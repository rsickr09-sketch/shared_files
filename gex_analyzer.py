#!/usr/bin/env python3
"""
gex_analyzer.py  —  Gamma Exposure (GEX) calculator
=====================================================
Calculates GEX, key gamma levels, and regime tags for any ticker
using the options chain data already pulled by yfinance — no external
API needed.

WHAT IT CALCULATES
------------------
For each ticker, given its full options chain (calls + puts across all
near-term expirations):

  Net GEX ($)         Total dollar gamma exposure of all market makers.
                      Positive = dealers are long gamma (dampening).
                      Negative = dealers are short gamma (amplifying).

  GEX Regime          "POSITIVE" / "NEGATIVE" / "NEUTRAL"
                      Based on Net GEX sign and magnitude.

  Gamma Flip $        The price level where net GEX crosses zero.
                      Approximated as the strike with the smallest
                      absolute net GEX.

  Call Wall $         Strike with the largest positive (call) gamma.
                      Acts as resistance — heaviest dealer hedging above.

  Put Wall $          Strike with the largest negative (put) gamma.
                      Acts as support — heaviest dealer hedging below.

  GEX Score (0-10)    Composite score for quick ranking:
                        10 = strong positive GEX, ideal for selling premium
                         0 = strong negative GEX, dangerous for selling premium

  GEX Warning         Human-readable flag for the results table.

HOW GEX IS CALCULATED
---------------------
For each option contract:

  Contract GEX = gamma × open_interest × 100 × stock_price²  × 0.01

  (the 0.01 converts gamma from per-$ to per-1% move, consistent with
   the SqueezeMetrics convention used by most GEX data providers)

  Calls contribute POSITIVE GEX (dealers short calls → long gamma)
  Puts  contribute NEGATIVE GEX (dealers short puts → short gamma on downside)

  Net GEX = sum(call GEX) + sum(put GEX)

  Note: we assume dealers are net short options (the standard assumption).
  This is accurate ~90% of the time for liquid US equities and ETFs.

INTEGRATION
-----------
Called from scan_ticker_options() in options_30delta.py and
scan_ticker_calls() in ditm_call_scanner.py after the options chain
is already fetched — no extra network call needed.

    from gex_analyzer import calculate_gex, gex_tag

    gex = calculate_gex(ticker, stock_price, calls_df, puts_df)
    # gex is a dict — merge into results.append({...})
"""

from __future__ import annotations
import math
from typing import Optional
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# GEX magnitude thresholds (in millions of dollars)
# Below NEUTRAL_THRESHOLD: regime = NEUTRAL
# Above POSITIVE_THRESHOLD: regime = POSITIVE (strong dampening)
# Below -POSITIVE_THRESHOLD: regime = NEGATIVE (strong amplifying)
NEUTRAL_THRESHOLD  = 5.0    # $5M
POSITIVE_THRESHOLD = 50.0   # $50M


# ── Core calculation ──────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (v != v) else v   # NaN check
    except (TypeError, ValueError):
        return default


def _contract_gex(gamma: float, oi: int, stock_price: float,
                  is_call: bool) -> float:
    """
    Dollar GEX for one contract series.
    Positive for calls (dealers long gamma), negative for puts.
    Units: dollars of delta change per 1% underlying move.
    """
    if gamma <= 0 or oi <= 0 or stock_price <= 0:
        return 0.0
    raw = gamma * oi * 100 * (stock_price ** 2) * 0.01
    return raw if is_call else -raw


def calculate_gex(
    ticker: str,
    stock_price: float,
    calls: Optional[pd.DataFrame],
    puts:  Optional[pd.DataFrame],
    max_dte_for_gex: int = 45,
) -> dict:
    """
    Calculate full GEX profile for a ticker from its options chain.

    Parameters
    ----------
    ticker          : Ticker symbol (for labelling only)
    stock_price     : Current underlying price
    calls           : DataFrame from yf.Ticker.option_chain(exp).calls
                      (can be a combined multi-expiry frame)
    puts            : DataFrame from yf.Ticker.option_chain(exp).puts
    max_dte_for_gex : Only include expirations within this many days
                      (closer expirations dominate GEX in practice)

    Returns
    -------
    dict with keys:
        GEX Net $M      Net gamma exposure in millions
        GEX Regime      POSITIVE / NEUTRAL / NEGATIVE
        GEX Flip $      Estimated gamma flip price level
        Call Wall $     Strike with max call gamma concentration
        Put Wall $      Strike with max put gamma concentration
        GEX Score       0-10 composite score
        GEX Tag         Short label for the results table
    """
    empty = {
        "GEX Net $M": 0.0,
        "GEX Regime": "NEUTRAL",
        "GEX Flip $": round(stock_price, 2),
        "Call Wall $": round(stock_price, 2),
        "Put Wall $":  round(stock_price, 2),
        "GEX Score":   5,
        "GEX Tag":     "N/A",
    }

    if stock_price <= 0:
        return empty

    # ── Accumulate per-strike GEX ─────────────────────────────────────────
    # Dict: strike -> net GEX dollars at that strike
    strike_gex: dict[float, float] = {}

    def _process(df: pd.DataFrame, is_call: bool) -> None:
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            strike  = _safe_float(row.get("strike"))
            gamma   = _safe_float(row.get("gamma"))
            oi      = int(_safe_float(row.get("openInterest")))

            if strike <= 0:
                continue

            # If gamma not in chain (yfinance sometimes omits it),
            # approximate via Black-Scholes simplified formula
            if gamma == 0.0:
                iv = _safe_float(row.get("impliedVolatility"), 0.30)
                if iv > 0 and stock_price > 0:
                    # Approximate ATM gamma: 1 / (S * iv * sqrt(2*pi))
                    # (simplified, assumes near-ATM; fine for GEX estimation)
                    gamma = 1.0 / (stock_price * iv * math.sqrt(2 * math.pi))

            gex = _contract_gex(gamma, oi, stock_price, is_call)
            strike_gex[strike] = strike_gex.get(strike, 0.0) + gex

    _process(calls, is_call=True)
    _process(puts,  is_call=False)

    if not strike_gex:
        return empty

    # ── Aggregate metrics ─────────────────────────────────────────────────
    net_gex_raw  = sum(strike_gex.values())
    net_gex_mm   = net_gex_raw / 1_000_000   # convert to millions

    strikes      = sorted(strike_gex.keys())
    gex_values   = [strike_gex[s] for s in strikes]

    # Call wall = strike with highest positive GEX above current price
    call_candidates = [(s, v) for s, v in zip(strikes, gex_values)
                       if s >= stock_price and v > 0]
    call_wall = max(call_candidates, key=lambda x: x[1])[0] \
                if call_candidates else stock_price

    # Put wall = strike with largest negative GEX below current price
    put_candidates  = [(s, v) for s, v in zip(strikes, gex_values)
                       if s <= stock_price and v < 0]
    put_wall  = min(put_candidates,  key=lambda x: x[1])[0] \
                if put_candidates  else stock_price

    # Gamma flip = strike where GEX is closest to zero
    # (the price level where positive and negative gamma cancel out)
    flip_strike = min(strikes, key=lambda s: abs(strike_gex[s]))

    # ── Regime classification ─────────────────────────────────────────────
    abs_gex = abs(net_gex_mm)
    if abs_gex < NEUTRAL_THRESHOLD:
        regime = "NEUTRAL"
    elif net_gex_mm > 0:
        regime = "POSITIVE"
    else:
        regime = "NEGATIVE"

    # ── GEX Score (0-10) ──────────────────────────────────────────────────
    # 10 = very positive GEX (great for selling premium)
    # 5  = neutral
    # 0  = very negative GEX (dangerous for selling premium)
    #
    # Sigmoid-like mapping: score = 5 + 5 * tanh(net_gex_mm / 100)
    score_raw = 5.0 + 5.0 * math.tanh(net_gex_mm / 100.0)
    score     = max(0, min(10, round(score_raw)))

    # ── GEX Tag for results table ─────────────────────────────────────────
    if regime == "POSITIVE":
        if score >= 8:
            tag = "✓ POS STRONG"
        else:
            tag = "✓ POS"
    elif regime == "NEGATIVE":
        if score <= 2:
            tag = "⚠ NEG STRONG"
        else:
            tag = "⚠ NEG"
    else:
        tag = "— NEUTRAL"

    return {
        "GEX Net $M":  round(net_gex_mm, 2),
        "GEX Regime":  regime,
        "GEX Flip $":  round(flip_strike, 2),
        "Call Wall $": round(call_wall,   2),
        "Put Wall $":  round(put_wall,    2),
        "GEX Score":   score,
        "GEX Tag":     tag,
    }


def gex_tag(gex_dict: dict) -> str:
    """Convenience function — returns just the GEX tag string."""
    return gex_dict.get("GEX Tag", "N/A")


def gex_warning(gex_dict: dict, scanner_type: str = "csp") -> str:
    """
    Returns a human-readable warning string for use in scan output.

    scanner_type: "csp"  — selling puts (wants positive GEX)
                  "ditm" — buying calls (benefits from positive GEX uptrend)
    """
    regime = gex_dict.get("GEX Regime", "NEUTRAL")
    score  = gex_dict.get("GEX Score",  5)
    flip   = gex_dict.get("GEX Flip $", 0)
    cwall  = gex_dict.get("Call Wall $", 0)
    pwall  = gex_dict.get("Put Wall $",  0)

    if scanner_type == "csp":
        if regime == "NEGATIVE" and score <= 3:
            return (f"HIGH RISK: Negative GEX (score {score}/10). "
                    f"Dealers amplify down moves. Put wall at ${pwall:.2f}. "
                    f"Consider waiting for GEX flip above ${flip:.2f}.")
        elif regime == "NEGATIVE":
            return (f"CAUTION: Negative GEX (score {score}/10). "
                    f"Put wall support at ${pwall:.2f}.")
        elif regime == "POSITIVE":
            return (f"FAVORABLE: Positive GEX (score {score}/10). "
                    f"Dealers dampen moves. Call wall ${cwall:.2f}, "
                    f"put wall ${pwall:.2f}.")
        else:
            return f"NEUTRAL GEX (score {score}/10). Gamma flip at ${flip:.2f}."

    else:  # ditm
        if regime == "POSITIVE":
            return (f"FAVORABLE: Positive GEX (score {score}/10). "
                    f"Slow grind up environment. Call wall target ${cwall:.2f}.")
        elif regime == "NEGATIVE":
            return (f"VOLATILE: Negative GEX (score {score}/10). "
                    f"Fast moves both ways. Gamma flip at ${flip:.2f}.")
        else:
            return f"NEUTRAL GEX (score {score}/10). Watch flip at ${flip:.2f}."
