#!/usr/bin/env python3
"""
Deep In-The-Money (DITM) Long Call Options Scanner
====================================================
Scans ALL optionable US equities via Yahoo Finance for CALL options
that meet the following criteria:

  Stock price           : $5 – $500
  Days to expiration    : 7 – 500 days
  Call delta            : +0.70 to +0.90  (deep ITM)
  Bid price             : <= $100         (affordable entry)
  Ask price             : <= $100         (affordable entry)
  Open interest         : >= 1,000
  Extrinsic % filter    : 18% – 30%       (extrinsic / ask × 100)
                          keeps only contracts with meaningful but not
                          excessive time value — sweet spot for DITM calls
  Trend filter          : Stock last price > 8 EMA AND > 20 EMA
                          (confirmed uptrend — only buy calls on
                           stocks already trending up)

Results ranked by: Ann ROC % descending, then Prob Profit % descending.

Strategy rationale:
  A deep ITM call with delta 0.70–0.90 moves nearly $0.70–$0.90 for
  every $1.00 the stock moves up.  Buying deep ITM gives you stock-like
  upside exposure at a fraction of the capital, with defined downside
  (you can only lose the premium paid).  The EMA filter ensures you're
  not fighting the trend.

Key output metrics:
  Intrinsic $   — how much of the premium is real value (strike below stock)
  Extrinsic $   — the time-value "cost" you pay above intrinsic
  Extrinsic %   — extrinsic as a % of ask (lower = cheaper leverage)
  Leverage      — stock price / ask price (how many $ of stock exposure per $ spent)
  Delta         — sensitivity to $1 stock move
  IV %          — implied volatility
  Break-even $  — stock price at expiration where you profit
  BE % Above    — how much the stock needs to rise just to break even

Ticker universe: built from four sources merged and de-duplicated —
  1. yfinance built-in screener (~4,000–8,000 names, handles Yahoo auth)
  2. Russell 1000 via iShares CSV (~1,000 names)
  3. S&P 500 via datahub.io (~503 names)
  4. Nasdaq-100 via Wikipedia (~101 names)

Threading: 5 concurrent workers by default (--workers N to override).

IMPORTANT – run during NYSE market hours (Mon–Fri 9:30–16:00 ET) for
accurate volume/OI data.

Usage:
  pip install -r requirements.txt
  python ditm_call_scanner.py                        # full scan, 5 workers
  python ditm_call_scanner.py --workers 3            # slower, gentler on Yahoo
  python ditm_call_scanner.py --ticker AAPL          # single-ticker debug
  python ditm_call_scanner.py --ticker AAPL --relax  # ignore OI filter
  python ditm_call_scanner.py --relax                # full scan, no OI filter
"""

import sys
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from math import exp, log, sqrt

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Strategy screening parameters ────────────────────────────────────────────
RISK_FREE_RATE   = 0.05

MIN_STOCK_PRICE  = 5.0
MAX_STOCK_PRICE  = 500.0
MIN_DTE          = 7
MAX_DTE          = 500
MIN_DELTA        = 0.70    # deep ITM calls
MAX_DELTA        = 0.90
MAX_ASK          = 100.0   # keep options affordable
MAX_BID          = 100.0
MIN_OPEN_INTEREST = 1_000

# Extrinsic-to-Ask filter: (extrinsic / ask) × 100 must be in this range.
# Below 18% = too deep, illiquid, near zero time value.
# Above 30% = too much time decay drag, behaves more like ATM.
MIN_EXTRINSIC_PCT = 18.0
MAX_EXTRINSIC_PCT = 30.0

# EMA periods for trend filter
EMA_SHORT = 8
EMA_LONG  = 20
# How many daily bars to fetch to compute EMAs reliably
EMA_LOOKBACK_DAYS = 60

# Threading
DEFAULT_WORKERS = 5
WORKER_SLEEP    = 0.4   # seconds between yFinance calls per worker


# ── Market-hours check ────────────────────────────────────────────────────────

def market_is_open() -> bool:
    eastern = timezone(timedelta(hours=-4))
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time


def warn_if_market_closed() -> None:
    if not market_is_open():
        print("=" * 72)
        print("  WARNING: NYSE appears to be CLOSED right now.")
        print("  Volume and OI data may be stale outside market hours.")
        print("  Run during market hours (Mon–Fri 9:30–16:00 ET) OR use --relax")
        print("  to bypass the OI filter for testing purposes.")
        print("=" * 72)
        print()


# ── NaN-safe helpers ──────────────────────────────────────────────────────────

def _f(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def _i(val, default: int = 0) -> int:
    try:
        v = float(val)
        return default if v != v else int(v)
    except (TypeError, ValueError):
        return default


# ── EMA calculation ───────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> float:
    """
    Return the most recent EMA value for the given period.
    Uses pandas ewm with adjust=False (standard EMA formula).
    Returns 0.0 if the series is too short.
    """
    if len(series) < period:
        return 0.0
    ema = series.ewm(span=period, adjust=False).mean()
    return float(ema.iloc[-1])


def stock_above_emas(ticker: str, price: float) -> tuple[bool, float, float]:
    """
    Fetch recent daily closes and check whether the last price is above
    both the 8 EMA and 20 EMA.

    Returns (passes_filter, ema8, ema20).
    If data cannot be fetched, returns (False, 0.0, 0.0).
    """
    try:
        hist = yf.Ticker(ticker).history(period=f"{EMA_LOOKBACK_DAYS}d", interval="1d")
        if hist.empty or len(hist) < EMA_LONG:
            return False, 0.0, 0.0
        closes = hist["Close"].dropna()
        ema8   = compute_ema(closes, EMA_SHORT)
        ema20  = compute_ema(closes, EMA_LONG)
        passes = price > ema8 and price > ema20
        return passes, round(ema8, 2), round(ema20, 2)
    except Exception:
        return False, 0.0, 0.0


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta = N(d1)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.5
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return float(norm.cdf(d1))
    except Exception:
        return 0.5


def implied_volatility_call(
    mid: float, S: float, K: float, T: float, r: float = RISK_FREE_RATE
) -> float:
    """Newton-Raphson IV solver for calls."""
    if T <= 0 or mid <= 0 or S <= 0 or K <= 0:
        return 0.30
    sigma = 0.30
    for _ in range(200):
        sigma = max(sigma, 1e-6)
        try:
            d1    = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2    = d1 - sigma * sqrt(T)
            price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            vega  = S * norm.pdf(d1) * sqrt(T)
            if abs(vega) < 1e-10:
                break
            diff = mid - price
            if abs(diff) < 1e-5:
                break
            sigma += diff / vega
        except Exception:
            break
    return float(np.clip(sigma, 0.01, 5.0))


# ── Ticker universe — multi-strategy builder ─────────────────────────────────

def _fetch_via_yfinance_screener() -> list:
    """
    Use yfinance built-in screener, trying multiple EquityQuery forms
    since Yahoo silently changes which operators are valid.
    """
    try:
        from yfinance import EquityQuery, screen as yf_screen
    except ImportError:
        raise RuntimeError("yfinance EquityQuery unavailable — upgrade: pip install -U yfinance")

    # Try several query forms in order — Yahoo changes supported operators without notice
    query_attempts = [
        lambda: EquityQuery("gt", ["intradaymarketcap", 0]),   # any positive market cap
        lambda: EquityQuery("eq", ["region", "us"]),            # all US equities
        lambda: EquityQuery("gt", ["regularmarketprice", 1]),   # price > $1
    ]

    last_err = None
    for attempt in query_attempts:
        try:
            q       = attempt()
            tickers = []
            offset  = 0
            while True:
                result = yf_screen(q, offset=offset, size=250)
                quotes = result.get("quotes", [])
                total  = result.get("total", 0)
                if not quotes:
                    break
                tickers.extend(r["symbol"] for r in quotes if r.get("symbol"))
                offset += len(quotes)
                print(f"    [yf-screener] fetched {offset:>5} / {total}", end="\r", flush=True)
                if offset >= total:
                    break
                time.sleep(0.3)
            if tickers:
                print(f"    [yf-screener] fetched {len(tickers):>5} tickers ✓       ")
                return tickers
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All screener query forms failed: {last_err}")


def _fetch_russell1000() -> list:
    url = (
        "https://www.ishares.com/us/products/239707/ISHARES-RUSSELL-1000-ETF"
        "/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
    )
    df = pd.read_csv(url, skiprows=9)
    tickers = (
        df["Ticker"].dropna().astype(str).str.strip()
        .str.replace(".", "-", regex=False).tolist()
    )
    return [t for t in tickers if t.isalpha() or
            (t.replace("-", "").isalpha() and len(t) <= 5)]


def _fetch_sp500() -> list:
    df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def _fetch_nasdaq100() -> list:
    """Nasdaq-100 with two fallback sources in case Wikipedia blocks."""
    # Source 1: Nasdaq official API
    try:
        import requests as _req
        r = _req.get(
            "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        rows = r.json()["data"]["data"]["rows"]
        return [row["symbol"] for row in rows if row.get("symbol")]
    except Exception:
        pass

    # Source 2: Wikipedia fallback
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", flavor="lxml")
        for t in tables:
            for col in t.columns:
                if str(col).lower() in ("ticker", "symbol"):
                    return t[col].str.replace(".", "-", regex=False).dropna().tolist()
    except Exception:
        pass

    return []




_FALLBACK_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","META","NVDA","TSLA","AMD",
    "INTC","QCOM","MU","AMAT","LRCX","KLAC","MRVL","AVGO","TXN",
    "SMCI","ARM","CRWD","PANW","ZS","NET","DDOG","SNOW","MDB",
    "CRM","NOW","ORCL","ADBE","WDAY","TEAM","HUBS",
    "JPM","BAC","WFC","GS","MS","C","USB","PNC","COF","AXP",
    "SCHW","BLK","COIN","HOOD","SOFI","UPST","PYPL",
    "XOM","CVX","COP","SLB","OXY","DVN","HAL","PSX","VLO",
    "JNJ","PFE","MRK","ABBV","BMY","AMGN","GILD","BIIB","MRNA",
    "RXRX","NVAX","IONS","REGN","VRTX","ALNY","INCY",
    "HD","LOW","TGT","WMT","COST","KR","DG","DLTR","NKE","LULU","DECK",
    "F","GM","RIVN","LCID","NIO","LI","XPEV",
    "DAL","AAL","UAL","LUV","CCL","RCL","NCLH",
    "MGM","WYNN","LVS","PENN","DKNG",
    "BA","LMT","RTX","NOC","GE","MMM","CAT","DE","HON","ETN",
    "AA","NUE","FCX","CLF","MP","GOLD","NEM","AEM",
    "T","VZ","TMUS","DIS","NFLX","WBD",
    "SNAP","PINS","RBLX","U","MTCH",
    "MSTR","IBIT","MARA","RIOT","CLSK",
    "OKLO","SMR","NNE","IONQ","RGTI","QUBT","ASTS","RKLB","LUNR","ACHR",
    "ACAD","PRGO","NBIX","PTGX",
    "SPY","QQQ","IWM","XLF","XLE","XLK","SMH","ARKK",
    "GLD","SLV","USO","TLT","HYG",
]


def get_ticker_universe() -> list:
    print("Building ticker universe...")
    combined = []

    try:
        print("  [1/4] yfinance screener...", end=" ", flush=True)
        t = _fetch_via_yfinance_screener()
        combined.extend(t)
        if t:
            print(f"{len(t)} tickers ✓")
    except Exception as e:
        print(f"failed ({e})")

    try:
        print("  [2/4] Russell 1000 from iShares...", end=" ", flush=True)
        t = _fetch_russell1000()
        combined.extend(t)
        print(f"{len(t)} tickers ✓")
    except Exception as e:
        print(f"failed ({e})")

    try:
        print("  [3/4] S&P 500 from datahub.io...", end=" ", flush=True)
        t = _fetch_sp500()
        combined.extend(t)
        print(f"{len(t)} tickers ✓")
    except Exception as e:
        print(f"failed ({e})")

    try:
        print("  [4/4] Nasdaq-100 from Wikipedia...", end=" ", flush=True)
        t = _fetch_nasdaq100()
        combined.extend(t)
        print(f"{len(t)} tickers ✓")
    except Exception as e:
        print(f"failed ({e})")

    if not combined:
        print("\n  WARNING: All sources failed — using curated emergency list.\n")
        combined = _FALLBACK_TICKERS

    combined.extend(_FALLBACK_TICKERS)

    seen, cleaned = set(), []
    for t in combined:
        t = str(t).strip().upper()
        if not t or " " in t or len(t) > 6:
            continue
        if not all(c.isalpha() or c == "-" for c in t):
            continue
        if t not in seen:
            seen.add(t)
            cleaned.append(t)

    cleaned.sort()
    print(f"\n  Universe ready: {len(cleaned)} unique tickers to scan.\n")
    return cleaned


# ── Yahoo Finance connectivity check ─────────────────────────────────────────

def check_yfinance_connectivity() -> bool:
    try:
        p = _f(yf.Ticker("SPY").fast_info.last_price)
        if p > 0:
            return True
        raise ValueError("price=0")
    except Exception as exc:
        print("=" * 72)
        print("  ERROR: Cannot reach Yahoo Finance.")
        print(f"  Detail: {exc}")
        print("  → Run on your local machine (not a sandboxed server).")
        print("=" * 72)
        return False


# ── Per-ticker call scan ──────────────────────────────────────────────────────

def scan_ticker_calls(
    ticker: str,
    stock_price: float,
    ema8: float,
    ema20: float,
    debug: bool = False,
    relax: bool = False,
) -> list:
    """
    Scan all call option expirations for a single ticker and return
    contracts that pass all DITM filters.
    """
    results = []
    today   = datetime.today()

    try:
        tk          = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            if debug:
                print(f"  [{ticker}] No expirations returned.")
            return results

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte      = (exp_date - today).days

            if not (MIN_DTE <= dte <= MAX_DTE):
                if debug:
                    print(f"  [{ticker}] {exp_str}  DTE={dte}  → skip (DTE out of range)")
                continue

            T = dte / 365.0

            try:
                chain = tk.option_chain(exp_str)
                calls = chain.calls
            except Exception as e:
                if debug:
                    print(f"  [{ticker}] {exp_str}  → could not fetch chain: {e}")
                continue

            if calls is None or calls.empty:
                if debug:
                    print(f"  [{ticker}] {exp_str}  DTE={dte}  → no calls returned")
                continue

            if debug:
                print(f"\n  [{ticker}] {exp_str}  DTE={dte}  ({len(calls)} calls)  "
                      f"stock=${stock_price:.2f}  8EMA=${ema8:.2f}  20EMA=${ema20:.2f}")

            for _, row in calls.iterrows():
                strike = _f(row.get("strike"))
                bid    = _f(row.get("bid"))
                ask    = _f(row.get("ask"))
                oi     = _i(row.get("openInterest"))
                vol    = _i(row.get("volume"))

                if strike <= 0 or bid <= 0 or ask <= 0:
                    if debug:
                        print(f"    K={strike:<7.2f}  bid={bid}  ask={ask}  → no market")
                    continue

                mid = (bid + ask) / 2.0

                # ── Quick pre-filters before IV/delta calc ────────────────
                if ask > MAX_ASK:
                    if debug:
                        print(f"    K={strike:<7.2f}  ask={ask:.2f}  → skip (ask>${MAX_ASK})")
                    continue
                if bid > MAX_BID:
                    continue
                if not relax and oi < MIN_OPEN_INTEREST:
                    if debug:
                        print(f"    K={strike:<7.2f}  OI={oi}  → skip (OI<{MIN_OPEN_INTEREST})")
                    continue

                # ── IV and delta ──────────────────────────────────────────
                iv_raw = _f(row.get("impliedVolatility"))
                iv     = iv_raw if 0.01 < iv_raw < 5.0 else implied_volatility_call(mid, stock_price, strike, T)
                delta  = bs_call_delta(stock_price, strike, T, RISK_FREE_RATE, iv)

                if debug:
                    intrinsic_dbg = max(0.0, stock_price - strike)
                    extrinsic_dbg = max(0.0, mid - intrinsic_dbg)
                    extpct_dbg    = (extrinsic_dbg / ask * 100) if ask > 0 else 0.0
                    reasons = []
                    if not (MIN_DELTA <= delta <= MAX_DELTA):
                        reasons.append(f"delta={delta:.3f} not in [{MIN_DELTA},{MAX_DELTA}]")
                    if not (MIN_EXTRINSIC_PCT <= extpct_dbg <= MAX_EXTRINSIC_PCT):
                        reasons.append(f"extr%={extpct_dbg:.1f}% not in [{MIN_EXTRINSIC_PCT},{MAX_EXTRINSIC_PCT}]")
                    if ask > MAX_ASK:
                        reasons.append(f"ask={ask:.2f}>${MAX_ASK}")
                    if not relax and oi < MIN_OPEN_INTEREST:
                        reasons.append(f"OI={oi}<{MIN_OPEN_INTEREST}")
                    status = "PASS" if not reasons else "FAIL: " + " | ".join(reasons)
                    print(f"    K={strike:<7.2f}  mid={mid:.2f}  ask={ask:.2f}  "
                          f"OI={oi:<6}  IV={iv*100:.0f}%  delta={delta:+.3f}  "
                          f"extr%={extpct_dbg:.1f}%  → {status}")

                if not (MIN_DELTA <= delta <= MAX_DELTA):
                    continue

                # ── Compute output metrics ────────────────────────────────
                # Intrinsic value: how much the call is already in the money
                intrinsic     = max(0.0, stock_price - strike)
                # Extrinsic value: time premium you pay above intrinsic
                extrinsic     = max(0.0, mid - intrinsic)
                # Extrinsic as % of ask — the core quality filter
                extrinsic_pct = (extrinsic / ask * 100) if ask > 0 else 0.0

                # ── Extrinsic % hard filter (18–30%) ─────────────────────
                if not (MIN_EXTRINSIC_PCT <= extrinsic_pct <= MAX_EXTRINSIC_PCT):
                    if debug:
                        print(f"    K={strike:<7.2f}  extr%={extrinsic_pct:.1f}%  "
                              f"→ skip (not in {MIN_EXTRINSIC_PCT}–{MAX_EXTRINSIC_PCT}%)")
                    continue

                # Leverage: $ of stock exposure per $ of option cost
                leverage      = (stock_price / ask) if ask > 0 else 0.0
                # Break-even at expiration: strike + ask (what you paid)
                breakeven     = strike + ask
                # How much the stock needs to rise (%) to break even
                be_pct_above  = ((breakeven - stock_price) / stock_price) * 100
                # Contract cost (1 contract = 100 shares)
                contract_cost = ask * 100
                # ROC %: delta-adjusted stock exposure vs capital spent
                # = (delta × stock price) / ask × 100
                # Measures how much stock-price exposure you get per $ invested
                roc_pct       = (delta * stock_price / ask * 100) if ask > 0 else 0.0
                # Annualised ROC
                roc_annual    = roc_pct * (365.0 / dte) if dte > 0 else 0.0
                # Prob Profit %: probability the call expires ITM = delta × 100
                prob_profit   = delta * 100

                results.append({
                    "Ticker"         : ticker,
                    "Stock $"        : round(stock_price,    2),
                    "8 EMA"          : round(ema8,           2),
                    "20 EMA"         : round(ema20,          2),
                    "Strike"         : round(strike,         2),
                    "Expiration"     : exp_str,
                    "DTE"            : dte,
                    "Bid"            : round(bid,            2),
                    "Ask"            : round(ask,            2),
                    "Mid"            : round(mid,            2),
                    "Volume"         : vol,
                    "OI"             : oi,
                    "IV %"           : round(iv * 100,       1),
                    "Delta"          : round(delta,          3),
                    "Intrinsic $"    : round(intrinsic,      2),
                    "Extrinsic $"    : round(extrinsic,      2),
                    "Extrinsic %"    : round(extrinsic_pct,  1),
                    "Leverage"       : round(leverage,       2),
                    "Breakeven $"    : round(breakeven,      2),
                    "BE % Above"     : round(be_pct_above,   2),
                    "Contract Cost $": round(contract_cost,  2),
                    "ROC %"          : round(roc_pct,        2),
                    "Ann ROC %"      : round(roc_annual,     2),
                    "Prob Profit %"  : round(prob_profit,    1),
                })

            time.sleep(WORKER_SLEEP)

    except Exception as e:
        if debug:
            print(f"  [{ticker}] Exception: {e}")

    return results


# ── Threaded worker ───────────────────────────────────────────────────────────

def _scan_one(ticker: str, relax: bool) -> dict:
    """
    Fetch price, check EMA trend filter, then scan calls.
    Returns a result dict for the orchestrator to collect.
    """
    try:
        price = _f(yf.Ticker(ticker).fast_info.last_price)
    except Exception:
        price = 0.0

    in_price_range = MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE
    if not in_price_range:
        return {"ticker": ticker, "price": price,
                "eligible": False, "skip_reason": "price_range", "contracts": []}

    # EMA trend filter — fetches 60d of daily history
    passes_ema, ema8, ema20 = stock_above_emas(ticker, price)
    if not passes_ema:
        return {"ticker": ticker, "price": price,
                "eligible": False, "skip_reason": "ema_filter",
                "ema8": ema8, "ema20": ema20, "contracts": []}

    contracts = scan_ticker_calls(ticker, price, ema8, ema20, relax=relax)
    return {
        "ticker"    : ticker,
        "price"     : price,
        "eligible"  : True,
        "ema8"      : ema8,
        "ema20"     : ema20,
        "contracts" : contracts,
    }


# ── Single-ticker debug mode ──────────────────────────────────────────────────

def debug_single_ticker(ticker: str, relax: bool = False) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  DEBUG MODE  –  {ticker}{'  (--relax: OI filter OFF)' if relax else ''}")
    print(sep)

    try:
        price = _f(yf.Ticker(ticker).fast_info.last_price)
    except Exception as e:
        print(f"  Cannot fetch price: {e}")
        return

    print(f"  Current price : ${price:.2f}")

    if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
        print(f"  *** ${price:.2f} is outside the ${MIN_STOCK_PRICE}–${MAX_STOCK_PRICE} price filter ***")

    passes_ema, ema8, ema20 = stock_above_emas(ticker, price)
    print(f"  8 EMA         : ${ema8:.2f}  {'✓ above' if price > ema8 else '✗ BELOW'}")
    print(f"  20 EMA        : ${ema20:.2f}  {'✓ above' if price > ema20 else '✗ BELOW'}")
    if not passes_ema:
        print(f"  *** EMA filter FAIL — stock is not in confirmed uptrend ***")
        print(f"  (Showing contracts anyway in debug mode)")

    flag = "  [OI filter RELAXED]" if relax else ""
    print(f"\n  Criteria : DTE {MIN_DTE}–{MAX_DTE} | delta [{MIN_DELTA},{MAX_DELTA}] | "
          f"ask≤${MAX_ASK} | OI≥{MIN_OPEN_INTEREST}{flag}\n")

    results = scan_ticker_calls(ticker, price, ema8, ema20, debug=True, relax=relax)

    print(f"\n  → {len(results)} contracts passed all filters.\n")
    if results:
        df   = pd.DataFrame(results)
        cols = ["Strike", "Expiration", "DTE", "Ask", "Delta", "IV %",
                "Intrinsic $", "Extrinsic $", "Extrinsic %", "Leverage",
                "Breakeven $", "BE % Above", "ROC %", "Ann ROC %", "Prob Profit %"]
        print(tabulate(df[cols], headers="keys", tablefmt="simple",
                       showindex=False, floatfmt=".2f"))


# ── Full scan orchestrator ────────────────────────────────────────────────────

def run_scan(relax: bool = False, workers: int = DEFAULT_WORKERS) -> pd.DataFrame:
    sep = "=" * 72
    print(f"\n{sep}")
    mode_note = "  [RELAXED: OI filter OFF]" if relax else ""
    print(f"  DEEP ITM CALL OPTIONS SCANNER  –  Yahoo Finance{mode_note}")
    print(sep)
    print(f"  Stock price       : ${MIN_STOCK_PRICE:.0f} – ${MAX_STOCK_PRICE:.0f}")
    print(f"  DTE               : {MIN_DTE} – {MAX_DTE} days")
    print(f"  Call delta        : {MIN_DELTA} – {MAX_DELTA}  (deep ITM)")
    print(f"  Max ask price     : ${MAX_ASK:.0f}")
    print(f"  Max bid price     : ${MAX_BID:.0f}")
    if relax:
        print(f"  Min open interest : RELAXED (0)")
    else:
        print(f"  Min open interest : {MIN_OPEN_INTEREST:,}")
    print(f"  Extrinsic % filter: {MIN_EXTRINSIC_PCT}% – {MAX_EXTRINSIC_PCT}%  (extrinsic / ask × 100)")
    print(f"  Trend filter      : Price > {EMA_SHORT} EMA  AND  Price > {EMA_LONG} EMA")
    print(f"  Risk-free rate    : {RISK_FREE_RATE*100:.1f}%")
    print(f"  Universe source   : yfinance screener + Russell 1000 + S&P 500 + Nasdaq-100")
    print(f"  Concurrent workers: {workers}  (sleep {WORKER_SLEEP}s/call)")
    print(f"{sep}\n")

    universe = get_ticker_universe()
    total    = len(universe)

    all_results    = []
    eligible_count = 0
    ema_fail_count = 0
    found_lock     = threading.Lock()
    found_total    = [0]

    print(f"Scanning {total} tickers with {workers} workers...\n")

    with tqdm(
        total=total,
        unit="ticker",
        ncols=90,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    ) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_scan_one, ticker, relax): ticker
                for ticker in universe
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    pbar.update(1)
                    continue

                if result.get("eligible"):
                    eligible_count += 1
                    contracts = result["contracts"]
                    all_results.extend(contracts)
                    with found_lock:
                        found_total[0] += len(contracts)
                elif result.get("skip_reason") == "ema_filter":
                    ema_fail_count += 1

                pbar.set_postfix(
                    trend_pass=eligible_count,
                    contracts=found_total[0],
                    ticker=result["ticker"],
                )
                pbar.update(1)

    print(f"\nScan complete:")
    print(f"  Tickers scanned        : {total}")
    print(f"  Passed price filter    : {eligible_count + ema_fail_count}")
    print(f"  Passed EMA trend filter: {eligible_count}")
    print(f"  Qualifying contracts   : {len(all_results)}\n")

    if not all_results:
        print("No contracts met all criteria.")
        if not relax and not market_is_open():
            print("Tip: Market appears closed. Try:  python ditm_call_scanner.py --relax")
        return pd.DataFrame()

    # Primary sort: Ann ROC % descending, secondary: Prob Profit % descending
    df = (pd.DataFrame(all_results)
          .sort_values(["Ann ROC %", "Prob Profit %"], ascending=[False, False])
          .reset_index(drop=True))
    return df


# ── Results display ───────────────────────────────────────────────────────────

def display_results(df: pd.DataFrame) -> None:
    if df.empty:
        return

    sep = "=" * 72

    print(f"\n{sep}")
    print("  ALL QUALIFYING DITM CALLS  –  ranked by Ann ROC % then Prob Profit %")
    print(sep)
    display_cols = [
        "Ticker", "Stock $", "Strike", "Expiration", "DTE",
        "Ask", "Delta", "IV %", "Intrinsic $", "Extrinsic $",
        "Extrinsic %", "Leverage", "Breakeven $", "BE % Above",
        "ROC %", "Ann ROC %", "Prob Profit %", "Contract Cost $",
    ]
    print(tabulate(df[display_cols], headers="keys", tablefmt="rounded_outline",
                   showindex=True, floatfmt=".2f"))

    top_cols = ["Ticker", "Stock $", "Strike", "Expiration", "DTE",
                "Ask", "Delta", "Extrinsic %", "ROC %", "Ann ROC %",
                "Prob Profit %", "Leverage", "BE % Above", "8 EMA", "20 EMA"]

    print(f"\n{sep}")
    print("  TOP 10  –  Highest Ann ROC %  (best return on capital)")
    print(sep)
    print(tabulate(df.nlargest(10, "Ann ROC %")[top_cols].reset_index(drop=True),
                   headers="keys", tablefmt="simple", showindex=True, floatfmt=".2f"))

    print(f"\n{sep}")
    print("  TOP 10  –  Highest Prob Profit %  (highest probability of expiring ITM)")
    print(sep)
    print(tabulate(df.nlargest(10, "Prob Profit %")[top_cols].reset_index(drop=True),
                   headers="keys", tablefmt="simple", showindex=True, floatfmt=".2f"))

    print(f"\n{sep}")
    print("  TOP 10  –  Lowest Extrinsic %  (cheapest time decay / best leverage)")
    print(sep)
    print(tabulate(df.nsmallest(10, "Extrinsic %")[top_cols].reset_index(drop=True),
                   headers="keys", tablefmt="simple", showindex=True, floatfmt=".2f"))

    print(f"\n{sep}")
    print("  TOP 10  –  Highest Leverage  (most stock exposure per $ spent)")
    print(sep)
    print(tabulate(df.nlargest(10, "Leverage")[top_cols].reset_index(drop=True),
                   headers="keys", tablefmt="simple", showindex=True, floatfmt=".2f"))

    print(f"\n{sep}")
    print("  PORTFOLIO SUMMARY STATISTICS")
    print(sep)
    print(f"  Total qualifying contracts : {len(df)}")
    print(f"  Unique tickers             : {df['Ticker'].nunique()}")
    print(f"  Average delta              : {df['Delta'].mean():.3f}")
    print(f"  Average extrinsic %        : {df['Extrinsic %'].mean():.1f}%")
    print(f"  Average Ann ROC %          : {df['Ann ROC %'].mean():.2f}%")
    print(f"  Average prob profit        : {df['Prob Profit %'].mean():.1f}%")
    print(f"  Average leverage           : {df['Leverage'].mean():.2f}x")
    print(f"  Average ask price          : ${df['Ask'].mean():.2f}")
    print(f"  Average contract cost      : ${df['Contract Cost $'].mean():.2f}")
    print(f"  Average DTE                : {df['DTE'].mean():.0f} days")
    print()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = f"ditm_calls_{ts}.csv"
    xlsx_path = f"ditm_calls_{ts}.xlsx"

    df.to_csv(csv_path, index=False)
    print(f"  CSV  saved → {csv_path}")

    try:
        df.to_excel(xlsx_path, index=False)
        print(f"  XLSX saved → {xlsx_path}")
    except Exception as e:
        print(f"  XLSX save failed: {e}")

    print(f"  Scan timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> tuple:
    args    = sys.argv[1:]
    relax   = "--relax" in args
    workers = DEFAULT_WORKERS

    if "--workers" in args:
        idx = args.index("--workers")
        try:
            w = int(args[idx + 1])
            if w < 1:
                raise ValueError
            workers = w
        except (IndexError, ValueError):
            print("Usage: --workers N  (positive integer, e.g. --workers 3)")
            sys.exit(1)

    if "--ticker" in args:
        idx = args.index("--ticker")
        try:
            ticker = args[idx + 1].upper()
        except IndexError:
            print("Usage: python ditm_call_scanner.py --ticker SYMBOL [--relax]")
            sys.exit(1)
        return "single", ticker, relax, workers

    return "full", None, relax, workers


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode, ticker, relax, workers = parse_args()

    if mode == "single":
        if not check_yfinance_connectivity():
            sys.exit(1)
        debug_single_ticker(ticker, relax=relax)
    else:
        warn_if_market_closed()
        if not check_yfinance_connectivity():
            sys.exit(1)
        results_df = run_scan(relax=relax, workers=workers)
        display_results(results_df)
