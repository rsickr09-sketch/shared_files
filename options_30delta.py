#!/usr/bin/env python3
"""
Cash Secured Put (CSP) Options Scanner
=======================================
Scans NYSE/major-index optionable stocks via Yahoo Finance for put options
that meet the following criteria, then ranks them for a CSP portfolio:

  Stock price last     : $5 – $250
  Days to expiration   : 21 – 60 days
  Option volume (day)  : >= 250   (only meaningful during market hours)
  Open interest        : >= 1,000
  Put delta            : -0.30 to -0.01
  Credit (mid-price)   : >= $0.50
  Max cash / contract  : $20,000

IMPORTANT – run this script during NYSE market hours (Mon–Fri 9:30–16:00 ET).
Volume data from Yahoo Finance is 0 outside market hours, which causes the
volume filter to reject every contract.

Ticker universe: built from four sources merged and de-duplicated —
  1. yfinance built-in screener (~4,000–8,000 names, handles Yahoo auth)
  2. Russell 1000 via iShares CSV (~1,000 names)
  3. S&P 500 via datahub.io (~503 names)
  4. Nasdaq-100 via Nasdaq API + Wikipedia fallback (~101 names)

Threading: 5 concurrent workers by default (--workers N to override).

Usage:
  pip install -r requirements.txt
  python options_30delta.py                       # full scan, 5 workers
  python options_30delta.py --workers 3           # slower, gentler on Yahoo
  python options_30delta.py --ticker INTC         # single-ticker debug
  python options_30delta.py --ticker INTC --relax # ignore vol/OI, show all
  python options_30delta.py --relax               # full scan, no vol/OI filter
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
RISK_FREE_RATE        = 0.05

MIN_STOCK_PRICE       = 5.0
MAX_STOCK_PRICE       = 250.0
MIN_DTE               = 21
MAX_DTE               = 60
MIN_VOLUME            = 250    # requires market-hours data to be meaningful
MIN_OPEN_INTEREST     = 1_000
MIN_DELTA             = -0.30
MAX_DELTA             = -0.01
MIN_CREDIT            = 0.50
MAX_CASH_PER_CONTRACT = 20_000

# Threading
DEFAULT_WORKERS = 5
WORKER_SLEEP    = 0.4   # seconds between yFinance calls per worker


# ── Market-hours check ────────────────────────────────────────────────────────

def market_is_open() -> bool:
    """Return True if NYSE is currently in regular session (approximate)."""
    eastern = timezone(timedelta(hours=-4))   # EDT; use -5 for EST
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
        print("  Yahoo Finance returns volume=0 outside market hours.")
        print("  The MIN_VOLUME=250 filter will reject every contract.")
        print("  Run during market hours (Mon–Fri 9:30–16:00 ET) OR use --relax")
        print("  to bypass volume/OI filters for diagnostic purposes.")
        print("=" * 72)
        print()


# ── NaN-safe value extractors ─────────────────────────────────────────────────
def _f(val, default: float = 0.0) -> float:
    """Float, replacing None/NaN/invalid with default."""
    try:
        v = float(val)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def _i(val, default: int = 0) -> int:
    """Int, replacing None/NaN/invalid with default."""
    try:
        v = float(val)
        return default if v != v else int(v)
    except (TypeError, ValueError):
        return default


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return -0.5
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        return float(norm.cdf(d1) - 1.0)
    except Exception:
        return -0.5


def implied_volatility(
    mid: float, S: float, K: float, T: float, r: float = RISK_FREE_RATE
) -> float:
    if T <= 0 or mid <= 0 or S <= 0 or K <= 0:
        return 0.30
    sigma = 0.30
    for _ in range(200):
        sigma = max(sigma, 1e-6)
        try:
            d1    = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2    = d1 - sigma * sqrt(T)
            price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
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


# ── Ticker universe — multi-source builder ────────────────────────────────────

def _fetch_via_yfinance_screener() -> list:
    """
    Use yfinance built-in screener — handles Yahoo auth automatically.
    Tries multiple EquityQuery forms since Yahoo changes supported
    operators without notice.
    """
    try:
        from yfinance import EquityQuery, screen as yf_screen
    except ImportError:
        raise RuntimeError("yfinance EquityQuery unavailable — upgrade: pip install -U yfinance")

    query_attempts = [
        lambda: EquityQuery("gt", ["intradaymarketcap", 0]),
        lambda: EquityQuery("eq", ["region", "us"]),
        lambda: EquityQuery("gt", ["regularmarketprice", 1]),
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
    """Nasdaq-100 with two fallback sources in case one blocks."""
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


# Curated high-liquidity supplement — always appended as safety net
_SUPPLEMENT = [
    # Mega-cap tech & semis
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA", "AMD",
    "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "AVGO", "TXN",
    "SMCI", "ARM", "CRWD", "PANW", "ZS", "NET", "DDOG", "SNOW", "MDB",
    # Cloud / SaaS
    "CRM", "NOW", "ORCL", "SAP", "ADBE", "WDAY", "TEAM", "HUBS",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "COF", "AXP",
    "SCHW", "BLK", "COIN", "HOOD", "SOFI", "UPST", "SQ", "PYPL",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "DVN", "MRO", "HAL", "PSX", "VLO",
    # Healthcare & biotech
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "BIIB", "MRNA",
    "RXRX", "NVAX", "IONS", "REGN", "VRTX", "ALNY", "INCY",
    # Consumer / retail
    "HD", "LOW", "TGT", "WMT", "COST", "KR", "DG", "DLTR",
    "NKE", "LULU", "DECK", "SKX",
    # Autos & EVs
    "F", "GM", "RIVN", "LCID", "NIO", "LI", "XPEV",
    # Airlines / travel / leisure
    "DAL", "AAL", "UAL", "LUV", "CCL", "RCL", "NCLH",
    "MGM", "WYNN", "LVS", "PENN", "DKNG",
    # Industrials / defense
    "BA", "LMT", "RTX", "NOC", "GE", "MMM", "CAT", "DE", "HON", "ETN",
    # Materials / metals
    "AA", "NUE", "FCX", "CLF", "MP", "GOLD", "NEM", "AEM",
    # Telecom / media
    "T", "VZ", "TMUS", "DIS", "NFLX", "PARA", "WBD",
    # Social / internet
    "SNAP", "PINS", "RBLX", "U", "MTCH",
    # Crypto adjacent
    "MSTR", "IBIT", "MARA", "RIOT", "CLSK",
    # High-IV / active options
    "OKLO", "SMR", "NNE", "IONQ", "RGTI", "QUBT", "ARQQ",
    "ASTS", "RKLB", "LUNR", "ACHR",
    # Active biotech
    "SAVA", "ACAD", "PRGO", "NBIX", "PTGX", "KRTX",
    # ETFs with active options
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "SMH", "ARKK",
    "GLD", "SLV", "USO", "TLT", "HYG",
]


def get_ticker_universe() -> list:
    """
    Build a deduplicated universe from four sources:
      1. yfinance screener (~4,000–8,000 names)
      2. Russell 1000 via iShares (~1,000 names)
      3. S&P 500 via datahub.io (~503 names)
      4. Nasdaq-100 via Nasdaq API / Wikipedia (~101 names)
    Curated supplement always appended as a safety net.
    """
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
        print("  [4/4] Nasdaq-100...", end=" ", flush=True)
        t = _fetch_nasdaq100()
        combined.extend(t)
        print(f"{len(t)} tickers ✓")
    except Exception as e:
        print(f"failed ({e})")

    if not combined:
        print("\n  WARNING: All sources failed — using curated emergency list.\n")
        combined = list(_SUPPLEMENT)

    # Always append supplement as safety net
    combined.extend(_SUPPLEMENT)

    # De-duplicate, clean, sort
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
        print()
        print("  Possible causes:")
        print("  1. No internet connection.")
        print("  2. This script is running inside a server / sandbox environment")
        print("     that blocks outbound connections to yahoo.com.")
        print("     → Run the script on your LOCAL machine instead.")
        print("  3. Yahoo Finance is temporarily rate-limiting or down.")
        print("=" * 72)
        return False


# ── Per-ticker options scan ───────────────────────────────────────────────────

def scan_ticker_options(
    ticker: str,
    stock_price: float,
    debug: bool = False,
    relax: bool = False,
) -> list:
    results = []
    today   = datetime.today()

    try:
        tk          = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            if debug:
                print(f"  [{ticker}] No expirations returned by yfinance.")
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
                puts  = chain.puts
            except Exception as e:
                if debug:
                    print(f"  [{ticker}] {exp_str}  → could not fetch chain: {e}")
                continue

            if puts is None or puts.empty:
                if debug:
                    print(f"  [{ticker}] {exp_str}  DTE={dte}  → no puts returned")
                continue

            if debug:
                print(f"\n  [{ticker}] {exp_str}  DTE={dte}  ({len(puts)} puts)")

            for _, row in puts.iterrows():
                strike = _f(row.get("strike"))
                bid    = _f(row.get("bid"))
                ask    = _f(row.get("ask"))
                vol    = _i(row.get("volume"))
                oi     = _i(row.get("openInterest"))

                if strike <= 0 or bid <= 0 or ask <= 0:
                    if debug:
                        print(f"    K={strike:<6.2f}  bid={bid}  ask={ask}  → no market (skip)")
                    continue

                mid = (bid + ask) / 2.0

                if debug:
                    iv_raw = _f(row.get("impliedVolatility"))
                    iv_dbg = iv_raw if 0.01 < iv_raw < 5.0 else implied_volatility(mid, stock_price, strike, T)
                    delta_dbg = bs_put_delta(stock_price, strike, T, RISK_FREE_RATE, iv_dbg)
                    reasons = []
                    if mid < MIN_CREDIT:
                        reasons.append(f"credit={mid:.2f}<{MIN_CREDIT}")
                    if not relax:
                        if vol < MIN_VOLUME:
                            reasons.append(f"vol={vol}<{MIN_VOLUME}")
                        if oi  < MIN_OPEN_INTEREST:
                            reasons.append(f"OI={oi}<{MIN_OPEN_INTEREST}")
                    if strike * 100 > MAX_CASH_PER_CONTRACT:
                        reasons.append(f"cash={strike*100:.0f}>{MAX_CASH_PER_CONTRACT}")
                    if not (MIN_DELTA <= delta_dbg <= MAX_DELTA):
                        reasons.append(f"delta={delta_dbg:.3f} not in [{MIN_DELTA},{MAX_DELTA}]")
                    status = "PASS" if not reasons else "FAIL: " + " | ".join(reasons)
                    print(f"    K={strike:<6.2f}  mid={mid:.2f}  vol={vol:<5}  "
                          f"OI={oi:<6}  IV={iv_dbg*100:.0f}%  delta={delta_dbg:+.3f}  → {status}")

                if mid < MIN_CREDIT:
                    continue
                if not relax:
                    if vol < MIN_VOLUME:
                        continue
                    if oi  < MIN_OPEN_INTEREST:
                        continue
                if strike * 100 > MAX_CASH_PER_CONTRACT:
                    continue

                iv_raw = _f(row.get("impliedVolatility"))
                iv     = iv_raw if 0.01 < iv_raw < 5.0 else implied_volatility(mid, stock_price, strike, T)
                delta  = bs_put_delta(stock_price, strike, T, RISK_FREE_RATE, iv)

                if not (MIN_DELTA <= delta <= MAX_DELTA):
                    continue

                prob_profit   = (1.0 - abs(delta)) * 100
                cash_required = strike * 100
                premium       = mid    * 100
                roc_period    = (premium / cash_required) * 100
                roc_annual    = roc_period * (365.0 / dte)
                breakeven     = strike - mid
                be_pct_below  = ((stock_price - breakeven) / stock_price) * 100

                results.append({
                    "Ticker"        : ticker,
                    "Stock $"       : round(stock_price, 2),
                    "Strike"        : round(strike,      2),
                    "Expiration"    : exp_str,
                    "DTE"           : dte,
                    "Bid"           : round(bid,  2),
                    "Ask"           : round(ask,  2),
                    "Credit (mid)"  : round(mid,  2),
                    "Volume"        : vol,
                    "OI"            : oi,
                    "IV %"          : round(iv * 100, 1),
                    "Delta"         : round(delta,        3),
                    "Prob Profit %" : round(prob_profit,  1),
                    "Breakeven $"   : round(breakeven,    2),
                    "BE % Below"    : round(be_pct_below, 1),
                    "Cash Req $"    : round(cash_required, 2),
                    "Premium $"     : round(premium,      2),
                    "ROC %"         : round(roc_period,   2),
                    "Ann ROC %"     : round(roc_annual,   2),
                })

            time.sleep(WORKER_SLEEP)

    except Exception as e:
        if debug:
            print(f"  [{ticker}] Exception: {e}")

    return results


# ── Threaded worker ───────────────────────────────────────────────────────────

def _scan_one(ticker: str, relax: bool) -> dict:
    """
    Fetch price then scan puts for one ticker.
    Returns a result dict for the orchestrator to collect.
    """
    try:
        price = _f(yf.Ticker(ticker).fast_info.last_price)
    except Exception:
        price = 0.0

    if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
        return {"ticker": ticker, "price": price,
                "eligible": False, "contracts": []}

    contracts = scan_ticker_options(ticker, price, relax=relax)
    return {"ticker": ticker, "price": price,
            "eligible": True, "contracts": contracts}


# ── Single-ticker debug mode ──────────────────────────────────────────────────

def debug_single_ticker(ticker: str, relax: bool = False) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  DEBUG MODE  –  {ticker}{'  (--relax: vol/OI filters OFF)' if relax else ''}")
    print(sep)

    try:
        p = _f(yf.Ticker(ticker).fast_info.last_price)
    except Exception as e:
        print(f"  Cannot fetch price: {e}")
        return

    price = p
    print(f"  Current price : ${price:.2f}")
    if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
        print(f"  *** ${price:.2f} is outside the ${MIN_STOCK_PRICE}–${MAX_STOCK_PRICE} stock-price filter ***")
    flag = "  [vol/OI filters RELAXED]" if relax else ""
    print(f"  Criteria : DTE {MIN_DTE}–{MAX_DTE} | credit≥${MIN_CREDIT} | "
          f"vol≥{MIN_VOLUME} | OI≥{MIN_OPEN_INTEREST} | "
          f"delta[{MIN_DELTA},{MAX_DELTA}] | cash≤${MAX_CASH_PER_CONTRACT:,}{flag}\n")

    results = scan_ticker_options(ticker, price, debug=True, relax=relax)

    print(f"\n  → {len(results)} contracts passed all filters.\n")
    if results:
        df   = pd.DataFrame(results)
        cols = ["Strike", "Expiration", "DTE", "Credit (mid)", "IV %",
                "Delta", "Prob Profit %", "ROC %", "Ann ROC %"]
        print(tabulate(df[cols], headers="keys", tablefmt="simple",
                       showindex=False, floatfmt=".2f"))


# ── Full scan orchestrator ────────────────────────────────────────────────────

def run_scan(relax: bool = False, workers: int = DEFAULT_WORKERS) -> pd.DataFrame:
    sep = "=" * 72
    print(f"\n{sep}")
    mode_note = "  [RELAXED: vol/OI filters OFF]" if relax else ""
    print(f"  CASH SECURED PUT (CSP) OPTIONS SCANNER  –  Yahoo Finance{mode_note}")
    print(sep)
    print(f"  Stock price       : ${MIN_STOCK_PRICE:.0f} – ${MAX_STOCK_PRICE:.0f}")
    print(f"  DTE               : {MIN_DTE} – {MAX_DTE} days")
    if relax:
        print(f"  Min volume        : RELAXED (0)")
        print(f"  Min open interest : RELAXED (0)")
    else:
        print(f"  Min volume        : {MIN_VOLUME:,}  ← requires market-hours data")
        print(f"  Min open interest : {MIN_OPEN_INTEREST:,}")
    print(f"  Delta range       : {MIN_DELTA} to {MAX_DELTA}")
    print(f"  Min credit        : ${MIN_CREDIT:.2f}")
    print(f"  Max cash/contract : ${MAX_CASH_PER_CONTRACT:,.0f}")
    print(f"  Risk-free rate    : {RISK_FREE_RATE*100:.1f}%")
    print(f"  Universe source   : yfinance screener + Russell 1000 + S&P 500 + Nasdaq-100")
    print(f"  Concurrent workers: {workers}  (sleep {WORKER_SLEEP}s/call)")
    print(f"{sep}\n")

    universe       = get_ticker_universe()
    total          = len(universe)
    all_results    = []
    eligible_count = 0
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

                pbar.set_postfix(
                    eligible=eligible_count,
                    contracts=found_total[0],
                    ticker=result["ticker"],
                )
                pbar.update(1)

    print(f"\nScan complete:")
    print(f"  Tickers scanned       : {total}")
    print(f"  Passed price filter   : {eligible_count}")
    print(f"  Qualifying contracts  : {len(all_results)}\n")

    if not all_results:
        print("No contracts met all criteria.")
        if not relax and not market_is_open():
            print("Tip: Market appears closed. Try:  python options_30delta.py --relax")
        return pd.DataFrame()

    df = (pd.DataFrame(all_results)
          .sort_values("Ann ROC %", ascending=False)
          .reset_index(drop=True))
    return df


# ── Results display ───────────────────────────────────────────────────────────

def display_results(df: pd.DataFrame) -> None:
    if df.empty:
        return

    sep = "=" * 72

    print(f"\n{sep}")
    print("  ALL QUALIFYING PUTS  –  ranked by Annualised Return on Capital")
    print(sep)
    display_cols = [
        "Ticker", "Stock $", "Strike", "Expiration", "DTE",
        "Credit (mid)", "Delta", "Prob Profit %",
        "Breakeven $", "BE % Below", "ROC %", "Ann ROC %",
    ]
    print(tabulate(df[display_cols], headers="keys", tablefmt="rounded_outline",
                   showindex=True, floatfmt=".2f"))

    top_cols = ["Ticker", "Strike", "Expiration", "DTE",
                "Credit (mid)", "IV %", "Delta", "Prob Profit %",
                "BE % Below", "ROC %", "Ann ROC %"]

    print(f"\n{sep}")
    print("  TOP 10  –  Highest Annualised Return on Capital")
    print(sep)
    print(tabulate(df.nlargest(10, "Ann ROC %")[top_cols].reset_index(drop=True),
                   headers="keys", tablefmt="simple", showindex=True, floatfmt=".2f"))

    print(f"\n{sep}")
    print("  TOP 10  –  Highest Probability of Profit")
    print(sep)
    print(tabulate(df.nlargest(10, "Prob Profit %")[top_cols].reset_index(drop=True),
                   headers="keys", tablefmt="simple", showindex=True, floatfmt=".2f"))

    print(f"\n{sep}")
    print("  PORTFOLIO SUMMARY STATISTICS")
    print(sep)
    print(f"  Total qualifying contracts : {len(df)}")
    print(f"  Unique tickers             : {df['Ticker'].nunique()}")
    print(f"  Average Ann ROC            : {df['Ann ROC %'].mean():.2f}%")
    print(f"  Median Ann ROC             : {df['Ann ROC %'].median():.2f}%")
    print(f"  Best Ann ROC               : {df['Ann ROC %'].max():.2f}%")
    print(f"  Average Prob Profit        : {df['Prob Profit %'].mean():.1f}%")
    print(f"  Average DTE                : {df['DTE'].mean():.0f} days")
    print(f"  Avg credit collected       : ${df['Credit (mid)'].mean():.2f}")
    print()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = f"csp_scan_{ts}.csv"
    xlsx_path = f"csp_scan_{ts}.xlsx"

    df.to_csv(csv_path, index=False)
    print(f"  CSV  saved → {csv_path}")

    try:
        from export_to_excel import export_scan_results
        export_scan_results(df, output_path=xlsx_path, source_csv=csv_path)
        print(f"  XLSX saved → {xlsx_path}  (open in Excel)")
    except ImportError:
        print("  Note: export_to_excel.py not found — skipping Excel export.")

    try:
        from export_to_html import export_scan_results as _html
        html_path = f"csp_scan_{ts}.html"
        _html(df, output_path=html_path, source_csv=csv_path)
        print(f"  HTML saved → {html_path}  (open in any browser ✓)")
    except ImportError:
        print("  Note: export_to_html.py not found — skipping HTML export.")
    except Exception as e:
        print(f"  HTML export error: {e}")

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
            print("Usage: python options_30delta.py --ticker SYMBOL [--relax]")
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
