#!/usr/bin/env python3
"""
scanner.py  —  Web-friendly wrapper around options_30delta.py
=============================================================
Exposes run_scan_web() which calls the scanner's ThreadPoolExecutor
loop but fires progress callbacks instead of printing to stdout,
so the Flask SSE stream can relay live updates to the browser.
"""

from __future__ import annotations
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import pandas as pd

# ── Import everything from the main scanner ───────────────────────────────────
from options_30delta import (
    MIN_STOCK_PRICE, MAX_STOCK_PRICE,
    DEFAULT_WORKERS, WORKER_SLEEP,
    get_ticker_universe,
    scan_ticker_options,
    market_is_open,
    _f,
)
import yfinance as yf


def _scan_one_web(ticker: str, relax: bool) -> dict:
    """Worker: fetch price, scan puts, return result dict."""
    try:
        price = _f(yf.Ticker(ticker).fast_info.last_price)
    except Exception:
        price = 0.0

    if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
        return {"ticker": ticker, "price": price, "eligible": False, "contracts": []}

    contracts = scan_ticker_options(ticker, price, relax=relax)
    return {"ticker": ticker, "price": price, "eligible": True, "contracts": contracts}


def run_scan_web(
    relax: bool = False,
    workers: int = DEFAULT_WORKERS,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> pd.DataFrame:
    """
    Run a full CSP scan and stream progress via callbacks.

    Parameters
    ----------
    relax       : bypass volume/OI filters
    workers     : number of concurrent threads
    on_progress : called after each ticker completes
                  signature: (scanned, total, eligible, contracts, ticker)
    on_log      : called with plain-text log messages
                  signature: (msg: str)
    """

    def log(msg: str):
        if on_log:
            on_log(msg)

    log("Building ticker universe...")
    universe = get_ticker_universe()
    total    = len(universe)
    log(f"Universe ready: {total} unique tickers to scan.")

    if not market_is_open() and not relax:
        log("WARNING: Market appears CLOSED. Volume filter will reject most contracts. Consider enabling 'Relax Filters'.")

    all_results    = []
    eligible_count = 0
    found_lock     = threading.Lock()
    found_total    = [0]
    scanned_count  = [0]

    log(f"Starting scan — {workers} concurrent workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_scan_one_web, ticker, relax): ticker
            for ticker in universe
        }

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                with found_lock:
                    scanned_count[0] += 1
                if on_progress:
                    on_progress(scanned_count[0], total, eligible_count,
                                found_total[0], "")
                continue

            with found_lock:
                scanned_count[0] += 1
                sc = scanned_count[0]

            if result.get("eligible"):
                eligible_count += 1
                contracts = result["contracts"]
                all_results.extend(contracts)
                with found_lock:
                    found_total[0] += len(contracts)

            if on_progress:
                on_progress(sc, total, eligible_count,
                            found_total[0], result["ticker"])

    log(f"Scan complete — {eligible_count} in price range, {len(all_results)} qualifying contracts.")

    if not all_results:
        return pd.DataFrame()

    return (
        pd.DataFrame(all_results)
        .sort_values("Ann ROC %", ascending=False)
        .reset_index(drop=True)
    )
