#!/usr/bin/env python3
"""
scanner_ditm.py  —  Web-friendly wrapper around ditm_call_scanner.py
=====================================================================
Exposes run_ditm_scan_web() which fires progress callbacks instead of
printing to stdout, so the Flask SSE stream can relay live updates.
"""

from __future__ import annotations
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd
import yfinance as yf

from ditm_call_scanner import (
    MIN_STOCK_PRICE, MAX_STOCK_PRICE,
    DEFAULT_WORKERS,
    get_ticker_universe,
    scan_ticker_calls,
    stock_above_emas,
    market_is_open,
    _f,
)


def _scan_one_ditm(ticker: str, relax: bool) -> dict:
    """Worker: fetch price, check EMA trend, scan calls, return result dict."""
    try:
        price = _f(yf.Ticker(ticker).fast_info.last_price)
    except Exception:
        price = 0.0

    if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
        return {"ticker": ticker, "price": price,
                "eligible": False, "skip_reason": "price_range", "contracts": []}

    passes_ema, ema8, ema20 = stock_above_emas(ticker, price)
    if not passes_ema:
        return {"ticker": ticker, "price": price,
                "eligible": False, "skip_reason": "ema_filter",
                "ema8": ema8, "ema20": ema20, "contracts": []}

    contracts = scan_ticker_calls(ticker, price, ema8, ema20, relax=relax)
    return {"ticker": ticker, "price": price,
            "eligible": True, "ema8": ema8, "ema20": ema20,
            "contracts": contracts}


def run_ditm_scan_web(
    relax: bool = False,
    workers: int = DEFAULT_WORKERS,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> pd.DataFrame:
    """
    Run a full DITM call scan and stream progress via callbacks.

    Parameters
    ----------
    relax       : bypass OI filter
    workers     : concurrent threads
    on_progress : (scanned, total, eligible, ema_fail, contracts, ticker)
    on_log      : (msg: str)
    """

    def log(msg: str):
        if on_log:
            on_log(msg)

    log("Building ticker universe...")
    universe = get_ticker_universe()
    total    = len(universe)
    log(f"Universe ready: {total} unique tickers to scan.")

    if not market_is_open() and not relax:
        log("WARNING: Market appears CLOSED. OI data may be stale. Consider enabling 'Relax Filters'.")

    all_results    = []
    eligible_count = 0
    ema_fail_count = 0
    found_lock     = threading.Lock()
    found_total    = [0]
    scanned_count  = [0]

    log(f"Starting DITM scan — {workers} concurrent workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_scan_one_ditm, ticker, relax): ticker
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
                                ema_fail_count, found_total[0], "")
                continue

            with found_lock:
                scanned_count[0] += 1
                sc = scanned_count[0]

            skip = result.get("skip_reason", "")
            if result.get("eligible"):
                eligible_count += 1
                contracts = result["contracts"]
                all_results.extend(contracts)
                with found_lock:
                    found_total[0] += len(contracts)
            elif skip == "ema_filter":
                ema_fail_count += 1

            if on_progress:
                on_progress(sc, total, eligible_count,
                            ema_fail_count, found_total[0], result["ticker"])

    log(f"Scan complete — {eligible_count} passed price+EMA filters, "
        f"{len(all_results)} qualifying contracts.")

    if not all_results:
        return pd.DataFrame()

    return (
        pd.DataFrame(all_results)
        .sort_values(["Ann ROC %", "Prob Profit %"], ascending=[False, False])
        .reset_index(drop=True)
    )
