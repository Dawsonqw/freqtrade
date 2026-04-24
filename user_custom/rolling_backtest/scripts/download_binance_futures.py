#!/usr/bin/env python3
"""
Incremental Binance USDT-M Futures OHLCV downloader.

Key design:
  1. Single Exchange instance – created once, reused for all downloads.
  2. Persistent state file (download_state.json) – tracks completed
     (pair, timeframe) combos so a restart skips finished work.
  3. True incremental – freqtrade's download_data appends to existing
     feather files; we just make sure we call it with the right timerange.
  4. Graceful shutdown on SIGINT/SIGTERM – saves state before exit.
  5. Real-time progress counter written to state file every batch.

Usage:
  # Full run (all pairs, 6 timeframes, 5 years)
  python download_binance_futures.py

  # Resume after interruption (automatic – reads state file)
  python download_binance_futures.py

  # Manifest only (list pairs)
  python download_binance_futures.py --manifest-only

  # Smoke test
  python download_binance_futures.py --max-pairs 3 --timeframes 5m
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

from freqtrade.data.history.history_utils import download_data
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.resolvers.exchange_resolver import ExchangeResolver

from user_custom.rolling_backtest.src.manifest import (
    SymbolMeta,
    build_symbol_meta,
    read_manifest,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("download_binance_futures")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_SHUTDOWN_REQUESTED = False


def _handle_signal(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    logger.warning("Shutdown signal received (%s). Will stop after current batch.", signum)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# State management – survives restarts
# ---------------------------------------------------------------------------
STATE_VERSION = 2


@dataclass
class DownloadState:
    """Persistent download progress tracker."""

    completed: set[tuple[str, str]] = field(default_factory=set)  # (pair, timeframe)
    failed: dict[str, str] = field(default_factory=dict)  # "pair|tf" -> last error
    total_tasks: int = 0
    started_at: str = ""

    @classmethod
    def load(cls, path: Path) -> "DownloadState":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("version") != STATE_VERSION:
                logger.info("State version mismatch, starting fresh.")
                return cls()
            s = cls()
            s.completed = {tuple(x) for x in raw.get("completed", [])}
            s.failed = raw.get("failed", {})
            s.total_tasks = raw.get("total_tasks", 0)
            s.started_at = raw.get("started_at", "")
            return s
        except Exception:
            logger.warning("Corrupt state file, starting fresh.")
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": STATE_VERSION,
            "completed": sorted(list(self.completed)),
            "failed": self.failed,
            "total_tasks": self.total_tasks,
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
            "started_at": self.started_at,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
        # Atomic write
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.rename(path)

    def mark_done(self, pair: str, tf: str) -> None:
        self.completed.add((pair, tf))
        key = f"{pair}|{tf}"
        self.failed.pop(key, None)

    def mark_failed(self, pair: str, tf: str, error: str) -> None:
        self.failed[f"{pair}|{tf}"] = error

    def is_done(self, pair: str, tf: str) -> bool:
        return (pair, tf) in self.completed

    @property
    def progress_pct(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return len(self.completed) / self.total_tasks * 100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental Binance futures OHLCV downloader")
    p.add_argument("--data-dir", default="/data/freqtrade_data", help="Freqtrade datadir")
    p.add_argument("--meta-dir", default="/data/freqtrade_data/_meta", help="Metadata output dir")
    p.add_argument("--endpoint", default="https://fapi.binance.com", help="Binance fapi endpoint")
    p.add_argument("--quote", default="USDT")
    p.add_argument("--years", type=int, default=5, help="Lookback years")
    p.add_argument("--timeframes", nargs="+", default=["5m", "15m", "1h", "4h", "1d", "1w"])
    p.add_argument("--batch-size", type=int, default=40, help="Pairs per download_data call")
    p.add_argument("--max-pairs", type=int, default=0, help="Limit pairs (0=all, for smoke test)")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--coverage-only", action="store_true")
    p.add_argument("--reset-state", action="store_true", help="Ignore existing state, start fresh")
    p.add_argument("--prepend", action="store_true", help="Enable prepend mode (for incremental updates, disables parallel)")
    p.add_argument("--retry-failed", action="store_true", help="Retry previously failed pairs")
    p.add_argument("--retries", type=int, default=5, help="Retries per batch")
    p.add_argument("--retry-sleep", type=float, default=3.0, help="Base backoff seconds")
    p.add_argument("--sleep-between-batches", type=float, default=0.0)
    p.add_argument("--save-interval", type=int, default=1, help="Save state every N batches")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Exchange singleton
# ---------------------------------------------------------------------------
_EXCHANGE = None


def get_exchange(data_dir: Path, pairs: list[str]) -> Any:
    """Create exchange once and cache it."""
    global _EXCHANGE
    if _EXCHANGE is not None:
        return _EXCHANGE

    key = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_API_SECRET", "")

    cfg = {
        "runmode": RunMode.UTIL_EXCHANGE,
        "dry_run": True,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "tradable_balance_ratio": 0.99,
        "dataformat_ohlcv": "feather",
        "trading_mode": TradingMode.FUTURES,
        "margin_mode": "isolated",
        "candle_type_def": CandleType.FUTURES,
        "new_pairs_days": 30,
        "pairs": pairs[:1],  # minimal for init
        "timeframes": ["1h"],
        "datadir": data_dir,
        "exchange": {
            "name": "binance",
            "only_from_ccxt": True,
            "key": key,
            "secret": secret,
            "pair_whitelist": pairs[:1],
            "pair_blacklist": [],
            "ccxt_config": {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                },
            },
        },
    }
    logger.info("Initializing Binance exchange (one-time)...")
    _EXCHANGE = ExchangeResolver.load_exchange(cfg, validate=False)
    logger.info("Exchange ready.")
    return _EXCHANGE


# ---------------------------------------------------------------------------
# Build download config (reuses existing exchange)
# ---------------------------------------------------------------------------
def build_download_config(
    *,
    data_dir: Path,
    pairs: list[str],
    timeframe: str,
    timerange: str,
    prepend: bool = False,
) -> dict[str, Any]:
    key = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_API_SECRET", "")
    return {
        "runmode": RunMode.UTIL_EXCHANGE,
        "dry_run": True,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "tradable_balance_ratio": 0.99,
        "dataformat_ohlcv": "feather",
        "trading_mode": TradingMode.FUTURES,
        "margin_mode": "isolated",
        "candle_type_def": CandleType.FUTURES,
        "new_pairs_days": 30,
        "pairs": pairs,
        "timeframes": [timeframe],
        "timerange": timerange,
        "datadir": data_dir,
        "prepend_data": prepend,
        "erase": False,
        "candle_types": [CandleType.FUTURES],
        "no_parallel_download": prepend,  # parallel incompatible with prepend
        "exchange": {
            "name": "binance",
            "only_from_ccxt": True,
            "key": key,
            "secret": secret,
            "pair_whitelist": pairs,
            "pair_blacklist": [],
            "ccxt_config": {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Coverage writer
# ---------------------------------------------------------------------------
def write_coverage(
    *,
    data_dir: Path,
    manifest_rows: list[SymbolMeta],
    timeframes: list[str],
    out_csv: Path,
) -> None:
    from freqtrade.data.history import get_datahandler

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dh = get_datahandler(data_dir, "feather")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "timeframe", "candle_type", "min_date", "max_date", "candles"])
        for row in manifest_rows:
            for tf in timeframes:
                try:
                    start, end, length = dh.ohlcv_data_min_max(row.pair, tf, CandleType.FUTURES)
                    w.writerow([row.pair, tf, "futures", start.isoformat(), end.isoformat(), length])
                except Exception:
                    w.writerow([row.pair, tf, "futures", "", "", 0])


# ---------------------------------------------------------------------------
# Exchange info
# ---------------------------------------------------------------------------
def fetch_exchange_info(endpoint: str) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/fapi/v1/exchangeInfo"
    logger.info("Fetching exchange info: %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    meta_dir = Path(args.meta_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = meta_dir / "binance_futures_manifest.csv"
    coverage_path = meta_dir / "binance_futures_coverage.csv"
    state_path = meta_dir / "download_state.json"

    # ---- Manifest ----
    if not args.coverage_only:
        info = fetch_exchange_info(args.endpoint)
        symbols = info.get("symbols", [])
        rows = build_symbol_meta(symbols, args.quote, args.years)
        if args.max_pairs > 0:
            rows = rows[: args.max_pairs]
        write_manifest(rows, manifest_path)
        logger.info("Manifest: %s pairs written to %s", len(rows), manifest_path)
    else:
        rows = read_manifest(manifest_path)

    if args.manifest_only:
        logger.info("Manifest-only mode. Done.")
        return

    # ---- State ----
    state = DownloadState() if args.reset_state else DownloadState.load(state_path)
    if args.retry_failed:
        # Clear failed entries so they get retried
        logger.info("Retrying %d previously failed tasks.", len(state.failed))
        for key in list(state.failed.keys()):
            pair, tf = key.split("|", 1)
            state.completed.discard((pair, tf))
        state.failed.clear()

    # Build task list: (pair, timeframe, timerange_str)
    by_pair = {r.pair: r for r in rows}
    all_pairs = sorted(by_pair.keys())
    tasks: list[tuple[str, str, str]] = []
    for tf in args.timeframes:
        for pair in all_pairs:
            if not state.is_done(pair, tf):
                meta = by_pair[pair]
                tr = f"{int(meta.planned_start.timestamp())}-"
                tasks.append((pair, tf, tr))

    state.total_tasks = len(tasks) + len(state.completed)
    if not state.started_at:
        state.started_at = datetime.now(tz=UTC).isoformat()

    logger.info(
        "Download plan: %d tasks remaining, %d already done (%.1f%%), %d total pairs, %d timeframes",
        len(tasks),
        len(state.completed),
        state.progress_pct,
        len(all_pairs),
        len(args.timeframes),
    )

    if args.coverage_only or len(tasks) == 0:
        if len(tasks) == 0:
            logger.info("All tasks already completed!")
        else:
            logger.info("Coverage-only mode; skip download.")
        write_coverage(data_dir=data_dir, manifest_rows=rows, timeframes=args.timeframes, out_csv=coverage_path)
        logger.info("Coverage written: %s", coverage_path)
        state.save(state_path)
        return

    # ---- Init exchange once ----
    exchange = get_exchange(data_dir, all_pairs)

    # Group tasks by timeframe for efficient batching
    from collections import defaultdict
    tf_tasks: dict[str, list[str]] = defaultdict(list)  # tf -> [pair]
    for pair, tf, tr in tasks:
        tf_tasks[tf].append(pair)

    batch_counter = 0
    total_downloaded = 0

    for tf in args.timeframes:
        pending_pairs = tf_tasks.get(tf, [])
        if not pending_pairs:
            logger.info("Timeframe %s: all done, skipping.", tf)
            continue

        for batch in chunked(pending_pairs, args.batch_size):
            # Use the earliest planned_start in the batch as unified timerange
            # (freqtrade only fetches from actual pair listing date anyway)
            min_start = min(by_pair[p].planned_start for p in batch)
            tr = f"{int(min_start.timestamp())}-"

            if _SHUTDOWN_REQUESTED:
                logger.warning("Shutdown requested. Saving state and exiting.")
                state.save(state_path)
                sys.exit(0)

            batch_counter += 1
            logger.info(
                "[%s] Batch %d | tf=%s pairs=%d (%s..%s) | Progress: %d/%d (%.1f%%)",
                datetime.now(tz=UTC).strftime("%H:%M:%S"),
                batch_counter,
                tf,
                len(batch),
                batch[0][:20],
                batch[-1][:20],
                len(state.completed),
                state.total_tasks,
                state.progress_pct,
            )

            cfg = build_download_config(
                data_dir=data_dir,
                pairs=batch,
                timeframe=tf,
                timerange=tr,
                prepend=args.prepend,
            )

            success = False
            last_error = ""
            for attempt in range(1, args.retries + 1):
                try:
                    download_data(cfg, exchange)
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < args.retries:
                        wait = args.retry_sleep * (2 ** (attempt - 1))
                        logger.warning(
                            "Batch failed (attempt %d/%d): %s. Retrying in %.1fs",
                            attempt, args.retries, exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error("Batch failed after %d attempts: %s", args.retries, exc)

            if success:
                for p in batch:
                    state.mark_done(p, tf)
                    total_downloaded += 1
            else:
                # Check which pairs actually got data written despite batch-level error
                from freqtrade.data.history import get_datahandler
                dh = get_datahandler(data_dir, "feather")
                for p in batch:
                    try:
                        _, _, length = dh.ohlcv_data_min_max(p, tf, CandleType.FUTURES)
                        if length > 0:
                            state.mark_done(p, tf)
                            total_downloaded += 1
                            logger.info("Pair %s/%s partially succeeded (%d candles), marked done.", p, tf, length)
                            continue
                    except Exception:
                        pass
                    state.mark_failed(p, tf, last_error)

            # Periodic state save
            if batch_counter % args.save_interval == 0:
                state.save(state_path)

            if args.sleep_between_batches > 0:
                time.sleep(args.sleep_between_batches)

    # ---- Final save & coverage ----
    state.save(state_path)
    logger.info("Download complete. %d tasks done, %d failed.", len(state.completed), len(state.failed))

    write_coverage(data_dir=data_dir, manifest_rows=rows, timeframes=args.timeframes, out_csv=coverage_path)
    logger.info("Coverage written: %s", coverage_path)

    # Summary report (backward compatible)
    report_path = meta_dir / "download_report.json"
    summary = {
        "manifest": str(manifest_path),
        "coverage": str(coverage_path),
        "state": str(state_path),
        "pair_count": len(rows),
        "timeframes": args.timeframes,
        "total_tasks": state.total_tasks,
        "completed": len(state.completed),
        "failed_count": len(state.failed),
        "failures": state.failed,
        "progress_pct": round(state.progress_pct, 2),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if state.failed:
        logger.warning("Some tasks failed. Re-run with --retry-failed to retry them.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
