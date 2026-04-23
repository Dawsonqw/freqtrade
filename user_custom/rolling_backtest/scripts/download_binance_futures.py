#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

from freqtrade.data.history import get_datahandler
from freqtrade.data.history.history_utils import download_data
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.resolvers.exchange_resolver import ExchangeResolver

from user_custom.rolling_backtest.src.manifest import build_symbol_meta, read_manifest, write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("download_binance_futures")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Binance futures OHLCV in batches")
    p.add_argument("--data-dir", default="/data/freqtrade_data", help="freqtrade datadir")
    p.add_argument("--meta-dir", default="/data/freqtrade_data/_meta", help="metadata output directory")
    p.add_argument("--endpoint", default="https://demo-fapi.binance.com", help="Binance API base")
    p.add_argument("--quote", default="USDT", help="Quote asset")
    p.add_argument("--years", type=int, default=5, help="Lookback years")
    p.add_argument("--timeframes", nargs="+", default=["5m", "15m", "1h", "4h", "1d", "1w"])
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--max-pairs", type=int, default=0, help="For smoke test. 0 means all.")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--coverage-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def fetch_exchange_info(endpoint: str) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/fapi/v1/exchangeInfo"
    logger.info("Fetching exchange info: %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_download_config(
    *,
    data_dir: Path,
    endpoint: str,
    pairs: list[str],
    timeframe: str,
    timerange: str,
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
        "exchange": {
            "name": "binance",
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
                # keep public endpoint aligned with requested demo endpoint
                "urls": {
                    "api": {
                        "fapiPublic": f"{endpoint.rstrip('/')}/fapi/v1",
                        "fapiPrivate": f"{endpoint.rstrip('/')}/fapi/v1",
                    }
                },
            },
        },
    }


def write_coverage(
    *,
    data_dir: Path,
    manifest_rows,
    timeframes: list[str],
    out_csv: Path,
) -> None:
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


def chunked(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    meta_dir = Path(args.meta_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = meta_dir / "binance_futures_manifest.csv"
    coverage_path = meta_dir / "binance_futures_coverage.csv"

    if not args.coverage_only:
        info = fetch_exchange_info(args.endpoint)
        symbols = info.get("symbols", [])
        rows = build_symbol_meta(symbols, args.quote, args.years)
        if args.max_pairs and args.max_pairs > 0:
            rows = rows[: args.max_pairs]
        write_manifest(rows, manifest_path)
        logger.info("Manifest written: %s (%s pairs)", manifest_path, len(rows))
    else:
        rows = read_manifest(manifest_path)

    if args.manifest_only:
        return

    if args.coverage_only:
        logger.info("Coverage-only mode; skip download.")
    else:
        by_pair = {r.pair: r for r in rows}
        all_pairs = sorted(by_pair.keys())
        logger.info(
            "Start download: pairs=%s, timeframes=%s, batch_size=%s",
            len(all_pairs),
            args.timeframes,
            args.batch_size,
        )

        for tf in args.timeframes:
            for batch in chunked(all_pairs, args.batch_size):
                # global earliest start in this batch; pairs listed later naturally return shorter data
                min_start = min(by_pair[p].planned_start for p in batch)
                timerange = f"{int(min_start.timestamp())}-"

                cfg = build_download_config(
                    data_dir=data_dir,
                    endpoint=args.endpoint,
                    pairs=batch,
                    timeframe=tf,
                    timerange=timerange,
                )

                logger.info(
                    "Downloading timeframe=%s batch=%s..%s (%s pairs) timerange=%s",
                    tf,
                    batch[0],
                    batch[-1],
                    len(batch),
                    timerange,
                )

                if args.dry_run:
                    continue

                exchange = ExchangeResolver.load_exchange(cfg, validate=False)
                download_data(cfg, exchange)

    write_coverage(data_dir=data_dir, manifest_rows=rows, timeframes=args.timeframes, out_csv=coverage_path)
    logger.info("Coverage written: %s", coverage_path)

    summary = {
        "manifest": str(manifest_path),
        "coverage": str(coverage_path),
        "pair_count": len(rows),
        "timeframes": args.timeframes,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
