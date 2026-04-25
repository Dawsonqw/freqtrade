#!/usr/bin/env python3
"""Scan all feather data files and build a comprehensive pair_availability.json.

This script reads every *.feather file in the data directory, extracts the
min/max date per pair per timeframe, and writes a single JSON file that the
rolling backtest framework can use to know which pairs existed at any point
in time.

Usage:
    python build_pair_availability.py --datadir /data/freqtrade_data/futures \
        --output /data/freqtrade_data/_meta/pair_availability.json

Output JSON structure:
{
  "generated_at": "2026-04-25T...",
  "data_dir": "/data/freqtrade_data/futures",
  "total_pairs": 535,
  "pairs": {
    "BTC/USDT:USDT": {
      "earliest_start": "2021-04-25T12:00:00+00:00",
      "latest_end": "2026-04-24T18:00:00+00:00",
      "timeframes": {
        "5m":  {"start": "...", "end": "...", "rows": 123456},
        "15m": {"start": "...", "end": "...", "rows": 78901},
        ...
      }
    },
    ...
  }
}
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pattern: {BASE}_{QUOTE}_{MARGIN}-{TIMEFRAME}-futures.feather
FILE_RE = re.compile(
    r"^(?P<base>.+?)_(?P<quote>[A-Z]+)_(?P<margin>[A-Z]+)"
    r"-(?P<timeframe>\d+[mhdwM])-futures\.feather$"
)


def filename_to_pair(base: str, quote: str, margin: str) -> str:
    """Convert filename components to freqtrade pair format: BASE/QUOTE:MARGIN."""
    return f"{base}/{quote}:{margin}"


def scan_datadir(datadir: Path) -> dict:
    """Scan all feather files and return pair availability dict."""
    pairs: dict[str, dict] = {}
    files = sorted(datadir.glob("*-futures.feather"))
    total = len(files)
    logger.info("Found %d feather files in %s", total, datadir)

    for i, fpath in enumerate(files, 1):
        m = FILE_RE.match(fpath.name)
        if not m:
            logger.warning("Skipping unrecognized file: %s", fpath.name)
            continue

        base = m.group("base")
        quote = m.group("quote")
        margin = m.group("margin")
        tf = m.group("timeframe")
        pair = filename_to_pair(base, quote, margin)

        try:
            df = pd.read_feather(fpath, columns=["date"])
            if df.empty:
                logger.warning("Empty file: %s", fpath.name)
                continue
            start = df["date"].min()
            end = df["date"].max()
            rows = len(df)
        except Exception as exc:
            logger.error("Failed to read %s: %s", fpath.name, exc)
            continue

        # Ensure timezone-aware ISO strings
        start_iso = start.isoformat() if hasattr(start, "isoformat") else str(start)
        end_iso = end.isoformat() if hasattr(end, "isoformat") else str(end)

        if pair not in pairs:
            pairs[pair] = {
                "earliest_start": start_iso,
                "latest_end": end_iso,
                "timeframes": {},
            }

        pairs[pair]["timeframes"][tf] = {
            "start": start_iso,
            "end": end_iso,
            "rows": rows,
        }

        # Update global earliest/latest for the pair
        if start_iso < pairs[pair]["earliest_start"]:
            pairs[pair]["earliest_start"] = start_iso
        if end_iso > pairs[pair]["latest_end"]:
            pairs[pair]["latest_end"] = end_iso

        if i % 100 == 0 or i == total:
            logger.info("Progress: %d/%d files scanned", i, total)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pair availability JSON from feather data")
    parser.add_argument(
        "--datadir",
        type=Path,
        default=Path("/data/freqtrade_data/futures"),
        help="Directory containing feather files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/freqtrade_data/_meta/pair_availability.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not args.datadir.is_dir():
        logger.error("Data directory does not exist: %s", args.datadir)
        sys.exit(1)

    pairs = scan_datadir(args.datadir)

    result = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "data_dir": str(args.datadir),
        "total_pairs": len(pairs),
        "pairs": dict(sorted(pairs.items())),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Written pair_availability.json: %d pairs, %d total timeframe entries",
        len(pairs),
        sum(len(v["timeframes"]) for v in pairs.values()),
    )


if __name__ == "__main__":
    main()
