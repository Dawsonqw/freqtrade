#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freqtrade.commands.optimize_commands import setup_optimize_configuration
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting

from user_custom.rolling_backtest.src.rolling_runner import RollingBacktestRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("run_rolling_backtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run chunked rolling backtest")
    p.add_argument("--config", required=True, help="Freqtrade config path")
    p.add_argument("--strategy", default=None, help="Override strategy name")
    p.add_argument("--timerange", default=None, help="Override timerange")
    p.add_argument("--window-days", type=int, default=14, help="Chunk size in days")
    p.add_argument("--datadir", default="/data/freqtrade_data", help="OHLCV data directory")
    p.add_argument("--user-data-dir", default="/root/workspace/freqtrade/user_data", help="freqtrade user_data directory")
    p.add_argument("--pairs-file", default=None, help="CSV/TXT with pair list")
    p.add_argument("--output-json", default="user_custom/rolling_backtest/output/rolling_backtest_result.json")
    p.add_argument("--log-file", default=None)
    p.add_argument("--fail-fast", action="store_true", help="Stop immediately when a window fails")
    p.add_argument(
        "--max-failed-windows",
        type=int,
        default=0,
        help="Abort when failed window count exceeds this threshold (0 = unlimited)",
    )
    return p.parse_args()


def setup_file_logger(log_file: str | None) -> None:
    if not log_file:
        return
    fp = Path(log_file)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(fp, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def load_pairs(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")
    if p.suffix.lower() == ".csv":
        out = []
        with p.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            if "pair" not in rd.fieldnames:
                raise ValueError("CSV must include 'pair' column")
            for row in rd:
                out.append(row["pair"].strip())
        return sorted(set(x for x in out if x))

    # TXT fallback: one pair per line
    lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
    return sorted(set(x for x in lines if x and not x.startswith("#")))


def build_args(ns: argparse.Namespace) -> dict[str, Any]:
    args: dict[str, Any] = {
        "config": [ns.config],
        "strategy": ns.strategy,
        "timerange": ns.timerange,
        "verbosity": 0,
        "timeframe": None,
        "timeframe_detail": None,
        "datadir": ns.datadir,
        "user_data_dir": ns.user_data_dir,
        "export": "none",
        "cache": "none",
        "backtest_breakdown": [],
        "enable_protections": False,
    }
    return args


def main() -> None:
    ns = parse_args()
    setup_file_logger(ns.log_file)

    args = build_args(ns)
    config = setup_optimize_configuration(args, RunMode.BACKTEST)

    if ns.pairs_file:
        pairs = load_pairs(ns.pairs_file)
        if not pairs:
            raise RuntimeError("Pairs file is empty")
        config["pairs"] = pairs
        config["exchange"]["pair_whitelist"] = pairs
        config["pairlists"] = [{"method": "StaticPairList"}]
        logger.info("Loaded pairs from file: %s", len(pairs))

    backtesting = Backtesting(config)
    runner = RollingBacktestRunner(backtesting=backtesting, window_days=ns.window_days)
    result = runner.run(
        output_json=Path(ns.output_json),
        fail_fast=ns.fail_fast,
        max_failed_windows=ns.max_failed_windows,
    )

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
