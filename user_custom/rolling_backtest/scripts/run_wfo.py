#!/usr/bin/env python3
"""CLI entry point for Walk-Forward Optimization."""
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

from user_custom.rolling_backtest.src.wfo_engine import WFOEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("run_wfo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-Forward Optimization: IS optimize → OOS validate across sliding folds",
    )
    # Required
    p.add_argument("--config", required=True, help="Freqtrade config path")

    # Strategy
    p.add_argument("--strategy", default=None, help="Override strategy name")

    # WFO-specific
    p.add_argument("--is-days", type=int, default=90, help="In-sample window (days)")
    p.add_argument("--oos-days", type=int, default=30, help="Out-of-sample window (days)")
    p.add_argument("--step-days", type=int, default=None, help="Slide step (days, default=oos-days)")
    p.add_argument("--n-epochs", type=int, default=100, help="Optuna trials per fold")
    p.add_argument(
        "--loss-function",
        default="SharpeHyperOptLoss",
        help="Hyperopt loss function (default: SharpeHyperOptLoss)",
    )
    p.add_argument(
        "--spaces",
        nargs="+",
        default=["buy", "sell"],
        help="Hyperopt spaces to optimize (default: buy sell)",
    )

    # Data
    p.add_argument("--timerange", default=None, help="Override timerange (e.g. 20230101-20240101)")
    p.add_argument("--datadir", default="/data/freqtrade_data", help="OHLCV data directory")
    p.add_argument("--user-data-dir", default="/root/workspace/freqtrade/user_data", help="freqtrade user_data dir")
    p.add_argument("--pairs-file", default=None, help="CSV/TXT with pair list")

    # Output
    p.add_argument("--output-dir", default=None, help="Directory for WFO reports")
    p.add_argument("--log-file", default=None, help="Log file path")

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
    return {
        "config": [ns.config],
        "strategy": ns.strategy,
        "strategy_list": [],
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


def main() -> None:
    ns = parse_args()
    setup_file_logger(ns.log_file)

    logger.info("=" * 60)
    logger.info("Walk-Forward Optimization")
    logger.info("  IS: %d days | OOS: %d days | Step: %s days",
                ns.is_days, ns.oos_days, ns.step_days or ns.oos_days)
    logger.info("  Epochs: %d | Loss: %s | Spaces: %s",
                ns.n_epochs, ns.loss_function, ns.spaces)
    logger.info("=" * 60)

    args = build_args(ns)
    config = setup_optimize_configuration(args, RunMode.BACKTEST)

    if ns.pairs_file:
        pairs = load_pairs(ns.pairs_file)
        if not pairs:
            raise RuntimeError("Pairs file is empty")
        config["pairs"] = pairs
        config["exchange"]["pair_whitelist"] = pairs
        config["pairlists"] = [{"method": "StaticPairList"}]
        logger.info("Loaded %d pairs from file", len(pairs))

    engine = WFOEngine(
        config,
        is_days=ns.is_days,
        oos_days=ns.oos_days,
        step_days=ns.step_days,
        n_epochs=ns.n_epochs,
        loss_function=ns.loss_function,
        spaces=ns.spaces,
        output_dir=ns.output_dir,
    )

    report = engine.run(timerange_str=ns.timerange)

    # Print summary JSON
    print("\n" + json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nVerdict: {report['overfit_analysis']['verdict']}")
    print(f"Report: {report['metadata']['report_path']}")


if __name__ == "__main__":
    main()
