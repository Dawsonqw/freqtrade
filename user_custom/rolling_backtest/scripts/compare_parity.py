#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_custom.rolling_backtest.src.parity_tools import (
    compute_trade_digest,
    load_backtest_json,
    load_trades_from_backtest_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare rolling backtest result with native freqtrade result")
    p.add_argument("--rolling-json", required=True, help="rolling result json (from run_rolling_backtest.py)")
    p.add_argument("--native-json", required=True, help="native freqtrade backtest result json")
    p.add_argument("--strategy", default=None, help="strategy name in native result, when needed")
    p.add_argument("--balance-tol", type=float, default=1e-8, help="absolute tolerance of final balance")
    p.add_argument("--output-json", default=None, help="optional report output path")
    return p.parse_args()


def main() -> None:
    ns = parse_args()

    rolling_data = json.loads(Path(ns.rolling_json).read_text(encoding="utf-8"))
    rolling_summary = rolling_data.get("summary", {})

    native_trades = load_trades_from_backtest_json(ns.native_json, strategy=ns.strategy)
    native_digest = compute_trade_digest(native_trades)

    rolling_digest = rolling_summary.get("trade_digest")
    if not rolling_digest:
        raise RuntimeError("rolling result missing summary.trade_digest, please rerun with latest runner")

    rolling_balance = float(rolling_summary.get("final_balance", 0.0))

    native_balance = None
    native_data = load_backtest_json(ns.native_json)
    strategy_block = native_data.get("strategy") if isinstance(native_data, dict) else None
    if isinstance(strategy_block, dict):
        if ns.strategy and isinstance(strategy_block.get(ns.strategy), dict):
            native_balance = strategy_block[ns.strategy].get("final_balance")
        elif strategy_block:
            first_payload = next(iter(strategy_block.values()))
            if isinstance(first_payload, dict):
                native_balance = first_payload.get("final_balance")

    result = {
        "rolling_summary": {
            "total_trades": rolling_summary.get("total_trades"),
            "final_balance": rolling_balance,
            "trade_digest": rolling_digest,
        },
        "native_summary": {
            "total_trades": int(len(native_trades)),
            "final_balance": float(native_balance) if native_balance is not None else None,
            "trade_digest": native_digest,
        },
    }

    digest_match = rolling_digest == native_digest
    result["checks"] = {
        "trade_digest_match": digest_match,
    }

    if native_balance is not None:
        balance_diff = abs(float(native_balance) - rolling_balance)
        result["checks"]["final_balance_abs_diff"] = balance_diff
        result["checks"]["final_balance_within_tol"] = balance_diff <= ns.balance_tol

    result["all_passed"] = all(v is True for k, v in result["checks"].items() if isinstance(v, bool))

    if ns.output_json:
        out = Path(ns.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
