from __future__ import annotations

import json

import pandas as pd

from user_custom.rolling_backtest.src.parity_tools import compute_trade_digest, load_trades_from_backtest_json


def test_compute_trade_digest_is_order_insensitive() -> None:
    a = pd.DataFrame(
        [
            {"pair": "BTC/USDT:USDT", "open_date": "2026-01-01T00:00:00+00:00", "close_date": "2026-01-01T01:00:00+00:00", "profit_ratio": 0.01, "open_rate": 100.0, "close_rate": 101.0},
            {"pair": "ETH/USDT:USDT", "open_date": "2026-01-02T00:00:00+00:00", "close_date": "2026-01-02T02:00:00+00:00", "profit_ratio": -0.02, "open_rate": 200.0, "close_rate": 196.0},
        ]
    )
    b = a.iloc[::-1].reset_index(drop=True)

    assert compute_trade_digest(a) == compute_trade_digest(b)


def test_compute_trade_digest_changes_when_trade_changes() -> None:
    a = pd.DataFrame(
        [
            {"pair": "BTC/USDT:USDT", "open_date": "2026-01-01T00:00:00+00:00", "close_date": "2026-01-01T01:00:00+00:00", "profit_ratio": 0.01, "open_rate": 100.0, "close_rate": 101.0},
        ]
    )
    b = a.copy()
    b.loc[0, "close_rate"] = 102.0

    assert compute_trade_digest(a) != compute_trade_digest(b)


def test_load_trades_from_backtest_json_with_strategy_block(tmp_path) -> None:
    payload = {
        "strategy": {
            "RollingSmokeStrategy": {
                "final_balance": 123.0,
                "trades": [
                    {
                        "pair": "BTC/USDT:USDT",
                        "open_date": "2026-01-01T00:00:00+00:00",
                        "close_date": "2026-01-01T01:00:00+00:00",
                        "open_rate": 100.0,
                        "close_rate": 101.0,
                    }
                ],
            }
        }
    }
    fp = tmp_path / "native.json"
    fp.write_text(json.dumps(payload), encoding="utf-8")

    df = load_trades_from_backtest_json(fp, strategy="RollingSmokeStrategy")
    assert len(df) == 1
    assert df.iloc[0]["pair"] == "BTC/USDT:USDT"
