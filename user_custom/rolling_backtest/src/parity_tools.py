from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TRADE_COLUMNS = [
    "pair",
    "open_date",
    "close_date",
    "open_rate",
    "close_rate",
    "profit_ratio",
    "is_short",
    "exit_reason",
]


def _canonical_trade_rows(trades: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    cols = columns or DEFAULT_TRADE_COLUMNS
    available_cols = [c for c in cols if c in trades.columns]

    if not available_cols:
        return []

    normalized = trades[available_cols].copy()
    for c in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[c]):
            normalized[c] = pd.to_datetime(normalized[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for row in normalized.to_dict(orient="records"):
        canon: dict[str, Any] = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                canon[k] = None
            elif isinstance(v, bool):
                canon[k] = v
            elif isinstance(v, (int, float)):
                canon[k] = round(float(v), 10)
            else:
                try:
                    ts = pd.to_datetime(v, utc=True)
                    if pd.isna(ts):
                        canon[k] = str(v)
                    else:
                        canon[k] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    canon[k] = str(v)
        rows.append(canon)

    def _sort_key(row: dict[str, Any]) -> tuple:
        return tuple("" if row.get(c) is None else str(row.get(c)) for c in available_cols)

    return sorted(rows, key=_sort_key)


def compute_trade_digest(trades: pd.DataFrame, columns: list[str] | None = None) -> str:
    rows = _canonical_trade_rows(trades, columns=columns)
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_backtest_json(path: str | Path) -> dict[str, Any] | list[Any]:
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            candidates = [
                name
                for name in zf.namelist()
                if name.endswith(".json") and not name.endswith("_config.json") and not name.endswith(".meta.json")
            ]
            if not candidates:
                raise ValueError(f"No result json found in zip: {path}")
            with zf.open(candidates[0]) as fp:
                return json.loads(fp.read().decode("utf-8"))

    return json.loads(p.read_text(encoding="utf-8"))


def load_trades_from_backtest_json(path: str | Path, strategy: str | None = None) -> pd.DataFrame:
    data = load_backtest_json(path)

    if isinstance(data, list):
        return pd.DataFrame(data)

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported json format: {type(data)!r}")

    if "trades" in data and isinstance(data["trades"], list):
        return pd.DataFrame(data["trades"])

    strategy_block = data.get("strategy")
    if isinstance(strategy_block, dict):
        if strategy:
            entry = strategy_block.get(strategy)
            if isinstance(entry, dict) and isinstance(entry.get("trades"), list):
                return pd.DataFrame(entry["trades"])
            raise ValueError(f"Strategy {strategy!r} trades not found in {path}")

        for payload in strategy_block.values():
            if isinstance(payload, dict) and isinstance(payload.get("trades"), list):
                return pd.DataFrame(payload["trades"])

    raise ValueError(f"No trades found in {path}")
