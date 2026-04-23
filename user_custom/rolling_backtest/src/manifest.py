from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SymbolMeta:
    pair: str
    symbol: str
    onboard_date: datetime
    planned_start: datetime
    quote: str


def symbol_to_pair(symbol: str, quote: str) -> str:
    if not symbol.endswith(quote):
        raise ValueError(f"Symbol {symbol} does not end with quote {quote}")
    base = symbol[: -len(quote)]
    return f"{base}/{quote}:{quote}"


def compute_planned_start(onboard_ms: int, years: int) -> datetime:
    now = datetime.now(tz=UTC)
    five_years_ago = now - timedelta(days=365 * years)
    onboard = datetime.fromtimestamp(onboard_ms / 1000, tz=UTC)
    return max(onboard, five_years_ago)


def build_symbol_meta(symbols: Iterable[dict], quote: str, years: int) -> list[SymbolMeta]:
    out: list[SymbolMeta] = []
    for s in symbols:
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != quote:
            continue
        symbol = s["symbol"]
        onboard_ms = int(s.get("onboardDate") or 0)
        if onboard_ms <= 0:
            continue
        out.append(
            SymbolMeta(
                pair=symbol_to_pair(symbol, quote),
                symbol=symbol,
                onboard_date=datetime.fromtimestamp(onboard_ms / 1000, tz=UTC),
                planned_start=compute_planned_start(onboard_ms, years),
                quote=quote,
            )
        )
    out.sort(key=lambda x: x.pair)
    return out


def write_manifest(rows: list[SymbolMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "symbol", "quote", "onboard_date_utc", "planned_start_utc"])
        for r in rows:
            w.writerow(
                [
                    r.pair,
                    r.symbol,
                    r.quote,
                    r.onboard_date.isoformat(),
                    r.planned_start.isoformat(),
                ]
            )


def read_manifest(path: Path) -> list[SymbolMeta]:
    rows: list[SymbolMeta] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rows.append(
                SymbolMeta(
                    pair=row["pair"],
                    symbol=row["symbol"],
                    quote=row["quote"],
                    onboard_date=datetime.fromisoformat(row["onboard_date_utc"]),
                    planned_start=datetime.fromisoformat(row["planned_start_utc"]),
                )
            )
    rows.sort(key=lambda x: x.pair)
    return rows
