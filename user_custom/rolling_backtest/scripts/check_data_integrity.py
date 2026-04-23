#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freqtrade.data.history import get_datahandler
from freqtrade.enums import CandleType

from user_custom.rolling_backtest.src.manifest import read_manifest


@dataclass
class IntegrityRow:
    pair: str
    timeframe: str
    status: str
    expected_start_utc: str
    actual_start_utc: str
    actual_end_utc: str
    expected_min_candles: int
    actual_candles: int
    coverage_ratio: float


def tf_to_minutes(tf: str) -> int:
    unit = tf[-1]
    value = int(tf[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    if unit == "w":
        return value * 10080
    raise ValueError(f"Unsupported timeframe: {tf}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate downloaded futures data integrity")
    p.add_argument("--data-dir", default="/data/freqtrade_data")
    p.add_argument("--meta-dir", default="/data/freqtrade_data/_meta")
    p.add_argument("--manifest", default=None, help="Default: <meta-dir>/binance_futures_manifest.csv")
    p.add_argument("--timeframes", nargs="+", default=["5m", "15m", "1h", "4h", "1d", "1w"])
    p.add_argument("--min-coverage", type=float, default=0.95, help="Coverage ratio threshold")
    p.add_argument("--grace-candles", type=int, default=200, help="Allowed missing candles")
    p.add_argument("--output-csv", default=None, help="Default: <meta-dir>/integrity_report.csv")
    p.add_argument("--output-json", default=None, help="Default: <meta-dir>/integrity_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    meta_dir = Path(args.meta_dir)
    manifest_path = Path(args.manifest) if args.manifest else (meta_dir / "binance_futures_manifest.csv")
    out_csv = Path(args.output_csv) if args.output_csv else (meta_dir / "integrity_report.csv")
    out_json = Path(args.output_json) if args.output_json else (meta_dir / "integrity_summary.json")

    rows = read_manifest(manifest_path)
    dh = get_datahandler(data_dir, "feather")

    now = datetime.now(tz=UTC)
    report_rows: list[IntegrityRow] = []

    for row in rows:
        for tf in args.timeframes:
            expected_start = row.planned_start
            expected_frames = max(1, int((now - expected_start).total_seconds() // 60 // tf_to_minutes(tf)))
            expected_min = max(1, expected_frames - args.grace_candles)

            try:
                min_date, max_date, actual_len = dh.ohlcv_data_min_max(row.pair, tf, CandleType.FUTURES)
                ratio = actual_len / expected_frames if expected_frames > 0 else 0.0
                status = "ok" if (actual_len >= expected_min and ratio >= args.min_coverage) else "insufficient"
                report_rows.append(
                    IntegrityRow(
                        pair=row.pair,
                        timeframe=tf,
                        status=status,
                        expected_start_utc=expected_start.isoformat(),
                        actual_start_utc=min_date.isoformat(),
                        actual_end_utc=max_date.isoformat(),
                        expected_min_candles=expected_min,
                        actual_candles=int(actual_len),
                        coverage_ratio=round(ratio, 4),
                    )
                )
            except Exception:
                report_rows.append(
                    IntegrityRow(
                        pair=row.pair,
                        timeframe=tf,
                        status="missing",
                        expected_start_utc=expected_start.isoformat(),
                        actual_start_utc="",
                        actual_end_utc="",
                        expected_min_candles=expected_min,
                        actual_candles=0,
                        coverage_ratio=0.0,
                    )
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair",
                "timeframe",
                "status",
                "expected_start_utc",
                "actual_start_utc",
                "actual_end_utc",
                "expected_min_candles",
                "actual_candles",
                "coverage_ratio",
            ]
        )
        for rr in report_rows:
            w.writerow(
                [
                    rr.pair,
                    rr.timeframe,
                    rr.status,
                    rr.expected_start_utc,
                    rr.actual_start_utc,
                    rr.actual_end_utc,
                    rr.expected_min_candles,
                    rr.actual_candles,
                    rr.coverage_ratio,
                ]
            )

    total = len(report_rows)
    ok_count = sum(1 for x in report_rows if x.status == "ok")
    insufficient_count = sum(1 for x in report_rows if x.status == "insufficient")
    missing_count = sum(1 for x in report_rows if x.status == "missing")

    summary = {
        "manifest": str(manifest_path),
        "report_csv": str(out_csv),
        "total_checks": total,
        "ok_checks": ok_count,
        "insufficient_checks": insufficient_count,
        "missing_checks": missing_count,
        "pass_rate": round(ok_count / total, 4) if total else 0.0,
        "failed_examples": [asdict(x) for x in report_rows if x.status != "ok"][:50],
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
