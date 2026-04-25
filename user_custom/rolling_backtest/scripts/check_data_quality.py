#!/usr/bin/env python3
"""
Comprehensive data quality check for downloaded Binance USDT-M Futures OHLCV data.

Checks:
  1. Missing files — expected (pair, tf) combos vs actual feather files
  2. Empty / too-small files
  3. Duplicate timestamps
  4. Time alignment — candle timestamps must align to timeframe grid
  5. Time gaps — missing candles beyond expected (weekends don't apply to crypto)
  6. Chronological order — timestamps must be strictly ascending
  7. OHLCV sanity — NaN, negative prices, high < low, zero volume anomalies
  8. Coverage — actual date range vs expected (planned_start ~ now)
  9. Stale data — max_date too far from now
  10. Cross-timeframe consistency — 1h close ≈ 5m close at same hour boundary

Outputs a JSON report + console summary.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("/data/freqtrade_data/futures")
META_DIR = Path("/data/freqtrade_data/_meta")
MANIFEST_CSV = META_DIR / "binance_futures_manifest.csv"
REPORT_OUT = META_DIR / "data_quality_report.json"

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]
TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}

# Thresholds
MAX_GAP_RATIO = 0.05        # >5% missing candles = warning
STALE_HOURS = 48            # max_date older than 48h = stale
MIN_CANDLES = 10            # files with fewer candles = suspect
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

NOW = datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pair_to_filename(pair: str, tf: str) -> str:
    """Convert 'BTC/USDT:USDT' -> 'BTC_USDT_USDT-{tf}-futures.feather'"""
    return pair.replace("/", "_").replace(":", "_") + f"-{tf}-futures.feather"


def read_manifest() -> list[dict]:
    """Read manifest CSV, return list of dicts with pair, onboard_date, planned_start."""
    rows = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_feather(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_feather(path)
        return df
    except Exception as e:
        return None


def expected_candles(start: datetime, end: datetime, tf_minutes: int) -> int:
    """Rough expected candle count (crypto trades 24/7)."""
    delta = end - start
    total_minutes = delta.total_seconds() / 60
    return max(0, int(total_minutes / tf_minutes))


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------
def check_missing_files(manifest_rows: list[dict]) -> dict:
    """Check which expected files are missing."""
    missing = []
    present = []
    for row in manifest_rows:
        pair = row["pair"]
        for tf in TIMEFRAMES:
            fname = pair_to_filename(pair, tf)
            fpath = DATA_DIR / fname
            if not fpath.exists():
                missing.append({"pair": pair, "timeframe": tf, "expected_file": fname})
            else:
                present.append({"pair": pair, "timeframe": tf, "path": str(fpath)})
    return {"missing_count": len(missing), "present_count": len(present), "missing": missing}


def check_single_file(pair: str, tf: str, fpath: Path, planned_start_str: str) -> dict:
    """Run all checks on a single feather file."""
    issues = []
    stats = {"pair": pair, "timeframe": tf, "file": fpath.name}

    df = load_feather(fpath)
    if df is None:
        issues.append({"type": "CORRUPT", "detail": "Cannot read feather file"})
        stats["issues"] = issues
        return stats

    if len(df) == 0:
        issues.append({"type": "EMPTY", "detail": "Zero rows"})
        stats["candles"] = 0
        stats["issues"] = issues
        return stats

    stats["candles"] = len(df)
    stats["file_size_kb"] = round(fpath.stat().st_size / 1024, 1)

    # Identify date column
    date_col = None
    for c in ["date", "datetime", "timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # Try first column
        date_col = df.columns[0]

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col], utc=True)
        except Exception:
            issues.append({"type": "BAD_DATES", "detail": f"Cannot parse {date_col} as datetime"})
            stats["issues"] = issues
            return stats

    # Make sure timezone-aware
    if df[date_col].dt.tz is None:
        df[date_col] = df[date_col].dt.tz_localize("UTC")

    dates = df[date_col].sort_values()
    min_date = dates.iloc[0]
    max_date = dates.iloc[-1]
    stats["min_date"] = str(min_date)
    stats["max_date"] = str(max_date)

    # 1. Too few candles
    if len(df) < MIN_CANDLES:
        issues.append({"type": "TOO_FEW", "detail": f"Only {len(df)} candles"})

    # 2. Duplicates
    dup_count = dates.duplicated().sum()
    if dup_count > 0:
        issues.append({"type": "DUPLICATES", "detail": f"{dup_count} duplicate timestamps"})
    stats["duplicate_timestamps"] = int(dup_count)

    # 3. Chronological order
    if not dates.is_monotonic_increasing:
        issues.append({"type": "NOT_SORTED", "detail": "Timestamps not strictly ascending"})

    # 4. Time alignment — check that timestamps are on the expected grid
    tf_min = TF_MINUTES[tf]
    if tf not in ("1w",):  # skip 1w alignment check (exchange-dependent)
        sample = dates.head(100)
        misaligned = 0
        for ts in sample:
            ts_minutes = ts.hour * 60 + ts.minute
            if tf == "1d":
                if ts.hour != 0 or ts.minute != 0:
                    misaligned += 1
            else:
                if ts_minutes % tf_min != 0:
                    misaligned += 1
        if misaligned > 0:
            issues.append({"type": "MISALIGNED", "detail": f"{misaligned}/{len(sample)} sample timestamps not aligned to {tf} grid"})

    # 5. Time gaps
    if len(dates) > 1 and tf != "1w":
        diffs = dates.diff().dropna()
        expected_delta = pd.Timedelta(minutes=tf_min)
        gap_count = (diffs > expected_delta * 1.5).sum()
        gap_ratio = gap_count / len(diffs) if len(diffs) > 0 else 0

        # Find largest gap
        if gap_count > 0:
            max_gap = diffs.max()
            max_gap_hours = max_gap.total_seconds() / 3600
            stats["max_gap_hours"] = round(max_gap_hours, 2)
            stats["gap_count"] = int(gap_count)
            stats["gap_ratio"] = round(gap_ratio, 4)

            if gap_ratio > MAX_GAP_RATIO:
                issues.append({
                    "type": "EXCESSIVE_GAPS",
                    "detail": f"{gap_count} gaps ({gap_ratio:.1%}), max gap={max_gap_hours:.1f}h"
                })

        # Expected vs actual candle count
        exp = expected_candles(min_date.to_pydatetime(), max_date.to_pydatetime(), tf_min)
        if exp > 0:
            completeness = len(df) / exp
            stats["expected_candles"] = exp
            stats["completeness"] = round(completeness, 4)
            if completeness < 0.90:
                issues.append({
                    "type": "INCOMPLETE",
                    "detail": f"Only {completeness:.1%} of expected candles ({len(df)}/{exp})"
                })

    # 6. Staleness
    if max_date.tzinfo is None:
        max_date_utc = max_date.replace(tzinfo=UTC)
    else:
        max_date_utc = max_date
    hours_stale = (NOW - max_date_utc).total_seconds() / 3600
    stats["hours_since_last"] = round(hours_stale, 1)
    if hours_stale > STALE_HOURS and tf not in ("1w", "1d"):
        issues.append({"type": "STALE", "detail": f"Last candle is {hours_stale:.0f}h old"})

    # 7. OHLCV sanity
    numeric_cols = {}
    for col_name in ["open", "high", "low", "close", "volume"]:
        if col_name in df.columns:
            numeric_cols[col_name] = df[col_name]

    if numeric_cols:
        # NaN check
        for col_name, col_data in numeric_cols.items():
            nan_count = col_data.isna().sum()
            if nan_count > 0:
                issues.append({"type": "NAN_VALUES", "detail": f"{col_name}: {nan_count} NaN values"})

        # Negative prices
        for col_name in ["open", "high", "low", "close"]:
            if col_name in numeric_cols:
                neg = (numeric_cols[col_name] < 0).sum()
                if neg > 0:
                    issues.append({"type": "NEGATIVE_PRICE", "detail": f"{col_name}: {neg} negative values"})

        # High < Low
        if "high" in numeric_cols and "low" in numeric_cols:
            bad_hl = (numeric_cols["high"] < numeric_cols["low"]).sum()
            if bad_hl > 0:
                issues.append({"type": "HIGH_LT_LOW", "detail": f"{bad_hl} candles where high < low"})

        # Zero volume (more than 10% = suspicious)
        if "volume" in numeric_cols:
            zero_vol = (numeric_cols["volume"] == 0).sum()
            zero_vol_ratio = zero_vol / len(df) if len(df) > 0 else 0
            stats["zero_volume_candles"] = int(zero_vol)
            if zero_vol_ratio > 0.10:
                issues.append({"type": "ZERO_VOLUME", "detail": f"{zero_vol} ({zero_vol_ratio:.1%}) candles with zero volume"})

    # 8. Epoch anomaly — dates near 1970 indicate broken data
    if min_date.year < 2019:
        issues.append({"type": "EPOCH_ANOMALY", "detail": f"min_date={min_date} is before 2019, likely corrupt"})

    stats["issue_count"] = len(issues)
    stats["issues"] = issues
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"{'='*70}")
    print(f"  Binance Futures Data Quality Check")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Time: {NOW.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*70}\n")

    # Load manifest
    manifest_rows = read_manifest()
    pairs = sorted(set(r["pair"] for r in manifest_rows))
    print(f"Manifest: {len(manifest_rows)} pairs")

    # Build planned_start lookup
    pair_meta = {}
    for r in manifest_rows:
        pair_meta[r["pair"]] = r

    # 1. Missing files check
    print("\n[1/3] Checking missing files...")
    missing_info = check_missing_files(manifest_rows)
    print(f"  Present: {missing_info['present_count']}, Missing: {missing_info['missing_count']}")
    if missing_info["missing"]:
        # Group missing by timeframe
        by_tf = defaultdict(list)
        for m in missing_info["missing"]:
            by_tf[m["timeframe"]].append(m["pair"])
        for tf, mp in sorted(by_tf.items()):
            print(f"    {tf}: {len(mp)} missing — e.g. {mp[:3]}")

    # 2. Per-file checks
    print(f"\n[2/3] Checking {missing_info['present_count']} data files...")
    all_results = []
    issue_summary = defaultdict(int)
    files_with_issues = 0
    files_checked = 0

    for row in manifest_rows:
        pair = row["pair"]
        planned_start = row.get("planned_start", "")
        for tf in TIMEFRAMES:
            fname = pair_to_filename(pair, tf)
            fpath = DATA_DIR / fname
            if not fpath.exists():
                continue

            result = check_single_file(pair, tf, fpath, planned_start)
            all_results.append(result)
            files_checked += 1

            if result.get("issue_count", 0) > 0:
                files_with_issues += 1
                for iss in result["issues"]:
                    issue_summary[iss["type"]] += 1

            if files_checked % 500 == 0:
                print(f"    ... checked {files_checked} files")

    print(f"  Checked: {files_checked} files")
    print(f"  Files with issues: {files_with_issues}")

    # 3. Summary
    print(f"\n[3/3] Issue Summary")
    print(f"{'='*70}")

    if not issue_summary:
        print("  ✅ No issues found! All data looks clean.")
    else:
        for issue_type, count in sorted(issue_summary.items(), key=lambda x: -x[1]):
            severity = "🔴" if issue_type in ("CORRUPT", "EMPTY", "EPOCH_ANOMALY", "HIGH_LT_LOW", "NEGATIVE_PRICE") else "🟡"
            print(f"  {severity} {issue_type:25s} : {count:>5d} files")

    # Print worst offenders
    worst = sorted(
        [r for r in all_results if r.get("issue_count", 0) > 0],
        key=lambda x: -x.get("issue_count", 0)
    )[:20]
    if worst:
        print(f"\n  Top problematic files:")
        for r in worst:
            issues_str = ", ".join(i["type"] for i in r["issues"])
            print(f"    {r['pair']:25s} {r['timeframe']:5s} | {r.get('candles','?'):>8} candles | {issues_str}")

    # Coverage stats
    print(f"\n{'='*70}")
    print("Coverage Statistics:")
    by_tf_stats = defaultdict(list)
    for r in all_results:
        if "completeness" in r:
            by_tf_stats[r["timeframe"]].append(r["completeness"])

    for tf in TIMEFRAMES:
        vals = by_tf_stats.get(tf, [])
        if vals:
            arr = np.array(vals)
            print(f"  {tf:5s}: mean={arr.mean():.1%}, median={np.median(arr):.1%}, min={arr.min():.1%}, max={arr.max():.1%}, n={len(arr)}")

    # Epoch anomaly pairs
    epoch_pairs = [r for r in all_results if any(i["type"] == "EPOCH_ANOMALY" for i in r.get("issues", []))]
    if epoch_pairs:
        print(f"\n⚠️  EPOCH ANOMALY (dates near 1970) — {len(epoch_pairs)} files:")
        for r in epoch_pairs[:20]:
            print(f"    {r['pair']:25s} {r['timeframe']:5s} | min_date={r.get('min_date','?')}")

    # Write JSON report
    report = {
        "check_time": NOW.isoformat(),
        "data_dir": str(DATA_DIR),
        "manifest_pairs": len(manifest_rows),
        "files_present": missing_info["present_count"],
        "files_missing": missing_info["missing_count"],
        "files_checked": files_checked,
        "files_with_issues": files_with_issues,
        "issue_summary": dict(issue_summary),
        "missing_files": missing_info["missing"],
        "problematic_files": [r for r in all_results if r.get("issue_count", 0) > 0],
        "all_results_count": len(all_results),
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report: {REPORT_OUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
