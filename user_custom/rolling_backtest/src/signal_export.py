"""Signal export and deviation detection for rolling backtest.

Exports entry/exit signals per window and detects deviations (inconsistencies)
between adjacent windows — e.g., signals that appear in one window but vanish
in the next for the same candle timestamps.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pandas import DataFrame

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Compact signal representation for a single candle."""
    pair: str
    timestamp: str
    enter_long: bool
    exit_long: bool
    enter_short: bool
    exit_short: bool
    enter_tag: str
    exit_tag: str


@dataclass
class SignalDeviation:
    """Deviation between two adjacent windows for the same candle."""
    pair: str
    timestamp: str
    field: str
    window_a: int
    value_a: str
    window_b: int
    value_b: str


def extract_signals_from_dataframe(
    pair: str,
    df: DataFrame,
) -> list[SignalRecord]:
    """Extract entry/exit signal records from a processed dataframe."""
    if df.empty:
        return []

    records = []
    signal_cols = {
        "enter_long": False,
        "exit_long": False,
        "enter_short": False,
        "exit_short": False,
        "enter_tag": "",
        "exit_tag": "",
    }

    for _, row in df.iterrows():
        enter_long = bool(row.get("enter_long", signal_cols["enter_long"]))
        exit_long = bool(row.get("exit_long", signal_cols["exit_long"]))
        enter_short = bool(row.get("enter_short", signal_cols["enter_short"]))
        exit_short = bool(row.get("exit_short", signal_cols["exit_short"]))

        # Only record rows where at least one signal is active
        if not (enter_long or exit_long or enter_short or exit_short):
            continue

        ts = row.get("date", row.name)
        records.append(
            SignalRecord(
                pair=pair,
                timestamp=str(ts),
                enter_long=enter_long,
                exit_long=exit_long,
                enter_short=enter_short,
                exit_short=exit_short,
                enter_tag=str(row.get("enter_tag", "")),
                exit_tag=str(row.get("exit_tag", "")),
            )
        )
    return records


def detect_deviations(
    signals_a: list[SignalRecord],
    window_a_idx: int,
    signals_b: list[SignalRecord],
    window_b_idx: int,
) -> list[SignalDeviation]:
    """Compare signals from two windows on overlapping timestamps.

    Only checks candle timestamps that appear in both windows.
    Returns a list of deviations where signals differ for the same pair+timestamp.
    """
    # Build lookup: (pair, timestamp) -> SignalRecord
    index_a: dict[tuple[str, str], SignalRecord] = {
        (s.pair, s.timestamp): s for s in signals_a
    }
    index_b: dict[tuple[str, str], SignalRecord] = {
        (s.pair, s.timestamp): s for s in signals_b
    }

    overlapping_keys = set(index_a.keys()) & set(index_b.keys())
    deviations: list[SignalDeviation] = []
    compare_fields = ["enter_long", "exit_long", "enter_short", "exit_short", "enter_tag", "exit_tag"]

    for key in sorted(overlapping_keys):
        sa = index_a[key]
        sb = index_b[key]
        for field in compare_fields:
            va = getattr(sa, field)
            vb = getattr(sb, field)
            if va != vb:
                deviations.append(
                    SignalDeviation(
                        pair=key[0],
                        timestamp=key[1],
                        field=field,
                        window_a=window_a_idx,
                        value_a=str(va),
                        window_b=window_b_idx,
                        value_b=str(vb),
                    )
                )

    return deviations


def compute_signal_digest(signals: list[SignalRecord]) -> str:
    """Compute a deterministic hash digest of a signal list for quick comparison."""
    if not signals:
        return "empty"
    raw = json.dumps(
        [asdict(s) for s in sorted(signals, key=lambda x: (x.pair, x.timestamp))],
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SignalExporter:
    """Collects and exports signals across rolling windows.

    Usage:
        exporter = SignalExporter(output_dir)
        # After each window's indicator computation:
        exporter.collect_window(window_idx, preprocessed_data)
        # After all windows:
        exporter.export()
        deviations = exporter.detect_all_deviations()
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._window_signals: dict[int, list[SignalRecord]] = {}

    def collect_window(
        self,
        window_idx: int,
        preprocessed: dict[str, DataFrame],
    ) -> list[SignalRecord]:
        """Extract signals from all pairs in a preprocessed window."""
        all_signals: list[SignalRecord] = []
        for pair, df in preprocessed.items():
            signals = extract_signals_from_dataframe(pair, df)
            all_signals.extend(signals)

        self._window_signals[window_idx] = all_signals
        logger.debug(
            "[window %d] collected %d signals from %d pairs",
            window_idx,
            len(all_signals),
            len(preprocessed),
        )
        return all_signals

    def export(self) -> Path:
        """Export all collected signals to a JSON file."""
        output_file = self.output_dir / "rolling_signals.json"
        data = {}
        for widx in sorted(self._window_signals.keys()):
            sigs = self._window_signals[widx]
            data[f"window_{widx}"] = {
                "signal_count": len(sigs),
                "digest": compute_signal_digest(sigs),
                "signals": [asdict(s) for s in sigs],
            }

        output_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Signals exported to: %s", output_file)
        return output_file

    def detect_all_deviations(self) -> list[SignalDeviation]:
        """Detect deviations between all consecutive window pairs."""
        all_deviations: list[SignalDeviation] = []
        sorted_indices = sorted(self._window_signals.keys())

        for i in range(len(sorted_indices) - 1):
            idx_a = sorted_indices[i]
            idx_b = sorted_indices[i + 1]
            devs = detect_deviations(
                self._window_signals[idx_a],
                idx_a,
                self._window_signals[idx_b],
                idx_b,
            )
            all_deviations.extend(devs)

        return all_deviations

    def export_deviations(self) -> Path | None:
        """Detect and export deviations to a JSON file."""
        deviations = self.detect_all_deviations()
        if not deviations:
            logger.info("No signal deviations detected across windows.")
            return None

        output_file = self.output_dir / "signal_deviations.json"
        data = {
            "total_deviations": len(deviations),
            "deviations": [asdict(d) for d in deviations],
        }
        output_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.warning(
            "Detected %d signal deviations across windows → %s",
            len(deviations),
            output_file,
        )
        return output_file
