"""Tests for dynamic pairlist refresh and signal export/deviation detection."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest
from datetime import datetime, UTC, timedelta

from user_custom.rolling_backtest.src.signal_export import (
    SignalRecord,
    SignalDeviation,
    SignalExporter,
    extract_signals_from_dataframe,
    detect_deviations,
    compute_signal_digest,
)
from user_custom.rolling_backtest.src.windowing import Window


def _make_timerange(start: datetime, end: datetime):
    """Create a mock TimeRange from start/end datetimes."""
    tr = MagicMock()
    tr.startts = int(start.timestamp())
    tr.stopts = int(end.timestamp())
    return tr


def _make_window(idx: int, start: datetime, end: datetime) -> Window:
    return Window(
        index=idx,
        start=start,
        end=end,
        timerange=_make_timerange(start, end),
    )


# ──────────────────────────────────────────────
# Signal Export Tests
# ──────────────────────────────────────────────

class TestExtractSignals:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")
        return df

    def test_empty_df(self):
        df = pd.DataFrame()
        assert extract_signals_from_dataframe("BTC/USDT:USDT", df) == []

    def test_no_signals(self):
        df = self._make_df([
            {"date": "2024-01-01", "close": 100, "enter_long": 0, "exit_long": 0,
             "enter_short": 0, "exit_short": 0},
        ])
        assert extract_signals_from_dataframe("BTC/USDT:USDT", df) == []

    def test_extracts_entry_signal(self):
        df = self._make_df([
            {"date": "2024-01-01", "close": 100, "enter_long": 1, "exit_long": 0,
             "enter_short": 0, "exit_short": 0, "enter_tag": "rsi_buy", "exit_tag": ""},
            {"date": "2024-01-02", "close": 101, "enter_long": 0, "exit_long": 0,
             "enter_short": 0, "exit_short": 0, "enter_tag": "", "exit_tag": ""},
        ])
        signals = extract_signals_from_dataframe("BTC/USDT:USDT", df)
        assert len(signals) == 1
        assert signals[0].enter_long is True
        assert signals[0].enter_tag == "rsi_buy"

    def test_extracts_multiple_signal_types(self):
        df = self._make_df([
            {"date": "2024-01-01", "enter_long": 1, "exit_long": 0,
             "enter_short": 0, "exit_short": 0, "enter_tag": "", "exit_tag": ""},
            {"date": "2024-01-02", "enter_long": 0, "exit_long": 0,
             "enter_short": 1, "exit_short": 0, "enter_tag": "short_signal", "exit_tag": ""},
            {"date": "2024-01-03", "enter_long": 0, "exit_long": 1,
             "enter_short": 0, "exit_short": 0, "enter_tag": "", "exit_tag": "take_profit"},
        ])
        signals = extract_signals_from_dataframe("ETH/USDT:USDT", df)
        assert len(signals) == 3


class TestDetectDeviations:
    def test_no_overlap(self):
        """Non-overlapping timestamps produce no deviations."""
        sigs_a = [SignalRecord("BTC/USDT:USDT", "2024-01-01", True, False, False, False, "", "")]
        sigs_b = [SignalRecord("BTC/USDT:USDT", "2024-01-15", True, False, False, False, "", "")]
        devs = detect_deviations(sigs_a, 0, sigs_b, 1)
        assert devs == []

    def test_identical_signals_no_deviation(self):
        sig = SignalRecord("BTC/USDT:USDT", "2024-01-05", True, False, False, False, "tag_a", "")
        devs = detect_deviations([sig], 0, [sig], 1)
        assert devs == []

    def test_detects_signal_change(self):
        """Same pair+timestamp but different signal values → deviation."""
        # Both have active signals at the same timestamp, but different types
        sig_a = SignalRecord("BTC/USDT:USDT", "2024-01-05", True, False, False, False, "", "")
        sig_b = SignalRecord("BTC/USDT:USDT", "2024-01-05", False, True, False, False, "", "")
        devs = detect_deviations([sig_a], 0, [sig_b], 1)
        assert len(devs) == 2  # enter_long changed, exit_long changed
        fields = {d.field for d in devs}
        assert "enter_long" in fields
        assert "exit_long" in fields

    def test_detects_tag_change(self):
        sig_a = SignalRecord("BTC/USDT:USDT", "2024-01-05", True, False, False, False, "rsi", "")
        sig_b = SignalRecord("BTC/USDT:USDT", "2024-01-05", True, False, False, False, "macd", "")
        devs = detect_deviations([sig_a], 0, [sig_b], 1)
        assert len(devs) == 1
        assert devs[0].field == "enter_tag"

    def test_multiple_deviations(self):
        sig_a = SignalRecord("BTC/USDT:USDT", "2024-01-05", True, False, False, False, "tag1", "")
        sig_b = SignalRecord("BTC/USDT:USDT", "2024-01-05", False, True, False, False, "tag2", "")
        devs = detect_deviations([sig_a], 0, [sig_b], 1)
        # enter_long, exit_long, enter_tag changed = 3 deviations
        assert len(devs) == 3


class TestSignalDigest:
    def test_deterministic(self):
        sigs = [SignalRecord("BTC/USDT:USDT", "2024-01-01", True, False, False, False, "", "")]
        assert compute_signal_digest(sigs) == compute_signal_digest(sigs)

    def test_empty_returns_empty(self):
        assert compute_signal_digest([]) == "empty"

    def test_order_independent(self):
        sig_a = SignalRecord("AAA/USDT:USDT", "2024-01-01", True, False, False, False, "", "")
        sig_b = SignalRecord("BTC/USDT:USDT", "2024-01-01", True, False, False, False, "", "")
        assert compute_signal_digest([sig_a, sig_b]) == compute_signal_digest([sig_b, sig_a])


class TestSignalExporter:
    def test_collect_and_export(self, tmp_path):
        exporter = SignalExporter(tmp_path / "signals")

        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "close": [100, 101],
            "enter_long": [1, 0],
            "exit_long": [0, 0],
            "enter_short": [0, 0],
            "exit_short": [0, 0],
            "enter_tag": ["buy", ""],
            "exit_tag": ["", ""],
        }).set_index("date")

        exporter.collect_window(0, {"BTC/USDT:USDT": df})
        exporter.collect_window(1, {"BTC/USDT:USDT": df})

        out = exporter.export()
        assert out.exists()

    def test_detect_deviations_across_windows(self, tmp_path):
        """Two consecutive windows with the same timestamp but different signals → deviation."""
        exporter = SignalExporter(tmp_path / "signals")

        # Window 0: enter_long=1 at 2024-01-05
        df_a = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-05"], utc=True),
            "enter_long": [1], "exit_long": [0],
            "enter_short": [0], "exit_short": [0],
            "enter_tag": ["tag_a"], "exit_tag": [""],
        }).set_index("date")

        # Window 1: enter_short=1 at same timestamp (different signal)
        df_b = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-05"], utc=True),
            "enter_long": [0], "exit_long": [0],
            "enter_short": [1], "exit_short": [0],
            "enter_tag": ["tag_b"], "exit_tag": [""],
        }).set_index("date")

        exporter.collect_window(0, {"BTC/USDT:USDT": df_a})
        exporter.collect_window(1, {"BTC/USDT:USDT": df_b})

        devs = exporter.detect_all_deviations()
        assert len(devs) >= 1
        fields = {d.field for d in devs}
        # enter_long and enter_short should both have changed
        assert "enter_long" in fields or "enter_short" in fields


# ──────────────────────────────────────────────
# Dynamic Pairlist Tests
# ──────────────────────────────────────────────

class TestDynamicPairlist:
    """Test that _refresh_pairlist_for_window correctly refreshes pairlist in dynamic mode."""

    def _make_runner(self, dynamic: bool, whitelist: list[str], available_pairs=None):
        """Create a RollingBacktestRunner with mocked Backtesting."""
        from user_custom.rolling_backtest.src.rolling_runner import RollingBacktestRunner

        bt = MagicMock()
        bt.config = {"enable_dynamic_pairlist": dynamic}
        bt.pairlists.whitelist = whitelist
        if available_pairs is not None:
            bt.available_pairs = available_pairs
        return RollingBacktestRunner(bt, window_days=14)

    def test_static_mode_returns_whitelist_unchanged(self):
        """When dynamic pairlist is disabled, whitelist should not change."""
        runner = self._make_runner(False, ["BTC/USDT:USDT", "ETH/USDT:USDT"])
        window = _make_window(0, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 15, tzinfo=UTC))

        result = runner._refresh_pairlist_for_window(window)
        assert result == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        runner.bt.pairlists.refresh_pairlist.assert_not_called()

    def test_dynamic_mode_calls_refresh(self):
        """When dynamic pairlist is enabled, refresh_pairlist should be called."""
        runner = self._make_runner(
            True,
            ["BTC/USDT:USDT"],
            available_pairs=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
        )
        window = _make_window(0, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 15, tzinfo=UTC))

        runner._refresh_pairlist_for_window(window)
        runner.bt.pairlists.refresh_pairlist.assert_called_once_with(
            pairs=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
        )

    def test_dynamic_mode_no_available_pairs(self):
        """When available_pairs is empty, refresh_pairlist is called with pairs=None."""
        runner = self._make_runner(True, ["BTC/USDT:USDT"], available_pairs=[])
        window = _make_window(0, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 15, tzinfo=UTC))

        runner._refresh_pairlist_for_window(window)
        runner.bt.pairlists.refresh_pairlist.assert_called_once_with(pairs=None)
