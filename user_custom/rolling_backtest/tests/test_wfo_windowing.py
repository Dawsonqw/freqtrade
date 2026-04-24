"""Tests for WFO windowing (IS/OOS fold generation)."""
import pytest
from datetime import datetime, UTC
from freqtrade.configuration import TimeRange

from user_custom.rolling_backtest.src.wfo_windowing import WFOFold, build_wfo_folds


def test_build_wfo_folds_basic():
    """180d IS + 30d OOS, step=30d, over 1 year should give ~7 folds."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr,
        timeframe="5m",
        is_days=180,
        oos_days=30,
        step_days=30,
    )
    assert len(folds) >= 5
    # Each fold has IS + OOS
    for f in folds:
        assert f.is_start < f.is_end
        assert f.oos_start == f.is_end  # OOS immediately follows IS
        assert f.oos_end > f.oos_start
        assert (f.oos_end - f.oos_start).days <= 31  # ~30 days


def test_build_wfo_folds_no_overlap():
    """With step=oos_days, OOS windows should tile without gaps."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=90, oos_days=30, step_days=30,
    )
    # OOS end of fold i == OOS start of fold i+1
    for i in range(len(folds) - 1):
        assert folds[i].oos_end == folds[i + 1].oos_start


def test_build_wfo_folds_too_short():
    """Timerange shorter than IS+OOS should raise."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2024, 3, 1, tzinfo=UTC).timestamp()))
    with pytest.raises(ValueError, match="too short"):
        build_wfo_folds(tr, timeframe="5m", is_days=180, oos_days=30)


def test_build_wfo_folds_default_step():
    """Default step_days should equal oos_days."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=90, oos_days=30,
    )
    # Should be same as explicit step_days=30
    folds_explicit = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=90, oos_days=30, step_days=30,
    )
    assert len(folds) == len(folds_explicit)
    for a, b in zip(folds, folds_explicit):
        assert a.is_start == b.is_start
        assert a.oos_end == b.oos_end


def test_wfo_fold_timerange_properties():
    """WFOFold.is_timerange and oos_timerange should produce valid TimeRange objects."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=180, oos_days=30,
    )
    f = folds[0]
    is_tr = f.is_timerange
    oos_tr = f.oos_timerange
    assert is_tr.startts == int(f.is_start.timestamp())
    assert is_tr.stopts == int(f.is_end.timestamp())
    assert oos_tr.startts == int(f.oos_start.timestamp())
    assert oos_tr.stopts == int(f.oos_end.timestamp())


def test_wfo_fold_days_properties():
    """WFOFold.is_days and oos_days should return approximate day counts."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=180, oos_days=30,
    )
    f = folds[0]
    assert abs(f.is_days - 180) < 1
    assert abs(f.oos_days - 30) < 1


def test_build_wfo_folds_small_step_overlap():
    """With step < oos_days, IS windows should overlap."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=90, oos_days=30, step_days=15,
    )
    # More folds than non-overlapping
    folds_no_overlap = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=90, oos_days=30, step_days=30,
    )
    assert len(folds) > len(folds_no_overlap)


def test_build_wfo_folds_indices_sequential():
    """Fold indices should be 0, 1, 2, ..."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=90, oos_days=30,
    )
    for i, f in enumerate(folds):
        assert f.index == i
