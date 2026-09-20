"""
Tests for the day-window bias correction hook.

The correction table is empty by design: no day-window offset is applied to
any source or variable, and the archive carries provider values as issued.
An earlier version subtracted 0.303 degC from GSOD 'tas' on the strength of a
GHCN-D/GSOD *tasmax* median -- a statistic for a different variable and a
source pair that supplies no tas at all. It was withdrawn.

These tests pin the hook shut. They exist so that re-introducing a correction
cannot happen quietly: it has to come with a deliberate change to a test that
says, in as many words, that an offset must be measured on the same source
pair and variable it will be applied to.
"""
from __future__ import annotations

import pandas as pd
import pytest

from observations.standardize import (
    GSOD_DAY_WINDOW_BIAS_CORRECTION,
    apply_day_window_correction,
)


def _series(values):
    return pd.Series(values, index=pd.date_range("1990-01-01", periods=len(values), freq="D"))


def test_no_correction_table_entries_are_configured():
    """
    The archive applies no day-window correction at all.

    Adding an entry here must be accompanied by a measurement of that exact
    (source pair, variable) -- not a figure borrowed from another variable,
    and not one from a source pair the variable never falls back to.
    """
    assert GSOD_DAY_WINDOW_BIAS_CORRECTION == {}


def test_gsod_values_pass_through_unchanged():
    """GSOD 'tas' in particular: this is the withdrawn correction's target."""
    raw = _series([10.0, 20.303, 0.0])
    result = apply_day_window_correction(raw, "gsod", "tas")
    pd.testing.assert_series_equal(result, raw)


def test_every_source_and_variable_passes_through_unchanged():
    raw = _series([10.0, 20.0])
    for source in ("gsod", "ecad", "ghcnd", "c3s_land"):
        for variable in ("tas", "tasmax", "tasmin", "pr", "sfcWind", "gust", "hurs"):
            result = apply_day_window_correction(raw, source, variable)
            pd.testing.assert_series_equal(result, raw)


def test_a_configured_offset_would_still_be_subtracted(monkeypatch):
    """
    The mechanism itself is intact and correct -- it is the empty table, not a
    broken function, that makes this a no-op. A future measured offset must
    subtract, and must touch only the source and variable it was measured on.
    """
    monkeypatch.setitem(GSOD_DAY_WINDOW_BIAS_CORRECTION, "tas", 0.5)
    raw = _series([10.0, 20.0])

    corrected = apply_day_window_correction(raw, "gsod", "tas")
    assert list(corrected) == pytest.approx([9.5, 19.5])

    # and nothing else is affected by that entry
    pd.testing.assert_series_equal(apply_day_window_correction(raw, "ecad", "tas"), raw)
    pd.testing.assert_series_equal(apply_day_window_correction(raw, "gsod", "tasmax"), raw)


def test_correction_preserves_the_series_index():
    raw = _series([10.0, 20.0])
    result = apply_day_window_correction(raw, "gsod", "tas")
    pd.testing.assert_index_equal(result.index, raw.index)
