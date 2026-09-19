"""
Tests for the GSOD day-window bias correction (RESEARCH_PLAN_AND_DESIGN.md
"Open items carried into Step 2": "Day-window normalisation is diagnosed but
not applied. The +0.303 degC GSOD offset is detected and blocked, not
corrected.").
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


def test_gsod_tas_is_corrected_by_the_measured_offset():
    raw = _series([10.0, 20.303, 0.0])
    corrected = apply_day_window_correction(raw, "gsod", "tas")
    assert list(corrected) == pytest.approx([10.0 - 0.303, 20.0, -0.303])


def test_non_gsod_sources_are_never_corrected():
    raw = _series([10.0, 20.0])
    for source in ("ecad", "ghcnd", "c3s_land"):
        result = apply_day_window_correction(raw, source, "tas")
        pd.testing.assert_series_equal(result, raw)


def test_gsod_variables_without_a_measured_offset_are_unchanged():
    """
    tasmax/tasmin have no documented correction -- a check against measured
    fallback-candidate bias gave an inconsistent, near-zero result, so
    applying tas's 0.303 there would be guessing, not correcting.
    """
    raw = _series([10.0, 20.0])
    for variable in ("tasmax", "tasmin", "pr", "sfcWind", "gust", "hurs"):
        result = apply_day_window_correction(raw, "gsod", variable)
        pd.testing.assert_series_equal(result, raw)


def test_correction_table_only_documents_measured_offsets():
    assert GSOD_DAY_WINDOW_BIAS_CORRECTION == {"tas": 0.303}


def test_correction_preserves_the_series_index():
    raw = _series([10.0, 20.0])
    corrected = apply_day_window_correction(raw, "gsod", "tas")
    pd.testing.assert_index_equal(corrected.index, raw.index)
