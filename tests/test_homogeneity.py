"""
Tests for breakpoint/inhomogeneity diagnosis (RESEARCH_PLAN_AND_DESIGN.md
"Open items carried into Step 2": "Breakpoint/inhomogeneity diagnosis is not
implemented.").

Detection runs on monthly anomalies, with a conservative SNHT floor and a
two-neighbor minimum -- see the module docstring for why each of those is
there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from observations.homogeneity import (
    MIN_DAYS_PER_MONTH,
    MIN_NEIGHBORS,
    MIN_NEIGHBOR_OVERLAP_MONTHS,
    MIN_SEGMENT_MONTHS,
    MIN_SNHT_STATISTIC,
    _restricted_max,
    benjamini_hochberg,
    build_reference_difference,
    detect_breakpoint,
    run_report,
    select_neighbors,
    snht_statistic,
    to_monthly_anomalies,
    variable_series_by_station,
)

_DEFAULT_DAYS = 3650          # 10 years -> 120 usable months


def _correlated_series(n=_DEFAULT_DAYS, seed=0, shift_after=None, shift_amount=0.0,
                        start="1990-01-01", weather_seed=777):
    """
    A realistic nearby-station series: a deterministic seasonal cycle, plus
    a REGIONAL weather anomaly shared by every station (same weather_seed),
    plus smaller station-specific noise.

    The shared weather term is what makes real neighbors correlate after
    deseasonalisation -- a warm January is warm at both stations. Sharing
    only the seasonal cycle would not do: to_monthly_anomalies subtracts
    the calendar-month climatology, which removes exactly that term and
    would leave two stations with nothing in common.
    """
    idx = pd.date_range(start, periods=n, freq="D")
    seasonal = 15.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    weather = np.random.default_rng(weather_seed).normal(0, 3.0, n)
    local = np.random.default_rng(seed).normal(0, 0.5, n)
    values = seasonal + weather + local
    if shift_after is not None:
        values = values.copy()
        values[shift_after:] += shift_amount
    return pd.Series(values, index=idx)


def _monthly(n_months, seed=0, start="1990-01-01"):
    """A monthly-frequency series, as build_reference_difference expects."""
    idx = pd.date_range(start, periods=n_months, freq="MS")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1.0, n_months), index=idx)


# ---------------------------------------------------------------------------
# to_monthly_anomalies
# ---------------------------------------------------------------------------

def test_to_monthly_anomalies_averages_a_month():
    idx = pd.date_range("1990-01-01", periods=31, freq="D")
    series = pd.Series(np.full(31, 10.0), index=idx)
    monthly = to_monthly_anomalies(series, "tas")
    assert len(monthly) == 1
    # single month -> climatology equals the value -> anomaly is zero
    assert monthly.iloc[0] == pytest.approx(0.0)


def test_to_monthly_anomalies_drops_months_below_the_day_threshold():
    # 25 days in January (kept), 5 days in February (dropped)
    idx = pd.date_range("1990-01-01", periods=25, freq="D").append(
        pd.date_range("1990-02-01", periods=5, freq="D")
    )
    series = pd.Series(np.arange(30, dtype=float), index=idx)
    monthly = to_monthly_anomalies(series, "tas")
    assert len(monthly) == 1
    assert monthly.index[0].month == 1


def test_to_monthly_anomalies_sums_precipitation():
    idx = pd.date_range("1990-01-01", periods=31, freq="D")
    series = pd.Series(np.full(31, 2.0), index=idx)
    mean_based = to_monthly_anomalies(series, "tas")
    # A single month deseasonalises to zero either way; check the raw
    # aggregation by giving two Januaries with different totals.
    idx2 = pd.date_range("1991-01-01", periods=31, freq="D")
    two_years = pd.concat([series, pd.Series(np.full(31, 4.0), index=idx2)])
    as_sum = to_monthly_anomalies(two_years, "pr")
    as_mean = to_monthly_anomalies(two_years, "tas")
    # sums are 62 and 124 (anomalies -31/+31); means are 2 and 4 (-1/+1)
    assert as_sum.to_numpy() == pytest.approx([-31.0, 31.0])
    assert as_mean.to_numpy() == pytest.approx([-1.0, 1.0])
    assert len(mean_based) == 1


def test_to_monthly_anomalies_removes_the_seasonal_cycle():
    series = _correlated_series(seed=0)
    monthly = to_monthly_anomalies(series, "tas")
    # the raw seasonal swing is +/-10 degC; anomalies must be far smaller
    assert monthly.abs().max() < 2.0


def test_to_monthly_anomalies_empty_input():
    assert to_monthly_anomalies(pd.Series(dtype="float64"), "tas").empty


# ---------------------------------------------------------------------------
# select_neighbors
# ---------------------------------------------------------------------------

def test_select_neighbors_ranks_by_distance_and_respects_radius():
    candidates = pd.DataFrame({
        "canonical_station_id": ["target", "near", "mid", "far"],
        "lat": [38.0, 38.05, 38.5, 45.0],
        "lon": [24.0, 24.05, 24.5, 30.0],
    })
    result = select_neighbors("target", 38.0, 24.0, candidates, radius_km=200)
    assert result[0] == "near"
    assert "far" not in result
    assert "target" not in result


def test_select_neighbors_respects_max_neighbors():
    candidates = pd.DataFrame({
        "canonical_station_id": [f"s{i}" for i in range(10)],
        "lat": [38.0 + i * 0.01 for i in range(10)],
        "lon": [24.0] * 10,
    })
    result = select_neighbors("s0", 38.0, 24.0, candidates, max_neighbors=3, radius_km=1000)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# build_reference_difference
# ---------------------------------------------------------------------------

def test_build_reference_difference_averages_across_passing_neighbors():
    target = _monthly(60, seed=0)
    ref, n_used = build_reference_difference(
        target, {"a": target - 1.0, "b": target - 3.0}
    )
    assert n_used == 2
    assert ref.mean() == pytest.approx(2.0, abs=1e-6)


def test_build_reference_difference_excludes_short_overlap():
    target = _monthly(60, seed=0)
    short = target.iloc[: MIN_NEIGHBOR_OVERLAP_MONTHS - 1]
    ref, n_used = build_reference_difference(target, {"a": short})
    assert n_used == 0
    assert ref.empty


def test_build_reference_difference_excludes_low_correlation():
    target = _monthly(60, seed=0)
    unrelated = _monthly(60, seed=99)
    ref, n_used = build_reference_difference(target, {"a": unrelated}, min_correlation=0.6)
    assert n_used == 0


def test_build_reference_difference_excludes_zero_variance_neighbor():
    target = _monthly(60, seed=0)
    constant = pd.Series(5.0, index=target.index)
    ref, n_used = build_reference_difference(target, {"a": constant})
    assert n_used == 0


def test_build_reference_difference_one_bad_neighbor_does_not_poison_a_good_one():
    target = _monthly(60, seed=0)
    ref, n_used = build_reference_difference(
        target, {"good": target - 1.0, "bad": _monthly(60, seed=99)}
    )
    assert n_used == 1
    assert ref.mean() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# snht_statistic / _restricted_max
# ---------------------------------------------------------------------------

def test_snht_statistic_hand_computed_symmetric_step():
    z = np.array([-1.0, -1.0, 1.0, 1.0])
    profile = snht_statistic(z)
    assert profile.shape == (3,)
    assert profile[0] == pytest.approx(4.0 / 3.0, abs=1e-6)
    assert profile[1] == pytest.approx(4.0, abs=1e-6)
    assert profile[2] == pytest.approx(4.0 / 3.0, abs=1e-6)
    assert int(np.argmax(profile)) == 1   # k=2, the true midpoint break


def test_snht_statistic_empty_for_degenerate_input():
    assert snht_statistic(np.array([1.0])).size == 0
    assert snht_statistic(np.array([])).size == 0


def test_snht_statistic_peaks_higher_for_a_real_step_than_for_noise():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, 1000)
    noise = (noise - noise.mean()) / noise.std()
    stepped = noise.copy()
    stepped[500:] += 5.0
    stepped = (stepped - stepped.mean()) / stepped.std()
    assert np.max(snht_statistic(stepped)) > np.max(snht_statistic(noise)) * 3


def test_restricted_max_excludes_near_edge_candidates():
    n = 40
    z = np.zeros(n)
    z[:2] = 20.0              # extreme near-edge outlier segment
    z[20:] = 1.0               # mild genuine mid-series step
    z = (z - z.mean()) / z.std()
    profile = snht_statistic(z)

    unrestricted_k, _ = _restricted_max(profile, n, min_segment=1)
    restricted_k, _ = _restricted_max(profile, n, min_segment=10)

    assert unrestricted_k == 2
    assert restricted_k is not None
    assert 10 <= restricted_k <= n - 10
    assert restricted_k != unrestricted_k


def test_restricted_max_returns_none_when_no_candidate_fits():
    profile = snht_statistic(np.array([1.0, -1.0, 1.0, -1.0]))
    best_k, best_t = _restricted_max(profile, 4, min_segment=10)
    assert best_k is None
    assert best_t == float("-inf")


def test_restricted_max_rejects_a_near_midpoint_only_window():
    n, min_segment = 2 * 12 + 4, 12    # window only 4 wide
    z = np.random.default_rng(0).normal(0, 1, n)
    best_k, best_t = _restricted_max(snht_statistic(z), n, min_segment)
    assert best_k is None
    assert best_t == float("-inf")


def test_restricted_max_accepts_a_genuinely_wide_window():
    n, min_segment = 3 * 12, 12        # exactly at the 3x threshold
    z = np.random.default_rng(0).normal(0, 1, n)
    best_k, best_t = _restricted_max(snht_statistic(z), n, min_segment)
    assert best_k is not None
    assert min_segment <= best_k <= n - min_segment


# ---------------------------------------------------------------------------
# detect_breakpoint
# ---------------------------------------------------------------------------

def test_detect_breakpoint_finds_an_injected_shift():
    bp_idx = 2200        # ~month 72 of 120
    target = _correlated_series(seed=1, shift_after=bp_idx, shift_amount=3.0)
    neighbors = {f"n{i}": _correlated_series(seed=100 + i) for i in range(3)}

    result = detect_breakpoint("target", "tas", target, neighbors, n_permutations=200, seed=0)

    assert result.status == "tested"
    assert result.n_neighbors_used == 3
    assert result.significant
    true_date = target.index[bp_idx]
    assert abs((result.breakpoint_date - true_date).days) < 45   # within ~1 month
    assert result.shift == pytest.approx(3.0, abs=0.5)
    assert result.statistic >= MIN_SNHT_STATISTIC


def test_detect_breakpoint_no_injected_shift_is_not_significant():
    target = _correlated_series(seed=1)
    neighbors = {f"n{i}": _correlated_series(seed=100 + i) for i in range(3)}
    result = detect_breakpoint("target", "tas", target, neighbors, n_permutations=200, seed=0)
    assert not result.significant


def test_detect_breakpoint_requires_two_neighbors():
    """A single neighbor cannot say whether the target or the neighbor moved."""
    bp_idx = 2200
    target = _correlated_series(seed=1, shift_after=bp_idx, shift_amount=3.0)
    one_neighbor = {"n0": _correlated_series(seed=100)}

    result = detect_breakpoint("target", "tas", target, one_neighbor, n_permutations=50, seed=0)

    assert result.n_neighbors_used == 1
    assert result.status == "too_few_neighbors"
    assert not result.significant
    assert result.breakpoint_date is None


def test_detect_breakpoint_applies_the_conservative_statistic_floor():
    """A break that clears the permutation p-value but not Climatol's
    SNHT>=20 convention is reported, but not called significant."""
    bp_idx = 2200
    target = _correlated_series(seed=1, shift_after=bp_idx, shift_amount=3.0)
    neighbors = {f"n{i}": _correlated_series(seed=100 + i) for i in range(3)}

    strict = detect_breakpoint("target", "tas", target, neighbors,
                                n_permutations=100, min_statistic=1e9, seed=0)
    lenient = detect_breakpoint("target", "tas", target, neighbors,
                                 n_permutations=100, min_statistic=0.0, seed=0)

    assert lenient.significant
    assert not strict.significant                 # same p-value, floor not cleared
    assert strict.p_value == lenient.p_value
    assert strict.status == "tested"              # still reported, just not flagged


def test_detect_breakpoint_with_no_usable_neighbor_reports_status():
    target = _correlated_series(seed=1)
    unrelated = pd.Series(
        np.random.default_rng(9).normal(0, 1, len(target)), index=target.index
    )
    result = detect_breakpoint("target", "tas", target, {"n0": unrelated}, n_permutations=50)
    assert result.n_neighbors_used == 0
    assert result.status == "no_usable_neighbor"
    assert result.breakpoint_date is None
    assert not result.significant


def test_detect_breakpoint_reports_series_too_short():
    n = 30 * (3 * MIN_SEGMENT_MONTHS - 6)     # too few months for a real window
    target = _correlated_series(n=n, seed=1)
    neighbors = {f"n{i}": _correlated_series(n=n, seed=10 + i) for i in range(3)}
    result = detect_breakpoint("target", "tas", target, neighbors, n_permutations=20)
    assert result.status == "series_too_short"
    assert result.breakpoint_date is None
    assert not result.significant


def test_detect_breakpoint_never_mutates_input_series():
    target = _correlated_series(seed=1)
    neighbors = {f"n{i}": _correlated_series(seed=10 + i) for i in range(2)}
    target_before = target.copy()
    detect_breakpoint("target", "tas", target, neighbors, n_permutations=20)
    pd.testing.assert_series_equal(target, target_before)


# ---------------------------------------------------------------------------
# variable_series_by_station / run_report
# ---------------------------------------------------------------------------

def test_variable_series_by_station_pivots_correctly():
    values = pd.DataFrame({
        "canonical_station_id": ["a", "a", "b"],
        "date": pd.to_datetime(["1990-01-01", "1990-01-02", "1990-01-01"]),
        "variable": ["tas", "tas", "tas"],
        "value": [10.0, 11.0, 20.0],
    })
    result = variable_series_by_station(values, "tas")
    assert set(result) == {"a", "b"}
    assert list(result["a"].to_numpy()) == [10.0, 11.0]
    assert list(result["b"].to_numpy()) == [20.0]


def test_variable_series_by_station_ignores_other_variables():
    values = pd.DataFrame({
        "canonical_station_id": ["a"],
        "date": pd.to_datetime(["1990-01-01"]),
        "variable": ["pr"],
        "value": [1.0],
    })
    assert variable_series_by_station(values, "tas") == {}


def _archive_frame(series_by_id, variable="tas"):
    return pd.concat([
        pd.DataFrame({"canonical_station_id": sid, "date": s.index,
                       "variable": variable, "value": s.to_numpy()})
        for sid, s in series_by_id.items()
    ], ignore_index=True)


def test_run_report_skips_stations_too_short_to_test():
    long_a = _correlated_series(seed=1)
    long_b = _correlated_series(seed=2)
    short = _correlated_series(n=100, seed=3)
    values = _archive_frame({"a": long_a, "b": long_b, "short": short})
    stations_meta = pd.DataFrame({
        "canonical_station_id": ["a", "b", "short"],
        "lat": [38.0, 38.05, 38.1], "lon": [24.0, 24.05, 24.1],
    })

    report = run_report(values, stations_meta, ["tas"], n_permutations=30, seed=0)

    assert "short" not in set(report["canonical_station_id"])
    assert {"a", "b"} <= set(report["canonical_station_id"])


def test_run_report_adds_status_and_fdr_columns():
    values = _archive_frame({
        "a": _correlated_series(seed=1),
        "b": _correlated_series(seed=2),
        "c": _correlated_series(seed=3),
    })
    stations_meta = pd.DataFrame({
        "canonical_station_id": ["a", "b", "c"],
        "lat": [38.0, 38.05, 38.1], "lon": [24.0, 24.05, 24.1],
    })
    report = run_report(values, stations_meta, ["tas"], n_permutations=30, seed=0)
    assert "significant_fdr" in report.columns
    assert "status" in report.columns
    assert report["significant_fdr"].dtype == bool


def test_run_report_fdr_column_never_exceeds_uncorrected():
    """FDR narrows what `significant` allows; it can never flag more. The
    correction must respect the SNHT floor and neighbor gate too, not just
    the p-value."""
    values = _archive_frame({
        "a": _correlated_series(seed=1, shift_after=2200, shift_amount=3.0),
        "b": _correlated_series(seed=2),
        "c": _correlated_series(seed=3),
        "d": _correlated_series(seed=4),
    })
    stations_meta = pd.DataFrame({
        "canonical_station_id": ["a", "b", "c", "d"],
        "lat": [38.0, 38.05, 38.1, 38.15], "lon": [24.0, 24.05, 24.1, 24.15],
    })
    report = run_report(values, stations_meta, ["tas"], n_permutations=50, seed=0)
    assert report["significant_fdr"].sum() <= report["significant"].sum()
    # and every FDR-flagged row is also flagged uncorrected
    assert not (report["significant_fdr"] & ~report["significant"]).any()


def test_run_report_returns_empty_frame_for_no_data():
    stations_meta = pd.DataFrame(columns=["canonical_station_id", "lat", "lon"])
    values = pd.DataFrame(columns=["canonical_station_id", "date", "variable", "value"])
    assert run_report(values, stations_meta, ["tas"]).empty


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_hand_computed():
    p = np.array([0.001, 0.01, 0.02, 0.03, 0.5])
    np.testing.assert_array_equal(
        benjamini_hochberg(p, alpha=0.05), [True, True, True, True, False]
    )


def test_benjamini_hochberg_is_order_independent():
    p = np.array([0.5, 0.001, 0.03, 0.01, 0.02])
    flagged = benjamini_hochberg(p, alpha=0.05)
    expected = {0.001: True, 0.01: True, 0.02: True, 0.03: True, 0.5: False}
    for value, flag in zip(p, flagged):
        assert flag == expected[value]


def test_benjamini_hochberg_never_flags_nan():
    flagged = benjamini_hochberg(np.array([0.001, np.nan, 0.01, np.nan]), alpha=0.05)
    assert not flagged[1]
    assert not flagged[3]


def test_benjamini_hochberg_all_nan_returns_all_false():
    assert not benjamini_hochberg(np.array([np.nan, np.nan])).any()


def test_benjamini_hochberg_more_conservative_than_uncorrected():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    assert benjamini_hochberg(p, alpha=0.05).sum() <= (p < 0.05).sum()
