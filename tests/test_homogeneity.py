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

from station_archive.homogeneity import (
    FLAG_COLUMNS,
    FLAG_LAYER_COLUMNS,
    MIN_NEIGHBORS,
    MIN_NEIGHBOR_OVERLAP_MONTHS,
    MIN_SEGMENT_MONTHS,
    MIN_SNHT_STATISTIC,
    PRECIP_LOG_OFFSET_MM,
    _restricted_max,
    apply_flag_layer,
    benjamini_hochberg,
    build_flag_layer,
    build_reference_difference,
    detect_breakpoint,
    run_report,
    select_neighbors,
    snht_statistic,
    to_monthly_anomalies,
    transform_for,
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


def _precip_series(n=_DEFAULT_DAYS, seed=0, start="1990-01-01",
                    scale_after=None, factor=1.0, weather_seed=555):
    """
    A realistic daily precipitation series: a seasonal wet/dry cycle, a
    REGIONAL monthly wetness factor shared by every station (same
    weather_seed), and skewed day-to-day variation.

    Regional wetness is multiplicative, not additive, which is how rainfall
    actually varies between a wet winter and a dry one -- and why neighbors
    correlate on the LOG scale rather than in millimetres.

    `factor` scales every value from `scale_after` onward, the precipitation
    analogue of an instrument change: a gauge that under-catches by 30%
    shifts the record by a ratio, not by a fixed number of millimetres.
    """
    idx = pd.date_range(start, periods=n, freq="D")
    seasonal = 3.0 + 2.5 * np.cos(2 * np.pi * idx.dayofyear.to_numpy() / 365.25)
    months = idx.to_period("M")
    unique_months = months.unique()
    regional = pd.Series(
        np.exp(np.random.default_rng(weather_seed).normal(0, 0.4, len(unique_months))),
        index=unique_months,
    )
    local = np.random.default_rng(seed).gamma(shape=2.0, scale=0.5, size=n)
    values = seasonal * regional.reindex(months).to_numpy() * local
    if scale_after is not None:
        values = values.copy()
        values[scale_after:] *= factor
    return pd.Series(values, index=idx)


def _monthly(n_months, seed=0, start="1990-01-01"):
    """A monthly-frequency series, as build_reference_difference expects."""
    idx = pd.date_range(start, periods=n_months, freq="MS")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1.0, n_months), index=idx)


# ---------------------------------------------------------------------------
# The permutation p-value floor
# ---------------------------------------------------------------------------

def test_permutation_p_value_is_never_exactly_zero():
    """
    A permutation test with n shuffles cannot resolve a p-value below
    1/(n+1), so reporting 0.0 claims certainty the method does not have.
    The conventional estimator is (1 + count) / (1 + n).

    This matters because p feeds Benjamini-Hochberg. With p = 0 a strong break
    passes BH at any rank; with the floor it must clear i/m * alpha, so on a
    report where few series carry a break the two answers differ.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("1985-01-01", periods=365 * 14, freq="D")
    base = rng.normal(15.0, 3.0, len(idx))
    target = base.copy()
    target[len(idx) // 2:] += 12.0            # an unmistakable step
    neighbors = {
        f"n{i}": pd.Series(base + rng.normal(0, 0.3, len(idx)), index=idx)
        for i in range(3)
    }

    result = detect_breakpoint(
        "target", "tas", pd.Series(target, index=idx), neighbors,
        n_permutations=99, seed=1,
    )
    assert result.status == "tested"
    assert result.significant
    assert result.p_value == pytest.approx(1 / 100)
    assert result.p_value > 0.0


def test_the_floor_is_the_only_change_to_a_middling_p_value():
    """
    A p-value with permutations above the observed statistic keeps the
    (1 + count) / (1 + n) form and is not clamped to anything.
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("1985-01-01", periods=365 * 14, freq="D")
    base = rng.normal(15.0, 3.0, len(idx))
    neighbors = {
        f"n{i}": pd.Series(base + rng.normal(0, 0.3, len(idx)), index=idx)
        for i in range(3)
    }
    result = detect_breakpoint(
        "target", "tas", pd.Series(base, index=idx), neighbors,
        n_permutations=49, seed=2,
    )
    assert result.status == "tested"
    # No real break: p should sit well away from the floor, and be a multiple
    # of 1/50 because that is the resolution 49 permutations provide.
    assert result.p_value > 1 / 50
    assert (result.p_value * 50) == pytest.approx(round(result.p_value * 50))


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


def test_to_monthly_anomalies_counts_values_not_rows():
    """
    The day threshold is about how many OBSERVATIONS built the monthly value,
    so a NaN-padded month must not pass on its row count.

    This gated on grouped.size(), which counts rows including NaN, while the
    aggregation itself skips NaN -- so a 31-row January holding 5 real values
    cleared a 20-day minimum and produced a "monthly mean" of five days. A
    daily series reindexed to a complete calendar (asfreq, or a pivot) is the
    ordinary way to arrive here, and nothing in the signature forbids it.
    """
    idx = pd.date_range("1990-01-01", periods=31, freq="D")
    values = np.full(31, np.nan)
    values[:5] = [1.0, 2.0, 3.0, 4.0, 50.0]
    sparse = pd.Series(values, index=idx)
    assert to_monthly_anomalies(sparse, "tas").empty

    # 20 real values in the same 31-row month is enough, as it always was.
    values[:20] = 1.0
    assert len(to_monthly_anomalies(pd.Series(values, index=idx), "tas")) == 1


def test_to_monthly_anomalies_unaffected_when_no_values_are_missing():
    """The fix above must not move a dense series, which is every real case."""
    idx = pd.date_range("1990-01-01", periods=120, freq="D")
    dense = pd.Series(np.arange(120, dtype=float), index=idx)
    monthly = to_monthly_anomalies(dense, "tas")
    assert len(monthly) == 4


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
    # Precipitation totals 62 and 124 mm, but pr is tested multiplicatively,
    # so the anomaly is the log ratio to the calendar month's geometric mean
    # rather than the mm difference: +/- half of log(125/63) after the
    # PRECIP_LOG_OFFSET_MM offset. Temperature stays additive: means 2 and 4.
    half_log_ratio = 0.5 * np.log(
        (124 + PRECIP_LOG_OFFSET_MM) / (62 + PRECIP_LOG_OFFSET_MM)
    )
    assert as_sum.to_numpy() == pytest.approx([-half_log_ratio, half_log_ratio])
    assert as_mean.to_numpy() == pytest.approx([-1.0, 1.0])
    assert len(mean_based) == 1


def test_to_monthly_anomalies_precipitation_anomaly_is_scale_free():
    """
    The reason pr is tested on a log basis: the same PROPORTIONAL change
    must register the same whether it happens in a wet month or a dry one.
    On a difference basis a 50% change in a 200 mm month dwarfs a 50%
    change in a 20 mm one, which is precisely the non-constant variance
    SNHT assumes away.
    """
    wet = pd.concat([
        pd.Series(np.full(31, 6.0), index=pd.date_range("1990-01-01", periods=31, freq="D")),
        pd.Series(np.full(31, 9.0), index=pd.date_range("1991-01-01", periods=31, freq="D")),
    ])
    dry = pd.concat([
        pd.Series(np.full(30, 0.6), index=pd.date_range("1990-04-01", periods=30, freq="D")),
        pd.Series(np.full(30, 0.9), index=pd.date_range("1991-04-01", periods=30, freq="D")),
    ])
    wet_anomaly = to_monthly_anomalies(wet, "pr").abs().max()
    dry_anomaly = to_monthly_anomalies(dry, "pr").abs().max()

    # Both months change by the same factor (x1.5), so the log anomalies are
    # of the same order -- within a factor of two of each other, despite the
    # raw totals differing by 10x. In millimetres they would differ ~10x.
    assert dry_anomaly > 0
    assert 0.5 < wet_anomaly / dry_anomaly < 2.0


def test_to_monthly_anomalies_drops_near_dry_calendar_months():
    """
    A calendar month whose climatology is a millimetre or two carries no
    usable ratio -- one showery month would read as a huge multiplicative
    anomaly. Those months are dropped before the transform.
    """
    wet_july = pd.Series(
        np.full(31, 5.0), index=pd.date_range("1990-07-01", periods=31, freq="D")
    )
    bone_dry_august = pd.Series(
        np.full(31, 0.01), index=pd.date_range("1990-08-01", periods=31, freq="D")
    )
    result = to_monthly_anomalies(pd.concat([wet_july, bone_dry_august]), "pr")
    assert list(result.index.month) == [7]


def test_to_monthly_anomalies_handles_a_completely_dry_month_without_error():
    """log(0) must not appear: PRECIP_LOG_OFFSET_MM makes 0 mm representable."""
    wet = pd.Series(np.full(31, 5.0), index=pd.date_range("1990-01-01", periods=31, freq="D"))
    zero = pd.Series(np.zeros(31), index=pd.date_range("1991-01-01", periods=31, freq="D"))
    result = to_monthly_anomalies(pd.concat([wet, zero]), "pr")
    assert len(result) == 2
    assert np.isfinite(result.to_numpy()).all()


def test_transform_for_marks_only_precipitation_multiplicative():
    assert transform_for("pr") == "log_ratio"
    assert transform_for("tas") == "difference"
    assert transform_for("sfcWind") == "difference"


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


def test_detect_breakpoint_finds_a_multiplicative_precipitation_shift():
    """
    End to end on the log basis: a gauge that starts catching 50% more is a
    RATIO change, and must be recovered as one.
    """
    bp_idx = 2200
    target = _precip_series(seed=1, scale_after=bp_idx, factor=1.5)
    neighbors = {f"n{i}": _precip_series(seed=100 + i) for i in range(3)}

    result = detect_breakpoint("target", "pr", target, neighbors, n_permutations=200, seed=0)

    assert result.status == "tested"
    assert result.transform == "log_ratio"
    assert result.significant
    true_date = target.index[bp_idx]
    assert abs((result.breakpoint_date - true_date).days) < 45
    # `shift` is in log units; the factor it represents is what was injected.
    assert result.shift_as_ratio() == pytest.approx(1.5, abs=0.25)


def test_detect_breakpoint_precipitation_without_a_shift_is_not_significant():
    target = _precip_series(seed=1)
    neighbors = {f"n{i}": _precip_series(seed=100 + i) for i in range(3)}
    result = detect_breakpoint("target", "pr", target, neighbors, n_permutations=200, seed=0)
    assert not result.significant


def test_detect_breakpoint_records_the_transform_on_every_status():
    """`shift` cannot be read without knowing its basis, so the basis is
    recorded even on results where no test ran."""
    short = _precip_series(n=600, seed=1)
    null_result = detect_breakpoint("target", "pr", short, {}, n_permutations=10)
    assert null_result.status != "tested"
    assert null_result.transform == "log_ratio"
    assert np.isnan(null_result.shift_as_ratio())

    additive = detect_breakpoint("target", "tas", _correlated_series(n=600), {}, n_permutations=10)
    assert additive.transform == "difference"
    assert np.isnan(additive.shift_as_ratio())   # a ratio is undefined here


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
# build_flag_layer / apply_flag_layer
# ---------------------------------------------------------------------------

def _report_row(station, variable="tas", significant_fdr=True, significant=True,
                 breakpoint_date="1996-01-01"):
    return {
        "canonical_station_id": station,
        "variable": variable,
        "n_neighbors_used": 3,
        "n_months_tested": 120,
        "breakpoint_date": pd.Timestamp(breakpoint_date),
        "statistic": 55.0,
        "p_value": 0.002,
        "significant": significant,
        "segment_before_mean": -1.0,
        "segment_after_mean": 1.0,
        "shift": 2.0,
        "status": "tested",
        "transform": transform_for(variable),
        "shift_ratio": float("nan"),
        "significant_fdr": significant_fdr,
    }


def test_build_flag_layer_keeps_only_fdr_surviving_rows():
    report = pd.DataFrame([
        _report_row("a", significant_fdr=True),
        _report_row("b", significant_fdr=False, significant=True),
        _report_row("c", significant_fdr=False, significant=False),
    ])
    layer = build_flag_layer(report)
    assert list(layer["canonical_station_id"]) == ["a"]
    assert list(layer.columns) == list(FLAG_LAYER_COLUMNS)
    assert layer["flag"].unique().tolist() == ["breakpoint_suspected"]
    assert layer["detected_by"].nunique() == 1


def test_build_flag_layer_empty_report_returns_typed_empty_frame():
    empty = build_flag_layer(pd.DataFrame())
    assert empty.empty
    assert list(empty.columns) == list(FLAG_LAYER_COLUMNS)


def test_build_flag_layer_no_survivors_returns_typed_empty_frame():
    report = pd.DataFrame([_report_row("a", significant_fdr=False)])
    layer = build_flag_layer(report)
    assert layer.empty
    assert list(layer.columns) == list(FLAG_LAYER_COLUMNS)


def _values_frame():
    dates = pd.to_datetime(["1994-06-01", "1996-01-01", "1998-03-01"])
    return pd.DataFrame({
        "canonical_station_id": ["a"] * 3 + ["b"] * 3,
        "date": list(dates) * 2,
        "variable": ["tas"] * 6,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


def test_apply_flag_layer_marks_segments_either_side_of_the_break():
    layer = build_flag_layer(pd.DataFrame([_report_row("a", breakpoint_date="1996-01-01")]))
    flagged = apply_flag_layer(_values_frame(), layer)

    station_a = flagged[flagged["canonical_station_id"] == "a"]
    # the breakpoint date itself belongs to the segment AFTER the break
    assert list(station_a["homogeneity_segment"]) == [0, 1, 1]
    assert station_a["homogeneity_flag"].all()

    station_b = flagged[flagged["canonical_station_id"] == "b"]
    assert not station_b["homogeneity_flag"].any()
    assert station_b["homogeneity_segment"].isna().all()


def test_apply_flag_layer_never_alters_observations():
    """The archive diagnoses and never corrects: values, rows and existing
    columns must come through untouched."""
    values = _values_frame()
    layer = build_flag_layer(pd.DataFrame([_report_row("a")]))
    flagged = apply_flag_layer(values, layer)

    assert len(flagged) == len(values)
    pd.testing.assert_frame_equal(flagged[values.columns], values)


def test_apply_flag_layer_refuses_to_overwrite_an_existing_flag_column():
    """Plan 1.4: layer project flags rather than overwriting them."""
    values = _values_frame()
    values["homogeneity_flag"] = False
    with pytest.raises(ValueError, match="refusing to overwrite"):
        apply_flag_layer(values, build_flag_layer(pd.DataFrame([_report_row("a")])))


def test_apply_flag_layer_with_an_empty_layer_still_adds_the_columns():
    """A consumer reads the flag columns unconditionally, so an archive with
    no flags must still carry them."""
    flagged = apply_flag_layer(_values_frame(), build_flag_layer(pd.DataFrame()))
    for column in FLAG_COLUMNS:
        assert column in flagged.columns
    assert not flagged["homogeneity_flag"].any()
    assert flagged["homogeneity_segment"].isna().all()


def test_apply_flag_layer_is_variable_specific():
    """A break in tas says nothing about pr at the same station."""
    values = _values_frame()
    values = pd.concat([values, values.assign(variable="pr")], ignore_index=True)
    layer = build_flag_layer(pd.DataFrame([_report_row("a", variable="tas")]))
    flagged = apply_flag_layer(values, layer)

    assert flagged[(flagged["canonical_station_id"] == "a")
                   & (flagged["variable"] == "tas")]["homogeneity_flag"].all()
    assert not flagged[(flagged["canonical_station_id"] == "a")
                       & (flagged["variable"] == "pr")]["homogeneity_flag"].any()


def test_run_report_output_flows_into_the_flag_layer():
    """The two halves must actually fit together on real report output."""
    values = _archive_frame({
        "a": _correlated_series(seed=1, shift_after=2200, shift_amount=4.0),
        "b": _correlated_series(seed=2),
        "c": _correlated_series(seed=3),
    })
    stations_meta = pd.DataFrame({
        "canonical_station_id": ["a", "b", "c"],
        "lat": [38.0, 38.05, 38.1], "lon": [24.0, 24.05, 24.1],
    })
    report = run_report(values, stations_meta, ["tas"], n_permutations=50, seed=0)
    layer = build_flag_layer(report)

    assert len(layer) == int(report["significant_fdr"].sum())
    flagged = apply_flag_layer(values, layer)
    assert set(FLAG_COLUMNS) <= set(flagged.columns)


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


# ---------------------------------------------------------------------------
# Neighbour support at the breakpoint
#
# The two-neighbour rule was checked against a whole-series count while the
# SNHT search ran over months whose support could be 1, so a break belonging to
# a NEIGHBOUR was reported as the target's with high confidence. Measured on
# the real archive, 95 of 342 published flags had a detected break in a window
# where only one neighbour contributed.
# ---------------------------------------------------------------------------

def _staggered_neighbours(n=_DEFAULT_DAYS, half=None, neighbour_break_at=None,
                           neighbour_shift=4.0):
    """
    Two neighbours covering disjoint halves of the target's record, the second
    of which moves partway through its own window -- so during that window the
    only evidence about the target comes from a station that itself shifted.
    """
    half = n // 2 if half is None else half
    a = _correlated_series(n=n, seed=2).iloc[:half]
    values = _correlated_series(n=n, seed=3)
    if neighbour_break_at is not None:
        values = values.copy()
        values.iloc[neighbour_break_at:] += neighbour_shift
    return {"A": a, "B": values.iloc[half:]}


def test_break_belonging_to_a_solo_neighbour_is_not_flagged():
    target = _correlated_series(seed=1)          # the target is clean
    neighbours = _staggered_neighbours(neighbour_break_at=2725)
    result = detect_breakpoint("target", "tas", target, neighbours,
                                n_permutations=200, seed=0)
    assert result.n_neighbors_used == 2          # the whole-series count is met
    assert not result.significant                # but the break window is not


def test_result_reports_support_at_the_breakpoint():
    target = _correlated_series(seed=1)
    neighbours = _staggered_neighbours(neighbour_break_at=2725)
    result = detect_breakpoint("target", "tas", target, neighbours,
                                n_permutations=50, seed=0)
    assert result.min_support_at_break < MIN_NEIGHBORS or result.breakpoint_date is None
    assert result.n_months_min_support == 0      # no month has two neighbours here


def test_fully_supported_break_is_still_found():
    """Regression: the fix must not suppress a break the safeguard allows."""
    bp = 2200
    target = _correlated_series(seed=1, shift_after=bp, shift_amount=3.0)
    neighbours = {f"n{i}": _correlated_series(seed=100 + i) for i in range(3)}
    result = detect_breakpoint("target", "tas", target, neighbours,
                                n_permutations=200, seed=0)
    assert result.significant
    assert result.min_support_at_break >= MIN_NEIGHBORS
    assert result.n_months_min_support > 0
    assert abs((result.breakpoint_date - target.index[bp]).days) < 45


def test_support_is_counted_per_month_not_per_series():
    from station_archive.homogeneity import (
        neighbor_support, passing_neighbor_differences, to_monthly_anomalies,
    )
    target = _correlated_series(seed=1)
    neighbours = _staggered_neighbours()
    mt = to_monthly_anomalies(target, "tas")
    mn = {k: to_monthly_anomalies(v, "tas") for k, v in neighbours.items()}
    diffs = passing_neighbor_differences(mt, mn)
    support = neighbor_support(diffs)
    assert len(diffs) == 2                       # two neighbours passed overall
    assert support.max() == 1                    # yet no single month has both


# ---------------------------------------------------------------------------
# Neighbour candidates must be filtered by data before being truncated
# ---------------------------------------------------------------------------

def test_neighbours_are_filtered_by_data_before_truncation():
    """
    The five nearest stations were chosen from ALL canonical stations and only
    then checked for the variable, so a station ringed by neighbours that lack
    it was reported untestable while usable neighbours sat inside the radius.
    On the real archive this cost 84 of 126 otherwise-testable gust stations.
    """
    near_without_data = {f"empty{i}": None for i in range(5)}
    series = {"target": _correlated_series(seed=1),
              "far1": _correlated_series(seed=2),
              "far2": _correlated_series(seed=3)}
    rows = [{"canonical_station_id": "target", "lat": 38.00, "lon": 24.0}]
    for i, name in enumerate(near_without_data):
        rows.append({"canonical_station_id": name, "lat": 38.01 + i * 0.001, "lon": 24.0})
    rows.append({"canonical_station_id": "far1", "lat": 38.20, "lon": 24.0})
    rows.append({"canonical_station_id": "far2", "lat": 38.25, "lon": 24.0})
    stations_meta = pd.DataFrame(rows)
    values = _archive_frame(series)

    report = run_report(values, stations_meta, ["tas"], n_permutations=20, seed=0)
    reached = set(report["canonical_station_id"])
    assert "target" in reached, "target was skipped although two usable neighbours exist"
    assert int(report.loc[report.canonical_station_id == "target",
                          "n_neighbors_used"].iloc[0]) == 2


# ---------------------------------------------------------------------------
# One dedupe rule for both flag appliers
# ---------------------------------------------------------------------------

def test_flag_layer_keeps_one_row_per_station_variable():
    report = pd.DataFrame([
        _report_row("a", breakpoint_date="1999-01-01"),
        _report_row("a", breakpoint_date="1996-01-01"),
    ])
    layer = build_flag_layer(report)
    assert len(layer) == 1
    assert layer.iloc[0]["breakpoint_date"] == pd.Timestamp("1996-01-01")


# ---------------------------------------------------------------------------
# flagged_breakpoint_table -- the Excel listing of where breaks are
# ---------------------------------------------------------------------------

def test_flagged_breakpoint_table_lists_surviving_breaks_with_station_identity():
    from station_archive.homogeneity import flagged_breakpoint_table

    report = pd.DataFrame([
        {**_report_row("a", variable="pr"), "shift": 0.3, "shift_ratio": 1.35,
         "min_support_at_break": 3},
        {**_report_row("b", significant_fdr=False), "min_support_at_break": 2},
        {**_report_row("c", variable="tasmax", breakpoint_date="1993-06-01"),
         "min_support_at_break": 2},
    ])
    stations = pd.DataFrame({
        "canonical_station_id": ["a", "b", "c"],
        "name": ["ALPHA", "BETA", "GAMMA"],
        "lat": [38.0, 39.0, 40.0],
        "lon": [23.0, 24.0, 25.0],
    })
    table = flagged_breakpoint_table(report, stations)
    assert list(table["canonical_station_id"]) == ["a", "c"]
    assert list(table["name"]) == ["ALPHA", "GAMMA"]
    for column in ("lat", "lon", "variable", "breakpoint_date", "transform",
                   "shift", "shift_ratio", "statistic", "p_value", "min_support_at_break"):
        assert column in table.columns


def test_flagged_breakpoint_table_of_nothing_is_empty_with_columns():
    from station_archive.homogeneity import flagged_breakpoint_table

    table = flagged_breakpoint_table(pd.DataFrame(), pd.DataFrame(
        columns=["canonical_station_id", "name", "lat", "lon"]))
    assert table.empty
    assert "breakpoint_date" in table.columns
