"""
Tests for source selection and gap filling (plan section 1.6).

The behaviours under test are the ones section 1.6 mandates: priority is
per-variable and explicit, a fallback is only ever used for days the primary
lacks, an incompatible source leaves the gap open rather than splicing, no
value is ever interpolated or averaged, and every selected value carries its
source, rank and reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from station_archive.policy import DEFAULT_POLICY
from station_archive.consolidate import (
    CONSOLIDATED_COLUMNS,
    NON_SELECTED_COLUMNS,
    check_compatibility,
    consolidate_station_variable,
    source_contribution,
    wet_day_fraction,
)


class FakeLoader:
    """Serves prepared series in PROVIDER units, as bronze holds them."""

    def __init__(self, series: dict | None = None) -> None:
        self._series = series or {}

    def series(self, source, station_id, variable):
        return self._series.get((source, station_id, variable), pd.Series(dtype="float64"))


def _daily(values, start="1990-01-01"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="D"), dtype="float64"
    )


def _aliases(*entries):
    return pd.DataFrame(
        [{"source": s, "source_station_id": i} for s, i in entries]
    )


def _seasonal(n, offset=0.0, seed=0, noise=3.0):
    """Daily temperature in tenths of a degC, as ECA&D and GHCN-D store it."""
    rng = np.random.default_rng(seed)
    base = 200 + 100 * np.sin(np.arange(n) * 2 * np.pi / 365.25)
    return base + offset * 10 + rng.normal(0, noise, n)


# ---------------------------------------------------------------------------
# Compatibility test (the four checks section 1.6 names)
# ---------------------------------------------------------------------------

def test_compatible_sources_pass():
    a = _daily(_seasonal(800, seed=1) / 10)
    b = _daily(_seasonal(800, seed=2) / 10)
    result = check_compatibility(a, b, "tasmax")
    assert result.compatible
    assert result.n_overlap == 800


def test_biased_fallback_is_refused():
    """
    Sources disagree by station -- for tas with an ECA&D primary and a GSOD
    fallback, measured overlap bias spans -1.60 to +0.58 degC. A step beyond
    the configured tolerance must not be spliced in silently; the gap stays
    open instead.
    """
    a = _daily(_seasonal(800, seed=1) / 10)
    b = _daily(_seasonal(800, offset=2.0, seed=1) / 10)
    result = check_compatibility(a, b, "tasmax")
    assert not result.compatible
    assert "bias" in result.reason
    assert result.bias == pytest.approx(2.0, abs=0.2)


def test_dispersion_mismatch_is_refused():
    """A source recording the same quantity should not be markedly smoother."""
    a = _daily(_seasonal(800, seed=1) / 10)
    flat = np.full(800, 20.0)
    result = check_compatibility(a, _daily(flat), "tasmax")
    assert not result.compatible
    assert "dispersion" in result.reason


def test_wet_day_frequency_mismatch_is_refused():
    """
    GSOD's "'0' includes a trace" convention moves wet-day frequency without
    necessarily moving the mean, so it is tested separately from bias.
    """
    rng = np.random.default_rng(3)
    wet = rng.choice([0.0, 5.0], size=400, p=[0.4, 0.6])
    dry = rng.choice([0.0, 5.0], size=400, p=[0.9, 0.1])
    result = check_compatibility(_daily(wet), _daily(dry), "pr")
    assert not result.compatible
    assert "wet-day frequency" in result.reason


def test_discontinuity_is_refused_even_when_overall_bias_cancels():
    """
    Two records that agree on average only because a warm half cancels a cold
    half are not interchangeable.
    """
    n = 800
    base = _seasonal(n, seed=1) / 10
    shifted = base.copy()
    shifted[: n // 2] += 3.0
    shifted[n // 2 :] -= 3.0
    result = check_compatibility(_daily(base), _daily(shifted), "tasmax")
    assert not result.compatible
    assert abs(result.bias) < 0.5                 # overall bias cancels
    assert "monthly departure" in result.reason


def test_a_thinly_shared_month_cannot_drive_the_discontinuity_refusal():
    """
    The departure test is a statement about monthly means, so a month built
    from one or two shared days is not a monthly mean and must not be able to
    veto a splice on one outlier.

    HomogeneityPolicy has carried min_days_per_month for exactly this reason;
    FallbackPolicy did not, and this gate drives 48 of 157 refusals in the
    published archive.
    """
    n = 800
    base = _seasonal(n, seed=1) / 10
    twin = base.copy()

    primary = _daily(base)
    fallback = _daily(twin)
    assert check_compatibility(primary, fallback, "tasmax").compatible

    # Give the fallback one extra day, far outside the primary's span, sitting
    # alone in its own calendar month with a large offset.
    lone = pd.Series(
        [base[0] + 40.0],
        index=pd.DatetimeIndex([pd.Timestamp("1990-01-01") + pd.Timedelta(days=n + 400)]),
    )
    with_outlier = pd.concat([fallback, lone])

    result = check_compatibility(primary, with_outlier, "tasmax")
    # That day is not shared with the primary, so it cannot enter the overlap
    # at all -- this guards the construction of the test below.
    assert result.compatible


def test_a_month_with_too_few_shared_days_is_excluded_from_the_departure():
    n = 800
    base = _seasonal(n, seed=1) / 10
    primary = _daily(base)
    fallback = _daily(base.copy())

    # A single shared day in a month of its own, disagreeing wildly. Included,
    # it is a 40-degree "monthly departure"; excluded, the pair is compatible.
    stray_date = pd.Timestamp("1990-01-01") + pd.Timedelta(days=n + 400)
    primary_plus = pd.concat([primary, pd.Series([0.0], index=pd.DatetimeIndex([stray_date]))])
    fallback_plus = pd.concat([fallback, pd.Series([40.0], index=pd.DatetimeIndex([stray_date]))])

    result = check_compatibility(primary_plus, fallback_plus, "tasmax")
    assert result.compatible, result.reason
    assert result.max_monthly_departure < 1.0


def test_an_overlap_with_no_well_populated_month_is_refused_outright():
    """
    Excluding thin months is strictly more permissive, so it opens a hole: an
    overlap long enough to pass min_overlap_days but spread so thinly that NO
    month qualifies would leave the discontinuity gate with nothing to test,
    and a gate with nothing to test used to score 0.0 and pass.

    A test that cannot be run has not been passed. Section 1.6: a failed
    compatibility test leaves the gap open.
    """
    # 200 shared days spread one-per-week over four years: every month holds
    # about four, below the 20-day minimum.
    idx = pd.date_range("1990-01-01", periods=200, freq="7D")
    rng = np.random.default_rng(5)
    values = rng.normal(15.0, 3.0, 200)
    primary = pd.Series(values, index=idx)
    fallback = pd.Series(values, index=idx)          # identical: nothing else can fail

    result = check_compatibility(primary, fallback, "tasmax")
    assert result.n_overlap == 200                    # clears min_overlap_days
    assert not result.compatible
    assert "shared days" in result.reason
    assert np.isnan(result.max_monthly_departure)


def test_short_overlap_cannot_establish_compatibility():
    a = _daily(_seasonal(800, seed=1) / 10)
    b = _daily(_seasonal(30, seed=2) / 10)
    result = check_compatibility(a, b, "tasmax")
    assert not result.compatible
    assert "overlapping days" in result.reason


def test_compatibility_reports_statistics_even_when_it_fails():
    """A refusal must be reviewable, not merely obeyed."""
    a = _daily(_seasonal(800, seed=1) / 10)
    b = _daily(_seasonal(800, offset=2.0, seed=1) / 10)
    row = check_compatibility(a, b, "tasmax").to_dict()
    assert row["n_overlap"] == 800
    assert not np.isnan(row["bias"])
    assert not np.isnan(row["dispersion_ratio"])


def test_wet_day_fraction_counts_days_at_or_above_threshold():
    assert wet_day_fraction(np.array([0.0, 0.5, 1.0, 5.0])) == pytest.approx(0.5)
    assert np.isnan(wet_day_fraction(np.array([])))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_highest_ranked_source_supplies_the_primary_series():
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): _daily(_seasonal(500, seed=1)),
        ("gsod", "S", "tasmax"): _daily(_seasonal(500, seed=2) / 10 * 9 / 5 + 32),
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("gsod", "S")), "tasmax", loader,
        priority=("ecad", "gsod"),
    )
    assert set(out.values["selected_source"]) == {"ecad"}
    assert not out.values["is_fallback"].any()


def test_values_arrive_in_canonical_units():
    """ECA&D stores tenths of a degC; the gold layer stores degC."""
    loader = FakeLoader({("ecad", "E", "tasmax"): _daily([200.0, 210.0, 220.0])})
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E")), "tasmax", loader, priority=("ecad",)
    )
    assert list(out.values["value"]) == pytest.approx([20.0, 21.0, 22.0])
    assert set(out.values["units"]) == {"degC"}


def test_fallback_fills_only_days_the_primary_lacks():
    primary = _daily(_seasonal(400, seed=1), start="1990-01-01")
    fallback = _daily(_seasonal(800, seed=1), start="1990-01-01")
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): primary,
        ("ghcnd", "G", "tasmax"): fallback,
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    assert len(out.values) == 800
    assert int((~out.values["is_fallback"]).sum()) == 400
    assert int(out.values["is_fallback"].sum()) == 400
    # No day is represented twice.
    assert not out.values["date"].duplicated().any()


def test_incompatible_fallback_leaves_the_gap_open():
    """
    Section 1.6: a failed compatibility test "leaves a gap ... it does not
    justify blind splicing".
    """
    primary = _daily(_seasonal(400, seed=1), start="1990-01-01")
    fallback = _daily(_seasonal(800, offset=5.0, seed=1), start="1990-01-01")
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): primary,
        ("ghcnd", "G", "tasmax"): fallback,
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    assert len(out.values) == 400                  # gap left open
    assert not out.values["is_fallback"].any()
    assert out.compatibility[0]["days_filled"] == 0
    assert not out.compatibility[0]["compatible"]


def test_every_value_records_source_rank_and_reason():
    loader = FakeLoader({("ecad", "E", "tasmax"): _daily(_seasonal(500, seed=1))})
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E")), "tasmax", loader, priority=("ecad",)
    )
    assert list(out.values.columns) == list(CONSOLIDATED_COLUMNS)
    assert (out.values["source_rank"] == 1).all()
    assert out.values["selection_reason"].str.len().gt(0).all()
    assert out.values["source_record_id"].str.contains("ecad:E:").all()


def test_source_outside_the_priority_never_contributes():
    """
    c3s_land duplicates GHCN-D, so it is excluded from every priority list
    and must not become a silent fallback for the records it copies.
    """
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): _daily(_seasonal(200, seed=1)),
        ("c3s_land", "C", "tasmax"): _daily(_seasonal(800, seed=1)),
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("c3s_land", "C")), "tasmax", loader,
        priority=("ecad", "ghcnd", "gsod"),
    )
    assert set(out.values["selected_source"]) == {"ecad"}
    assert len(out.values) == 200


def test_missing_days_are_never_interpolated():
    """Section 1.6: "Do not interpolate missing station observations"."""
    values = _seasonal(400, seed=1)
    series = _daily(values)
    series = series.drop(series.index[100:200])     # a 100-day hole
    loader = FakeLoader({("ecad", "E", "tasmax"): series})
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E")), "tasmax", loader, priority=("ecad",)
    )
    assert len(out.values) == 300                   # hole preserved
    gap = pd.date_range("1990-01-01", periods=400, freq="D")[100:200]
    assert not out.values["date"].isin(gap).any()


def test_no_candidates_yields_an_empty_frame():
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E")), "hurs", FakeLoader(), priority=("ecad",)
    )
    assert out.values.empty
    assert list(out.values.columns) == list(CONSOLIDATED_COLUMNS)


def test_selection_is_deterministic():
    """Section 1.8: an unchanged archive must reproduce the same result."""
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): _daily(_seasonal(400, seed=1)),
        ("ghcnd", "G", "tasmax"): _daily(_seasonal(800, seed=1)),
    })
    aliases = _aliases(("ghcnd", "G"), ("ecad", "E"))
    first = consolidate_station_variable("c:1", aliases, "tasmax", loader,
                                         priority=("ecad", "ghcnd")).values
    second = consolidate_station_variable("c:1", aliases, "tasmax", loader,
                                          priority=("ecad", "ghcnd")).values
    pd.testing.assert_frame_equal(first, second)


def test_priority_order_actually_changes_the_winner():
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): _daily(_seasonal(500, seed=1)),
        ("gsod", "S", "tasmax"): _daily(_seasonal(500, seed=1) / 10 * 9 / 5 + 32),
    })
    aliases = _aliases(("ecad", "E"), ("gsod", "S"))
    ecad_first = consolidate_station_variable("c:1", aliases, "tasmax", loader,
                                              priority=("ecad", "gsod")).values
    gsod_first = consolidate_station_variable("c:1", aliases, "tasmax", loader,
                                              priority=("gsod", "ecad")).values
    assert set(ecad_first["selected_source"]) == {"ecad"}
    assert set(gsod_first["selected_source"]) == {"gsod"}


def test_configured_priority_matches_measured_coverage():
    """
    The plan calls its own ordering "a research hypothesis, not the final
    order".  hurs has exactly one source in this domain, and c3s_land appears
    in no list because it redelivers GHCN-D.
    """
    assert DEFAULT_POLICY.source_priority["hurs"] == ("ecad",)
    assert DEFAULT_POLICY.source_priority["gust"][0] == "gsod"       # 203 vs ecad's 65
    for variable, order in DEFAULT_POLICY.source_priority.items():
        assert "c3s_land" not in order, variable
        assert len(set(order)) == len(order), variable


def test_gsod_ranks_below_ghcnd_for_precipitation():
    """
    Coverage favours GSOD (203 stations vs 155); semantics do not -- GSOD's
    zero "includes a trace" and 46% of its zero rows are non-observations.
    """
    order = DEFAULT_POLICY.source_priority["pr"]
    assert order.index("ghcnd") < order.index("gsod")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def test_source_contribution_reports_days_by_variable_and_year():
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): _daily(_seasonal(400, seed=1), start="1990-01-01"),
        ("ghcnd", "G", "tasmax"): _daily(_seasonal(800, seed=1), start="1990-01-01"),
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    audit = source_contribution(out.values)
    assert {"variable", "year", "selected_source", "is_fallback", "days"} <= set(audit.columns)
    assert audit["days"].sum() == len(out.values)


def test_source_contribution_empty_for_empty_input():
    assert source_contribution(pd.DataFrame(columns=list(CONSOLIDATED_COLUMNS))).empty


# ---------------------------------------------------------------------------
# Loader efficiency (not cosmetic: it decides whether consolidation is runnable)
# ---------------------------------------------------------------------------

def test_station_snapshots_are_parsed_once_for_all_variables():
    """
    A GSOD station has 51 station-year files and seven variables. Parsing per
    variable would read 357 files where 51 suffice, which across 1600
    stations is the difference between minutes and hours.
    """
    from station_archive import bronze

    calls = {"n": 0}

    class CountingLoader(bronze.SeriesLoader):
        def _station_map(self, source):
            return {"S": ["snapshot"]}

    def fake_all(source, snapshots, period=None, rejections=None):
        calls["n"] += 1
        return {
            "tasmax": _daily([200.0] * 10),
            "tasmin": _daily([100.0] * 10),
            "pr": _daily([0.0] * 10),
        }

    original = bronze.all_daily_series
    bronze.all_daily_series = fake_all
    try:
        loader = CountingLoader()
        for variable in ("tasmax", "tasmin", "pr", "tasmax"):
            loader.series("gsod", "S", variable)
    finally:
        bronze.all_daily_series = original

    assert calls["n"] == 1


def test_loader_returns_empty_for_a_variable_the_station_lacks():
    from station_archive import bronze

    class StubLoader(bronze.SeriesLoader):
        def _station_map(self, source):
            return {"S": ["snapshot"]}

    original = bronze.all_daily_series
    bronze.all_daily_series = (
        lambda s, snaps, period=None, rejections=None: {"tasmax": _daily([1.0])}
    )
    try:
        loader = StubLoader()
        assert loader.series("gsod", "S", "hurs").empty
        assert len(loader.series("gsod", "S", "tasmax")) == 1
    finally:
        bronze.all_daily_series = original


def test_clear_releases_cached_series():
    from station_archive import bronze

    class StubLoader(bronze.SeriesLoader):
        def _station_map(self, source):
            return {"S": ["snapshot"]}

    calls = {"n": 0}

    def fake_all(source, snapshots, period=None, rejections=None):
        calls["n"] += 1
        return {"tasmax": _daily([1.0])}

    original = bronze.all_daily_series
    bronze.all_daily_series = fake_all
    try:
        loader = StubLoader()
        loader.series("gsod", "S", "tasmax")
        loader.clear()
        loader.series("gsod", "S", "tasmax")
    finally:
        bronze.all_daily_series = original

    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Quality control (plan section 1.4)
# ---------------------------------------------------------------------------

def test_gsod_flag_i_zero_is_rejected_as_a_non_observation():
    """
    NOAA's readme: "I = Station did not report any precipitation data for the
    day".  46% of GSOD's zero-precipitation rows carry it; reading them as
    0.0 mm inflates dry-day frequency.
    """
    from station_archive import qc

    reason = qc.reject_reason("gsod", "pr", 0.0, {"attributes": "I"})
    assert reason is not None and "flag I" in reason


def test_gsod_flag_h_zero_is_rejected_on_the_providers_own_words():
    from station_archive import qc

    reason = qc.reject_reason("gsod", "pr", 0.0, {"attributes": "H"})
    assert reason is not None and "incomplete" in reason


def test_gsod_genuine_zero_is_kept():
    """Flags A-G describe which hours the total covers: real, if partial."""
    from station_archive import qc

    assert qc.reject_reason("gsod", "pr", 0.0, {"attributes": "F"}) is None
    assert qc.reject_reason("gsod", "pr", 0.04, {"attributes": "C"}) is None


def test_flag_rules_do_not_leak_to_other_sources_or_variables():
    from station_archive import qc

    assert qc.reject_reason("ecad", "pr", 0.0, {"attributes": "I"}) is None
    assert qc.reject_reason("gsod", "tasmax", 0.0, {"attributes": "I"}) is None


def test_negative_precipitation_is_rejected():
    from station_archive import qc

    assert qc.reject_reason("ecad", "pr", -5.0, {}) is not None


def test_gross_range_keeps_extremes_but_drops_the_impossible():
    """
    Deliberately wide: an aggressive filter would delete the extremes a
    downscaling model and a bias-adjustment reference most need.
    """
    from station_archive import qc

    values = pd.Series([-45.0, 48.0, 200.0, np.nan])
    mask = qc.check_gross_range(values, "tasmax")
    assert list(mask) == [True, True, False, False]


def test_relative_humidity_bounds_enforced():
    from station_archive import qc

    mask = qc.check_gross_range(pd.Series([0.0, 50.0, 100.0, 105.0]), "hurs")
    assert list(mask) == [True, True, True, False]


def test_a_variable_with_no_declared_range_still_rejects_nan():
    """
    The docstring's promise -- "NaN is treated as out of range so it cannot
    reach the gold layer as an accepted value" -- has to hold for every
    variable, not only the seven with bounds.

    The silver layer deliberately carries dew_point, tavg_unspecified and
    wind_max_sustained (standardize.CANONICAL_UNITS) so a later decision can
    use them. None has a GROSS_RANGE entry, and the unbounded branch used to
    return an all-True mask -- admitting NaN, and admitting the undocumented
    sentinels this check exists to catch.
    """
    from station_archive import qc

    values = pd.Series([12.0, np.nan, 999.0])
    mask = qc.check_gross_range(values, "dew_point")
    assert list(mask) == [True, False, True]     # NaN out; 999.0 not judged


def test_every_variable_the_silver_layer_carries_is_covered_by_the_nan_guard():
    from station_archive import qc, standardize

    for variable in standardize.CANONICAL_UNITS:
        mask = qc.check_gross_range(pd.Series([1.0, np.nan]), variable)
        assert not bool(mask.iloc[1]), f"{variable} admitted NaN"


def test_internal_consistency_flags_tmin_above_tmax():
    from station_archive import qc

    idx = pd.date_range("1990-01-01", periods=3, freq="D")
    out = qc.check_internal_consistency(
        pd.Series([5.0, 20.0, 1.0], index=idx),
        pd.Series([10.0, 15.0, 5.0], index=idx),
        pd.Series([15.0, 10.0, 9.0], index=idx),
    )
    assert len(out) == 1
    assert "tasmin > tasmax" in out.iloc[0]["violation"]


def test_internal_consistency_flags_mean_outside_the_range():
    from station_archive import qc

    idx = pd.date_range("1990-01-01", periods=2, freq="D")
    out = qc.check_internal_consistency(
        pd.Series([5.0, 5.0], index=idx),
        pd.Series([2.0, 20.0], index=idx),
        pd.Series([15.0, 15.0], index=idx),
    )
    assert len(out) == 2
    assert "tas < tasmin" in out.iloc[0]["violation"]
    assert "tas > tasmax" in out.iloc[1]["violation"]


def test_internal_consistency_ignores_days_with_missing_partners():
    """A station reporting tasmax but not tas is not inconsistent."""
    from station_archive import qc

    idx = pd.date_range("1990-01-01", periods=2, freq="D")
    out = qc.check_internal_consistency(
        pd.Series([np.nan, 5.0], index=idx),
        pd.Series([np.nan, 10.0], index=idx),
        pd.Series([15.0, 15.0], index=idx),
    )
    assert out.empty


def test_repeated_days_detects_a_stuck_sensor():
    from station_archive import qc

    idx = pd.date_range("1990-01-01", periods=20, freq="D")
    values = pd.Series([20.0] * 12 + list(range(8)), index=idx, dtype="float64")
    mask = qc.check_repeated_days(values, min_run=10)
    assert mask.iloc[:12].all()
    assert not mask.iloc[12:].any()


def test_coverage_summary_reports_missing_fraction_by_month():
    from station_archive import qc

    dates = pd.date_range("1990-01-01", periods=15, freq="D")
    values = pd.DataFrame({
        "canonical_station_id": "c:1", "date": dates, "variable": "tasmax",
    })
    out = qc.coverage_summary(values)
    assert len(out) == 1
    assert out.iloc[0]["observed_days"] == 15
    assert out.iloc[0]["days_in_month"] == 31
    assert out.iloc[0]["missing_fraction"] == pytest.approx(1 - 15 / 31, abs=1e-4)


def test_impossible_values_are_dropped_before_selection():
    """
    GHCN-D carries 999.0 degC at some Italian stations -- an undocumented
    sentinel the adapter's -9999 filter misses.  One such value dragged a
    station's standard deviation from 6.8 to 46.1 degC.
    """
    loader = FakeLoader({
        ("ghcnd", "G", "tasmax"): _daily([200.0, 9990.0, 210.0]),   # tenths
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ghcnd", "G")), "tasmax", loader, priority=("ghcnd",)
    )
    assert list(out.values["value"]) == pytest.approx([20.0, 21.0])
    assert out.values["value"].max() < 60.0


def test_gross_range_does_not_clip_genuine_extremes():
    """An aggressive filter would delete the extremes the model most needs."""
    loader = FakeLoader({("ecad", "E", "tasmax"): _daily([-450.0, 480.0])})
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E")), "tasmax", loader, priority=("ecad",)
    )
    assert list(out.values["value"]) == pytest.approx([-45.0, 48.0])


def test_gsod_values_reach_selection_unadjusted():
    """
    The gold table carries provider values as issued, converted to canonical
    units and nothing else. No day-window offset is applied to GSOD tas: the
    one that used to be (0.303 degC) was withdrawn as unsupported -- see
    standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION. A disagreement between
    sources is handled by refusing the splice, never by adjusting a value.
    """
    loader = FakeLoader({("gsod", "S", "tas"): _daily([68.0])})   # 68 degF = 20.0 degC
    out = consolidate_station_variable(
        "c:1", _aliases(("gsod", "S")), "tas", loader, priority=("gsod",)
    )
    assert out.values["value"].iloc[0] == pytest.approx(20.0, abs=1e-6)


def test_impossible_values_cannot_poison_a_compatibility_test():
    """
    The sentinel's real damage was statistical: it wrecked correlation and
    dispersion, so a legitimate source switch could be refused for a reason
    that was pure artefact.
    """
    clean = _daily(_seasonal(800, seed=1))
    poisoned = clean.copy()
    poisoned.iloc[100] = 9990.0
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): clean,
        ("ghcnd", "G", "tasmax"): poisoned,
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    assert out.values["value"].max() < 60.0


# ---------------------------------------------------------------------------
# Non-selected candidates audit (plan section 1.6 point 5)
# ---------------------------------------------------------------------------

def test_outranked_candidate_is_retained_in_non_selected():
    """A fallback value on a day the primary already covers is not deleted;
    it is recorded as outranked."""
    primary = _daily(_seasonal(500, seed=1), start="1990-01-01")
    fallback = _daily(_seasonal(500, seed=2), start="1990-01-01")   # identical range => fully outranked
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): primary,
        ("ghcnd", "G", "tasmax"): fallback,
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    assert list(out.non_selected.columns) == list(NON_SELECTED_COLUMNS)
    assert len(out.non_selected) == 500
    assert (out.non_selected["source"] == "ghcnd").all()
    assert out.non_selected["rejection_reason"].str.startswith("outranked").all()
    assert set(out.values["selected_source"]) == {"ecad"}   # never made it into the gold table


def test_incompatible_fallback_candidates_are_retained_in_non_selected():
    """A fallback tested and refused on a genuine gap keeps its values in
    the audit table, tagged with the same reason check_compatibility gave."""
    primary = _daily(_seasonal(400, seed=1), start="1990-01-01")
    fallback = _daily(_seasonal(800, offset=5.0, seed=1), start="1990-01-01")
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): primary,
        ("ghcnd", "G", "tasmax"): fallback,
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    assert len(out.non_selected) == 800   # 400 outranked + 400 incompatible-gap
    outranked = out.non_selected[out.non_selected["rejection_reason"].str.startswith("outranked")]
    incompatible = out.non_selected[out.non_selected["rejection_reason"].str.startswith("incompatible")]
    assert len(outranked) == 400
    assert len(incompatible) == 400
    assert "bias" in incompatible["rejection_reason"].iloc[0]


def test_primary_source_never_appears_in_non_selected():
    loader = FakeLoader({("ecad", "E", "tasmax"): _daily(_seasonal(500, seed=1))})
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E")), "tasmax", loader, priority=("ecad",)
    )
    assert out.non_selected.empty


def test_no_candidates_yields_an_empty_non_selected_frame():
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E")), "hurs", FakeLoader(), priority=("ecad",)
    )
    assert out.non_selected.empty
    assert list(out.non_selected.columns) == list(NON_SELECTED_COLUMNS)


def test_non_selected_values_are_in_canonical_units():
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): _daily([200.0, 210.0]),
        ("ghcnd", "G", "tasmax"): _daily([200.0, 210.0]),   # tenths, fully outranked
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    assert list(out.non_selected["value"]) == pytest.approx([20.0, 21.0])
    assert set(out.non_selected["units"]) == {"degC"}


def test_selected_fallback_values_do_not_also_appear_as_non_selected():
    """A day that WAS successfully filled by a compatible fallback must not
    be double-booked in the audit table."""
    primary = _daily(_seasonal(400, seed=1), start="1990-01-01")
    fallback = _daily(_seasonal(800, seed=1), start="1990-01-01")   # compatible, fills the 400-day gap
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): primary,
        ("ghcnd", "G", "tasmax"): fallback,
    })
    out = consolidate_station_variable(
        "c:1", _aliases(("ecad", "E"), ("ghcnd", "G")), "tasmax", loader,
        priority=("ecad", "ghcnd"),
    )
    selected_ghcnd_dates = set(out.values.loc[out.values["is_fallback"], "date"])
    non_selected_ghcnd_dates = set(out.non_selected["date"])
    assert selected_ghcnd_dates.isdisjoint(non_selected_ghcnd_dates)
