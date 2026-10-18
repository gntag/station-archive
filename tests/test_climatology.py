"""
Tests for the climatological plausibility screen.

What is under test is the contract the review relies on: a record's departure
from an independent reference is measured on matched days only, scored against
the spread of every other record of the same variable, sorted into the two
tiers the reviewer was promised, and given the neighbour context that tells a
broken record from a site the reference cannot resolve. Nothing here removes a
value; exclusions only happen in consolidation, and only on a reviewer's 'yes'.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from station_archive import climatology
from station_archive.climatology import (
    TIER_EXCLUDE,
    TIER_OK,
    TIER_REVIEW,
    Reference,
    add_neighbor_context,
    departure,
    exclusions_from_decisions,
    load_exclusion_decisions,
    reference_from_wide,
    reference_keys,
    run_screen,
    score,
    screen_series,
)
from station_archive.policy import DEFAULT_POLICY, ClimatologyPolicy, Policy


def _daily(values, start="1990-01-01"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="D"), dtype="float64"
    )


# ---------------------------------------------------------------------------
# departure
# ---------------------------------------------------------------------------

def test_wind_departure_is_a_log_ratio():
    assert departure(5.0, 10.0, "gust") == pytest.approx(math.log(0.5))


def test_precipitation_that_is_always_zero_departs_infinitely():
    """An all-zero series against a wet reference is the strongest possible flag."""
    assert departure(0.0, 2.0, "pr") == -math.inf


def test_multiplicative_departure_is_undefined_without_a_positive_reference():
    assert math.isnan(departure(1.0, 0.0, "pr"))


def test_temperature_departure_is_in_degrees_and_lapse_adjusted():
    # Station 1000 m above the reference point: 6.5 degC colder is expected,
    # so a station exactly that much colder departs by zero.
    d = departure(3.5, 10.0, "tasmax", station_elevation_m=1000.0,
                  reference_elevation_m=0.0)
    assert d == pytest.approx(0.0)


def test_temperature_departure_without_elevations_is_unadjusted():
    assert departure(12.0, 10.0, "tas") == pytest.approx(2.0)


def test_humidity_is_never_lapse_adjusted():
    d = departure(60.0, 70.0, "hurs", station_elevation_m=1000.0, reference_elevation_m=0.0)
    assert d == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# screen_series
# ---------------------------------------------------------------------------

def test_only_matched_days_enter_the_means():
    """
    The reference is compared on the record's own days. GSOD reports a gust
    only on gusty days, so comparing its mean with an all-days reference mean
    would manufacture a departure out of sampling alone.
    """
    reference = _daily([10.0] * 400 + [100.0] * 400)
    station = _daily([10.0] * 400)
    row = screen_series("gsod:1", "gust", station, reference, days_reporting_any=800)
    assert row["status"] == "tested"
    assert row["n_matched_days"] == 400
    assert row["reference_mean"] == pytest.approx(10.0)
    assert row["departure"] == pytest.approx(0.0)


def test_too_few_matched_days_is_a_status_not_a_departure():
    row = screen_series("ecad:1", "tas", _daily([1.0] * 100), _daily([1.0] * 100),
                        days_reporting_any=100)
    assert row["status"] == "too_few_matched_days"
    assert math.isnan(row["departure"])


def test_missing_reference_is_reported():
    row = screen_series("ecad:1", "tas", _daily([1.0] * 400), pd.Series(dtype="float64"),
                        days_reporting_any=400)
    assert row["status"] == "no_reference"


def test_reporting_fraction_and_the_gsod_gust_note():
    row = screen_series("gsod:1", "gust", _daily([12.0] * 400), _daily([10.0] * 4000),
                        days_reporting_any=4000)
    assert row["reporting_fraction"] == pytest.approx(0.1)
    assert "gusty days" in row["sampling_note"]


def test_no_sampling_note_for_an_every_day_record():
    row = screen_series("ecad:1", "gust", _daily([12.0] * 400), _daily([10.0] * 400),
                        days_reporting_any=400)
    assert row["sampling_note"] == ""


def test_nan_values_do_not_count_as_matched_days():
    station = _daily([np.nan] * 100 + [5.0] * 400)
    reference = _daily([5.0] * 500)
    row = screen_series("ecad:1", "tas", station, reference, days_reporting_any=500)
    assert row["n_matched_days"] == 400


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _screen_frame(departures, variable="tas", status="tested"):
    return pd.DataFrame(
        {
            "alias": [f"ecad:{i}" for i in range(len(departures))],
            "variable": variable,
            "status": status,
            "departure": departures,
        }
    )


def test_robust_z_and_tiers_follow_the_agreed_thresholds():
    # Median 0, MAD 1 -> robust SD 1.4826.
    base = [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    sd = 1.4826
    frame = _screen_frame(base + [3.5 * sd, 4.5 * sd, -5.0 * sd])
    out = score(frame)
    z = dict(zip(out["alias"], out["robust_z"]))
    tiers = dict(zip(out["alias"], out["tier"]))
    assert z["ecad:7"] == pytest.approx(3.5, rel=1e-3)
    assert tiers["ecad:7"] == TIER_REVIEW
    assert tiers["ecad:8"] == TIER_EXCLUDE
    assert tiers["ecad:9"] == TIER_EXCLUDE
    assert tiers["ecad:2"] == TIER_OK


def test_exactly_four_is_review_not_exclusion():
    """'More than 4 robust SDs' is the exclusion rule; 4 itself is review."""
    sd = 1.4826
    frame = _screen_frame([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0 * sd])
    out = score(frame)
    assert out.loc[out["alias"] == "ecad:7", "tier"].item() == TIER_REVIEW


def test_infinite_departure_is_proposed_for_exclusion_without_distorting_the_spread():
    frame = _screen_frame([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -math.inf], variable="pr")
    out = score(frame)
    assert out.loc[out["alias"] == "ecad:7", "tier"].item() == TIER_EXCLUDE
    # the spread is computed from the finite departures alone
    assert out.loc[out["alias"] == "ecad:6", "robust_z"].item() == pytest.approx(1 / 1.4826)


def test_variables_are_scored_separately():
    tas = _screen_frame([0.0] * 6 + [10.0], variable="tas")
    gust = _screen_frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], variable="gust")
    gust["alias"] = [f"gsod:{i}" for i in range(7)]
    out = score(pd.concat([tas, gust], ignore_index=True))
    assert (out.loc[out["variable"] == "gust", "tier"] == TIER_OK).all()


def test_untested_rows_get_no_tier():
    frame = _screen_frame([0.0, 1.0, -1.0, 0.0, 0.5, -0.5])
    frame.loc[0, "status"] = "too_few_matched_days"
    frame.loc[0, "departure"] = np.nan
    out = score(frame)
    assert out.loc[0, "tier"] == ""
    assert math.isnan(out.loc[0, "robust_z"])


def test_no_spread_scores_nothing():
    out = score(_screen_frame([1.0] * 6))
    assert (out["tier"] == TIER_OK).all()
    assert out["robust_z"].isna().all()


def test_thresholds_come_from_the_policy():
    sd = 1.4826
    frame = _screen_frame([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.5 * sd])
    strict = ClimatologyPolicy(review_z=2.0, exclude_z=3.0)
    out = score(frame, strict)
    assert out.loc[out["alias"] == "ecad:7", "tier"].item() == TIER_REVIEW


# ---------------------------------------------------------------------------
# add_neighbor_context
# ---------------------------------------------------------------------------

def _context_frame():
    return pd.DataFrame(
        [
            # a cluster of three, all low, inside 100 km of each other
            {"alias": "ecad:a", "canonical_station_id": "ecad:a", "variable": "tasmax",
             "lat": 41.80, "lon": 13.10, "status": "tested", "departure": 5.0,
             "robust_z": 4.5, "tier": TIER_EXCLUDE},
            {"alias": "ecad:b", "canonical_station_id": "ecad:b", "variable": "tasmax",
             "lat": 41.85, "lon": 13.20, "status": "tested", "departure": 5.5,
             "robust_z": 5.0, "tier": TIER_EXCLUDE},
            {"alias": "ecad:c", "canonical_station_id": "ecad:c", "variable": "tasmax",
             "lat": 41.70, "lon": 13.00, "status": "tested", "departure": 0.1,
             "robust_z": 0.1, "tier": TIER_OK},
            # same physical station as ecad:a: never its own neighbour
            {"alias": "gsod:a", "canonical_station_id": "ecad:a", "variable": "tasmax",
             "lat": 41.80, "lon": 13.10, "status": "tested", "departure": 5.0,
             "robust_z": 4.5, "tier": TIER_EXCLUDE},
            # far away
            {"alias": "ecad:far", "canonical_station_id": "ecad:far", "variable": "tasmax",
             "lat": 35.0, "lon": 25.0, "status": "tested", "departure": 0.0,
             "robust_z": 0.0, "tier": TIER_OK},
            # other variable at the same place
            {"alias": "ecad:w", "canonical_station_id": "ecad:w", "variable": "gust",
             "lat": 41.80, "lon": 13.10, "status": "tested", "departure": 0.0,
             "robust_z": 0.0, "tier": TIER_OK},
        ]
    )


def test_neighbours_are_other_stations_of_the_same_variable_within_the_radius():
    out = add_neighbor_context(_context_frame()).set_index("alias")
    assert out.loc["ecad:a", "n_neighbors"] == 2               # b and c, not gsod:a
    assert out.loc["ecad:a", "neighbor_median_departure"] == pytest.approx((5.5 + 0.1) / 2)
    assert out.loc["ecad:far", "n_neighbors"] == 0
    assert math.isnan(out.loc["ecad:far", "neighbor_median_departure"])


def test_same_direction_count_reveals_a_network_offset():
    out = add_neighbor_context(_context_frame()).set_index("alias")
    assert out.loc["ecad:a", "n_neighbors_same_direction"] == 1     # ecad:b
    assert out.loc["ecad:c", "n_neighbors_same_direction"] == 0     # c itself is ok


def test_departure_versus_neighbours():
    out = add_neighbor_context(_context_frame()).set_index("alias")
    assert out.loc["ecad:c", "departure_vs_neighbors"] == pytest.approx(0.1 - 5.25)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def _wide():
    dates = pd.date_range("1990-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "id": ["ecad:1"] * 3 + ["ecad:2"] * 3,
            "date": list(dates) * 2,
            "cerra_tmax": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cerra_gust": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )


def test_reference_from_wide_serves_date_indexed_series():
    ref = reference_from_wide(
        _wide(), id_column="id", date_column="date",
        columns={"tasmax": "cerra_tmax", "gust": "cerra_gust"},
        elevation={"ecad:1": 100.0},
    )
    assert isinstance(ref, Reference)
    s = ref.series("ecad:2", "gust")
    assert list(s.to_numpy()) == [10.0, 11.0, 12.0]
    assert isinstance(s.index, pd.DatetimeIndex)
    assert ref.series("ecad:9", "gust").empty
    assert ref.series("ecad:1", "pr").empty
    assert ref.elevation("ecad:1") == 100.0
    assert math.isnan(ref.elevation("ecad:2"))


def test_reference_keys_prefer_the_record_itself_then_its_group():
    aliases = pd.DataFrame(
        {
            "canonical_station_id": ["ecad:1", "ecad:1", "ecad:5"],
            "source": ["ecad", "gsod", "ecad"],
            "source_station_id": ["1", "16000099999", "5"],
        }
    )
    keys = reference_keys(aliases, {"ecad:1"})
    assert keys["ecad:1"] == "ecad:1"
    assert keys["gsod:16000099999"] == "ecad:1"
    assert "ecad:5" not in keys


def test_reference_keys_use_any_alias_of_the_group_when_the_canonical_id_moved():
    """A merge can change the group's canonical id; any member's key still resolves."""
    aliases = pd.DataFrame(
        {
            "canonical_station_id": ["c3s:A", "c3s:A"],
            "source": ["ecad", "gsod"],
            "source_station_id": ["1", "2"],
        }
    )
    keys = reference_keys(aliases, {"gsod:2"})
    assert keys == {"ecad:1": "gsod:2", "gsod:2": "gsod:2"}


# ---------------------------------------------------------------------------
# run_screen, end to end on fake bronze
# ---------------------------------------------------------------------------

class FakeLoader:
    """Serves prepared series in PROVIDER units, as bronze holds them."""

    def __init__(self, series: dict) -> None:
        self._series = series
        self.cleared = 0

    def series(self, source, station_id, variable):
        return self._series.get((source, station_id, variable), pd.Series(dtype="float64"))

    def clear(self):
        self.cleared += 1


def test_run_screen_end_to_end_flags_the_broken_record_only():
    n_days = 500
    dates = pd.date_range("1990-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    stations = [f"{i:06d}" for i in range(12)]
    wide_rows = []
    bronze_series = {}
    for i, sid in enumerate(stations):
        wet = rng.gamma(0.5, 4.0, n_days)
        for d, v in zip(dates, wet):
            wide_rows.append({"id": f"ecad:{sid}", "date": d, "cerra_precip": v})
        # ECA&D pr is 0.1 mm; a modest station-to-station spread around the reference
        factor = 1.0 + 0.05 * (i - 6) / 6
        raw = wet * factor * 10.0
        if sid == "000003":
            raw = np.zeros(n_days)              # the always-zero series
        bronze_series[("ecad", sid, "pr")] = pd.Series(raw, index=dates)
    reference = reference_from_wide(
        pd.DataFrame(wide_rows), id_column="id", date_column="date",
        columns={"pr": "cerra_precip"},
    )
    aliases = pd.DataFrame(
        {
            "canonical_station_id": [f"ecad:{s}" for s in stations],
            "source": "ecad",
            "source_station_id": stations,
            "lat": [38.0 + 0.1 * i for i in range(12)],
            "lon": 23.0,
            "elevation_m": 100.0,
            "name": [f"S{i}" for i in range(12)],
        }
    )
    policy = Policy.from_dict({"variables": list(DEFAULT_POLICY.variables)})
    screen = run_screen(aliases, FakeLoader(bronze_series), reference, policy=policy)
    pr = screen[screen["variable"] == "pr"].set_index("alias")
    assert pr.loc["ecad:000003", "tier"] == TIER_EXCLUDE
    assert (pr.drop(index="ecad:000003")["tier"] == TIER_OK).all()
    assert set(climatology.SCREEN_COLUMNS) <= set(screen.columns)
    # the screen reports, it never changes the input
    assert (bronze_series[("ecad", "000003", "pr")] == 0).all()


def test_run_screen_skips_sources_outside_the_priority_list():
    """c3s_land is audit-only: excluding it would change nothing, so it is not screened."""
    dates = pd.date_range("1990-01-01", periods=400, freq="D")
    reference = reference_from_wide(
        pd.DataFrame({"id": "c3s_land:X", "date": dates, "cerra_tmean": 280.0 - 273.15}),
        id_column="id", date_column="date", columns={"tas": "cerra_tmean"},
    )
    aliases = pd.DataFrame(
        {"canonical_station_id": ["c3s_land:X"], "source": ["c3s_land"],
         "source_station_id": ["X"], "lat": [40.0], "lon": [20.0],
         "elevation_m": [10.0], "name": ["X"]}
    )
    loader = FakeLoader({("c3s_land", "X", "tas"): pd.Series(280.0, index=dates)})
    screen = run_screen(aliases, loader, reference)
    assert screen.empty


# ---------------------------------------------------------------------------
# Reading the reviewer's answers back
# ---------------------------------------------------------------------------

def _workbook(tmp_path, series_rows, ledger_rows=None):
    path = tmp_path / "review.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(series_rows).to_excel(writer, sheet_name="Series", index=False)
        if ledger_rows is not None:
            pd.DataFrame(ledger_rows).to_excel(writer, sheet_name="Series_decisions", index=False)
    return path


def test_only_yes_and_no_are_decisions(tmp_path):
    path = _workbook(tmp_path, [
        {"alias": "ecad:1", "variable": "pr", "DECISION_exclude": "Yes",
         "DECISION_note": "all zeros"},
        {"alias": "ecad:2", "variable": "pr", "DECISION_exclude": "no", "DECISION_note": ""},
        {"alias": "ecad:3", "variable": "pr", "DECISION_exclude": "maybe", "DECISION_note": ""},
        {"alias": "ecad:4", "variable": "pr", "DECISION_exclude": None, "DECISION_note": ""},
    ])
    decisions = load_exclusion_decisions(path)
    assert decisions[("ecad:1", "pr")] == ("yes", "all zeros")
    assert decisions[("ecad:2", "pr")] == ("no", "")
    assert ("ecad:3", "pr") not in decisions
    assert ("ecad:4", "pr") not in decisions


def test_ledger_decisions_are_read_when_the_row_is_no_longer_offered(tmp_path):
    path = _workbook(
        tmp_path,
        [{"alias": "ecad:1", "variable": "pr", "DECISION_exclude": "", "DECISION_note": ""}],
        [{"alias": "ecad:9", "variable": "tasmax", "DECISION_exclude": "yes",
          "DECISION_note": "Lazio block", "still_offered": False}],
    )
    assert load_exclusion_decisions(path)[("ecad:9", "tasmax")] == ("yes", "Lazio block")


def test_the_offered_sheet_wins_over_the_ledger(tmp_path):
    path = _workbook(
        tmp_path,
        [{"alias": "ecad:1", "variable": "pr", "DECISION_exclude": "no", "DECISION_note": ""}],
        [{"alias": "ecad:1", "variable": "pr", "DECISION_exclude": "yes",
          "DECISION_note": "", "still_offered": True}],
    )
    assert load_exclusion_decisions(path)[("ecad:1", "pr")][0] == "no"


def test_a_workbook_without_series_sheets_has_no_decisions(tmp_path):
    path = tmp_path / "old.xlsx"
    pd.DataFrame({"left": ["a"], "right": ["b"], "DECISION_merge": ["yes"]}).to_excel(
        path, sheet_name="Review", index=False
    )
    assert load_exclusion_decisions(path) == {}


def test_exclusions_are_the_yes_answers_with_their_reason():
    out = exclusions_from_decisions({
        ("ecad:1", "pr"): ("yes", "all zeros"),
        ("ecad:2", "pr"): ("no", ""),
        ("ecad:3", "gust"): ("yes", ""),
    })
    assert out == {
        ("ecad:1", "pr"): "all zeros",
        ("ecad:3", "gust"): "excluded by reviewer",
    }


def test_default_climatology_policy_is_the_agreed_rule():
    c = DEFAULT_POLICY.climatology
    assert (c.review_z, c.exclude_z) == (3.0, 4.0)
    assert c.min_matched_days == 365
    assert c.multiplicative_variables == frozenset({"pr", "sfcWind", "gust"})
    assert c.lapse_variables == frozenset({"tas", "tasmin", "tasmax"})
    assert c.lapse_rate_k_per_m == 0.0065
    assert c.neighbor_radius_km == 100.0


def test_climatology_policy_round_trips_through_a_policy_file():
    p = Policy.from_dict({"climatology": {"exclude_z": 5.0}})
    assert p.climatology.exclude_z == 5.0
    assert Policy.from_dict(p.to_dict()).climatology == p.climatology


def test_review_threshold_must_sit_below_the_exclusion_threshold():
    with pytest.raises(ValueError):
        ClimatologyPolicy(review_z=4.0, exclude_z=3.0)
