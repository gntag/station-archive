"""
Tests for the canonical station crosswalk (plan section 1.3) and the unit
primitives it depends on (section 1.4).

The behaviours under test are the ones section 1.3 actually mandates: exact
identifiers outrank geometry, coordinate proximity alone never merges,
unresolved candidates stay separate with a risk flag, duplicates are never
averaged, and the result is deterministic.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from observations import config, identity, standardize
from observations.identity import (
    MatchPair,
    OverlapStats,
    StationRecord,
    TIER_COINCIDENT,
    TIER_IDENTIFIER,
    TIER_NEAR,
    TIER_UNRESOLVED,
    build_crosswalk,
    candidate_pairs,
    evaluate_pair,
    haversine_km,
    match_tier1_identifier,
    name_similarity,
    normalize_name,
    normalized_wmo,
    overlap_statistics,
    records_from_inventories,
)
from observations.types import INVENTORY_COLUMNS, StationInventory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _station(source, sid, lat, lon, elev=100.0, name="ATHENS", wmo="", variables=("tasmax",)):
    return StationRecord(
        source=source,
        source_station_id=sid,
        name=name,
        lat=lat,
        lon=lon,
        elevation_m=elev,
        wmo_id=wmo,
        wigos_id="",
        variables=frozenset(variables),
        first_date="1985-01-01",
        last_date="2005-12-31",
    )


class FakeLoader:
    """Serves prepared daily series without touching the archive."""

    def __init__(self, series: dict | None = None) -> None:
        self._series = series or {}

    def series(self, source, station_id, variable):
        return self._series.get((source, station_id, variable), pd.Series(dtype="float64"))


def _daily(values, start="1985-01-01"):
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype="float64")


# ---------------------------------------------------------------------------
# Unit primitives
# ---------------------------------------------------------------------------

def test_fahrenheit_conversion_matches_known_points():
    assert standardize.fahrenheit_to_celsius(32.0) == pytest.approx(0.0)
    assert standardize.fahrenheit_to_celsius(212.0) == pytest.approx(100.0)


def test_kelvin_conversion():
    assert standardize.kelvin_to_celsius(273.15) == pytest.approx(0.0)


def test_tenths_scaling():
    """GHCN-D TMAX 142 means 14.2 degC, not 142."""
    assert standardize.tenths(142.0) == pytest.approx(14.2)


def test_inch_and_knot_conversions():
    assert standardize.inch_to_mm(1.0) == pytest.approx(25.4)
    assert standardize.knot_to_m_s(1.0) == pytest.approx(0.514444, abs=1e-6)


def test_unknown_unit_raises_rather_than_passing_through():
    """Guessing an unrecognised unit is canonical would corrupt every value."""
    with pytest.raises(standardize.UnknownUnitError):
        standardize.to_canonical(1.0, "furlongs_per_fortnight")


def test_to_canonical_dispatches_by_provider_unit():
    assert standardize.to_canonical(142.0, "degC_tenths") == pytest.approx(14.2)
    assert standardize.to_canonical(77.0, "degF") == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Geometry and names
# ---------------------------------------------------------------------------

def test_haversine_zero_for_identical_points():
    assert haversine_km(37.9, 23.7, 37.9, 23.7) == pytest.approx(0.0)


def test_haversine_one_degree_latitude_is_about_111_km():
    assert haversine_km(37.0, 23.0, 38.0, 23.0) == pytest.approx(111.2, abs=0.5)


def test_haversine_longitude_is_shorter_at_latitude():
    """
    The reason the gate is distance-based, not a degree box: at 38N a degree
    of longitude is far shorter than a degree of latitude.
    """
    lat_km = haversine_km(38.0, 23.0, 39.0, 23.0)
    lon_km = haversine_km(38.0, 23.0, 38.0, 24.0)
    assert lon_km < lat_km * 0.85


def test_normalize_name_strips_punctuation_and_country_suffix():
    assert normalize_name("PIACENZA, IT") == "PIACENZA IT"
    assert normalize_name("ATHENS-HELLINIKON") == "ATHENS HELLINIKON"


def test_name_similarity_rewards_shared_tokens():
    assert name_similarity("ATHENS", "ATHENS") == pytest.approx(1.0)
    assert 0.0 < name_similarity("ATHENS", "ATHENS AIRPORT") < 1.0
    assert name_similarity("ATHENS", "HERAKLION") == 0.0


def test_name_similarity_zero_when_a_name_is_missing():
    assert name_similarity("", "ATHENS") == 0.0


# ---------------------------------------------------------------------------
# Tier 1 -- exact identifiers
# ---------------------------------------------------------------------------

def test_normalized_wmo_strips_zero_padding():
    """GSOD zero-pads its WMO number where GHCN-D does not."""
    assert normalized_wmo("016716") == normalized_wmo("16716")


def test_normalized_wmo_empty_for_absent_values():
    assert normalized_wmo("") == ""
    assert normalized_wmo("nan") == ""


def test_c3s_links_to_ghcnd_by_shared_identifier():
    """
    C3S primary_station_id is a GHCN-D identifier verbatim -- the 'documented
    cross-source identifier' of tier 1.
    """
    records = [
        _station("ghcnd", "ALE00100939", 41.3, 19.8),
        _station("c3s_land", "ALE00100939", 41.3, 19.8),
    ]
    pairs = match_tier1_identifier(records)
    assert len(pairs) == 1
    assert pairs[0].tier == TIER_IDENTIFIER
    assert pairs[0].accepted
    assert "ghcnd_id" in pairs[0].reason


def test_ghcnd_and_gsod_link_by_wmo_number():
    records = [
        _station("ghcnd", "GRE00105241", 37.9, 23.7, wmo="16716"),
        _station("gsod", "16716099999", 37.93, 23.73, wmo="16716"),
    ]
    pairs = match_tier1_identifier(records)
    assert len(pairs) == 1
    assert "wmo" in pairs[0].reason


def test_tier1_ignores_same_source_identifier_collisions():
    """Two stations from one source sharing an ID is that source's business."""
    records = [
        _station("ghcnd", "A1", 37.9, 23.7, wmo="16716"),
        _station("ghcnd", "A2", 37.9, 23.7, wmo="16716"),
    ]
    assert match_tier1_identifier(records) == []


def test_tier1_ignores_blank_identifiers():
    """ECA&D publishes no WMO number; blanks must not all match each other."""
    records = [
        _station("ecad", "000247", 37.9, 23.7, wmo=""),
        _station("gsod", "16716099999", 37.9, 23.7, wmo=""),
    ]
    assert match_tier1_identifier(records) == []


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def test_candidates_only_pair_across_sources():
    records = [
        _station("ecad", "1", 37.900, 23.700),
        _station("ecad", "2", 37.901, 23.701),
        _station("ghcnd", "G", 37.900, 23.700),
    ]
    pairs = candidate_pairs(records, config.IDENTITY_NEAR_DEG)
    assert all(a.source != b.source for a, b in pairs)


def test_candidates_find_neighbours_across_bucket_boundaries():
    """
    Two stations metres apart but on opposite sides of a bucket edge must
    still be compared, or the blocking would silently lose real duplicates.
    """
    tol = config.IDENTITY_NEAR_DEG
    edge = math.floor(37.9 / tol) * tol
    records = [
        _station("ecad", "1", edge - tol / 1000, 23.7),
        _station("ghcnd", "G", edge + tol / 1000, 23.7),
    ]
    assert len(candidate_pairs(records, tol)) == 1


def test_distant_stations_are_not_candidates():
    records = [
        _station("ecad", "1", 37.9, 23.7),
        _station("ghcnd", "G", 35.3, 25.1),
    ]
    assert candidate_pairs(records, config.IDENTITY_NEAR_DEG) == []


def test_candidate_generation_is_deterministic():
    records = [
        _station("ecad", "1", 37.900, 23.700),
        _station("ghcnd", "G", 37.901, 23.701),
        _station("gsod", "S", 37.902, 23.702),
    ]
    first = [(a.key, b.key) for a, b in candidate_pairs(records, config.IDENTITY_NEAR_DEG)]
    second = [(a.key, b.key) for a, b in candidate_pairs(records, config.IDENTITY_NEAR_DEG)]
    assert first == second


def test_candidates_skip_stations_without_coordinates():
    records = [
        _station("ecad", "1", float("nan"), float("nan")),
        _station("ghcnd", "G", 37.9, 23.7),
    ]
    assert candidate_pairs(records, config.IDENTITY_NEAR_DEG) == []


# ---------------------------------------------------------------------------
# Overlap statistics
# ---------------------------------------------------------------------------

def test_overlap_converts_units_before_comparing():
    """
    The same observations in Fahrenheit and in tenths of a degree must show
    zero bias, not a 30-degree offset.
    """
    celsius = np.array([10.0, 15.0, 20.0, 25.0])
    left = _daily(celsius * 10)                       # degC_tenths
    right = _daily(celsius * 9 / 5 + 32)              # degF
    stats = overlap_statistics(left, right, "tasmax", "degC_tenths", "degF")
    assert stats.n_overlap == 4
    assert stats.bias == pytest.approx(0.0, abs=1e-9)
    assert stats.exact_fraction == pytest.approx(1.0)


def test_overlap_uses_only_shared_days():
    left = _daily([1.0, 2.0, 3.0], start="1985-01-01")
    right = _daily([2.0, 3.0, 4.0], start="1985-01-02")
    stats = overlap_statistics(left, right, "tasmax", "degC", "degC")
    assert stats.n_overlap == 2


def test_overlap_empty_when_no_shared_days():
    left = _daily([1.0, 2.0], start="1985-01-01")
    right = _daily([1.0, 2.0], start="1990-01-01")
    stats = overlap_statistics(left, right, "tasmax", "degC", "degC")
    assert stats.n_overlap == 0
    assert not stats.usable


def test_overlap_correlation_is_nan_for_constant_series():
    """A flat series has no defined correlation; NaN, not a spurious value."""
    left = _daily([5.0] * 400)
    right = _daily([5.0] * 400)
    stats = overlap_statistics(left, right, "tasmax", "degC", "degC")
    assert math.isnan(stats.correlation)
    assert not stats.usable


def test_overlap_detects_bias_between_sources():
    left = _daily([10.0, 12.0, 14.0])
    right = _daily([11.0, 13.0, 15.0])
    stats = overlap_statistics(left, right, "tasmax", "degC", "degC")
    assert stats.bias == pytest.approx(-1.0)
    assert stats.mean_abs_diff == pytest.approx(1.0)
    assert stats.exact_fraction == pytest.approx(0.0)


def test_overlap_stats_unusable_below_minimum_days():
    left = _daily([float(i % 20) for i in range(10)])
    right = _daily([float(i % 20) for i in range(10)])
    stats = overlap_statistics(left, right, "tasmax", "degC", "degC")
    assert stats.n_overlap == 10
    assert not stats.usable


# ---------------------------------------------------------------------------
# The merge gate
# ---------------------------------------------------------------------------

def _correlated_pair(n=800, offset=0.0, noise=0.0):
    """
    Two well-correlated but NOT identical daily series, in provider units.

    Values are returned in tenths of a degree because the sources used in
    these tests declare 'degC_tenths'; emitting degC here would let the
    conversion shrink the noise tenfold and make merely-correlated series
    register as identical, which is the distinction several tests turn on.
    """
    rng = np.random.default_rng(0)
    base = 20 + 10 * np.sin(np.arange(n) * 2 * math.pi / 365.25)
    left = base + rng.normal(0, 0.3, n)
    right = base + offset + rng.normal(0, 0.3 + noise, n)
    return _daily(left * 10), _daily(right * 10)


def test_pair_merges_when_all_gates_pass():
    left, right = _correlated_pair()
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): left, ("ghcnd", "G", "tasmax"): right}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=100.0)
    b = _station("ghcnd", "G", 37.9005, 23.7005, elev=105.0)
    pair = evaluate_pair(a, b, loader)
    assert pair.accepted
    assert pair.tier in (TIER_COINCIDENT, TIER_NEAR)


def test_proximity_alone_never_merges():
    """
    Section 1.3: "Coordinate tolerance alone is insufficient because airports
    and nearby climatological stations can be distinct."  Co-located stations
    with uncorrelated series must stay separate.
    """
    rng = np.random.default_rng(1)
    loader = FakeLoader(
        {
            ("ecad", "1", "tasmax"): _daily(rng.normal(20, 5, 800)),
            ("ghcnd", "G", "tasmax"): _daily(rng.normal(20, 5, 800)),
        }
    )
    a = _station("ecad", "1", 37.9000, 23.7000)
    b = _station("ghcnd", "G", 37.9001, 23.7001)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert "correlation" in pair.reason


def test_distance_gate_rejects_beyond_tolerance():
    a = _station("ecad", "1", 37.9, 23.7)
    b = _station("ghcnd", "G", 38.5, 23.7)
    pair = evaluate_pair(a, b, FakeLoader())
    assert not pair.accepted
    assert pair.tier == TIER_UNRESOLVED
    assert "distance" in pair.reason


def test_elevation_gate_rejects_incompatible_height():
    """A valley station and a hilltop station can be 300 m apart horizontally."""
    left, right = _correlated_pair()
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): left, ("ghcnd", "G", "tasmax"): right}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=50.0)
    b = _station("ghcnd", "G", 37.9005, 23.7005, elev=900.0)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert "elevation" in pair.reason


def test_missing_elevation_blocks_the_merge():
    """Absent evidence is not evidence of compatibility."""
    left, right = _correlated_pair()
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): left, ("ghcnd", "G", "tasmax"): right}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=float("nan"))
    b = _station("ghcnd", "G", 37.9005, 23.7005, elev=105.0)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert "elevation missing" in pair.reason


def test_insufficient_overlap_blocks_the_merge():
    """r=1.0 over five days is not evidence."""
    short = _daily([10.0, 11.0, 12.0, 13.0, 14.0])
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): short, ("ghcnd", "G", "tasmax"): short}
    )
    a = _station("ecad", "1", 37.9000, 23.7000)
    b = _station("ghcnd", "G", 37.9005, 23.7005)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert "overlapping days" in pair.reason


def test_rejected_pair_still_records_its_statistics():
    """The evidence table must explain every non-merge, not just record it."""
    rng = np.random.default_rng(2)
    loader = FakeLoader(
        {
            ("ecad", "1", "tasmax"): _daily(rng.normal(20, 5, 800)),
            ("ghcnd", "G", "tasmax"): _daily(rng.normal(20, 5, 800)),
        }
    )
    a = _station("ecad", "1", 37.9000, 23.7000)
    b = _station("ghcnd", "G", 37.9001, 23.7001)
    pair = evaluate_pair(a, b, loader)
    assert pair.stats is not None
    assert pair.stats.n_overlap == 800
    assert "correlation" in pair.to_dict()


# ---------------------------------------------------------------------------
# Crosswalk assembly
# ---------------------------------------------------------------------------

def _inventory(rows):
    records = []
    for row in rows:
        base = {c: "" for c in INVENTORY_COLUMNS}
        base.update(row)
        records.append(base)
    return StationInventory.from_records(records)


def test_crosswalk_gives_every_station_a_canonical_id():
    inventories = {
        "ecad": _inventory(
            [dict(source="ecad", source_station_id="000247", name="ATHENS",
                  lat=37.9, lon=23.7, elevation_m=107, variable="tasmax")]
        ),
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="GRE00105241", name="HERAKLION",
                  lat=35.3, lon=25.1, elevation_m=39, variable="tasmax")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    assert crosswalk.n_aliases() == 2
    assert crosswalk.n_canonical() == 2
    assert crosswalk.aliases["canonical_station_id"].notna().all()


def test_crosswalk_merges_tier1_identifier_matches():
    inventories = {
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tasmax")]
        ),
        "c3s_land": _inventory(
            [dict(source="c3s_land", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tas")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    assert crosswalk.n_aliases() == 2
    assert crosswalk.n_canonical() == 1
    assert set(crosswalk.aliases["match_tier"]) == {TIER_IDENTIFIER}


def test_canonical_id_is_the_smallest_member_key():
    """
    Deterministic and stable: adding unrelated stations later must not
    renumber existing canonical IDs, which fold manifests depend on.
    """
    inventories = {
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tasmax")]
        ),
        "c3s_land": _inventory(
            [dict(source="c3s_land", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tas")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    assert crosswalk.aliases["canonical_station_id"].iloc[0] == "c3s_land:ALE00100939"


def test_unmerged_station_is_not_flagged_when_nothing_was_near():
    """Isolation is not duplicate risk; only a failed nearby test is."""
    inventories = {
        "ecad": _inventory(
            [dict(source="ecad", source_station_id="000247", name="ATHENS",
                  lat=37.9, lon=23.7, elevation_m=107, variable="tasmax")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    assert not crosswalk.aliases["duplicate_risk"].any()


def test_failed_nearby_pair_raises_duplicate_risk_on_both():
    """
    Section 1.3 tier 4: "unresolved candidates remain separate and receive a
    duplicate-risk flag".  A failed test means insufficient evidence, not
    proven distinctness.
    """
    inventories = {
        "ecad": _inventory(
            [dict(source="ecad", source_station_id="000247", name="ATHENS",
                  lat=37.9000, lon=23.7000, elevation_m=107, variable="tasmax")]
        ),
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="GRE00105241", name="ATHENS AP",
                  lat=37.9005, lon=23.7005, elevation_m=110, variable="tasmax")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())     # no series -> fails
    assert crosswalk.n_canonical() == 2
    assert crosswalk.aliases["duplicate_risk"].all()


def test_metadata_history_keeps_one_segment_per_alias():
    """
    Section 1.3 forbids "silent coordinate averaging"; two aliases with
    different coordinates must remain two time-bounded segments.
    """
    inventories = {
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.330, lon=19.780, elevation_m=89, variable="tasmax",
                  first_date="1985-01-01", last_date="1995-12-31")]
        ),
        "c3s_land": _inventory(
            [dict(source="c3s_land", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.335, lon=19.785, elevation_m=91, variable="tas",
                  first_date="1996-01-01", last_date="2005-12-31")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    history = crosswalk.metadata_history
    assert len(history) == 2
    assert history["canonical_station_id"].nunique() == 1
    # Coordinates preserved per segment, not averaged into one position.
    assert sorted(history["lat"].round(3)) == [41.330, 41.335]


def test_crosswalk_is_deterministic_across_runs():
    """Section 1.8: an unchanged archive must reproduce the same result."""
    inventories = {
        "ecad": _inventory(
            [dict(source="ecad", source_station_id="000247", name="ATHENS",
                  lat=37.9, lon=23.7, elevation_m=107, variable="tasmax")]
        ),
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tasmax")]
        ),
        "c3s_land": _inventory(
            [dict(source="c3s_land", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tas")]
        ),
    }
    first = build_crosswalk(inventories, FakeLoader()).aliases
    second = build_crosswalk(inventories, FakeLoader()).aliases
    pd.testing.assert_frame_equal(first, second)


def test_evidence_records_both_accepted_and_rejected_pairs():
    inventories = {
        "ecad": _inventory(
            [dict(source="ecad", source_station_id="000247", name="ATHENS",
                  lat=37.9000, lon=23.7000, elevation_m=107, variable="tasmax")]
        ),
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="GRE00105241", name="ATHENS AP",
                  lat=37.9005, lon=23.7005, elevation_m=110, variable="tasmax")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    assert len(crosswalk.evidence) == 1
    assert not crosswalk.evidence["accepted"].iloc[0]
    assert crosswalk.evidence["reason"].iloc[0]


def test_records_from_inventories_collapses_station_variable_rows():
    inventory = _inventory(
        [
            dict(source="ecad", source_station_id="000247", name="ATHENS",
                 lat=37.9, lon=23.7, elevation_m=107, variable="tasmax"),
            dict(source="ecad", source_station_id="000247", name="ATHENS",
                 lat=37.9, lon=23.7, elevation_m=107, variable="pr"),
        ]
    )
    records = records_from_inventories({"ecad": inventory})
    assert len(records) == 1
    assert records[0].variables == frozenset({"tasmax", "pr"})


# ---------------------------------------------------------------------------
# Transitive grouping
# ---------------------------------------------------------------------------

def test_transitive_group_is_flagged_for_review():
    """
    A links to B and B to C, but A and C were never accepted directly.
    Section 1.3 gives no rule for chained links, so the group is flagged.
    """
    from observations.identity import _DisjointSet, _assemble

    records = {
        "ecad:1": _station("ecad", "1", 37.9000, 23.7000),
        "ghcnd:G": _station("ghcnd", "G", 37.9005, 23.7005),
        "gsod:S": _station("gsod", "S", 37.9010, 23.7010),
    }
    groups = _DisjointSet()
    for key in records:
        groups.add(key)
    groups.union("ecad:1", "ghcnd:G")
    groups.union("ghcnd:G", "gsod:S")

    evidence = [
        MatchPair("ecad:1", "ghcnd:G", TIER_NEAR, True, 0.07, 5.0, 1.0, None, "ok"),
        MatchPair("ghcnd:G", "gsod:S", TIER_NEAR, True, 0.07, 5.0, 1.0, None, "ok"),
    ]
    crosswalk = _assemble(records, groups, evidence)
    assert crosswalk.n_canonical() == 1
    assert crosswalk.aliases["group_formed_transitively"].all()


def test_disjoint_set_representative_is_deterministic():
    from observations.identity import _DisjointSet

    first, second = _DisjointSet(), _DisjointSet()
    first.union("b", "a")
    second.union("a", "b")
    assert first.find("a") == second.find("a") == "a"


def test_sources_without_loadable_series_skip_tier23_tests():
    """
    C3S delivers whole-year files, so it has no per-station series.  It must
    join only at tier 1 and never be rejected for "insufficient overlapping
    days", which would imply a comparison that never happened.
    """
    inventories = {
        "ecad": _inventory(
            [dict(source="ecad", source_station_id="000247", name="ATHENS",
                  lat=37.9000, lon=23.7000, elevation_m=107, variable="tasmax")]
        ),
        "c3s_land": _inventory(
            [dict(source="c3s_land", source_station_id="GRE00105241", name="ATHENS",
                  lat=37.9001, lon=23.7001, elevation_m=107, variable="tas")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    reasons = " ".join(crosswalk.evidence["reason"]) if not crosswalk.evidence.empty else ""
    assert "overlapping days" not in reasons
    assert crosswalk.n_canonical() == 2


def test_c3s_still_joins_ghcnd_at_tier1_despite_no_series():
    inventories = {
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tasmax")]
        ),
        "c3s_land": _inventory(
            [dict(source="c3s_land", source_station_id="ALE00100939", name="TIRANA",
                  lat=41.33, lon=19.78, elevation_m=89, variable="tas")]
        ),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    assert crosswalk.n_canonical() == 1
    assert set(crosswalk.aliases["match_tier"]) == {TIER_IDENTIFIER}


def test_elevation_rejection_still_records_series_evidence():
    """
    Section 1.6 wants pairwise overlap statistics in the audit.  An elevation
    conflict between two archives describing one station is exactly the case
    needing them: without the statistics the audit records a disagreement and
    hides the proof of which side is wrong.
    """
    left, right = _correlated_pair()
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): left, ("ghcnd", "G", "tasmax"): right}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=50.0)
    b = _station("ghcnd", "G", 37.9005, 23.7005, elev=900.0)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert "elevation" in pair.reason
    assert pair.stats is not None                 # evidence retained
    assert pair.stats.n_overlap > 0
    assert pair.to_dict()["correlation"] > 0.9


def test_distance_rejection_skips_series_load():
    """
    Beyond the radius nothing can merge, so the series is not worth parsing;
    the distance reason is self-explanatory without statistics.
    """
    a = _station("ecad", "1", 37.9, 23.7)
    b = _station("ghcnd", "G", 38.5, 23.7)
    pair = evaluate_pair(a, b, FakeLoader())
    assert not pair.accepted
    assert pair.stats is None


def test_accepted_merge_with_poor_value_agreement_is_flagged():
    """
    Correlation measures covariation, not agreement.  A merge can clear
    r>=0.95 while the two records differ by more than a degree every day.
    """
    from observations.identity import agreement_needs_review

    good = OverlapStats("tasmax", 800, 0.999, 0.0, 0.02, 0.99)
    poor = OverlapStats("tasmax", 800, 0.981, -1.3, 1.31, 0.03)
    assert not agreement_needs_review(good)
    assert agreement_needs_review(poor)


def test_agreement_review_threshold_is_per_unit():
    """2 mm of precipitation disagreement is not 2 degC of temperature."""
    from observations.identity import agreement_needs_review

    assert not agreement_needs_review(OverlapStats("pr", 800, 0.99, 0.0, 1.5, 0.5))
    assert agreement_needs_review(OverlapStats("tasmax", 800, 0.99, 0.0, 1.5, 0.5))


def test_identical_records_are_recognised():
    from observations.identity import records_are_identical

    assert records_are_identical(OverlapStats("tasmax", 3210, 1.0, 0.0, 0.0, 1.0))
    assert not records_are_identical(OverlapStats("tasmax", 3210, 0.99, 0.2, 0.74, 0.06))


def test_metadata_conflict_reported_when_series_identical_but_elevation_differs():
    """
    Two archives publishing one instrument under irreconcilable elevations is
    a metadata error, not two stations -- and elevation is a model predictor,
    so it must surface rather than be silently absorbed.
    """
    from observations.identity import metadata_conflicts

    stats = OverlapStats("tasmax", 7670, 1.0, 0.0, 0.0, 1.0)
    evidence = [
        MatchPair("ecad:010963", "ghcnd:HRE00155411", TIER_COINCIDENT, False,
                  0.0, 113.0, 1.0, stats, "elevation differs by 113 m"),
    ]
    conflicts = metadata_conflicts(evidence)
    assert len(conflicts) == 1
    assert "elevation" in conflicts.iloc[0]["conflict"]
    assert conflicts.iloc[0]["n_overlap_days"] == 7670


def test_no_conflict_reported_when_metadata_agrees():
    from observations.identity import metadata_conflicts

    stats = OverlapStats("tasmax", 3000, 1.0, 0.0, 0.0, 1.0)
    evidence = [
        MatchPair("ecad:1", "ghcnd:G", TIER_COINCIDENT, True,
                  0.05, 2.0, 1.0, stats, "ok"),
    ]
    assert metadata_conflicts(evidence).empty


def test_merely_similar_stations_are_not_called_a_metadata_conflict():
    """Distinct nearby stations at different elevations are not an error."""
    from observations.identity import metadata_conflicts

    stats = OverlapStats("tasmax", 1208, 0.9916, -1.2, 1.289, 0.0099)
    evidence = [
        MatchPair("ecad:028043", "ghcnd:ITM00016410", TIER_NEAR, False,
                  2.41, 54.2, 0.5, stats, "elevation differs by 54 m"),
    ]
    assert metadata_conflicts(evidence).empty


def test_identical_series_override_the_elevation_gate():
    """
    Two archives publishing one instrument under irreconcilable elevations:
    the observations disprove the metadata, so the merge stands.
    """
    values = [100.0 + (i % 37) * 4.0 for i in range(800)]     # tenths of degC
    same = _daily(values)
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): same, ("ghcnd", "G", "tasmax"): same.copy()}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=50.0)
    b = _station("ghcnd", "G", 37.9000, 23.7000, elev=613.0)   # 563 m apart
    pair = evaluate_pair(a, b, loader)
    assert pair.accepted
    assert "overridden by identical series" in pair.reason


def test_merely_correlated_series_do_not_override_elevation():
    """The override is narrow: correlation is not identity."""
    left, right = _correlated_pair(offset=2.0)
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): left, ("ghcnd", "G", "tasmax"): right}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=50.0)
    b = _station("ghcnd", "G", 37.9000, 23.7000, elev=613.0)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert "elevation differs" in pair.reason


def test_identical_but_short_series_do_not_override_elevation():
    """Ten identical days is not proof; the minimum overlap still applies."""
    short = _daily([100.0, 110.0, 120.0, 130.0, 140.0] * 2)   # tenths of degC
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): short, ("ghcnd", "G", "tasmax"): short.copy()}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=50.0)
    b = _station("ghcnd", "G", 37.9000, 23.7000, elev=613.0)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted


def test_identical_series_override_missing_elevation():
    values = [100.0 + (i % 37) * 4.0 for i in range(800)]     # tenths of degC
    same = _daily(values)
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): same, ("ghcnd", "G", "tasmax"): same.copy()}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=float("nan"))
    b = _station("ghcnd", "G", 37.9000, 23.7000, elev=100.0)
    pair = evaluate_pair(a, b, loader)
    assert pair.accepted
    assert "missing elevation overridden" in pair.reason


def test_conflict_queue_requires_the_minimum_overlap():
    """
    One shared day that happens to agree is not evidence that two archives
    describe one instrument; it would only put noise in the review queue.
    """
    from observations.identity import metadata_conflicts

    one_day = OverlapStats("tasmax", 1, float("nan"), 0.0, 0.0, 1.0)
    evidence = [
        MatchPair("ecad:023329", "gsod:16765099999", TIER_NEAR, False,
                  0.78, 14.0, 0.5, one_day, "insufficient overlap"),
    ]
    assert metadata_conflicts(evidence).empty


def test_merge_requires_agreement_not_only_correlation():
    """
    Measured on the archive: pairs 2.4 km apart correlate at 0.9915 while
    differing 1.3 degC every day.  Correlation alone cannot separate a
    duplicate from a distinct neighbouring site.
    """
    left, right = _correlated_pair(offset=1.3)     # strongly correlated, offset
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): left, ("ghcnd", "G", "tasmax"): right}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=100.0)
    b = _station("ghcnd", "G", 37.9005, 23.7005, elev=105.0)
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert "on average" in pair.reason
    assert pair.stats.correlation > 0.99          # it did pass correlation


def test_close_agreement_still_merges():
    values = [100.0 + (i % 53) * 3.0 for i in range(900)]
    left = _daily(values)
    right = _daily([v + 1.0 for v in values])      # 0.1 degC apart in tenths
    loader = FakeLoader(
        {("ecad", "1", "tasmax"): left, ("ghcnd", "G", "tasmax"): right}
    )
    a = _station("ecad", "1", 37.9000, 23.7000, elev=100.0)
    b = _station("ghcnd", "G", 37.9005, 23.7005, elev=105.0)
    assert evaluate_pair(a, b, loader).accepted


# ---------------------------------------------------------------------------
# Canonical metadata resolution
# ---------------------------------------------------------------------------

def _alias_row(cid, source, sid, lat, lon, elev, name="X", variables="tasmax", risk=False):
    return {
        "canonical_station_id": cid, "source": source, "source_station_id": sid,
        "name": name, "lat": lat, "lon": lon, "elevation_m": elev,
        "wmo_id": "", "wigos_id": "", "n_variables": 1, "variables": variables,
        "is_representative": False, "match_tier": 1, "match_tier_name": "x",
        "match_evidence": "", "duplicate_risk": risk, "agreement_review": False,
        "group_formed_transitively": False, "valid_from": "", "valid_to": "",
    }


def test_metadata_winner_is_the_alias_with_most_observations():
    from observations.identity import resolve_canonical_metadata

    aliases = pd.DataFrame([
        _alias_row("c:1", "ecad", "1", 37.90, 23.70, 31.0, name="LONG"),
        _alias_row("c:1", "ghcnd", "G", 37.91, 23.71, 595.0, name="SHORT"),
    ])
    loader = FakeLoader({
        ("ecad", "1", "tasmax"): _daily([1.0] * 900),
        ("ghcnd", "G", "tasmax"): _daily([1.0] * 300),
    })
    out = resolve_canonical_metadata(aliases, loader, variables=("tasmax",))
    assert out.loc[0, "metadata_source"] == "ecad:1"
    assert out.loc[0, "elevation_m"] == 31.0
    assert out.loc[0, "name"] == "LONG"
    assert out.loc[0, "metadata_n_observations"] == 900


def test_metadata_fields_come_from_one_alias_as_a_block():
    """
    Mixing fields from different winners could manufacture a
    coordinate/elevation pair no archive ever published.
    """
    from observations.identity import resolve_canonical_metadata

    aliases = pd.DataFrame([
        _alias_row("c:1", "ecad", "1", 37.90, 23.70, 31.0),
        _alias_row("c:1", "gsod", "S", 41.00, 19.00, 595.0),
    ])
    loader = FakeLoader({
        ("ecad", "1", "tasmax"): _daily([1.0] * 900),
        ("gsod", "S", "tasmax"): _daily([1.0] * 100),
    })
    out = resolve_canonical_metadata(aliases, loader, variables=("tasmax",))
    assert (out.loc[0, "lat"], out.loc[0, "lon"], out.loc[0, "elevation_m"]) == (
        37.90, 23.70, 31.0
    )


def test_metadata_resolution_is_deterministic_on_ties():
    from observations.identity import resolve_canonical_metadata

    aliases = pd.DataFrame([
        _alias_row("c:1", "gsod", "S", 1.0, 1.0, 10.0),
        _alias_row("c:1", "ecad", "E", 2.0, 2.0, 20.0),
    ])
    loader = FakeLoader({
        ("ecad", "E", "tasmax"): _daily([1.0] * 500),
        ("gsod", "S", "tasmax"): _daily([1.0] * 500),
    })
    first = resolve_canonical_metadata(aliases, loader, variables=("tasmax",))
    second = resolve_canonical_metadata(aliases, loader, variables=("tasmax",))
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "metadata_source"] == "ecad:E"     # tie broken on key


def test_single_source_station_needs_no_contest():
    from observations.identity import resolve_canonical_metadata

    aliases = pd.DataFrame([_alias_row("c:1", "ecad", "1", 37.9, 23.7, 100.0)])
    out = resolve_canonical_metadata(aliases, FakeLoader(), variables=("tasmax",))
    assert out.loc[0, "selection"] == "single source"
    assert out.loc[0, "n_aliases"] == 1


def test_resolution_records_the_full_contest_for_audit():
    from observations.identity import resolve_canonical_metadata

    aliases = pd.DataFrame([
        _alias_row("c:1", "ecad", "1", 37.90, 23.70, 31.0),
        _alias_row("c:1", "ghcnd", "G", 37.91, 23.71, 595.0),
    ])
    loader = FakeLoader({
        ("ecad", "1", "tasmax"): _daily([1.0] * 900),
        ("ghcnd", "G", "tasmax"): _daily([1.0] * 300),
    })
    out = resolve_canonical_metadata(aliases, loader, variables=("tasmax",))
    assert "ecad:1(900)" in out.loc[0, "selection"]
    assert "ghcnd:G(300)" in out.loc[0, "selection"]


def test_overlap_statistics_ignore_impossible_sentinel_values():
    """
    GHCN-D's undocumented 999.0 degC sentinel made two identical records of
    MONTE CIMONE report r=0.14 and a 2.10 degC mean difference, so the pair
    was rejected on a statistic that was pure artefact.
    """
    values = [100.0 + (i % 53) * 4.0 for i in range(800)]
    clean = _daily(values)
    poisoned = _daily(values)
    poisoned.iloc[10] = 9990.0                    # 999.0 degC once converted

    stats = overlap_statistics(clean, poisoned, "tasmax", "degC_tenths", "degC_tenths")
    assert stats.n_overlap == 799                 # the sentinel day is excluded
    assert stats.correlation == pytest.approx(1.0, abs=1e-9)
    assert stats.mean_abs_diff == pytest.approx(0.0, abs=1e-9)


def test_sentinel_removal_does_not_discard_real_extremes():
    values = [-450.0] + [200.0] * 799            # -45 degC is cold but real
    stats = overlap_statistics(
        _daily(values), _daily(values), "tasmax", "degC_tenths", "degC_tenths"
    )
    assert stats.n_overlap == 800


def test_zero_overlap_pairs_are_never_merged():
    """
    Project rule, stricter than plan tier 2: two records sharing no
    observation day cannot be shown to be one instrument, however well their
    coordinates, names and elevations agree.
    """
    loader = FakeLoader()          # no series at all for either station
    a = _station("ecad", "1", 37.9000, 23.7000, elev=100.0, name="KAIROUAN")
    b = _station("ghcnd", "G", 37.9000, 23.7000, elev=100.0, name="KAIROUAN")
    pair = evaluate_pair(a, b, loader)
    assert not pair.accepted
    assert pair.reason == identity.NEVER_MERGE_REASON


def test_overlap_in_any_variable_escapes_the_never_merge_rule():
    """
    A pair whose temperatures never coincide but whose wind records do has
    evidence, and must still be evaluated normally.
    """
    wind = _daily([50.0 + (i % 17) for i in range(800)])
    loader = FakeLoader({
        ("ecad", "1", "sfcWind"): wind,
        ("ghcnd", "G", "sfcWind"): wind.copy(),
    })
    a = _station("ecad", "1", 37.9000, 23.7000, variables=("sfcWind",))
    b = _station("ghcnd", "G", 37.9005, 23.7005, variables=("sfcWind",))
    pair = evaluate_pair(a, b, loader)
    assert pair.reason != identity.NEVER_MERGE_REASON


def test_total_overlap_counts_across_every_variable():
    from observations.identity import total_overlap_days

    loader = FakeLoader({
        ("ecad", "1", "tasmax"): _daily([1.0] * 10, start="1990-01-01"),
        ("ghcnd", "G", "tasmax"): _daily([1.0] * 10, start="1995-01-01"),
        ("ecad", "1", "pr"): _daily([1.0] * 10, start="1990-01-01"),
        ("ghcnd", "G", "pr"): _daily([1.0] * 10, start="1990-01-05"),
    })
    a = _station("ecad", "1", 37.9, 23.7)
    b = _station("ghcnd", "G", 37.9, 23.7)
    assert total_overlap_days(a, b, loader) == 6      # pr overlaps, tasmax does not


# ---------------------------------------------------------------------------
# Reviewer decisions
# ---------------------------------------------------------------------------

def _decision_inventories():
    return {
        "ecad": _inventory(
            [dict(source="ecad", source_station_id="000247", name="ATHENS",
                  lat=37.9000, lon=23.7000, elevation_m=107, variable="tasmax")]
        ),
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="GRE00105241", name="ATHENS AP",
                  lat=37.9005, lon=23.7005, elevation_m=110, variable="tasmax")]
        ),
    }


def test_reviewer_yes_merges_a_pair_the_gates_rejected():
    """The review exists precisely to overrule the automated gates."""
    values = [100.0 + (i % 37) * 4.0 for i in range(800)]
    loader = FakeLoader({
        ("ecad", "000247", "tasmax"): _daily(values),
        ("ghcnd", "GRE00105241", "tasmax"): _daily([v + 9.0 for v in values]),
    })
    without = build_crosswalk(_decision_inventories(), loader)
    assert without.n_canonical() == 2                    # gates refuse it

    with_decision = build_crosswalk(
        _decision_inventories(), loader,
        decisions={("ecad:000247", "ghcnd:GRE00105241"): "yes"},
    )
    assert with_decision.n_canonical() == 1
    assert with_decision.evidence["reason"].str.contains("reviewer decision").any()


def test_reviewer_no_unmerges_a_pair_the_gates_accepted():
    values = [100.0 + (i % 37) * 4.0 for i in range(800)]
    loader = FakeLoader({
        ("ecad", "000247", "tasmax"): _daily(values),
        ("ghcnd", "GRE00105241", "tasmax"): _daily(values),
    })
    without = build_crosswalk(_decision_inventories(), loader)
    assert without.n_canonical() == 1                    # gates accept it

    with_decision = build_crosswalk(
        _decision_inventories(), loader,
        decisions={("ecad:000247", "ghcnd:GRE00105241"): "no"},
    )
    assert with_decision.n_canonical() == 2


def test_never_merge_rule_outranks_a_reviewer_yes():
    """
    The project declared the zero-overlap rule absolute, with no override.
    A decision to merge such a pair must not take effect.
    """
    crosswalk = build_crosswalk(
        _decision_inventories(), FakeLoader(),      # no series -> zero overlap
        decisions={("ecad:000247", "ghcnd:GRE00105241"): "yes"},
    )
    assert crosswalk.n_canonical() == 2
    assert crosswalk.evidence["reason"].str.contains("rule is absolute").any()


def test_decisions_other_than_yes_or_no_are_ignored():
    """A stray note must not be mistaken for an instruction."""
    values = [100.0 + (i % 37) * 4.0 for i in range(800)]
    loader = FakeLoader({
        ("ecad", "000247", "tasmax"): _daily(values),
        ("ghcnd", "GRE00105241", "tasmax"): _daily([v + 9.0 for v in values]),
    })
    crosswalk = build_crosswalk(
        _decision_inventories(), loader,
        decisions={("ecad:000247", "ghcnd:GRE00105241"): "maybe later"},
    )
    assert crosswalk.n_canonical() == 2
