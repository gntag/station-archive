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

from station_archive import identity, standardize
from station_archive.policy import DEFAULT_POLICY
from station_archive.identity import (
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
from station_archive.types import INVENTORY_COLUMNS, StationInventory


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
    pairs = candidate_pairs(records, DEFAULT_POLICY.identity.near_deg)
    assert all(a.source != b.source for a, b in pairs)


def test_candidates_find_neighbours_across_bucket_boundaries():
    """
    Two stations metres apart but on opposite sides of a bucket edge must
    still be compared, or the blocking would silently lose real duplicates.
    """
    tol = DEFAULT_POLICY.identity.near_deg
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
    assert candidate_pairs(records, DEFAULT_POLICY.identity.near_deg) == []


def test_candidate_generation_is_deterministic():
    records = [
        _station("ecad", "1", 37.900, 23.700),
        _station("ghcnd", "G", 37.901, 23.701),
        _station("gsod", "S", 37.902, 23.702),
    ]
    first = [(a.key, b.key) for a, b in candidate_pairs(records, DEFAULT_POLICY.identity.near_deg)]
    second = [(a.key, b.key) for a, b in candidate_pairs(records, DEFAULT_POLICY.identity.near_deg)]
    assert first == second


def test_candidates_skip_stations_without_coordinates():
    records = [
        _station("ecad", "1", float("nan"), float("nan")),
        _station("ghcnd", "G", 37.9, 23.7),
    ]
    assert candidate_pairs(records, DEFAULT_POLICY.identity.near_deg) == []


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
    from station_archive.identity import _DisjointSet, _assemble

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
    from station_archive.identity import _DisjointSet

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
    from station_archive.identity import agreement_needs_review

    good = OverlapStats("tasmax", 800, 0.999, 0.0, 0.02, 0.99)
    poor = OverlapStats("tasmax", 800, 0.981, -1.3, 1.31, 0.03)
    assert not agreement_needs_review(good)
    assert agreement_needs_review(poor)


def test_agreement_review_threshold_is_per_unit():
    """2 mm of precipitation disagreement is not 2 degC of temperature."""
    from station_archive.identity import agreement_needs_review

    assert not agreement_needs_review(OverlapStats("pr", 800, 0.99, 0.0, 1.5, 0.5))
    assert agreement_needs_review(OverlapStats("tasmax", 800, 0.99, 0.0, 1.5, 0.5))


def test_identical_records_are_recognised():
    from station_archive.identity import records_are_identical

    assert records_are_identical(OverlapStats("tasmax", 3210, 1.0, 0.0, 0.0, 1.0))
    assert not records_are_identical(OverlapStats("tasmax", 3210, 0.99, 0.2, 0.74, 0.06))


def test_metadata_conflict_reported_when_series_identical_but_elevation_differs():
    """
    Two archives publishing one instrument under irreconcilable elevations is
    a metadata error, not two stations -- and elevation is a model predictor,
    so it must surface rather than be silently absorbed.
    """
    from station_archive.identity import metadata_conflicts

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
    from station_archive.identity import metadata_conflicts

    stats = OverlapStats("tasmax", 3000, 1.0, 0.0, 0.0, 1.0)
    evidence = [
        MatchPair("ecad:1", "ghcnd:G", TIER_COINCIDENT, True,
                  0.05, 2.0, 1.0, stats, "ok"),
    ]
    assert metadata_conflicts(evidence).empty


def test_merely_similar_stations_are_not_called_a_metadata_conflict():
    """Distinct nearby stations at different elevations are not an error."""
    from station_archive.identity import metadata_conflicts

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
    from station_archive.identity import metadata_conflicts

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
    from station_archive.identity import resolve_canonical_metadata

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
    from station_archive.identity import resolve_canonical_metadata

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
    from station_archive.identity import resolve_canonical_metadata

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
    from station_archive.identity import resolve_canonical_metadata

    aliases = pd.DataFrame([_alias_row("c:1", "ecad", "1", 37.9, 23.7, 100.0)])
    out = resolve_canonical_metadata(aliases, FakeLoader(), variables=("tasmax",))
    assert out.loc[0, "selection"] == "single source"
    assert out.loc[0, "n_aliases"] == 1


def test_resolution_records_the_full_contest_for_audit():
    from station_archive.identity import resolve_canonical_metadata

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
    from station_archive.identity import total_overlap_days

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


# ---------------------------------------------------------------------------
# Reviewer decisions on IDENTIFIER-tier pairs
#
# These are the pairs the workbook's Already_merged and Beyond_radius sheets
# offer a decision on. The tier-1 links are applied before the candidate loop
# and excluded from it, so the override used to be read and then dropped: 156
# accepted identifier merges and one too-far refusal in the published archive
# carried a DECISION_merge column that could not act.
# ---------------------------------------------------------------------------

def _identifier_inventories():
    """A c3s_land/ghcnd pair sharing an exact GHCN-D identifier: tier 1."""
    return {
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="GRE00105241", name="ATHENS",
                  lat=37.9000, lon=23.7500, elevation_m=107, variable="tasmax")]
        ),
        "c3s_land": _inventory(
            [dict(source="c3s_land", source_station_id="GRE00105241", name="ATHENS",
                  lat=37.9000, lon=23.7500, elevation_m=107, variable="tasmax")]
        ),
    }


_IDENTIFIER_PAIR = ("c3s_land:GRE00105241", "ghcnd:GRE00105241")


def test_the_pair_under_test_really_is_tier_one():
    """Guards the two tests below: if this stops being tier 1 they prove nothing."""
    crosswalk = build_crosswalk(_identifier_inventories(), FakeLoader())
    assert list(crosswalk.evidence["tier"]) == [TIER_IDENTIFIER]
    assert crosswalk.n_canonical() == 1


def test_reviewer_no_splits_an_identifier_merge():
    """
    'Write no in DECISION_merge to split one' is what the workbook's summary
    sheet promises for Already_merged, which lists every automatic merge --
    identifier merges included.
    """
    crosswalk = build_crosswalk(
        _identifier_inventories(), FakeLoader(),
        decisions={_IDENTIFIER_PAIR: "no"},
    )
    assert crosswalk.n_canonical() == 2
    assert crosswalk.evidence["reason"].str.contains("reviewer decision").any()


def test_reviewer_yes_merges_an_identifier_pair_rejected_as_too_far():
    """
    match_tier1_identifier's own docstring: a pair that fails the distance
    check 'appears in the review workbook, where a reviewer can overturn it'.
    """
    inventories = {
        "ghcnd": _inventory(
            [dict(source="ghcnd", source_station_id="GR000167460", name="METHONI",
                  lat=36.83, lon=21.70, elevation_m=52, wmo_id="16746",
                  variable="tasmax")]
        ),
        "gsod": _inventory(
            [dict(source="gsod", source_station_id="16746099999", name="METHONI",
                  lat=36.88, lon=21.70, elevation_m=52, wmo_id="16746",
                  variable="tasmax")]
        ),
    }
    pair = ("ghcnd:GR000167460", "gsod:16746099999")

    refused = build_crosswalk(inventories, FakeLoader())
    assert refused.n_canonical() == 2
    assert not refused.evidence["accepted"].any()

    overturned = build_crosswalk(inventories, FakeLoader(), decisions={pair: "yes"})
    assert overturned.n_canonical() == 1
    assert overturned.evidence["reason"].str.contains("reviewer decision").any()


def test_an_overturned_identifier_pair_keeps_the_automated_verdict_on_record():
    """
    The evidence table must say both things: what the rules concluded and that
    a human overruled them. A reason that recorded only the human decision
    would erase the basis for revisiting it.

    The tier deliberately stays as the rules assigned it, so a reader cannot
    mistake a human decision for machine evidence.
    """
    crosswalk = build_crosswalk(
        _identifier_inventories(), FakeLoader(),
        decisions={_IDENTIFIER_PAIR: "no"},
    )
    reason = crosswalk.evidence["reason"].iloc[0]
    assert reason.startswith("reviewer decision: do not merge")
    assert "exact ghcnd_id identifier GRE00105241" in reason
    assert list(crosswalk.evidence["tier"]) == [TIER_IDENTIFIER]


def test_an_undecided_identifier_pair_is_untouched():
    """A decision about some other pair must not leak onto this one."""
    crosswalk = build_crosswalk(
        _identifier_inventories(), FakeLoader(),
        decisions={("ecad:000247", "ghcnd:GRE00105241"): "no"},
    )
    assert crosswalk.n_canonical() == 1
    assert not crosswalk.evidence["reason"].str.contains("reviewer").any()


# ---------------------------------------------------------------------------
# Tier-1 distance sanity check
#
# An exact identifier used to merge with no distance test whatsoever. On the
# real archive that merged METHONI with CYCLADES ISLANDS 327 km apart, KALAMATA
# with MIKONOS and SYROS, and HERAKLION with KARPATHOS -- because 18 GSOD
# stations carried a WMO number belonging to a different station.
# ---------------------------------------------------------------------------

def test_identifier_match_beyond_the_distance_gate_is_refused():
    records = [
        _station("ghcnd", "GR000016734", 36.8331, 21.7000, wmo="16734"),
        _station("gsod", "16734199999", 37.0833, 25.3667, wmo="16734"),
    ]
    pairs = match_tier1_identifier(records)
    assert len(pairs) == 1
    assert not pairs[0].accepted
    assert pairs[0].tier == TIER_UNRESOLVED
    assert "km apart" in pairs[0].reason
    assert "16734" in pairs[0].reason      # the identifier is still named


def test_identifier_match_within_the_gate_still_merges():
    records = [
        _station("ghcnd", "GR000016716", 37.9000, 23.7500, wmo="16716"),
        _station("gsod", "16716099999", 37.8822, 23.7348, wmo="16716"),
    ]
    pairs = match_tier1_identifier(records)
    assert len(pairs) == 1
    assert pairs[0].accepted
    assert pairs[0].tier == TIER_IDENTIFIER


def test_identifier_distance_gate_ignores_elevation():
    """Elevation is deliberately not part of this check."""
    a = _station("ghcnd", "X", 37.9000, 23.7500, elev=10.0, wmo="16716")
    b = _station("gsod", "16716099999", 37.9005, 23.7505, elev=900.0, wmo="16716")
    pairs = match_tier1_identifier([a, b])
    assert pairs[0].accepted


def test_identifier_match_with_missing_coordinates_is_not_refused():
    """A source publishing no coordinates must not lose its identifier link."""
    a = _station("ghcnd", "X", float("nan"), float("nan"), wmo="16716")
    b = _station("gsod", "16716099999", 37.9, 23.7, wmo="16716")
    pairs = match_tier1_identifier([a, b])
    assert pairs[0].accepted
    assert "distance unknown" in pairs[0].reason


def test_refused_identifier_pair_raises_duplicate_risk_on_both():
    inventories = {
        "ghcnd": _inventory([dict(source="ghcnd", source_station_id="GR000016734",
                                  name="METHONI", lat=36.8331, lon=21.7,
                                  elevation_m=52, variable="tasmax", wmo_id="16734")]),
        "gsod": _inventory([dict(source="gsod", source_station_id="16734199999",
                                 name="CYCLADES ISLANDS", lat=37.0833, lon=25.3667,
                                 elevation_m=3, variable="tasmax", wmo_id="16734")]),
    }
    crosswalk = build_crosswalk(inventories, FakeLoader())
    assert crosswalk.n_canonical() == 2             # not merged
    assert crosswalk.aliases["duplicate_risk"].all()


# ---------------------------------------------------------------------------
# Visibility of merges nobody approved
# ---------------------------------------------------------------------------

def _accepted_evidence():
    import pandas as pd
    return pd.DataFrame([
        {"left": "ecad:1", "right": "ghcnd:A", "tier": TIER_IDENTIFIER,
         "tier_name": "exact_identifier", "accepted": True, "distance_km": 0.02,
         "elevation_diff_m": 1.0, "name_similarity": 1.0,
         "reason": "exact wmo identifier 16716", "match_variable": "",
         "n_overlap_days": 0, "correlation": float("nan"), "bias": float("nan"),
         "mean_abs_diff": float("nan"), "exact_fraction": float("nan")},
        {"left": "ecad:2", "right": "gsod:B", "tier": TIER_NEAR,
         "tier_name": "near_coordinates_plus_series", "accepted": True,
         "distance_km": 2.4, "elevation_diff_m": 40.0, "name_similarity": 0.0,
         "reason": "2.40 km apart, elevation within 40 m, tasmax r=0.999 over 900 days",
         "match_variable": "tasmax", "n_overlap_days": 900, "correlation": 0.999,
         "bias": 0.0, "mean_abs_diff": 0.02, "exact_fraction": 0.99},
        {"left": "ecad:3", "right": "gsod:C", "tier": TIER_NEAR,
         "tier_name": "near_coordinates_plus_series", "accepted": True,
         "distance_km": 0.5, "elevation_diff_m": 2.0, "name_similarity": 0.9,
         "reason": "reviewer decision: merge (automated gate said: only 0 usable days)",
         "match_variable": "tasmax", "n_overlap_days": 0, "correlation": float("nan"),
         "bias": float("nan"), "mean_abs_diff": float("nan"),
         "exact_fraction": float("nan")},
    ])


def test_review_lists_merges_made_without_approval():
    import pandas as pd
    from station_archive import review as review_module

    aliases = pd.DataFrame([
        _alias_row("ecad:1", "ecad", "1", 37.9, 23.7, 10.0),
        _alias_row("ecad:1", "ghcnd", "A", 37.9, 23.7, 11.0),
        _alias_row("ecad:2", "ecad", "2", 38.0, 24.0, 10.0),
        _alias_row("ecad:2", "gsod", "B", 38.02, 24.0, 50.0),
        _alias_row("ecad:3", "ecad", "3", 39.0, 25.0, 10.0),
        _alias_row("ecad:3", "gsod", "C", 39.0, 25.0, 12.0),
    ])
    sheets = review_module.build_review(
        _accepted_evidence(), aliases, FakeLoader(), progress=False
    )
    assert "Already_merged" in sheets
    tab = sheets["Already_merged"]
    pairs = set(zip(tab["left"], tab["right"]))
    assert ("ecad:1", "ghcnd:A") in pairs          # automatic, identifier
    assert ("ecad:2", "gsod:B") in pairs           # automatic, series
    assert ("ecad:3", "gsod:C") not in pairs       # you approved this one


def test_already_merged_tab_carries_a_decision_column():
    import pandas as pd
    from station_archive import review as review_module

    aliases = pd.DataFrame([
        _alias_row("ecad:1", "ecad", "1", 37.9, 23.7, 10.0),
        _alias_row("ecad:1", "ghcnd", "A", 37.9, 23.7, 11.0),
    ])
    tab = review_module.build_review(
        _accepted_evidence().head(1), aliases, FakeLoader(), progress=False
    )["Already_merged"]
    for column in review_module.DECISION_COLUMNS:
        assert column in tab.columns
    assert (tab[list(review_module.DECISION_COLUMNS)] == "").all().all()


def test_already_merged_tab_puts_the_least_defensible_first():
    import pandas as pd
    from station_archive import review as review_module

    aliases = pd.DataFrame([
        _alias_row("ecad:1", "ecad", "1", 37.9, 23.7, 10.0),
        _alias_row("ecad:1", "ghcnd", "A", 37.9, 23.7, 11.0),
        _alias_row("ecad:2", "ecad", "2", 38.0, 24.0, 10.0),
        _alias_row("ecad:2", "gsod", "B", 38.02, 24.0, 50.0),
    ])
    tab = review_module.build_review(
        _accepted_evidence().head(2), aliases, FakeLoader(), progress=False
    )["Already_merged"]
    assert tab.iloc[0]["right"] == "gsod:B"        # 2.4 km beats 0.02 km


def test_decisions_are_read_from_every_decision_sheet(tmp_path):
    """
    The Beyond_radius sheet used to carry a decision column that nothing read,
    so a reviewer could fill it in and change nothing.
    """
    import pandas as pd

    path = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([{"left": "ecad:1", "right": "ghcnd:A",
                       "DECISION_merge": "yes"}]).to_excel(
            writer, sheet_name="Review", index=False)
        pd.DataFrame([{"left": "ecad:2", "right": "gsod:B",
                       "DECISION_merge": "no"}]).to_excel(
            writer, sheet_name="Already_merged", index=False)
        pd.DataFrame([{"left": "ecad:9", "right": "gsod:Z",
                       "DECISION_merge": "yes"}]).to_excel(
            writer, sheet_name="Beyond_radius", index=False)

    decisions = identity.load_decisions(path)
    assert decisions[("ecad:1", "ghcnd:A")] == "yes"
    assert decisions[("ecad:2", "gsod:B")] == "no"
    assert decisions[("ecad:9", "gsod:Z")] == "yes"


def test_load_decisions_tolerates_a_workbook_without_the_new_sheet(tmp_path):
    import pandas as pd

    path = tmp_path / "old.xlsx"
    pd.DataFrame([{"left": "ecad:1", "right": "ghcnd:A",
                   "DECISION_merge": "yes"}]).to_excel(
        path, sheet_name="Review", index=False)
    assert identity.load_decisions(path) == {("ecad:1", "ghcnd:A"): "yes"}


def test_refusing_an_automatic_merge_unmerges_it():
    values = [100.0 + (i % 37) * 4.0 for i in range(800)]
    loader = FakeLoader({
        ("ecad", "000247", "tasmax"): _daily(values),
        ("ghcnd", "GRE00105241", "tasmax"): _daily(values),
    })
    assert build_crosswalk(_decision_inventories(), loader).n_canonical() == 1
    refused = build_crosswalk(
        _decision_inventories(), loader,
        decisions={("ecad:000247", "ghcnd:GRE00105241"): "no"},
    )
    assert refused.n_canonical() == 2
    assert refused.aliases["duplicate_risk"].all()


def test_rebuilding_the_workbook_keeps_decisions_already_made(tmp_path):
    """
    Regenerating the workbook used to blank every DECISION_merge, so a rebuild
    silently destroyed the reviewer's work -- which is why hand-made backup
    copies of it accumulate beside it. Decisions already made must survive.
    """
    import pandas as pd
    from station_archive import review as review_module

    path = tmp_path / "wb.xlsx"
    review_module.write_workbook(path, {
        "Review": pd.DataFrame([{"left": "ecad:1", "right": "ghcnd:A",
                                 "DECISION_merge": "yes", "DECISION_note": "same site"}]),
    })
    previous = identity.load_decisions(path)
    assert previous == {("ecad:1", "ghcnd:A"): "yes"}

    rebuilt = review_module.carry_forward_decisions(
        {"Review": pd.DataFrame([
            {"left": "ecad:1", "right": "ghcnd:A", "DECISION_merge": "", "DECISION_note": ""},
            {"left": "ecad:2", "right": "ghcnd:B", "DECISION_merge": "", "DECISION_note": ""},
        ])},
        previous,
    )
    row = rebuilt["Review"].set_index(["left", "right"]).loc[("ecad:1", "ghcnd:A")]
    assert row["DECISION_merge"] == "yes"
    assert rebuilt["Review"].set_index(["left", "right"]).loc[
        ("ecad:2", "ghcnd:B"), "DECISION_merge"] == ""


def test_carrying_decisions_forward_ignores_sheets_without_the_column():
    import pandas as pd
    from station_archive import review as review_module

    sheets = {"Per_variable": pd.DataFrame([{"left": "a", "right": "b", "variable": "tas"}])}
    out = review_module.carry_forward_decisions(sheets, {("a", "b"): "yes"})
    assert "DECISION_merge" not in out["Per_variable"].columns


def test_a_decision_survives_even_when_its_pair_is_no_longer_offered():
    """
    A 'yes' decision merges a pair, so the next rebuild presents it nowhere:
    not in Review (it is no longer rejected) and not in Already_merged (which
    excludes reviewer-decided pairs). Without a durable ledger the decision
    vanishes, and the merge it authorised is silently lost on the next run.
    """
    import pandas as pd
    from station_archive import review as review_module

    sheets = {"Review": pd.DataFrame([
        {"left": "ecad:2", "right": "ghcnd:B", "DECISION_merge": "", "DECISION_note": ""},
    ])}
    previous = {("ecad:1", "ghcnd:A"): "yes", ("ecad:2", "ghcnd:B"): "no"}
    out = review_module.carry_forward_decisions(sheets, previous)

    assert "Decisions" in out
    ledger = out["Decisions"].set_index(["left", "right"])
    assert ledger.loc[("ecad:1", "ghcnd:A"), "DECISION_merge"] == "yes"
    assert ledger.loc[("ecad:2", "ghcnd:B"), "DECISION_merge"] == "no"
    assert not ledger.loc[("ecad:1", "ghcnd:A"), "still_offered"]
    assert ledger.loc[("ecad:2", "ghcnd:B"), "still_offered"]


def test_the_ledger_is_read_back_as_a_decision_source(tmp_path):
    import pandas as pd
    from station_archive import review as review_module

    path = tmp_path / "wb.xlsx"
    review_module.write_workbook(path, {
        "Decisions": pd.DataFrame([{"left": "ecad:1", "right": "ghcnd:A",
                                    "DECISION_merge": "yes", "still_offered": False}]),
    })
    assert identity.load_decisions(path) == {("ecad:1", "ghcnd:A"): "yes"}


def test_blank_decisions_stay_blank_not_nan():
    import pandas as pd
    from station_archive import review as review_module

    sheets = {"Review": pd.DataFrame([
        {"left": "a", "right": "b", "DECISION_merge": float("nan"), "DECISION_note": ""},
    ])}
    out = review_module.carry_forward_decisions(sheets, {("x", "y"): "yes"})
    assert out["Review"].loc[0, "DECISION_merge"] == ""


# ---------------------------------------------------------------------------
# A recorded conflict should say how it was resolved
# ---------------------------------------------------------------------------

def test_conflicts_record_which_alias_supplied_the_coordinates():
    """
    metadata_conflicts.csv said two archives disagree and stopped there, so a
    reader could not tell which coordinate the archive actually uses. The rule
    is the alias with the most observations across all variables; the file now
    says so, per row, with the count behind it.
    """
    import pandas as pd
    from station_archive.identity import annotate_conflicts

    conflicts = pd.DataFrame([
        {"left": "ecad:1", "right": "gsod:B", "conflict": "coordinates",
         "distance_km": 1.2, "elevation_diff_m": 3.0},
    ])
    aliases = pd.DataFrame([
        _alias_row("c:1", "ecad", "1", 37.90, 23.70, 10.0),
        _alias_row("c:1", "gsod", "B", 37.91, 23.71, 13.0),
    ])
    canonical = pd.DataFrame([{
        "canonical_station_id": "c:1", "metadata_source": "gsod:B",
        "metadata_n_observations": 39239, "lat": 37.91, "lon": 23.71,
        "elevation_m": 13.0,
        "selection": "gsod:B(39239) > ecad:1(38325)",
    }])
    out = annotate_conflicts(conflicts, aliases, canonical)
    row = out.iloc[0]
    assert row["coordinates_used_from"] == "gsod:B"
    assert row["coordinates_used_n_obs"] == 39239
    assert row["coordinates_used_side"] == "right"
    assert row["lat_used"] == 37.91 and row["lon_used"] == 23.71
    assert "most observations" in row["coordinate_rule"]


def test_conflict_resolution_names_a_third_alias_when_one_wins():
    """A group can have more than two aliases, and the winner may be neither
    side of the reported pair."""
    import pandas as pd
    from station_archive.identity import annotate_conflicts

    conflicts = pd.DataFrame([
        {"left": "ecad:2", "right": "gsod:C", "conflict": "coordinates",
         "distance_km": 0.7, "elevation_diff_m": 14.0},
    ])
    aliases = pd.DataFrame([
        _alias_row("c:9", "ecad", "2", 35.42, 27.14, 20.0),
        _alias_row("c:9", "gsod", "C", 35.43, 27.15, 6.0),
        _alias_row("c:9", "ecad", "3", 35.42, 27.14, 20.0),
    ])
    canonical = pd.DataFrame([{
        "canonical_station_id": "c:9", "metadata_source": "ecad:3",
        "metadata_n_observations": 13696, "lat": 35.42, "lon": 27.14,
        "elevation_m": 20.0, "selection": "ecad:3(13696) > ecad:2(13696)",
    }])
    out = annotate_conflicts(conflicts, aliases, canonical)
    assert out.iloc[0]["coordinates_used_side"] == "another alias"
    assert out.iloc[0]["coordinates_used_from"] == "ecad:3"


def test_annotating_an_empty_conflict_table_is_harmless():
    import pandas as pd
    from station_archive.identity import annotate_conflicts

    out = annotate_conflicts(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert out.empty
