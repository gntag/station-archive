"""
Tests for scientific policy, the source registry, and the candidate-generation
guarantee that depends on both.

The behaviours under test are the ones that were previously unenforced or
untrue: that a policy can be overridden without editing source, that a
configuration naming a source which cannot do what is asked of it is refused at
construction rather than failing three stages later, and that no station pair
inside the distance gate can escape candidate generation.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from station_archive import settings
from station_archive.identity import (
    StationRecord,
    _degrees_to_km,
    candidate_pairs,
    haversine_km,
    longitude_scale,
    provider_units,
)
from station_archive.policy import DEFAULT_POLICY, Policy
from station_archive.sources import registry
from station_archive.types import Domain


def _station(source, sid, lat, lon):
    return StationRecord(
        source=source, source_station_id=sid, name="X", lat=lat, lon=lon,
        elevation_m=100.0, wmo_id="", wigos_id="",
        variables=frozenset({"tasmax"}), first_date="", last_date="",
    )


# ---------------------------------------------------------------------------
# The default policy is the one this archive was built with
# ---------------------------------------------------------------------------

def test_default_policy_matches_the_published_archive():
    """
    These numbers produced the consolidated archive that is published. Changing
    one changes what a rebuilt archive says, so a change here should break this
    test and be a deliberate, written-up decision -- the measurement goes in the
    comment beside the value, and the change goes in the GitHub release notes.
    """
    d = DEFAULT_POLICY
    assert (d.domain.lat_min, d.domain.lat_max) == (33.0, 43.0)
    assert (d.domain.lon_min, d.domain.lon_max) == (11.0, 32.0)
    assert d.domain.buffer_deg == 2.0
    assert d.period == (1985, 2005)
    assert d.fetch_years == (1970, 2020)
    assert d.identity.near_deg == 0.027
    assert d.identity.coincident_deg == 0.002
    assert d.identity.correlation_threshold == 0.995
    assert d.identity.min_overlap_days == 365
    assert d.identity.elevation_tolerance_m == 50.0
    assert d.identity.max_mean_abs_diff["degC"] == 0.5
    assert d.fallback.min_overlap_days == 180
    assert d.fallback.min_days_per_month == 20
    assert d.fallback.dispersion_range == (0.75, 1.33)
    assert d.fallback.max_wet_day_diff == 0.10
    assert d.homogeneity.min_neighbors == 2
    assert d.homogeneity.min_snht_statistic == 20.0
    assert d.source_priority["hurs"] == ("ecad",)
    assert d.source_priority["gust"][0] == "gsod"
    assert d.source_priority["pr"].index("ghcnd") < d.source_priority["pr"].index("gsod")


def test_policy_is_immutable():
    with pytest.raises(Exception):
        DEFAULT_POLICY.period = (1990, 2000)       # type: ignore[misc]
    with pytest.raises(TypeError):
        DEFAULT_POLICY.identity.max_mean_abs_diff["degC"] = 99.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# Overriding it, which is what makes the tool usable on another region
# ---------------------------------------------------------------------------

def test_a_partial_policy_file_inherits_everything_it_omits():
    iberia = Policy.from_dict(
        {"domain": {"lat_min": 36.0, "lat_max": 44.0, "lon_min": -10.0,
                    "lon_max": 4.0, "buffer_deg": 2.0},
         "period": [1990, 2010]}
    )
    assert iberia.domain.lon_min == -10.0
    assert iberia.period == (1990, 2010)
    # every threshold is inherited, not silently reset
    assert iberia.identity == DEFAULT_POLICY.identity
    assert iberia.fallback == DEFAULT_POLICY.fallback
    assert iberia.source_priority == DEFAULT_POLICY.source_priority


def test_policy_round_trips_through_json(tmp_path):
    path = DEFAULT_POLICY.to_json(tmp_path / "policy.json")
    assert Policy.from_json(path) == DEFAULT_POLICY


def test_policy_to_dict_is_json_serialisable():
    json.dumps(DEFAULT_POLICY.to_dict())        # must not raise


def test_a_threshold_can_be_overridden_alone():
    strict = Policy.from_dict({"identity": {"near_deg": 0.009}})
    assert strict.identity.near_deg == 0.009
    assert strict.identity.correlation_threshold == 0.995   # untouched
    assert strict.domain == DEFAULT_POLICY.domain


# ---------------------------------------------------------------------------
# The consistency check that did not exist before
# ---------------------------------------------------------------------------

def test_priority_naming_an_unknown_source_is_refused():
    with pytest.raises(registry.PolicyConsistencyError, match="unknown source"):
        replace(DEFAULT_POLICY, source_priority={"pr": ("nope",)})


def test_priority_naming_an_audit_only_source_is_refused():
    """
    c3s_land redelivers GHCN-D, so it is matched and audited but never selected.
    Listing it in a priority order used to be accepted in silence.
    """
    with pytest.raises(registry.PolicyConsistencyError, match="not selectable"):
        replace(DEFAULT_POLICY, source_priority={"pr": ("c3s_land",)})


def test_priority_repeating_a_source_is_refused():
    with pytest.raises(registry.PolicyConsistencyError, match="repeats"):
        replace(DEFAULT_POLICY, source_priority={"pr": ("ecad", "ecad")})


def test_priority_for_an_unlisted_variable_is_refused():
    with pytest.raises(ValueError, match="absent from"):
        replace(DEFAULT_POLICY, variables=("tas",))    # priority still names pr etc.


def test_selecting_a_variable_with_no_gross_range_is_refused():
    """
    dew_point, tavg_unspecified and wind_max_sustained are carried through the
    silver layer but have no qc.GROSS_RANGE entry, so consolidating one would
    write values past no range check at all -- the check that exists because
    GHCN-D carries 999.0 degC as an undocumented sentinel.

    A --policy file naming one used to be accepted in silence, which is the
    same class of failure the source-priority checks above were added for.
    """
    with pytest.raises(ValueError, match="no gross-range"):
        Policy.from_dict({
            "variables": ["tas", "dew_point"],
            "source_priority": {"tas": ["ecad", "gsod"], "dew_point": ["gsod"]},
        })


def test_the_published_policy_selects_only_range_checked_variables():
    """The rule above, asserted against the policy this archive was built with."""
    from station_archive import qc

    assert set(DEFAULT_POLICY.source_priority) <= qc.gated_variables()


def test_a_variable_may_be_declared_without_being_selected():
    """
    The refusal is about SELECTION, not about the vocabulary. Carrying a
    variable in 'variables' so the inventory reports it, while selecting
    nothing from it, stays legal -- that is how dew_point reaches the silver
    layer today.
    """
    policy = replace(
        DEFAULT_POLICY,
        variables=DEFAULT_POLICY.variables + ("dew_point",),
    )
    assert "dew_point" in policy.variables
    assert "dew_point" not in policy.source_priority


def test_reversed_period_is_refused():
    with pytest.raises(ValueError, match="period start after end"):
        replace(DEFAULT_POLICY, period=(2005, 1985))


# ---------------------------------------------------------------------------
# The registry replaces seven unsynchronised lists
# ---------------------------------------------------------------------------

def test_registry_declares_every_source_once():
    assert set(registry.all_sources()) == {"ghcnd", "ecad", "gsod", "c3s_land"}


def test_c3s_joins_the_crosswalk_but_is_neither_readable_nor_selectable():
    spec = registry.spec("c3s_land")
    assert spec.joins_crosswalk
    assert not spec.has_station_series
    assert not spec.selectable
    assert "c3s_land" in registry.crosswalk_sources()
    assert "c3s_land" not in registry.series_sources()
    assert "c3s_land" not in registry.selectable_sources()


def test_unknown_source_names_fail_loudly():
    with pytest.raises(registry.UnknownSourceError):
        registry.adapter_for("nope")


def test_adapters_declare_their_own_units():
    """
    Units come from the adapter, so the three duplicated per-source lookup
    ladders that could drift from it are gone.
    """
    assert provider_units("ghcnd", "tasmax") == "degC_tenths"
    assert provider_units("ecad", "pr") == "mm_tenths"
    assert provider_units("gsod", "tas") == "degF"
    assert provider_units("gsod", "sfcWind") == "knot"
    # C3S declares units per row, and is never read for a series
    assert provider_units("c3s_land", "tasmax") == ""


# ---------------------------------------------------------------------------
# Candidate generation: the guarantee the docstring asserts must be true
# ---------------------------------------------------------------------------

def test_east_west_pairs_inside_the_gate_are_always_candidates():
    """
    Bucketing raw degrees on both axes made the east-west bucket narrower than
    the gate -- 2.37 km against 3.00 km at 38N -- so a pair inside the gate
    could land two buckets apart and never be tested at all.
    """
    tolerance = DEFAULT_POLICY.identity.near_deg
    gate_km = _degrees_to_km(tolerance, DEFAULT_POLICY)
    for lat in (33.0, 38.0, 43.0, 45.0):
        km_per_degree = haversine_km(lat, 0.0, lat, 1.0)
        separation = (gate_km - 1e-6) / km_per_degree
        for step in range(60):
            base = 21.0 + step * tolerance / 60.0
            pairs = candidate_pairs(
                [_station("ecad", "A", lat, base),
                 _station("ghcnd", "B", lat, base + separation)],
                tolerance,
                DEFAULT_POLICY,
            )
            assert len(pairs) == 1, f"missed an in-gate pair at lat {lat}, lon {base}"


def test_north_south_pairs_inside_the_gate_are_always_candidates():
    tolerance = DEFAULT_POLICY.identity.near_deg
    gate_km = _degrees_to_km(tolerance, DEFAULT_POLICY)
    km_per_degree = haversine_km(38.0, 0.0, 39.0, 0.0)
    separation = (gate_km - 1e-6) / km_per_degree
    for step in range(60):
        base = 37.0 + step * tolerance / 60.0
        pairs = candidate_pairs(
            [_station("ecad", "A", base, 24.0),
             _station("ghcnd", "B", base + separation, 24.0)],
            tolerance,
            DEFAULT_POLICY,
        )
        assert len(pairs) == 1


def test_the_parma_pair_is_a_candidate():
    """
    Regression on a real miss: ecad:001729 and gsod:16130099999, both PARMA,
    2.859 km apart, never reached the evidence table because of the bucketing.
    """
    pairs = candidate_pairs(
        [_station("ecad", "001729", 44.80416666666667, 10.315),
         _station("gsod", "16130099999", 44.8166666, 10.2833333)],
        DEFAULT_POLICY.identity.near_deg,
        DEFAULT_POLICY,
    )
    assert len(pairs) == 1


def test_pairs_beyond_the_gate_are_still_excluded():
    """The fix must not turn the blocking into a full quadratic comparison."""
    far = candidate_pairs(
        [_station("ecad", "A", 38.0, 24.0), _station("ghcnd", "B", 38.0, 25.0)],
        DEFAULT_POLICY.identity.near_deg,
        DEFAULT_POLICY,
    )
    assert far == []


def test_longitude_scale_uses_the_domain_edge_not_its_middle():
    """
    Scaling by the middle latitude would leave the guarantee false at the edge,
    which is exactly where the real miss happened.
    """
    scale = longitude_scale(DEFAULT_POLICY)
    edge = DEFAULT_POLICY.domain.lat_max + DEFAULT_POLICY.domain.buffer_deg
    assert scale == pytest.approx(math.cos(math.radians(edge)), abs=1e-9)
    mid = (DEFAULT_POLICY.domain.lat_min + DEFAULT_POLICY.domain.lat_max) / 2
    assert scale < math.cos(math.radians(mid))


def test_longitude_scale_survives_a_polar_domain():
    polar = replace(DEFAULT_POLICY, domain=Domain(85.0, 89.0, 0.0, 30.0, 0.0))
    assert longitude_scale(polar) > 0.0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_provenance_record_describes_policy_and_libraries(tmp_path, monkeypatch):
    from station_archive import manifests

    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    record = manifests.build_provenance(DEFAULT_POLICY, registry.all_sources())
    assert record["policy"]["period"] == [1985, 2005]
    assert "numpy" in record["libraries"] and "pandas" in record["libraries"]
    assert record["station_archive_version"]
    # a source never acquired is reported as such rather than omitted
    assert record["sources"]["ghcnd"] == {"status": "not acquired"}
    json.dumps(record)


def test_provenance_path_sits_with_the_other_manifests(tmp_path, monkeypatch):
    from station_archive import manifests

    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    assert manifests.provenance_path() == tmp_path / "manifests" / "provenance.json"


# ---------------------------------------------------------------------------
# The demo, which is the one path a reader can run with no network
# ---------------------------------------------------------------------------

def test_demo_runs_the_whole_chain(tmp_path, capsys):
    """
    A smoke test over the real adapters, crosswalk, consolidator and breakpoint
    diagnosis, on synthetic files in the providers' own formats. Small and fast
    on purpose; it guards the demo staying runnable, which is what a reader
    meets first.
    """
    from station_archive import demo

    assert demo.run(root=tmp_path, n_stations=6, n_years=6, keep=True) == 0
    out = capsys.readouterr().out
    assert "planted duplicate" in out
    assert "merged: YES" in out
    assert (tmp_path / "raw" / "ecad").exists()
    assert (tmp_path / "inventory" / "ecad_inventory.csv").exists()


# ---------------------------------------------------------------------------
# The climatology reference: an input that changes the archive, so provenance
# must say which one was used
# ---------------------------------------------------------------------------

def test_reference_record_identifies_each_file_by_checksum(tmp_path, monkeypatch):
    import hashlib

    from station_archive import manifests

    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    reference = tmp_path / "cerra.parquet"
    reference.write_bytes(b"reference bytes")
    screen = tmp_path / "screen.csv"
    screen.write_bytes(b"alias,variable\n")
    record = manifests.record_reference(
        {"reference": reference, "reference_static": None, "screen": screen}
    )
    assert record["reference"]["file"] == "cerra.parquet"
    assert record["reference"]["bytes"] == len(b"reference bytes")
    assert record["reference"]["sha256"] == hashlib.sha256(b"reference bytes").hexdigest()
    assert record["reference_static"] == {"status": "not given"}
    assert record["screen"]["sha256"] == hashlib.sha256(b"alias,variable\n").hexdigest()
    # written where provenance will find it, and read back unchanged
    assert manifests.read_reference_record() == record


def test_provenance_carries_the_reference_record(tmp_path, monkeypatch):
    from station_archive import manifests

    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    assert manifests.build_provenance(DEFAULT_POLICY, [])["climatology_reference"] == {
        "status": "not used"
    }
    reference = tmp_path / "cerra.parquet"
    reference.write_bytes(b"x")
    manifests.record_reference({"reference": reference})
    record = manifests.build_provenance(DEFAULT_POLICY, [])
    assert record["climatology_reference"]["reference"]["file"] == "cerra.parquet"
    json.dumps(record)


def test_reference_record_never_stores_an_absolute_path(tmp_path, monkeypatch):
    """A provenance record is shared; a local path in it says nothing to anyone else."""
    from station_archive import manifests

    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    reference = tmp_path / "deep" / "cerra.parquet"
    reference.parent.mkdir()
    reference.write_bytes(b"x")
    text = json.dumps(manifests.record_reference({"reference": reference}))
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
