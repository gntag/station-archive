import numpy as np
import pandas as pd
import pytest

from observations.fold_manifest import (
    assign_spatial_folds,
    merge_duplicate_risk_groups,
    build_fold_manifest,
    load_fold_manifest,
    save_fold_manifest,
)


def _grid_stations(n_per_cluster=15):
    """Two well-separated clusters of stations, ~5 degrees apart."""
    rng = np.random.default_rng(0)
    lat_a = 35.0 + rng.normal(0, 0.1, n_per_cluster)
    lon_a = 20.0 + rng.normal(0, 0.1, n_per_cluster)
    lat_b = 40.0 + rng.normal(0, 0.1, n_per_cluster)
    lon_b = 28.0 + rng.normal(0, 0.1, n_per_cluster)
    lat = np.concatenate([lat_a, lat_b])
    lon = np.concatenate([lon_a, lon_b])
    return lat, lon


def test_assign_spatial_folds_separates_distant_clusters():
    lat, lon = _grid_stations()
    folds = assign_spatial_folds(lat, lon, n_folds=2, seed=0)
    assert folds.shape == lat.shape
    # every station in the first half should share one fold, second half another
    first_half_fold = set(folds[:15])
    second_half_fold = set(folds[15:])
    assert len(first_half_fold) == 1
    assert len(second_half_fold) == 1
    assert first_half_fold != second_half_fold


def test_assign_spatial_folds_is_deterministic():
    lat, lon = _grid_stations()
    a = assign_spatial_folds(lat, lon, n_folds=2, seed=42)
    b = assign_spatial_folds(lat, lon, n_folds=2, seed=42)
    np.testing.assert_array_equal(a, b)


def test_assign_spatial_folds_respects_n_folds():
    lat, lon = _grid_stations(n_per_cluster=30)
    folds = assign_spatial_folds(lat, lon, n_folds=5, seed=0)
    assert set(folds).issubset(set(range(5)))


def test_merge_duplicate_risk_groups_pins_pair_together():
    station_ids = np.array(["s1", "s2", "s3", "s4"])
    folds = np.array([0, 1, 0, 1])
    pairs = pd.DataFrame({"a": ["s1"], "b": ["s2"]})
    merged = merge_duplicate_risk_groups(folds, station_ids, pairs)
    idx = {sid: i for i, sid in enumerate(station_ids)}
    assert merged[idx["s1"]] == merged[idx["s2"]]
    # unrelated stations untouched
    assert merged[idx["s3"]] == folds[idx["s3"]]


def test_merge_duplicate_risk_groups_propagates_transitively():
    # s1-s2 and s2-s3 pinned => s1, s2, s3 all end up in the same fold
    station_ids = np.array(["s1", "s2", "s3", "s4"])
    folds = np.array([0, 1, 2, 3])
    pairs = pd.DataFrame({"a": ["s1", "s2"], "b": ["s2", "s3"]})
    merged = merge_duplicate_risk_groups(folds, station_ids, pairs)
    idx = {sid: i for i, sid in enumerate(station_ids)}
    assert merged[idx["s1"]] == merged[idx["s2"]] == merged[idx["s3"]]
    assert merged[idx["s4"]] == folds[idx["s4"]]


def test_merge_duplicate_risk_groups_ignores_unknown_ids():
    station_ids = np.array(["s1", "s2"])
    folds = np.array([0, 1])
    pairs = pd.DataFrame({"a": ["s1"], "b": ["s99"]})  # s99 not present
    merged = merge_duplicate_risk_groups(folds, station_ids, pairs)
    np.testing.assert_array_equal(merged, folds)


def _canonical_stations_fixture():
    lat, lon = _grid_stations(n_per_cluster=10)
    ids = [f"st{i}" for i in range(len(lat))]
    return pd.DataFrame({
        "canonical_station_id": ids,
        "lat": lat,
        "lon": lon,
        "elevation_m": 100.0,
        "duplicate_risk": False,
    })


def test_build_fold_manifest_excludes_stations_without_coverage():
    stations = _canonical_stations_fixture()
    covered = set(stations["canonical_station_id"].iloc[:15])  # drop last 5
    manifest = build_fold_manifest(
        stations, pinnable_pairs=pd.DataFrame(columns=["a", "b"]),
        stations_with_coverage=covered, n_spatial_folds=2, seed=0,
    )
    excluded = manifest[~manifest["included"]]
    assert set(excluded["canonical_station_id"]) == set(stations["canonical_station_id"]) - covered
    assert (excluded["exclusion_reason"] == "no_1985_2005_coverage").all()


def test_build_fold_manifest_excludes_unresolved_duplicate_risk():
    stations = _canonical_stations_fixture()
    stations.loc[stations["canonical_station_id"] == "st0", "duplicate_risk"] = True
    covered = set(stations["canonical_station_id"])
    manifest = build_fold_manifest(
        stations, pinnable_pairs=pd.DataFrame(columns=["a", "b"]),
        stations_with_coverage=covered, n_spatial_folds=2, seed=0,
    )
    row = manifest[manifest["canonical_station_id"] == "st0"].iloc[0]
    assert not row["included"]
    assert row["exclusion_reason"] == "duplicate_risk_unresolved_partner"


def test_build_fold_manifest_keeps_pinned_duplicate_risk_pair_included():
    stations = _canonical_stations_fixture()
    stations.loc[stations["canonical_station_id"].isin(["st0", "st1"]), "duplicate_risk"] = True
    covered = set(stations["canonical_station_id"])
    pairs = pd.DataFrame({"a": ["st0"], "b": ["st1"]})
    manifest = build_fold_manifest(
        stations, pinnable_pairs=pairs,
        stations_with_coverage=covered, n_spatial_folds=2, seed=0,
    )
    rows = manifest.set_index("canonical_station_id")
    assert rows.loc["st0", "included"]
    assert rows.loc["st1", "included"]
    assert rows.loc["st0", "spatial_fold"] == rows.loc["st1", "spatial_fold"]


def test_save_and_load_fold_manifest_roundtrip(tmp_path):
    stations = _canonical_stations_fixture()
    covered = set(stations["canonical_station_id"])
    manifest = build_fold_manifest(
        stations, pinnable_pairs=pd.DataFrame(columns=["a", "b"]),
        stations_with_coverage=covered, n_spatial_folds=2, seed=0,
    )
    path = tmp_path / "fold_manifest.parquet"
    save_fold_manifest(manifest, path)
    reloaded = load_fold_manifest(path)
    pd.testing.assert_frame_equal(
        manifest.reset_index(drop=True), reloaded.reset_index(drop=True)
    )
