import numpy as np
import pandas as pd
import pytest

from observations.archive_ingest import (
    apply_elevation_overrides,
    find_pinnable_duplicate_pairs,
    load_canonical_stations,
    load_consolidated_archive,
)


def _write_partitioned_observations(root, rows):
    """Write one long-format observations parquet part file."""
    (root / "observations_consolidated").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["canonical_station_id", "date", "variable", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(root / "observations_consolidated" / "part-0000.parquet")


def _write_canonical_stations(root, rows):
    (root).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "canonical_stations.csv", index=False)


def test_load_canonical_stations_basic_columns(tmp_path):
    canon_dir = tmp_path / "canonical"
    _write_canonical_stations(canon_dir, [
        {"canonical_station_id": "st1", "name": "A", "lat": 38.0, "lon": 24.0,
         "elevation_m": 10.0, "duplicate_risk": False},
    ])
    df = load_canonical_stations(canon_dir)
    assert set(["canonical_station_id", "lat", "lon", "elevation_m", "duplicate_risk"]).issubset(df.columns)
    assert df.loc[0, "elevation_m"] == 10.0


def test_apply_elevation_overrides_replaces_matching_row():
    stations = pd.DataFrame({
        "canonical_station_id": ["st1", "st2"],
        "elevation_m": [52.0, 100.0],
    })
    overrides = pd.DataFrame({
        "canonical_station_id": ["st1"],
        "elevation_m": [165.0],
        "reason": ["cerra orography check"],
    })
    out = apply_elevation_overrides(stations, overrides)
    assert out.set_index("canonical_station_id").loc["st1", "elevation_m"] == 165.0
    assert out.set_index("canonical_station_id").loc["st2", "elevation_m"] == 100.0


def test_apply_elevation_overrides_unknown_station_is_noop():
    stations = pd.DataFrame({"canonical_station_id": ["st1"], "elevation_m": [52.0]})
    overrides = pd.DataFrame({
        "canonical_station_id": ["st_missing"], "elevation_m": [999.0], "reason": ["x"],
    })
    out = apply_elevation_overrides(stations, overrides)
    assert out.set_index("canonical_station_id").loc["st1", "elevation_m"] == 52.0


def test_load_consolidated_archive_pivots_to_wide_schema(tmp_path):
    consolidated_dir = tmp_path / "consolidated"
    _write_partitioned_observations(consolidated_dir, [
        ("st1", "1990-01-01", "tas", 20.0),
        ("st1", "1990-01-01", "tasmin", 15.0),
        ("st1", "1990-01-01", "tasmax", 25.0),
        ("st1", "1990-01-01", "hurs", 60.0),
        ("st1", "1990-01-01", "pr", 0.0),
        ("st1", "1990-01-01", "sfcWind", 5.0),
        ("st1", "1990-01-01", "gust", 9.0),
        ("st1", "1990-01-02", "tas", 21.0),
    ])
    canon_dir = tmp_path / "canonical"
    _write_canonical_stations(canon_dir, [
        {"canonical_station_id": "st1", "name": "STATION ONE", "lat": 38.0, "lon": 24.0,
         "elevation_m": 10.0, "duplicate_risk": False},
    ])
    stations_meta = load_canonical_stations(canon_dir)

    df = load_consolidated_archive(consolidated_dir, stations_meta, year_start=1990, year_end=1990)

    expected_cols = {"station_id", "date", "lat", "lon", "elevation_m", "name",
                      "tmean_c", "tmin_c", "tmax_c", "rh_pct", "prcp_mm", "wdsp_ms", "gust_ms"}
    assert expected_cols.issubset(df.columns)
    row = df[df["date"] == pd.Timestamp("1990-01-01")].iloc[0]
    assert row["tmean_c"] == 20.0
    assert row["tmin_c"] == 15.0
    assert row["tmax_c"] == 25.0
    assert row["rh_pct"] == 60.0
    assert row["prcp_mm"] == 0.0
    assert row["wdsp_ms"] == 5.0
    assert row["gust_ms"] == 9.0
    assert len(df) == 2  # two distinct dates for st1


def test_load_consolidated_archive_filters_year_range(tmp_path):
    consolidated_dir = tmp_path / "consolidated"
    _write_partitioned_observations(consolidated_dir, [
        ("st1", "1985-01-01", "tas", 20.0),
        ("st1", "2010-01-01", "tas", 30.0),  # outside range
    ])
    canon_dir = tmp_path / "canonical"
    _write_canonical_stations(canon_dir, [
        {"canonical_station_id": "st1", "name": "A", "lat": 38.0, "lon": 24.0,
         "elevation_m": 10.0, "duplicate_risk": False},
    ])
    stations_meta = load_canonical_stations(canon_dir)
    df = load_consolidated_archive(consolidated_dir, stations_meta, year_start=1985, year_end=2005)
    assert len(df) == 1
    assert df.iloc[0]["date"] == pd.Timestamp("1985-01-01")


def test_find_pinnable_duplicate_pairs_resolves_aliases_and_excludes_never_merge():
    duplicate_evidence = pd.DataFrame([
        {"left": "ecad:001", "right": "gsod:001", "accepted": False,
         "reason": "distance 3.5 km exceeds tolerance"},
        {"left": "ecad:002", "right": "gsod:002", "accepted": False,
         "reason": "no overlapping days in any variable; never merge (project rule)"},
        {"left": "ecad:003", "right": "gsod:003", "accepted": True,
         "reason": "exact identifier"},
    ])
    station_aliases = pd.DataFrame([
        {"source": "ecad", "source_station_id": "001", "canonical_station_id": "canon_A"},
        {"source": "gsod", "source_station_id": "001", "canonical_station_id": "canon_B"},
        {"source": "ecad", "source_station_id": "002", "canonical_station_id": "canon_C"},
        {"source": "gsod", "source_station_id": "002", "canonical_station_id": "canon_D"},
        {"source": "ecad", "source_station_id": "003", "canonical_station_id": "canon_E"},
        {"source": "gsod", "source_station_id": "003", "canonical_station_id": "canon_E"},
    ])
    canonical_ids = {"canon_A", "canon_B", "canon_C", "canon_D", "canon_E"}

    pairs = find_pinnable_duplicate_pairs(duplicate_evidence, station_aliases, canonical_ids)

    resolved = set(zip(pairs["a"], pairs["b"])) | set(zip(pairs["b"], pairs["a"]))
    assert ("canon_A", "canon_B") in resolved
    assert not any({"canon_C", "canon_D"} <= {x, y} for x, y in resolved)  # never-merge excluded
    assert not any({"canon_E"} <= {x, y} for x, y in resolved)  # already same canonical id
