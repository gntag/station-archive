"""
Tests for DEM-based adjudication of elevation disagreements.

Two archives publishing different heights for one station is common, and until
now it was only ever diagnosed -- metadata_conflicts.csv recorded the
disagreement and nothing resolved it. These tests cover resolving it against an
independent source: the AWS Terrain Tiles ('elevation-tiles-prod') in Terrarium
encoding, the same tileset the parent downscaling project reads.

No test touches the network. The tile fetcher is substituted throughout.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from station_archive import settings
from station_archive import dem
from station_archive.policy import DEFAULT_POLICY


@pytest.fixture(autouse=True)
def _isolate_tile_cache():
    """No decoded tile may leak from one test into the next."""
    dem._TILE_CACHE.clear()
    yield
    dem._TILE_CACHE.clear()


def _tile_png(fill_rgb=(128, 0, 0), overrides: dict | None = None) -> bytes:
    """A 256x256 Terrarium tile, optionally with specific pixels set."""
    arr = np.zeros((dem.TILE_PX, dem.TILE_PX, 3), dtype=np.uint8)
    arr[:, :] = fill_rgb
    for (px, py), rgb in (overrides or {}).items():
        arr[py, px] = rgb
    buffer = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def served_tiles(monkeypatch, tmp_path):
    """Serve a constant tile, counting fetches so caching can be observed."""
    calls = {"n": 0}

    def fake_get(url, byte_range=None):
        calls["n"] += 1
        return _tile_png((128, 100, 0))          # 100 m everywhere

    monkeypatch.setattr(dem.net, "get_bytes", fake_get)
    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    return calls


# ---------------------------------------------------------------------------
# Terrarium encoding
# ---------------------------------------------------------------------------

def test_terrarium_decodes_the_documented_formula():
    """elevation = R*256 + G + B/256 - 32768, in metres."""
    arr = np.array([[[128, 0, 0], [128, 10, 128], [127, 246, 0]]], dtype=np.uint8)
    out = dem.terrarium_decode(arr)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(10.5)
    assert out[0, 2] == pytest.approx(-10.0)


def test_terrarium_handles_below_sea_level():
    arr = np.array([[[127, 0, 0]]], dtype=np.uint8)
    assert dem.terrarium_decode(arr)[0, 0] == pytest.approx(-256.0)


# ---------------------------------------------------------------------------
# Tile geometry -- must match the reference implementation in the parent repo
# ---------------------------------------------------------------------------

def test_tile_index_matches_the_slippy_map_convention():
    def reference(lon, lat, z):
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        lr = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n)
        return x, y

    for lon, lat in [(25.27, 37.40), (23.7348, 37.8822), (-10.0, 44.0), (0.0, 0.0)]:
        assert dem.lon_lat_to_tile(lon, lat, 15) == reference(lon, lat, 15)


def test_pixel_within_tile_is_in_range():
    for lon, lat in [(25.27, 37.40), (23.7348, 37.8822), (11.0, 33.0)]:
        x, y, px, py = dem.tile_pixel(lon, lat, 15)
        assert 0 <= px < dem.TILE_PX
        assert 0 <= py < dem.TILE_PX
        assert (x, y) == dem.lon_lat_to_tile(lon, lat, 15)


def test_neighbouring_points_land_in_the_same_tile():
    """A zoom-15 tile spans ~1 km, so two points 100 m apart usually share one."""
    a = dem.lon_lat_to_tile(23.7348, 37.8822, 15)
    b = dem.lon_lat_to_tile(23.7358, 37.8822, 15)
    assert a == b


# ---------------------------------------------------------------------------
# Lookup and caching
# ---------------------------------------------------------------------------

def test_elevation_at_returns_the_decoded_value(served_tiles):
    assert dem.elevation_at(37.8822, 23.7348) == pytest.approx(100.0)


def test_elevation_reads_the_right_pixel(monkeypatch, tmp_path):
    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    lat, lon = 37.8822, 23.7348
    _, _, px, py = dem.tile_pixel(lon, lat, 15)
    png = _tile_png((128, 0, 0), {(px, py): (128, 200, 0)})   # 200 m at that pixel only
    monkeypatch.setattr(dem.net, "get_bytes", lambda url, byte_range=None: png)
    assert dem.elevation_at(lat, lon) == pytest.approx(200.0)


def test_a_tile_is_fetched_once_and_then_cached(served_tiles):
    dem.elevation_at(37.8822, 23.7348)
    dem.elevation_at(37.8823, 23.7349)          # same tile
    assert served_tiles["n"] == 1


def test_the_cache_survives_a_new_process(served_tiles, monkeypatch):
    dem.elevation_at(37.8822, 23.7348)
    dem._TILE_CACHE.clear()                     # as if restarted
    dem.elevation_at(37.8822, 23.7348)
    assert served_tiles["n"] == 1, "the on-disk cache was not used"


def test_a_missing_tile_yields_nan_rather_than_failing(monkeypatch, tmp_path):
    """One unreachable tile must not abort adjudication of every other station."""
    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))

    def refuse(url, byte_range=None):
        raise RuntimeError("404")

    monkeypatch.setattr(dem.net, "get_bytes", refuse)
    assert math.isnan(dem.elevation_at(37.8822, 23.7348))


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

def test_the_candidate_nearest_the_dem_wins():
    chosen, gap = dem.adjudicate({"ecad:1": 52.0, "ghcnd:A": 165.0}, 164.03)
    assert chosen == "ghcnd:A"
    assert gap == pytest.approx(0.97, abs=0.01)


def test_adjudication_reports_the_gap_even_when_it_is_large():
    """
    The nearest candidate always wins, so the gap is the only thing telling a
    reader the DEM endorsed nothing. It has to be recorded.
    """
    chosen, gap = dem.adjudicate({"ecad:1": 52.0, "ghcnd:A": 165.0}, 400.0)
    assert chosen == "ghcnd:A"
    assert gap == pytest.approx(235.0)


def test_adjudication_is_deterministic_on_a_tie():
    chosen, _ = dem.adjudicate({"b:1": 90.0, "a:1": 110.0}, 100.0)
    assert chosen == "a:1"                      # tie broken on the alias key


def test_adjudication_without_a_dem_reading_decides_nothing():
    assert dem.adjudicate({"ecad:1": 52.0}, float("nan")) == (None, float("inf"))


# ---------------------------------------------------------------------------
# Building the override table
# ---------------------------------------------------------------------------

def _aliases(rows):
    base = {"canonical_station_id": "", "source": "", "source_station_id": "",
            "name": "X", "lat": 37.9, "lon": 23.7, "elevation_m": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _canonical(rows):
    base = {"canonical_station_id": "", "name": "X", "lat": 37.9, "lon": 23.7,
            "elevation_m": 0.0, "duplicate_risk": False}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_only_stations_disagreeing_by_more_than_the_threshold_are_adjudicated(served_tiles):
    aliases = _aliases([
        {"canonical_station_id": "c:1", "source": "ecad", "source_station_id": "1",
         "elevation_m": 52.0},
        {"canonical_station_id": "c:1", "source": "ghcnd", "source_station_id": "A",
         "elevation_m": 165.0},
        {"canonical_station_id": "c:2", "source": "ecad", "source_station_id": "2",
         "elevation_m": 40.0},
        {"canonical_station_id": "c:2", "source": "ghcnd", "source_station_id": "B",
         "elevation_m": 44.0},          # only 4 m apart
    ])
    canonical = _canonical([
        {"canonical_station_id": "c:1", "elevation_m": 52.0},
        {"canonical_station_id": "c:2", "elevation_m": 40.0},
    ])
    out = dem.build_elevation_overrides(aliases, canonical, DEFAULT_POLICY)
    assert list(out.canonical_station_id) == ["c:1"]


def test_the_override_records_the_dem_evidence(served_tiles):
    aliases = _aliases([
        {"canonical_station_id": "c:1", "source": "ecad", "source_station_id": "1",
         "elevation_m": 52.0},
        {"canonical_station_id": "c:1", "source": "ghcnd", "source_station_id": "A",
         "elevation_m": 101.0},
    ])
    canonical = _canonical([{"canonical_station_id": "c:1", "elevation_m": 52.0}])
    out = dem.build_elevation_overrides(aliases, canonical, DEFAULT_POLICY)
    row = out.iloc[0]
    assert row.elevation_m == 101.0             # nearest to the 100 m tile
    assert set(out.columns) >= {"canonical_station_id", "elevation_m", "reason"}
    assert "100" in row.reason                  # the DEM value
    assert "terrarium" in row.reason.lower() or "terrain" in row.reason.lower()
    assert "52" in row.reason                   # the candidate it beat


def test_a_station_whose_dem_lookup_failed_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setenv(settings.ARCHIVE_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(dem.net, "get_bytes",
                        lambda url, byte_range=None: (_ for _ in ()).throw(RuntimeError()))
    aliases = _aliases([
        {"canonical_station_id": "c:1", "source": "ecad", "source_station_id": "1",
         "elevation_m": 52.0},
        {"canonical_station_id": "c:1", "source": "ghcnd", "source_station_id": "A",
         "elevation_m": 165.0},
    ])
    canonical = _canonical([{"canonical_station_id": "c:1", "elevation_m": 52.0}])
    out = dem.build_elevation_overrides(aliases, canonical, DEFAULT_POLICY)
    assert out.empty


def test_existing_overrides_are_preserved(served_tiles):
    """A correction already reviewed by a person is not silently replaced."""
    aliases = _aliases([
        {"canonical_station_id": "c:1", "source": "ecad", "source_station_id": "1",
         "elevation_m": 52.0},
        {"canonical_station_id": "c:1", "source": "ghcnd", "source_station_id": "A",
         "elevation_m": 165.0},
    ])
    canonical = _canonical([{"canonical_station_id": "c:1", "elevation_m": 52.0}])
    existing = pd.DataFrame([{"canonical_station_id": "c:1", "elevation_m": 999.0,
                              "reason": "decided by hand"}])
    out = dem.build_elevation_overrides(aliases, canonical, DEFAULT_POLICY,
                                        existing=existing)
    row = out[out.canonical_station_id == "c:1"].iloc[0]
    assert row.elevation_m == 999.0
    assert row.reason == "decided by hand"


def test_single_alias_stations_are_never_adjudicated(served_tiles):
    aliases = _aliases([{"canonical_station_id": "c:1", "source": "ecad",
                         "source_station_id": "1", "elevation_m": 52.0}])
    canonical = _canonical([{"canonical_station_id": "c:1", "elevation_m": 52.0}])
    assert dem.build_elevation_overrides(aliases, canonical, DEFAULT_POLICY).empty


def test_the_override_is_always_a_published_height_never_the_dem_reading(served_tiles):
    """
    The DEM's job is to say which archive is closer to right, not to supply the
    number. Section 1.3 selects metadata from what a provider asserted, and a
    DEM pixel is not a station's recorded height -- so the written value must
    always be one of the candidates, including when the DEM agrees with none of
    them.
    """
    aliases = _aliases([
        {"canonical_station_id": "c:1", "source": "ecad", "source_station_id": "1",
         "elevation_m": 46.0},
        {"canonical_station_id": "c:1", "source": "ecad", "source_station_id": "2",
         "elevation_m": 312.0},
    ])
    canonical = _canonical([{"canonical_station_id": "c:1", "elevation_m": 46.0}])
    out = dem.build_elevation_overrides(aliases, canonical, DEFAULT_POLICY)
    written = out.iloc[0].elevation_m
    assert written in (46.0, 312.0), "the override must be a published height"
    assert written != 100.0, "the DEM reading must never be written as the elevation"
    # the tile reads 100 m, so 46 is the nearer published value
    assert written == 46.0
