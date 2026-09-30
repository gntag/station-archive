"""
dem.py
======
Resolve elevation disagreements against an independent digital elevation model.

THE PROBLEM THIS SETTLES
------------------------
Two archives routinely publish different heights for one station, and until
now the pipeline only ever *diagnosed* that: identity.py proves from the
observations that two records are one instrument, notices their metadata
cannot both be right, and writes the pair to metadata_conflicts.csv. Section
1.4 forbids silently homogenising, so nothing resolved it, and the canonical
elevation stayed whichever alias happened to have the most observations.

That is not good enough for elevation specifically, because elevation is a
model predictor (delta_elev_m): a station carrying the wrong height corrupts
the fitted relationship however its identity is resolved. A 113 m error, or
the 564 m one in this archive, is not a rounding difference.

So where the archives disagree by more than the policy threshold, a third,
independent source breaks the tie.

THE SOURCE
----------
AWS Terrain Tiles -- the 'elevation-tiles-prod' bucket, Terrarium encoding --
which is the same tileset the parent downscaling project reads in
raster/delos_mapzen_s2coast.py. Terrarium packs metres into RGB:

    elevation_m = R * 256 + G + B / 256 - 32768

A caution worth carrying: zoom 15 gives ~4.8 m pixels at these latitudes, but
the underlying data over Europe is ~30 m SRTM/EU-DEM, so the extra detail is
interpolation, not information. It is ample for deciding between candidates
tens or hundreds of metres apart, which is all it is used for.

WHAT IT DOES AND DOES NOT DECIDE
--------------------------------
It chooses among the elevations the archives ACTUALLY PUBLISHED. It never
writes the DEM's own reading into the archive: section 1.3 selects metadata
from what a provider asserted and never synthesises a value, and a DEM pixel
is not a station's recorded height.

The nearest candidate always wins, so the recorded reason carries the gap
between the DEM and the winner. A large gap means the DEM endorsed nothing and
the choice rests on relative proximity alone -- read the reason before trusting
such a row.

WHEN A LARGE GAP MEANS THE COORDINATE IS WRONG, NOT THE ELEVATION
-----------------------------------------------------------------
Measured on the first run over this archive, and worth knowing before reading
any gap: of 25 overrides, the 9 with a gap above 20 m ALL sat at coordinates
rounded to a whole arcminute, and none of the 12 precisely-located stations
produced one. Median gap was 2.6 m with a precise coordinate and 36.7 m with a
coarse one.

A whole-arcminute coordinate -- ECA&D publishes degrees:minutes:seconds and
many stations carry zero seconds -- is +/-900 m on the ground at this latitude.
On a coastal cliff or a mountainside the pixel sampled is then simply not the
station's pixel, and the DEM is answering a question about somewhere nearby.
So a large gap is evidence about the COORDINATE, not about which elevation is
right, and such a row deserves a human before it is trusted.

Overrides are applied at READ time by whatever consumes the archive, never
written back into the consolidated table, so the archive stays reproducible
from its snapshots and two consumers may disagree about corrections without
forking the data.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from station_archive import net, settings
from station_archive.policy import DEFAULT_POLICY, Policy

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_PX = 256

# Decoded tiles, per process. The on-disk cache below survives restarts; this
# one avoids re-decoding a PNG for every station sharing a tile.
#
# Keyed on the cache directory as well as the tile, because the archive root is
# resolved at call time and can change within one process -- the demo builds a
# whole archive under a temporary root. Keying on the tile alone would serve
# one archive's tiles to another.
_TILE_CACHE: dict[tuple[str, int, int, int], np.ndarray] = {}


def cache_dir() -> Path:
    """Where fetched tiles are kept, so a re-run costs no network."""
    return settings.archive_root() / "cache" / "dem"


# ---------------------------------------------------------------------------
# Terrarium encoding and slippy-map geometry
# ---------------------------------------------------------------------------

def terrarium_decode(rgb: np.ndarray) -> np.ndarray:
    """Decode a Terrarium RGB array to metres."""
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    return r * 256.0 + g + b / 256.0 - 32768.0


def lon_lat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Slippy-map tile containing a coordinate (Web Mercator, XYZ scheme)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return x, y


def tile_pixel(lon: float, lat: float, zoom: int) -> tuple[int, int, int, int]:
    """
    Tile index and the pixel inside it for a coordinate.

    Returns (tile_x, tile_y, px, py). The pixel matters: a zoom-15 tile spans
    about a kilometre, so reading the tile's corner instead of the station's
    own pixel would answer a question about somewhere else.
    """
    n = 2 ** zoom
    fx = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    fy = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    x, y = int(fx), int(fy)
    px = min(int((fx - x) * TILE_PX), TILE_PX - 1)
    py = min(int((fy - y) * TILE_PX), TILE_PX - 1)
    return x, y, px, py


# ---------------------------------------------------------------------------
# Tile fetching
# ---------------------------------------------------------------------------

def _tile_path(x: int, y: int, zoom: int) -> Path:
    return cache_dir() / str(zoom) / str(x) / f"{y}.png"


def _decode_png(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGB"))


def load_tile(x: int, y: int, zoom: int) -> Optional[np.ndarray]:
    """
    One decoded elevation tile, from cache where possible.

    Returns None when the tile cannot be had. A single unreachable tile must
    not abort adjudication of every other station, so the failure is local.
    """
    path = _tile_path(x, y, zoom)
    key = (str(cache_dir()), zoom, x, y)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]

    data: Optional[bytes] = None
    if path.exists():
        data = path.read_bytes()
    else:
        try:
            data = net.get_bytes(TILE_URL.format(z=zoom, x=x, y=y))
        except Exception:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError:
            pass                    # an uncacheable tile is still usable now

    try:
        elevation = terrarium_decode(_decode_png(data))
    except Exception:
        return None
    _TILE_CACHE[key] = elevation
    return elevation


def elevation_at(
    lat: float,
    lon: float,
    zoom: Optional[int] = None,
    policy: Policy = DEFAULT_POLICY,
) -> float:
    """
    DEM elevation at one coordinate, in metres, or NaN if it cannot be read.
    """
    zoom = policy.dem.zoom if zoom is None else zoom
    if pd.isna(lat) or pd.isna(lon):
        return float("nan")
    x, y, px, py = tile_pixel(float(lon), float(lat), zoom)
    tile = load_tile(x, y, zoom)
    if tile is None:
        return float("nan")
    return float(tile[py, px])


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

def adjudicate(
    candidates: dict[str, float],
    dem_elevation: float,
) -> tuple[Optional[str], float]:
    """
    Which published elevation the DEM sits closest to, and by how much.

    Returns (alias_key, gap_m). The gap is what makes a weak decision visible:
    the nearest candidate always wins, so only the gap distinguishes "the DEM
    confirms this one" from "the DEM agrees with neither, and this one is
    merely less wrong".

    Ties break on the alias key, so a re-run cannot reorder them.
    """
    usable = {k: float(v) for k, v in candidates.items() if not pd.isna(v)}
    if not usable or pd.isna(dem_elevation):
        return None, float("inf")
    key = min(sorted(usable), key=lambda k: abs(usable[k] - dem_elevation))
    return key, abs(usable[key] - dem_elevation)


def build_elevation_overrides(
    aliases: pd.DataFrame,
    canonical: pd.DataFrame,
    policy: Policy = DEFAULT_POLICY,
    existing: Optional[pd.DataFrame] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """
    An elevation override per canonical station whose aliases disagree.

    A station qualifies when the spread between its aliases' published
    elevations exceeds policy.dem.elevation_disagreement_m. The DEM is read at
    the station's canonical position, and the alias elevation nearest to it
    becomes the override.

    Rows already present in 'existing' are carried through untouched: a
    correction a person reviewed outranks one this function would compute, and
    silently replacing it would discard the human decision this whole workbook
    machinery exists to capture.
    """
    if aliases.empty or canonical.empty:
        return pd.DataFrame(columns=["canonical_station_id", "elevation_m", "reason"])

    reviewed = (
        existing.copy() if existing is not None and not existing.empty
        else pd.DataFrame(columns=["canonical_station_id", "elevation_m", "reason"])
    )
    already = set(reviewed.get("canonical_station_id", []))

    working = aliases.copy()
    working["elev"] = pd.to_numeric(working["elevation_m"], errors="coerce")
    working["key"] = working["source"] + ":" + working["source_station_id"].astype(str)
    meta = canonical.set_index("canonical_station_id")

    rows: list[dict] = []
    grouped = working.dropna(subset=["elev"]).groupby("canonical_station_id")
    for index, (canonical_id, group) in enumerate(grouped, start=1):
        if canonical_id in already or len(group) < 2:
            continue
        spread = float(group["elev"].max() - group["elev"].min())
        if spread <= policy.dem.elevation_disagreement_m:
            continue
        if canonical_id not in meta.index:
            continue

        station = meta.loc[canonical_id]
        reading = elevation_at(station["lat"], station["lon"], policy=policy)
        candidates = dict(zip(group["key"], group["elev"]))
        winner, gap = adjudicate(candidates, reading)
        if winner is None:
            continue                    # no DEM reading: decide nothing

        others = ", ".join(
            f"{k} {v:.0f} m" for k, v in sorted(candidates.items()) if k != winner
        )
        rows.append(
            {
                "canonical_station_id": canonical_id,
                "elevation_m": float(candidates[winner]),
                "reason": (
                    f"elevation disagreement of {spread:.0f} m between aliases; "
                    f"AWS Terrain Tiles (terrarium, zoom {policy.dem.zoom}) reads "
                    f"{reading:.2f} m at ({station['lat']:.4f}, {station['lon']:.4f}), "
                    f"nearest published value is {winner} at "
                    f"{candidates[winner]:.0f} m (gap {gap:.2f} m), "
                    f"in preference to {others}"
                ),
            }
        )
        if progress and index % 10 == 0:
            print(f"[dem] examined {index} stations")

    computed = pd.DataFrame(rows, columns=["canonical_station_id", "elevation_m", "reason"])
    if reviewed.empty:
        return computed
    return pd.concat([reviewed, computed], ignore_index=True)
