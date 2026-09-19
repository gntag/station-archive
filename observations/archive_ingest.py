"""
archive_ingest.py
==================
Reads the consolidated archive back out for modelling.

`consolidate` writes the gold table in long format -- one row per
canonical_station_id x date x variable. Most downstream consumers want the
opposite shape: one row per station-day, one column per variable. This
module does that pivot, joins the canonical station metadata onto it, and
applies any reviewed elevation corrections on the way through:

    station_id, date, lat, lon, elevation_m, name,
    tmean_c, tmin_c, tmax_c, rh_pct, prcp_mm, wdsp_ms, gust_ms

The column names are the conventional ones used by the downscaling pipeline
this archive was built for, so the archive can stand in directly for a
single-source station extract.

Nothing here writes to the archive. Corrections are layered on at read time
from a versioned override table rather than edited into the consolidated
output, so the archive stays reproducible from its source snapshots.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# archive variable name -> wide output column name
_VARIABLE_TO_COLUMN = {
    "tas":     "tmean_c",
    "tasmin":  "tmin_c",
    "tasmax":  "tmax_c",
    "hurs":    "rh_pct",
    "pr":      "prcp_mm",
    "sfcWind": "wdsp_ms",
    "gust":    "gust_ms",
}


def load_canonical_stations(canonical_dir: Path) -> pd.DataFrame:
    """
    Load canonical_stations.csv.

    Returns
    -------
    DataFrame with (at least) columns: canonical_station_id, name, lat, lon,
    elevation_m, duplicate_risk.
    """
    path = Path(canonical_dir) / "canonical_stations.csv"
    df = pd.read_csv(path)
    df["duplicate_risk"] = df["duplicate_risk"].astype(bool)
    return df


def apply_elevation_overrides(
    stations: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace elevation_m for any canonical_station_id present in `overrides`
    (columns: canonical_station_id, elevation_m, reason). Stations not named
    in overrides are left untouched. See observations/data/elevation_overrides.csv
    and E:/station_archive/review/station_merge_review.xlsx sheet
    'Elevation_overrides' for the documented Step 2 correction this applies.
    """
    out = stations.copy()
    override_map = overrides.set_index("canonical_station_id")["elevation_m"]
    replace_mask = out["canonical_station_id"].isin(override_map.index)
    out.loc[replace_mask, "elevation_m"] = (
        out.loc[replace_mask, "canonical_station_id"].map(override_map)
    )
    return out


def load_elevation_overrides(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_consolidated_archive(
    consolidated_dir: Path,
    stations_meta: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    Load the partitioned long-format observation archive and pivot to the
    wide per-station-day schema.

    Parameters
    ----------
    consolidated_dir : directory containing observations_consolidated/part-*.parquet
    stations_meta     : from load_canonical_stations() (optionally with
                        apply_elevation_overrides applied); must have columns
                        canonical_station_id, name, lat, lon, elevation_m.
    year_start, year_end : inclusive calendar-year filter.

    Returns
    -------
    DataFrame with columns: station_id, date, lat, lon, elevation_m, name,
    tmean_c, tmin_c, tmax_c, rh_pct, prcp_mm, wdsp_ms, gust_ms
    (one row per canonical_station_id x date that has at least one variable).
    """
    parts = sorted(glob.glob(str(Path(consolidated_dir) / "observations_consolidated" / "*.parquet")))
    frames = [pd.read_parquet(p) for p in parts]
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["canonical_station_id", "date", "variable", "value"]
    )
    long_df["date"] = pd.to_datetime(long_df["date"])

    start = pd.Timestamp(year=year_start, month=1, day=1)
    end = pd.Timestamp(year=year_end, month=12, day=31)
    long_df = long_df[(long_df["date"] >= start) & (long_df["date"] <= end)]
    long_df = long_df[long_df["variable"].isin(_VARIABLE_TO_COLUMN)]

    wide = long_df.pivot_table(
        index=["canonical_station_id", "date"], columns="variable", values="value", aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns=_VARIABLE_TO_COLUMN)
    for col in _VARIABLE_TO_COLUMN.values():
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.rename(columns={"canonical_station_id": "station_id"})

    meta = stations_meta.rename(columns={"canonical_station_id": "station_id"})[
        ["station_id", "name", "lat", "lon", "elevation_m"]
    ]
    result = wide.merge(meta, on="station_id", how="inner")

    ordered = ["station_id", "date", "lat", "lon", "elevation_m", "name"] + list(_VARIABLE_TO_COLUMN.values())
    return result[ordered].sort_values(["station_id", "date"]).reset_index(drop=True)


def find_pinnable_duplicate_pairs(
    duplicate_evidence: pd.DataFrame,
    station_aliases: pd.DataFrame,
    canonical_ids: set[str],
) -> pd.DataFrame:
    """
    Resolve duplicate_evidence.csv's rejected (accepted=False) left/right
    source-id pairs to canonical_station_id pairs, via station_aliases.csv's
    (source, source_station_id) -> canonical_station_id mapping.

    A pair is "pinnable" (must be pinned to the same fold) when:
      - it was rejected for merging (accepted == False),
      - both sides resolve to a still-separate canonical station, and
      - the rejection reason is not the project's absolute zero-overlap
        "never merge" rule (RESEARCH_PLAN_AND_DESIGN.md section 1.3: two
        records sharing no observation day are permanently unmergeable,
        which the project treats as sufficiently resolved to not need
        fold-level pinning).

    Returns
    -------
    DataFrame with columns 'a', 'b' of canonical_station_id pairs.
    """
    aliases = station_aliases.copy()
    aliases["src_key"] = aliases["source"] + ":" + aliases["source_station_id"].astype(str)
    lookup = dict(zip(aliases["src_key"], aliases["canonical_station_id"]))

    def resolve(key: str) -> str:
        return lookup.get(key, key)

    df = duplicate_evidence.copy()
    df = df[df["accepted"] == False]  # noqa: E712 (pandas boolean mask)
    df["a"] = df["left"].map(resolve)
    df["b"] = df["right"].map(resolve)

    df = df[df["a"].isin(canonical_ids) & df["b"].isin(canonical_ids)]
    df = df[df["a"] != df["b"]]
    df = df[~df["reason"].str.contains("never merge", case=False, na=False)]

    return df[["a", "b"]].drop_duplicates().reset_index(drop=True)
