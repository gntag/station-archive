"""
fold_manifest.py
=================
Builds and loads the frozen Step 2 fold manifest: one row per canonical
station, assigning a grouped spatial-fold id used identically by every
benchmark model (LightGBM today; TabM/patch-CNN later) so no model script
invents its own split (RESEARCH_PLAN_AND_DESIGN.md, "Architecture
guardrails": "The fold manifest is created once from canonical identities;
individual model scripts may read it but may not invent their own split.").

This replaces true per-station LOOCV (1,197 retrains per model) with grouped
spatial-fold cross-validation for every model, including LightGBM -- a
deliberate, documented deviation from the plan's literal "exact
canonical-station LOOCV" wording, made for compute-cost reasons on this
hardware (see Step 2 Design Decision Record).

METHOD
------
Spatial blocking via KMeans clustering on station coordinates -- the
standard Python approach to blockCV-style spatial blocks (Valavi et al.
2019, "blockCV: An R package for generating spatially or environmentally
separated folds", https://doi.org/10.1111/2041-210X.13107). Coordinates are
projected to a locally equal-angle space (lon * cos(mean_lat)) before
clustering so Euclidean KMeans distance approximates great-circle distance.
Each fold is then a real geographic cluster, so it doubles as the plan's
"leave-region-out" harder test (Appendix A) without a second mechanism.

DUPLICATE-RISK PINNING
-----------------------
Stations flagged `duplicate_risk=True` in canonical_stations.csv may still
be the same physical station as another canonical station (Step 1 left this
unresolved). Where a specific candidate partner can be identified (from
duplicate_evidence.csv or the human merge-review sheet, see
archive_ingest.find_pinnable_duplicate_pairs), the pair is pinned to the
same fold via union-find so they can never land on opposite sides of a
split. A duplicate-risk station with no identifiable partner is excluded
from scoring entirely (included=False) rather than guessed at.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def assign_spatial_folds(
    lat: np.ndarray,
    lon: np.ndarray,
    n_folds: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Cluster stations into n_folds spatial groups via KMeans.

    Longitude is scaled by cos(mean latitude) so that one unit of scaled
    longitude and one unit of latitude correspond to approximately equal
    arc-length (same convention as cerra_extract._precompute_patches).

    Returns
    -------
    Integer array of shape (n_stations,) with values in [0, n_folds).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    cos_lat = np.cos(np.radians(float(np.mean(lat))))
    coords = np.column_stack([lat, lon * cos_lat])
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def merge_duplicate_risk_groups(
    spatial_fold: np.ndarray,
    station_ids: np.ndarray,
    pinnable_pairs: pd.DataFrame,
) -> np.ndarray:
    """
    Force every station in a pinned duplicate-risk pair (or transitive
    chain of pairs) onto the same fold, via union-find.

    Parameters
    ----------
    spatial_fold  : per-station fold assignment from assign_spatial_folds.
    station_ids   : canonical_station_id per row, aligned with spatial_fold.
    pinnable_pairs: DataFrame with columns 'a', 'b' of canonical_station_id
                    pairs that must never be split. Ids absent from
                    station_ids are ignored (e.g. already excluded upstream).

    Returns
    -------
    A copy of spatial_fold with pinned groups collapsed onto one fold
    (the fold of the lowest-index member of each group). Unpinned stations
    keep their original fold.
    """
    index_of = {sid: i for i, sid in enumerate(station_ids)}
    parent = list(range(len(station_ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for a, b in zip(pinnable_pairs["a"], pinnable_pairs["b"]):
        if a in index_of and b in index_of:
            union(index_of[a], index_of[b])

    merged = spatial_fold.copy()
    for i in range(len(station_ids)):
        root = find(i)
        merged[i] = spatial_fold[root]
    return merged


def build_fold_manifest(
    canonical_stations: pd.DataFrame,
    pinnable_pairs: pd.DataFrame,
    stations_with_coverage: set[str],
    n_spatial_folds: int,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build the frozen fold manifest.

    Parameters
    ----------
    canonical_stations     : columns canonical_station_id, lat, lon,
                              elevation_m, duplicate_risk.
    pinnable_pairs          : columns 'a', 'b' -- canonical_station_id pairs
                              with an identifiable duplicate-risk partner.
    stations_with_coverage  : canonical_station_ids present in the
                              1985-2005 consolidated observation archive.
    n_spatial_folds         : number of KMeans spatial groups.
    seed                    : KMeans random_state, for reproducibility.

    Returns
    -------
    DataFrame with columns: canonical_station_id, lat, lon, elevation_m,
    spatial_fold (int, NaN if excluded), included (bool), exclusion_reason.
    """
    df = canonical_stations.copy().reset_index(drop=True)
    df["included"] = df["canonical_station_id"].isin(stations_with_coverage)
    df["exclusion_reason"] = np.where(df["included"], "", "no_1985_2005_coverage")

    has_partner = set(pinnable_pairs["a"]) | set(pinnable_pairs["b"])
    unresolved_risk = df["included"] & df["duplicate_risk"] & ~df["canonical_station_id"].isin(has_partner)
    df.loc[unresolved_risk, "included"] = False
    df.loc[unresolved_risk, "exclusion_reason"] = "duplicate_risk_unresolved_partner"

    df["spatial_fold"] = np.nan
    included = df[df["included"]]
    if len(included) > 0:
        raw_folds = assign_spatial_folds(
            included["lat"].values, included["lon"].values, n_spatial_folds, seed,
        )
        merged_folds = merge_duplicate_risk_groups(
            raw_folds, included["canonical_station_id"].values, pinnable_pairs,
        )
        df.loc[df["included"], "spatial_fold"] = merged_folds

    return df[[
        "canonical_station_id", "lat", "lon", "elevation_m",
        "spatial_fold", "included", "exclusion_reason",
    ]]


def save_fold_manifest(manifest: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)


def load_fold_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
