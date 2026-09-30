"""
inventory.py
============
Coverage-only analysis across source inventories.

Plan section 1.2 asks four questions before any bulk download:

  1. How many genuinely new physical stations and station-years does each
     source add?
  2. Which variables become denser -- temperature, precipitation,
     humidity/dew point, wind and gust?
  3. Where do gaps remain around the Cyclades, Crete, mainland Greece,
     Turkiye and the eastern Mediterranean?
  4. Which candidate records cannot be redistributed or used outside
     non-commercial research?

This module answers them and nothing else.  In particular, question 1 says
"genuinely new physical stations", which is a duplicate-detection question --
and duplicate detection belongs to identity.py.  So the proximity grouping
here is labelled and used as a COVERAGE ESTIMATE ONLY.  It must never be
mistaken for the canonical station crosswalk of plan section 1.3, which
requires WMO/WIGOS identifiers, name and elevation compatibility, and
overlapping-series correlation before two records may be called one station.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from station_archive import settings
from station_archive.identity import longitude_scale
from station_archive.policy import DEFAULT_POLICY, Policy
from station_archive.types import INVENTORY_COLUMNS, StationInventory

# Grouping radius for the coverage estimate.  Deliberately tight: at this
# latitude 0.02 degrees is roughly 2 km, which catches the same airport listed
# by two portals without merging a coastal station into an inland one.
COVERAGE_PROXIMITY_DEG = 0.02

# Sub-regions used for the geographic gap question.  Boxes are approximate and
# exist only to structure the report.
SUBREGIONS: dict[str, tuple[float, float, float, float]] = {
    # name: (lat_min, lat_max, lon_min, lon_max)
    "Cyclades": (36.2, 37.9, 24.0, 26.0),
    "Crete": (34.7, 35.8, 23.4, 26.4),
    "Mainland Greece": (36.5, 41.8, 20.0, 26.5),
    "Turkiye (west/south)": (36.0, 41.5, 26.0, 32.0),
    "Eastern Mediterranean (other)": (33.0, 36.0, 26.0, 32.0),
}


def merge(inventories: Iterable[StationInventory]) -> pd.DataFrame:
    """Concatenate per-source inventories into one frame."""
    frames = [inv.frame for inv in inventories if len(inv) > 0]
    if not frames:
        return pd.DataFrame(columns=list(INVENTORY_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _coverage_cluster(
    frame: pd.DataFrame,
    policy: Policy = DEFAULT_POLICY,
) -> pd.Series:
    """
    Approximate physical-site label from rounded coordinates.

    COVERAGE ESTIMATE ONLY -- see the module docstring.  Rounding to the
    proximity radius is a cheap stand-in for the real crosswalk; it will merge
    genuinely distinct nearby stations and separate a station that moved.

    LONGITUDE IS SCALED FIRST, for the reason identity.candidate_pairs spells
    out at length: a degree of longitude is shorter than a degree of latitude
    everywhere off the equator, so rounding raw degrees on both axes makes the
    east-west cell narrower than the north-south one -- about 1.6 km against
    2.2 km at this latitude. East-west duplicates then escape the cluster and
    gain_over_baseline overstates how many genuinely new stations a source
    adds. Same bug, same fix, one estimate later.
    """
    lat = pd.to_numeric(frame["lat"], errors="coerce")
    lon = pd.to_numeric(frame["lon"], errors="coerce") * longitude_scale(policy)
    step = COVERAGE_PROXIMITY_DEG
    return (
        (lat / step).round().astype("Int64").astype(str)
        + "_"
        + (lon / step).round().astype("Int64").astype(str)
    )


def source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-source station, variable and station-year totals."""
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["n_obs_period"] = pd.to_numeric(working["n_obs_period"], errors="coerce")

    return (
        working.groupby("source")
        .agg(
            stations=("source_station_id", "nunique"),
            variables=("variable", "nunique"),
            rows=("source_station_id", "size"),
            measured_station_days=("n_obs_period", "sum"),
        )
        .reset_index()
    )


def gain_over_baseline(
    frame: pd.DataFrame,
    baseline: str = "gsod",
    policy: Policy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """
    Stations each source adds beyond the baseline source's footprint.

    Answers plan question 1, and the Step 1 acceptance item "The incremental
    station and station-year gain over GSOD is quantified by variable" --
    subject to the coverage-estimate caveat above.
    """
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    working["cluster"] = _coverage_cluster(working, policy)

    baseline_sites = set(
        working.loc[working["source"] == baseline, "cluster"].dropna().unique()
    )

    rows = []
    for (source, variable), group in working.groupby(["source", "variable"]):
        if source == baseline:
            continue
        clusters = set(group["cluster"].dropna().unique())
        rows.append(
            {
                "source": source,
                "variable": variable,
                "stations": group["source_station_id"].nunique(),
                "sites_not_in_baseline": len(clusters - baseline_sites),
                "sites_shared_with_baseline": len(clusters & baseline_sites),
            }
        )

    # An inventory containing only the baseline source yields no rows; return
    # the empty table with its columns intact so the report still renders.
    if not rows:
        return pd.DataFrame(
            columns=["source", "variable", "stations", "sites_not_in_baseline",
                     "sites_shared_with_baseline"]
        )
    return pd.DataFrame(rows).sort_values(["variable", "source"]).reset_index(drop=True)


def variable_density(frame: pd.DataFrame) -> pd.DataFrame:
    """Stations per variable per source — plan question 2."""
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.pivot_table(
            index="variable",
            columns="source",
            values="source_station_id",
            aggfunc="nunique",
            fill_value=0,
        )
        .reset_index()
    )


def subregion_coverage(
    frame: pd.DataFrame,
    policy: Policy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """Distinct coverage sites per sub-region — plan question 3."""
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    working["lat"] = pd.to_numeric(working["lat"], errors="coerce")
    working["lon"] = pd.to_numeric(working["lon"], errors="coerce")
    working["cluster"] = _coverage_cluster(working, policy)

    rows = []
    for name, (lat_min, lat_max, lon_min, lon_max) in SUBREGIONS.items():
        inside = working[
            working["lat"].between(lat_min, lat_max)
            & working["lon"].between(lon_min, lon_max)
        ]
        row = {"subregion": name, "sites": inside["cluster"].nunique()}
        for source in sorted(working["source"].unique()):
            row[source] = inside.loc[inside["source"] == source, "cluster"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def elevation_coverage(
    frame: pd.DataFrame,
    policy: Policy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """
    Sites per elevation band.

    The Step 1 acceptance criteria ask for a report identifying remaining
    "spatial, elevation and variable gaps"; elevation is the one most easily
    forgotten, and it matters directly for a 100 m target grid over islands.
    """
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    working["elevation_m"] = pd.to_numeric(working["elevation_m"], errors="coerce")
    working["cluster"] = _coverage_cluster(working, policy)
    bands = [-np.inf, 0, 100, 300, 600, 1000, np.inf]
    labels = ["<=0 m", "0-100 m", "100-300 m", "300-600 m", "600-1000 m", ">1000 m"]
    working["band"] = pd.cut(working["elevation_m"], bins=bands, labels=labels)

    return (
        working.groupby("band", observed=False)["cluster"]
        .nunique()
        .reset_index()
        .rename(columns={"cluster": "sites"})
    )


def licence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Access status and licence per source — plan question 4."""
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["source", "access_status", "licence"])
        .size()
        .reset_index(name="rows")
    )


def _table(frame: pd.DataFrame) -> str:
    """
    Render a frame as a Markdown table.

    Written out by hand rather than via DataFrame.to_markdown, which needs the
    optional 'tabulate' package.  A coverage report is not worth a new runtime
    dependency in pyproject.toml.
    """
    if frame is None or frame.empty:
        return "_(no rows)_\n"

    columns = [str(c) for c in frame.columns]
    rows = [[_cell(value) for value in record] for record in frame.itertuples(index=False)]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]

    def line(cells: list[str]) -> str:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(columns), separator, *(line(row) for row in rows)]) + "\n"


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def coverage_report(
    frame: pd.DataFrame,
    baseline: str = "gsod",
    period: Optional[tuple[int, int]] = None,
    policy: Policy = DEFAULT_POLICY,
) -> str:
    """Render the discovery findings as Markdown."""
    period = policy.period if period is None else period
    domain = policy.domain
    lines = [
        "# Station source coverage report",
        "",
        f"Domain: lat {domain.lat_min}..{domain.lat_max}, "
        f"lon {domain.lon_min}..{domain.lon_max} "
        f"(+{domain.buffer_deg} deg buffer)",
        f"Admission window: {period[0]}-{period[1]}",
        f"Inventory rows: {len(frame)}",
        "",
        "> Station grouping in this report is a COVERAGE ESTIMATE from "
        f"{COVERAGE_PROXIMITY_DEG} degree proximity only. It is not the canonical "
        "station crosswalk: identity resolution (plan section 1.3) requires "
        "WMO/WIGOS identifiers, name and elevation compatibility, and overlapping "
        "series correlation, and is performed by identity.py.",
        "",
        "## Per-source totals",
        "",
        _table(source_summary(frame)),
        f"## Question 1 - stations gained over {baseline}",
        "",
        _table(gain_over_baseline(frame, baseline, policy)),
        "## Question 2 - stations per variable",
        "",
        _table(variable_density(frame)),
        "## Question 3 - sub-regional coverage",
        "",
        _table(subregion_coverage(frame, policy)),
        "### Elevation bands",
        "",
        _table(elevation_coverage(frame, policy)),
        "## Question 4 - access and redistribution",
        "",
        _table(licence_summary(frame)),
    ]
    return "\n".join(lines)


def write_report(
    frame: pd.DataFrame,
    path: Path | None = None,
    baseline: str = "gsod",
    policy: Policy = DEFAULT_POLICY,
) -> Path:
    path = path or (settings.inventory_dir() / "coverage_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        coverage_report(frame, baseline=baseline, policy=policy), encoding="utf-8"
    )
    return path


def load_inventories(sources: Sequence[str]) -> list[StationInventory]:
    """Read previously written per-source inventory CSVs."""
    inventories = []
    for source in sources:
        path = settings.inventory_dir() / f"{source}_inventory.csv"
        if path.exists():
            inventories.append(StationInventory.read_csv(path))
    return inventories
