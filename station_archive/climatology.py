"""
climatology.py
==============
A climatological plausibility screen: which source records sit so far from an
independent reference, on average, that their values cannot be describing the
quantity they claim to (plan 1.4: project QC layered over provider flags, with
transparent rejection reasons).

WHAT IT CATCHES
---------------
Gross-range QC removes the impossible value; this removes nothing, and looks for
the impossible *record*. Measured on this archive against CERRA:

  * ECA&D precipitation series that are exactly zero on 400-6,600 matched days at
    Greek, Bulgarian and Turkish airports (Elefsis, Kos, Santorini, Chios ...).
    ECA&D ranks first for pr, so those zeros were the archive's values.
  * ECA&D precipitation at Monte Cimone (0.19 mm/day) and Vigna di Valle
    (0.32 mm/day), where the GHCN-D and GSOD candidates they outranked, and
    CERRA, all give several mm. The wet-day fraction matches (0.77); the
    amounts are 7-40x too small.
  * A Lazio block of ECA&D stations reading +5 to +6 degC above the reference
    in tasmax after the elevation adjustment -- a whole network offset at once.

WHY MATCHED DAYS, AND NOT STATION MEANS
---------------------------------------
Every departure is computed on the record's own days only, because a record's
sampling can manufacture a departure by itself. GSOD reports a gust only on
gusty days: a median 10% of the days a GSOD station reports wind, on which the
reference gust runs 1.45x its all-days mean. Its all-days mean therefore
describes the windiest tenth of days. Compared that way, a Sicilian network
reporting every day looked half as gusty as its GSOD neighbours (median 0.55);
on shared days it is 0.86, and against the reference it sits at a normal 0.94.
The reporting fraction is recorded on every row so a reviewer can see it.

WHAT THE REFERENCE CANNOT TELL YOU
----------------------------------
A 5.5 km reanalysis cannot resolve a summit, a cape or a sheltered valley:
summit wind runs 3x the reference while being perfectly real. So the departure
alone never decides. Each row carries neighbour context -- how other stations
nearby depart -- which separates "this station disagrees with everything around
it" from "this place is unlike its grid cell", and a same-direction count that
makes a network-wide offset visible as such.

WHAT THIS MODULE NEVER DOES
---------------------------
Change or remove a value. It scores and proposes; review.py shows the proposal;
a reviewer answers DECISION_exclude; consolidate.py acts only on 'yes', moving
the record's values to the audit table so the next-ranked source can take over.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from station_archive import consolidate, standardize
from station_archive.identity import haversine_km
from station_archive.policy import DEFAULT_POLICY, ClimatologyPolicy, Policy

TIER_EXCLUDE = "propose_exclusion"
TIER_REVIEW = "review"
TIER_OK = "ok"

# 1.4826 * MAD estimates the standard deviation of a normal sample, and unlike
# the standard deviation it is not inflated by the very records being looked for.
MAD_TO_SD = 1.4826

SCREEN_COLUMNS: tuple[str, ...] = (
    "alias",
    "canonical_station_id",
    "source",
    "source_station_id",
    "name",
    "lat",
    "lon",
    "variable",
    "status",
    "reference_key",
    "n_matched_days",
    "station_mean",
    "reference_mean",
    "elevation_adjustment",
    "departure",
    "departure_units",
    "robust_z",
    "tier",
    "reporting_fraction",
    "sampling_note",
    "n_neighbors",
    "neighbor_median_departure",
    "departure_vs_neighbors",
    "n_neighbors_same_direction",
)

# Read from the provider's own convention rather than inferred per station:
# GSOD's GUST field is the day's maximum reported gust, and a gust is reported
# only when one occurred, so the GSOD gust sample is conditioned on the outcome.
_SAMPLING_NOTES = {
    ("gsod", "gust"): (
        "GSOD reports a gust only on gusty days; its mean describes those days, "
        "not the climate"
    ),
}


def departure(
    station_mean: float,
    reference_mean: float,
    variable: str,
    station_elevation_m: float = float("nan"),
    reference_elevation_m: float = float("nan"),
    policy: ClimatologyPolicy = DEFAULT_POLICY.climatology,
) -> float:
    """
    How far a record's matched-day mean sits from the reference's.

    A log ratio for multiplicative variables, -inf for a zero mean against a
    positive reference, NaN where the ratio is undefined. Otherwise a
    difference in the variable's units, with the reference moved to the
    station's height for temperatures when both elevations are known.
    """
    if variable in policy.multiplicative_variables:
        if not reference_mean > 0 or station_mean < 0 or math.isnan(station_mean):
            return float("nan")
        if station_mean == 0:
            return -math.inf
        return math.log(station_mean / reference_mean)
    return station_mean - reference_mean - _lapse_adjustment(
        variable, station_elevation_m, reference_elevation_m, policy
    )


def _lapse_adjustment(
    variable: str,
    station_elevation_m: float,
    reference_elevation_m: float,
    policy: ClimatologyPolicy,
) -> float:
    """The temperature change expected from the reference's height to the station's."""
    if variable not in policy.lapse_variables:
        return 0.0
    if pd.isna(station_elevation_m) or pd.isna(reference_elevation_m):
        return 0.0
    return -policy.lapse_rate_k_per_m * (station_elevation_m - reference_elevation_m)


def screen_series(
    alias: str,
    variable: str,
    station: pd.Series,
    reference: pd.Series,
    days_reporting_any: int,
    station_elevation_m: float = float("nan"),
    reference_elevation_m: float = float("nan"),
    reference_key: str = "",
    policy: ClimatologyPolicy = DEFAULT_POLICY.climatology,
) -> dict:
    """
    One record x variable compared with the reference on matched days.

    Both series must be date-indexed and in canonical units.
    """
    source = alias.split(":", 1)[0]
    multiplicative = variable in policy.multiplicative_variables
    station = station.dropna()
    row = {
        "alias": alias,
        "variable": variable,
        "reference_key": reference_key,
        "n_matched_days": 0,
        "station_mean": float("nan"),
        "reference_mean": float("nan"),
        "elevation_adjustment": float("nan"),
        "departure": float("nan"),
        "departure_units": "log ratio" if multiplicative else _units(variable),
        "reporting_fraction": (
            len(station) / days_reporting_any if days_reporting_any > 0 else float("nan")
        ),
        "sampling_note": _SAMPLING_NOTES.get((source, variable), ""),
    }
    reference = reference.dropna()
    if reference.empty:
        return {**row, "status": "no_reference"}

    shared = station.index.intersection(reference.index)
    row["n_matched_days"] = len(shared)
    if len(shared) < policy.min_matched_days:
        return {**row, "status": "too_few_matched_days"}

    station_mean = float(station.loc[shared].mean())
    reference_mean = float(reference.loc[shared].mean())
    value = departure(
        station_mean, reference_mean, variable,
        station_elevation_m, reference_elevation_m, policy,
    )
    adjustment = _lapse_adjustment(
        variable, station_elevation_m, reference_elevation_m, policy
    )
    return {
        **row,
        "station_mean": station_mean,
        "reference_mean": reference_mean,
        "elevation_adjustment": adjustment if variable in policy.lapse_variables else float("nan"),
        "departure": value,
        "status": "tested" if not math.isnan(value) else "undefined_departure",
    }


def _units(variable: str) -> str:
    return standardize.canonical_unit_for(variable) or ""


def score(
    screen: pd.DataFrame,
    policy: ClimatologyPolicy = DEFAULT_POLICY.climatology,
) -> pd.DataFrame:
    """
    Robust z per variable and the tier it puts each tested record in.

    The centre and spread come from the finite departures of tested records
    only, so an infinite departure is flagged without dragging the spread.
    A variable with no spread at all scores nothing rather than dividing by
    zero: identical departures give no basis for calling any one of them odd.
    """
    out = screen.copy()
    out["robust_z"] = np.nan
    out["tier"] = ""
    if out.empty:
        return out
    tested = out["status"] == "tested"
    for variable in sorted(out.loc[tested, "variable"].unique()):
        rows = tested & (out["variable"] == variable)
        values = out.loc[rows, "departure"].astype("float64")
        finite = values[np.isfinite(values)]
        centre = float(finite.median()) if not finite.empty else float("nan")
        spread = MAD_TO_SD * float((finite - centre).abs().median()) if not finite.empty else 0.0
        if not spread > 0:
            out.loc[rows, "tier"] = TIER_OK
            continue
        z = (values - centre) / spread
        out.loc[rows, "robust_z"] = z
        out.loc[rows, "tier"] = np.where(
            z.abs() > policy.exclude_z, TIER_EXCLUDE,
            np.where(z.abs() > policy.review_z, TIER_REVIEW, TIER_OK),
        )
    return out


def add_neighbor_context(
    screen: pd.DataFrame,
    policy: ClimatologyPolicy = DEFAULT_POLICY.climatology,
) -> pd.DataFrame:
    """
    How the other stations nearby depart, for the reviewer.

    A neighbour is another canonical station with a tested record of the same
    variable within the radius. Each neighbouring station counts once, at the
    median of its records' departures, so a station delivered by three archives
    does not outvote one delivered by one. A record is never its own
    neighbour, and nor are the other records of its own station.
    """
    out = screen.copy()
    out["n_neighbors"] = 0
    out["neighbor_median_departure"] = np.nan
    out["departure_vs_neighbors"] = np.nan
    out["n_neighbors_same_direction"] = 0
    if out.empty:
        return out

    tested = out[out["status"] == "tested"]
    for variable, group in tested.groupby("variable"):
        flagged = group["tier"].isin([TIER_EXCLUDE, TIER_REVIEW])
        stations = (
            group.assign(
                up=flagged & (group["robust_z"] > 0),
                down=flagged & (group["robust_z"] < 0),
            )
            .groupby("canonical_station_id")
            .agg(
                lat=("lat", "first"),
                lon=("lon", "first"),
                departure=("departure", _station_departure),
                up=("up", "any"),
                down=("down", "any"),
            )
        )
        lat = stations["lat"].to_numpy(dtype="float64")
        lon = stations["lon"].to_numpy(dtype="float64")
        for index, row in group.iterrows():
            distance = np.array([
                haversine_km(row["lat"], row["lon"], a, b) for a, b in zip(lat, lon)
            ])
            near = (distance <= policy.neighbor_radius_km) & (
                stations.index != row["canonical_station_id"]
            )
            neighbours = stations[near]
            out.at[index, "n_neighbors"] = len(neighbours)
            if neighbours.empty:
                continue
            median = float(neighbours["departure"].median())
            out.at[index, "neighbor_median_departure"] = median
            out.at[index, "departure_vs_neighbors"] = row["departure"] - median
            if row["tier"] in (TIER_EXCLUDE, TIER_REVIEW):
                column = "up" if row["robust_z"] > 0 else "down"
                out.at[index, "n_neighbors_same_direction"] = int(neighbours[column].sum())
    return out


def _station_departure(values: pd.Series) -> float:
    """One departure per station: the median of its finite records, else any."""
    finite = values[np.isfinite(values)]
    return float(finite.median()) if not finite.empty else float(values.median())


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reference:
    """
    Daily reference values per reference point, in canonical units.

    Kept wide and split by point rather than melted to one long table: the
    CERRA cache is ~9 million rows, and seven variables long would be ~60
    million rows held for lookups that only ever ask for one point.
    """

    frames: Mapping[str, pd.DataFrame]
    elevations: Mapping[str, float] = field(default_factory=dict)

    def keys(self) -> set[str]:
        return set(self.frames)

    def series(self, key: str, variable: str) -> pd.Series:
        frame = self.frames.get(key)
        if frame is None or variable not in frame.columns:
            return pd.Series(dtype="float64")
        return frame[variable]

    def elevation(self, key: str) -> float:
        value = self.elevations.get(key)
        return float(value) if value is not None and not pd.isna(value) else float("nan")


def reference_from_wide(
    frame: pd.DataFrame,
    id_column: str,
    date_column: str,
    columns: Mapping[str, str],
    elevation: Optional[Mapping[str, float]] = None,
) -> Reference:
    """
    A Reference from a wide table keyed by point and date.

    `columns` maps a canonical variable name to the table's column for it. The
    values must already be in canonical units -- this converts nothing.
    """
    present = {v: c for v, c in columns.items() if c in frame.columns}
    working = frame[[id_column, date_column, *present.values()]].rename(
        columns={c: v for v, c in present.items()}
    )
    working[date_column] = pd.to_datetime(working[date_column])
    frames = {
        str(key): group.drop(columns=[id_column]).set_index(date_column).sort_index()
        for key, group in working.groupby(id_column, sort=True)
    }
    return Reference(frames=frames, elevations=dict(elevation or {}))


def reference_keys(aliases: pd.DataFrame, reference_ids: set[str]) -> dict[str, str]:
    """
    Which reference point each source record is compared with.

    The record's own key when the reference has it; otherwise the smallest
    reference key among the other records of its station. A reference built
    for an earlier crosswalk is keyed by that crosswalk's canonical ids, and a
    canonical id is always one of its group's alias keys, so any member's key
    still resolves after a merge has renamed the group. Records of one station
    sit within a few kilometres, inside one reanalysis cell.
    """
    keys = aliases["source"] + ":" + aliases["source_station_id"].astype(str)
    working = aliases.assign(alias=keys)
    out: dict[str, str] = {}
    for _, group in working.groupby("canonical_station_id", sort=True):
        members = sorted(group["alias"])
        available = [k for k in members if k in reference_ids]
        for alias in members:
            if alias in reference_ids:
                out[alias] = alias
            elif available:
                out[alias] = available[0]
    return out


# ---------------------------------------------------------------------------
# The whole screen
# ---------------------------------------------------------------------------

def run_screen(
    aliases: pd.DataFrame,
    loader,
    reference: Reference,
    policy: Policy = DEFAULT_POLICY,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Screen every record whose values consolidation could select.

    A source outside a variable's priority list is never screened for it: an
    exclusion acts only in consolidation, so judging a record nothing can
    select would ask the reviewer a question whose answer changes nothing.
    """
    keys = reference_keys(aliases, reference.keys())
    rows: list[dict] = []
    groups = aliases.groupby("canonical_station_id", sort=True)
    for index, (canonical_id, group) in enumerate(groups, start=1):
        for record in group.itertuples(index=False):
            source = str(record.source)
            station_id = str(record.source_station_id)
            alias = f"{source}:{station_id}"
            variables = [v for v in policy.source_priority if source in policy.priority_for(v)]
            if not variables:
                continue
            series = {
                v: consolidate.canonical_series(source, station_id, v, loader)
                for v in variables
            }
            reporting = set()
            for s in series.values():
                reporting.update(s.dropna().index)
            key = keys.get(alias, "")
            for variable, values in series.items():
                if values.dropna().empty:
                    continue
                row = screen_series(
                    alias, variable, values,
                    reference.series(key, variable) if key else pd.Series(dtype="float64"),
                    days_reporting_any=len(reporting),
                    station_elevation_m=pd.to_numeric(
                        getattr(record, "elevation_m", float("nan")), errors="coerce"
                    ),
                    reference_elevation_m=reference.elevation(key) if key else float("nan"),
                    reference_key=key,
                    policy=policy.climatology,
                )
                row.update(
                    canonical_station_id=canonical_id,
                    source=source,
                    source_station_id=station_id,
                    name=getattr(record, "name", ""),
                    lat=getattr(record, "lat", float("nan")),
                    lon=getattr(record, "lon", float("nan")),
                )
                rows.append(row)
        if hasattr(loader, "clear"):
            loader.clear()
        if progress and index % 100 == 0:
            print(f"[climatology] {index} stations screened")

    if not rows:
        return pd.DataFrame(columns=list(SCREEN_COLUMNS))
    screen = add_neighbor_context(score(pd.DataFrame(rows), policy.climatology), policy.climatology)
    return screen[list(SCREEN_COLUMNS)].sort_values(["variable", "alias"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reading the reviewer's answers
# ---------------------------------------------------------------------------

SERIES_SHEET = "Series"
SERIES_LEDGER_SHEET = "Series_decisions"


def load_exclusion_decisions(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Reviewer exclusion decisions, keyed by (alias, variable), as (answer, note).

    Read from the ledger first and the offered sheet second, so an answer
    changed on the sheet the reviewer is looking at wins over the ledger copy
    of the old one. Only 'yes' and 'no' are answers; anything else is
    undecided, so a stray note cannot be mistaken for an instruction.
    """
    decisions: dict[tuple[str, str], tuple[str, str]] = {}
    for sheet in (SERIES_LEDGER_SHEET, SERIES_SHEET):
        try:
            frame = pd.read_excel(path, sheet_name=sheet, dtype=str)
        except ValueError:
            continue                      # sheet absent from this workbook
        if "DECISION_exclude" not in frame.columns:
            continue
        for row in frame.fillna("").itertuples(index=False):
            answer = str(row.DECISION_exclude).strip().lower()
            if answer in ("yes", "no"):
                note = str(getattr(row, "DECISION_note", "") or "").strip()
                decisions[(str(row.alias), str(row.variable))] = (answer, note)
    return decisions


def exclusions_from_decisions(
    decisions: Mapping[tuple[str, str], tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """The 'yes' answers, as the reason consolidation records for each."""
    return {
        key: (note or "excluded by reviewer")
        for key, (answer, note) in sorted(decisions.items())
        if answer == "yes"
    }
