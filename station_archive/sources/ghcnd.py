"""
sources/ghcnd.py
================
NOAA GHCN-Daily adapter.

DATA SOURCE
-----------
https://www.ncei.noaa.gov/pub/data/ghcn/daily/
Documentation: GHCND_documentation.pdf and readme.txt at that location.

Files used (all verified reachable on 13 September 2026):
  ghcnd-version.txt        release string, e.g. "3.34-upd-2026091118"
  ghcnd-stations.txt       fixed-width station metadata, ~11 MB
  ghcnd-inventory.txt      per station-element first/last year, ~36 MB
  by_station/<ID>.csv.gz   one gzipped CSV per station, ~260 kB typical

Per-station files make this the cheapest of the three new sources: the domain
subset is a few hundred stations rather than the whole global archive.

FORMATS
-------
ghcnd-stations.txt is fixed-width (readme.txt section IV):
    1-11 ID, 13-20 LATITUDE, 22-30 LONGITUDE, 32-37 ELEVATION,
    39-40 STATE, 42-71 NAME, 73-75 GSN FLAG, 77-79 HCN/CRN FLAG, 81-85 WMO ID

ghcnd-inventory.txt is fixed-width:
    1-11 ID, 13-20 LATITUDE, 22-30 LONGITUDE, 32-35 ELEMENT,
    37-40 FIRSTYEAR, 42-45 LASTYEAR

by_station/<ID>.csv.gz is headerless with eight columns:
    ID, YYYYMMDD, ELEMENT, VALUE, M-FLAG, Q-FLAG, S-FLAG, OBS-TIME

CAUTION FROM THE PLAN
---------------------
Section 1.1: GHCN-Daily "is itself a composite of merged sources, so overlap
with GSOD/C3S/ECA&D is expected".  This adapter does not attempt to resolve
that overlap -- duplicate detection is identity.py's job.
"""
from __future__ import annotations

import gzip
import io
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import pandas as pd

from station_archive import net
from station_archive.sources.base import StationSource
from station_archive.types import (
    Domain,
    DownloadRequest,
    RawObservation,
    SchemaError,
    SourceSnapshot,
    StationInventory,
)

if TYPE_CHECKING:            # runtime import would cycle via the registry
    from station_archive.policy import Policy

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"

# GHCN-D element -> canonical variable.  Only the elements this project uses
# are mapped; the archive carries many more.
#
# There is deliberately no 'tas' entry: GHCN-D has TAVG, but it is reported by
# only a subset of stations and its definition varies by contributing network
# (true 24-hour mean vs (TMAX+TMIN)/2).  Mapping it to 'tas' here would hand
# consolidation two incompatible quantities under one name.  TAVG is still
# carried through with its own label so a later step can decide.
ELEMENT_TO_VARIABLE: dict[str, str] = {
    "TMAX": "tasmax",
    "TMIN": "tasmin",
    "TAVG": "tavg_unspecified",
    "PRCP": "pr",
    "AWND": "sfcWind",
    "WSFG": "gust",
    "WSF2": "gust",
    "WSF5": "gust",
}

# Provider units, kept with the raw values (readme.txt section III).
ELEMENT_UNITS: dict[str, str] = {
    "TMAX": "degC_tenths",
    "TMIN": "degC_tenths",
    "TAVG": "degC_tenths",
    "PRCP": "mm_tenths",
    "AWND": "m_s_tenths",
    "WSFG": "m_s_tenths",
    "WSF2": "m_s_tenths",
    "WSF5": "m_s_tenths",
}

# Column position through which every ghcnd-stations.txt line must extend:
# the end of NAME.  The GSN, HCN/CRN and WMO fields after it are optional and
# are trimmed away entirely on stations that lack them.
_STATION_REQUIRED_WIDTH = 71

_STATION_FIELDS = [
    ("source_station_id", 0, 11),
    ("lat", 12, 20),
    ("lon", 21, 30),
    ("elevation_m", 31, 37),
    ("state", 38, 40),
    ("name", 41, 71),
    ("gsn_flag", 72, 75),
    ("hcn_crn_flag", 76, 79),
    ("wmo_id", 80, 85),
]

_INVENTORY_FIELDS = [
    ("source_station_id", 0, 11),
    ("lat", 12, 20),
    ("lon", 21, 30),
    ("element", 31, 35),
    ("first_year", 36, 40),
    ("last_year", 41, 45),
]

_DLY_COLUMNS = ["id", "date", "element", "value", "m_flag", "q_flag", "s_flag", "obs_time"]

# Value sentinel for missing data in the by_station CSV (readme.txt section III).
_MISSING_VALUE = -9999


class GhcndSource(StationSource):
    name = "ghcnd"
    licence = "NOAA open data; no redistribution restriction stated"
    licence_url = "https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf"
    redistribution = "unrestricted"
    element_to_variable = ELEMENT_TO_VARIABLE
    element_units = ELEMENT_UNITS

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url
        self._release: str | None = None
        self._station_frame: pd.DataFrame | None = None

    # ---- release ---------------------------------------------------------

    def release(self) -> str:
        """
        Release string parsed from ghcnd-version.txt.

        The file is a sentence, not a bare version, so the version token is
        extracted from it; the raw text is kept in the release notes only if
        parsing fails, which then raises rather than inventing a version.
        """
        if self._release is None:
            text = net.get_bytes(self.base_url + "ghcnd-version.txt").decode("utf-8", "replace")
            token = self._parse_version(text)
            self._release = token
        return self._release

    @staticmethod
    def _parse_version(text: str) -> str:
        """Extract e.g. '3.34-upd-2026091118' from the version sentence."""
        for word in text.replace(",", " ").split():
            stripped = word.strip("().")
            if stripped.count("-") >= 2 and stripped[0].isdigit() and "." in stripped:
                return stripped
        raise SchemaError(f"could not parse a version token from ghcnd-version.txt: {text[:120]!r}")

    # ---- discovery -------------------------------------------------------

    def _stations(self) -> pd.DataFrame:
        if self._station_frame is None:
            raw = net.get_bytes(self.base_url + "ghcnd-stations.txt").decode("utf-8", "replace")
            self._station_frame = _parse_fixed_width(
                raw, _STATION_FIELDS, "ghcnd-stations.txt", _STATION_REQUIRED_WIDTH
            )
        return self._station_frame

    def _inventory_table(self) -> pd.DataFrame:
        raw = net.get_bytes(self.base_url + "ghcnd-inventory.txt").decode("utf-8", "replace")
        return _parse_fixed_width(raw, _INVENTORY_FIELDS, "ghcnd-inventory.txt")

    def discover(
        self,
        domain: Domain,
        variables: Sequence[str],
        period: tuple[int, int],
    ) -> StationInventory:
        """
        Build the inventory from ghcnd-stations.txt and ghcnd-inventory.txt.

        The inventory file gives first/last year per station-element but not
        observation counts, so 'n_obs_period' and 'missing_fraction_period'
        are left empty here and filled by parse() once a station's records are
        on disk.  Reporting a count that was never measured would defeat the
        purpose of plan section 1.2's coverage questions.
        """
        period_start, period_end = period
        wanted_elements = {
            element
            for element, variable in ELEMENT_TO_VARIABLE.items()
            if variable in variables
        }

        # Copy before coercing: _stations() caches the parsed file, and
        # discover() may be called more than once per adapter instance.
        stations = self._stations().copy()
        for column in ("lat", "lon", "elevation_m"):
            stations[column] = pd.to_numeric(stations[column], errors="coerce")

        in_domain = stations[
            stations.apply(lambda row: domain.contains(row["lat"], row["lon"]), axis=1)
        ]
        # A repeated ID would make the .loc lookup below return a frame rather
        # than a row, silently corrupting every field it reads.  Keep the first
        # and say so, rather than failing obscurely later.
        duplicates = in_domain["source_station_id"].duplicated()
        if duplicates.any():
            print(
                f"[ghcnd] {int(duplicates.sum())} duplicate station IDs in "
                f"ghcnd-stations.txt; keeping the first occurrence of each"
            )
            in_domain = in_domain[~duplicates]
        metadata = in_domain.set_index("source_station_id")

        inventory = self._inventory_table()
        inventory["first_year"] = pd.to_numeric(inventory["first_year"], errors="coerce")
        inventory["last_year"] = pd.to_numeric(inventory["last_year"], errors="coerce")

        selected = inventory[
            inventory["source_station_id"].isin(metadata.index)
            & inventory["element"].isin(wanted_elements)
            # Overlap with the admission window, not containment: a station
            # covering part of 1985-2005 still contributes training days.
            & (inventory["first_year"] <= period_end)
            & (inventory["last_year"] >= period_start)
        ]

        release = self.release()
        retrieved_at = net.utc_now()
        records = []
        for row in selected.itertuples(index=False):
            station = metadata.loc[row.source_station_id]
            element = row.element
            records.append(
                {
                    "source": self.name,
                    "source_release": release,
                    "source_station_id": row.source_station_id,
                    "wmo_id": station["wmo_id"] or "",
                    "wigos_id": "",
                    "name": station["name"],
                    "lat": station["lat"],
                    "lon": station["lon"],
                    "elevation_m": station["elevation_m"],
                    "variable": ELEMENT_TO_VARIABLE[element],
                    "provider_element": element,
                    "units": ELEMENT_UNITS.get(element, "unknown"),
                    "observing_interval": "daily",
                    # readme.txt: the day is the local calendar day at the
                    # station; TMAX/TMIN refer to the observation-day window,
                    # which varies by network.
                    "timestamp_meaning": "local calendar day",
                    "first_date": f"{int(row.first_year)}-01-01",
                    "last_date": f"{int(row.last_year)}-12-31",
                    "n_obs_period": pd.NA,
                    "missing_fraction_period": pd.NA,
                    "access_status": "public",
                    "licence": self.licence,
                    "request": f"{self.base_url}by_station/{row.source_station_id}.csv.gz",
                    "retrieved_at": retrieved_at,
                }
            )
        return StationInventory.from_records(records)

    # ---- planning and transfer ------------------------------------------

    def plan(
        self,
        inventory: StationInventory,
        policy: Optional["Policy"] = None,
    ) -> list[DownloadRequest]:
        """
        One request per station, not per station-element.

        'policy' is unused: a by_station file covers the whole record, so there
        is no year window to apply.

        by_station files hold every element for a station, so planning by
        element would download the same file repeatedly.
        """
        station_ids = sorted(set(inventory.frame["source_station_id"]))
        return [
            DownloadRequest(
                source=self.name,
                key=f"ghcnd/{station_id}",
                relative_path=f"by_station/{station_id}.csv.gz",
                url=f"{self.base_url}by_station/{station_id}.csv.gz",
                metadata={"source_station_id": station_id},
            )
            for station_id in station_ids
        ]

    def _transfer(self, request: DownloadRequest, destination: Path) -> None:
        assert request.url is not None  # guaranteed by DownloadRequest.__post_init__
        net.download_to(request.url, destination)

    # ---- parsing ---------------------------------------------------------

    def parse(self, snapshot: SourceSnapshot) -> Iterator[RawObservation]:
        """
        Stream one by_station file as untouched provider values.

        Values stay in GHCN-D tenths; the three provider flags are preserved
        per plan section 1.4 ("retain provider QC flags ... rather than
        overwriting them").
        """
        with gzip.open(snapshot.path, "rt", encoding="utf-8", errors="replace") as handle:
            frame = pd.read_csv(
                io.StringIO(handle.read()),
                header=None,
                names=_DLY_COLUMNS,
                dtype=str,
                keep_default_na=False,
            )

        if frame.shape[1] != len(_DLY_COLUMNS):
            raise SchemaError(
                f"{snapshot.path.name}: expected {len(_DLY_COLUMNS)} columns, "
                f"found {frame.shape[1]}"
            )
        if not frame.empty and not frame["date"].iloc[0].isdigit():
            raise SchemaError(
                f"{snapshot.path.name}: column 2 is not a YYYYMMDD date "
                f"({frame['date'].iloc[0]!r}); by_station format may have changed"
            )

        release = snapshot.release
        for row in frame.itertuples(index=False):
            value = float(row.value) if row.value else float("nan")
            if value == _MISSING_VALUE:
                continue
            yield RawObservation(
                source=self.name,
                source_release=release,
                source_station_id=row.id,
                obs_date=_ghcnd_date(row.date, snapshot.path.name),
                element=row.element,
                value=value,
                units=ELEMENT_UNITS.get(row.element, "unknown"),
                provider_flags={
                    "m_flag": row.m_flag,
                    "q_flag": row.q_flag,
                    "s_flag": row.s_flag,
                    "obs_time": row.obs_time,
                },
                source_record_id=f"{row.id}:{row.date}:{row.element}",
            )


def _ghcnd_date(token: str, context: str) -> date:
    try:
        return datetime.strptime(token, "%Y%m%d").date()
    except ValueError as exc:
        raise SchemaError(f"{context}: unparseable date {token!r}") from exc


def _parse_fixed_width(
    text: str,
    fields: list[tuple[str, int, int]],
    context: str,
    required_width: int | None = None,
) -> pd.DataFrame:
    """
    Parse a NOAA fixed-width metadata file.

    'required_width' is the column position through which every line must
    extend.  It is not simply the last field's end, because NOAA right-trims
    these files: a station with no WMO ID -- the final 81-85 field of
    ghcnd-stations.txt -- ends after its name, and thousands of real lines are
    shorter than the nominal record width.

    Lines shorter than the required width are still a format change and raise,
    so a genuine column shift is caught while ordinary trailing-field
    truncation is not mistaken for one.  Slicing past the end of a short line
    yields an empty string, which is the correct reading of an absent optional
    field.
    """
    if required_width is None:
        required_width = max(end for _, _, end in fields)

    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line) < required_width:
            raise SchemaError(
                f"{context}: line {number} is {len(line)} characters, "
                f"expected at least {required_width}"
            )
        rows.append({name: line[start:end].strip() for name, start, end in fields})
    if not rows:
        raise SchemaError(f"{context}: no data lines")
    return pd.DataFrame(rows)
