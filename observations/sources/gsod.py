"""
sources/gsod.py
===============
NOAA Global Summary of the Day adapter.

DATA SOURCE
-----------
https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<year>/
One CSV per station per year.  Station filenames follow the WMO convention;
the existing pipeline uses prefix "16", which covers Greece, Turkey, Cyprus
and neighbouring countries.

RELATIONSHIP TO Station_Data.py
-------------------------------
Station_Data/Station_Data.py already downloads exactly these files and its
output at Station_Data/downloaded_station_data/ is what
downscaling/station_ingest.py reads today.  That script is untouched and
keeps working.

This adapter re-acquires the same files into the versioned archive so GSOD
carries the same provenance as the new sources: a release stamp, a request
manifest, per-file checksums and a licence record.  Plan section 1.1 keeps
GSOD as "the current reproducible baseline", and the coverage report cannot
quantify "incremental station and station-year gain over GSOD" unless GSOD is
inventoried the same way as everything it is compared against.

DISCOVERY COST
--------------
GSOD publishes no station-metadata file alongside the yearly directories, and
a file's coordinates live inside the file itself.  Discovery therefore lists
each year's directory (cheap) and reads the first data line of each candidate
CSV to learn its coordinates (one small ranged read per station-year).  To
keep that bounded, coordinates are resolved once per station from the first
year in which it appears.

UNITS
-----
GSOD reports Fahrenheit, inches and knots.  Those values are stored unchanged,
per plan section 1.4; conversion belongs to standardize.py.  The missing-value
sentinels are recorded as flags rather than applied, so nothing is discarded
at the bronze layer beyond exact sentinel matches.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from observations import config, net
from observations.sources.base import StationSource
from observations.types import (
    Domain,
    DownloadRequest,
    RawObservation,
    SchemaError,
    SourceSnapshot,
    StationInventory,
)

BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"

DEFAULT_STATION_PREFIX = "16"

# GSOD column -> canonical variable.
#
# TEMP is the daily mean of the available hourly observations, MAX/MIN the
# reported daily extremes, WDSP the mean wind, GUST the maximum gust.
# DEWP is carried because relative humidity is derivable from TEMP and DEWP,
# but it is NOT labelled 'hurs' here: plan section 1.4 allows computing RH
# "only when both inputs and the formula are recorded", which is a
# standardization decision, not a download one.
COLUMN_TO_VARIABLE: dict[str, str] = {
    "TEMP": "tas",
    "MAX": "tasmax",
    "MIN": "tasmin",
    "PRCP": "pr",
    "WDSP": "sfcWind",
    "GUST": "gust",
    "DEWP": "dew_point",
}

COLUMN_UNITS: dict[str, str] = {
    "TEMP": "degF",
    "MAX": "degF",
    "MIN": "degF",
    "DEWP": "degF",
    "PRCP": "inch",
    "WDSP": "knot",
    "GUST": "knot",
}

# NOAA sentinels meaning "not recorded" (GSOD readme).
_SENTINELS: dict[str, float] = {
    "TEMP": 9999.9,
    "MAX": 9999.9,
    "MIN": 9999.9,
    "DEWP": 9999.9,
    "WDSP": 999.9,
    "GUST": 999.9,
    "PRCP": 99.99,
}

# Columns whose value carries a companion attribute column.
_ATTRIBUTE_COLUMNS = {"TEMP", "MAX", "MIN", "DEWP", "WDSP", "PRCP"}

_REQUIRED_COLUMNS = {"STATION", "DATE", "LATITUDE", "LONGITUDE"}


class _LinkParser(HTMLParser):
    """Extract href values from NOAA's Apache-style directory index pages."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


class GsodSource(StationSource):
    name = "gsod"
    licence = "NOAA open data; no redistribution restriction stated"
    licence_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt"
    redistribution = "unrestricted"

    def __init__(
        self,
        base_url: str = BASE_URL,
        station_prefix: str = DEFAULT_STATION_PREFIX,
        year_start: int = config.FETCH_YEAR_START,
        year_end: int = config.FETCH_YEAR_END,
    ) -> None:
        self.base_url = base_url
        self.station_prefix = station_prefix
        self.year_start = year_start
        self.year_end = year_end
        self._release: str | None = None

    # ---- release ---------------------------------------------------------

    def release(self) -> str:
        """
        Retrieval date, because GSOD publishes no version identifier.

        Named 'retrieved-' rather than given a bare date so nothing downstream
        mistakes it for a provider version.
        """
        if self._release is None:
            self._release = "retrieved-" + datetime.now(timezone.utc).strftime("%Y%m%d")
        return self._release

    # ---- discovery -------------------------------------------------------

    def list_year_files(self, year: int) -> list[str]:
        """Station CSV filenames in one year directory, filtered by prefix."""
        url = f"{self.base_url}{year}/"
        html = net.get_bytes(url).decode("utf-8", errors="ignore")
        parser = _LinkParser()
        parser.feed(html)
        names = {
            Path(link).name
            for link in parser.links
            if Path(link).name.endswith(".csv")
            and Path(link).name.startswith(self.station_prefix)
        }
        if not names:
            raise SchemaError(
                f"no station CSVs matching prefix {self.station_prefix!r} in {url}; "
                f"directory listing format may have changed"
            )
        return sorted(names)

    def station_header(self, year: int, filename: str) -> dict[str, str] | None:
        """
        Read a station's identity from the first data row of its yearly file.

        Only the first few kilobytes are fetched, because GSOD puts the
        coordinates on every row and the first one is enough.
        """
        url = f"{self.base_url}{year}/{filename}"
        try:
            head = net.get_bytes(url, (0, 4096)).decode("utf-8", errors="replace")
        except Exception:
            return None

        lines = head.splitlines()
        if len(lines) < 2:
            return None
        reader = csv.DictReader(io.StringIO("\n".join(lines[:2])))
        row = next(reader, None)
        if row is None:
            return None
        if not _REQUIRED_COLUMNS.issubset({k.strip() for k in row}):
            raise SchemaError(
                f"{filename}: missing required GSOD columns; found {sorted(row)}"
            )
        return {k.strip(): (v or "").strip() for k, v in row.items() if k}

    def discover(
        self,
        domain: Domain,
        variables: Sequence[str],
        period: tuple[int, int],
    ) -> StationInventory:
        """
        Walk the year directories, resolve each station's coordinates once,
        and emit one inventory row per station-variable.

        'first_date' and 'last_date' come from the range of years in which the
        station's file appears, which is the only date information available
        without reading whole files.
        """
        period_start, period_end = period
        release = self.release()
        retrieved_at = net.utc_now()

        years_present: dict[str, list[int]] = {}
        metadata: dict[str, dict[str, str]] = {}

        for year in range(self.year_start, self.year_end + 1):
            try:
                filenames = self.list_year_files(year)
            except Exception as exc:
                print(f"[gsod] {year}: listing unavailable ({exc})")
                continue

            for filename in filenames:
                station_id = Path(filename).stem
                years_present.setdefault(station_id, []).append(year)
                if station_id in metadata:
                    continue
                header = self.station_header(year, filename)
                if header is not None:
                    metadata[station_id] = header

        wanted_columns = [
            column for column, variable in COLUMN_TO_VARIABLE.items() if variable in variables
        ]

        records = []
        for station_id, years in sorted(years_present.items()):
            header = metadata.get(station_id)
            if header is None:
                continue
            lat = pd.to_numeric(header.get("LATITUDE"), errors="coerce")
            lon = pd.to_numeric(header.get("LONGITUDE"), errors="coerce")
            if not domain.contains(lat, lon):
                continue
            if min(years) > period_end or max(years) < period_start:
                continue

            for column in wanted_columns:
                records.append(
                    {
                        "source": self.name,
                        "source_release": release,
                        "source_station_id": station_id,
                        # GSOD filenames are the 11-digit USAF-WBAN pair; the
                        # leading five digits are the WMO number when the
                        # trailing WBAN is the 99999 filler.
                        "wmo_id": station_id[:5] if station_id.endswith("99999") else "",
                        "wigos_id": "",
                        "name": header.get("NAME", ""),
                        "lat": lat,
                        "lon": lon,
                        "elevation_m": pd.to_numeric(header.get("ELEVATION"), errors="coerce"),
                        "variable": COLUMN_TO_VARIABLE[column],
                        "provider_element": column,
                        "units": COLUMN_UNITS.get(column, "unknown"),
                        "observing_interval": "daily",
                        "timestamp_meaning": "UTC day summary",
                        "first_date": f"{min(years)}-01-01",
                        "last_date": f"{max(years)}-12-31",
                        "n_obs_period": pd.NA,
                        "missing_fraction_period": pd.NA,
                        "access_status": "public",
                        "licence": self.licence,
                        "request": f"{self.base_url}<year>/{station_id}.csv",
                        "retrieved_at": retrieved_at,
                    }
                )

        return StationInventory.from_records(records)

    # ---- planning and transfer ------------------------------------------

    def plan(self, inventory: StationInventory) -> list[DownloadRequest]:
        """
        One request per station-year within the configured fetch window.

        The inventory's first/last year bound the years requested for each
        station, so absent years are not requested at all.
        """
        frame = inventory.frame
        requests: list[DownloadRequest] = []
        seen: set[str] = set()

        for station_id, group in frame.groupby("source_station_id", sort=True):
            first = int(str(group["first_date"].iloc[0])[:4])
            last = int(str(group["last_date"].iloc[0])[:4])
            for year in range(max(first, self.year_start), min(last, self.year_end) + 1):
                key = f"gsod/{year}/{station_id}"
                if key in seen:
                    continue
                seen.add(key)
                requests.append(
                    DownloadRequest(
                        source=self.name,
                        key=key,
                        relative_path=f"{year}/{station_id}.csv",
                        url=f"{self.base_url}{year}/{station_id}.csv",
                        metadata={"source_station_id": station_id, "year": year},
                    )
                )
        return requests

    def _transfer(self, request: DownloadRequest, destination: Path) -> None:
        assert request.url is not None
        net.download_to(request.url, destination)

    # ---- parsing ---------------------------------------------------------

    def parse(self, snapshot: SourceSnapshot) -> Iterator[RawObservation]:
        """
        Stream one station-year CSV as untouched provider values.

        Sentinel values are skipped (they encode absence, not a measurement),
        but Fahrenheit, inches and knots are preserved exactly as reported.
        """
        frame = pd.read_csv(snapshot.path, dtype=str, keep_default_na=False)
        frame.columns = frame.columns.str.strip()

        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise SchemaError(
                f"{snapshot.path.name}: missing required columns {sorted(missing)}"
            )

        release = snapshot.release
        present = [c for c in COLUMN_TO_VARIABLE if c in frame.columns]

        for row in frame.to_dict("records"):
            station_id = str(row["STATION"]).strip()
            obs_date = _gsod_date(str(row["DATE"]).strip(), snapshot.path.name)
            for column in present:
                text = str(row.get(column, "")).strip()
                if not text:
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                if value == _SENTINELS.get(column):
                    continue

                flags: dict[str, str] = {}
                if column in _ATTRIBUTE_COLUMNS:
                    attribute = row.get(f"{column}_ATTRIBUTES")
                    if attribute is not None:
                        flags["attributes"] = str(attribute).strip()

                yield RawObservation(
                    source=self.name,
                    source_release=release,
                    source_station_id=station_id,
                    obs_date=obs_date,
                    element=column,
                    value=value,
                    units=COLUMN_UNITS.get(column, "unknown"),
                    provider_flags=flags,
                    source_record_id=f"{station_id}:{row['DATE']}:{column}",
                )


def _gsod_date(token: str, context: str) -> date:
    try:
        return datetime.strptime(token, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SchemaError(f"{context}: unparseable date {token!r}") from exc
