"""
sources/ecad.py
===============
European Climate Assessment & Dataset (ECA&D) daily station adapter.

DATA SOURCE
-----------
https://www.ecad.eu/dailydata/  -- predefined blended series, published as one
ZIP archive per element on the KNMI asset bucket:
  https://knmi-ecad-assets-prd.s3.amazonaws.com/download/ECA_blend_<elem>.zip

LICENCE
-------
The header of every series file states: "THESE DATA CAN BE USED FOR
NON-COMMERCIAL RESEARCH AND EDUCATION PROVIDED THAT THE FOLLOWING SOURCE IS
ACKNOWLEDGED".  Plan section 1.1 flags this as the principal caution for
ECA&D, and section 1.2 question 4 asks which records cannot be redistributed.
The restriction is recorded on every snapshot.

WHY RANGED EXTRACTION
---------------------
ECA&D offers no per-station programmatic endpoint; the archives are the only
scriptable route, and the seven elements this project needs total about
5.4 GB compressed (rr alone is 1.84 GB).  Of roughly 3000 stations per
archive, only the few dozen inside the eastern-Mediterranean domain are
wanted.

The S3 bucket honours HTTP Range requests, so ziprange.RemoteZip reads each
archive's index from its tail and fetches only the wanted members.  This
replaces a 5.4 GB transfer with a few megabytes -- which matters on a machine
whose repository drive has 64 MB free.

FORMATS
-------
stations.txt inside each archive (19 header lines, then six comma-separated
fields): STAID, STANAME, CN, LAT, LON, HGHT.  Its header block describes fixed
column positions, but the delivered rows do not honour them exactly (CN is
padded wider than its documented two characters), so the fields are split on
commas instead.  LAT/LON are degrees:minutes:seconds, e.g. "+56:52:00" /
"+014:48:00".

<ELEM>_STAID<6 digits>.txt (19 header lines, then comma-separated):
    STAID, SOUID, DATE, <ELEM>, Q_<ELEM>
    Missing value -9999; Q is 0 valid, 1 suspect, 9 missing.

date_timestamp.txt inside the archive carries the release stamp.  It is read
from inside the ZIP rather than from the standalone stations.txt on the
bucket, because those two were observed to disagree (12-08-2026 inside the
archive vs 18-07-2023 for the loose file).
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import pandas as pd

from station_archive import net, ziprange
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

BASE_URL = "https://knmi-ecad-assets-prd.s3.amazonaws.com/download/"

# ECA&D element code -> canonical variable.
# TG is the blended daily mean; TN/TX the daily minimum/maximum; RR daily
# precipitation; HU relative humidity; FG daily mean wind speed; FX wind gust.
ELEMENT_TO_VARIABLE: dict[str, str] = {
    "TG": "tas",
    "TN": "tasmin",
    "TX": "tasmax",
    "RR": "pr",
    "HU": "hurs",
    "FG": "sfcWind",
    "FX": "gust",
}

# Units as declared in each series file's header block.
ELEMENT_UNITS: dict[str, str] = {
    "TG": "degC_tenths",
    "TN": "degC_tenths",
    "TX": "degC_tenths",
    "RR": "mm_tenths",
    "HU": "percent",
    "FG": "m_s_tenths",
    "FX": "m_s_tenths",
}

# The six comma-separated fields of stations.txt, in order:
# STAID, STANAME, CN (country), LAT, LON, HGHT.
_STATION_FIELDS = ("source_station_id", "name", "country", "lat", "lon", "elevation_m")

_MISSING_VALUE = -9999

# Quality codes from the file header: 0 valid, 1 suspect, 9 missing.
_QUALITY_MISSING = 9

_MEMBER_PATTERN = re.compile(r"^(?P<element>[A-Z]{2})_STAID(?P<staid>\d{6})\.txt$")


class EcadSource(StationSource):
    name = "ecad"
    licence = "Non-commercial research and education only; acknowledgement required"
    licence_url = "https://www.ecad.eu/documents/ECAD_datapolicy.pdf"
    redistribution = (
        "Not redistributable. Public ECA&D series are for non-commercial research "
        "and education; some contributing series are not downloadable at all."
    )
    element_to_variable = ELEMENT_TO_VARIABLE
    element_units = ELEMENT_UNITS

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url
        self._release: str | None = None
        self._archives: dict[str, ziprange.RemoteZip] = {}

    # ---- release ---------------------------------------------------------

    def archive_url(self, element: str) -> str:
        return f"{self.base_url}ECA_blend_{element.lower()}.zip"

    def _archive(self, element: str) -> ziprange.RemoteZip:
        """Open (once per element) a ranged reader over that element's archive."""
        if element not in self._archives:
            self._archives[element] = ziprange.RemoteZip(self.archive_url(element))
        return self._archives[element]

    def release(self) -> str:
        """
        Release stamp from date_timestamp.txt inside the TG archive.

        All element archives are rebuilt together, so one stamp identifies the
        release.  Read from inside the ZIP because the loose stations.txt on
        the bucket carries a different, older date.
        """
        if self._release is None:
            archive = self._archive("TG")
            if "date_timestamp.txt" not in archive.entries:
                raise SchemaError("ECA_blend_tg.zip no longer contains date_timestamp.txt")
            stamp = archive.read("date_timestamp.txt").decode("latin1").strip()
            self._release = _release_date(stamp)
        return self._release

    # ---- discovery -------------------------------------------------------

    def discover(
        self,
        domain: Domain,
        variables: Sequence[str],
        period: tuple[int, int],
    ) -> StationInventory:
        """
        Inventory from each element archive's own stations.txt and member list.

        A station appears for an element only when that element's archive
        actually contains its series file, so the inventory reflects what can
        be downloaded rather than what the metadata advertises.

        Observation counts are left empty here: counting them would mean
        fetching every series, which is exactly what the plan's
        discovery-before-download rule is trying to avoid.  parse() fills them
        once the data is on disk.
        """
        wanted_elements = [
            element
            for element, variable in ELEMENT_TO_VARIABLE.items()
            if variable in variables
        ]

        release = self.release()
        retrieved_at = net.utc_now()
        records = []

        for element in wanted_elements:
            archive = self._archive(element)
            if "stations.txt" not in archive.entries:
                raise SchemaError(f"ECA_blend_{element.lower()}.zip has no stations.txt")

            stations = _parse_stations(archive.read("stations.txt").decode("latin1"), element)
            available = {
                match.group("staid")
                for name in archive.namelist()
                if (match := _MEMBER_PATTERN.match(name)) and match.group("element") == element
            }

            for row in stations.itertuples(index=False):
                if not domain.contains(row.lat, row.lon):
                    continue
                staid = f"{int(row.source_station_id):06d}"
                if staid not in available:
                    # Advertised in metadata but the series is not in the
                    # public archive -- plan 1.1: "some series are
                    # non-downloadable".  Recorded so the gap is visible.
                    access_status = "restricted_or_absent"
                else:
                    access_status = "public"

                records.append(
                    {
                        "source": self.name,
                        "source_release": release,
                        "source_station_id": staid,
                        "wmo_id": "",
                        "wigos_id": "",
                        "name": row.name,
                        "lat": row.lat,
                        "lon": row.lon,
                        "elevation_m": row.elevation_m,
                        "variable": ELEMENT_TO_VARIABLE[element],
                        "provider_element": element,
                        "units": ELEMENT_UNITS[element],
                        "observing_interval": "daily",
                        "timestamp_meaning": "station-defined daily window (blended series)",
                        "first_date": pd.NA,
                        "last_date": pd.NA,
                        "n_obs_period": pd.NA,
                        "missing_fraction_period": pd.NA,
                        "access_status": access_status,
                        "licence": self.licence,
                        "request": f"{self.archive_url(element)}#{element}_STAID{staid}.txt",
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
        One request per station-element, each carrying the byte range of its
        member inside the remote archive.

        'policy' is unused: ECA&D publishes whole series with no year window.

        The byte range is computed from the central directory alone, with no
        further network access, so planning a thousand stations costs nothing
        beyond the index already read.  Recording it puts the exact transfer
        into the request manifest, so the manifest describes a replayable
        operation rather than "some part of a 1.8 GB file".
        """
        frame = inventory.frame
        downloadable = frame[frame["access_status"] == "public"]

        requests: list[DownloadRequest] = []
        for row in downloadable.itertuples(index=False):
            element = row.provider_element
            staid = str(row.source_station_id).zfill(6)
            member = f"{element}_STAID{staid}.txt"
            archive = self._archive(element)
            if member not in archive.entries:
                continue
            offset, length = archive.fetch_range(member)
            requests.append(
                DownloadRequest(
                    source=self.name,
                    key=f"ecad/{element}/{staid}",
                    relative_path=f"{element}/{member}",
                    url=self.archive_url(element),
                    byte_ranges=((offset, length),),
                    metadata={
                        "member": member,
                        "element": element,
                        "source_station_id": staid,
                    },
                )
            )
        return requests

    def _transfer(self, request: DownloadRequest, destination: Path) -> None:
        """
        Fetch one member's compressed bytes and store it decompressed.

        RemoteZip.read verifies the member CRC, which is what catches a
        mis-offset range request -- the failure mode a plain byte-range copy
        would store as a valid-looking snapshot.
        """
        member = request.metadata["member"]
        element = request.metadata["element"]
        data = self._archive(element).read(member)
        net.write_bytes_atomic(data, destination)

    # ---- parsing ---------------------------------------------------------

    def parse(self, snapshot: SourceSnapshot) -> Iterator[RawObservation]:
        """
        Stream one ECA&D series file as untouched provider values.

        Values stay in the provider's 0.1 units.  Rows flagged Q=9 (missing)
        are skipped; Q=1 (suspect) rows are kept with the flag attached,
        because deciding what to do with a suspect value is qc.py's call, not
        the downloader's.
        """
        text = snapshot.path.read_text(encoding="latin1")
        header, body = _split_header(text, snapshot.path.name)
        element = _element_from_header(header, snapshot.path.name)

        frame = pd.read_csv(
            io.StringIO(body),
            skipinitialspace=True,
            dtype=str,
            keep_default_na=False,
        )
        frame.columns = [c.strip() for c in frame.columns]

        expected = {"STAID", "SOUID", "DATE", element, f"Q_{element}"}
        missing = expected - set(frame.columns)
        if missing:
            raise SchemaError(
                f"{snapshot.path.name}: missing expected columns {sorted(missing)}; "
                f"found {list(frame.columns)}"
            )

        release = snapshot.release
        units = ELEMENT_UNITS.get(element, "unknown")
        for row in frame.itertuples(index=False):
            quality = int(getattr(row, f"Q_{element}"))
            value = float(getattr(row, element))
            if quality == _QUALITY_MISSING or value == _MISSING_VALUE:
                continue
            staid = str(row.STAID).strip()
            souid = str(row.SOUID).strip()
            yield RawObservation(
                source=self.name,
                source_release=release,
                source_station_id=staid.zfill(6),
                obs_date=_ecad_date(str(row.DATE).strip(), snapshot.path.name),
                element=element,
                value=value,
                units=units,
                # SOUID identifies which contributing series a blended value
                # came from; identity.py needs it to reason about overlap.
                provider_flags={"quality_code": quality, "souid": souid},
                source_record_id=f"{staid}:{souid}:{row.DATE}:{element}",
            )


def _release_date(stamp: str) -> str:
    """
    Reduce date_timestamp.txt to an ISO date usable as a directory name.

    The file holds a sentence, not a bare timestamp -- observed as
    "EUROPEAN CLIMATE ASSESSMENT & DATASET (ECA&D), file created on 12-08-2026".
    Using it verbatim would put ampersands, parentheses and commas into the
    archive path, so the DD-MM-YYYY date is extracted and reversed to sort
    chronologically.
    """
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", stamp)
    if not match:
        raise SchemaError(
            f"no DD-MM-YYYY date in ECA&D date_timestamp.txt: {stamp[:120]!r}"
        )
    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


def _split_header(text: str, context: str) -> tuple[str, str]:
    """
    Separate the descriptive header from the CSV body.

    The body begins at the line naming the STAID column; searching for that
    rather than skipping a fixed 19 lines means a changed header length is
    tolerated while a changed column layout still raises downstream.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.replace(" ", "").startswith("STAID,"):
            return "\n".join(lines[:index]), "\n".join(lines[index:])
    raise SchemaError(f"{context}: no 'STAID,' column header line found")


def _element_from_header(header: str, context: str) -> str:
    """Read the element code from the file's own FILE FORMAT block."""
    for line in header.splitlines():
        match = re.match(r"\s*\d+-\d+\s+([A-Z]{2})\s*:", line)
        if match and match.group(1) not in ("Q",):
            code = match.group(1)
            if code in ELEMENT_UNITS:
                return code
    raise SchemaError(f"{context}: could not identify the element code from the header")


def _ecad_date(token: str, context: str) -> date:
    try:
        return datetime.strptime(token, "%Y%m%d").date()
    except ValueError as exc:
        raise SchemaError(f"{context}: unparseable date {token!r}") from exc


def _parse_stations(text: str, context: str) -> pd.DataFrame:
    """
    Parse an ECA&D stations.txt into a frame with decimal coordinates.

    The file's own header describes fixed-width columns, but the delivered
    data does not match those offsets exactly (the CN field is padded wider
    than its documented two characters).  Splitting on commas is what the data
    actually supports: every data line in the published file resolves to
    exactly six fields, and no station name contains a comma.

    The field-count check doubles as the schema guard -- a station name that
    ever did contain a comma, or a new column, would raise here rather than
    silently shift every coordinate.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.replace(" ", "").startswith("STAID,STANAME"):
            start = index + 1
            break
    if start is None:
        raise SchemaError(f"{context} stations.txt: no 'STAID,STANAME' header line")

    rows = []
    for number, line in enumerate(lines[start:], start=start + 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(_STATION_FIELDS):
            raise SchemaError(
                f"{context} stations.txt line {number}: expected "
                f"{len(_STATION_FIELDS)} comma-separated fields, found {len(parts)}"
            )
        staid, name, country, lat_dms, lon_dms, height = parts
        rows.append(
            {
                "source_station_id": staid,
                "name": name,
                "country": country,
                "lat": _dms_to_degrees(lat_dms, context),
                "lon": _dms_to_degrees(lon_dms, context),
                "elevation_m": pd.to_numeric(height, errors="coerce"),
            }
        )

    if not rows:
        raise SchemaError(f"{context} stations.txt: no station rows")
    return pd.DataFrame(rows)


def _dms_to_degrees(token: str, context: str) -> float:
    """
    Convert ECA&D's '+DD:MM:SS' / '+DDD:MM:SS' coordinates to decimal degrees.

    The sign belongs to the whole coordinate, so minutes and seconds are
    subtracted for southern/western stations rather than added -- getting this
    backwards would place them on the wrong side of the equator or meridian.
    """
    token = token.strip().rstrip(",")
    parts = token.split(":")
    if len(parts) != 3:
        raise SchemaError(f"{context}: malformed coordinate {token!r}")
    try:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError as exc:
        raise SchemaError(f"{context}: malformed coordinate {token!r}") from exc

    sign = -1.0 if token.lstrip().startswith("-") else 1.0
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)
