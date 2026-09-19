"""
types.py
========
Shared value types for the source-adapter contract defined in
RESEARCH_PLAN_AND_DESIGN.md section 1.5.

These types are the only vocabulary the adapters share with the rest of the
pipeline.  Keeping them here (rather than letting each adapter invent its own
return shape) is what stops a provider API change from reaching into
consolidation, QC or modelling code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class SchemaError(RuntimeError):
    """
    Raised when a provider's file or response no longer matches the structure
    this adapter was written against.

    Plan section 1.5 requires "an upstream schema-change failure that stops
    loudly rather than silently dropping columns".  Every parser therefore
    validates its expected fields and raises this rather than returning a
    partially-populated result.
    """


class AccessError(RuntimeError):
    """Raised when a provider refuses access (licence, credentials, embargo)."""


@dataclass(frozen=True)
class Domain:
    """
    Geographic admission box for the station search.

    'buffer_deg' widens the box on every side.  Plan section 1.2 asks for the
    training domain "and a configurable synoptic buffer" -- stations just
    outside the Delos target area still carry information about the weather
    systems that cross it, so they are admitted to the inventory.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    buffer_deg: float = 0.0

    def contains(self, lat: float, lon: float) -> bool:
        """True when (lat, lon) falls inside the buffered box."""
        if lat is None or lon is None:
            return False
        if pd.isna(lat) or pd.isna(lon):
            return False
        b = self.buffer_deg
        return (
            (self.lat_min - b) <= lat <= (self.lat_max + b)
            and (self.lon_min - b) <= lon <= (self.lon_max + b)
        )

    def as_cds_area(self) -> list[float]:
        """
        Buffered box as the CDS 'area' convention: [north, west, south, east].
        """
        b = self.buffer_deg
        return [
            self.lat_max + b,
            self.lon_min - b,
            self.lat_min - b,
            self.lon_max + b,
        ]


@dataclass(frozen=True)
class SourceRelease:
    """
    Provider release identity recorded once per acquisition run.

    Corresponds to the 'source_releases' table in plan section 1.7: source,
    version, retrieval time, licence, checksum and request manifest.
    """

    source: str
    release: str
    retrieved_at: str          # ISO-8601 UTC
    licence: str
    licence_url: str
    redistribution: str        # plain-language restriction, plan 1.2 question 4
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "release": self.release,
            "retrieved_at": self.retrieved_at,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "redistribution": self.redistribution,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DownloadRequest:
    """
    One unit of transfer, described completely enough to be replayed.

    'key' is the stable identity used for idempotency and for matching a
    request to an existing snapshot across runs.  It must not contain the
    release string, so that re-running against a new provider release can be
    recognised as the same logical request.

    Exactly one of 'url' or 'payload' is set: 'url' for plain HTTP sources,
    'payload' for API sources (Copernicus CDS) where there is no stable URL.

    'byte_ranges' carries (offset, length) pairs when only part of a remote
    file is needed -- used to pull single stations out of the multi-gigabyte
    ECA&D element archives without transferring the whole archive.
    """

    source: str
    key: str
    relative_path: str
    url: Optional[str] = None
    payload: Optional[dict[str, Any]] = None
    byte_ranges: Optional[tuple[tuple[int, int], ...]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.url is None) == (self.payload is None):
            raise ValueError(
                f"DownloadRequest {self.key!r} must set exactly one of url/payload"
            )

    def request_hash(self) -> str:
        """
        Deterministic hash of what was asked for (not of what came back).

        Lets a re-run prove that the same request was issued even when the
        provider has no version endpoint of its own.
        """
        body = json.dumps(
            {
                "source": self.source,
                "key": self.key,
                "url": self.url,
                "payload": self.payload,
                "byte_ranges": self.byte_ranges,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "key": self.key,
            "relative_path": self.relative_path,
            "url": self.url,
            "payload": self.payload,
            "byte_ranges": self.byte_ranges,
            "request_hash": self.request_hash(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SourceSnapshot:
    """
    A file that landed on disk, with everything needed to verify it again.

    Snapshots are immutable: plan section 1.5 says "never edit a raw snapshot
    in place".  A changed provider file produces a new snapshot under a new
    release directory rather than overwriting the old bytes.
    """

    request: DownloadRequest
    path: Path
    sha256: str
    n_bytes: int
    retrieved_at: str          # ISO-8601 UTC
    release: str
    status: str                # "downloaded" | "skipped_checksum_match"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.request.to_dict(),
            "path": str(self.path),
            "sha256": self.sha256,
            "n_bytes": self.n_bytes,
            "retrieved_at": self.retrieved_at,
            "release": self.release,
            "status": self.status,
        }


@dataclass(frozen=True)
class RawObservation:
    """
    One provider value, untouched.

    Plan section 1.4: "Store provider values and flags unchanged in a
    raw/bronze layer".  'value' keeps the provider's own number in the
    provider's own units (GHCN-D tenths of a degree, ECA&D 0.1 m/s, GSOD
    Fahrenheit), and 'units' records what those are.  Conversion belongs to
    standardize.py, which is outside this package.
    """

    source: str
    source_release: str
    source_station_id: str
    obs_date: date
    element: str               # the provider's own element name, e.g. "TMAX"
    value: float
    units: str
    provider_flags: dict[str, Any] = field(default_factory=dict)
    source_record_id: str = ""


# Inventory schema from plan section 1.2.  Every adapter's discover() must
# return exactly these columns so the coverage report can concatenate sources
# without special-casing any of them.
INVENTORY_COLUMNS: tuple[str, ...] = (
    "source",
    "source_release",
    "source_station_id",
    "wmo_id",
    "wigos_id",
    "name",
    "lat",
    "lon",
    "elevation_m",
    "variable",
    "provider_element",
    "units",
    "observing_interval",
    "timestamp_meaning",
    "first_date",
    "last_date",
    "n_obs_period",
    "missing_fraction_period",
    "access_status",
    "licence",
    "request",
    "retrieved_at",
)


# Columns that are identifiers rather than numbers, regardless of how they
# happen to be spelled.  See StationInventory.read_csv.
_IDENTIFIER_DTYPES: dict[str, type] = {
    "source_station_id": str,
    "wmo_id": str,
    "wigos_id": str,
    "provider_element": str,
}


class StationInventory:
    """
    Per-source coverage table: one row per station-variable combination.

    Wraps a DataFrame only to enforce the column contract at construction
    time, so a malformed adapter fails at its own boundary instead of
    producing a coverage report with silently missing fields.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [c for c in INVENTORY_COLUMNS if c not in frame.columns]
        if missing:
            raise SchemaError(f"inventory is missing required columns: {missing}")
        self.frame = frame.loc[:, list(INVENTORY_COLUMNS)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    @classmethod
    def empty(cls) -> "StationInventory":
        return cls(pd.DataFrame(columns=list(INVENTORY_COLUMNS)))

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "StationInventory":
        if not records:
            return cls.empty()
        return cls(pd.DataFrame.from_records(records))

    def n_stations(self) -> int:
        return int(self.frame["source_station_id"].nunique())

    def station_years(self) -> float:
        """
        Station-years present in the admission window.

        Counts observed days rather than nominal record length, so a station
        with a long but mostly empty record is not credited with coverage it
        does not have.
        """
        return float(pd.to_numeric(self.frame["n_obs_period"], errors="coerce").fillna(0).sum()) / 365.25

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)

    @classmethod
    def read_csv(cls, path: Path) -> "StationInventory":
        """
        Read an inventory back, keeping identifiers as text.

        Without an explicit dtype, pandas infers numeric types and silently
        strips leading zeros: the GSOD station '03772099999' comes back as
        3772099999 and its download URL then points at a file that does not
        exist, while the ECA&D station '000247' loses its zero padding.  Every
        identifier-like column is therefore read as a string.
        """
        return cls(pd.read_csv(path, dtype=_IDENTIFIER_DTYPES, keep_default_na=True))
