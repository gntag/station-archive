"""
bronze.py
=========
Read parsed observations back out of the bronze snapshot layer.

WHY THIS MODULE EXISTS
----------------------
The plan's module list in section 1.5 is described as "Recommended module
boundaries", and does not name a reader.  Both identity.py (which needs
overlapping daily series to test whether two source stations are one physical
station) and the forthcoming standardize.py (which needs every value) have to
locate and parse snapshots.  Duplicating that lookup in each would be worse
than naming it once here.

This module reads; it does not convert, judge or select.  Unit conversion is
standardize.py's, quality judgement is qc.py's, and source ranking is
consolidate.py's.  Values come back exactly as the adapters parsed them, in
provider units.

HOW SNAPSHOTS ARE LOCATED
-------------------------
From the snapshot manifests written during acquisition, not by walking the
directory tree.  The manifest is the record of what was actually fetched and
verified, so a file on disk that no manifest vouches for is not read -- which
is what keeps "the archive" a defined set rather than whatever happens to be
in the folder.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from observations import config, manifests, qc
from observations.sources.base import StationSource
from observations.sources.c3s_land import C3sLandSource
from observations.sources.ecad import EcadSource
from observations.sources.ghcnd import GhcndSource
from observations.sources.gsod import GsodSource
from observations.types import DownloadRequest, SourceSnapshot

_ADAPTERS: dict[str, type[StationSource]] = {
    "ghcnd": GhcndSource,
    "ecad": EcadSource,
    "gsod": GsodSource,
    "c3s_land": C3sLandSource,
}


def adapter_for(source: str) -> StationSource:
    if source not in _ADAPTERS:
        raise KeyError(f"no adapter for source {source!r}")
    return _ADAPTERS[source]()


def latest_release(source: str) -> Optional[str]:
    """
    Most recently recorded release for a source.

    Reads source_releases.jsonl rather than listing directories, so a
    half-written or hand-copied release folder is not mistaken for an
    acquisition.
    """
    releases = [r for r in manifests.read_releases() if r.get("source") == source]
    if not releases:
        return None
    return max(releases, key=lambda r: r.get("retrieved_at", ""))["release"]


def snapshot_records(source: str, release: str) -> list[dict]:
    """Raw snapshot manifest rows for one source release."""
    path = manifests.snapshots_path(source, release)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _rebuild_snapshot(record: dict) -> SourceSnapshot:
    """
    Reconstruct a SourceSnapshot from its manifest row so an adapter's
    parse() can consume it.

    Only the fields parse() actually reads are needed, but the request is
    rebuilt faithfully so anything inspecting provenance downstream sees the
    same object shape the fetch produced.
    """
    request = DownloadRequest(
        source=record["source"],
        key=record["key"],
        relative_path=record["relative_path"],
        url=record.get("url"),
        payload=record.get("payload") if record.get("url") is None else None,
        metadata=record.get("metadata") or {},
    )
    return SourceSnapshot(
        request=request,
        path=Path(record["path"]),
        sha256=record["sha256"],
        n_bytes=record["n_bytes"],
        retrieved_at=record["retrieved_at"],
        release=record["release"],
        status=record["status"],
    )


def snapshots_by_station(source: str, release: str) -> dict[str, list[SourceSnapshot]]:
    """
    Map source_station_id -> its snapshots.

    A station has one snapshot for GHCN-D (a whole-record file), one per
    element for ECA&D, and one per year for GSOD.  C3S delivers whole years
    covering every station at once, so it has no per-station mapping and
    returns empty -- callers needing C3S series must scan its yearly
    deliveries instead.
    """
    grouped: dict[str, list[SourceSnapshot]] = defaultdict(list)
    for record in snapshot_records(source, release):
        station_id = (record.get("metadata") or {}).get("source_station_id")
        if station_id is None:
            continue
        grouped[str(station_id)].append(_rebuild_snapshot(record))
    return dict(grouped)


def daily_series(
    source: str,
    station_id: str,
    variable: str,
    snapshots: list[SourceSnapshot],
    period: Optional[tuple[int, int]] = None,
) -> pd.Series:
    """
    Daily values for one station-variable, indexed by date, in PROVIDER units.

    Returns an empty Series when the station has no values for the variable,
    which is an ordinary outcome rather than an error: most stations report
    only a subset of variables.

    Duplicate dates within a source are collapsed by keeping the first, and
    the count of collapsed rows is discarded here -- detecting and reporting
    repeated days is a quality-control check (section 1.4), not a reader's
    job.  Keeping the first rather than averaging them preserves section
    1.3's rule never to average duplicates.
    """
    adapter = adapter_for(source)
    wanted_elements = _elements_for(source, variable)
    if not wanted_elements:
        return pd.Series(dtype="float64")

    dates: list = []
    values: list[float] = []
    for snapshot in snapshots:
        for observation in adapter.parse(snapshot):
            if observation.element not in wanted_elements:
                continue
            if period is not None and not (period[0] <= observation.obs_date.year <= period[1]):
                continue
            dates.append(observation.obs_date)
            values.append(observation.value)

    if not dates:
        return pd.Series(dtype="float64")

    series = pd.Series(values, index=pd.to_datetime(dates), dtype="float64")
    series = series[~series.index.duplicated(keep="first")]
    return series.sort_index()


def _element_to_variable(source: str) -> dict[str, str]:
    """
    A source's full provider-element -> canonical-variable mapping.

    Read from each adapter rather than restated, so the two cannot drift.
    """
    if source == "ghcnd":
        from observations.sources.ghcnd import ELEMENT_TO_VARIABLE

        return dict(ELEMENT_TO_VARIABLE)
    if source == "ecad":
        from observations.sources.ecad import ELEMENT_TO_VARIABLE

        return dict(ELEMENT_TO_VARIABLE)
    if source == "gsod":
        from observations.sources.gsod import COLUMN_TO_VARIABLE

        return dict(COLUMN_TO_VARIABLE)
    if source == "c3s_land":
        from observations.sources.c3s_land import VARIABLE_TO_CDS

        return {cds: canonical for canonical, cds in VARIABLE_TO_CDS.items()}
    raise KeyError(f"no element mapping for source {source!r}")


def all_daily_series(
    source: str,
    snapshots: list[SourceSnapshot],
    period: Optional[tuple[int, int]] = None,
    rejections: Optional[dict[tuple[str, str], int]] = None,
) -> dict[str, pd.Series]:
    """
    Every canonical variable a station offers, from ONE pass over its files.

    Calling daily_series once per variable re-reads the same snapshots: a GSOD
    station has 51 station-year files and seven variables, so the naive route
    parses 357 files where 51 suffice.  Across 1600 stations that is the
    difference between minutes and hours.

    Values stay in provider units, as everywhere in this module.

    Project QC is applied here rather than later because it needs the
    provider flags, which exist on the parsed observation and are not carried
    into the series.  Rejections are counted into 'rejections' by
    (variable, reason) so the gold layer can report "accepted values plus
    transparent rejection reasons" (plan section 1.4) without materialising a
    flag string for every one of tens of millions of rows.
    """
    adapter = adapter_for(source)
    mapping = _element_to_variable(source)

    dates: dict[str, list] = {}
    values: dict[str, list[float]] = {}

    for snapshot in snapshots:
        for observation in adapter.parse(snapshot):
            variable = mapping.get(observation.element)
            if variable is None:
                continue
            if period is not None and not (period[0] <= observation.obs_date.year <= period[1]):
                continue

            reason = qc.reject_reason(
                source, variable, observation.value, observation.provider_flags
            )
            if reason is not None:
                if rejections is not None:
                    key = (variable, reason)
                    rejections[key] = rejections.get(key, 0) + 1
                continue

            dates.setdefault(variable, []).append(observation.obs_date)
            values.setdefault(variable, []).append(observation.value)

    out: dict[str, pd.Series] = {}
    for variable, when in dates.items():
        series = pd.Series(values[variable], index=pd.to_datetime(when), dtype="float64")
        series = series[~series.index.duplicated(keep="first")]
        out[variable] = series.sort_index()
    return out


def _elements_for(source: str, variable: str) -> set[str]:
    """
    Provider element names that carry a canonical variable for one source.

    Read from each adapter's own mapping rather than restated here, so the
    two cannot drift apart.
    """
    if source == "ghcnd":
        from observations.sources.ghcnd import ELEMENT_TO_VARIABLE

        return {e for e, v in ELEMENT_TO_VARIABLE.items() if v == variable}
    if source == "ecad":
        from observations.sources.ecad import ELEMENT_TO_VARIABLE

        return {e for e, v in ELEMENT_TO_VARIABLE.items() if v == variable}
    if source == "gsod":
        from observations.sources.gsod import COLUMN_TO_VARIABLE

        return {c for c, v in COLUMN_TO_VARIABLE.items() if v == variable}
    if source == "c3s_land":
        from observations.sources.c3s_land import VARIABLE_TO_CDS

        cds = VARIABLE_TO_CDS.get(variable)
        return {cds} if cds else set()
    raise KeyError(f"no element mapping for source {source!r}")


class SeriesLoader:
    """
    Lazily loads and caches daily series per (source, station, variable).

    Identity matching asks for the same station's series repeatedly as it is
    compared against several neighbours, and parsing an ECA&D or GHCN-D file
    costs tens of thousands of rows each time.  Caching makes the crosswalk
    practical; the cache is per-instance so it is released with the loader.
    """

    def __init__(self, period: Optional[tuple[int, int]] = None) -> None:
        self.period = period
        self._releases: dict[str, Optional[str]] = {}
        self._snapshots: dict[str, dict[str, list[SourceSnapshot]]] = {}
        self._station_series: dict[tuple[str, str], dict[str, pd.Series]] = {}
        self.misses: list[tuple[str, str]] = []
        # (source, variable, reason) -> count, accumulated across the run.
        self.rejections: dict[tuple[str, str, str], int] = {}

    def _release(self, source: str) -> Optional[str]:
        if source not in self._releases:
            self._releases[source] = latest_release(source)
        return self._releases[source]

    def _station_map(self, source: str) -> dict[str, list[SourceSnapshot]]:
        if source not in self._snapshots:
            release = self._release(source)
            self._snapshots[source] = (
                snapshots_by_station(source, release) if release else {}
            )
        return self._snapshots[source]

    def clear(self) -> None:
        """
        Drop cached series, keeping the snapshot index.

        Consolidation walks 1600+ stations across seven variables; holding
        every series would exhaust memory long before the end.  The snapshot
        index is cheap and stays, so clearing costs only re-parsing.
        """
        self._station_series.clear()      # rejection tallies deliberately survive

    def series(self, source: str, station_id: str, variable: str) -> pd.Series:
        """
        One station-variable series, in provider units.

        The first request for a station parses its snapshots once and keeps
        every variable it found; later requests for the same station are
        served from that single parse.
        """
        key = (source, station_id)
        if key not in self._station_series:
            snapshots = self._station_map(source).get(station_id, [])
            if not snapshots:
                self.misses.append((source, station_id))
                self._station_series[key] = {}
            else:
                counts: dict[tuple[str, str], int] = {}
                self._station_series[key] = all_daily_series(
                    source, snapshots, self.period, counts
                )
                for (variable_name, reason), n in counts.items():
                    tally = (source, variable_name, reason)
                    self.rejections[tally] = self.rejections.get(tally, 0) + n
        return self._station_series[key].get(variable, pd.Series(dtype="float64"))
