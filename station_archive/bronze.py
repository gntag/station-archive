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
from typing import Optional

import pandas as pd

from station_archive import manifests, qc
from station_archive.sources import registry
from station_archive.sources.base import StationSource
from station_archive.types import DownloadRequest, SourceSnapshot


def adapter_for(source: str) -> StationSource:
    """
    A fresh adapter for one source.

    Delegates to the registry rather than keeping a second adapter map here --
    two maps of the same thing is how a source ends up fetchable but
    unreadable, or readable but absent from the CLI.
    """
    return registry.adapter_for(source)


def latest_release(source: str) -> Optional[str]:
    """
    The release to read a source's series from: the most recent one that
    actually has snapshots.

    Reads source_releases.jsonl rather than listing directories, so a
    half-written or hand-copied release folder is not mistaken for an
    acquisition.

    The "has snapshots" condition is not a refinement, it is the fix for a
    silent-emptying failure. A release row is written by `discover` as well as
    by `fetch`, and a source whose release string is a retrieval date gets a
    NEW row every day. So running `discover` the day after a `fetch` recorded a
    release with no snapshots behind it, and taking the newest row made every
    station in that source resolve to an empty series -- the source dropped out
    of the crosswalk and the gold table with nothing but a miss count to show
    for it.

    Falls back to the newest recorded release when none has snapshots, so a
    freshly discovered, not-yet-fetched source still reports a release rather
    than None.
    """
    releases = [r for r in manifests.read_releases() if r.get("source") == source]
    if not releases:
        return None
    newest_first = sorted(
        releases, key=lambda r: r.get("retrieved_at", ""), reverse=True
    )
    for record in newest_first:
        release = record["release"]
        if manifests.snapshots_path(source, release).exists():
            return release
    return newest_first[0]["release"]


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


def all_daily_series(
    source: str,
    snapshots: list[SourceSnapshot],
    period: Optional[tuple[int, int]] = None,
    rejections: Optional[dict[tuple[str, str], int]] = None,
) -> dict[str, pd.Series]:
    """
    Every canonical variable a station offers, from ONE pass over its files.

    Reading one variable at a time would re-parse the same snapshots: a GSOD
    station has 51 station-year files and seven variables, so the naive route
    parses 357 files where 51 suffice.  Across 1600 stations that is the
    difference between minutes and hours.

    This is the ONLY reader. A per-variable daily_series() used to sit beside
    it, unreferenced, and it skipped the qc.reject_reason step below -- so the
    obvious-sounding name returned values with GSOD's flag-I false zeros left
    in. Removed rather than kept as a convenience.

    Values stay in provider units, as everywhere in this module.

    Project QC is applied here rather than later because it needs the
    provider flags, which exist on the parsed observation and are not carried
    into the series.  Rejections are counted into 'rejections' by
    (variable, reason) so the gold layer can report "accepted values plus
    transparent rejection reasons" (plan section 1.4) without materialising a
    flag string for every one of tens of millions of rows.
    """
    adapter = adapter_for(source)
    mapping = adapter.element_to_variable

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
