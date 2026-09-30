"""
sources/registry.py
===================
One place that says which sources exist and what each is allowed to do.

WHY THIS EXISTS
---------------
Source identity used to be spread across seven unsynchronised lists -- an
adapter map in the reader, another in the CLI, a default-sources tuple, and
four tuples in the configuration naming which sources join the crosswalk,
which can supply a per-station series, and which may have their values
selected. Nothing checked them against each other, so adding or renaming a
source meant editing six places and a miss was silent: omit a source from the
series list and it quietly joins only at tier 1 with no error anywhere.

This module replaces all of them. A source is declared once, with the three
capabilities the pipeline actually branches on, and validate() refuses a
configuration that names a source which cannot do what is being asked of it.

WHAT THE CAPABILITIES MEAN
--------------------------
has_station_series
    A per-station daily series can be read from bronze. False for a provider
    that delivers whole-period files covering every station at once (C3S), so
    tier-2/3 identity tests -- which need a series -- must not be attempted.
    Attempting them would produce rejections reading "insufficient overlapping
    days", which falsely implies the series had been compared.

joins_crosswalk
    The source is admitted to the canonical station crosswalk. A source can
    join on an exact identifier (tier 1) without having a readable series.

selectable
    The consolidator may choose this source's values. Kept separate from
    joins_crosswalk so an audit-only source cannot silently become a fallback
    value provider: C3S redelivers GHCN-D in this domain, so it is matched and
    audited but never selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from station_archive.sources.base import StationSource
from station_archive.sources.c3s_land import C3sLandSource
from station_archive.sources.ecad import EcadSource
from station_archive.sources.ghcnd import GhcndSource
from station_archive.sources.gsod import GsodSource


@dataclass(frozen=True)
class SourceSpec:
    """One provider and what the pipeline may do with it."""

    name: str
    adapter: type[StationSource]
    has_station_series: bool
    joins_crosswalk: bool
    selectable: bool
    notes: str = ""


class UnknownSourceError(KeyError):
    """Raised when a source name has no entry in the registry."""


class PolicyConsistencyError(ValueError):
    """
    Raised when a configuration asks a source to do something it cannot.

    This is the check whose absence made the old seven-list arrangement
    dangerous: a typo or an omission produced silently wrong behaviour rather
    than a failure.
    """


# The four sources this archive was built from. Adding a fifth means adding one
# entry here -- not six edits across three modules.
REGISTRY: Mapping[str, SourceSpec] = {
    "ghcnd": SourceSpec(
        name="ghcnd",
        adapter=GhcndSource,
        has_station_series=True,
        joins_crosswalk=True,
        selectable=True,
    ),
    "ecad": SourceSpec(
        name="ecad",
        adapter=EcadSource,
        has_station_series=True,
        joins_crosswalk=True,
        selectable=True,
    ),
    "gsod": SourceSpec(
        name="gsod",
        adapter=GsodSource,
        has_station_series=True,
        joins_crosswalk=True,
        selectable=True,
    ),
    "c3s_land": SourceSpec(
        name="c3s_land",
        adapter=C3sLandSource,
        has_station_series=False,
        joins_crosswalk=True,
        selectable=False,
        notes=(
            "Delivers whole-year files covering every station at once, so there "
            "is no per-station snapshot to read a series from. Every one of its "
            "stations in the original domain carries an exact GHCN-D identifier, "
            "making it a redelivery rather than independent evidence: matched at "
            "tier 1 and kept as a value-level cross-check, never selected."
        ),
    ),
}


def spec(name: str) -> SourceSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise UnknownSourceError(
            f"no source named {name!r}; known sources are {sorted(REGISTRY)}"
        ) from exc


def adapter_for(name: str) -> StationSource:
    """A fresh adapter instance for one source."""
    return spec(name).adapter()


def all_sources() -> tuple[str, ...]:
    return tuple(REGISTRY)


def crosswalk_sources() -> tuple[str, ...]:
    """Sources admitted to the canonical crosswalk."""
    return tuple(n for n, s in REGISTRY.items() if s.joins_crosswalk)


def series_sources() -> tuple[str, ...]:
    """Sources whose per-station daily series can be read from bronze."""
    return tuple(n for n, s in REGISTRY.items() if s.has_station_series)


def selectable_sources() -> tuple[str, ...]:
    """Sources whose values the consolidator may choose."""
    return tuple(n for n, s in REGISTRY.items() if s.selectable)


def validate_priority(priority: Mapping[str, Iterable[str]]) -> None:
    """
    Check a per-variable source priority against the registry.

    Refuses three things the old arrangement would have accepted silently:
    an unknown source name, a source that may not be selected appearing in a
    priority list, and the same source listed twice for one variable.
    """
    for variable, order in priority.items():
        order = tuple(order)
        for name in order:
            if name not in REGISTRY:
                raise PolicyConsistencyError(
                    f"source priority for {variable!r} names unknown source "
                    f"{name!r}; known sources are {sorted(REGISTRY)}"
                )
            if not REGISTRY[name].selectable:
                raise PolicyConsistencyError(
                    f"source priority for {variable!r} names {name!r}, which is "
                    f"not selectable: {REGISTRY[name].notes or 'audit-only source'}"
                )
        if len(set(order)) != len(order):
            raise PolicyConsistencyError(
                f"source priority for {variable!r} repeats a source: {order}"
            )
