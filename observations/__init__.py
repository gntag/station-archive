"""
observations
============
Source adapters and discovery for the Delos station-observation archive
(RESEARCH_PLAN_AND_DESIGN.md, Step 1).

Scope of this package is deliberately narrow.  Per the Step 1 architecture
guardrail: "Source adapters fetch and parse provider data; they do not decide
station identity or source priority."  Identity matching, quality control,
per-variable source priority and gap filling live in separate modules
(identity.py, qc.py, consolidate.py) that consume what this package produces.

What this package does:
  * discover   -- build a per-source station/variable inventory (plan 1.2)
  * plan       -- turn an inventory into concrete download requests
  * fetch      -- restartable, idempotent, checksummed transfer (plan 1.5)
  * parse      -- read a snapshot into untouched provider values (bronze)

What this package must never do:
  * convert units, reconcile duplicate stations, rank sources, fill gaps,
    or interpolate anything.
"""

from observations.types import (
    Domain,
    DownloadRequest,
    RawObservation,
    SchemaError,
    SourceRelease,
    SourceSnapshot,
    StationInventory,
)

__all__ = [
    "Domain",
    "DownloadRequest",
    "RawObservation",
    "SchemaError",
    "SourceRelease",
    "SourceSnapshot",
    "StationInventory",
]
