"""
station_archive
===============
Reconcile several overlapping public climate sources into one daily
station-observation archive where every value is traceable.

Four providers (GHCN-Daily, ECA&D, GSOD, C3S land in-situ) describe an
overlapping set of physical stations under different identifiers and in
different units.  This package resolves that into one archive: which records
are one station, which value wins on a given day, and why -- recording the
evidence for both at every step.

The pipeline, each stage reading what the previous one wrote:

    discover -> report -> fetch -> identity -> review -> consolidate
                                                      -> homogeneity
                                                      -> provenance

`python -m station_archive.cli demo` runs the whole chain on synthetic data in
seconds, with no network and no credentials.  It exercises the real adapters
and the real decision code, so it is the cheapest accurate picture of what the
pipeline does.

WHAT THIS PACKAGE MUST NEVER DO
-------------------------------
Invent a value.  No interpolation, no averaging of two sources, no bias
correction, no coordinate averaging.  A day with no accepted candidate stays
missing, and a breakpoint is diagnosed rather than corrected.  AGENTS.md states
the seven invariants in full; each one is tested.

Reading the archive into a modelling schema, fold design and the choice of
which stations to score belong to the consuming project, not here.

Where the documentation lives, in two files:
  * README.md  -- the architecture, how to run it, and what it produces
  * AGENTS.md  -- everything else: the seven invariants, every threshold with the
                  measurement behind it, the full output schema column by column,
                  how each stage decides, how to add a source, and the design
                  requirements the docstrings cite by section number
Version history is in the GitHub releases and the commit log.
"""

__version__ = "2.2.0"

from station_archive.types import (
    Domain,
    DownloadRequest,
    RawObservation,
    SchemaError,
    SourceRelease,
    SourceSnapshot,
    StationInventory,
)

__all__ = [
    "__version__",
    "Domain",
    "DownloadRequest",
    "RawObservation",
    "SchemaError",
    "SourceRelease",
    "SourceSnapshot",
    "StationInventory",
]
