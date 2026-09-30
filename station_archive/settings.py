"""
settings.py
===========
Machine settings: where the archive lives and how transfers behave.

This file holds nothing scientific. Every number here can be changed on a
different machine, or a different network, without changing a single value in
the archive the code produces. Scientific policy -- domain, admission window,
identity thresholds, source priority -- lives in policy.py, which is versioned
separately precisely so that a published archive can state which policy built
it (plan section 1.6).

STORAGE
-------
The archive is large (a few GB of provider snapshots) and deliberately lives
outside the repository, so a checkout never carries licensed provider data.
Set ARCHIVE_ROOT_ENV_VAR to choose where:

    <archive root>/
        raw/<source>/<release>/...          immutable provider bytes (bronze)
        manifests/source_releases.jsonl     one line per source acquisition
        manifests/<source>/<release>/requests.jsonl
        inventory/<source>_inventory.csv
        inventory/coverage_report.md
        canonical/                          identity crosswalk and metadata
        consolidated/                       the gold table and its audits
        review/                             human decisions (inputs)
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Archive root
# ---------------------------------------------------------------------------
# One variable, deliberately. A second name (DELOS_STATION_ARCHIVE, from the
# project this package was extracted from) was honoured here for continuity, and
# was removed: it was set on no machine, so the compatibility was with nothing,
# while the two-name precedence rule had to be documented, tested and guarded
# against in every other test that touches the environment.
ARCHIVE_ROOT_ENV_VAR = "STATION_ARCHIVE_ROOT"

def default_archive_root() -> Path:
    """
    Where the archive lives when no environment variable says otherwise.

    Home-relative on every platform, so a fresh clone resolves somewhere that
    exists and belongs to whoever is running it.

    This used to be a per-platform branch returning a hardcoded drive-letter
    path on Windows, and the hazard it created is worth recording: a bare
    drive-letter path is absolute on Windows and *relative* on POSIX, so the
    same default would create './<drive>:/station_archive' inside the checkout
    on Linux or macOS -- inside a repository whose ignore rules exist to keep
    licensed provider data out of git. A home-relative path removes the hazard
    instead of branching around it, and belongs to whoever is running the code
    rather than to whoever wrote it.

    Setting STATION_ARCHIVE_ROOT explicitly is the supported route on every
    platform; this default is a convenience, not a recommendation.
    """
    return Path.home() / "station_archive"


# Retained for callers that want to report the default without resolving it.
DEFAULT_ARCHIVE_ROOT = default_archive_root()


def archive_root() -> Path:
    """
    Root directory for raw snapshots, manifests and inventories.

    Read at call time rather than import time so tests can redirect it with
    monkeypatch.setenv without reloading the module.
    """
    # An exported-but-empty variable must not resolve the archive to '', so the
    # test is truthiness rather than presence.
    value = os.environ.get(ARCHIVE_ROOT_ENV_VAR)
    return Path(value) if value else default_archive_root()


def raw_dir(source: str, release: str) -> Path:
    return archive_root() / "raw" / source / release


def manifest_dir(source: str, release: str) -> Path:
    return archive_root() / "manifests" / source / release


def inventory_dir() -> Path:
    return archive_root() / "inventory"


def canonical_dir() -> Path:
    return archive_root() / "canonical"


def consolidated_dir() -> Path:
    return archive_root() / "consolidated"


def review_dir() -> Path:
    return archive_root() / "review"


def _version() -> str:
    """
    The package version, for the User-Agent.

    Imported inside the function because station_archive/__init__.py is what
    defines it, and importing the package at module scope from one of its own
    modules is how an import cycle starts.
    """
    from station_archive import __version__

    return __version__


# ---------------------------------------------------------------------------
# Transfer behaviour
# ---------------------------------------------------------------------------
HTTP_TIMEOUT = 120           # seconds
HTTP_RETRIES = 4
HTTP_BACKOFF_BASE = 2.0      # seconds; delay is base * 2**(attempt-1)
HTTP_MIN_INTERVAL = 0.2      # seconds between requests to one host
# Derived, not restated: this said 1.0 while the package was at 2.0.0, so a
# provider log could not tell which build made a request.
USER_AGENT = f"station-archive/{_version()} (research; contact via repository)"

# Free space required before a fetch run starts. A half-written archive on a
# full drive is worse than a run that refuses to start.
MIN_FREE_BYTES = 2 * 1024**3
