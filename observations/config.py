"""
config.py
=========
Acquisition policy for the station archive.

Plan section 1.6: "Scientific policy belongs in versioned configuration, while
download, modelling, gridding and adjustment code remain separate."  Nothing
in this file is a scientific decision about which source wins -- source
priority is deliberately absent, because ranking sources is consolidation's
job, not the downloader's.

STORAGE
-------
The archive is large (a few GB of provider snapshots) and deliberately lives
outside the repository, so a checkout never carries licensed provider data.
Set STATION_ARCHIVE_ROOT to choose where:

    <archive root>/
        raw/<source>/<release>/...          immutable provider bytes (bronze)
        manifests/source_releases.jsonl     one line per source acquisition
        manifests/<source>/<release>/requests.jsonl
        inventory/<source>_inventory.csv
        inventory/coverage_report.md
        canonical/                          identity crosswalk and metadata
        consolidated/                       the gold table and its audits
"""
from __future__ import annotations

import os
from pathlib import Path

from observations.types import Domain

# ---------------------------------------------------------------------------
# Archive root
# ---------------------------------------------------------------------------
DEFAULT_ARCHIVE_ROOT = Path("E:/station_archive")

# STATION_ARCHIVE_ROOT is the name to use. DELOS_STATION_ARCHIVE is the
# original name from the project this package was extracted from, still
# honoured so an existing environment keeps working.
ARCHIVE_ROOT_ENV_VARS = ("STATION_ARCHIVE_ROOT", "DELOS_STATION_ARCHIVE")


def archive_root() -> Path:
    """
    Root directory for raw snapshots, manifests and inventories.

    Read at call time rather than import time so tests can redirect it with
    monkeypatch.setenv without reloading the module.
    """
    for name in ARCHIVE_ROOT_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path(DEFAULT_ARCHIVE_ROOT)


def raw_dir(source: str, release: str) -> Path:
    return archive_root() / "raw" / source / release


def manifest_dir(source: str, release: str) -> Path:
    return archive_root() / "manifests" / source / release


def inventory_dir() -> Path:
    return archive_root() / "inventory"


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
# Matches downscaling.config.STATION_BBOX, which is the box the existing
# LightGBM training set is filtered to.  The buffer admits stations outside
# that box whose records still describe systems crossing the target area.
DOMAIN = Domain(
    lat_min=33.0,
    lat_max=43.0,
    lon_min=11.0,
    lon_max=32.0,
    buffer_deg=2.0,
)

# ---------------------------------------------------------------------------
# Admission window
# ---------------------------------------------------------------------------
# Plan section 1.2: "The historical overlap period 1985-2005 is the minimum
# admission window for the present CERRA model.  Longer records should be
# retained for QC, homogeneity diagnosis and later QDM sensitivity, but must
# not leak future information into a historical calibration experiment."
#
# Completeness statistics in the inventory are computed over this window.
# Downloads are NOT truncated to it: GHCN-D and ECA&D publish whole-series
# files with no subsetting option, so the longer record costs nothing extra.
PERIOD_START = 1985
PERIOD_END = 2005

# Years fetched where the provider does subset by year.  Wider than the
# admission window so the retained-record requirement above is satisfied.
FETCH_YEAR_START = 1970
FETCH_YEAR_END = 2020

# ---------------------------------------------------------------------------
# Canonical variable names
# ---------------------------------------------------------------------------
# CMIP/CORDEX short names, used only as a common label in the inventory so the
# coverage report can compare sources.  Each adapter maps its provider's own
# element names onto these; the provider name is kept alongside in
# 'provider_element' so nothing is lost.
VARIABLES: tuple[str, ...] = (
    "tas",       # daily mean near-surface air temperature
    "tasmin",
    "tasmax",
    "pr",        # precipitation
    "hurs",      # relative humidity
    "sfcWind",   # daily mean wind speed
    "gust",      # wind gust
)

# ---------------------------------------------------------------------------
# Transfer behaviour
# ---------------------------------------------------------------------------
HTTP_TIMEOUT = 120           # seconds
HTTP_RETRIES = 4
HTTP_BACKOFF_BASE = 2.0      # seconds; delay is base * 2**(attempt-1)
HTTP_MIN_INTERVAL = 0.2      # seconds between requests to one host
USER_AGENT = "delos-station-archive/1.0 (research; contact via repository)"

# Free space required before a fetch run starts.  The plan's storage table is
# the reason this check exists at all: a half-written archive on a full drive
# is worse than a run that refuses to start.
MIN_FREE_BYTES = 2 * 1024**3


# ---------------------------------------------------------------------------
# Canonical station identity (plan section 1.3)
# ---------------------------------------------------------------------------
# Thresholds live here, not in identity.py, because they are scientific policy:
# "Scientific policy belongs in versioned configuration, while download,
# modelling, gridding and adjustment code remain separate."
#
# The plan fixes no numbers, so these are the project's declared choices.
# They are deliberately conservative: section 1.3 warns that "Coordinate
# tolerance alone is insufficient because airports and nearby climatological
# stations can be distinct", and a false merge leaks a station across
# validation folds, which is the specific harm identity control exists to
# prevent.

# Tier-2 "coincident coordinates": same site to within ~200 m.
IDENTITY_COINCIDENT_DEG = 0.002

# Tier-3 "near coordinates within a conservative tolerance": ~3 km.
#
# Widened from an initial ~1 km after the first crosswalk run showed the
# tighter gate rejecting pairs that are provably one record: ecad:002346 and
# ghcnd:BUM00015502 sit 1.26 km apart and are identical to the last decimal
# on all 3210 days they share (r = 1.000000, mean absolute difference 0.000
# degC).  Archives disagree about a station's "official" coordinates by more
# than a kilometre, so distance alone cannot separate a duplicate from a
# neighbour at this scale.
#
# This is a candidate-generation bound, not the decision: section 1.3 tier 3
# reads "near coordinates within a conservative tolerance PLUS overlapping
# daily-series correlation and metadata review", and the elevation,
# correlation and minimum-overlap gates still have to pass.
IDENTITY_NEAR_DEG = 0.027

# Compatible elevation. Two reports of one station rarely differ by more than
# a few tens of metres; a larger gap suggests genuinely different sites.
IDENTITY_ELEVATION_TOLERANCE_M = 50.0

# Overlapping daily-series correlation required before any non-identifier
# merge is accepted.
IDENTITY_CORRELATION_THRESHOLD = 0.995

# Minimum shared days before a correlation is trusted at all.  Without this a
# pair sharing five days could reach r = 0.99 by chance and be merged.  The
# plan does not fix a value; one year of overlap is the project's choice.
IDENTITY_MIN_OVERLAP_DAYS = 365

# Maximum mean absolute difference, per canonical unit, for a merge.
#
# Correlation measures covariation; this measures agreement, and the two come
# apart.  Measured on the archive: pairs 2.4 km apart correlate at 0.9915
# while differing 1.3 degC every day (a lapse-rate/exposure difference between
# genuinely distinct sites), whereas true duplicates cluster at a median
# difference of 0.000 degC.  Correlation alone therefore cannot separate them.
#
# Non-temperature thresholds are set by analogy rather than measurement: the
# match variables in practice are tasmax and tasmin, so pr and wind values
# here have not been calibrated against this archive.
IDENTITY_MAX_MEAN_ABS_DIFF = {
    "degC": 0.5,
    "mm": 0.5,
    "m s-1": 0.5,
    "percent": 3.0,
}

# Never merge two source stations that share no observation day at all.
#
# A project rule, stricter than anything plan section 1.3 requires: tier 2
# there would allow a merge on "coincident coordinates plus compatible name
# and elevation" with no series evidence whatsoever.  This forbids that
# outright.  Two records with zero overlapping days cannot be shown to be one
# instrument by any amount of metadata agreement, and a wrong merge puts one
# physical station on both sides of a validation fold.
#
# The rule is absolute and admits no manual override: a pair failing it is
# permanently unmergeable and is not offered for review.  Overlap is counted
# across every canonical variable, not just the matching variables, so a pair
# sharing only wind days still counts as having evidence.
IDENTITY_REQUIRE_ANY_OVERLAP = True

# Variables used to test series agreement, in order of preference.  tasmax is
# first because it is the best-populated variable common to ECA&D, GHCN-D and
# GSOD, and daily maxima are less sensitive than means to differing daily
# aggregation windows.
IDENTITY_MATCH_VARIABLES: tuple[str, ...] = ("tasmax", "tasmin", "pr")

# Sources admitted to the canonical crosswalk.  c3s_land is present in the
# archive but excluded from source selection: every one of its stations in
# this domain is an exact GHCN-D identifier, so it is a redelivery of GHCN-D
# rather than independent evidence.  It is still matched (tier 1) and carried
# in the audit as a value-level cross-check against GHCN-D.
IDENTITY_SOURCES: tuple[str, ...] = ("ecad", "ghcnd", "gsod", "c3s_land")

# Value agreement above which an ACCEPTED merge is sent for review, per
# canonical unit.  A pair can clear the correlation gate while still
# disagreeing in value: correlation measures covariation, not agreement, so
# two genuinely distinct stations a few hundred metres apart correlate above
# 0.98 while differing by more than a degree every day.  These merges stand,
# but they are marked so "metadata review" (section 1.3 tier 3) has a queue.
IDENTITY_AGREEMENT_REVIEW = {
    "degC": 1.0,
    "mm": 2.0,
    "m s-1": 1.0,
    "percent": 5.0,
}

# Agreement at or beyond which two records are treated as the SAME underlying
# observation for metadata-conflict reporting: near-zero mean absolute
# difference on a high fraction of shared days.  Used only for diagnostics --
# it never merges anything on its own.
IDENTITY_IDENTICAL_MAD = 0.05
IDENTITY_IDENTICAL_EXACT_FRACTION = 0.99

# Sources whose per-station daily series can be loaded from bronze.
#
# C3S is absent deliberately, and not as policy: it delivers whole-year files
# covering every station at once, so there is no per-station snapshot to read
# a series from.  Stations from a source outside this set can still join the
# crosswalk at tier 1 on an exact identifier -- which is how every C3S station
# joins -- but they are excluded from the tier-2/3 candidate tests, because
# those tests require a series.  Including them would produce rejections
# reading "insufficient overlapping days", which would falsely imply the
# series had been compared.
IDENTITY_SERIES_SOURCES: tuple[str, ...] = ("ecad", "ghcnd", "gsod")

# Sources whose values may be selected by the consolidator.  Kept separate
# from IDENTITY_SOURCES so an audit-only source cannot silently become a
# fallback value provider.
CONSOLIDATION_SOURCES: tuple[str, ...] = ("ecad", "ghcnd", "gsod")


# ---------------------------------------------------------------------------
# Source selection and gap filling (plan section 1.6)
# ---------------------------------------------------------------------------
# "Do not encode one universal claim that a portal is always 'best.'  Store a
# priority list per variable because coverage and semantics differ."
#
# The plan offers a starting order and calls it "a research hypothesis, not
# the final order.  Confirm it from the Step 1 coverage/QC pilot, variable
# definitions, daily aggregation semantics, provider flags and overlap
# behaviour."  The pilot has now run, and it contradicts the hypothesis in
# three places:
#
#   * c3s_land is ranked first for hurs and gust in the hypothesis but
#     carries neither -- and every one of its stations in this domain is an
#     exact GHCN-D identifier, so it is a redelivery rather than independent
#     evidence.  It is excluded from selection entirely (see
#     CONSOLIDATION_SOURCES) and kept as an audit cross-check.
#   * hurs has exactly one source in this domain: ECA&D, at 425 stations.
#   * gsod is ranked above ghcnd for pr in the hypothesis, but GSOD's
#     precipitation is the least trustworthy of the three: its own readme
#     states "'0' indicates no measurable precipitation (includes a trace)",
#     and 46% of its zero rows carry flag I, meaning the station reported no
#     precipitation data at all that day.  Coverage favours GSOD (203
#     stations against GHCN-D's 155); semantics do not.  Semantics win.
#
# Station counts behind the coverage half of each ordering:
#   tas      ecad 939  gsod 203  ghcnd   0     (GHCN-D TAVG is excluded; its
#                                               definition varies by network)
#   tasmin   ecad 1011 gsod 203  ghcnd 152
#   tasmax   ecad 1008 gsod 203  ghcnd 152
#   pr       ecad 1015 gsod 203  ghcnd 155
#   hurs     ecad 425
#   sfcWind  ecad 443  gsod 203  ghcnd   3
#   gust     gsod 203  ecad  65  ghcnd   3
SOURCE_PRIORITY: dict[str, tuple[str, ...]] = {
    "tas":     ("ecad", "gsod"),
    "tasmin":  ("ecad", "ghcnd", "gsod"),
    "tasmax":  ("ecad", "ghcnd", "gsod"),
    "pr":      ("ecad", "ghcnd", "gsod"),
    "hurs":    ("ecad",),
    "sfcWind": ("ecad", "gsod", "ghcnd"),
    "gust":    ("gsod", "ecad", "ghcnd"),
}

# Same-station fallback compatibility (plan section 1.6).
#
# "Before allowing a source switch, compare overlapping records by month for
# bias, dispersion, wet-day frequency and discontinuities.  A failed
# compatibility test leaves a gap or creates a time-bounded station segment;
# it does not justify blind splicing."
#
# Measured on confirmed duplicates in this archive, as median values. These
# come from duplicate_evidence.csv, whose comparisons are almost entirely
# tasmax -- the variable identity matching happened to use -- so each figure
# below is a TASMAX statistic and describes no other variable:
#   ecad vs ghcnd   bias  0.0000 degC   (byte-identical records)
#   ecad vs gsod    bias  0.0002 degC
#   ghcnd vs gsod   bias  0.3030 degC
#
# Do not read the third line as "GSOD runs warm". It is one median over 19
# tasmax pairs, it is not reproduced for other variables or other source
# pairs, and GHCN-D supplies no tas at all, so it says nothing about the
# ECA&D/GSOD tas fallbacks that make up most of this archive's switches --
# those run at a median -0.16 degC, the opposite sign. A correction built on
# that misreading was applied and withdrawn; see
# standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION.
#
# The default admits an offset of this size rather than blocking it; tighten
# FALLBACK_MAX_BIAS below 0.30 to refuse those splices instead.
FALLBACK_MIN_OVERLAP_DAYS = 180

FALLBACK_MAX_BIAS = {
    "degC": 0.5,
    "mm": 1.0,
    "m s-1": 0.5,
    "percent": 5.0,
}

# Ratio of fallback standard deviation to primary. A source recording the same
# quantity should not be materially smoother or noisier than the one it fills.
FALLBACK_DISPERSION_RANGE = (0.75, 1.33)

# Absolute difference in wet-day frequency permitted before a precipitation
# source switch is refused. GSOD's trace-as-zero convention moves this
# directly, which is why it is tested separately from bias.
FALLBACK_MAX_WET_DAY_DIFF = 0.10
WET_DAY_THRESHOLD_MM = 1.0

# A month whose bias departs this far from the overall bias is a
# discontinuity: the two records agree on average only by cancellation.
FALLBACK_MAX_MONTHLY_DEPARTURE = {
    "degC": 1.0,
    "mm": 2.0,
    "m s-1": 1.0,
    "percent": 10.0,
}
