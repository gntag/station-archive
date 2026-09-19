"""
identity.py
===========
Canonical station identity and duplicate control (plan section 1.3).

THE PROBLEM
-----------
"Simple concatenation would inflate the apparent station count and leak
duplicates across validation folds."  Four sources describe an overlapping set
of physical stations under different identifiers; the same airport may appear
as an ECA&D STAID, a GHCN-D ID, a GSOD USAF-WBAN pair and a C3S primary ID.
Treating those as four stations would both overstate the network and let one
physical site appear in training and validation at once.

THE MATCHING LADDER
-------------------
Section 1.3, in descending confidence:

  1. exact WIGOS/WMO or documented cross-source identifier;
  2. coincident coordinates plus compatible name and elevation;
  3. near coordinates within a conservative tolerance plus overlapping
     daily-series correlation and metadata review;
  4. unresolved candidates remain separate and receive a duplicate-risk flag.

WHAT THE ARCHIVE SUPPORTS
-------------------------
Measured from the downloaded inventories rather than assumed:

  * No source publishes a WIGOS identifier.
  * ECA&D publishes no WMO number at all, for any of its 1463 domain
    stations -- so tier 1 is unavailable to the largest source and every
    ECA&D link must be earned at tier 2 or 3.
  * C3S primary_station_id is literally a GHCN-D identifier
    (e.g. 'EGM00062318'), and every C3S station in this domain matches a
    GHCN-D one exactly.  That is a documented cross-source identifier, so C3S
    joins at tier 1.
  * GHCN-D and GSOD share WMO numbers for 40 stations.

PROJECT POLICY (recorded in config.py, not here)
------------------------------------------------
A non-identifier merge requires ALL of: coordinates within tolerance,
elevation within tolerance, and overlapping daily-series correlation at or
above threshold over a minimum number of shared days.  Section 1.3 warns that
"Coordinate tolerance alone is insufficient because airports and nearby
climatological stations can be distinct", and a false merge is the more
damaging error: it leaks a station across folds, which is the exact harm this
module exists to prevent.

A CAVEAT RECORDED IN THE EVIDENCE
---------------------------------
At ~1 km separation, daily maximum temperature at two genuinely distinct
stations still correlates above 0.98, so correlation alone is a weak
duplicate discriminator.  Near-identity of values is the stronger signal.
This module therefore also records mean absolute difference, bias and an
exact-agreement fraction for every candidate pair -- statistics section 1.6
requires in the consolidation audit regardless -- and flags pairs that pass
the correlation gate while disagreeing in value, so "metadata review" has
something to review.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from observations import config, qc, standardize
from observations.types import StationInventory

# Match tiers, in the order section 1.3 lists them.
TIER_IDENTIFIER = 1
TIER_COINCIDENT = 2
TIER_NEAR = 3
TIER_UNRESOLVED = 4

# Reason string for the never-merge rule.  Matched by review.py to exclude
# these pairs from the decision workbook, so it must stay in one place.
NEVER_MERGE_REASON = (
    "no overlapping days in any variable; never merge (project rule)"
)

TIER_NAMES = {
    TIER_IDENTIFIER: "exact_identifier",
    TIER_COINCIDENT: "coincident_coordinates_plus_series",
    TIER_NEAR: "near_coordinates_plus_series",
    TIER_UNRESOLVED: "unresolved",
}


@dataclass(frozen=True)
class StationRecord:
    """One source station, as described by its source's inventory."""

    source: str
    source_station_id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    wmo_id: str
    wigos_id: str
    variables: frozenset[str]
    first_date: str
    last_date: str

    @property
    def key(self) -> str:
        """Stable cross-source key, also the basis of the canonical ID."""
        return f"{self.source}:{self.source_station_id}"


@dataclass(frozen=True)
class OverlapStats:
    """
    Agreement between two source stations on a shared daily series.

    'exact_fraction' counts days whose canonical-unit values agree to within
    half a provider quantum; it is the statistic that separates a genuine
    redelivery of one record from two nearby but distinct stations.
    """

    variable: str
    n_overlap: int
    correlation: float
    bias: float
    mean_abs_diff: float
    exact_fraction: float

    @property
    def usable(self) -> bool:
        return self.n_overlap >= config.IDENTITY_MIN_OVERLAP_DAYS and not math.isnan(
            self.correlation
        )


@dataclass(frozen=True)
class MatchPair:
    """A proposed or rejected link between two source stations."""

    left: str                      # StationRecord.key
    right: str
    tier: int
    accepted: bool
    distance_km: float
    elevation_diff_m: float
    name_similarity: float
    stats: Optional[OverlapStats]
    reason: str

    def to_dict(self) -> dict:
        row = {
            "left": self.left,
            "right": self.right,
            "tier": self.tier,
            "tier_name": TIER_NAMES.get(self.tier, str(self.tier)),
            "accepted": self.accepted,
            "distance_km": round(self.distance_km, 4),
            "elevation_diff_m": self.elevation_diff_m,
            "name_similarity": round(self.name_similarity, 3),
            "reason": self.reason,
        }
        if self.stats is not None:
            row.update(
                {
                    "match_variable": self.stats.variable,
                    "n_overlap_days": self.stats.n_overlap,
                    "correlation": round(self.stats.correlation, 5),
                    "bias": round(self.stats.bias, 4),
                    "mean_abs_diff": round(self.stats.mean_abs_diff, 4),
                    "exact_fraction": round(self.stats.exact_fraction, 4),
                }
            )
        else:
            row.update(
                {
                    "match_variable": "",
                    "n_overlap_days": 0,
                    "correlation": float("nan"),
                    "bias": float("nan"),
                    "mean_abs_diff": float("nan"),
                    "exact_fraction": float("nan"),
                }
            )
        return row


# ---------------------------------------------------------------------------
# Geometry and text helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in kilometres.

    Used for reporting and for the distance gate.  A plain degree box would
    be anisotropic: at 38 degrees north one degree of longitude is about
    79 km against 111 km for latitude, so a degree tolerance alone would
    admit pairs nearly half again as far apart east-west as north-south.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def normalize_name(name: str) -> str:
    """
    Reduce a station name to comparable tokens.

    Sources spell the same site differently ('ATHENS', 'ATHENS HELLINIKON',
    'PIACENZA, IT'), so punctuation, country suffixes and case are stripped
    before comparison.  Name similarity is supporting evidence only -- it
    never decides a merge on its own.
    """
    text = str(name or "").upper()
    for separator in (",", "/", "-", "(", ")", "."):
        text = text.replace(separator, " ")
    return " ".join(text.split())


def name_similarity(left: str, right: str) -> float:
    """
    Jaccard overlap of name tokens, 0.0 when either name is empty.

    Token overlap rather than edit distance because the common real
    difference is an added qualifier ('AIRPORT', 'INTL'), not a typo.
    """
    a = set(normalize_name(left).split())
    b = set(normalize_name(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Tier 1 -- exact identifiers
# ---------------------------------------------------------------------------

def normalized_wmo(value: str) -> str:
    """
    WMO number without leading zeros, or '' when absent.

    GSOD encodes it as the first five digits of the USAF-WBAN filename while
    GHCN-D publishes it as a separate field, so zero-padding differs between
    the two and must be removed before comparison.
    """
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    stripped = text.lstrip("0")
    return stripped if stripped else "0"


def ghcnd_style_id(record: StationRecord) -> str:
    """
    The GHCN-D identifier a record carries, if any.

    C3S delivers GHCN-D identifiers verbatim in primary_station_id, which is
    the "documented cross-source identifier" of section 1.3 tier 1.  Verified
    against the archive: all 118 C3S domain stations match a GHCN-D ID.
    """
    if record.source in ("ghcnd", "c3s_land"):
        return record.source_station_id.strip()
    return ""


def match_tier1_identifier(records: Sequence[StationRecord]) -> list[MatchPair]:
    """
    Link stations that share an exact identifier.

    Two independent identifier spaces are used: the GHCN-D identifier (shared
    by GHCN-D and C3S) and the WMO number (published by GHCN-D and derivable
    from GSOD filenames).  ECA&D participates in neither.
    """
    pairs: list[MatchPair] = []

    for scheme, extract in (
        ("ghcnd_id", ghcnd_style_id),
        ("wmo", lambda r: normalized_wmo(r.wmo_id)),
        ("wigos", lambda r: str(r.wigos_id or "").strip()),
    ):
        groups: dict[str, list[StationRecord]] = {}
        for record in records:
            value = extract(record)
            if value and value.lower() != "nan":
                groups.setdefault(value, []).append(record)

        for value, members in groups.items():
            # Only cross-source links matter: two stations from one source
            # sharing an identifier is that source's own business.
            by_source = {}
            for member in members:
                by_source.setdefault(member.source, []).append(member)
            if len(by_source) < 2:
                continue

            ordered = sorted(members, key=lambda r: r.key)
            anchor = ordered[0]
            for other in ordered[1:]:
                if other.source == anchor.source:
                    continue
                pairs.append(
                    MatchPair(
                        left=anchor.key,
                        right=other.key,
                        tier=TIER_IDENTIFIER,
                        accepted=True,
                        distance_km=haversine_km(
                            anchor.lat, anchor.lon, other.lat, other.lon
                        ),
                        elevation_diff_m=_elevation_gap(anchor, other),
                        name_similarity=name_similarity(anchor.name, other.name),
                        stats=None,
                        reason=f"exact {scheme} identifier {value}",
                    )
                )
    return pairs


def _elevation_gap(a: StationRecord, b: StationRecord) -> float:
    if _missing(a.elevation_m) or _missing(b.elevation_m):
        return float("nan")
    return abs(float(a.elevation_m) - float(b.elevation_m))


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


# ---------------------------------------------------------------------------
# Candidate generation and series agreement
# ---------------------------------------------------------------------------

def candidate_pairs(
    records: Sequence[StationRecord],
    tolerance_deg: float,
) -> list[tuple[StationRecord, StationRecord]]:
    """
    Cross-source station pairs close enough to be worth testing.

    Blocked on a coordinate grid so the comparison is not quadratic over the
    whole network: each station is bucketed by rounded coordinates and only
    compared against its own and neighbouring buckets.  The bucket is the
    tolerance, so no pair within tolerance can fall in non-adjacent buckets.
    """
    buckets: dict[tuple[int, int], list[StationRecord]] = {}
    for record in records:
        if _missing(record.lat) or _missing(record.lon):
            continue
        key = (
            int(math.floor(record.lat / tolerance_deg)),
            int(math.floor(record.lon / tolerance_deg)),
        )
        buckets.setdefault(key, []).append(record)

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[StationRecord, StationRecord]] = []

    for (bi, bj), members in buckets.items():
        neighbourhood: list[StationRecord] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                neighbourhood.extend(buckets.get((bi + di, bj + dj), ()))

        for left in members:
            for right in neighbourhood:
                if left.source == right.source:
                    continue
                ordered = tuple(sorted((left.key, right.key)))
                if ordered in seen:
                    continue
                seen.add(ordered)
                a, b = (left, right) if left.key <= right.key else (right, left)
                pairs.append((a, b))

    return sorted(pairs, key=lambda p: (p[0].key, p[1].key))


# Half a provider quantum, per canonical unit, for the exact-agreement test.
# GHCN-D and ECA&D report temperature in tenths of a degree, so two records of
# one observation agree exactly when they differ by less than 0.05 degC.
_EXACT_TOLERANCE = {
    "degC": 0.05,
    "mm": 0.05,
    "m s-1": 0.05,
    "percent": 0.5,
}


def overlap_statistics(
    left: pd.Series,
    right: pd.Series,
    variable: str,
    left_units: str,
    right_units: str,
) -> OverlapStats:
    """
    Agreement statistics between two daily series on their shared days.

    Both series are converted to canonical units first.  Correlation would
    survive without conversion -- Pearson r is invariant under affine
    transforms -- but bias, mean absolute difference and exact agreement
    would be meaningless when one source reports Fahrenheit and the other
    tenths of a degree Celsius.
    """
    if left.empty or right.empty:
        return OverlapStats(variable, 0, float("nan"), float("nan"), float("nan"), 0.0)

    shared = left.index.intersection(right.index)
    if len(shared) == 0:
        return OverlapStats(variable, 0, float("nan"), float("nan"), float("nan"), 0.0)

    a = np.array([standardize.to_canonical(v, left_units) for v in left.loc[shared]])
    b = np.array([standardize.to_canonical(v, right_units) for v in right.loc[shared]])

    # Drop physically impossible values before measuring agreement.
    #
    # GHCN-D carries 999.0 degC at some stations -- an undocumented sentinel
    # the -9999 filter misses.  Measured on ecad:002082 / ghcnd:IT000016134,
    # two records of MONTE CIMONE that agree exactly on the days they share:
    # leaving the sentinels in reported a correlation of 0.14 and a mean
    # difference of 2.10 degC, so an identical pair was rejected for
    # "correlation below 0.995" on a statistic that was pure artefact.
    plausible = (
        qc.check_gross_range(pd.Series(a), variable).to_numpy()
        & qc.check_gross_range(pd.Series(b), variable).to_numpy()
    )
    a, b = a[plausible], b[plausible]

    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = int(a.size)
    if n == 0:
        return OverlapStats(variable, 0, float("nan"), float("nan"), float("nan"), 0.0)

    # A constant series has zero variance and no defined correlation; report
    # NaN rather than letting numpy warn and return a meaningless value.
    if n < 2 or np.std(a) == 0 or np.std(b) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])

    difference = a - b
    tolerance = _EXACT_TOLERANCE.get(standardize.canonical_unit_for(variable) or "", 0.05)

    return OverlapStats(
        variable=variable,
        n_overlap=n,
        correlation=correlation,
        bias=float(np.mean(difference)),
        mean_abs_diff=float(np.mean(np.abs(difference))),
        exact_fraction=float(np.mean(np.abs(difference) < tolerance)),
    )


# ---------------------------------------------------------------------------
# Union-find over accepted links
# ---------------------------------------------------------------------------

class _DisjointSet:
    """
    Groups source stations into canonical stations.

    Merging is transitive: if A links to B and B to C, all three become one
    canonical station even when A and C were never compared directly.  That
    is the correct reading of "one canonical_station_id and all source
    aliases", but it can absorb a station on a chain of pairwise links, so
    build_crosswalk records which groups were formed transitively for review.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:      # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Attach the lexicographically larger root to the smaller so the
        # representative is deterministic across runs.
        if ra <= rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return {root: sorted(members) for root, members in out.items()}


# ---------------------------------------------------------------------------
# Crosswalk
# ---------------------------------------------------------------------------

def agreement_needs_review(stats: Optional[OverlapStats]) -> bool:
    """
    Whether an accepted merge disagrees enough in value to warrant review.

    Correlation measures covariation, not agreement.  Two distinct stations a
    few hundred metres apart correlate above 0.98 while differing by more
    than a degree every day, so a merge can pass the gate and still be wrong.
    """
    if stats is None or math.isnan(stats.mean_abs_diff):
        return False
    unit = standardize.canonical_unit_for(stats.variable) or ""
    threshold = config.IDENTITY_AGREEMENT_REVIEW.get(unit)
    if threshold is None:
        return False
    return stats.mean_abs_diff > threshold


def records_are_identical(stats: Optional[OverlapStats]) -> bool:
    """
    Whether two series are the same underlying observation, not merely alike.

    Diagnostic only: this never merges anything.  It is what distinguishes an
    archive metadata error from two genuinely different nearby stations.
    """
    if stats is None or math.isnan(stats.mean_abs_diff):
        return False
    return (
        stats.mean_abs_diff <= config.IDENTITY_IDENTICAL_MAD
        and stats.exact_fraction >= config.IDENTITY_IDENTICAL_EXACT_FRACTION
    )


def metadata_conflicts(evidence: list[MatchPair]) -> pd.DataFrame:
    """
    Pairs whose values prove one record but whose metadata disagrees.

    Section 1.3 makes "metadata review" part of tier 3, and section 1.4 asks
    for QC diagnostics rather than silent correction.  This table is that
    review queue: each row is a station where two archives publish
    irreconcilable coordinates or elevation for what the observations show to
    be a single instrument.

    It matters beyond identity.  Elevation is a model predictor
    (delta_elev_m in the existing pipeline), so a station carrying a wrong
    elevation corrupts the fitted relationship however its identity is
    resolved.
    """
    rows = []
    for pair in evidence:
        # The same minimum overlap the merge gate uses: a single shared day
        # that happens to agree is not evidence that two archives describe
        # one instrument, and would put noise in the review queue.
        if pair.stats is None or not pair.stats.usable:
            continue
        if not records_are_identical(pair.stats):
            continue
        elevation_conflict = (
            not math.isnan(pair.elevation_diff_m)
            and pair.elevation_diff_m > config.IDENTITY_ELEVATION_TOLERANCE_M
        )
        distance_conflict = pair.distance_km > _degrees_to_km(
            config.IDENTITY_COINCIDENT_DEG
        )
        if not (elevation_conflict or distance_conflict):
            continue
        rows.append(
            {
                "left": pair.left,
                "right": pair.right,
                "merged": pair.accepted,
                "distance_km": round(pair.distance_km, 4),
                "elevation_diff_m": pair.elevation_diff_m,
                "match_variable": pair.stats.variable,
                "n_overlap_days": pair.stats.n_overlap,
                "correlation": round(pair.stats.correlation, 6),
                "mean_abs_diff": round(pair.stats.mean_abs_diff, 4),
                "exact_fraction": round(pair.stats.exact_fraction, 4),
                "conflict": ", ".join(
                    filter(
                        None,
                        [
                            "elevation" if elevation_conflict else "",
                            "coordinates" if distance_conflict else "",
                        ],
                    )
                ),
                "note": "series identical; one archive's metadata is wrong",
            }
        )
    return pd.DataFrame(rows)


def resolve_canonical_metadata(
    aliases: pd.DataFrame,
    series_loader,
    variables: Sequence[str] = config.VARIABLES,
) -> pd.DataFrame:
    """
    One authoritative coordinate, elevation and name per canonical station.

    The alias contributing the most observations wins every field, as a
    block.  Record length is a proxy for how well a station is characterised
    by its archive, and taking all fields from one alias keeps the result a
    position some archive actually published -- picking each field from a
    different winner could manufacture a coordinate/elevation pair that no
    source ever asserted.

    This is selection, not averaging: section 1.3 forbids "silent coordinate
    averaging", and station_metadata_history keeps every alias's own
    coordinates as its own time-bounded segment regardless of which one wins
    here.

    Only groups with more than one alias are counted, so the observation
    tally costs a read of the merged stations rather than the whole archive.
    """
    rows = []
    for canonical_id, group in aliases.groupby("canonical_station_id", sort=True):
        if len(group) == 1:
            winner = group.iloc[0]
            n_obs = pd.NA
            contest = "single source"
        else:
            counts = []
            for row in group.itertuples(index=False):
                declared = str(row.variables or "").split("|")
                total = 0
                for variable in variables:
                    if variable in declared:
                        total += len(
                            series_loader.series(
                                row.source, str(row.source_station_id), variable
                            )
                        )
                counts.append(total)
            group = group.assign(alias_n_obs=counts)
            # Ties break on the alias key so a re-run cannot reorder them.
            ordered = group.sort_values(
                ["alias_n_obs", "source", "source_station_id"],
                ascending=[False, True, True],
            )
            winner = ordered.iloc[0]
            n_obs = int(winner["alias_n_obs"])
            contest = " > ".join(
                f"{r.source}:{r.source_station_id}({int(r.alias_n_obs)})"
                for r in ordered.itertuples(index=False)
            )

        rows.append(
            {
                "canonical_station_id": canonical_id,
                "n_aliases": len(group),
                "metadata_source": f"{winner['source']}:{winner['source_station_id']}",
                "metadata_n_observations": n_obs,
                "name": winner["name"],
                "lat": winner["lat"],
                "lon": winner["lon"],
                "elevation_m": winner["elevation_m"],
                "wmo_id": winner["wmo_id"],
                "sources": "|".join(sorted(set(group["source"]))),
                "variables": "|".join(
                    sorted(
                        {
                            v
                            for entry in group["variables"].dropna()
                            for v in str(entry).split("|")
                            if v
                        }
                    )
                ),
                "duplicate_risk": bool(group["duplicate_risk"].any()),
                "selection": contest,
            }
        )

    return pd.DataFrame(rows).sort_values("canonical_station_id").reset_index(drop=True)


@dataclass
class Crosswalk:
    """
    The durable tables of plan section 1.7 that this module owns.

    station_aliases              canonical ID, source ID, matching evidence
    station_metadata_history     coordinates/elevation/name with effective dates
    evidence                     every pair tested, accepted or not
    """

    aliases: pd.DataFrame
    metadata_history: pd.DataFrame
    evidence: pd.DataFrame
    conflicts: pd.DataFrame = field(default_factory=pd.DataFrame)

    def n_canonical(self) -> int:
        return int(self.aliases["canonical_station_id"].nunique())

    def n_aliases(self) -> int:
        return len(self.aliases)


def records_from_inventories(
    inventories: dict[str, StationInventory],
) -> list[StationRecord]:
    """
    Collapse per-source inventories (one row per station-variable) into one
    record per source station.

    Coordinates and elevation are taken from the first row rather than
    averaged: a single source reports one position per station, and averaging
    would be the "silent coordinate averaging" section 1.3 forbids.
    """
    records: list[StationRecord] = []
    for source, inventory in inventories.items():
        frame = inventory.frame
        if frame.empty:
            continue
        for station_id, group in frame.groupby("source_station_id", sort=True):
            head = group.iloc[0]
            records.append(
                StationRecord(
                    source=source,
                    source_station_id=str(station_id),
                    name=str(head.get("name", "") or ""),
                    lat=float(pd.to_numeric(head.get("lat"), errors="coerce")),
                    lon=float(pd.to_numeric(head.get("lon"), errors="coerce")),
                    elevation_m=float(
                        pd.to_numeric(head.get("elevation_m"), errors="coerce")
                    ),
                    wmo_id=str(head.get("wmo_id", "") or ""),
                    wigos_id=str(head.get("wigos_id", "") or ""),
                    variables=frozenset(group["variable"].dropna().astype(str)),
                    first_date=str(head.get("first_date", "") or ""),
                    last_date=str(head.get("last_date", "") or ""),
                )
            )
    return sorted(records, key=lambda r: r.key)


def total_overlap_days(a: StationRecord, b: StationRecord, series_loader) -> int:
    """
    Days the two records share, summed over every canonical variable.

    Counted across all variables rather than just the matching ones: a pair
    whose temperature records never coincide but whose wind records do has
    evidence, and the never-merge rule must not treat it as evidence-free.
    No unit conversion is needed to count shared dates.
    """
    total = 0
    for variable in config.VARIABLES:
        left = series_loader.series(a.source, a.source_station_id, variable)
        right = series_loader.series(b.source, b.source_station_id, variable)
        if not left.empty and not right.empty:
            total += len(left.index.intersection(right.index))
    return total


def _best_overlap(
    a: StationRecord,
    b: StationRecord,
    series_loader,
) -> Optional[OverlapStats]:
    """
    Agreement on the first shared match variable with usable overlap.

    Variables are tried in the configured order rather than pooled, because
    mixing variables would make the correlation uninterpretable.
    """
    shared_variables = [
        v
        for v in config.IDENTITY_MATCH_VARIABLES
        if v in a.variables and v in b.variables
    ]
    best: Optional[OverlapStats] = None
    for variable in shared_variables:
        left = series_loader.series(a.source, a.source_station_id, variable)
        right = series_loader.series(b.source, b.source_station_id, variable)
        units_left = _provider_units(a.source, variable)
        units_right = _provider_units(b.source, variable)
        if not units_left or not units_right:
            continue
        stats = overlap_statistics(left, right, variable, units_left, units_right)
        if stats.usable:
            return stats
        if best is None or stats.n_overlap > best.n_overlap:
            best = stats
    return best


def _provider_units(source: str, variable: str) -> str:
    """Provider unit string a source uses for a canonical variable."""
    if source == "ghcnd":
        from observations.sources.ghcnd import ELEMENT_TO_VARIABLE, ELEMENT_UNITS

        for element, mapped in ELEMENT_TO_VARIABLE.items():
            if mapped == variable:
                return ELEMENT_UNITS.get(element, "")
    if source == "ecad":
        from observations.sources.ecad import ELEMENT_TO_VARIABLE, ELEMENT_UNITS

        for element, mapped in ELEMENT_TO_VARIABLE.items():
            if mapped == variable:
                return ELEMENT_UNITS.get(element, "")
    if source == "gsod":
        from observations.sources.gsod import COLUMN_TO_VARIABLE, COLUMN_UNITS

        for column, mapped in COLUMN_TO_VARIABLE.items():
            if mapped == variable:
                return COLUMN_UNITS.get(column, "")
    if source == "c3s_land":
        # C3S declares units per row rather than per variable; identity
        # matching does not use C3S series because it joins at tier 1.
        return ""
    return ""


def evaluate_pair(
    a: StationRecord,
    b: StationRecord,
    series_loader,
) -> MatchPair:
    """
    Test one candidate pair against the project's merge policy.

    All three gates must pass: distance, elevation and series correlation.
    The rejection reason names the first gate that failed, so the evidence
    table explains every non-merge rather than merely recording it.
    """
    distance = haversine_km(a.lat, a.lon, b.lat, b.lon)
    elevation_gap = _elevation_gap(a, b)
    similarity = name_similarity(a.name, b.name)

    near_km = _degrees_to_km(config.IDENTITY_NEAR_DEG)
    coincident_km = _degrees_to_km(config.IDENTITY_COINCIDENT_DEG)
    tier = TIER_COINCIDENT if distance <= coincident_km else TIER_NEAR

    if distance > near_km:
        return MatchPair(
            a.key, b.key, TIER_UNRESOLVED, False, distance, elevation_gap,
            similarity, None,
            f"distance {distance:.2f} km exceeds {near_km:.2f} km tolerance",
        )

    # Series statistics are computed before the metadata gates rather than
    # after, so a pair rejected on elevation still carries the evidence that
    # would let a reviewer overturn the rejection.  Section 1.6 requires
    # "pairwise overlap statistics" in the audit, and an elevation conflict
    # between two archives describing one station is exactly the case that
    # needs them: without the statistics the audit records a disagreement and
    # hides the proof of which side is wrong.
    # The never-merge rule, applied before any other evidence is weighed.
    # Two records sharing no observation day cannot be shown to be one
    # instrument, however well their coordinates and names agree.
    if config.IDENTITY_REQUIRE_ANY_OVERLAP:
        if total_overlap_days(a, b, series_loader) == 0:
            return MatchPair(
                a.key, b.key, TIER_UNRESOLVED, False, distance, elevation_gap,
                similarity, None, NEVER_MERGE_REASON,
            )

    stats = _best_overlap(a, b, series_loader)

    # Proven-identical observations outrank the elevation metadata.
    #
    # Two archives were found publishing one instrument under irreconcilable
    # elevations -- ecad:010963 and ghcnd:HRE00155411 sit at identical
    # coordinates and agree to the last decimal on all 7670 days they share,
    # yet differ by 113 m in stated elevation.  Rejecting that pair would
    # leave one physical station in two folds on the strength of a metadata
    # error the observations themselves disprove.
    #
    # The override is deliberately narrow: near-zero mean absolute difference
    # on over 99% of shared days, over at least the minimum overlap.  Merely
    # correlated stations never qualify, so genuinely distinct sites at
    # different altitudes are still separated by the gate.  The conflict is
    # always recorded in metadata_conflicts.csv -- the elevation is not
    # corrected here, because section 1.4 asks to "diagnose ... but do not
    # silently homogenize".
    identical = stats is not None and stats.usable and records_are_identical(stats)

    if math.isnan(elevation_gap) and not identical:
        return MatchPair(
            a.key, b.key, tier, False, distance, elevation_gap, similarity, stats,
            "elevation missing for one station; cannot confirm compatibility",
        )

    if (
        not math.isnan(elevation_gap)
        and elevation_gap > config.IDENTITY_ELEVATION_TOLERANCE_M
        and not identical
    ):
        return MatchPair(
            a.key, b.key, tier, False, distance, elevation_gap, similarity, stats,
            f"elevation differs by {elevation_gap:.0f} m, above "
            f"{config.IDENTITY_ELEVATION_TOLERANCE_M:.0f} m tolerance",
        )

    if stats is None or not stats.usable:
        overlap = 0 if stats is None else stats.n_overlap
        return MatchPair(
            a.key, b.key, tier, False, distance, elevation_gap, similarity, stats,
            f"only {overlap} usable overlapping days, below the "
            f"{config.IDENTITY_MIN_OVERLAP_DAYS}-day minimum",
        )

    if stats.correlation < config.IDENTITY_CORRELATION_THRESHOLD:
        return MatchPair(
            a.key, b.key, tier, False, distance, elevation_gap, similarity, stats,
            f"{stats.variable} correlation {stats.correlation:.3f} below "
            f"{config.IDENTITY_CORRELATION_THRESHOLD}",
        )

    # Agreement, not merely covariation.  A pair can clear the correlation
    # gate while differing by more than a degree every day, which is a real
    # difference between sites rather than two records of one instrument.
    agreement_limit = config.IDENTITY_MAX_MEAN_ABS_DIFF.get(
        standardize.canonical_unit_for(stats.variable) or ""
    )
    if agreement_limit is not None and stats.mean_abs_diff > agreement_limit:
        return MatchPair(
            a.key, b.key, tier, False, distance, elevation_gap, similarity, stats,
            f"{stats.variable} differs by {stats.mean_abs_diff:.2f} on average, "
            f"above the {agreement_limit} limit",
        )

    elevation_overridden = identical and (
        math.isnan(elevation_gap)
        or elevation_gap > config.IDENTITY_ELEVATION_TOLERANCE_M
    )
    if elevation_overridden:
        detail = (
            f"elevation conflict of {elevation_gap:.0f} m overridden by identical series"
            if not math.isnan(elevation_gap)
            else "missing elevation overridden by identical series"
        )
    else:
        detail = f"elevation within {elevation_gap:.0f} m"

    return MatchPair(
        a.key, b.key, tier, True, distance, elevation_gap, similarity, stats,
        f"{distance:.2f} km apart, {detail}, "
        f"{stats.variable} r={stats.correlation:.3f} over {stats.n_overlap} days",
    )


def _degrees_to_km(degrees: float) -> float:
    """
    Degree tolerance expressed as a distance at the domain's mid-latitude.

    Converting once here keeps the configured policy in the familiar degree
    units while the actual gate stays isotropic.
    """
    mid_lat = (config.DOMAIN.lat_min + config.DOMAIN.lat_max) / 2.0
    return haversine_km(mid_lat, 0.0, mid_lat + degrees, 0.0)


def load_decisions(path) -> dict[tuple[str, str], str]:
    """
    Reviewer merge decisions, keyed by the ordered pair of alias keys.

    Read from the review workbook's Review sheet.  Only 'yes' and 'no' are
    recognised; anything else is treated as undecided, so a stray note or a
    half-typed value cannot be mistaken for an instruction.
    """
    frame = pd.read_excel(path, sheet_name="Review")
    decisions: dict[tuple[str, str], str] = {}
    for row in frame.itertuples(index=False):
        value = str(getattr(row, "DECISION_merge", "") or "").strip().lower()
        if value in ("yes", "no"):
            decisions[(row.left, row.right)] = value
    return decisions


def build_crosswalk(
    inventories: dict[str, StationInventory],
    series_loader,
    progress: bool = False,
    decisions: Optional[dict[tuple[str, str], str]] = None,
) -> Crosswalk:
    """
    Build the canonical crosswalk from per-source inventories.

    Order matters: tier-1 identifier links are applied first and never
    re-tested, because an exact cross-source identifier is stronger evidence
    than any coordinate or correlation test could be.  Remaining candidates
    are then evaluated against the full gate.
    """
    records = records_from_inventories(inventories)
    by_key = {r.key: r for r in records}

    groups = _DisjointSet()
    for record in records:
        groups.add(record.key)

    evidence: list[MatchPair] = []

    tier1 = match_tier1_identifier(records)
    for pair in tier1:
        evidence.append(pair)
        groups.union(pair.left, pair.right)

    already_linked = {tuple(sorted((p.left, p.right))) for p in tier1}

    # Tier 2 and 3 both need a daily series, so a source whose series cannot
    # be read per station is excluded from candidate generation rather than
    # tested and rejected for "insufficient overlap" -- a reason that would
    # falsely imply the series had been compared.  Such sources still hold
    # whatever tier-1 identity they earned above.
    series_records = [
        r for r in records if r.source in config.IDENTITY_SERIES_SOURCES
    ]
    decisions = decisions or {}

    candidates = candidate_pairs(series_records, config.IDENTITY_NEAR_DEG)
    for index, (a, b) in enumerate(candidates, start=1):
        if tuple(sorted((a.key, b.key))) in already_linked:
            continue
        pair = evaluate_pair(a, b, series_loader)

        # A reviewer's decision overrides the automated gates -- that is what
        # the review exists for.  It cannot override the never-merge rule,
        # which the project declared absolute: a pair sharing no observation
        # day stays separate however it was adjudicated.
        decision = decisions.get((pair.left, pair.right))
        if decision == "yes" and not pair.accepted:
            if pair.reason == NEVER_MERGE_REASON:
                pair = replace(
                    pair,
                    reason=(
                        f"{NEVER_MERGE_REASON}; reviewer said merge but the "
                        f"rule is absolute and takes precedence"
                    ),
                )
            else:
                pair = replace(
                    pair,
                    accepted=True,
                    reason=f"reviewer decision: merge (automated gate said: {pair.reason})",
                )
        elif decision == "no" and pair.accepted:
            pair = replace(
                pair,
                accepted=False,
                reason=f"reviewer decision: do not merge (automated gate said: {pair.reason})",
            )

        evidence.append(pair)
        if pair.accepted:
            groups.union(pair.left, pair.right)
        if progress and index % 100 == 0:
            print(f"[identity] evaluated {index}/{len(candidates)} candidate pairs")

    return _assemble(by_key, groups, evidence)


def _assemble(
    by_key: dict[str, StationRecord],
    groups: _DisjointSet,
    evidence: list[MatchPair],
) -> Crosswalk:
    """Turn accepted links into the three durable tables."""
    accepted_pairs = {
        tuple(sorted((p.left, p.right))) for p in evidence if p.accepted
    }
    membership = groups.groups()

    alias_rows = []
    history_rows = []

    for canonical_id, members in sorted(membership.items()):
        # A group larger than two whose members were not all compared
        # directly was formed by chaining links; section 1.3 gives no rule
        # for that, so it is flagged rather than silently trusted.
        transitive = False
        if len(members) > 2:
            for i, left in enumerate(members):
                for right in members[i + 1 :]:
                    if tuple(sorted((left, right))) not in accepted_pairs:
                        transitive = True

        for key in members:
            record = by_key[key]
            link = _link_for(key, canonical_id, evidence)
            alias_rows.append(
                {
                    "canonical_station_id": canonical_id,
                    "source": record.source,
                    "source_station_id": record.source_station_id,
                    "name": record.name,
                    "lat": record.lat,
                    "lon": record.lon,
                    "elevation_m": record.elevation_m,
                    "wmo_id": record.wmo_id,
                    "wigos_id": record.wigos_id,
                    "n_variables": len(record.variables),
                    "variables": "|".join(sorted(record.variables)),
                    "is_representative": key == canonical_id,
                    "match_tier": link["tier"],
                    "match_tier_name": TIER_NAMES.get(link["tier"], ""),
                    "match_evidence": link["reason"],
                    # Section 1.3 tier 4: anything not linked to another
                    # source stays separate and carries the risk flag.
                    "duplicate_risk": link["duplicate_risk"],
                    "agreement_review": link["agreement_review"],
                    "never_merge_no_overlap": link["never_merge_no_overlap"],
                    "group_formed_transitively": transitive,
                    "valid_from": record.first_date,
                    "valid_to": record.last_date,
                }
            )

            # Section 1.3: relocation "should create time-bounded metadata
            # segments under the same canonical history ... not silent
            # coordinate averaging".  None of the four sources publishes a
            # relocation history, so each alias contributes one segment with
            # its own coordinates, bounded by its own record dates.  A moved
            # station therefore appears as two segments rather than one
            # averaged position.
            history_rows.append(
                {
                    "canonical_station_id": canonical_id,
                    "segment_source": record.source,
                    "segment_source_station_id": record.source_station_id,
                    "name": record.name,
                    "lat": record.lat,
                    "lon": record.lon,
                    "elevation_m": record.elevation_m,
                    "effective_from": record.first_date,
                    "effective_to": record.last_date,
                    "provenance": "source inventory; no provider relocation history",
                }
            )

    aliases = pd.DataFrame(alias_rows).sort_values(
        ["canonical_station_id", "source", "source_station_id"]
    ).reset_index(drop=True)
    history = pd.DataFrame(history_rows).sort_values(
        ["canonical_station_id", "effective_from", "segment_source"]
    ).reset_index(drop=True)
    evidence_frame = pd.DataFrame([p.to_dict() for p in evidence])
    if not evidence_frame.empty:
        evidence_frame = evidence_frame.sort_values(["left", "right"]).reset_index(drop=True)

    return Crosswalk(
        aliases=aliases,
        metadata_history=history,
        evidence=evidence_frame,
        conflicts=metadata_conflicts(evidence),
    )


def _link_for(key: str, canonical_id: str, evidence: list[MatchPair]) -> dict:
    """
    How one alias came to sit under its canonical station.

    A station that is its own canonical station and was never accepted into
    any group is unresolved: either nothing was close enough to test, or
    every test failed.  Both carry the tier-4 duplicate-risk flag, because a
    failed test is not proof of distinctness -- only of insufficient evidence.
    """
    for pair in evidence:
        if pair.accepted and key in (pair.left, pair.right):
            return {
                "tier": pair.tier,
                "reason": pair.reason,
                "duplicate_risk": False,
                "agreement_review": agreement_needs_review(pair.stats),
                "never_merge_no_overlap": False,
            }

    rejected = [
        p for p in evidence if not p.accepted and key in (p.left, p.right)
    ]
    if rejected:
        closest = min(rejected, key=lambda p: p.distance_km)
        return {
            "tier": TIER_UNRESOLVED,
            "reason": f"nearest untested/failed link: {closest.reason}",
            "duplicate_risk": True,
            "agreement_review": False,
            "never_merge_no_overlap": NEVER_MERGE_REASON in closest.reason,
        }
    return {
        "tier": TIER_UNRESOLVED,
        "reason": "no cross-source station within tolerance",
        "duplicate_risk": False,
        "agreement_review": False,
        "never_merge_no_overlap": False,
    }
