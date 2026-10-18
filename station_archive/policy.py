"""
policy.py
=========
Scientific policy: the domain, the admission window, and every threshold that
decides what the archive says.

Plan section 1.6: "Scientific policy belongs in versioned configuration, while
download, modelling, gridding and adjustment code remain separate."

Nothing in this file is a machine setting -- no timeouts, no paths, no disk
limits; those live in settings.py. Nothing in this file is a claim that a
portal is always best either: source priority is per variable, and the order
was set by a coverage/QC pilot rather than assumed.

WHY A DATACLASS AND NOT MODULE CONSTANTS
----------------------------------------
Every number here used to be a module-level global read directly from 85 sites
across 13 modules. That made the policy inseparable from the process: one
domain per interpreter, thresholds changeable only by editing source, and no
way to record in an archive which policy produced it. A frozen Policy value
fixes all three -- it can be constructed per run, passed explicitly, compared
between runs, and serialised straight into a provenance record.

DEFAULT_POLICY carries exactly the values this archive was built with. Changing
a default changes what a rebuilt archive says, so treat an edit here as a
scientific decision: state the measurement in the comment beside the value, and
write the change up in the GitHub release notes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from station_archive.sources import registry
from station_archive.types import Domain


def _frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """A read-only view, so a Policy cannot be edited through a nested dict."""
    return MappingProxyType(dict(mapping))


# ---------------------------------------------------------------------------
# Canonical variable names
# ---------------------------------------------------------------------------
# CMIP/CORDEX short names, used as the common label across sources. Each
# adapter maps its provider's own element names onto these; the provider name
# is kept alongside in 'provider_element' so nothing is lost.
DEFAULT_VARIABLES: tuple[str, ...] = (
    "tas",       # daily mean near-surface air temperature
    "tasmin",
    "tasmax",
    "pr",        # precipitation
    "hurs",      # relative humidity
    "sfcWind",   # daily mean wind speed
    "gust",      # wind gust
)


@dataclass(frozen=True)
class IdentityPolicy:
    """
    When two source records may be called one physical station (plan 1.3).

    The thresholds are deliberately conservative. Section 1.3 warns that
    "Coordinate tolerance alone is insufficient because airports and nearby
    climatological stations can be distinct", and a false merge leaks a station
    across validation folds, which is the specific harm identity control exists
    to prevent.
    """

    # Tier-2 "coincident coordinates": same site to within ~200 m.
    coincident_deg: float = 0.002

    # Tier-3 "near coordinates within a conservative tolerance": ~3 km.
    #
    # Widened from an initial ~1 km after the first crosswalk run showed the
    # tighter gate rejecting pairs that are provably one record: ecad:002346
    # and ghcnd:BUM00015502 sit 1.26 km apart and are identical to the last
    # decimal on all 3210 days they share (r = 1.000000, mean absolute
    # difference 0.000 degC). Archives disagree about a station's "official"
    # coordinates by more than a kilometre, so distance alone cannot separate a
    # duplicate from a neighbour at this scale.
    #
    # This is a candidate-generation bound, not the decision: the elevation,
    # correlation and minimum-overlap gates still have to pass.
    near_deg: float = 0.027

    # Compatible elevation. Two reports of one station rarely differ by more
    # than a few tens of metres; a larger gap suggests genuinely different sites.
    elevation_tolerance_m: float = 50.0

    # Overlapping daily-series correlation required before any non-identifier
    # merge is accepted.
    correlation_threshold: float = 0.995

    # Minimum shared days before a correlation is trusted at all. Without this
    # a pair sharing five days could reach r = 0.99 by chance and be merged.
    min_overlap_days: int = 365

    # Maximum mean absolute difference, per canonical unit, for a merge.
    #
    # Correlation measures covariation; this measures agreement, and the two
    # come apart. Measured on the archive: pairs 2.4 km apart correlate at
    # 0.9915 while differing 1.3 degC every day (a lapse-rate/exposure
    # difference between genuinely distinct sites), whereas true duplicates
    # cluster at a median difference of 0.000 degC.
    #
    # Non-temperature thresholds are set by analogy rather than measurement:
    # the match variables in practice are tasmax and tasmin, so pr and wind
    # values here have not been calibrated against this archive.
    max_mean_abs_diff: Mapping[str, float] = field(
        default_factory=lambda: _frozen(
            {"degC": 0.5, "mm": 0.5, "m s-1": 0.5, "percent": 3.0}
        )
    )

    # Never merge two source stations that share no observation day at all.
    #
    # A project rule, stricter than anything plan section 1.3 requires: tier 2
    # there would allow a merge on "coincident coordinates plus compatible name
    # and elevation" with no series evidence whatsoever. This forbids that
    # outright, and admits no manual override -- a pair failing it is
    # permanently unmergeable and is not offered for review. Overlap is counted
    # across every canonical variable, so a pair sharing only wind days still
    # counts as having evidence.
    require_any_overlap: bool = True

    # Variables used to test series agreement, in order of preference. tasmax
    # is first because it is the best-populated variable common to ECA&D,
    # GHCN-D and GSOD, and daily maxima are less sensitive than means to
    # differing daily aggregation windows.
    match_variables: tuple[str, ...] = ("tasmax", "tasmin", "pr")

    # Value agreement above which an ACCEPTED merge is sent for review, per
    # canonical unit. A pair can clear the correlation gate while still
    # disagreeing in value, so these merges stand but are marked.
    agreement_review: Mapping[str, float] = field(
        default_factory=lambda: _frozen(
            {"degC": 1.0, "mm": 2.0, "m s-1": 1.0, "percent": 5.0}
        )
    )

    # Agreement at or beyond which two records are treated as the SAME
    # underlying observation for metadata-conflict reporting. Diagnostic only --
    # it never merges anything on its own.
    identical_mad: float = 0.05
    identical_exact_fraction: float = 0.99


@dataclass(frozen=True)
class FallbackPolicy:
    """
    When a lower-priority alias may fill a gap in a higher-priority series.

    Plan section 1.6: "Before allowing a source switch, compare overlapping
    records by month for bias, dispersion, wet-day frequency and
    discontinuities. A failed compatibility test leaves a gap or creates a
    time-bounded station segment; it does not justify blind splicing."

    Measured on confirmed duplicates in this archive, as median values, almost
    entirely tasmax -- so each figure is a TASMAX statistic and describes no
    other variable:
        ecad vs ghcnd   bias  0.0000 degC   (byte-identical records)
        ecad vs gsod    bias  0.0002 degC
        ghcnd vs gsod   bias  0.3030 degC

    Do not read the third line as "GSOD runs warm". It is one median over 19
    tasmax pairs, it is not reproduced for other variables or source pairs, and
    GHCN-D supplies no tas at all, so it says nothing about the ECA&D/GSOD tas
    fallbacks that make up most of this archive's switches -- those run at a
    median -0.16 degC, the opposite sign. A correction built on that misreading
    was applied and withdrawn; see standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION.
    """

    min_overlap_days: int = 180

    # Minimum shared days within a calendar month before that month contributes
    # a monthly mean to the discontinuity test below.
    #
    # The same reasoning as HomogeneityPolicy.min_days_per_month, which has
    # always carried it: a month built from one or two shared days is not a
    # monthly mean, and letting it compete lets a single outlier veto a splice
    # on the max_monthly_departure gate. That gate is the second-largest
    # refusal cause in this archive (48 of 157), so what enters it matters.
    #
    # Excluding thin months is strictly more permissive, which opens a hole at
    # the other end: an overlap long enough to clear min_overlap_days but spread
    # so thinly that no month qualifies would leave the gate with nothing to
    # test. check_compatibility refuses outright in that case rather than
    # passing a test it could not run.
    min_days_per_month: int = 20

    # The default admits an offset of the size measured above rather than
    # blocking it; tighten max_bias below 0.30 to refuse those splices instead.
    max_bias: Mapping[str, float] = field(
        default_factory=lambda: _frozen(
            {"degC": 0.5, "mm": 1.0, "m s-1": 0.5, "percent": 5.0}
        )
    )

    # Ratio of fallback standard deviation to primary. A source recording the
    # same quantity should not be materially smoother or noisier.
    dispersion_range: tuple[float, float] = (0.75, 1.33)

    # Absolute difference in wet-day frequency permitted before a precipitation
    # source switch is refused. GSOD's trace-as-zero convention moves this
    # directly, which is why it is tested separately from bias.
    max_wet_day_diff: float = 0.10
    wet_day_threshold_mm: float = 1.0

    # A month whose bias departs this far from the overall bias is a
    # discontinuity: the two records agree on average only by cancellation.
    max_monthly_departure: Mapping[str, float] = field(
        default_factory=lambda: _frozen(
            {"degC": 1.0, "mm": 2.0, "m s-1": 1.0, "percent": 10.0}
        )
    )


@dataclass(frozen=True)
class HomogeneityPolicy:
    """
    How conservatively breakpoints are detected (plan 1.4: diagnose, never
    silently homogenize).

    Published method comparisons (MULTITEST) show the fraction of series judged
    homogeneous swinging from 85% (HOMER) to 22% (ACMANT) on identical data --
    how conservative a method's detection is matters more than which statistic
    it uses. Since this is a flag consumed downstream rather than an
    adjustment, it is tuned towards missing a marginal break rather than
    flagging a sound station.
    """

    # Minimum days of data within a calendar month before that month yields a
    # usable monthly value. A month built from a handful of days is not a
    # monthly mean.
    min_days_per_month: int = 20

    # Minimum overlapping months between target and a candidate neighbour
    # before the neighbour is trusted to contribute to the reference series.
    min_neighbor_overlap_months: int = 24

    # Minimum Pearson correlation on the shared overlap, measured on FIRST
    # DIFFERENCES of the deseasonalised anomalies, before a candidate neighbour
    # is considered to be tracking the same weather.
    min_neighbor_correlation: float = 0.6

    # Neighbours are searched within this radius before concluding none exist.
    neighbor_search_radius_km: float = 100.0

    # Cap on how many passing neighbours are used. More neighbours stabilise
    # the reference series; this only bounds how many are tried.
    max_neighbors: int = 5

    # A break cannot be attributed with a single neighbour: target-minus-
    # neighbour shows the same step whether the target moved or the neighbour
    # did. PHA resolves attribution across the whole network before adjusting
    # anything; the cheap equivalent here is to refuse to call anything
    # significant on one neighbour's word.
    min_neighbors: int = 2

    # Minimum length, in months, either side of a candidate breakpoint. Without
    # this, a split at the very edge of the record lets that short segment's
    # mean be dominated by one or two extreme values, producing a spuriously
    # huge SNHT statistic that reflects outliers, not a sustained shift.
    min_segment_months: int = 12

    # Conservative floor on the SNHT statistic itself, on top of the
    # permutation p-value. Climatol applies a threshold of SNHT = 20 "in order
    # to avoid false positives" rather than relying on nominal significance.
    min_snht_statistic: float = 20.0

    # Variables aggregated to a monthly TOTAL rather than a monthly mean.
    sum_variables: frozenset[str] = frozenset({"pr"})

    # Variables tested on a MULTIPLICATIVE rather than an additive basis.
    # SNHT assumes constant variance; precipitation's spread grows with its
    # level, so a difference in millimetres is heteroscedastic by construction.
    # Taking logarithms turns the ratio into a difference again, so the whole
    # additive machinery applies unchanged while the variance is stabilised.
    log_variables: frozenset[str] = frozenset({"pr"})

    # Offset added before taking logarithms, in the variable's own units, so
    # that a genuinely dry month (0 mm) is representable at all.
    precip_log_offset_mm: float = 1.0

    # Calendar months whose climatological total is below this are dropped from
    # a log-basis series entirely. At a 2 mm climatology the log ratio is
    # driven by the offset and by trace amounts rather than by the station.
    min_precip_climatology_mm: float = 5.0

    # Significance defaults, overridable per run from the CLI.
    n_permutations: int = 500
    alpha: float = 0.05


@dataclass(frozen=True)
class DemPolicy:
    """
    How an elevation disagreement between archives is settled (plan 1.4).

    Section 1.4 forbids silently homogenising a raw observation, and nothing
    here does: a DEM decides only which of the elevations the ARCHIVES already
    published is right. Elevation is metadata and a model predictor, not an
    observation, and a station carrying the wrong height corrupts the fitted
    relationship however its identity is resolved.
    """

    # Spread between an aggregated station's alias elevations, above which the
    # disagreement is worth settling. Below it, the archives differ by about as
    # much as a DEM pixel could resolve, so a DEM cannot improve on them.
    elevation_disagreement_m: float = 10.0

    # AWS Terrain Tiles zoom. 15 matches the parent downscaling project's own
    # DEM script. Over Europe the underlying data is ~30 m SRTM/EU-DEM, so a
    # zoom-15 pixel (~4.8 m here) is interpolated rather than independent
    # detail -- ample for choosing between candidates tens of metres apart,
    # which is the only thing it is asked to do.
    zoom: int = 15


@dataclass(frozen=True)
class ClimatologyPolicy:
    """
    When a record's long-term mean is too far from an independent reference
    to be believed (plan 1.4: QC layered over provider flags, with transparent
    rejection reasons).

    The screen measures each source record's departure from a reference
    sampled at the same place on the same days, then scores it against the
    spread of every other record of that variable. It proposes; a reviewer
    decides; only consolidation acts, and only on an explicit 'yes'.

    Measured on this archive against CERRA before the rule was set (n >= 365
    matched days): the fixed "half or double" rule flagged no gust record,
    18 sfcWind and 31 pr; the robust-z rule at 4 flagged gust 0, sfcWind 5,
    pr 29, tas 4, tasmax 3, tasmin 1, hurs 0. The pr flags are dominated by
    ECA&D series reporting exactly zero on 400-6,600 matched days, and the tasmax
    flags by a Lazio block reading +5 to +6 degC above the reference.
    """

    # Matched station/reference days before a departure is computed. A year,
    # so a seasonal record cannot pass for a climatology.
    min_matched_days: int = 365

    # Robust z = (departure - median) / (1.4826 * MAD), per variable over every
    # tested record. Above exclude_z the record is proposed for exclusion;
    # above review_z it is shown for review only. Agreed with the archive's
    # owner, not fitted: tuning these until a particular network is caught
    # would build that network's comparison artefact into the rule.
    review_z: float = 3.0
    exclude_z: float = 4.0

    # Compared as a log ratio: their spread grows with their level, so a
    # difference would make a wet or windy site look anomalous by being wet or
    # windy. A record whose mean is zero against a positive reference departs
    # by -inf, which is the strongest flag there is.
    multiplicative_variables: frozenset[str] = frozenset({"pr", "sfcWind", "gust"})

    # Temperature is compared in degC after moving the reference to the
    # station's height at the standard-atmosphere lapse rate. A diagnostic
    # adjustment only: no value is ever changed by it. Without it every
    # mountain station would sit several degrees from a reference whose
    # orography is a 5.5 km average.
    lapse_rate_k_per_m: float = 0.0065
    lapse_variables: frozenset[str] = frozenset({"tas", "tasmin", "tasmax"})

    # Neighbour context for the reviewer, never a gate: the same radius the
    # homogeneity neighbour search uses.
    neighbor_radius_km: float = 100.0

    def __post_init__(self) -> None:
        if not self.review_z < self.exclude_z:
            raise ValueError(
                f"review_z ({self.review_z}) must be below exclude_z "
                f"({self.exclude_z}): the review tier sits between them"
            )


# ---------------------------------------------------------------------------
# Source selection (plan section 1.6)
# ---------------------------------------------------------------------------
# "Do not encode one universal claim that a portal is always 'best.' Store a
# priority list per variable because coverage and semantics differ."
#
# The plan offers a starting order and calls it "a research hypothesis, not the
# final order. Confirm it from the Step 1 coverage/QC pilot." The pilot has run,
# and it contradicts the hypothesis in three places:
#
#   * c3s_land is ranked first for hurs and gust in the hypothesis but carries
#     neither -- and every one of its stations in this domain is an exact
#     GHCN-D identifier, so it is a redelivery rather than independent
#     evidence. It is excluded from selection entirely (registry: selectable
#     is False) and kept as an audit cross-check.
#   * hurs has exactly one source in this domain: ECA&D, at 425 stations.
#   * gsod is ranked above ghcnd for pr in the hypothesis, but GSOD's
#     precipitation is the least trustworthy of the three: its own readme
#     states "'0' indicates no measurable precipitation (includes a trace)",
#     and 46% of its zero rows carry flag I, meaning the station reported no
#     precipitation data at all that day. Coverage favours GSOD (203 stations
#     against GHCN-D's 155); semantics do not. Semantics win.
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
DEFAULT_SOURCE_PRIORITY: Mapping[str, tuple[str, ...]] = _frozen(
    {
        "tas":     ("ecad", "gsod"),
        "tasmin":  ("ecad", "ghcnd", "gsod"),
        "tasmax":  ("ecad", "ghcnd", "gsod"),
        "pr":      ("ecad", "ghcnd", "gsod"),
        "hurs":    ("ecad",),
        "sfcWind": ("ecad", "gsod", "ghcnd"),
        "gust":    ("gsod", "ecad", "ghcnd"),
    }
)


@dataclass(frozen=True)
class Policy:
    """
    Everything scientific about one run, in one immutable value.

    Pass it explicitly rather than reading globals, so that two domains can be
    built in one process, a run can be reproduced from a recorded policy, and
    an archive can state which policy produced it (see cli 'provenance').
    """

    domain: Domain
    period: tuple[int, int]
    fetch_years: tuple[int, int]
    variables: tuple[str, ...] = DEFAULT_VARIABLES
    # default_factory, not a bare default: dataclasses reject a mappingproxy as
    # a class-level default even though it is read-only.
    source_priority: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_SOURCE_PRIORITY
    )
    identity: IdentityPolicy = field(default_factory=IdentityPolicy)
    fallback: FallbackPolicy = field(default_factory=FallbackPolicy)
    homogeneity: HomogeneityPolicy = field(default_factory=HomogeneityPolicy)
    dem: DemPolicy = field(default_factory=DemPolicy)
    climatology: ClimatologyPolicy = field(default_factory=ClimatologyPolicy)

    def __post_init__(self) -> None:
        # Fail at construction rather than three stages later. The old
        # arrangement had no check at all, so a source named in a priority list
        # but absent from the adapter map failed with a KeyError deep in a
        # loader, and one omitted from the series list failed silently.
        registry.validate_priority(self.source_priority)
        if self.period[0] > self.period[1]:
            raise ValueError(f"period start after end: {self.period}")
        if self.fetch_years[0] > self.fetch_years[1]:
            raise ValueError(f"fetch_years start after end: {self.fetch_years}")
        unknown = [v for v in self.source_priority if v not in self.variables]
        if unknown:
            raise ValueError(
                f"source priority names variables absent from 'variables': {unknown}"
            )
        # A selected variable must be range-checkable. Deferred import because
        # qc is a consumer of policy everywhere else; this is the one edge that
        # runs the other way, and importing at module scope would make the
        # cycle real rather than merely awkward.
        from station_archive import qc

        ungated = sorted(set(self.source_priority) - qc.gated_variables())
        if ungated:
            raise ValueError(
                f"source priority selects variables with no gross-range check: "
                f"{ungated}. Values for these would reach the gold table with no "
                f"range check at all -- add bounds to qc.GROSS_RANGE first, or "
                f"carry the variable in 'variables' without selecting it."
            )

    # ---- source capabilities, read from the registry ---------------------

    def crosswalk_sources(self) -> tuple[str, ...]:
        return registry.crosswalk_sources()

    def series_sources(self) -> tuple[str, ...]:
        return registry.series_sources()

    def selectable_sources(self) -> tuple[str, ...]:
        return registry.selectable_sources()

    def priority_for(self, variable: str) -> tuple[str, ...]:
        return tuple(self.source_priority.get(variable, ()))

    # ---- serialisation, for provenance and for --policy ------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Plain-data form, suitable for a provenance record or a policy file.

        Mappings are converted out of their read-only views and sets out of
        frozenset so the result is JSON-serialisable.
        """

        # A manual walk rather than dataclasses.asdict, which deep-copies and
        # therefore cannot handle the read-only mappings that make a Policy
        # genuinely immutable.
        def plain(value: Any) -> Any:
            if is_dataclass(value):
                return {
                    f.name: plain(getattr(value, f.name))
                    for f in dataclass_fields(value)
                }
            if isinstance(value, Mapping):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (frozenset, set)):
                return sorted(value)
            if isinstance(value, tuple):
                return list(value)
            return value

        return {f.name: plain(getattr(self, f.name)) for f in dataclass_fields(self)}

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Policy":
        """
        Build a Policy from plain data, filling anything absent from the
        defaults. A partial policy file is therefore an override list rather
        than a full restatement, which keeps a domain change to three lines.
        """
        domain = data.get("domain")
        if isinstance(domain, Mapping):
            domain = Domain(**domain)
        kwargs: dict[str, Any] = {}
        if domain is not None:
            kwargs["domain"] = domain
        for name in ("period", "fetch_years"):
            if name in data:
                kwargs[name] = tuple(data[name])
        if "variables" in data:
            kwargs["variables"] = tuple(data["variables"])
        if "source_priority" in data:
            kwargs["source_priority"] = _frozen(
                {k: tuple(v) for k, v in data["source_priority"].items()}
            )
        for name, klass in (
            ("identity", IdentityPolicy),
            ("fallback", FallbackPolicy),
            ("homogeneity", HomogeneityPolicy),
            ("dem", DemPolicy),
            ("climatology", ClimatologyPolicy),
        ):
            if name in data:
                kwargs[name] = _sub_policy(klass, data[name])
        # Always an override of the default, never a construction from nothing:
        # a policy file that names only a domain is a complete, valid policy,
        # and every threshold it omits is inherited and recorded as such.
        return replace(DEFAULT_POLICY, **kwargs)

    @classmethod
    def from_json(cls, path: Path) -> "Policy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sub_policy(klass: type, data: Mapping[str, Any]) -> Any:
    """
    Construct a sub-policy, coercing JSON types back to the field types.

    JSON has no tuple, frozenset or read-only mapping, so each value is coerced
    to whatever the default for that field already is -- one defaults instance,
    built once, rather than one per key.
    """
    fields = {f.name: f for f in klass.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    defaults = klass()
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"{klass.__name__} has no field {key!r}")
        current = getattr(defaults, key)
        if isinstance(current, Mapping):
            kwargs[key] = _frozen(value)
        elif isinstance(current, frozenset):
            kwargs[key] = frozenset(value)
        elif isinstance(current, tuple):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return klass(**kwargs)


# ---------------------------------------------------------------------------
# The policy this archive was built with
# ---------------------------------------------------------------------------
# Domain matches the training box the original LightGBM set was filtered to.
# The buffer admits stations outside that box whose records still describe
# systems crossing the target area (plan 1.2 asks for the training domain "and
# a configurable synoptic buffer").
DEFAULT_DOMAIN = Domain(
    lat_min=33.0,
    lat_max=43.0,
    lon_min=11.0,
    lon_max=32.0,
    buffer_deg=2.0,
)

# Plan section 1.2: "The historical overlap period 1985-2005 is the minimum
# admission window for the present CERRA model. Longer records should be
# retained for QC, homogeneity diagnosis and later QDM sensitivity, but must
# not leak future information into a historical calibration experiment."
#
# Completeness statistics in the inventory are computed over this window.
# Downloads are NOT truncated to it: GHCN-D and ECA&D publish whole-series
# files with no subsetting option, so the longer record costs nothing extra.
DEFAULT_PERIOD = (1985, 2005)

# Years fetched where the provider does subset by year. Wider than the
# admission window so the retained-record requirement above is satisfied.
DEFAULT_FETCH_YEARS = (1970, 2020)

DEFAULT_POLICY = Policy(
    domain=DEFAULT_DOMAIN,
    period=DEFAULT_PERIOD,
    fetch_years=DEFAULT_FETCH_YEARS,
)
