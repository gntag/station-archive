"""
homogeneity.py
===============
Breakpoint / inhomogeneity diagnosis (plan section 1.4: "diagnose
breakpoints/inhomogeneity, but do not silently homogenize raw observations").

WHAT A BREAKPOINT IS
---------------------
A date at which a station's record shifts in a way that is not genuine
climate variability -- a relocation, an instrument change, a change in
observation time, growth around the sensor. The standard way to find one
without independently knowing station history is RELATIVE homogeneity
testing: build a difference series between the target station and several
nearby, independently-operating stations (real weather moves both together;
an artificial break shows up only in the target), then apply the Standard
Normal Homogeneity Test (SNHT; Alexandersson 1986) to that difference series
to find the single most likely breakpoint and how strong the evidence for it
is.

WHY THIS MATTERS FOR THIS ARCHIVE
-----------------------------------
An undetected breakpoint stitches two different physical regimes into one
series that downscaling models and, more importantly, the downstream
QDM/ISIMIP3BASD bias-adjustment stage both treat as one continuous physical
record. Since separating genuine climate-change signal from historical bias
is the entire point of that stage, an artificial jump can be misread as
trend, or mask a real one.

HOW IT IS TUNED
----------------
Published method comparisons (MULTITEST) show the fraction of series judged
homogeneous swinging from 85% (HOMER) to 22% (ACMANT) on identical data --
how conservative a method's detection is matters more than which statistic
it uses. Since this is a flag consumed downstream, not an adjustment, it is
tuned towards missing a marginal break rather than flagging a sound station:
testing runs on monthly anomalies rather than daily values, the SNHT
statistic must clear Climatol's conservative threshold of 20 as well as the
permutation p-value, at least two neighbors must agree, and the whole report
carries a Benjamini-Hochberg correction across every test in it.

WHAT THIS MODULE DOES NOT DO
------------------------------
- It never alters a value. Every function here returns a diagnostic, never a
  corrected series -- "do not silently homogenize raw observations". Whether
  to segment a flagged station, drop a segment, or leave it is a policy
  decision this module does not make.
- It does not attribute a break across the network the way PHA does. With
  at least two neighbors a break is more likely the target's than any one
  neighbor's, but this module never proves which station moved.
- It finds one candidate breakpoint per call, not an iterative multi-break
  decomposition. RHtests/HOMER/ACMANT-style tools recurse (split at the
  found break, test each half again); that is future work.
- Significance is judged by a permutation test on the station's own data
  rather than a published critical-value table. Alexandersson & Moberg
  (1997) is the textbook table, but transcribing specific numbers from
  memory risks getting them wrong; a permutation test answers the same
  question -- "how extreme is this statistic relative to what pure noise
  from this series would produce?" -- directly from the data under test,
  with no number that has to be trusted rather than computed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from station_archive.identity import haversine_km
from station_archive.policy import DEFAULT_POLICY

# Testing runs on MONTHLY anomalies, not daily values. Established practice
# (Climatol, HOMER, ACMANT, PHA) detects breaks on monthly aggregates and
# only then splits the daily series at the detected dates: a daily
# difference series has far worse signal-to-noise, and break detection gets
# harder as that ratio falls. Deseasonalising (subtracting each series' own
# calendar-month climatology) additionally keeps the neighbor-correlation
# gate meaningful -- raw monthly temperature series all correlate at ~0.99
# purely through the shared annual cycle, so a correlation gate on raw
# monthly values would pass essentially anything.

# Minimum days of data within a calendar month before that month yields a
# usable monthly value. A month built from a handful of days is not a
# monthly mean.
# Every threshold below is scientific policy and lives in policy.py. The names
# are kept here as read-only aliases because they are the vocabulary the
# docstrings above use, and because a reader looking for "how conservative is
# this" should find the answer next to the method that applies it.
_H = DEFAULT_POLICY.homogeneity

MIN_DAYS_PER_MONTH = _H.min_days_per_month
MIN_NEIGHBOR_OVERLAP_MONTHS = _H.min_neighbor_overlap_months
MIN_NEIGHBOR_CORRELATION = _H.min_neighbor_correlation
NEIGHBOR_SEARCH_RADIUS_KM = _H.neighbor_search_radius_km
MAX_NEIGHBORS = _H.max_neighbors
MIN_NEIGHBORS = _H.min_neighbors
MIN_SEGMENT_MONTHS = _H.min_segment_months
MIN_SNHT_STATISTIC = _H.min_snht_statistic
SUM_VARIABLES = _H.sum_variables
LOG_VARIABLES = _H.log_variables
PRECIP_LOG_OFFSET_MM = _H.precip_log_offset_mm
MIN_PRECIP_CLIMATOLOGY_MM = _H.min_precip_climatology_mm


@dataclass(frozen=True)
class BreakpointResult:
    """
    One candidate breakpoint for one station and variable.

    `status` says whether a test actually ran, so a row that produced no
    breakpoint is distinguishable from one that was never testable:
      tested              -- the test ran and this is its result
      no_usable_neighbor  -- no candidate neighbor passed overlap/correlation
      too_few_neighbors   -- fewer than MIN_NEIGHBORS passed; a break cannot
                             be attributed to target rather than neighbor
      no_supported_window -- neighbors passed overall, but no candidate
                             breakpoint has MIN_NEIGHBORS of them behind it,
                             so nothing in this series can be attributed
      series_too_short    -- too few months for a non-degenerate search
      no_variance         -- the difference series is constant

    `transform` says how to read `shift`, and must be consulted before the
    number is used:
      difference -- `shift` is in the variable's own monthly units: degC
                    (or %, m/s) for a monthly mean.
      log_ratio  -- `shift` is in natural-log units, because the variable is
                    tested multiplicatively (see LOG_VARIABLES).
                    exp(shift) is the factor the station's monthly totals
                    changed by; `shift_as_ratio` does that conversion.
    """

    canonical_station_id: str
    variable: str
    n_neighbors_used: int
    n_months_tested: int
    breakpoint_date: Optional[pd.Timestamp]
    statistic: float
    p_value: float
    significant: bool
    segment_before_mean: float
    segment_after_mean: float
    shift: float
    status: str = "tested"
    transform: str = "difference"
    # How many neighbours actually stood behind the month the break was found
    # in, and how many months in the series meet the minimum at all. Without
    # these, n_neighbors_used reads as though the two-neighbour safeguard
    # applied at the breakpoint when it may not have.
    min_support_at_break: int = 0
    n_months_min_support: int = 0

    def shift_as_ratio(self) -> float:
        """
        `shift` expressed as a multiplicative factor, for log_ratio results.
        1.0 means no change; 1.2 means 20% wetter after the breakpoint.
        Returns NaN for additive results, where a ratio is not defined.
        """
        if self.transform != "log_ratio" or not np.isfinite(self.shift):
            return float("nan")
        return float(np.exp(self.shift))

    def to_dict(self) -> dict:
        return {
            "canonical_station_id": self.canonical_station_id,
            "variable": self.variable,
            "n_neighbors_used": self.n_neighbors_used,
            "n_months_tested": self.n_months_tested,
            "breakpoint_date": self.breakpoint_date,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "segment_before_mean": self.segment_before_mean,
            "segment_after_mean": self.segment_after_mean,
            "shift": self.shift,
            "status": self.status,
            "transform": self.transform,
            "shift_ratio": self.shift_as_ratio(),
            "min_support_at_break": self.min_support_at_break,
            "n_months_min_support": self.n_months_min_support,
        }


def transform_for(variable: str) -> str:
    """
    Which basis a variable is tested on: "log_ratio" for LOG_VARIABLES,
    "difference" otherwise. Recorded on every result so `shift` is never
    read in the wrong units.
    """
    return "log_ratio" if variable in LOG_VARIABLES else "difference"


def to_monthly_anomalies(
    series: pd.Series,
    variable: str,
    min_days: int = MIN_DAYS_PER_MONTH,
) -> pd.Series:
    """
    Aggregate a daily series to monthly values, then deseasonalise.

    Months with fewer than min_days observations are dropped rather than
    averaged from a handful of days. Precipitation aggregates to a monthly
    total (SUM_VARIABLES); everything else to a monthly mean. The calendar-
    month climatology is then subtracted, so what remains is the anomaly --
    the part of the signal a breakpoint can actually show up in, with the
    annual cycle (which every neighbor shares anyway) taken out.

    For LOG_VARIABLES the same steps run in log space, which makes the
    anomaly a log ratio to the calendar month's geometric mean rather than a
    difference from its arithmetic mean -- see LOG_VARIABLES for why
    precipitation must be tested multiplicatively. Calendar months too dry
    for a ratio to mean anything (below MIN_PRECIP_CLIMATOLOGY_MM) are
    dropped before the transform.
    """
    if series.empty:
        return pd.Series(dtype="float64")

    grouped = series.groupby(series.index.to_period("M"))
    # count(), not size(): size() counts rows including NaN while the
    # aggregation below skips them, so a NaN-padded month cleared the threshold
    # on rows it had no values for. min_days is a statement about how many
    # observations built the monthly value.
    counts = grouped.count()
    monthly = grouped.sum() if variable in SUM_VARIABLES else grouped.mean()
    monthly = monthly[counts >= min_days]
    if monthly.empty:
        return pd.Series(dtype="float64")

    monthly.index = monthly.index.to_timestamp()

    if variable in LOG_VARIABLES:
        # Drop near-dry calendar months on the RAW totals, before the
        # transform, so the threshold keeps its physical meaning in mm.
        wet_enough = monthly.groupby(monthly.index.month).transform("mean") >= MIN_PRECIP_CLIMATOLOGY_MM
        monthly = monthly[wet_enough]
        if monthly.empty:
            return pd.Series(dtype="float64")
        monthly = np.log(monthly.clip(lower=0.0) + PRECIP_LOG_OFFSET_MM)

    climatology = monthly.groupby(monthly.index.month).transform("mean")
    return (monthly - climatology).sort_index()


def select_neighbors(
    target_id: str,
    target_lat: float,
    target_lon: float,
    candidates: pd.DataFrame,
    max_neighbors: int = MAX_NEIGHBORS,
    radius_km: float = NEIGHBOR_SEARCH_RADIUS_KM,
) -> list[str]:
    """
    Nearest candidate canonical stations within radius_km, closest first.

    candidates: DataFrame with canonical_station_id, lat, lon (e.g.
    canonical_stations.csv). This proposes candidates by distance only --
    the overlap/correlation gates in build_reference_difference run later,
    once each candidate's actual series is loaded, because distance alone
    cannot decide those.
    """
    others = candidates[candidates["canonical_station_id"] != target_id]
    if others.empty:
        return []
    distance_km = [
        haversine_km(target_lat, target_lon, lat, lon)
        for lat, lon in zip(others["lat"], others["lon"])
    ]
    ranked = others.assign(distance_km=distance_km)
    ranked = ranked[ranked["distance_km"] <= radius_km].sort_values("distance_km")
    return ranked["canonical_station_id"].head(max_neighbors).tolist()


def passing_neighbor_differences(
    target: pd.Series,
    neighbors: dict[str, pd.Series],
    min_overlap: int = MIN_NEIGHBOR_OVERLAP_MONTHS,
    min_correlation: float = MIN_NEIGHBOR_CORRELATION,
) -> list[pd.Series]:
    """
    One (target - neighbor) series per neighbor that passes the gates, kept
    SEPARATE rather than averaged.

    Keeping them separate is what makes per-period support countable. A single
    averaged reference series cannot say how many neighbors stood behind any
    given month, and that number is the whole content of the two-neighbor rule:
    a month supported by one neighbor carries no more attribution power than a
    one-neighbor test does, however many neighbors the series has in total.

    Each neighbor is gated independently before contributing, rather than
    averaging raw neighbor values first, so one bad or unrelated neighbor
    cannot poison the reference the way it could if it were blended in before
    any check ran.
    """
    differences: list[pd.Series] = []
    for series in neighbors.values():
        shared = target.index.intersection(series.index)
        if len(shared) < min_overlap:
            continue
        a = target.loc[shared].to_numpy(dtype="float64")
        b = series.loc[shared].to_numpy(dtype="float64")
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() < min_overlap:
            continue
        a, b, idx = a[finite], b[finite], shared[finite]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        # Correlate FIRST DIFFERENCES, not levels. A step change in the
        # target inflates its own variance without the neighbor sharing it,
        # which drags the level correlation down and would reject a perfectly
        # good neighbor exactly when the target's break is largest -- the
        # gate would be self-defeating. Differencing turns a step into a
        # single spike, so the measure stays robust to the very
        # inhomogeneities being looked for. This is why PHA selects its
        # neighbors on first-differenced series too.
        da, db = np.diff(a), np.diff(b)
        if da.size == 0 or np.std(da) == 0 or np.std(db) == 0:
            continue
        correlation = float(np.corrcoef(da, db)[0, 1])
        if correlation < min_correlation:
            continue
        differences.append(pd.Series(a - b, index=idx))

    return differences


def neighbor_support(differences: list[pd.Series]) -> pd.Series:
    """
    How many neighbors contribute to each period, indexed like the reference.

    This is the number the two-neighbor rule has to be checked against. The
    whole-series count answers a different and weaker question -- "did two
    neighbors pass anywhere in the record" -- which a break sitting in a
    single-neighbor window satisfies without being supported by anything.
    """
    if not differences:
        return pd.Series(dtype="int64")
    combined = pd.concat(differences, axis=1)
    support = combined.notna().sum(axis=1)
    return support[support > 0].sort_index().astype("int64")


def combine_reference(
    differences: list[pd.Series],
) -> tuple[pd.Series, pd.Series]:
    """The mean difference series and its per-period support, index-aligned."""
    if not differences:
        return pd.Series(dtype="float64"), pd.Series(dtype="int64")
    combined = pd.concat(differences, axis=1)
    reference = combined.mean(axis=1).dropna().sort_index()
    support = combined.notna().sum(axis=1).reindex(reference.index).astype("int64")
    return reference, support


def build_reference_difference(
    target: pd.Series,
    neighbors: dict[str, pd.Series],
    min_overlap: int = MIN_NEIGHBOR_OVERLAP_MONTHS,
    min_correlation: float = MIN_NEIGHBOR_CORRELATION,
) -> tuple[pd.Series, int]:
    """
    Period-by-period mean of (target - neighbor), across every neighbor that
    individually passes the overlap and correlation gates. Callers pass
    monthly anomalies (see to_monthly_anomalies), so `min_overlap` counts
    shared months.

    Returns (difference_series, n_neighbors_used). An empty series and 0
    means no candidate neighbor was usable -- the caller must not attempt a
    breakpoint test without a reference.

    Note that n_neighbors_used counts neighbors over the WHOLE series. Use
    neighbor_support() for the per-period count that the two-neighbor rule is
    actually about; detect_breakpoint does.
    """
    differences = passing_neighbor_differences(
        target, neighbors, min_overlap, min_correlation
    )
    reference, _ = combine_reference(differences)
    return reference, len(differences)


def snht_statistic(standardized: np.ndarray) -> np.ndarray:
    """
    Standard Normal Homogeneity Test statistic T(k) at every candidate
    breakpoint k = 1..n-1 (Alexandersson 1986).

    `standardized` must already be a zero-mean, unit-variance series (a
    difference series standardized over its own full period). For a
    candidate split into z[:k] and z[k:]:

        T(k) = k * mean(z[:k])**2 + (n-k) * mean(z[k:])**2

    A genuine single step change maximises T(k) at the true break; pure
    noise produces a comparatively flat, low profile. Returns the profile
    indexed 0..n-2, where profile[i] is T(k=i+1).
    """
    n = len(standardized)
    if n < 2:
        return np.array([])
    cumulative = np.cumsum(standardized)
    total = cumulative[-1]
    k = np.arange(1, n)
    mean_before = cumulative[k - 1] / k
    mean_after = (total - cumulative[k - 1]) / (n - k)
    return k * mean_before ** 2 + (n - k) * mean_after ** 2


def _restricted_max(
    profile: np.ndarray,
    n: int,
    min_segment: int,
    eligible: Optional[np.ndarray] = None,
) -> tuple[Optional[int], float]:
    """
    argmax of `profile` (as returned by snht_statistic, indexed 0..n-2 for
    k=1..n-1) restricted to candidates with at least min_segment days on
    both sides, and only when that leaves a genuinely searchable window
    (n >= 3*min_segment -- see below). Returns (best_k, best_t), or
    (None, -inf) if no candidate in range qualifies -- callers must check
    for None before using best_k.
    """
    if profile.size == 0:
        return None, float("-inf")
    if n < 3 * min_segment:
        # A total length only just above 2*min_segment leaves almost no
        # freedom in where the split can fall (near n = 2*min_segment, the
        # valid range collapses to a single point at the midpoint) -- not
        # enough choice for "this is the best split" to mean anything, so
        # any noise looks like a confident detection. Requiring the search
        # window itself (n - 2*min_segment) to be at least min_segment wide
        # rules that degenerate case out before it can produce a result.
        return None, float("-inf")
    k = np.arange(1, n)
    in_range = (k >= min_segment) & (k <= n - min_segment)
    if eligible is not None:
        # A candidate the evidence cannot support is not a candidate. Applying
        # this to the observed statistic AND to every permutation keeps the null
        # distribution comparable: both maximise over the same admissible set.
        in_range = in_range & np.asarray(eligible, dtype=bool)
    if not in_range.any():
        return None, float("-inf")
    masked = np.where(in_range, profile, -np.inf)
    best_idx = int(np.argmax(masked))
    return int(k[best_idx]), float(profile[best_idx])


def detect_breakpoint(
    canonical_station_id: str,
    variable: str,
    target: pd.Series,
    neighbors: dict[str, pd.Series],
    n_permutations: int = 500,
    alpha: float = 0.05,
    min_segment_months: int = MIN_SEGMENT_MONTHS,
    min_neighbors: int = MIN_NEIGHBORS,
    min_statistic: float = MIN_SNHT_STATISTIC,
    seed: int = 0,
) -> BreakpointResult:
    """
    Test one station/variable series for a single candidate breakpoint.

    `target` and `neighbors` are DAILY series; they are aggregated to
    monthly anomalies here (see to_monthly_anomalies) before anything is
    tested, because break detection on daily values has far worse
    signal-to-noise than on monthly aggregates.

    Candidate splits within min_segment_months of either end are never
    considered -- a short segment's mean is not meaningfully comparable to
    the rest of the series, and letting it compete produces spuriously
    large statistics driven by a handful of extreme values.

    A result is called significant only when all three hold:
      * the permutation p-value is below alpha -- the observed max T(k) is
        beyond what shuffles of this same series produce by chance;
      * the statistic clears min_statistic (Climatol's conservative
        SNHT >= 20 convention, to suppress marginal detections); and
      * at least min_neighbors neighbors contributed, since one neighbor
        cannot say whether the target or the neighbor moved.
    """
    transform = transform_for(variable)
    monthly_target = to_monthly_anomalies(target, variable)
    monthly_neighbors = {
        name: to_monthly_anomalies(series, variable)
        for name, series in neighbors.items()
    }
    differences = passing_neighbor_differences(monthly_target, monthly_neighbors)
    n_used = len(differences)
    reference, support = combine_reference(differences)
    n_supported = int((support >= min_neighbors).sum()) if len(support) else 0

    def _null(status: str) -> BreakpointResult:
        return BreakpointResult(
            canonical_station_id, variable, n_used, len(reference),
            None, float("nan"), float("nan"), False,
            float("nan"), float("nan"), float("nan"), status, transform,
            0, n_supported,
        )

    if n_used == 0:
        return _null("no_usable_neighbor")
    if n_used < min_neighbors:
        return _null("too_few_neighbors")
    if len(reference) < 3 * min_segment_months:
        return _null("series_too_short")

    values = reference.to_numpy(dtype="float64")
    mean, std = float(np.mean(values)), float(np.std(values))
    if std == 0:
        return BreakpointResult(
            canonical_station_id, variable, n_used, len(reference),
            None, 0.0, 1.0, False, mean, mean, 0.0, "no_variance", transform,
            0, n_supported,
        )
    standardized = (values - mean) / std
    n = len(standardized)

    # A candidate breakpoint at k means reference.index[k] is the first period
    # of the shifted segment, so eligibility is the support at that period.
    eligible = (support.to_numpy() >= min_neighbors)[1:]

    profile = snht_statistic(standardized)
    best_k, observed_t = _restricted_max(profile, n, min_segment_months, eligible)
    if best_k is None:
        # Two different failures, and the distinction matters to a reader: a
        # series too short to search at all, versus one long enough but with no
        # candidate the neighbours can speak for.
        unrestricted_k, _ = _restricted_max(profile, n, min_segment_months)
        return _null(
            "no_supported_window" if unrestricted_k is not None else "series_too_short"
        )

    rng = np.random.default_rng(seed)
    shuffled = standardized.copy()
    null_max = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(shuffled)
        _, null_max[i] = _restricted_max(
            snht_statistic(shuffled), n, min_segment_months, eligible
        )
    # (1 + count) / (1 + n), not count / n. A permutation test with n shuffles
    # cannot resolve a p-value below 1/(n+1), so a bare mean reports 0.0 for
    # any statistic no shuffle beat -- a certainty 500 draws do not buy. The
    # add-one form is the standard estimator (Davison & Hinkley) and is the
    # conservative direction, which is what this module is tuned for
    # throughout.
    #
    # It matters because p feeds Benjamini-Hochberg: at p = 0 a break passes
    # BH at any rank, whereas the floor must clear i/m * alpha -- which needs
    # rank i >= m / (alpha * (n + 1)), so the flagged fraction has to reach
    # 1 / (alpha * (n + 1)). At alpha = 0.05 and n = 500 that is about 4%.
    # On this archive 600 of 1253 tested series sat at 0.0, far above the
    # pivot, and the flagged set is 481 either way; on a report where fewer
    # than ~4% of series carry a break, the two answers differ.
    p_value = float((1.0 + np.sum(null_max >= observed_t)) / (1.0 + n_permutations))

    before, after = values[:best_k], values[best_k:]
    return BreakpointResult(
        canonical_station_id=canonical_station_id,
        variable=variable,
        n_neighbors_used=n_used,
        n_months_tested=len(reference),
        breakpoint_date=reference.index[best_k],   # first month of the shifted segment
        statistic=observed_t,
        p_value=p_value,
        significant=bool(p_value < alpha and observed_t >= min_statistic),
        segment_before_mean=float(np.mean(before)),
        segment_after_mean=float(np.mean(after)),
        shift=float(np.mean(after) - np.mean(before)),
        status="tested",
        transform=transform,
        min_support_at_break=int(support.iloc[best_k]),
        n_months_min_support=n_supported,
    )


# Columns of the project flag layer (plan section 1.4: "retain provider QC
# flags and layer project flags rather than overwriting them"). One row per
# flagged canonical_station_id x variable.
FLAG_LAYER_COLUMNS: tuple[str, ...] = (
    "canonical_station_id",
    "variable",
    "flag",
    "breakpoint_date",
    "statistic",
    "p_value",
    "shift",
    "shift_ratio",
    "transform",
    "n_neighbors_used",
    "min_support_at_break",
    "n_months_tested",
    "detected_by",
)


def dedupe_flags(layer: pd.DataFrame) -> pd.DataFrame:
    """
    One flag per station and variable, the earliest breakpoint winning.

    Shared by every consumer of the layer. It exists because the two appliers
    used to disagree: the long-format one kept the earliest date and the wide
    one kept whichever row happened to come last in the file, so one archive
    could answer the same question two ways. Which is not hypothetical --
    build_flag_layer's own contract anticipates a second detector adding rows
    to this layer.
    """
    if layer.empty:
        return layer
    out = layer.copy()
    out["breakpoint_date"] = pd.to_datetime(out["breakpoint_date"])
    return (
        out.sort_values("breakpoint_date")
        .drop_duplicates(["canonical_station_id", "variable"], keep="first")
        .reset_index(drop=True)
    )

# Columns apply_flag_layer adds to an observation table. Named distinctly
# from anything a provider writes, so layering can never collide with a
# retained provider QC flag.
FLAG_COLUMNS: tuple[str, ...] = (
    "homogeneity_flag",
    "homogeneity_breakpoint_date",
    "homogeneity_segment",
)


def build_flag_layer(
    report: pd.DataFrame,
    detected_by: str = "snht_relative_monthly",
) -> pd.DataFrame:
    """
    Reduce a run_report result to the project flag layer: one row per
    station/variable that survived the FDR correction, in a form that joins
    onto the consolidated table.

    Only FDR-surviving rows become flags. `significant` alone is the
    uncorrected column kept for reference (see run_report); publishing it as
    an archive-level flag would push the report's own multiple-testing
    problem onto everything downstream.

    `detected_by` names the method, so a later detector (an iterative
    multi-break decomposition, or PHA-style network attribution) can add
    rows to the same layer without its flags being mistaken for these.
    """
    if report.empty or "significant_fdr" not in report.columns:
        return pd.DataFrame(columns=list(FLAG_LAYER_COLUMNS))

    flagged = report[report["significant_fdr"].astype(bool)].copy()
    if flagged.empty:
        return pd.DataFrame(columns=list(FLAG_LAYER_COLUMNS))

    flagged["flag"] = "breakpoint_suspected"
    flagged["detected_by"] = detected_by
    defaults = {"transform": "difference", "shift_ratio": float("nan"),
                "min_support_at_break": 0}
    for column, default in defaults.items():
        if column not in flagged.columns:
            flagged[column] = default
    return dedupe_flags(flagged[list(FLAG_LAYER_COLUMNS)].reset_index(drop=True))


BREAKPOINT_TABLE_COLUMNS: tuple[str, ...] = (
    "canonical_station_id",
    "name",
    "lat",
    "lon",
    "variable",
    "breakpoint_date",
    "transform",
    "shift",
    "shift_ratio",
    "segment_before_mean",
    "segment_after_mean",
    "statistic",
    "p_value",
    "min_support_at_break",
    "n_neighbors_used",
)


def flagged_breakpoint_table(report: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    Every FDR-surviving break with the station's name and position, for a
    person reading the list in a spreadsheet rather than joining it in code.

    A listing only: the flags stay diagnostic, and nothing reads this back.
    Sorted by station and date, so a cluster of same-dated breaks at
    neighbouring stations -- one event seen several times, since this module
    does not attribute a break across the network -- reads as a cluster.
    """
    if report.empty or "significant_fdr" not in report.columns:
        return pd.DataFrame(columns=list(BREAKPOINT_TABLE_COLUMNS))
    flagged = report[report["significant_fdr"].astype(bool)]
    identity = stations[["canonical_station_id", "name", "lat", "lon"]]
    table = flagged.merge(identity, on="canonical_station_id", how="left")
    for column in BREAKPOINT_TABLE_COLUMNS:
        if column not in table.columns:
            table[column] = float("nan")
    return (
        table[list(BREAKPOINT_TABLE_COLUMNS)]
        .sort_values(["canonical_station_id", "variable", "breakpoint_date"])
        .reset_index(drop=True)
    )


def apply_flag_layer(values: pd.DataFrame, layer: pd.DataFrame) -> pd.DataFrame:
    """
    Layer the breakpoint flags onto a long-format observation table, so the
    diagnosis is queryable alongside every other QC flag rather than living
    in a table someone has to know to join by hand.

    Adds FLAG_COLUMNS and changes nothing else -- no value is altered, no
    row is dropped, and no existing column is touched (plan section 1.4:
    layer project flags rather than overwriting them; section 1.6: this
    module diagnoses and never corrects):

      homogeneity_flag            True where this station/variable has a
                                  suspected breakpoint at all.
      homogeneity_breakpoint_date the detected date, repeated on every row
                                  of that station/variable.
      homogeneity_segment         0 for observations before the breakpoint,
                                  1 from the breakpoint onward, <NA> where
                                  unflagged. This is what makes the flag
                                  usable: section 4.6's reference-
                                  sensitivity test needs to hold a segment
                                  out, or compare the two, which needs each
                                  observation to know which side it is on.

    `values` needs canonical_station_id, date and variable; anything else it
    carries is passed through untouched.
    """
    collisions = [c for c in FLAG_COLUMNS if c in values.columns]
    if collisions:
        raise ValueError(
            f"refusing to overwrite existing column(s) {collisions}; "
            "flags are layered, never overwritten"
        )

    out = values.copy()
    if layer.empty:
        out["homogeneity_flag"] = False
        out["homogeneity_breakpoint_date"] = pd.NaT
        out["homogeneity_segment"] = pd.array([pd.NA] * len(out), dtype="Int8")
        return out

    # One flag per station/variable, via the shared rule -- see dedupe_flags.
    keyed = dedupe_flags(layer)[
        ["canonical_station_id", "variable", "breakpoint_date"]
    ].rename(columns={"breakpoint_date": "homogeneity_breakpoint_date"})

    out = out.merge(keyed, on=["canonical_station_id", "variable"], how="left")
    out["homogeneity_flag"] = out["homogeneity_breakpoint_date"].notna()

    dates = pd.to_datetime(out["date"])
    after = dates >= out["homogeneity_breakpoint_date"]
    segment = np.where(after, 1, 0)
    out["homogeneity_segment"] = pd.array(
        np.where(out["homogeneity_flag"], segment, None), dtype="Int8"
    )
    return out


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg false-discovery-rate correction.

    run_report tests many station/variable series at once (thousands, across
    the full archive); at alpha=0.05 with no correction, roughly 5% of them
    are expected to look significant from chance alone even if the test is
    perfectly calibrated -- with 2,000+ tests that is on the order of a
    hundred spurious "breakpoints". BH controls the expected proportion of
    false discoveries among the rows flagged, rather than the per-test
    false-positive rate, which is the standard correction for exactly this
    many-simultaneous-tests situation.

    NaN p-values (series that were never tested -- no usable neighbor, or a
    too-short/too-narrow window) are never flagged.

    Returns a boolean array the same length as p_values.
    """
    p_values = np.asarray(p_values, dtype="float64")
    flagged = np.zeros(p_values.shape, dtype=bool)
    valid = np.where(np.isfinite(p_values))[0]
    if valid.size == 0:
        return flagged

    order = valid[np.argsort(p_values[valid])]
    m = order.size
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = p_values[order] <= thresholds
    if not below.any():
        return flagged
    last_true = np.max(np.where(below)[0])
    flagged[order[: last_true + 1]] = True
    return flagged


def variable_series_by_station(values: pd.DataFrame, variable: str) -> dict[str, pd.Series]:
    """
    From the consolidated (gold) long-format table, one date-indexed Series
    per canonical_station_id for a single variable -- the shape both
    select_neighbors's callers and detect_breakpoint need.
    """
    subset = values[values["variable"] == variable]
    if subset.empty:
        return {}
    result = {}
    for station_id, group in subset.groupby("canonical_station_id"):
        result[station_id] = pd.Series(
            group["value"].to_numpy(),
            index=pd.DatetimeIndex(group["date"]),
        ).sort_index()
    return result


def run_report(
    values: pd.DataFrame,
    stations_meta: pd.DataFrame,
    variables: list[str],
    n_permutations: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Batch driver: test every station with enough data, for every requested
    variable, against its own nearest passing neighbors.

    Parameters
    ----------
    values       : the consolidated long-format table (canonical_station_id,
                   date, variable, value, ...).
    stations_meta: canonical_station_id, lat, lon (e.g. canonical_stations.csv).
    variables    : which variables to test (e.g. config.SOURCE_PRIORITY keys).

    Returns
    -------
    One row per station x variable reached (its `status` column says whether
    a test actually ran; stations too short to produce even the minimum
    number of monthly values, or with no neighbor at all, are skipped
    outright rather than reported as a spurious non-result), plus a
    `significant_fdr` column: `significant` re-evaluated with a
    Benjamini-Hochberg false-discovery-rate correction across every row in
    this report (one family -- see benjamini_hochberg). With thousands of
    station/variable tests in one report, `significant_fdr` is the column to
    read; the uncorrected `significant` column is kept for reference.
    """
    meta_by_id = stations_meta.set_index("canonical_station_id")[["lat", "lon"]]
    rows: list[dict] = []
    # Cheapest possible pre-filter: a series that cannot yield 3*MIN_SEGMENT_MONTHS
    # usable months can never produce a non-degenerate search window.
    min_daily_observations = 3 * MIN_SEGMENT_MONTHS * MIN_DAYS_PER_MONTH

    for variable in variables:
        series_by_station = variable_series_by_station(values, variable)
        # Only stations that actually carry this variable are candidate
        # neighbours. Selecting the nearest few from ALL canonical stations and
        # filtering afterwards discarded usable neighbours that sat just beyond
        # the ones without data -- measured on the real archive, 84 of 126
        # otherwise-testable gust stations, 112 for tas and 52 for hurs, each
        # reported as no_usable_neighbor or too_few_neighbors rather than
        # tested.
        candidates = stations_meta[
            stations_meta["canonical_station_id"].isin(series_by_station)
        ]
        for station_id, target in series_by_station.items():
            if len(target) < min_daily_observations or station_id not in meta_by_id.index:
                continue
            lat, lon = meta_by_id.loc[station_id, ["lat", "lon"]]
            neighbor_ids = select_neighbors(station_id, lat, lon, candidates)
            neighbors = {
                nid: series_by_station[nid]
                for nid in neighbor_ids
                if nid in series_by_station
            }
            if not neighbors:
                continue
            result = detect_breakpoint(
                station_id, variable, target, neighbors,
                n_permutations=n_permutations, alpha=alpha, seed=seed,
            )
            rows.append(result.to_dict())

    report = pd.DataFrame(rows)
    if not report.empty:
        # FDR narrows what `significant` already allows -- it never widens it.
        # Applying the correction to p-values alone would ignore the SNHT
        # floor and the two-neighbor gate baked into `significant`, and could
        # then flag MORE rows than the uncorrected column, which is nonsense.
        survives_fdr = benjamini_hochberg(report["p_value"].to_numpy(), alpha)
        report["significant_fdr"] = survives_fdr & report["significant"].to_numpy()
    return report
