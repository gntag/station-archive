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

from observations.identity import haversine_km

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
MIN_DAYS_PER_MONTH = 20

# Minimum overlapping months between target and a candidate neighbor before
# the neighbor is trusted to contribute to the reference series.
MIN_NEIGHBOR_OVERLAP_MONTHS = 24

# Minimum Pearson correlation on the shared overlap, measured on FIRST
# DIFFERENCES of the deseasonalised anomalies (see build_reference_difference
# for why differences rather than levels), before a candidate neighbor is
# considered to be tracking the same weather rather than a different
# micro-climate that would inject its own noise into the difference series.
MIN_NEIGHBOR_CORRELATION = 0.6

# Neighbors are searched within this radius before concluding none exist.
NEIGHBOR_SEARCH_RADIUS_KM = 100.0

# Cap on how many passing neighbors are used. More neighbors stabilise the
# reference series, but each one must still individually pass the overlap
# and correlation gates above -- this only bounds how many are tried.
MAX_NEIGHBORS = 5

# A break cannot be attributed with a single neighbor: target-minus-neighbor
# shows the same step whether the target moved or the neighbor did. PHA
# (NOAA's algorithm for GHCN itself) resolves attribution across the whole
# network before adjusting anything; the cheap equivalent of that here is to
# refuse to call anything significant on one neighbor's word.
MIN_NEIGHBORS = 2

# Minimum length, in months, either side of a candidate breakpoint must have
# before that split is even considered. Without this, a candidate split at
# the very edge of the record lets that short segment's mean be dominated by
# one or two extreme values, producing a spuriously huge SNHT statistic that
# reflects outliers, not a sustained shift. One full annual cycle is the
# defensible minimum: shorter than that, a segment mean is not meaningfully
# comparable to the rest of the series at all, breakpoint or not.
MIN_SEGMENT_MONTHS = 12

# Conservative floor on the SNHT statistic itself, on top of the permutation
# p-value. Climatol applies a threshold of SNHT = 20 "in order to avoid
# false positives" rather than relying on nominal significance alone, and
# the published method comparisons show why: the fraction of series judged
# homogeneous swings from 85% (HOMER) to 22% (ACMANT) on identical data,
# almost entirely through how conservative each method's detection is. This
# report is a diagnostic flag, so it errs towards missing a marginal break
# rather than flagging a station that is fine.
MIN_SNHT_STATISTIC = 20.0

# Variables aggregated to a monthly TOTAL rather than a monthly mean.
SUM_VARIABLES = frozenset({"pr"})


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
      series_too_short    -- too few months for a non-degenerate search
      no_variance         -- the difference series is constant

    `shift` is in the variable's monthly units: degC (or %, m/s) for a
    monthly mean, mm/month for precipitation's monthly total.
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
        }


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
    """
    if series.empty:
        return pd.Series(dtype="float64")

    grouped = series.groupby(series.index.to_period("M"))
    counts = grouped.size()
    monthly = grouped.sum() if variable in SUM_VARIABLES else grouped.mean()
    monthly = monthly[counts >= min_days]
    if monthly.empty:
        return pd.Series(dtype="float64")

    monthly.index = monthly.index.to_timestamp()
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

    Each neighbor is gated independently before contributing, rather than
    averaging raw neighbor values first, so one bad or unrelated neighbor
    cannot poison the reference the way it could if it were blended in
    before any check ran.

    Returns (difference_series, n_neighbors_used). An empty series and 0
    means no candidate neighbor was usable -- the caller must not attempt a
    breakpoint test without a reference.
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

    if not differences:
        return pd.Series(dtype="float64"), 0

    combined = pd.concat(differences, axis=1)
    return combined.mean(axis=1).dropna().sort_index(), len(differences)


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


def _restricted_max(profile: np.ndarray, n: int, min_segment: int) -> tuple[Optional[int], float]:
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
    monthly_target = to_monthly_anomalies(target, variable)
    monthly_neighbors = {
        name: to_monthly_anomalies(series, variable)
        for name, series in neighbors.items()
    }
    reference, n_used = build_reference_difference(monthly_target, monthly_neighbors)

    def _null(status: str) -> BreakpointResult:
        return BreakpointResult(
            canonical_station_id, variable, n_used, len(reference),
            None, float("nan"), float("nan"), False,
            float("nan"), float("nan"), float("nan"), status,
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
            None, 0.0, 1.0, False, mean, mean, 0.0, "no_variance",
        )
    standardized = (values - mean) / std
    n = len(standardized)

    profile = snht_statistic(standardized)
    best_k, observed_t = _restricted_max(profile, n, min_segment_months)
    if best_k is None:
        return _null("series_too_short")

    rng = np.random.default_rng(seed)
    shuffled = standardized.copy()
    null_max = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(shuffled)
        _, null_max[i] = _restricted_max(snht_statistic(shuffled), n, min_segment_months)
    p_value = float(np.mean(null_max >= observed_t))

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
    )


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
        for station_id, target in series_by_station.items():
            if len(target) < min_daily_observations or station_id not in meta_by_id.index:
                continue
            lat, lon = meta_by_id.loc[station_id, ["lat", "lon"]]
            neighbor_ids = select_neighbors(station_id, lat, lon, stations_meta)
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
