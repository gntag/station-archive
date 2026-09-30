"""
consolidate.py
==============
Deterministic source selection and gap filling (plan section 1.6).

WHAT THIS MODULE DECIDES
------------------------
For each `canonical_station_id x date x variable`, which single value becomes
the canonical observation, and why.  Section 1.6 sets the procedure:

  1. collect only values that pass mandatory provider and project QC;
  2. sort candidates by configured source rank, then by declared tie-breakers;
  3. choose the highest-ranked accepted value and record selected_source,
     source_rank, selection_reason and raw-record identifier;
  4. if that value is missing or rejected, move down the priority list and
     label the accepted record as a fallback;
  5. retain all non-selected candidates in an audit table;
  6. leave the canonical value missing when no candidate passes QC.

THE TWO MODES THAT MUST NOT BE CONFLATED
----------------------------------------
Section 1.6 is explicit that gap filling has two distinct meanings:

  * **Same-station fallback** fills a missing day from a lower-priority alias
    of the *same physical station*.  This module does that, and only after
    the compatibility test below passes.
  * **Network expansion** admits a *different* physical station because it
    fills geographic or temporal coverage.  That is not gap filling at all;
    such a station keeps its own canonical identity and fills nobody's row.
    Nothing here ever writes one station's value into another's series.

WHY THE COMPATIBILITY TEST EXISTS
---------------------------------
"Before allowing a source switch, compare overlapping records by month for
bias, dispersion, wet-day frequency and discontinuities.  A failed
compatibility test leaves a gap or creates a time-bounded station segment; it
does not justify blind splicing."

This is not hypothetical here.  Sources disagree by station: for tas with an
ECA&D primary and a GSOD fallback, the measured overlap bias runs from -1.60
to +0.58 degC across 14 station pairs.  Splicing GSOD days into an ECA&D
series without testing would inject a step change into a series later used to
fit a downscaling model and to define a bias-adjustment reference
distribution.

Note the spread, and that it straddles zero.  It is why this module tests
every switch at the station it is about to make it at, and why no
archive-wide offset is applied to stand in for that test: see
standardize.GSOD_DAY_WINDOW_BIAS_CORRECTION for a withdrawn correction that
took one source pair's median for one variable and applied it to a different
variable of a different pair.

WHAT THIS MODULE MUST NEVER DO
------------------------------
Interpolate.  Section 1.6: "Do not interpolate missing station observations
in Step 1."  A day with no accepted candidate stays missing.  It also never
averages two sources (section 1.3), and never rewrites bronze.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from station_archive import qc, standardize
from station_archive.policy import DEFAULT_POLICY, Policy

# Columns of the consolidated (gold) table.
CONSOLIDATED_COLUMNS: tuple[str, ...] = (
    "canonical_station_id",
    "date",
    "variable",
    "value",
    "units",
    "selected_source",
    "source_station_id",
    "source_rank",
    "selection_reason",
    "is_fallback",
    "source_record_id",
)

# Columns of the non-selected-candidates audit table (plan section 1.6 point
# 5: "retain all non-selected candidates in an audit table rather than
# deleting or averaging them"). One row per candidate value that existed
# for a canonical_station_id x date x variable but was not chosen as the
# canonical value -- either because a higher-priority source already
# covered that date, or because this source was tested as a fallback for a
# genuine gap and check_compatibility refused it.
NON_SELECTED_COLUMNS: tuple[str, ...] = (
    "canonical_station_id",
    "date",
    "variable",
    "source",
    "source_station_id",
    "value",
    "units",
    "source_rank",
    "rejection_reason",
)


@dataclass(frozen=True)
class CompatibilityResult:
    """
    Whether a lower-priority alias may be spliced into a primary series.

    Every statistic section 1.6 names is reported whether or not the test
    passes, so a refusal can be reviewed rather than merely obeyed.
    """

    compatible: bool
    n_overlap: int
    bias: float
    dispersion_ratio: float
    wet_day_diff: float
    max_monthly_departure: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "n_overlap": self.n_overlap,
            "bias": _round(self.bias, 4),
            "dispersion_ratio": _round(self.dispersion_ratio, 4),
            "wet_day_diff": _round(self.wet_day_diff, 4),
            "max_monthly_departure": _round(self.max_monthly_departure, 4),
            "reason": self.reason,
        }


def _round(value: float, digits: int) -> float:
    return value if value is None or math.isnan(value) else round(value, digits)


def wet_day_fraction(
    values: np.ndarray,
    threshold_mm: Optional[float] = None,
) -> float:
    """
    Fraction of days at or above the wet-day threshold.

    Tested separately from bias because a source can match another's mean
    precipitation while disagreeing about how often it rains -- which is
    exactly what GSOD's "'0' ... includes a trace" convention produces, and
    what a downscaling model trained on wet-day frequency would inherit.
    """
    if threshold_mm is None:
        threshold_mm = DEFAULT_POLICY.fallback.wet_day_threshold_mm
    if values.size == 0:
        return float("nan")
    return float(np.mean(values >= threshold_mm))


def check_compatibility(
    primary: pd.Series,
    fallback: pd.Series,
    variable: str,
    policy: Policy = DEFAULT_POLICY,
) -> CompatibilityResult:
    """
    Compare two same-station series on their shared days.

    Both series must already be in canonical units.  The four checks are the
    ones section 1.6 names: bias, dispersion, wet-day frequency and
    discontinuities, the last measured as the largest monthly departure from
    the overall bias -- two records that agree on average only because a warm
    half cancels a cold half are not interchangeable.
    """
    limits = policy.fallback
    unit = standardize.canonical_unit_for(variable) or ""
    shared = primary.index.intersection(fallback.index)

    if len(shared) < limits.min_overlap_days:
        return CompatibilityResult(
            compatible=False,
            n_overlap=len(shared),
            bias=float("nan"),
            dispersion_ratio=float("nan"),
            wet_day_diff=float("nan"),
            max_monthly_departure=float("nan"),
            reason=(
                f"only {len(shared)} overlapping days, below the "
                f"{limits.min_overlap_days}-day minimum; "
                f"compatibility cannot be established"
            ),
        )

    a = primary.loc[shared].to_numpy(dtype="float64")
    b = fallback.loc[shared].to_numpy(dtype="float64")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    index = shared[finite]

    if a.size < limits.min_overlap_days:
        return CompatibilityResult(
            False, int(a.size), float("nan"), float("nan"), float("nan"),
            float("nan"),
            f"only {a.size} finite overlapping days, below the "
            f"{limits.min_overlap_days}-day minimum",
        )

    difference = b - a
    bias = float(np.mean(difference))

    std_primary = float(np.std(a))
    dispersion = float(np.std(b) / std_primary) if std_primary > 0 else float("nan")

    if variable == "pr":
        wet_diff = abs(
            wet_day_fraction(b, limits.wet_day_threshold_mm)
            - wet_day_fraction(a, limits.wet_day_threshold_mm)
        )
    else:
        wet_diff = float("nan")

    # Only well-populated months contribute a monthly mean. 'difference' is
    # already finite-filtered above, so a group's size is its value count.
    by_month = pd.Series(difference, index=index).groupby([index.year, index.month])
    populated = by_month.size() >= limits.min_days_per_month
    monthly = by_month.mean()[populated]

    if monthly.empty:
        # Nothing to test the discontinuity against. Scoring this 0.0 and
        # passing would let an overlap that clears min_overlap_days purely by
        # being spread thin skip the gate entirely -- section 1.6 leaves the
        # gap open on a failed test, and a test that could not run has not
        # been passed.
        return CompatibilityResult(
            False, int(a.size), bias, dispersion, wet_diff, float("nan"),
            f"no calendar month has {limits.min_days_per_month} shared days, so "
            f"the monthly discontinuity test cannot be run; compatibility "
            f"cannot be established",
        )

    departure = float(np.max(np.abs(monthly.to_numpy() - bias)))

    bias_limit = limits.max_bias.get(unit)
    departure_limit = limits.max_monthly_departure.get(unit)
    low, high = limits.dispersion_range

    failures = []
    if bias_limit is not None and abs(bias) > bias_limit:
        failures.append(f"bias {bias:+.2f} exceeds +/-{bias_limit}")
    if not math.isnan(dispersion) and not (low <= dispersion <= high):
        failures.append(f"dispersion ratio {dispersion:.2f} outside {low}-{high}")
    if not math.isnan(wet_diff) and wet_diff > limits.max_wet_day_diff:
        failures.append(
            f"wet-day frequency differs by {wet_diff:.1%}, above "
            f"{limits.max_wet_day_diff:.0%}"
        )
    if departure_limit is not None and departure > departure_limit:
        failures.append(
            f"largest monthly departure {departure:.2f} exceeds {departure_limit}"
        )

    if failures:
        return CompatibilityResult(
            False, int(a.size), bias, dispersion, wet_diff, departure,
            "; ".join(failures),
        )

    return CompatibilityResult(
        True, int(a.size), bias, dispersion, wet_diff, departure,
        f"compatible over {a.size} days (bias {bias:+.2f}, "
        f"dispersion {dispersion:.2f})",
    )


@dataclass
class StationConsolidation:
    """Consolidated values for one canonical station and variable."""

    values: pd.DataFrame
    compatibility: list[dict] = field(default_factory=list)
    non_selected: pd.DataFrame = field(default_factory=lambda: _empty_non_selected())


def _canonical_series(
    source: str,
    station_id: str,
    variable: str,
    series_loader,
) -> pd.Series:
    """
    One alias's daily series, converted to canonical units.

    Conversion happens here rather than in the loader because bronze is
    defined as provider values in provider units (section 1.4); the silver
    layer is where they become comparable.
    """
    from station_archive.identity import provider_units

    raw = series_loader.series(source, station_id, variable)
    if raw.empty:
        return raw
    units = provider_units(source, variable)
    if not units:
        return pd.Series(dtype="float64")

    # Vectorised, and bit-identical to the per-value loop it replaces: a full
    # consolidation converts ~9.5 million values.
    values = standardize.to_canonical_series(raw, units)
    values = standardize.apply_day_window_correction(values, source, variable)

    # Gross-range QC, applied here because it needs canonical units.
    #
    # Not hypothetical: GHCN-D was found carrying 999.0 degC temperatures at
    # Italian stations -- an undocumented missing-value sentinel that is not
    # the -9999 the format documents, so the adapter's sentinel filter does
    # not catch it.  Left in, a single such value destroys any statistic
    # computed over the series: it dragged one station's standard deviation
    # from 6.8 to 46.1 degC and its correlation against the same record in
    # another archive from ~1.0 to 0.14.
    return values[qc.check_gross_range(values, variable)]


def consolidate_station_variable(
    canonical_station_id: str,
    aliases: pd.DataFrame,
    variable: str,
    series_loader,
    priority: Optional[Sequence[str]] = None,
    policy: Policy = DEFAULT_POLICY,
) -> StationConsolidation:
    """
    Select one value per day for one canonical station and variable.

    The highest-ranked source supplies the primary series outright.  Every
    lower-ranked source is a candidate only for days the primary lacks, and
    only after check_compatibility accepts it -- an incompatible alias leaves
    the gap open rather than filling it, per section 1.6.

    Sources absent from the configured priority never contribute, which is
    how c3s_land stays an audit cross-check rather than becoming a silent
    fallback for the GHCN-D records it duplicates.
    """
    ranking = list(
        priority if priority is not None else policy.priority_for(variable)
    )
    units = standardize.canonical_unit_for(variable) or ""

    available: list[tuple[int, str, str, pd.Series]] = []
    for rank, source in enumerate(ranking, start=1):
        for row in aliases[aliases["source"] == source].itertuples(index=False):
            series = _canonical_series(
                source, str(row.source_station_id), variable, series_loader
            )
            if not series.empty:
                available.append((rank, source, str(row.source_station_id), series))

    if not available:
        return StationConsolidation(values=_empty_consolidation())

    non_selected_frames: list[pd.DataFrame] = []

    available.sort(key=lambda item: (item[0], item[1], item[2]))
    primary_rank, primary_source, primary_id, primary_series = available[0]

    frame = pd.DataFrame(
        {
            "canonical_station_id": canonical_station_id,
            "date": primary_series.index,
            "variable": variable,
            "value": primary_series.to_numpy(),
            "units": units,
            "selected_source": primary_source,
            "source_station_id": primary_id,
            "source_rank": primary_rank,
            "selection_reason": f"highest-ranked source with data (rank {primary_rank})",
            "is_fallback": False,
            "source_record_id": [
                f"{primary_source}:{primary_id}:{d.date().isoformat()}:{variable}"
                for d in primary_series.index
            ],
        }
    )

    compatibility: list[dict] = []
    covered = set(primary_series.index)

    for rank, source, station_id, series in available[1:]:
        gaps = series.index.difference(pd.Index(sorted(covered)))

        # Every day this source already lost to a higher-priority selection
        # (primary, or an earlier, higher-ranked fallback) is a non-selected
        # candidate: the value existed, it just wasn't the one chosen.
        outranked = series.index.difference(gaps)
        if len(outranked) > 0:
            non_selected_frames.append(
                _non_selected_frame(
                    canonical_station_id, variable, source, station_id,
                    series.loc[outranked], units, rank,
                    "outranked: date already covered by a higher-priority source",
                )
            )

        record = {
            "canonical_station_id": canonical_station_id,
            "variable": variable,
            "primary": f"{primary_source}:{primary_id}",
            "fallback": f"{source}:{station_id}",
            "fallback_rank": rank,
            "candidate_days": len(gaps),
        }

        if len(gaps) == 0:
            record.update(
                {
                    **CompatibilityResult(
                        False, 0, float("nan"), float("nan"), float("nan"),
                        float("nan"), "no gaps to fill",
                    ).to_dict(),
                    "days_filled": 0,
                }
            )
            compatibility.append(record)
            continue

        result = check_compatibility(primary_series, series, variable, policy)
        record.update(result.to_dict())

        if not result.compatible:
            # Section 1.6: a failed compatibility test "leaves a gap ...
            # it does not justify blind splicing". The candidate values
            # still existed, so they go to the audit table rather than
            # vanishing.
            non_selected_frames.append(
                _non_selected_frame(
                    canonical_station_id, variable, source, station_id,
                    series.loc[gaps], units, rank,
                    f"incompatible fallback: {result.reason}",
                )
            )
            record["days_filled"] = 0
            compatibility.append(record)
            continue

        filled = series.loc[gaps]
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "canonical_station_id": canonical_station_id,
                        "date": filled.index,
                        "variable": variable,
                        "value": filled.to_numpy(),
                        "units": units,
                        "selected_source": source,
                        "source_station_id": station_id,
                        "source_rank": rank,
                        "selection_reason": (
                            f"fallback from rank {rank}; {result.reason}"
                        ),
                        "is_fallback": True,
                        "source_record_id": [
                            f"{source}:{station_id}:{d.date().isoformat()}:{variable}"
                            for d in filled.index
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
        covered |= set(filled.index)
        record["days_filled"] = len(filled)
        compatibility.append(record)

    frame = frame.sort_values("date").reset_index(drop=True)
    non_selected = (
        pd.concat(non_selected_frames, ignore_index=True)
        if non_selected_frames
        else _empty_non_selected()
    )
    return StationConsolidation(values=frame, compatibility=compatibility, non_selected=non_selected)


def _non_selected_frame(
    canonical_station_id: str,
    variable: str,
    source: str,
    source_station_id: str,
    series: pd.Series,
    units: str,
    source_rank: int,
    rejection_reason: str,
) -> pd.DataFrame:
    """One row per (source, date) candidate value that was not selected."""
    return pd.DataFrame(
        {
            "canonical_station_id": canonical_station_id,
            "date": series.index,
            "variable": variable,
            "source": source,
            "source_station_id": source_station_id,
            "value": series.to_numpy(),
            "units": units,
            "source_rank": source_rank,
            "rejection_reason": rejection_reason,
        }
    )


def _empty_consolidation() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CONSOLIDATED_COLUMNS))


def _empty_non_selected() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NON_SELECTED_COLUMNS))


def merge_source_contributions(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Sum per-part contribution tables into one.

    The audit is computed part by part and summed here, rather than by
    concatenating every part back into one frame. That concatenation is what
    the buffered writer exists to avoid: the gold table runs to ~9.6 million
    rows with string columns, so rebuilding it in memory for a group-by
    reinstated exactly the cost the streaming write had just paid to avoid.
    Each part's aggregate is a few hundred rows, so the sum is free.
    """
    frames = [p for p in parts if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["variable", "year", "selected_source", "is_fallback"], as_index=False)[
            "days"
        ]
        .sum()
        .sort_values(["variable", "year", "selected_source"])
        .reset_index(drop=True)
    )


def source_contribution(values: pd.DataFrame) -> pd.DataFrame:
    """
    Source contribution by variable and year, for the section 1.6 audit.

    Reported as counts of selected days rather than percentages, so a year
    with little data cannot look like full coverage.
    """
    if values.empty:
        return pd.DataFrame()
    working = values.copy()
    working["year"] = pd.to_datetime(working["date"]).dt.year
    return (
        working.groupby(["variable", "year", "selected_source", "is_fallback"])
        .size()
        .reset_index(name="days")
        .sort_values(["variable", "year", "selected_source"])
        .reset_index(drop=True)
    )
