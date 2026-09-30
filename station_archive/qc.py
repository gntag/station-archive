"""
qc.py
=====
Project quality control, layered over provider flags (plan section 1.4).

THE CONTRACT
------------
"Store provider values and flags unchanged in a raw/bronze layer; perform
conversions and project QC in a standardized/silver layer.  The
modelling/gold layer should contain only accepted values plus transparent
rejection reasons."

So nothing here edits bronze.  A rejection removes a value from the
consolidated table and is counted, by station, variable and reason, in a
rejection summary -- the "transparent rejection reasons" half of the contract.

"retain provider QC flags and layer project flags rather than overwriting
them": provider flags decide nothing on their own here.  Where a provider
flag is used, it is because the provider's own documentation says the value
is not an observation.

THE GSOD PRECIPITATION PROBLEM
------------------------------
This is the rule that matters most in this archive, and it is quoted from
NOAA's own readme rather than inferred:

    "'0' indicates no measurable precipitation (includes a trace)"
    "I = Station did not report any precipitation data for the day and did
     not report any occurrences of precipitation in its hourly observations"
    "H = Station reported '0' as the amount for the day ... but also reported
     at least one occurrence of precipitation in hourly observations.  This
     could indicate a trace occurred, but should be considered as incomplete
     data for the day."

Measured on the downloaded archive: of 85,968 GSOD rows recording 0.00 in a
400-file sample, 39,361 -- 45.8% -- carry flag I.  Those are not measured
zeros; they are days the station reported nothing.  Reading them as 0.0 mm
inflates dry-day frequency, and the existing pipeline
(downscaling/station_ingest.py) does exactly that: it drops whole
precipitation-inactive stations but converts these per-day false zeros
straight through.

Flag H is rejected on the provider's own words -- "should be considered as
incomplete data for the day".

Section 1.4 also requires that trace precipitation be preserved separately
from zero.  GSOD cannot satisfy that at all: its readme states plainly that
a reported zero includes a trace, and the information is not recoverable
from the file.  That is recorded as a known limitation of the source rather
than papered over, and is why GSOD ranks last for pr in SOURCE_PRIORITY.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

# GSOD PRCP_ATTRIBUTES values whose own documentation says the day is not a
# measurement.  Every other letter (A-G) describes which hours the total
# covers and is a genuine, if partial, observation.
GSOD_PRCP_NON_OBSERVATION_FLAGS = {
    "I": "GSOD flag I: station reported no precipitation data for the day",
    "H": "GSOD flag H: provider documents this as incomplete data for the day",
}

# Physically impossible values, used only to catch gross failures.  These are
# deliberately wide: QC here removes the impossible, not the merely unusual,
# because an aggressive range filter would silently delete the extremes a
# downscaling model and a bias-adjustment reference most need.
GROSS_RANGE = {
    "tas": (-60.0, 60.0),
    "tasmin": (-70.0, 55.0),
    "tasmax": (-60.0, 60.0),
    "pr": (0.0, 2000.0),
    "hurs": (0.0, 100.0),
    "sfcWind": (0.0, 120.0),
    "gust": (0.0, 150.0),
}


@dataclass(frozen=True)
class Rejection:
    """One rejected value, identified well enough to audit."""

    source: str
    source_station_id: str
    variable: str
    reason: str
    n_values: int


def reject_reason(
    source: str,
    variable: str,
    value: float,
    provider_flags: dict[str, Any],
) -> Optional[str]:
    """
    Why this value must not enter the consolidated table, or None to accept.

    Value is in the PROVIDER's units, because this runs at parse time; only
    flag-based and sign-based rules are applied here.  Unit-dependent range
    checks run later, in canonical units, via check_gross_range.
    """
    if source == "gsod" and variable == "pr":
        attributes = str(provider_flags.get("attributes", "") or "").strip().upper()
        reason = GSOD_PRCP_NON_OBSERVATION_FLAGS.get(attributes)
        if reason is not None:
            return reason

    # A negative accumulation is not a small amount; it is a broken record.
    if variable == "pr" and value < 0:
        return "negative precipitation"

    if variable in ("sfcWind", "gust") and value < 0:
        return "negative wind speed"

    return None


def gated_variables() -> frozenset[str]:
    """
    Variables this module can range-check.

    Read by Policy.__post_init__, which refuses to select a variable absent
    from here: a selected variable with no bounds would reach the gold table
    with no range check at all, and the undocumented-sentinel case below is
    exactly what that check is for.
    """
    return frozenset(GROSS_RANGE)


def check_gross_range(values: pd.Series, variable: str) -> pd.Series:
    """
    Boolean mask of values inside the physically possible range.

    Applied in canonical units.  NaN is treated as out of range so it cannot
    reach the gold layer as an accepted value -- and that holds whether or not
    the variable has declared bounds.  The unbounded branch used to return an
    all-True mask, which admitted NaN and so contradicted the sentence above
    for the three variables the silver layer carries without bounds
    (dew_point, tavg_unspecified, wind_max_sustained).

    An unbounded variable is still not range-JUDGED -- nothing here invents a
    plausible range for it.  Policy.__post_init__ refuses to select one, so
    the two rules together mean a value reaching the gold table has always been
    checked against bounds someone wrote down.
    """
    finite = values.notna()
    bounds = GROSS_RANGE.get(variable)
    if bounds is None:
        return finite
    low, high = bounds
    return values.between(low, high) & finite


def check_internal_consistency(
    tasmin: pd.Series,
    tas: pd.Series,
    tasmax: pd.Series,
) -> pd.DataFrame:
    """
    Days violating `tasmin <= tas <= tasmax` (plan section 1.4).

    Returns one row per violating day with the three values, so the failure
    can be attributed rather than merely counted.  This module reports the
    violation; deciding which of the three values to drop is a policy
    question that has not been settled, so nothing is removed here.

    Comparisons run only on days where the relevant pair is present -- a
    station reporting tasmax but not tas is not inconsistent.
    """
    frame = pd.DataFrame({"tasmin": tasmin, "tas": tas, "tasmax": tasmax})

    min_over_max = frame["tasmin"] > frame["tasmax"]
    mean_below_min = frame["tas"] < frame["tasmin"]
    mean_above_max = frame["tas"] > frame["tasmax"]

    violations = frame[min_over_max | mean_below_min | mean_above_max].copy()
    if violations.empty:
        return pd.DataFrame(columns=["tasmin", "tas", "tasmax", "violation"])

    def describe(row: pd.Series) -> str:
        problems = []
        if pd.notna(row["tasmin"]) and pd.notna(row["tasmax"]) and row["tasmin"] > row["tasmax"]:
            problems.append("tasmin > tasmax")
        if pd.notna(row["tas"]) and pd.notna(row["tasmin"]) and row["tas"] < row["tasmin"]:
            problems.append("tas < tasmin")
        if pd.notna(row["tas"]) and pd.notna(row["tasmax"]) and row["tas"] > row["tasmax"]:
            problems.append("tas > tasmax")
        return "; ".join(problems)

    violations["violation"] = violations.apply(describe, axis=1)
    return violations[violations["violation"] != ""]


def check_repeated_days(values: pd.Series, min_run: int = 10) -> pd.Series:
    """
    Runs of an identical value lasting at least `min_run` days.

    Section 1.4 asks for "repeated days, impossible persistence".  A stuck
    sensor reports the same number indefinitely; genuine weather rarely
    does for temperature, though a dry spell legitimately repeats zero
    precipitation, so callers must not apply this to pr without thought.

    Returns a boolean mask of days belonging to such a run.
    """
    if values.empty:
        return pd.Series(dtype=bool, index=values.index)
    # Each change of value starts a new group; groups at least min_run long
    # are flagged in full.
    groups = (values != values.shift()).cumsum()
    sizes = values.groupby(groups).transform("size")
    return sizes >= min_run


def coverage_summary(values: pd.DataFrame) -> pd.DataFrame:
    """
    Completeness by station, variable, year and month (plan section 1.4).

    "create coverage and missingness summaries by station, variable, month
    and year".  Reported as observed-day counts against days in the month, so
    a partially observed month is visible rather than averaged away.
    """
    if values.empty:
        return pd.DataFrame()

    working = values.copy()
    working["date"] = pd.to_datetime(working["date"])
    working["year"] = working["date"].dt.year
    working["month"] = working["date"].dt.month

    summary = (
        working.groupby(["canonical_station_id", "variable", "year", "month"])
        .size()
        .reset_index(name="observed_days")
    )
    summary["days_in_month"] = [
        pd.Period(year=int(y), month=int(m), freq="M").days_in_month
        for y, m in zip(summary["year"], summary["month"])
    ]
    summary["missing_fraction"] = (
        1.0 - summary["observed_days"] / summary["days_in_month"]
    ).clip(lower=0.0).round(4)
    return summary
