"""
standardize.py
==============
Unit and variable harmonization for the silver layer (plan section 1.4).

Plan section 1.4: "Store provider values and flags unchanged in a raw/bronze
layer; perform conversions and project QC in a standardized/silver layer" and
"normalize units without destroying original units or precision".

SCOPE OF THIS MODULE SO FAR
---------------------------
Only the conversion primitives and the provider-element -> canonical-variable
mapping are implemented.  Full silver-layer materialization, the QC contract
and the trace/missing policy are the next stage of Step 1.

They exist now because identity.py needs them: Pearson correlation is
invariant under affine transforms, so series agreement could be tested in raw
provider units -- but the mean-absolute-difference and bias statistics that
plan section 1.6 requires as "pairwise overlap statistics" cannot.  Writing
the conversions here rather than inside identity.py keeps one source of truth
for what a degree, a millimetre and a metre per second are.

UNITS
-----
Canonical units follow the existing pipeline (downscaling/station_ingest.py):
temperature degC, precipitation mm, wind m/s, relative humidity percent.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

# Canonical unit per variable.  Anything converted by this module ends up here.
CANONICAL_UNITS: dict[str, str] = {
    "tas": "degC",
    "tasmin": "degC",
    "tasmax": "degC",
    "pr": "mm",
    "hurs": "percent",
    "sfcWind": "m s-1",
    "gust": "m s-1",
    # Not canonical targets, but carried through the silver layer so a later
    # decision can use them.  See the notes on each below.
    "dew_point": "degC",
    "tavg_unspecified": "degC",
    "wind_max_sustained": "m s-1",
}

# Exact conversion constants.  Spelled out rather than inlined so a reader can
# check them against the provider documentation without reading arithmetic.
_FAHRENHEIT_OFFSET = 32.0
_FAHRENHEIT_SCALE = 5.0 / 9.0
_INCH_TO_MM = 25.4
_KNOT_TO_M_S = 0.514444          # 1 international knot = 1852 m / 3600 s
_KELVIN_OFFSET = 273.15
_TENTHS = 0.1


def fahrenheit_to_celsius(value: float) -> float:
    return (value - _FAHRENHEIT_OFFSET) * _FAHRENHEIT_SCALE


def kelvin_to_celsius(value: float) -> float:
    return value - _KELVIN_OFFSET


def inch_to_mm(value: float) -> float:
    return value * _INCH_TO_MM


def knot_to_m_s(value: float) -> float:
    return value * _KNOT_TO_M_S


def tenths(value: float) -> float:
    """
    Scale a provider's tenths-of-a-unit integer to whole units.

    GHCN-D and ECA&D both store temperature in tenths of a degree, ECA&D wind
    in 0.1 m/s and precipitation in tenths of a millimetre.  The scale factor
    is identical; only the resulting unit differs.
    """
    return value * _TENTHS


# Provider unit string -> conversion to the canonical unit.  Keyed by the
# 'units' field that each adapter records on every RawObservation, so a
# provider that changes its units surfaces here as a missing key rather than
# as silently wrong numbers.
_CONVERSIONS = {
    "degC_tenths": tenths,
    "mm_tenths": tenths,
    "m_s_tenths": tenths,
    "degF": fahrenheit_to_celsius,
    "inch": inch_to_mm,
    "knot": knot_to_m_s,
    "K": kelvin_to_celsius,
    # Already canonical; listed explicitly so an unknown unit still raises.
    "degC": lambda v: v,
    "mm": lambda v: v,
    "percent": lambda v: v,
    "m s-1": lambda v: v,
    "m/s": lambda v: v,
}


class UnknownUnitError(ValueError):
    """
    Raised when a provider unit has no declared conversion.

    Plan section 1.5 wants an upstream change to "stop loudly rather than
    silently dropping columns"; the same applies to units.  Guessing that an
    unrecognised unit is already canonical would silently corrupt every value
    carrying it.
    """


def to_canonical(value: float, provider_units: str) -> float:
    """
    Convert one provider value to its canonical unit.

    The original value and unit are never modified in place -- bronze keeps
    them, and the silver layer records 'source_units' alongside the converted
    number, per section 1.4's "without destroying original units or precision".
    """
    conversion = _CONVERSIONS.get(provider_units)
    if conversion is None:
        raise UnknownUnitError(
            f"no conversion declared for provider unit {provider_units!r}; "
            f"known units are {sorted(_CONVERSIONS)}"
        )
    return conversion(value)


def canonical_unit_for(variable: str) -> Optional[str]:
    return CANONICAL_UNITS.get(variable)


# ---------------------------------------------------------------------------
# GSOD day-window bias correction
# ---------------------------------------------------------------------------
# NOAA GSOD aggregates its daily summary over the UTC calendar day; GHCN-D
# and ECA&D aggregate over the station's local civil day (consolidate.py:
# "GSOD summarises a UTC day where GHCN-D and ECA&D use a local climatological
# day"). For an Eastern Mediterranean station (UTC+2/+3), that shifts GSOD's
# aggregation window several hours relative to every other source in this
# archive, producing a small, systematic bias that is not genuine climate
# signal.
#
# No sub-daily GSOD data exists in this archive to re-aggregate over the
# correct window directly, so the correction here is the best available
# proxy: an additive bias measured on confirmed-duplicate GSOD/GHCN-D station
# pairs -- "GSOD runs 0.303 degC warmer than GHCN-D on the days they share"
# (consolidate.py; the same measurement check_compatibility's bias test
# catches and refuses to splice, rather than correcting).
#
# Scoped to 'tas' only. A parallel check on tasmax fallback-candidate pairs
# (fallback_compatibility.csv, 48 tested GSOD/tasmax candidates) gave a mean
# bias of -0.13 degC with a near-zero median -- inconsistent in both
# magnitude and sign with a uniform day-window offset. Applying 0.303 to
# tasmax (or tasmin, untested) would be guessing a number the evidence does
# not support; it stays at 0.0 until a dedicated remeasurement documents one.
GSOD_DAY_WINDOW_BIAS_CORRECTION: dict[str, float] = {
    "tas": 0.303,
}


def apply_day_window_correction(
    values: pd.Series, source: str, variable: str
) -> pd.Series:
    """
    Subtract the measured day-window systematic bias for (source, variable),
    if one has been documented in GSOD_DAY_WINDOW_BIAS_CORRECTION. A no-op
    for every other source or variable -- silence here means "not measured",
    never "measured as zero".

    Applied in the silver layer, after to_canonical() and before any
    fallback-compatibility comparison, so GSOD's own known offset cannot
    itself trigger check_compatibility's bias guardrail nor get spliced into
    a series carrying it uncorrected.
    """
    if source != "gsod":
        return values
    offset = GSOD_DAY_WINDOW_BIAS_CORRECTION.get(variable, 0.0)
    if offset == 0.0:
        return values
    return values - offset
