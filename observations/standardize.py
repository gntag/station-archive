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
# THIS TABLE IS DELIBERATELY EMPTY. No day-window correction is applied to
# any source or variable, and none may be added without a measurement of the
# exact (source pair, variable) it will be applied to.
#
# An earlier version of this file subtracted 0.303 degC from GSOD 'tas'. That
# was withdrawn as unsupported. The number itself is real -- it is the median
# difference on confirmed-duplicate GHCN-D/GSOD pairs in this archive -- but
# every step from there to "subtract it from GSOD tas" was wrong:
#
#   * WRONG VARIABLE. Of the 20 GHCN-D/GSOD duplicate pairs carrying a bias,
#     19 are match_variable 'tasmax' and one is 'pr'. duplicate_evidence.csv
#     contains no 'tas' comparison at all. The constant is a tasmax
#     statistic; it was applied to tas.
#   * WRONG SOURCE PAIR. SOURCE_PRIORITY['tas'] is ('ecad', 'gsod') and
#     GHCN-D contributes no tas values whatsoever, so the correction could
#     only ever act where GSOD meets ECA&D -- a pair whose measured bias is
#     0.0002 degC (config.py).
#   * WRONG SIGN. Measured on this archive before the correction existed,
#     tas with an ecad primary and a gsod fallback ran at a median -0.16 degC
#     over 14 station pairs: GSOD is COLDER than ECA&D here, not warmer.
#     Subtracting a further 0.303 pushed two Greek stations past the +/-0.5
#     degC fallback guardrail and discarded 4,750 previously accepted days.
#
# A UTC-versus-local aggregation window is a real phenomenon, and it would be
# expected to show up most in tasmax/tasmin -- where a shifted window can
# capture a different diurnal extremum -- and least in tas, a daily mean.
# The withdrawn correction was applied in exactly the opposite pattern.
#
# The archive therefore carries provider values as issued, converted to
# canonical units and nothing more. check_compatibility already refuses to
# splice a source whose overlap bias is too large, which is the honest
# response to a suspected offset: leave the gap, do not manufacture the
# value. Diagnose, never silently adjust (plan section 1.4).
GSOD_DAY_WINDOW_BIAS_CORRECTION: dict[str, float] = {}


def apply_day_window_correction(
    values: pd.Series, source: str, variable: str
) -> pd.Series:
    """
    Subtract a measured day-window systematic bias for (source, variable),
    if one has been documented in GSOD_DAY_WINDOW_BIAS_CORRECTION.

    That table is empty, so this is currently a no-op for every input: the
    archive carries provider values as issued. It is kept as the single,
    named place such a correction would have to go, precisely so that adding
    one is a visible, reviewable change to a table that states what was
    measured, on which source pair, and for which variable -- rather than an
    arithmetic adjustment buried in a parsing path.

    Anything added here must be measured on the same (source pair, variable)
    it will be applied to. A statistic borrowed from a different variable, or
    from a source pair the variable never uses, is not evidence. See the
    table's comment for the withdrawn correction that ignored both rules.
    """
    if source != "gsod":
        return values
    offset = GSOD_DAY_WINDOW_BIAS_CORRECTION.get(variable, 0.0)
    if offset == 0.0:
        return values
    return values - offset
