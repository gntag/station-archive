"""
review.py
=========
Build a human-review workbook for station pairs the rules could not resolve.

WHY THIS EXISTS
---------------
Plan section 1.3 makes "metadata review" an explicit part of the matching
ladder, and tier 4 says unresolved candidates "remain separate and receive a
duplicate-risk flag" -- a flag is a request for review, not a verdict.  This
module turns those flags into something a person can actually adjudicate: for
every rejected pair, what the two archives say about the station, and what
their observations say, variable by variable.

The rules deliberately refuse to merge on insufficient evidence.  That is the
right default for an automated pass, but it leaves real duplicates split, and
a split duplicate puts one physical station in both training and validation
folds.  This workbook is how those get caught.

WHAT IT DOES NOT DO
-------------------
Decide anything.  Every row carries a blank decision column.  Nothing here
changes the crosswalk; applying decisions is a separate, deliberate step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from station_archive import identity
from station_archive.policy import DEFAULT_POLICY, Policy

# Variables compared side by side for every pair.  All seven, not just the
# one the matching gate happened to use: two archives can agree closely on
# temperature and disagree completely on precipitation, and that pattern is
# itself evidence about whether they are the same instrument.
REVIEW_VARIABLES: tuple[str, ...] = DEFAULT_POLICY.variables

# Columns the reviewer fills in.  Left blank deliberately.
DECISION_COLUMNS: tuple[str, ...] = ("DECISION_merge", "DECISION_note")


# Identity columns added to every sheet.  A reader adjudicating a merge should
# never have to cross-reference a station ID against another sheet to find out
# what the station is called or where it sits.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "left_name", "left_lat", "left_lon", "left_elevation_m",
    "right_name", "right_lat", "right_lon", "right_elevation_m",
)


def add_station_identity(
    frame: pd.DataFrame,
    lookup: dict[str, pd.Series],
    after: str = "right",
) -> pd.DataFrame:
    """
    Add name, latitude, longitude and elevation for both sides of each pair.

    Placed immediately after the station identifiers rather than appended, so
    the identity of a row is readable without scrolling. Names come through
    blank where a source publishes none ("where applicable"); coordinates are
    always present, since a station without them never entered the crosswalk.
    """
    if frame.empty or "left" not in frame.columns or "right" not in frame.columns:
        return frame

    working = frame.copy()
    for side in ("left", "right"):
        rows = [lookup.get(key) for key in working[side]]
        working[f"{side}_name"] = [getattr(r, "name", "") if r is not None else "" for r in rows]
        working[f"{side}_lat"] = [getattr(r, "lat", None) if r is not None else None for r in rows]
        working[f"{side}_lon"] = [getattr(r, "lon", None) if r is not None else None for r in rows]
        working[f"{side}_elevation_m"] = [
            getattr(r, "elevation_m", None) if r is not None else None for r in rows
        ]

    # Reorder so identity sits next to the identifiers it describes.
    identity_cols = [c for c in IDENTITY_COLUMNS if c in working.columns]
    rest = [c for c in working.columns if c not in identity_cols]
    anchor = rest.index(after) + 1 if after in rest else len(rest)
    ordered = rest[:anchor] + identity_cols + rest[anchor:]
    return working.loc[:, ordered]


def _station_lookup(aliases: pd.DataFrame) -> dict[str, pd.Series]:
    """Map 'source:station_id' -> its alias row."""
    lookup = {}
    for row in aliases.itertuples(index=False):
        lookup[f"{row.source}:{row.source_station_id}"] = row
    return lookup


def per_variable_comparison(
    left_key: str,
    right_key: str,
    series_loader,
    variables: Sequence[str] = REVIEW_VARIABLES,
) -> list[dict]:
    """
    Overlap statistics for one pair, for every variable both sides report.

    Returns one row per variable with data on both sides; variables only one
    side reports are reported with n_overlap 0 so their absence is visible
    rather than silently omitted.
    """
    left_source, left_id = left_key.split(":", 1)
    right_source, right_id = right_key.split(":", 1)

    rows = []
    for variable in variables:
        left = series_loader.series(left_source, left_id, variable)
        right = series_loader.series(right_source, right_id, variable)

        row = {
            "left": left_key,
            "right": right_key,
            "variable": variable,
            "left_days": len(left),
            "right_days": len(right),
        }

        left_units = identity.provider_units(left_source, variable)
        right_units = identity.provider_units(right_source, variable)

        if left.empty or right.empty or not left_units or not right_units:
            row.update(
                {
                    "n_overlap": 0,
                    "correlation": None,
                    "bias": None,
                    "mean_abs_diff": None,
                    "exact_fraction": None,
                    "verdict": "no comparable data",
                }
            )
        else:
            stats = identity.overlap_statistics(
                left, right, variable, left_units, right_units
            )
            row.update(
                {
                    "n_overlap": stats.n_overlap,
                    "correlation": _clean(stats.correlation, 5),
                    "bias": _clean(stats.bias, 3),
                    "mean_abs_diff": _clean(stats.mean_abs_diff, 3),
                    "exact_fraction": _clean(stats.exact_fraction, 3),
                    "verdict": _verdict(stats),
                }
            )
        rows.append(row)
    return rows


def _clean(value: float, digits: int) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), digits)


def _verdict(stats, policy: Policy = DEFAULT_POLICY) -> str:
    """
    A plain-language reading of one variable's comparison.

    Written for a reader deciding a merge, not for a machine: the question is
    always "does this look like one instrument reported twice".
    """
    if stats.n_overlap < policy.identity.min_overlap_days:
        return f"too few shared days ({stats.n_overlap}) to judge"
    if pd.isna(stats.correlation):
        return "no variation; cannot judge"
    if stats.exact_fraction >= 0.99 and stats.mean_abs_diff <= 0.05:
        return "IDENTICAL - same record"
    if stats.exact_fraction >= 0.5:
        return "very close - likely same station"
    if stats.correlation >= 0.99 and stats.mean_abs_diff <= 0.5:
        return "close - plausibly same station"
    if stats.correlation >= 0.99:
        return f"correlated but differs by {stats.mean_abs_diff:.2f} - likely different sites"
    return "poor agreement - different stations"


def _pair_summary(
    evidence_row: pd.Series,
    lookup: dict[str, pd.Series],
    variable_rows: list[dict],
) -> dict:
    """One review row: what the metadata says, what the observations say."""
    left_key = evidence_row["left"]
    right_key = evidence_row["right"]
    left = lookup.get(left_key)
    right = lookup.get(right_key)

    comparable = [r for r in variable_rows if r["n_overlap"] > 0]
    identical = [r for r in comparable if str(r["verdict"]).startswith("IDENTICAL")]
    close = [
        r for r in comparable
        if str(r["verdict"]).startswith(("very close", "close"))
    ]
    poor = [
        r for r in comparable
        if "different" in str(r["verdict"])
    ]

    if identical:
        recommendation = (
            f"LIKELY MERGE - {len(identical)} variable(s) identical: "
            + ", ".join(r["variable"] for r in identical)
        )
    elif close and not poor:
        recommendation = (
            f"CONSIDER MERGE - {len(close)} variable(s) close, none contradicting"
        )
    elif poor and not identical:
        recommendation = (
            f"LIKELY DISTINCT - {len(poor)} variable(s) disagree: "
            + ", ".join(r["variable"] for r in poor)
        )
    elif not comparable:
        recommendation = "NO EVIDENCE - decide on metadata alone"
    else:
        recommendation = "MIXED - variables disagree with each other"

    return {
        "left": left_key,
        "right": right_key,
        "rejected_because": evidence_row["reason"],
        "recommendation": recommendation,
        **{c: "" for c in DECISION_COLUMNS},
        "distance_km": round(float(evidence_row["distance_km"]), 3),
        "elevation_diff_m": evidence_row["elevation_diff_m"],
        "name_similarity": evidence_row["name_similarity"],
        "left_name": getattr(left, "name", ""),
        "right_name": getattr(right, "name", ""),
        "left_lat": getattr(left, "lat", None),
        "right_lat": getattr(right, "lat", None),
        "left_lon": getattr(left, "lon", None),
        "right_lon": getattr(right, "lon", None),
        "left_elevation_m": getattr(left, "elevation_m", None),
        "right_elevation_m": getattr(right, "elevation_m", None),
        "left_wmo": getattr(left, "wmo_id", ""),
        "right_wmo": getattr(right, "wmo_id", ""),
        "left_variables": getattr(left, "variables", ""),
        "right_variables": getattr(right, "variables", ""),
        "variables_compared": len(comparable),
        "variables_identical": len(identical),
        "variables_disagreeing": len(poor),
        "best_overlap_days": max((r["n_overlap"] for r in comparable), default=0),
    }


def already_merged(evidence: pd.DataFrame) -> pd.DataFrame:
    """
    Pairs the pipeline merged on its own authority, least defensible first.

    Tier 4 gives a reviewer every pair the rules REFUSED. Nothing showed the
    opposite: the merges the rules made without being asked. Those are the
    consequential ones -- a refusal leaves two stations that were always two,
    while a wrong merge silently fuses two sites and puts one of them in the
    other's series and folds.

    Pairs a reviewer already decided are excluded: they had permission, and
    re-presenting them as unapproved would bury the ones that did not.

    Ordered by how much doubt the evidence itself carries -- distance first,
    then elevation disagreement, then name dissimilarity -- so attention goes
    to the merges least supported by their own numbers rather than to whatever
    sorted first alphabetically.
    """
    if evidence.empty:
        return pd.DataFrame()
    merged = evidence[evidence["accepted"].astype(bool)].copy()
    merged = merged[
        ~merged["reason"].astype(str).str.startswith("reviewer decision")
    ]
    if merged.empty:
        return pd.DataFrame()

    merged = merged.rename(columns={"reason": "merged_because"})
    merged["distance_km"] = pd.to_numeric(merged["distance_km"], errors="coerce")
    merged["elevation_diff_m"] = pd.to_numeric(
        merged["elevation_diff_m"], errors="coerce"
    )
    merged = merged.sort_values(
        ["distance_km", "elevation_diff_m", "name_similarity"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    for column in DECISION_COLUMNS:
        merged[column] = ""

    columns = [
        "left", "right", "tier_name", "merged_because",
        *DECISION_COLUMNS,
        "distance_km", "elevation_diff_m", "name_similarity",
        "match_variable", "n_overlap_days", "correlation",
        "mean_abs_diff", "exact_fraction",
    ]
    return merged.loc[:, [c for c in columns if c in merged.columns]]


def build_review(
    evidence: pd.DataFrame,
    aliases: pd.DataFrame,
    series_loader,
    max_distance_km: float = 3.0,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Assemble every sheet of the review workbook.

    Only rejected pairs are reviewable -- an accepted pair has already been
    merged.  Pairs beyond 'max_distance_km' are kept on their own sheet
    rather than dropped, because a station whose two archives disagree about
    its position by several kilometres is exactly the case where distance is
    the least trustworthy evidence.
    """
    lookup = _station_lookup(aliases)
    rejected = evidence[~evidence["accepted"]].copy()

    # Pairs the never-merge rule forbids are not decisions -- they are
    # settled.  They are listed on their own sheet for transparency but
    # carry no decision column, because no answer would change anything.
    never = rejected[rejected["reason"] == identity.NEVER_MERGE_REASON]
    rejected = rejected[rejected["reason"] != identity.NEVER_MERGE_REASON]

    near = rejected[rejected["distance_km"] <= max_distance_km]
    far = rejected[rejected["distance_km"] > max_distance_km]

    review_rows = []
    variable_rows = []

    for index, (_, row) in enumerate(near.iterrows(), start=1):
        per_variable = per_variable_comparison(row["left"], row["right"], series_loader)
        variable_rows.extend(per_variable)
        review_rows.append(_pair_summary(row, lookup, per_variable))
        if progress and index % 25 == 0:
            print(f"[review] compared {index}/{len(near)} pairs")

    review = pd.DataFrame(review_rows)
    if not review.empty:
        # Most likely merges first: that is the order a reviewer wants.
        order = {
            "LIKELY MERGE": 0, "CONSIDER MERGE": 1, "MIXED": 2,
            "NO EVIDENCE": 3, "LIKELY DISTINCT": 4,
        }
        review["_sort"] = review["recommendation"].str.split(" -").str[0].map(order).fillna(9)
        review = review.sort_values(["_sort", "distance_km"]).drop(columns="_sort")
        review = review.reset_index(drop=True)

    far_rows = []
    for _, row in far.iterrows():
        far_rows.append(
            {
                "left": row["left"],
                "right": row["right"],
                "distance_km": round(float(row["distance_km"]), 3),
                "elevation_diff_m": row["elevation_diff_m"],
                "rejected_because": row["reason"],
                **{c: "" for c in DECISION_COLUMNS},
            }
        )

    never_rows = [
        {
            "left": row["left"],
            "right": row["right"],
            "distance_km": round(float(row["distance_km"]), 3),
            "elevation_diff_m": row["elevation_diff_m"],
            "forbidden_because": (
                "zero overlapping days in any variable - project rule "
                "forbids merging, no override"
            ),
        }
        for _, row in never.iterrows()
    ]

    return {
        "Review": review,
        "Already_merged": add_station_identity(already_merged(evidence), lookup),
        "Per_variable": add_station_identity(pd.DataFrame(variable_rows), lookup),
        "Beyond_radius": add_station_identity(pd.DataFrame(far_rows), lookup),
        "Never_merge_rule": add_station_identity(pd.DataFrame(never_rows), lookup),
    }


def summary_sheet(sheets: dict[str, pd.DataFrame], conflicts: pd.DataFrame) -> pd.DataFrame:
    """Counts and a plain-language guide to the other sheets."""
    review = sheets["Review"]
    counts = (
        review["recommendation"].str.split(" -").str[0].value_counts()
        if not review.empty
        else pd.Series(dtype=int)
    )

    rows = [
        ("Pairs needing a decision", len(review),
         "Sheet 'Review'. Rejected pairs within 3 km. Fill DECISION_merge with yes/no."),
        ("  LIKELY MERGE", int(counts.get("LIKELY MERGE", 0)),
         "At least one variable is identical - almost certainly one station."),
        ("  CONSIDER MERGE", int(counts.get("CONSIDER MERGE", 0)),
         "Close on every comparable variable, nothing contradicting."),
        ("  MIXED", int(counts.get("MIXED", 0)),
         "Variables disagree with each other - needs a human."),
        ("  NO EVIDENCE", int(counts.get("NO EVIDENCE", 0)),
         "No overlapping observations. Decide on coordinates, name, elevation."),
        ("  LIKELY DISTINCT", int(counts.get("LIKELY DISTINCT", 0)),
         "Observations disagree - probably genuinely different stations."),
        ("Merged WITHOUT your approval", len(sheets.get("Already_merged", [])),
         "Sheet 'Already_merged'. The rules merged these on their own. Widest "
         "and least-agreeing first. Write 'no' in DECISION_merge to split one."),
        ("Forbidden by never-merge rule", len(sheets.get("Never_merge_rule", [])),
         "Sheet 'Never_merge_rule'. Zero overlapping days in any variable. "
         "Permanently unmergeable - no decision column, nothing to decide."),
        ("Pairs beyond 3 km", len(sheets["Beyond_radius"]),
         "Sheet 'Beyond_radius'. Never tested on series; listed for completeness."),
        ("Already-merged conflicts", len(conflicts),
         "Sheet 'Metadata_conflicts'. Merged, but the archives disagree on "
         "coordinates or elevation - one of them is wrong."),
        ("Per-variable comparisons", len(sheets["Per_variable"]),
         "Sheet 'Per_variable'. Every variable for every reviewed pair."),
    ]
    return pd.DataFrame(rows, columns=["item", "count", "what it means"])


def carry_forward_decisions(
    sheets: dict[str, pd.DataFrame],
    previous: dict[tuple[str, str], str],
) -> dict[str, pd.DataFrame]:
    """
    Re-apply decisions already made to a freshly built workbook.

    Rebuilding used to blank every DECISION_merge, so regenerating the workbook
    silently destroyed the reviewer's work -- which is why hand-made backup
    copies of it accumulate in the review directory. A rebuild is a routine
    thing to want (the crosswalk changed, so the pairs did), and it must not
    cost a day of adjudication.

    Only pairs that still exist are carried: a decision about a pair the new
    crosswalk no longer generates has nothing to attach to, and inventing a row
    for it would resurrect a question that is no longer being asked.
    """
    if not previous:
        return sheets

    offered: set[tuple[str, str]] = set()
    out: dict[str, pd.DataFrame] = {}
    for name, frame in sheets.items():
        if frame.empty or "DECISION_merge" not in frame.columns:
            out[name] = frame
            continue
        working = frame.copy()
        decided = []
        for left, right, existing in zip(
            working["left"], working["right"], working["DECISION_merge"]
        ):
            offered.add((left, right))
            answer = previous.get((left, right))
            if answer is None:
                # Keep a blank blank. Read back from a spreadsheet an empty
                # cell arrives as NaN, and letting that through would write the
                # string "nan" into the next workbook as though it were an
                # answer.
                answer = "" if pd.isna(existing) else existing
            decided.append(answer)
        working["DECISION_merge"] = decided
        out[name] = working

    # The ledger. A 'yes' merges its pair, so the next rebuild presents it
    # nowhere -- not in Review, which holds only refusals, and not in
    # Already_merged, which excludes reviewer-decided pairs. Without somewhere
    # durable to live, the decision disappears on the rebuild and the merge it
    # authorised is silently lost the next time the crosswalk is built. This
    # sheet is that somewhere: every decision ever made, whether or not the
    # current crosswalk still asks the question.
    out["Decisions"] = pd.DataFrame(
        [
            {
                "left": left,
                "right": right,
                "DECISION_merge": answer,
                "still_offered": (left, right) in offered,
            }
            for (left, right), answer in sorted(previous.items())
        ],
        columns=["left", "right", "DECISION_merge", "still_offered"],
    )
    return out


# ---------------------------------------------------------------------------
# The Series sheets: the climatology screen's proposals, in the same workbook
# ---------------------------------------------------------------------------
# One workbook, one pass. A record's exclusion does not depend on any merge
# decision -- the screen judges source records, not merged stations -- so both
# kinds of question can be answered at the same sitting.

SERIES_SHEET_COLUMNS: tuple[str, ...] = (
    "alias", "canonical_station_id", "name", "lat", "lon", "variable",
    "RECOMMENDATION", "robust_z", "departure", "departure_units",
    "station_mean", "reference_mean", "n_matched_days", "reporting_fraction",
    "sampling_note", "n_neighbors", "neighbor_median_departure",
    "departure_vs_neighbors", "n_neighbors_same_direction", "reference_key",
)
SERIES_DECISION_COLUMNS: tuple[str, ...] = ("DECISION_exclude", "DECISION_note")
SERIES_LEDGER_COLUMNS: tuple[str, ...] = (
    "alias", "variable", "DECISION_exclude", "DECISION_note", "still_offered",
)
_RECOMMENDATION = {"propose_exclusion": "EXCLUDE", "review": "REVIEW"}


def series_sheet(screen: pd.DataFrame) -> pd.DataFrame:
    """
    The records the screen proposes for exclusion or review, strongest first.

    Recommendations only: the decision column is blank, and a blank keeps the
    record. Nothing leaves the gold table without a reviewer's 'yes'.
    """
    columns = [*SERIES_SHEET_COLUMNS, *SERIES_DECISION_COLUMNS]
    if screen.empty or "tier" not in screen.columns:
        return pd.DataFrame(columns=columns)
    flagged = screen[screen["tier"].isin(list(_RECOMMENDATION))].copy()
    flagged["RECOMMENDATION"] = flagged["tier"].map(_RECOMMENDATION)
    for column in SERIES_DECISION_COLUMNS:
        flagged[column] = ""
    for column in SERIES_SHEET_COLUMNS:
        if column not in flagged.columns:
            flagged[column] = float("nan")
    flagged["_strength"] = flagged["robust_z"].abs()
    flagged = flagged.sort_values(
        ["_strength", "variable", "alias"], ascending=[False, True, True]
    )
    return flagged[columns].reset_index(drop=True)


def carry_forward_exclusions(
    sheet: pd.DataFrame,
    previous: dict[tuple[str, str], tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-apply exclusion decisions already made, and keep every one in a ledger.

    The same reasoning as carry_forward_decisions: a rebuild must not cost an
    answer, and an answer whose row is no longer offered (the screen stopped
    flagging it, or the record now passes) still governs consolidation, so it
    needs somewhere durable to live -- the Series_decisions sheet.
    """
    ledger_empty = pd.DataFrame(columns=list(SERIES_LEDGER_COLUMNS))
    if not previous:
        return sheet, ledger_empty
    working = sheet.copy()
    offered = set(zip(working["alias"], working["variable"]))
    answers = [previous.get((a, v)) for a, v in zip(working["alias"], working["variable"])]
    working["DECISION_exclude"] = [
        answer[0] if answer else current
        for answer, current in zip(answers, working["DECISION_exclude"])
    ]
    working["DECISION_note"] = [
        answer[1] if answer else current
        for answer, current in zip(answers, working["DECISION_note"])
    ]
    ledger = pd.DataFrame(
        [
            {
                "alias": alias,
                "variable": variable,
                "DECISION_exclude": answer,
                "DECISION_note": note,
                "still_offered": (alias, variable) in offered,
            }
            for (alias, variable), (answer, note) in sorted(previous.items())
        ],
        columns=list(SERIES_LEDGER_COLUMNS),
    )
    return working, ledger


# ---------------------------------------------------------------------------
# The Start_here tab: what each tab is and what the reviewer writes where
# ---------------------------------------------------------------------------
# Qualitative only, so it never goes stale against the numbers on the other
# tabs and needs no analysis to rebuild. It carries no DECISION_ column of its
# own: nothing is read back from it.

INSTRUCTIONS_SHEET = "Start_here"

# Every tab the review command writes, in the order it writes them.
WORKBOOK_TABS: tuple[str, ...] = (
    INSTRUCTIONS_SHEET,
    "Summary",
    "Review",
    "Already_merged",
    "Series",
    "Per_variable",
    "Metadata_conflicts",
    "Beyond_radius",
    "Never_merge_rule",
    "Decisions",
    "Series_decisions",
)

_ANSWERS = (
    "yes / no. Capitals and spaces do not matter. A blank cell is no decision. "
    "Anything else (e.g. 'maybe') is ignored as undecided."
)

_INSTRUCTIONS: tuple[dict[str, str], ...] = (
    {"tab": "HOW THIS WORKS",
     "what it shows": "One workbook holds every decision the pipeline needs from a "
                      "person: which records are one station (merges) and which "
                      "records are excluded for a variable (series exclusions).",
     "what you do": "1. Fill in the decision columns on the tabs marked 'FILL IN'. "
                    "2. Save the file and close Excel. "
                    "3. Run 'station-archive apply'.",
     "column to fill": "", "write": ""},
    {"tab": "HOW THIS WORKS",
     "what it shows": "Rebuilding this workbook never loses an answer: every "
                      "decision is carried into the new copy and kept in the "
                      "ledger tabs.",
     "what you do": "Answer each row once. A row you answered will not be asked "
                    "again unless its evidence changes.",
     "column to fill": "", "write": ""},
    {"tab": INSTRUCTIONS_SHEET,
     "what it shows": "This page.",
     "what you do": "Read only.",
     "column to fill": "", "write": ""},
    {"tab": "Summary",
     "what it shows": "How many rows each tab holds and what the recommendations mean.",
     "what you do": "Read only.",
     "column to fill": "", "write": ""},
    {"tab": "Review",
     "what it shows": "FILL IN. Pairs of records within 3 km that the rules REFUSED "
                      "to merge, with the evidence and a recommendation "
                      "(LIKELY MERGE ... LIKELY DISTINCT).",
     "what you do": "Decide whether the two records are the same physical station.",
     "column to fill": "DECISION_merge (and optionally DECISION_note)",
     "write": "yes = merge them into one station. no = keep them separate. "
              "Blank = the rules stand (separate). " + _ANSWERS},
    {"tab": "Already_merged",
     "what it shows": "FILL IN. Pairs the rules MERGED on their own, widest and "
                      "least-agreeing first. A wrong merge puts two sites into one "
                      "series, so this is the more consequential direction.",
     "what you do": "Confirm or split each merge.",
     "column to fill": "DECISION_merge (and optionally DECISION_note)",
     "write": "yes = confirm the merge. no = split into two stations. "
              "Blank = the merge stands. " + _ANSWERS},
    {"tab": "Series",
     "what it shows": "FILL IN. Source records whose long-term mean, on the days "
                      "they share with CERRA, sits far from CERRA compared with "
                      "every other record of that variable. RECOMMENDATION "
                      "EXCLUDE = more than 4 robust SDs; REVIEW = 3 to 4. "
                      "Neighbour columns show how stations within 100 km compare: "
                      "many flagged the same way suggests a network problem, no "
                      "neighbours suggests a site CERRA cannot resolve (summit, "
                      "cape, valley).",
     "what you do": "Decide whether that record's values for that one variable are "
                    "believable. Compare station_mean with reference_mean, and "
                    "read the neighbour columns.",
     "column to fill": "DECISION_exclude (and optionally DECISION_note)",
     "write": "yes = exclude this record for this variable; the next-ranked "
              "source of the same station takes over, or the days stay empty. "
              "no = keep it. Blank = keep it (nothing is excluded without a yes). "
              "Your note is written into the audit table as the reason. " + _ANSWERS},
    {"tab": "Per_variable",
     "what it shows": "Every variable compared side by side for each pair on the "
                      "Review tab.",
     "what you do": "Read only: evidence for the Review tab.",
     "column to fill": "", "write": ""},
    {"tab": "Metadata_conflicts",
     "what it shows": "Merged pairs whose observations are identical but whose "
                      "archives disagree on coordinates or elevation, and which "
                      "values the archive uses.",
     "what you do": "Read only. Elevations are settled by the DEM rule during "
                    "'apply' (coordinates from the record with the most data, the "
                    "published elevation nearest the DEM).",
     "column to fill": "", "write": ""},
    {"tab": "Beyond_radius",
     "what it shows": "Pairs more than 3 km apart that were never tested on their "
                      "series.",
     "what you do": "Optional. Fill in only if you know two of them are one station.",
     "column to fill": "DECISION_merge (and optionally DECISION_note)",
     "write": "yes = merge. no = keep separate. Blank = no decision. " + _ANSWERS},
    {"tab": "Never_merge_rule",
     "what it shows": "Pairs with no observation day in common. They can never be "
                      "merged, whatever is written anywhere.",
     "what you do": "Read only. There is no decision column, because no answer "
                    "would change anything.",
     "column to fill": "", "write": ""},
    {"tab": "Decisions",
     "what it shows": "The ledger of every merge decision ever made, including "
                      "pairs no longer shown on another tab (still_offered = False).",
     "what you do": "Do not edit. Change a decision on the tab where the pair is "
                    "offered; the ledger follows.",
     "column to fill": "", "write": ""},
    {"tab": "Series_decisions",
     "what it shows": "The ledger of every series exclusion decision ever made.",
     "what you do": "Do not edit. Change a decision on the Series tab; the ledger "
                    "follows.",
     "column to fill": "", "write": ""},
)


def instructions_sheet() -> pd.DataFrame:
    """The Start_here tab: every tab, what it shows, and what to write where."""
    return pd.DataFrame(
        list(_INSTRUCTIONS),
        columns=["tab", "what it shows", "what you do", "column to fill", "write"],
    )


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    """
    Write the sheets to one .xlsx, with sensible column widths.

    If the target is locked -- which on Windows means the reviewer has it open
    in Excel, the most likely moment for a rebuild to be requested -- the
    workbook is written beside it with a timestamp rather than lost.  Failing
    the whole run because a file is open would discard several minutes of
    comparison work for a reason the reviewer can fix in a second.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    target = path
    if path.exists():
        try:
            with path.open("a"):
                pass
        except PermissionError:
            from datetime import datetime

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            print(
                f"[review] {path.name} is open in another program; "
                f"writing {target.name} instead"
            )

    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            output = frame if not frame.empty else pd.DataFrame({"(no rows)": []})
            output.to_excel(writer, sheet_name=name[:31], index=False)
            worksheet = writer.sheets[name[:31]]
            for column_cells in worksheet.columns:
                longest = max(
                    (len(str(cell.value)) for cell in column_cells if cell.value is not None),
                    default=10,
                )
                letter = column_cells[0].column_letter
                worksheet.column_dimensions[letter].width = min(max(longest + 2, 10), 55)
            worksheet.freeze_panes = "A2"
    return target
