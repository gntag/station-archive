"""
Tests for the Series sheets of the review workbook: what the reviewer is shown
from the climatology screen, and that a rebuild never costs a decision.
"""
from __future__ import annotations

import pandas as pd

from station_archive.climatology import TIER_EXCLUDE, TIER_OK, TIER_REVIEW
from station_archive.review import SERIES_SHEET_COLUMNS, carry_forward_exclusions, series_sheet


def _screen():
    return pd.DataFrame(
        [
            {"alias": "ecad:1", "variable": "pr", "robust_z": -math_inf(), "tier": TIER_EXCLUDE},
            {"alias": "ecad:2", "variable": "gust", "robust_z": -3.5, "tier": TIER_REVIEW},
            {"alias": "ecad:3", "variable": "tasmax", "robust_z": 4.6, "tier": TIER_EXCLUDE},
            {"alias": "ecad:4", "variable": "tas", "robust_z": 0.2, "tier": TIER_OK},
            {"alias": "ecad:5", "variable": "tas", "robust_z": float("nan"), "tier": ""},
        ]
    )


def math_inf():
    return float("inf")


def test_series_sheet_holds_flagged_rows_strongest_first():
    sheet = series_sheet(_screen())
    assert list(sheet["alias"]) == ["ecad:1", "ecad:3", "ecad:2"]
    assert list(sheet["RECOMMENDATION"]) == ["EXCLUDE", "EXCLUDE", "REVIEW"]
    assert (sheet["DECISION_exclude"] == "").all()
    assert list(sheet.columns)[-2:] == ["DECISION_exclude", "DECISION_note"]


def test_series_sheet_of_an_empty_screen_has_its_columns():
    sheet = series_sheet(pd.DataFrame())
    assert sheet.empty
    assert "DECISION_exclude" in sheet.columns
    assert set(SERIES_SHEET_COLUMNS) <= set(sheet.columns)


def test_decisions_are_carried_into_a_rebuilt_sheet_and_kept_in_the_ledger():
    sheet = series_sheet(_screen())
    previous = {
        ("ecad:1", "pr"): ("yes", "all zeros"),
        ("ecad:9", "tasmax"): ("yes", "no longer flagged, still decided"),
    }
    out, ledger = carry_forward_exclusions(sheet, previous)
    row = out[out["alias"] == "ecad:1"].iloc[0]
    assert (row["DECISION_exclude"], row["DECISION_note"]) == ("yes", "all zeros")
    assert out.loc[out["alias"] == "ecad:2", "DECISION_exclude"].item() == ""
    led = ledger.set_index(["alias", "variable"])
    assert bool(led.loc[("ecad:1", "pr"), "still_offered"]) is True
    assert bool(led.loc[("ecad:9", "tasmax"), "still_offered"]) is False
    assert led.loc[("ecad:9", "tasmax"), "DECISION_exclude"] == "yes"


def test_carry_forward_with_no_history_returns_an_empty_ledger():
    sheet = series_sheet(_screen())
    out, ledger = carry_forward_exclusions(sheet, {})
    pd.testing.assert_frame_equal(out, sheet)
    assert ledger.empty
    assert list(ledger.columns) == ["alias", "variable", "DECISION_exclude",
                                    "DECISION_note", "still_offered"]


def test_an_empty_sheet_still_keeps_every_decision_in_the_ledger():
    """With no screen to show, the ledger is the only place a 'yes' survives a rebuild."""
    out, ledger = carry_forward_exclusions(series_sheet(pd.DataFrame()),
                                           {("ecad:1", "pr"): ("yes", "all zeros")})
    assert out.empty
    assert ledger.iloc[0]["DECISION_exclude"] == "yes"
    assert not bool(ledger.iloc[0]["still_offered"])


# ---------------------------------------------------------------------------
# The Start_here tab
# ---------------------------------------------------------------------------

def test_instructions_cover_every_tab_the_review_writes():
    from station_archive.review import INSTRUCTIONS_SHEET, WORKBOOK_TABS, instructions_sheet

    sheet = instructions_sheet()
    assert WORKBOOK_TABS[0] == INSTRUCTIONS_SHEET
    described = set(sheet["tab"])
    assert set(WORKBOOK_TABS) <= described


def test_instructions_name_the_decision_columns_and_the_answers():
    from station_archive.review import instructions_sheet

    text = " ".join(instructions_sheet().astype(str).to_numpy().ravel())
    for phrase in ("DECISION_merge", "DECISION_exclude", "yes", "no", "blank"):
        assert phrase in text


def test_instructions_carry_no_decision_column_of_their_own():
    """The sheet must never be mistaken for one the pipeline reads answers from."""
    from station_archive.review import instructions_sheet

    assert not [c for c in instructions_sheet().columns if c.startswith("DECISION_")]
