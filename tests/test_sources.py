"""
Adapter tests against fixed provider fixtures.

Plan section 1.5 lists the required coverage: "inventory parsing, pagination,
retry/resume, checksum mismatch, missing/trace values, unit metadata, quality
flags and an upstream schema-change failure that stops loudly rather than
silently dropping columns."
"""
from __future__ import annotations

import gzip
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from observations import net
from observations.sources.c3s_land import C3sLandSource, VARIABLE_TO_CDS
from observations.sources.ecad import EcadSource, _dms_to_degrees, _parse_stations
from observations.sources.ghcnd import GhcndSource
from observations.sources.gsod import GsodSource
from observations.types import (
    Domain,
    DownloadRequest,
    SchemaError,
    SourceSnapshot,
    StationInventory,
)
from tests import fixtures

DOMAIN = Domain(33.0, 43.0, 11.0, 32.0, buffer_deg=2.0)
PERIOD = (1985, 2005)
VARIABLES = ("tas", "tasmin", "tasmax", "pr", "hurs", "sfcWind", "gust")


@pytest.fixture(autouse=True)
def archive(tmp_path, monkeypatch):
    monkeypatch.setenv("STATION_ARCHIVE_ROOT", str(tmp_path / "archive"))


def _snapshot(path: Path, source: str, release: str) -> SourceSnapshot:
    return SourceSnapshot(
        request=DownloadRequest(source=source, key="k", relative_path=path.name, url="https://x"),
        path=path,
        sha256="x",
        n_bytes=path.stat().st_size,
        retrieved_at="2026-09-13T00:00:00+00:00",
        release=release,
        status="downloaded",
    )


# ===========================================================================
# GHCN-Daily
# ===========================================================================

@pytest.fixture
def ghcnd(monkeypatch):
    source = GhcndSource()
    responses = {
        "ghcnd-version.txt": fixtures.GHCND_VERSION_TEXT,
        "ghcnd-stations.txt": fixtures.GHCND_STATIONS_TEXT,
        "ghcnd-inventory.txt": fixtures.GHCND_INVENTORY_TEXT,
    }

    def fake_get(url, byte_range=None):
        for name, text in responses.items():
            if url.endswith(name):
                return text.encode("utf-8")
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(net, "get_bytes", fake_get)
    return source


def test_ghcnd_release_is_parsed_from_the_version_sentence(ghcnd):
    assert ghcnd.release() == "3.34-upd-2026091118"


def test_ghcnd_release_raises_when_the_version_file_changes(monkeypatch):
    with pytest.raises(SchemaError, match="version token"):
        GhcndSource._parse_version("GHCN Daily was updated recently.")


def test_ghcnd_discovery_keeps_only_in_domain_stations(ghcnd):
    inventory = ghcnd.discover(DOMAIN, VARIABLES, PERIOD)
    found = set(inventory.frame["source_station_id"])
    assert "GRE00105241" in found
    assert "ACW00011604" not in found       # Antigua
    assert "AE000041196" not in found       # Sharjah


def test_ghcnd_discovery_drops_elements_outside_the_window(ghcnd):
    """GRE00105242 PRCP runs 2012-2020, entirely after the admission window."""
    inventory = ghcnd.discover(DOMAIN, VARIABLES, PERIOD)
    rows = inventory.frame
    heraklion = rows[rows["source_station_id"] == "GRE00105242"]
    assert set(heraklion["provider_element"]) == {"TMAX"}


def test_ghcnd_discovery_maps_elements_to_canonical_variables(ghcnd):
    inventory = ghcnd.discover(DOMAIN, VARIABLES, PERIOD)
    mapping = dict(zip(inventory.frame["provider_element"], inventory.frame["variable"]))
    assert mapping["TMAX"] == "tasmax"
    assert mapping["TMIN"] == "tasmin"
    assert mapping["PRCP"] == "pr"


def test_ghcnd_discovery_records_provider_units(ghcnd):
    inventory = ghcnd.discover(DOMAIN, VARIABLES, PERIOD)
    units = dict(zip(inventory.frame["provider_element"], inventory.frame["units"]))
    assert units["TMAX"] == "degC_tenths"
    assert units["PRCP"] == "mm_tenths"


def test_ghcnd_discovery_leaves_uncounted_statistics_empty(ghcnd):
    """Counts are not available from the inventory file and must not be invented."""
    inventory = ghcnd.discover(DOMAIN, VARIABLES, PERIOD)
    assert inventory.frame["n_obs_period"].isna().all()


def test_ghcnd_plan_is_one_request_per_station(ghcnd):
    inventory = ghcnd.discover(DOMAIN, VARIABLES, PERIOD)
    requests = ghcnd.plan(inventory)
    assert len(requests) == inventory.n_stations()
    assert all(r.url.endswith(".csv.gz") for r in requests)


def test_ghcnd_fixed_width_parser_rejects_short_lines():
    from observations.sources.ghcnd import _parse_fixed_width, _STATION_FIELDS

    with pytest.raises(SchemaError, match="expected at least"):
        _parse_fixed_width("TOO SHORT\n", _STATION_FIELDS, "ghcnd-stations.txt")


def test_ghcnd_parse_preserves_tenths_and_flags(tmp_path):
    path = tmp_path / "GRE00105241.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(fixtures.GHCND_STATION_CSV)

    observations = list(GhcndSource().parse(_snapshot(path, "ghcnd", "3.34")))

    tmax = next(o for o in observations if o.element == "TMAX" and o.obs_date == date(1985, 1, 1))
    assert tmax.value == 142.0            # tenths, not 14.2
    assert tmax.units == "degC_tenths"

    flagged = next(o for o in observations if o.obs_date == date(1985, 1, 2) and o.element == "TMAX")
    assert flagged.provider_flags["q_flag"] == "G"


def test_ghcnd_parse_skips_the_missing_sentinel(tmp_path):
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(fixtures.GHCND_STATION_CSV)
    observations = list(GhcndSource().parse(_snapshot(path, "ghcnd", "3.34")))
    assert not any(o.value == -9999 for o in observations)
    assert not any(o.obs_date == date(1985, 1, 2) and o.element == "PRCP" for o in observations)


def test_ghcnd_parse_keeps_zero_precipitation(tmp_path):
    """Zero is a measurement; only -9999 means absent."""
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(fixtures.GHCND_STATION_CSV)
    observations = list(GhcndSource().parse(_snapshot(path, "ghcnd", "3.34")))
    zeros = [o for o in observations if o.element == "PRCP" and o.value == 0.0]
    assert len(zeros) == 1


def test_ghcnd_parse_stops_loudly_on_a_schema_change(tmp_path):
    path = tmp_path / "s.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("id,not-a-date,TMAX,1,,,,\n")
    with pytest.raises(SchemaError, match="YYYYMMDD"):
        list(GhcndSource().parse(_snapshot(path, "ghcnd", "3.34")))


# ===========================================================================
# ECA&D
# ===========================================================================

def test_ecad_dms_conversion_north_east():
    assert _dms_to_degrees("+37:58:00", "t") == pytest.approx(37.9667, abs=1e-4)


def test_ecad_dms_conversion_south_west_subtracts():
    """The sign applies to the whole coordinate, not just the degrees field."""
    assert _dms_to_degrees("-22:54:00", "t") == pytest.approx(-22.9, abs=1e-4)
    assert _dms_to_degrees("-043:10:00", "t") == pytest.approx(-43.1667, abs=1e-4)


def test_ecad_dms_rejects_malformed_coordinates():
    with pytest.raises(SchemaError):
        _dms_to_degrees("37.9667", "t")


def test_ecad_stations_parse_into_decimal_degrees():
    frame = _parse_stations(fixtures.ECAD_STATIONS_TEXT, "TG")
    athens = frame[frame["source_station_id"] == "247"].iloc[0]
    assert athens["name"] == "ATHENS"
    assert athens["lat"] == pytest.approx(37.9667, abs=1e-3)
    assert athens["lon"] == pytest.approx(23.7167, abs=1e-3)
    assert athens["elevation_m"] == 107


def test_ecad_stations_parse_raises_without_the_header():
    with pytest.raises(SchemaError, match="STAID,STANAME"):
        _parse_stations("no header here\n", "TG")


def test_ecad_parse_preserves_tenths_and_quality_codes(tmp_path):
    path = tmp_path / "TG_STAID000247.txt"
    path.write_text(fixtures.ECAD_SERIES_TEXT, encoding="latin1")

    observations = list(EcadSource().parse(_snapshot(path, "ecad", "12-08-2026")))

    first = observations[0]
    assert first.element == "TG"
    assert first.value == 112.0                 # 0.1 degC, unconverted
    assert first.units == "degC_tenths"
    assert first.source_station_id == "000247"
    assert first.provider_flags["quality_code"] == 0


def test_ecad_parse_keeps_suspect_values_with_the_flag(tmp_path):
    """Q=1 is qc.py's decision to make, not the downloader's."""
    path = tmp_path / "TG_STAID000247.txt"
    path.write_text(fixtures.ECAD_SERIES_TEXT, encoding="latin1")
    observations = list(EcadSource().parse(_snapshot(path, "ecad", "rel")))
    suspect = [o for o in observations if o.provider_flags["quality_code"] == 1]
    assert len(suspect) == 1
    assert suspect[0].value == 128.0


def test_ecad_parse_drops_rows_flagged_missing(tmp_path):
    path = tmp_path / "TG_STAID000247.txt"
    path.write_text(fixtures.ECAD_SERIES_TEXT, encoding="latin1")
    observations = list(EcadSource().parse(_snapshot(path, "ecad", "rel")))
    assert not any(o.obs_date == date(1985, 1, 3) for o in observations)
    assert not any(o.value == -9999 for o in observations)


def test_ecad_parse_records_the_contributing_source_id(tmp_path):
    """SOUID is what lets identity.py reason about blended-series overlap."""
    path = tmp_path / "TG_STAID000247.txt"
    path.write_text(fixtures.ECAD_SERIES_TEXT, encoding="latin1")
    observations = list(EcadSource().parse(_snapshot(path, "ecad", "rel")))
    assert {o.provider_flags["souid"] for o in observations} == {"125522", "136679"}


def test_ecad_parse_stops_loudly_when_columns_change(tmp_path):
    broken = fixtures.ECAD_SERIES_TEXT.replace("STAID, SOUID,    DATE,   TG, Q_TG", "STAID, DATE, TG")
    path = tmp_path / "TG_STAID000247.txt"
    path.write_text(broken, encoding="latin1")
    with pytest.raises(SchemaError, match="missing expected columns"):
        list(EcadSource().parse(_snapshot(path, "ecad", "rel")))


def test_ecad_discovery_filters_by_domain_and_marks_absent_series(monkeypatch):
    """
    End-to-end discovery over a synthetic archive:
    Athens has a series, Heraklion is advertised but absent, Rio is out of domain.
    """
    members = {
        "date_timestamp.txt": b"12-08-2026",
        "stations.txt": fixtures.ECAD_STATIONS_TEXT.encode("latin1"),
        "TG_STAID000247.txt": fixtures.ECAD_SERIES_TEXT.encode("latin1"),
    }
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    payload = blob.getvalue()

    monkeypatch.setattr(net, "content_length", lambda url: len(payload))
    monkeypatch.setattr(net, "supports_ranges", lambda url: True)
    monkeypatch.setattr(
        net,
        "get_bytes",
        lambda url, byte_range=None: payload
        if byte_range is None
        else payload[byte_range[0] : byte_range[0] + byte_range[1]],
    )

    source = EcadSource()
    inventory = source.discover(DOMAIN, ("tas",), PERIOD)
    frame = inventory.frame

    assert set(frame["source_station_id"]) == {"000247", "000248"}
    assert frame.loc[frame["source_station_id"] == "000247", "access_status"].iloc[0] == "public"
    assert (
        frame.loc[frame["source_station_id"] == "000248", "access_status"].iloc[0]
        == "restricted_or_absent"
    )


def test_ecad_plan_skips_stations_whose_series_is_absent(monkeypatch):
    """Only downloadable series become requests."""
    members = {
        "date_timestamp.txt": b"12-08-2026",
        "stations.txt": fixtures.ECAD_STATIONS_TEXT.encode("latin1"),
        "TG_STAID000247.txt": fixtures.ECAD_SERIES_TEXT.encode("latin1"),
    }
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    payload = blob.getvalue()

    monkeypatch.setattr(net, "content_length", lambda url: len(payload))
    monkeypatch.setattr(net, "supports_ranges", lambda url: True)
    monkeypatch.setattr(
        net,
        "get_bytes",
        lambda url, byte_range=None: payload
        if byte_range is None
        else payload[byte_range[0] : byte_range[0] + byte_range[1]],
    )

    source = EcadSource()
    requests = source.plan(source.discover(DOMAIN, ("tas",), PERIOD))
    assert [r.key for r in requests] == ["ecad/TG/000247"]
    assert requests[0].byte_ranges is not None   # ranged, not a full-archive pull


def test_ecad_licence_restriction_is_declared():
    """Plan 1.2 question 4: non-redistributable records must be identifiable."""
    assert "non-commercial" in EcadSource.licence.lower()
    assert "not redistributable" in EcadSource.redistribution.lower()


# ===========================================================================
# GSOD
# ===========================================================================

def test_gsod_directory_listing_filters_by_prefix(monkeypatch):
    monkeypatch.setattr(
        net, "get_bytes", lambda url, byte_range=None: fixtures.GSOD_DIRECTORY_HTML.encode()
    )
    names = GsodSource().list_year_files(1985)
    assert names == ["16716099999.csv", "16754099999.csv"]


def test_gsod_directory_listing_raises_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(net, "get_bytes", lambda url, byte_range=None: b"<html></html>")
    with pytest.raises(SchemaError, match="format may have changed"):
        GsodSource().list_year_files(1985)


def test_gsod_parse_keeps_fahrenheit_and_inches(tmp_path):
    path = tmp_path / "16716099999.csv"
    path.write_text(fixtures.GSOD_CSV, encoding="utf-8")

    observations = list(GsodSource().parse(_snapshot(path, "gsod", "retrieved-20260913")))
    temp = next(o for o in observations if o.element == "TEMP")
    assert temp.value == 77.0
    assert temp.units == "degF"

    precipitation = next(o for o in observations if o.element == "PRCP")
    assert precipitation.value == 0.04
    assert precipitation.units == "inch"


def test_gsod_parse_skips_each_columns_own_sentinel(tmp_path):
    """TEMP uses 9999.9, GUST 999.9, PRCP 99.99 — one sentinel per column."""
    path = tmp_path / "s.csv"
    path.write_text(fixtures.GSOD_CSV, encoding="utf-8")
    observations = list(GsodSource().parse(_snapshot(path, "gsod", "rel")))

    assert not any(o.obs_date == date(1985, 6, 16) and o.element == "TEMP" for o in observations)
    assert not any(o.obs_date == date(1985, 6, 15) and o.element == "GUST" for o in observations)
    # The second day's gust is a real 35.0 knots and must survive.
    gust = next(o for o in observations if o.element == "GUST")
    assert gust.value == 35.0


def test_gsod_parse_preserves_attribute_flags(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text(fixtures.GSOD_CSV, encoding="utf-8")
    observations = list(GsodSource().parse(_snapshot(path, "gsod", "rel")))
    precipitation = [o for o in observations if o.element == "PRCP"]
    assert precipitation[0].provider_flags["attributes"] == "C"
    assert precipitation[1].provider_flags["attributes"] == "I"


def test_gsod_parse_stops_loudly_on_missing_columns(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("FOO,BAR\n1,2\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="missing required columns"):
        list(GsodSource().parse(_snapshot(path, "gsod", "rel")))


def test_gsod_dewpoint_is_not_labelled_relative_humidity():
    """
    Plan 1.4 allows deriving RH only where inputs and formula are recorded,
    which makes it a standardization step, not a download-time relabelling.
    """
    from observations.sources.gsod import COLUMN_TO_VARIABLE

    assert COLUMN_TO_VARIABLE["DEWP"] == "dew_point"
    assert "hurs" not in COLUMN_TO_VARIABLE.values()


def test_gsod_plan_covers_the_station_year_range():
    inventory = StationInventory.from_records(
        [
            {
                **{c: "" for c in StationInventory.empty().frame.columns},
                "source": "gsod",
                "source_station_id": "16716099999",
                "variable": "tas",
                "first_date": "1985-01-01",
                "last_date": "1987-12-31",
            }
        ]
    )
    requests = GsodSource(year_start=1970, year_end=2020).plan(inventory)
    assert [r.metadata["year"] for r in requests] == [1985, 1986, 1987]


# ===========================================================================
# C3S land in-situ
# ===========================================================================

def test_c3s_request_matches_the_published_form():
    """
    Every key and value below was read from the dataset's form.json;
    'year' is single-valued because the form declares a single-choice widget.
    """
    request = C3sLandSource().build_request(DOMAIN, ("tas", "pr"), 1995)
    assert request["version"] == "3_0_0"
    assert request["time_aggregation"] == "daily"
    assert request["year"] == "1995"
    assert isinstance(request["year"], str)
    assert request["month"] == [f"{m:02d}" for m in range(1, 13)]
    assert len(request["day"]) == 31
    assert request["data_format"] == "csv"
    assert request["area"] == DOMAIN.as_cds_area()


def test_c3s_requests_only_variables_the_dataset_offers():
    variables = C3sLandSource().cds_variables(("tas", "pr", "sfcWind", "hurs", "tasmax"))
    offered = {
        "accumulated_precipitation",
        "air_pressure",
        "air_pressure_at_sea_level",
        "air_temperature",
        "dew_point_temperature",
        "fresh_snow",
        "snow_depth",
        "snow_water_equivalent",
        "wind_from_direction",
        "wind_speed",
    }
    assert set(variables).issubset(offered)


def test_c3s_does_not_claim_humidity_or_daily_extremes():
    """The dataset has no hurs/tasmin/tasmax variable; the mapping must not pretend."""
    assert "hurs" not in VARIABLE_TO_CDS
    assert "tasmin" not in VARIABLE_TO_CDS
    assert "tasmax" not in VARIABLE_TO_CDS


def test_c3s_always_requests_dew_point_for_downstream_humidity():
    assert "dew_point_temperature" in C3sLandSource().cds_variables(("tas",))


def test_c3s_plan_is_one_request_per_year(monkeypatch):
    from observations import config

    monkeypatch.setattr(config, "FETCH_YEAR_START", 1990)
    monkeypatch.setattr(config, "FETCH_YEAR_END", 1992)
    inventory = StationInventory.from_records(
        [
            {
                **{c: "" for c in StationInventory.empty().frame.columns},
                "source": "c3s_land",
                "source_station_id": "GR-ATH-001",
                "variable": "tas",
            }
        ]
    )
    requests = C3sLandSource().plan(inventory)
    assert [r.metadata["year"] for r in requests] == [1990, 1991, 1992]
    assert all(r.payload is not None and r.url is None for r in requests)


def test_c3s_parse_reads_a_zipped_csv_delivery(tmp_path):
    path = tmp_path / "1995.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("observations.csv", fixtures.C3S_CSV)

    observations = list(C3sLandSource().parse(_snapshot(path, "c3s_land", "3_0_0-daily")))
    temperatures = [o for o in observations if o.element == "air_temperature"]
    assert temperatures[0].value == 284.15      # kelvin, unconverted
    assert temperatures[0].units == "K"
    assert temperatures[0].obs_date == date(1995, 3, 1)


def test_c3s_parse_stops_loudly_on_an_unrecognised_schema(tmp_path):
    path = tmp_path / "1995.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("observations.csv", "a,b\n1,2\n")
    with pytest.raises(SchemaError, match="could not resolve required C3S columns"):
        list(C3sLandSource().parse(_snapshot(path, "c3s_land", "rel")))


def test_c3s_inventory_filters_out_of_domain_stations(tmp_path):
    source = C3sLandSource()
    frame = pd.read_csv(io.StringIO(fixtures.C3S_CSV))
    inventory = source._inventory_from_frame(frame, DOMAIN, 1995, tmp_path / "x.zip")
    assert set(inventory.frame["source_station_id"]) == {"GR-ATH-001"}


def test_c3s_client_is_not_needed_for_planning():
    """Discovery and planning must work on a machine with no CDS credentials."""
    source = C3sLandSource()
    source.build_request(DOMAIN, ("tas",), 1995)
    assert source._client is None


def test_ecad_release_is_reduced_to_an_iso_date():
    """
    date_timestamp.txt holds a sentence; using it verbatim would put '&', '('
    and ',' into the archive directory name.
    """
    from observations.sources.ecad import _release_date

    stamp = "EUROPEAN CLIMATE ASSESSMENT & DATASET (ECA&D), file created on 12-08-2026"
    assert _release_date(stamp) == "2026-08-12"


def test_ecad_release_raises_when_no_date_is_present():
    from observations.sources.ecad import _release_date

    with pytest.raises(SchemaError, match="no DD-MM-YYYY date"):
        _release_date("ECA&D archive, created recently")


def test_ghcnd_station_line_without_a_wmo_id_is_accepted():
    """
    NOAA right-trims ghcnd-stations.txt: a station with no WMO ID ends after
    its name, and thousands of real lines are shorter than the nominal record
    width.  The optional trailing fields must read as empty, not raise.
    """
    from observations.sources.ghcnd import (
        _parse_fixed_width,
        _STATION_FIELDS,
        _STATION_REQUIRED_WIDTH,
    )

    # Name padded to the end of its field (column 71), then nothing: exactly
    # the shape of a real station line whose GSN/HCN/WMO fields are all absent.
    line = "GRE00105241  37.9000   23.7500   107.0   ATHENS HELLINIKON".ljust(71)
    assert len(line) == 71
    frame = _parse_fixed_width(
        line + "\n", _STATION_FIELDS, "ghcnd-stations.txt", _STATION_REQUIRED_WIDTH
    )
    assert frame.loc[0, "source_station_id"] == "GRE00105241"
    assert frame.loc[0, "name"] == "ATHENS HELLINIKON"
    assert frame.loc[0, "wmo_id"] == ""


def test_ghcnd_station_line_truncated_mid_record_still_raises():
    """A genuine column shift must still be caught."""
    from observations.sources.ghcnd import (
        _parse_fixed_width,
        _STATION_FIELDS,
        _STATION_REQUIRED_WIDTH,
    )

    with pytest.raises(SchemaError, match="expected at least"):
        _parse_fixed_width(
            "GRE00105241  37.9000\n",
            _STATION_FIELDS,
            "ghcnd-stations.txt",
            _STATION_REQUIRED_WIDTH,
        )
