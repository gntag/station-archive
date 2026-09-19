"""
Coverage-report tests and the end-to-end acquisition test.

The end-to-end case exercises the whole Step 1 download path against a
synthetic provider: discover -> plan -> manifest -> fetch -> re-fetch, and
checks the properties the plan's acceptance criteria name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pytest

from observations import config, inventory as inventory_module, manifests, net
from observations.sources.base import StationSource
from observations.types import (
    Domain,
    DownloadRequest,
    RawObservation,
    SourceSnapshot,
    StationInventory,
    INVENTORY_COLUMNS,
)


@pytest.fixture(autouse=True)
def archive(tmp_path, monkeypatch):
    monkeypatch.setenv("STATION_ARCHIVE_ROOT", str(tmp_path / "archive"))
    return tmp_path / "archive"


def _row(**overrides):
    row = {column: "" for column in INVENTORY_COLUMNS}
    row.update({"n_obs_period": 0})
    row.update(overrides)
    return row


# ===========================================================================
# Coverage report
# ===========================================================================

def test_gain_counts_a_site_only_gsod_has_as_shared():
    """A station both sources report at the same place is not a new site."""
    frame = pd.DataFrame(
        [
            _row(source="gsod", source_station_id="G1", variable="tas", lat=37.90, lon=23.75),
            _row(source="ghcnd", source_station_id="H1", variable="tas", lat=37.901, lon=23.751),
        ]
    )
    gain = inventory_module.gain_over_baseline(frame, baseline="gsod")
    assert gain.loc[0, "sites_not_in_baseline"] == 0
    assert gain.loc[0, "sites_shared_with_baseline"] == 1


def test_gain_counts_a_distant_station_as_new():
    frame = pd.DataFrame(
        [
            _row(source="gsod", source_station_id="G1", variable="tas", lat=37.90, lon=23.75),
            _row(source="ecad", source_station_id="E1", variable="tas", lat=35.33, lon=25.18),
        ]
    )
    gain = inventory_module.gain_over_baseline(frame, baseline="gsod")
    assert gain.loc[0, "sites_not_in_baseline"] == 1


def test_variable_density_counts_stations_per_variable():
    frame = pd.DataFrame(
        [
            _row(source="ecad", source_station_id="E1", variable="pr", lat=37.9, lon=23.7),
            _row(source="ecad", source_station_id="E2", variable="pr", lat=35.3, lon=25.1),
            _row(source="ecad", source_station_id="E1", variable="hurs", lat=37.9, lon=23.7),
        ]
    )
    density = inventory_module.variable_density(frame).set_index("variable")
    assert density.loc["pr", "ecad"] == 2
    assert density.loc["hurs", "ecad"] == 1


def test_subregion_coverage_places_stations_in_the_right_region():
    frame = pd.DataFrame(
        [
            _row(source="ecad", source_station_id="E1", variable="tas", lat=37.45, lon=25.33),
            _row(source="ecad", source_station_id="E2", variable="tas", lat=35.33, lon=25.18),
        ]
    )
    coverage = inventory_module.subregion_coverage(frame).set_index("subregion")
    assert coverage.loc["Cyclades", "sites"] == 1
    assert coverage.loc["Crete", "sites"] == 1


def test_elevation_coverage_bands_stations():
    frame = pd.DataFrame(
        [
            _row(source="ecad", source_station_id="E1", variable="tas",
                 lat=37.9, lon=23.7, elevation_m=15),
            _row(source="ecad", source_station_id="E2", variable="tas",
                 lat=39.0, lon=22.0, elevation_m=1400),
        ]
    )
    bands = inventory_module.elevation_coverage(frame).set_index("band")["sites"]
    assert bands["0-100 m"] == 1
    assert bands[">1000 m"] == 1


def test_licence_summary_separates_restricted_rows():
    frame = pd.DataFrame(
        [
            _row(source="ecad", source_station_id="E1", variable="tas",
                 access_status="public", licence="non-commercial"),
            _row(source="ecad", source_station_id="E2", variable="tas",
                 access_status="restricted_or_absent", licence="non-commercial"),
        ]
    )
    summary = inventory_module.licence_summary(frame).set_index("access_status")["rows"]
    assert summary["public"] == 1
    assert summary["restricted_or_absent"] == 1


def test_report_states_that_grouping_is_not_identity_resolution():
    """
    The proximity grouping must never be mistaken for the plan 1.3 crosswalk,
    so the report says so in its own text.
    """
    frame = pd.DataFrame([_row(source="gsod", source_station_id="G1", variable="tas",
                               lat=37.9, lon=23.7)])
    report = inventory_module.coverage_report(frame)
    assert "COVERAGE ESTIMATE" in report
    assert "identity.py" in report


def test_report_answers_all_four_discovery_questions():
    frame = pd.DataFrame([_row(source="gsod", source_station_id="G1", variable="tas",
                               lat=37.9, lon=23.7)])
    report = inventory_module.coverage_report(frame)
    for question in ("Question 1", "Question 2", "Question 3", "Question 4"):
        assert question in report


def test_report_handles_an_empty_inventory():
    report = inventory_module.coverage_report(pd.DataFrame(columns=list(INVENTORY_COLUMNS)))
    assert "Inventory rows: 0" in report


# ===========================================================================
# End-to-end acquisition
# ===========================================================================

class FakeSource(StationSource):
    """
    A minimal provider: two stations, one file each, served from memory.

    Stands in for a real adapter so the shared fetch/manifest/idempotency
    behaviour in StationSource can be exercised without a network.
    """

    name = "fake"
    licence = "test licence"
    licence_url = "https://example/licence"
    redistribution = "unrestricted"

    FILES = {
        "fake/S1": b"station one payload",
        "fake/S2": b"station two payload",
    }

    def __init__(self) -> None:
        self.transfers: list[str] = []

    def release(self) -> str:
        return "r1"

    def discover(self, domain: Domain, variables: Sequence[str], period) -> StationInventory:
        return StationInventory.from_records(
            [
                _row(source=self.name, source_station_id="S1", variable="tas",
                     lat=37.9, lon=23.7, licence=self.licence),
                _row(source=self.name, source_station_id="S2", variable="tas",
                     lat=35.3, lon=25.1, licence=self.licence),
            ]
        )

    def plan(self, inventory: StationInventory) -> list[DownloadRequest]:
        return [
            DownloadRequest(
                source=self.name,
                key=f"fake/{station}",
                relative_path=f"{station}.txt",
                url=f"https://example/{station}",
            )
            for station in sorted(inventory.frame["source_station_id"])
        ]

    def _transfer(self, request: DownloadRequest, destination: Path) -> None:
        self.transfers.append(request.key)
        net.write_bytes_atomic(self.FILES[request.key], destination)

    def parse(self, snapshot: SourceSnapshot) -> Iterator[RawObservation]:
        return iter(())


def test_end_to_end_discover_plan_fetch(archive):
    source = FakeSource()
    inventory = source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005))
    requests = source.plan(inventory)
    snapshots = source.fetch_all(requests, progress=False)

    assert len(snapshots) == 2
    assert all(s.status == "downloaded" for s in snapshots)
    for snapshot in snapshots:
        assert snapshot.path.exists()
        assert snapshot.sha256 == net.sha256_file(snapshot.path)


def test_rerun_skips_unchanged_files(archive):
    """Plan 1.5: skip a file only when its checksum matches."""
    source = FakeSource()
    requests = source.plan(source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005)))
    source.fetch_all(requests, progress=False)

    second = FakeSource()
    snapshots = second.fetch_all(requests, progress=False)
    assert second.transfers == []
    assert all(s.status == "skipped_checksum_match" for s in snapshots)


def test_rerun_refetches_a_corrupted_file(archive):
    """A file that no longer matches its recorded checksum must be fetched again."""
    source = FakeSource()
    requests = source.plan(source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005)))
    snapshots = source.fetch_all(requests, progress=False)
    snapshots[0].path.write_bytes(b"corrupted")

    second = FakeSource()
    second.fetch_all(requests, progress=False)
    assert second.transfers == ["fake/S1"]


def test_rerun_reproduces_the_archive_checksum(archive):
    """Plan 1.8: unchanged snapshots must produce the identical checksum."""
    source = FakeSource()
    requests = source.plan(source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005)))
    source.fetch_all(requests, progress=False)
    first = manifests.archive_checksum("fake", "r1")

    FakeSource().fetch_all(requests, progress=False)
    assert manifests.archive_checksum("fake", "r1") == first


def test_requests_are_manifested_even_when_a_transfer_fails(archive):
    """A run that dies partway must still record what it intended to fetch."""

    class BrokenSource(FakeSource):
        def _transfer(self, request, destination):
            raise RuntimeError("provider unavailable")

    source = BrokenSource()
    requests = source.plan(source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005)))
    snapshots = source.fetch_all(requests, progress=False)

    assert snapshots == []
    manifested = manifests.requests_path("fake", "r1").read_text(encoding="utf-8").splitlines()
    assert len(manifested) == 2


def test_licence_is_recorded_for_every_run(archive):
    source = FakeSource()
    requests = source.plan(source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005)))
    source.fetch_all(requests, progress=False)
    releases = manifests.read_releases()
    assert releases[-1]["licence"] == "test licence"
    assert releases[-1]["redistribution"] == "unrestricted"


def test_fetch_refuses_to_start_on_a_full_drive(archive, monkeypatch):
    """
    The plan records G: at ~0.1 GB free; starting there would leave an archive
    whose checksums describe incomplete files.
    """
    import shutil
    from collections import namedtuple

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1, 1, 1024))

    source = FakeSource()
    requests = source.plan(source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005)))
    with pytest.raises(RuntimeError, match="free"):
        source.fetch_all(requests, progress=False)


def test_snapshots_are_written_under_source_and_release(archive):
    source = FakeSource()
    requests = source.plan(source.discover(config.DOMAIN, config.VARIABLES, (1985, 2005)))
    snapshots = source.fetch_all(requests, progress=False)
    assert snapshots[0].path.parent == archive / "raw" / "fake" / "r1"
