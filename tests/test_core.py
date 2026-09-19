"""
Tests for the shared vocabulary, transfer layer, ranged ZIP reader and
manifests -- everything the adapters build on.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from observations import manifests, net, ziprange
from observations.types import (
    Domain,
    DownloadRequest,
    SchemaError,
    SourceRelease,
    SourceSnapshot,
    StationInventory,
    INVENTORY_COLUMNS,
)


@pytest.fixture(autouse=True)
def archive(tmp_path, monkeypatch):
    """Redirect the archive root so no test touches the real one."""
    monkeypatch.setenv("STATION_ARCHIVE_ROOT", str(tmp_path / "archive"))
    return tmp_path / "archive"


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_contains_interior_point():
    domain = Domain(33.0, 43.0, 11.0, 32.0)
    assert domain.contains(37.9, 23.7)


def test_domain_excludes_point_outside():
    domain = Domain(33.0, 43.0, 11.0, 32.0)
    assert not domain.contains(-22.9, -43.2)


def test_domain_buffer_admits_nearby_point():
    """A synoptic buffer must widen the box, not shift it."""
    tight = Domain(33.0, 43.0, 11.0, 32.0)
    buffered = Domain(33.0, 43.0, 11.0, 32.0, buffer_deg=2.0)
    assert not tight.contains(44.0, 23.0)
    assert buffered.contains(44.0, 23.0)


def test_domain_rejects_nan_coordinates():
    domain = Domain(33.0, 43.0, 11.0, 32.0)
    assert not domain.contains(float("nan"), 23.0)


def test_domain_cds_area_is_north_west_south_east():
    domain = Domain(33.0, 43.0, 11.0, 32.0, buffer_deg=1.0)
    assert domain.as_cds_area() == [44.0, 10.0, 32.0, 33.0]


# ---------------------------------------------------------------------------
# DownloadRequest
# ---------------------------------------------------------------------------

def test_request_requires_exactly_one_of_url_or_payload():
    with pytest.raises(ValueError):
        DownloadRequest(source="s", key="k", relative_path="p")
    with pytest.raises(ValueError):
        DownloadRequest(source="s", key="k", relative_path="p", url="u", payload={"a": 1})


def test_request_hash_is_stable_and_sensitive():
    a = DownloadRequest(source="s", key="k", relative_path="p", url="https://x/1")
    b = DownloadRequest(source="s", key="k", relative_path="p", url="https://x/1")
    c = DownloadRequest(source="s", key="k", relative_path="p", url="https://x/2")
    assert a.request_hash() == b.request_hash()
    assert a.request_hash() != c.request_hash()


def test_request_hash_ignores_relative_path():
    """The hash describes what was asked for, not where it was filed."""
    a = DownloadRequest(source="s", key="k", relative_path="one", url="https://x/1")
    b = DownloadRequest(source="s", key="k", relative_path="two", url="https://x/1")
    assert a.request_hash() == b.request_hash()


# ---------------------------------------------------------------------------
# StationInventory
# ---------------------------------------------------------------------------

def _inventory_row(**overrides):
    row = {column: "" for column in INVENTORY_COLUMNS}
    row.update({"source": "ghcnd", "source_station_id": "S1", "n_obs_period": 365})
    row.update(overrides)
    return row


def test_inventory_rejects_missing_columns():
    with pytest.raises(SchemaError):
        StationInventory(pd.DataFrame({"source": ["ghcnd"]}))


def test_inventory_orders_columns_to_the_contract():
    inventory = StationInventory.from_records([_inventory_row()])
    assert list(inventory.frame.columns) == list(INVENTORY_COLUMNS)


def test_inventory_counts_distinct_stations():
    inventory = StationInventory.from_records(
        [
            _inventory_row(source_station_id="S1", variable="tasmax"),
            _inventory_row(source_station_id="S1", variable="tasmin"),
            _inventory_row(source_station_id="S2", variable="tasmax"),
        ]
    )
    assert inventory.n_stations() == 2


def test_inventory_station_years_uses_measured_days():
    inventory = StationInventory.from_records(
        [_inventory_row(n_obs_period=365), _inventory_row(n_obs_period=365)]
    )
    assert inventory.station_years() == pytest.approx(730 / 365.25)


def test_inventory_round_trips_through_csv(tmp_path):
    original = StationInventory.from_records([_inventory_row(name="ATHENS")])
    path = tmp_path / "inv.csv"
    original.to_csv(path)
    assert StationInventory.read_csv(path).frame.loc[0, "name"] == "ATHENS"


def test_inventory_csv_preserves_leading_zero_station_ids(tmp_path):
    """
    GSOD '03772099999' and ECA&D '000247' must survive the round trip: numeric
    inference would strip the zeros and send the fetch at a nonexistent file.
    """
    original = StationInventory.from_records(
        [
            _inventory_row(source_station_id="03772099999"),
            _inventory_row(source_station_id="000247"),
        ]
    )
    path = tmp_path / "inv.csv"
    original.to_csv(path)
    ids = list(StationInventory.read_csv(path).frame["source_station_id"])
    assert ids == ["03772099999", "000247"]


def test_inventory_csv_preserves_numeric_wmo_ids(tmp_path):
    original = StationInventory.from_records([_inventory_row(wmo_id="16716")])
    path = tmp_path / "inv.csv"
    original.to_csv(path)
    assert StationInventory.read_csv(path).frame.loc[0, "wmo_id"] == "16716"


# ---------------------------------------------------------------------------
# Transfer layer
# ---------------------------------------------------------------------------

def test_download_writes_file_and_reports_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(net, "_with_retries", lambda operation, description: operation())
    monkeypatch.setattr(net, "_open", lambda *a, **k: _FakeResponse(b"payload"))
    destination = tmp_path / "out.bin"
    assert net.download_to("https://example/x", destination) == "downloaded"
    assert destination.read_bytes() == b"payload"


def test_download_skips_when_checksum_matches(tmp_path, monkeypatch):
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"payload")
    checksum = net.sha256_bytes(b"payload")

    def explode(*args, **kwargs):
        raise AssertionError("must not transfer when the checksum already matches")

    monkeypatch.setattr(net, "_open", explode)
    assert net.download_to("https://example/x", destination, checksum) == "skipped_checksum_match"


def test_download_refetches_when_checksum_differs(tmp_path, monkeypatch):
    """A truncated file from an interrupted run must not be reused."""
    destination = tmp_path / "out.bin"
    destination.write_bytes(b"trunc")
    monkeypatch.setattr(net, "_with_retries", lambda operation, description: operation())
    monkeypatch.setattr(net, "_open", lambda *a, **k: _FakeResponse(b"payload"))
    status = net.download_to("https://example/x", destination, net.sha256_bytes(b"payload"))
    assert status == "downloaded"
    assert destination.read_bytes() == b"payload"


def test_download_raises_and_cleans_up_on_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(net, "_with_retries", lambda operation, description: operation())
    monkeypatch.setattr(net, "_open", lambda *a, **k: _FakeResponse(b"wrong"))
    destination = tmp_path / "out.bin"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        net.download_to("https://example/x", destination, net.sha256_bytes(b"right"))
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_retries_are_bounded_and_then_raise(monkeypatch):
    monkeypatch.setattr(net.config, "HTTP_RETRIES", 3)
    monkeypatch.setattr(net.config, "HTTP_BACKOFF_BASE", 0.0)
    attempts = []

    def failing():
        attempts.append(1)
        raise TimeoutError("slow")

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        net._with_retries(failing, "test")
    assert len(attempts) == 3


def test_permanent_errors_are_not_retried(monkeypatch):
    from urllib.error import HTTPError

    monkeypatch.setattr(net.config, "HTTP_BACKOFF_BASE", 0.0)
    attempts = []

    def missing():
        attempts.append(1)
        raise HTTPError("https://x", 404, "Not Found", {}, None)

    with pytest.raises(HTTPError):
        net._with_retries(missing, "test")
    assert len(attempts) == 1


def test_access_refusal_surfaces_as_access_error(monkeypatch):
    from urllib.error import HTTPError

    from observations.types import AccessError

    def forbidden():
        raise HTTPError("https://x", 403, "Forbidden", {}, None)

    with pytest.raises(AccessError):
        net._with_retries(forbidden, "test")


def test_free_space_check_refuses_a_full_drive(tmp_path, monkeypatch):
    import shutil as shutil_module
    from collections import namedtuple

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil_module, "disk_usage", lambda p: Usage(1, 1, 64 * 1024**2))
    with pytest.raises(RuntimeError, match="below the"):
        net.check_free_space(tmp_path, required_bytes=2 * 1024**3)


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data)), "Accept-Ranges": "bytes"}

    def read(self, *args):
        return self._buffer.read(*args)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# Ranged ZIP reader
# ---------------------------------------------------------------------------

def _build_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def served_zip(monkeypatch):
    """Serve an in-memory ZIP through the net layer's range interface."""
    members = {
        "date_timestamp.txt": b"12-08-2026 03:15",
        "stations.txt": b"station metadata",
        "TG_STAID000247.txt": b"series for Athens" * 40,
        "TG_STAID000248.txt": b"series for Heraklion" * 40,
    }
    blob = _build_zip(members)

    monkeypatch.setattr(net, "content_length", lambda url: len(blob))
    monkeypatch.setattr(net, "supports_ranges", lambda url: True)
    monkeypatch.setattr(
        net,
        "get_bytes",
        lambda url, byte_range=None: blob
        if byte_range is None
        else blob[byte_range[0] : byte_range[0] + byte_range[1]],
    )
    return blob, members


def test_remote_zip_lists_members(served_zip):
    _, members = served_zip
    archive = ziprange.RemoteZip("https://example/a.zip")
    assert set(archive.namelist()) == set(members)


def test_remote_zip_reads_one_member_without_the_whole_archive(served_zip):
    _, members = served_zip
    archive = ziprange.RemoteZip("https://example/a.zip")
    assert archive.read("TG_STAID000247.txt") == members["TG_STAID000247.txt"]


def test_remote_zip_fetch_range_is_inside_the_file(served_zip):
    blob, _ = served_zip
    archive = ziprange.RemoteZip("https://example/a.zip")
    offset, length = archive.fetch_range("TG_STAID000248.txt")
    assert 0 < offset < len(blob)
    assert offset + length <= len(blob)


def test_remote_zip_raises_on_unknown_member(served_zip):
    archive = ziprange.RemoteZip("https://example/a.zip")
    with pytest.raises(KeyError):
        archive.read("TG_STAID999999.txt")


def test_remote_zip_detects_corruption():
    """
    A mis-offset range request returns plausible bytes that inflate fine.
    Only the CRC recorded in the central directory catches it, which is why
    read() checks it rather than trusting the transfer.
    """
    import zlib

    payload = b"the wrong series entirely"
    entry = ziprange.ZipEntry(
        name="TG_STAID000247.txt",
        compress_type=0,                 # stored, so no inflate step to fail first
        compressed_size=len(payload),
        uncompressed_size=len(payload),
        header_offset=0,
        crc32=zlib.crc32(b"the right series") & 0xFFFFFFFF,
    )
    with pytest.raises(SchemaError, match="CRC mismatch"):
        ziprange.RemoteZip._inflate(entry, payload)


def test_remote_zip_accepts_a_member_whose_crc_matches():
    import zlib

    payload = b"the right series"
    entry = ziprange.ZipEntry(
        name="TG_STAID000247.txt",
        compress_type=0,
        compressed_size=len(payload),
        uncompressed_size=len(payload),
        header_offset=0,
        crc32=zlib.crc32(payload) & 0xFFFFFFFF,
    )
    assert ziprange.RemoteZip._inflate(entry, payload) == payload


def test_remote_zip_rejects_unknown_compression():
    entry = ziprange.ZipEntry(
        name="x.txt",
        compress_type=99,
        compressed_size=1,
        uncompressed_size=1,
        header_offset=0,
        crc32=0,
    )
    with pytest.raises(SchemaError, match="unsupported compression"):
        ziprange.RemoteZip._inflate(entry, b"x")


def test_remote_zip_requires_range_support(monkeypatch):
    monkeypatch.setattr(net, "content_length", lambda url: 100)
    monkeypatch.setattr(net, "supports_ranges", lambda url: False)
    with pytest.raises(SchemaError, match="Accept-Ranges"):
        ziprange.RemoteZip("https://example/a.zip")


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def test_release_is_recorded_with_licence():
    manifests.record_release(
        SourceRelease(
            source="ecad",
            release="12-08-2026",
            retrieved_at="2026-09-13T00:00:00+00:00",
            licence="Non-commercial research and education only",
            licence_url="https://www.ecad.eu/documents/ECAD_datapolicy.pdf",
            redistribution="Not redistributable.",
        )
    )
    records = manifests.read_releases()
    assert records[0]["licence"].startswith("Non-commercial")


def test_requests_are_manifested_before_transfer():
    request = DownloadRequest(source="ghcnd", key="k", relative_path="p", url="https://x")
    assert manifests.record_requests("ghcnd", "3.34", [request]) == 1
    lines = manifests.requests_path("ghcnd", "3.34").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["key"] == "k"


def test_credentials_never_reach_a_manifest():
    payload = {"dataset": "x", "api_key": "SECRET", "nested": {"token": "SECRET"}}
    redacted = manifests.redact_payload(payload)
    assert redacted["api_key"] == "<redacted>"
    assert redacted["nested"]["token"] == "<redacted>"
    assert redacted["dataset"] == "x"
    assert "SECRET" not in json.dumps(redacted)


def test_snapshot_payload_is_redacted_on_disk():
    request = DownloadRequest(
        source="c3s_land",
        key="c3s_land/1995",
        relative_path="1995/x.zip",
        payload={"year": "1995", "key": "SECRET"},
    )
    manifests.record_snapshot(
        SourceSnapshot(
            request=request,
            path=Path("x.zip"),
            sha256="abc",
            n_bytes=1,
            retrieved_at="2026-09-13T00:00:00+00:00",
            release="3_0_0-daily",
            status="downloaded",
        )
    )
    text = manifests.snapshots_path("c3s_land", "3_0_0-daily").read_text(encoding="utf-8")
    assert "SECRET" not in text


def test_known_checksums_drive_idempotency():
    request = DownloadRequest(source="ghcnd", key="ghcnd/S1", relative_path="p", url="https://x")
    manifests.record_snapshot(
        SourceSnapshot(
            request=request,
            path=Path("p"),
            sha256="deadbeef",
            n_bytes=1,
            retrieved_at="2026-09-13T00:00:00+00:00",
            release="3.34",
            status="downloaded",
        )
    )
    assert manifests.known_checksums("ghcnd", "3.34") == {"ghcnd/S1": "deadbeef"}


def test_archive_checksum_is_order_independent():
    """Plan 1.8: an unchanged snapshot set must reproduce the same checksum."""

    def record(keys):
        for key in keys:
            manifests.record_snapshot(
                SourceSnapshot(
                    request=DownloadRequest(
                        source="ghcnd", key=key, relative_path=key, url="https://x"
                    ),
                    path=Path(key),
                    sha256=f"hash-{key}",
                    n_bytes=1,
                    retrieved_at="2026-09-13T00:00:00+00:00",
                    release="rel",
                    status="downloaded",
                )
            )

    record(["b", "a", "c"])
    first = manifests.archive_checksum("ghcnd", "rel")
    manifests.snapshots_path("ghcnd", "rel").unlink()
    record(["c", "b", "a"])
    assert manifests.archive_checksum("ghcnd", "rel") == first
