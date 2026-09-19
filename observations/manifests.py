"""
manifests.py
============
Provenance records: what was asked for, what came back, and under whose licence.

Plan section 1.7 lists 'source_releases' as a durable table holding "source,
version, retrieval time, licence, checksum and request manifest".  This module
owns that table and the per-request manifest beside it.

Manifests are append-only JSON Lines.  A run appends; it never rewrites
history, so an archive carries the full record of how it came to exist even
when a provider release changed mid-project.

Plan section 1.5: "Credentials, if required by Copernicus, belong in
environment configuration and never in the manifest or repository."
redact_payload() enforces that for the CDS adapter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from observations import config
from observations.types import DownloadRequest, SourceRelease, SourceSnapshot

# Payload keys that must never reach a manifest file, matched case-insensitively
# as substrings so 'cds_api_key' and 'Authorization' are both caught.
_SECRET_KEY_MARKERS = ("key", "token", "password", "secret", "credential", "authorization")

_REDACTED = "<redacted>"


def redact_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Replace any credential-looking value with a placeholder, recursively."""
    if payload is None:
        return None
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS):
            clean[key] = _REDACTED
        elif isinstance(value, dict):
            clean[key] = redact_payload(value)
        else:
            clean[key] = value
    return clean


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())

    def generate() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return generate()


def source_releases_path() -> Path:
    return config.archive_root() / "manifests" / "source_releases.jsonl"


def requests_path(source: str, release: str) -> Path:
    return config.manifest_dir(source, release) / "requests.jsonl"


def snapshots_path(source: str, release: str) -> Path:
    return config.manifest_dir(source, release) / "snapshots.jsonl"


def record_release(release: SourceRelease) -> None:
    """Append one source_releases row."""
    _append_jsonl(source_releases_path(), release.to_dict())


def record_requests(source: str, release: str, requests: Iterable[DownloadRequest]) -> int:
    """
    Write the request manifest.

    Plan section 1.5 requires this before transfer, not after: a run that dies
    partway must still leave evidence of everything it intended to fetch, or
    the gap between plan and archive cannot be audited.
    """
    path = requests_path(source, release)
    count = 0
    for request in requests:
        record = request.to_dict()
        record["payload"] = redact_payload(record.get("payload"))
        _append_jsonl(path, record)
        count += 1
    return count


def record_snapshot(snapshot: SourceSnapshot) -> None:
    """Append one snapshot row once its bytes are on disk and checksummed."""
    record = snapshot.to_dict()
    record["payload"] = redact_payload(record.get("payload"))
    _append_jsonl(snapshots_path(snapshot.request.source, snapshot.release), record)


def known_checksums(source: str, release: str) -> dict[str, str]:
    """
    Map request key -> recorded sha256 for snapshots already in the archive.

    This is what makes a re-run idempotent: a request whose key is here and
    whose file still hashes to the recorded value is skipped, while a file
    that no longer matches is fetched again.
    """
    return {
        record["key"]: record["sha256"]
        for record in _read_jsonl(snapshots_path(source, release))
        if record.get("key") and record.get("sha256")
    }


def read_releases() -> list[dict[str, Any]]:
    return list(_read_jsonl(source_releases_path()))


def archive_checksum(source: str, release: str) -> str:
    """
    Single checksum over a source's snapshot set.

    Plan section 1.8: "Re-running consolidation from unchanged snapshots
    produces the identical archive checksum."  Consolidation computes its own,
    but the acquisition layer needs one too, so a downstream step can assert
    that the bronze layer it is reading is the one it was tested against.

    Computed from sorted (key, sha256) pairs, so it is independent of the
    order in which files happened to be downloaded.
    """
    import hashlib

    pairs = sorted(known_checksums(source, release).items())
    digest = hashlib.sha256()
    for key, checksum in pairs:
        digest.update(f"{key}:{checksum}\n".encode("utf-8"))
    return digest.hexdigest()
