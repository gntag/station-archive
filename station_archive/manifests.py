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
from typing import Any, Iterable, Iterator, Mapping, Optional

from station_archive import settings
from station_archive.types import DownloadRequest, SourceRelease, SourceSnapshot

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
    return settings.archive_root() / "manifests" / "source_releases.jsonl"


def requests_path(source: str, release: str) -> Path:
    return settings.manifest_dir(source, release) / "requests.jsonl"


def snapshots_path(source: str, release: str) -> Path:
    return settings.manifest_dir(source, release) / "snapshots.jsonl"


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


def build_provenance(policy: Any, sources: Iterable[str]) -> dict[str, Any]:
    """
    One record describing what built this archive.

    The per-request and per-snapshot manifests already say how every file
    arrived, but nothing said what the archive AS A WHOLE was produced by --
    which policy, which provider releases, which library versions. A paper
    citing this archive needs one artefact it can point at, and a reader
    checking whether two archives are comparable needs to diff something
    smaller than the archive.

    Records only what is already on disk; it computes nothing and changes
    nothing, so it is safe to run at any time.
    """
    import platform
    import sys

    from station_archive import __version__

    releases = read_releases()
    per_source: dict[str, Any] = {}
    for source in sources:
        rows = [r for r in releases if r.get("source") == source]
        if not rows:
            per_source[source] = {"status": "not acquired"}
            continue
        newest = max(rows, key=lambda r: r.get("retrieved_at", ""))
        release = newest["release"]
        checksums = known_checksums(source, release)
        per_source[source] = {
            "release": release,
            "retrieved_at": newest.get("retrieved_at"),
            "licence": newest.get("licence"),
            "redistribution": newest.get("redistribution"),
            "n_snapshots": len(checksums),
            "archive_checksum": archive_checksum(source, release),
        }

    def version_of(name: str) -> str:
        try:
            module = __import__(name)
        except Exception:
            return "not installed"
        return getattr(module, "__version__", "unknown")

    return {
        "generated_at": _utc_now(),
        "station_archive_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # The libraries this package actually imports. It used to record
        # sklearn -- left over from fold_manifest.py, which moved out -- and
        # omitted PIL, which dem.py needs to decode elevation tiles. A
        # provenance record that names a library the run never used, and
        # misses one it did, is not a reproducibility record.
        "libraries": {
            name: version_of(name)
            for name in ("numpy", "pandas", "pyarrow", "openpyxl", "PIL")
        },
        "policy": policy.to_dict(),
        "sources": per_source,
        # The climatology screen's reference is an input the archive depends
        # on -- it decides which series are proposed for exclusion -- without
        # being a source. Two archives built from the same snapshots but
        # screened against different references must not share a record.
        "climatology_reference": read_reference_record() or {"status": "not used"},
    }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def provenance_path() -> Path:
    return settings.archive_root() / "manifests" / "provenance.json"


def reference_record_path() -> Path:
    return settings.archive_root() / "manifests" / "climatology_reference.json"


def _file_record(path: Optional[Path]) -> dict[str, Any]:
    """
    A file identified by name, size and sha256 -- never by its full path.

    The path is where the file sits on one machine; the checksum is what the
    file is, and the only part of the record another reader can check.
    """
    import hashlib

    if path is None:
        return {"status": "not given"}
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "file": Path(path).name,
        "bytes": Path(path).stat().st_size,
        "sha256": digest.hexdigest(),
    }


def record_reference(files: Mapping[str, Optional[Path]]) -> dict[str, Any]:
    """
    Record which reference files the climatology screen ran on, and the
    screen it produced, for build_provenance to carry.

    Written by the screen itself, at the moment it reads the files, so the
    record describes what was realised rather than what a later command was
    told. Replaced on every screen run: provenance describes the screen behind
    the current proposals, not every screen ever run.
    """
    record: dict[str, Any] = {
        "recorded_at": _utc_now(),
        **{role: _file_record(path) for role, path in files.items()},
    }
    path = reference_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def read_reference_record() -> Optional[dict[str, Any]]:
    path = reference_record_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
