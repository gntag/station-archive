"""
net.py
======
Restartable, idempotent HTTP transfer.

Plan section 1.5: "Every downloader must be restartable and idempotent:
create a request manifest before transfer; keep provider release/version, URL
or API payload, retrieval time, response checksum and licence; skip a file
only when its checksum matches; use bounded retries and provider rate limits;
and never edit a raw snapshot in place."

The checksum rule is the subtle one.  Skipping on mere file existence would
silently keep a truncated file from an interrupted run, so a snapshot is
reused only when its recorded checksum still matches the bytes on disk.
"""
from __future__ import annotations

import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from station_archive import settings
from station_archive.types import AccessError

# Last request time per host, for the rate limit.  Module-level because the
# limit is a courtesy to the provider, not a property of any one adapter.
_last_request: dict[str, float] = {}


def utc_now() -> str:
    """Retrieval timestamp in ISO-8601 UTC, recorded on every snapshot."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _throttle(url: str) -> None:
    """Hold at least HTTP_MIN_INTERVAL seconds between requests to one host."""
    host = urlsplit(url).netloc
    previous = _last_request.get(host)
    if previous is not None:
        wait = settings.HTTP_MIN_INTERVAL - (time.monotonic() - previous)
        if wait > 0:
            time.sleep(wait)
    _last_request[host] = time.monotonic()


def _open(url: str, byte_range: Optional[tuple[int, int]] = None, method: str = "GET"):
    headers = {"User-Agent": settings.USER_AGENT}
    if byte_range is not None:
        offset, length = byte_range
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
    _throttle(url)
    return urlopen(Request(url, headers=headers, method=method), timeout=settings.HTTP_TIMEOUT)


def _is_permanent(exc: Exception) -> bool:
    """
    True for errors that retrying cannot fix.

    404 means the provider does not have this station/year at all, which is a
    normal outcome when a request is planned from an inventory that is broader
    than the archive.  401/403 mean a licence or credential problem, which
    must surface rather than be retried into a timeout.
    """
    return isinstance(exc, HTTPError) and exc.code in (400, 401, 403, 404, 410)


def _with_retries(operation, description: str):
    """
    Run 'operation', retrying transient failures with exponential back-off.

    Bounded by settings.HTTP_RETRIES.  Permanent errors are re-raised at once so
    a run of 404s does not spend the whole retry budget.
    """
    last: Optional[Exception] = None
    for attempt in range(1, settings.HTTP_RETRIES + 1):
        try:
            return operation()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if _is_permanent(exc):
                if isinstance(exc, HTTPError) and exc.code in (401, 403):
                    raise AccessError(f"{description}: access refused ({exc})") from exc
                raise
            last = exc
            if attempt < settings.HTTP_RETRIES:
                time.sleep(settings.HTTP_BACKOFF_BASE * 2 ** (attempt - 1))
    raise RuntimeError(f"{description}: failed after {settings.HTTP_RETRIES} attempts: {last}")


def get_bytes(url: str, byte_range: Optional[tuple[int, int]] = None) -> bytes:
    """Fetch a URL (or one byte range of it) into memory, with retries."""

    def operation() -> bytes:
        with _open(url, byte_range) as response:
            return response.read()

    return _with_retries(operation, f"GET {url}")


def content_length(url: str) -> int:
    """
    Size of a remote file from its headers.

    Needed before a ranged read, because the ZIP central directory is located
    relative to the end of the file.
    """

    def operation() -> int:
        with _open(url, method="HEAD") as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise RuntimeError(f"no Content-Length for {url}")
            return int(length)

    return _with_retries(operation, f"HEAD {url}")


def supports_ranges(url: str) -> bool:
    """
    Whether the server advertises byte-range support.

    Checked before planning a ranged extraction: without ranges the ECA&D
    strategy silently degrades into downloading gigabytes, so the adapter
    needs to know rather than discover it by filling the disk.
    """

    def operation() -> bool:
        with _open(url, method="HEAD") as response:
            return response.headers.get("Accept-Ranges", "").lower() == "bytes"

    return _with_retries(operation, f"HEAD {url}")


def download_to(url: str, destination: Path, expected_sha256: Optional[str] = None) -> str:
    """
    Download 'url' to 'destination', returning "downloaded" or
    "skipped_checksum_match".

    Written to a .part file and renamed only on success, so an interrupted
    transfer can never be mistaken for a complete snapshot on the next run.
    """
    if expected_sha256 and destination.exists():
        if sha256_file(destination) == expected_sha256:
            return "skipped_checksum_match"

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    def operation() -> str:
        with _open(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        return "downloaded"

    try:
        status = _with_retries(operation, f"GET {url}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    if expected_sha256:
        actual = sha256_file(partial)
        if actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
            )

    partial.replace(destination)
    return status


def write_bytes_atomic(data: bytes, destination: Path) -> None:
    """Write in-memory bytes through a .part file, as download_to does."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(destination)


def check_free_space(path: Path, required_bytes: int = settings.MIN_FREE_BYTES) -> None:
    """
    Refuse to start a fetch run that the target drive cannot hold.

    The plan's hardware table records G: at ~0.1 GB free; a run that starts
    there would fail partway and leave an archive whose checksums no longer
    describe complete files.
    """
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required_bytes:
        raise RuntimeError(
            f"{path} has {free / 1024**3:.2f} GB free, "
            f"below the {required_bytes / 1024**3:.2f} GB required. "
            f"Set STATION_ARCHIVE_ROOT to a drive with room."
        )
