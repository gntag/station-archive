"""
sources/base.py
===============
The source-adapter contract from RESEARCH_PLAN_AND_DESIGN.md section 1.5:

    class StationSource:
        def discover(self, domain, variables, period) -> StationInventory: ...
        def plan(self, inventory) -> list[DownloadRequest]: ...
        def fetch(self, request, raw_directory) -> SourceSnapshot: ...
        def parse(self, snapshot) -> list[RawObservation]: ...

Implemented as written, with one documented deviation: parse() returns an
Iterator[RawObservation] rather than a list.  The element type is unchanged;
materialising every value in the domain as a dataclass instance at once would
need tens of millions of objects, so parsing streams instead.  Callers that
genuinely want a list can wrap the call.

The guardrail this file defends: "Source adapters fetch and parse provider
data; they do not decide station identity or source priority."  No method here
may compare stations across sources, rank them, convert units or fill gaps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Sequence

from observations import config, manifests, net
from observations.types import (
    Domain,
    DownloadRequest,
    RawObservation,
    SourceRelease,
    SourceSnapshot,
    StationInventory,
)


class StationSource(ABC):
    """
    One provider behind a stable interface.

    Subclasses set 'name' and the licence fields, then implement the four
    contract methods.  Everything shared -- idempotent fetching, checksum
    bookkeeping, manifest writing -- lives here so an adapter cannot
    accidentally opt out of provenance.
    """

    name: str = ""
    licence: str = ""
    licence_url: str = ""
    redistribution: str = ""

    # ---- contract -------------------------------------------------------

    @abstractmethod
    def release(self) -> str:
        """
        Provider release/version identifier for this acquisition.

        Prefer a value the provider itself publishes (GHCN-D's
        ghcnd-version.txt, ECA&D's date_timestamp.txt, the CDS dataset
        version).  Only fall back to a retrieval date when the provider
        publishes nothing, and say so in the release string.
        """

    @abstractmethod
    def discover(
        self,
        domain: Domain,
        variables: Sequence[str],
        period: tuple[int, int],
    ) -> StationInventory:
        """
        Coverage-only query: which stations and variables this source offers
        inside 'domain', and how complete they are over 'period'.

        Plan section 1.2 requires this before bulk download, so the value of a
        source can be judged before its bytes are transferred.
        """

    @abstractmethod
    def plan(self, inventory: StationInventory) -> list[DownloadRequest]:
        """Turn a (possibly filtered) inventory into concrete transfers."""

    @abstractmethod
    def _transfer(self, request: DownloadRequest, destination: Path) -> None:
        """
        Move the bytes for one request to 'destination'.

        Adapter-specific; fetch() wraps it with the provenance rules that
        apply to every source.
        """

    @abstractmethod
    def parse(self, snapshot: SourceSnapshot) -> Iterator[RawObservation]:
        """
        Read a snapshot into untouched provider values.

        Must raise SchemaError if the file no longer matches the expected
        structure -- plan section 1.5 wants a loud stop, not silently dropped
        columns.
        """

    # ---- shared behaviour ------------------------------------------------

    def source_release(self) -> SourceRelease:
        return SourceRelease(
            source=self.name,
            release=self.release(),
            retrieved_at=net.utc_now(),
            licence=self.licence,
            licence_url=self.licence_url,
            redistribution=self.redistribution,
        )

    def fetch(
        self,
        request: DownloadRequest,
        raw_directory: Path,
        known: dict[str, str] | None = None,
    ) -> SourceSnapshot:
        """
        Execute one request idempotently and return its snapshot.

        'known' maps request key -> previously recorded checksum.  A file is
        reused only when it exists AND still hashes to that value; existence
        alone is not enough, because an interrupted transfer leaves a file
        that looks complete.
        """
        release = self.release()
        destination = raw_directory / request.relative_path

        expected = (known or {}).get(request.key)
        if expected and destination.exists() and net.sha256_file(destination) == expected:
            return SourceSnapshot(
                request=request,
                path=destination,
                sha256=expected,
                n_bytes=destination.stat().st_size,
                retrieved_at=net.utc_now(),
                release=release,
                status="skipped_checksum_match",
            )

        self._transfer(request, destination)

        return SourceSnapshot(
            request=request,
            path=destination,
            sha256=net.sha256_file(destination),
            n_bytes=destination.stat().st_size,
            retrieved_at=net.utc_now(),
            release=release,
            status="downloaded",
        )

    def fetch_all(
        self,
        requests: Sequence[DownloadRequest],
        progress: bool = True,
    ) -> list[SourceSnapshot]:
        """
        Run a planned set of requests, writing manifests around the transfer.

        Individual permanent failures (a station absent for a year, say) are
        recorded and skipped rather than aborting the run; anything else
        propagates, because a silent partial archive is the failure mode the
        plan's provenance rules exist to prevent.
        """
        release = self.release()
        raw_directory = config.raw_dir(self.name, release)
        net.check_free_space(raw_directory)

        manifests.record_release(self.source_release())
        manifests.record_requests(self.name, release, requests)
        known = manifests.known_checksums(self.name, release)

        snapshots: list[SourceSnapshot] = []
        failures: list[tuple[str, str]] = []

        for index, request in enumerate(requests, start=1):
            try:
                snapshot = self.fetch(request, raw_directory, known)
            except Exception as exc:  # recorded, not swallowed: see summary below
                failures.append((request.key, str(exc)))
                if progress:
                    print(f"[{self.name}] {index}/{len(requests)} FAILED {request.key}: {exc}")
                continue

            manifests.record_snapshot(snapshot)
            snapshots.append(snapshot)
            if progress:
                print(
                    f"[{self.name}] {index}/{len(requests)} "
                    f"{snapshot.status} {request.key} ({snapshot.n_bytes} bytes)"
                )

        if progress:
            print(
                f"[{self.name}] release={release} "
                f"ok={len(snapshots)} failed={len(failures)} "
                f"archive_checksum={manifests.archive_checksum(self.name, release)}"
            )
            for key, message in failures[:20]:
                print(f"[{self.name}] failure: {key}: {message}")

        return snapshots
