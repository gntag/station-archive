"""
ziprange.py
===========
Read individual members out of a remote ZIP archive over HTTP byte ranges,
without downloading the archive.

WHY THIS EXISTS
---------------
ECA&D publishes its daily station series only as whole-element archives:
ECA_blend_tg.zip (881 MB), ECA_blend_rr.zip (1.84 GB), and five more -- about
5.4 GB compressed for the seven elements this project needs, and far more
unpacked.  Of the ~3000 stations in each archive, only the few dozen inside
the eastern-Mediterranean domain are wanted.

A ZIP file stores its index (the "central directory") at the end, and each
member's compressed bytes at a known offset.  Given a server that honours
Range requests -- the ECA&D S3 bucket does -- the index can be read from the
tail and then each wanted member fetched individually.  This turns a 5.4 GB
transfer into a few megabytes.

The alternative, downloading each archive and extracting a handful of files,
would need ~5.4 GB of transient space on a machine whose repository drive has
64 MB free.

FORMAT REFERENCE
----------------
PKWARE APPNOTE, sections 4.3.6 (local header), 4.3.12 (central directory) and
4.3.16 (end of central directory).  Only the fields needed to locate and
inflate a member are parsed; ZIP64 is detected and rejected loudly rather
than misread.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Optional

from station_archive import net
from station_archive.types import SchemaError

_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"

# The end-of-central-directory record is 22 bytes plus a comment of up to
# 65535 bytes, so this much of the tail always contains it.
_EOCD_SEARCH_BYTES = 66000

# 0xFFFFFFFF in a size or offset field means "see the ZIP64 extra field".
_ZIP64_SENTINEL = 0xFFFFFFFF

_STORED = 0
_DEFLATED = 8

# Slack allowed for a local header's variable-length extra field, so a member
# can be fetched in one range request instead of two.  ZIP extra fields are
# typically tens of bytes; this is generous without being wasteful.
_EXTRA_FIELD_SLACK = 4096


@dataclass(frozen=True)
class ZipEntry:
    """One member of the remote archive, as described by the central directory."""

    name: str
    compress_type: int
    compressed_size: int
    uncompressed_size: int
    header_offset: int
    crc32: int


class RemoteZip:
    """
    Random access to a remote ZIP over HTTP range requests.

    The central directory is read once on construction; each read() then costs
    a single further range request in the common case.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.size = net.content_length(url)
        if not net.supports_ranges(url):
            raise SchemaError(
                f"{url} does not advertise Accept-Ranges: bytes; "
                f"ranged member extraction is not possible"
            )
        self.entries: dict[str, ZipEntry] = self._read_central_directory()

    def _read_central_directory(self) -> dict[str, ZipEntry]:
        tail_length = min(_EOCD_SEARCH_BYTES, self.size)
        tail = net.get_bytes(self.url, (self.size - tail_length, tail_length))

        position = tail.rfind(_EOCD_SIGNATURE)
        if position < 0:
            raise SchemaError(f"no end-of-central-directory record in {self.url}")

        eocd = tail[position : position + 22]
        directory_size = struct.unpack("<I", eocd[12:16])[0]
        directory_offset = struct.unpack("<I", eocd[16:20])[0]

        if directory_offset == _ZIP64_SENTINEL or directory_size == _ZIP64_SENTINEL:
            raise SchemaError(
                f"{self.url} is a ZIP64 archive; this reader handles only "
                f"classic ZIP offsets"
            )

        directory = net.get_bytes(self.url, (directory_offset, directory_size))
        return self._parse_central_directory(directory)

    @staticmethod
    def _parse_central_directory(directory: bytes) -> dict[str, ZipEntry]:
        entries: dict[str, ZipEntry] = {}
        position = 0
        while position + 46 <= len(directory):
            if directory[position : position + 4] != _CENTRAL_SIGNATURE:
                break
            compress_type = struct.unpack("<H", directory[position + 10 : position + 12])[0]
            crc32 = struct.unpack("<I", directory[position + 16 : position + 20])[0]
            compressed_size, uncompressed_size = struct.unpack(
                "<II", directory[position + 20 : position + 28]
            )
            name_length, extra_length, comment_length = struct.unpack(
                "<HHH", directory[position + 28 : position + 34]
            )
            header_offset = struct.unpack("<I", directory[position + 42 : position + 46])[0]

            if header_offset == _ZIP64_SENTINEL or compressed_size == _ZIP64_SENTINEL:
                raise SchemaError("ZIP64 member encountered; not supported")

            name = directory[position + 46 : position + 46 + name_length].decode(
                "utf-8", errors="replace"
            )
            entries[name] = ZipEntry(
                name=name,
                compress_type=compress_type,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                header_offset=header_offset,
                crc32=crc32,
            )
            position += 46 + name_length + extra_length + comment_length

        if not entries:
            raise SchemaError("central directory contained no members")
        return entries

    def namelist(self) -> list[str]:
        return list(self.entries)

    def fetch_range(self, name: str) -> tuple[int, int]:
        """
        Byte range that read() will request for this member, computed offline.

        A member's data does not start at its local header: the header is 30
        bytes plus a variable-length name and extra field, and only the name
        length is knowable from the central directory.  Rather than spend an
        HTTP request discovering the extra-field length -- which would double
        the request count across a thousand stations -- the range covers the
        header plus _EXTRA_FIELD_SLACK, and read() parses the real offset out
        of the bytes that come back.

        Clamped to the file size so the last member does not request past the
        end of the archive.
        """
        entry = self._entry(name)
        length = 30 + len(name.encode("utf-8")) + _EXTRA_FIELD_SLACK + entry.compressed_size
        return entry.header_offset, min(length, self.size - entry.header_offset)

    def read(self, name: str) -> bytes:
        """
        Fetch and decompress one member, verifying its CRC.

        One range request in the common case: the local header and the member
        data arrive together and the header is parsed from the front of the
        response.  A second, exact request is issued only if the extra field
        turns out to be larger than the slack allowed for.

        The CRC check matters more here than for a whole-file download: a
        truncated or mis-offset range request yields plausible-looking bytes
        that would otherwise be stored as a valid snapshot.
        """
        entry = self._entry(name)
        offset, length = self.fetch_range(name)
        block = net.get_bytes(self.url, (offset, length))

        if block[:4] != _LOCAL_SIGNATURE:
            raise SchemaError(f"no local header for {name!r} at {entry.header_offset}")
        name_length, extra_length = struct.unpack("<HH", block[26:30])
        data_start = 30 + name_length + extra_length
        data_end = data_start + entry.compressed_size

        if data_end > len(block):
            # Unusually large extra field; fetch the exact range instead.
            raw = net.get_bytes(
                self.url, (entry.header_offset + data_start, entry.compressed_size)
            )
        else:
            raw = block[data_start:data_end]

        return self._inflate(entry, raw)

    @staticmethod
    def _inflate(entry: ZipEntry, raw: bytes) -> bytes:
        if entry.compress_type == _DEFLATED:
            data = zlib.decompress(raw, -zlib.MAX_WBITS)
        elif entry.compress_type == _STORED:
            data = raw
        else:
            raise SchemaError(
                f"member {entry.name!r} uses unsupported compression "
                f"method {entry.compress_type}"
            )
        if zlib.crc32(data) & 0xFFFFFFFF != entry.crc32:
            raise SchemaError(f"CRC mismatch for member {entry.name!r}")
        return data

    def _entry(self, name: str) -> ZipEntry:
        entry: Optional[ZipEntry] = self.entries.get(name)
        if entry is None:
            raise KeyError(f"{name!r} is not in {self.url}")
        return entry
