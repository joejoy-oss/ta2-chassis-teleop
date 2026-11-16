#!/usr/bin/env python3
"""The .rtpcap file: RTP packets with the time each was received or is to be sent.

rtp_capture.py writes one from a UDP port, make_synthetic.sh builds them from an encoder's RTP output with
synthesized send times; rtp_replay.py sends them again and stream_info.py analyses them.

Layout:
    b"RTPCAP1\\n"
    one line of UTF-8 JSON metadata ending in b"\\n" (source, port, clock, generator settings, ...)
    then per packet: struct "<QI" (time in ns, packet length) followed by the packet (RTP header included)

Times are time.monotonic_ns() on the capturing host, or synthesized send times (metadata "clock" says which); only
differences between them mean anything. Pure Python, no gi.

Command line (used by make_synthetic.sh and the closed-loop checks):
    python3 rtpcap.py from-rfc4571 IN.rtpstream OUT.rtpcap [--fps 45] [--line-rate-bps 1e9] [--meta KEY=VALUE ...]
    python3 rtpcap.py digest FILE.rtpcap ...      # packet count and packets_sha256 per file, as JSON lines
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import datetime
import hashlib
import json
import os
import struct
import sys
from typing import BinaryIO

MAGIC = b"RTPCAP1\n"
RECORD = struct.Struct("<QI")
MAX_PACKET = 65535
CLOCK_MONOTONIC = "monotonic_ns"
CLOCK_SYNTHETIC = "synthetic_send_ns"


def utc_now() -> str:
    """Current UTC time for metadata, e.g. 2026-11-03T07:40:00Z."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017 (3.10)


class Writer:
    """Writes a .rtpcap file: `meta` goes into the header line, then write(time_ns, packet) per packet."""

    def __init__(self, path: str | os.PathLike, meta: dict, *, buffer_size: int = 1 << 20):
        line = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode()
        self._file = open(path, "wb", buffering=buffer_size)  # noqa: SIM115 (closed in close())
        self._file.write(MAGIC + line + b"\n")
        self._pack = RECORD.pack
        self.packets = 0

    def write(self, time_ns: int, packet: bytes | bytearray | memoryview) -> None:
        self._file.write(self._pack(time_ns, len(packet)))
        self._file.write(packet)
        self.packets += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class Reader:
    """Reads a .rtpcap file: `meta` is the header's metadata; iterating yields (time_ns, packet) in file order.

    A file that ends inside a record (the writer was killed) raises ValueError at that point.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._file = open(self.path, "rb")  # noqa: SIM115 (closed in close())
        try:
            if self._file.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{self.path}: not an rtpcap file")
            line = self._file.readline()
            if not line.endswith(b"\n"):
                raise ValueError(f"{self.path}: truncated metadata line")
            self.meta: dict = json.loads(line)
        except Exception:
            self._file.close()
            raise

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        read = self._file.read
        size = RECORD.size
        unpack = RECORD.unpack
        while True:
            head = read(size)
            if not head:
                return
            if len(head) < size:
                raise ValueError(f"{self.path}: truncated record header at the end")
            time_ns, length = unpack(head)
            packet = read(length)
            if len(packet) < length:
                raise ValueError(f"{self.path}: truncated packet at the end ({len(packet)} of {length} bytes)")
            yield time_ns, packet

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Reader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_meta(path: str | os.PathLike) -> dict:
    with Reader(path) as reader:
        return reader.meta


def iter_packets(path: str | os.PathLike) -> Iterator[tuple[int, bytes]]:
    """(time_ns, packet) for every packet of the file."""
    with Reader(path) as reader:
        yield from reader


def packets_sha256(packets: Iterable[bytes]) -> str:
    """sha256 over the packets in order, each prefixed by its length (4 bytes, big endian). Times do not enter, so a
    capture and its replayed-and-recaptured copy hash the same when nothing was lost, reordered or changed."""
    digest = hashlib.sha256()
    for packet in packets:
        digest.update(len(packet).to_bytes(4, "big"))
        digest.update(packet)
    return digest.hexdigest()


def file_digest(path: str | os.PathLike) -> dict:
    """{"packets": count, "packets_sha256": ...} of a .rtpcap file."""
    count = 0

    def packets() -> Iterator[bytes]:
        nonlocal count
        for _time_ns, packet in iter_packets(path):
            count += 1
            yield packet

    digest = packets_sha256(packets())
    return {"packets": count, "packets_sha256": digest}


def iter_rfc4571(stream: BinaryIO) -> Iterator[bytes]:
    """The packets of an RFC 4571 stream (each prefixed by a 16-bit big-endian length), as GStreamer's rtpstreampay
    writes them."""
    while True:
        head = stream.read(2)
        if not head:
            return
        if len(head) < 2:
            raise ValueError("truncated RFC 4571 length")
        length = int.from_bytes(head, "big")
        packet = stream.read(length)
        if len(packet) < length:
            raise ValueError(f"truncated RFC 4571 packet ({len(packet)} of {length} bytes)")
        yield packet


def synthesize_send_times(
    packets: Iterable[bytes],
    *,
    fps: float,
    line_rate_bps: float,
    clock_rate: int = 90_000,
    start_ns: int = 0,
) -> Iterator[tuple[int, bytes]]:
    """Give RTP packets the send times of a camera sender: access unit i (counted from the RTP timestamps, i = round(
    elapsed ticks * fps / clock_rate), so rounding in the timestamps does not matter) starts at i / fps; the packets of
    one access unit follow each other at `line_rate_bps`, counting the RTP packet bytes only. Times never go back: an
    access unit that would start before the previous one finished starts right after it."""
    first_timestamp = None
    current_timestamp = None
    next_free_ns = start_ns
    send_ns = start_ns
    for packet in packets:
        timestamp = int.from_bytes(packet[4:8], "big")
        if timestamp != current_timestamp:
            if first_timestamp is None:
                first_timestamp = timestamp
            ticks = (timestamp - first_timestamp) & 0xFFFFFFFF  # the 32-bit timestamp may wrap
            index = round(ticks * fps / clock_rate)
            send_ns = max(start_ns + round(index * 1e9 / fps), next_free_ns)
            current_timestamp = timestamp
        else:
            send_ns = next_free_ns
        yield send_ns, packet
        next_free_ns = send_ns + round(len(packet) * 8e9 / line_rate_bps)


def _from_rfc4571(args: argparse.Namespace) -> None:
    meta = {
        "format": "rtpcap",
        "version": 1,
        "clock": CLOCK_SYNTHETIC,
        "created_utc": utc_now(),
        "source": "rfc4571:" + os.path.basename(args.input),
        "send_model": {
            "fps": args.fps,
            "line_rate_bps": args.line_rate_bps,
            "bytes_counted": "RTP packet (UDP payload)",
            "description": "access unit i starts at i/fps; its packets follow back to back at the line rate",
        },
    }
    for item in args.meta:
        key, _, value = item.partition("=")
        try:
            meta[key] = json.loads(value)  # numbers, booleans, JSON objects
        except json.JSONDecodeError:
            meta[key] = value
    with open(args.input, "rb") as stream, Writer(args.output, meta) as writer:
        packets = iter_rfc4571(stream)
        for send_ns, packet in synthesize_send_times(packets, fps=args.fps, line_rate_bps=args.line_rate_bps):
            writer.write(send_ns, packet)
        print(f"{args.output}: {writer.packets} packets")


def _digest(args: argparse.Namespace) -> None:
    for path in args.files:
        print(json.dumps({"file": path, **file_digest(path)}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("from-rfc4571", help="RFC 4571 stream (rtpstreampay) -> .rtpcap with send times")
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument("--fps", type=float, default=45.0)
    convert.add_argument("--line-rate-bps", type=float, default=1e9)
    convert.add_argument(
        "--meta", action="append", default=[], help="KEY=VALUE for the file header (VALUE parsed as JSON if it is)"
    )
    convert.set_defaults(run=_from_rfc4571)
    digest = commands.add_parser("digest", help="packet count and packets_sha256 of .rtpcap files")
    digest.add_argument("files", nargs="+")
    digest.set_defaults(run=_digest)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
