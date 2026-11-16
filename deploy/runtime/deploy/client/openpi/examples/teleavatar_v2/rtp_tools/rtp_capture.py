#!/usr/bin/env python3
"""Record an RTP/H.265 stream from a UDP port into a .rtpcap file and report its packet statistics.

    python3 rtp_capture.py --port 8890 --duration 60 --out robot.rtpcap --stats-json robot_capture.json

Stops after --duration seconds, after --idle-timeout seconds without packets (once packets have arrived), or on
SIGINT/SIGTERM; the file is complete in every case. Nothing else may listen on the port: the socket is bound without
SO_REUSEADDR, so a second receiver fails to bind instead of silently sharing the packets.

The receive loop only stores each packet with its time.monotonic_ns() (and its first 64 bytes in memory); the
statistics are computed afterwards: packets, bytes, packets/s, bitrate, RTP timestamp steps and the frame rate they
imply, sequence gaps and reordering (rtp_pipeline.rtp_seq_step, the same counting as the receiver), packet sizes,
single NAL unit / AP / FU shares, marker bits (access units), SSRC and payload type, and the kernel's drop counter for
the socket (/proc/net/udp), which shows packets lost to a full receive buffer.

The requested --rcvbuf is capped by net.core.rmem_max (global, not per network namespace); the actual size Linux
reports (twice the granted value) and rmem_max are in the report.
"""

from __future__ import annotations

import argparse
from array import array
import collections
import json
import os
import pathlib
import signal
import socket
import struct
import sys
import time

import h265_rtp
import rtpcap

_TELEAVATAR_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if _TELEAVATAR_DIR not in sys.path:
    sys.path.append(_TELEAVATAR_DIR)  # rtp_pipeline.py lives one directory up

import rtp_pipeline  # noqa: E402

HEAD_BYTES = 64  # bytes kept per packet for the statistics: RTP header (with CSRCs) and payload header
KINDS = ("single", "ap", "fu", "paci", "unknown")


def percentiles(values, points: tuple[float, ...] = (50, 99)) -> dict[str, float]:
    """min, the given percentiles (nearest rank) and max of `values`; empty dict for no values."""
    ordered = sorted(values)
    if not ordered:
        return {}
    result = {"min": ordered[0]}
    for point in points:
        rank = max(1, -(-len(ordered) * point // 100))  # ceil(n * p / 100)
        result[f"p{point:g}"] = ordered[int(rank) - 1]
    result["max"] = ordered[-1]
    return result


class PacketStats:
    """Packet-level statistics of one RTP/H.265 stream, built from each packet's time, first bytes and length.

    Used by this tool and by stream_info.py. Access units here are runs of packets with the same RTP timestamp, in
    arrival order.
    """

    def __init__(self, clock_rate: int = h265_rtp.RTP_CLOCK_RATE):
        self.clock_rate = clock_rate
        self.packets = 0
        self.bytes = 0
        self.invalid = 0
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self.lengths = array("I")
        self.payload_types: collections.Counter[int] = collections.Counter()
        self.ssrcs: collections.Counter[int] = collections.Counter()
        self.kinds: collections.Counter[str] = collections.Counter()
        self.single_nal_types: collections.Counter[str] = collections.Counter()
        self.fu_starts = 0
        self.fu_ends = 0
        self.markers = 0
        self.seq_missing = 0
        self.seq_out_of_order = 0
        self._last_seq: int | None = None
        self.seq_wraps = 0
        self.timestamp_steps: collections.Counter[int] = collections.Counter()
        self.first_timestamp: int | None = None
        self.last_timestamp: int | None = None
        self.access_units = 0
        self.au_bytes = array("Q")
        self.au_packets = array("I")
        self.au_span_ns = array("Q")
        self._au_timestamp: int | None = None
        self._au_first_ns = 0
        self._au_last_ns = 0
        self._au_bytes = 0
        self._au_packets = 0
        self._first_au_ns: int | None = None
        self._last_au_ns = 0

    def add(self, time_ns: int, packet: bytes, length: int | None = None) -> None:
        """One packet: `packet` is the whole packet or (with `length`, its full size) its first bytes."""
        length = len(packet) if length is None else length
        self.packets += 1
        self.bytes += length
        self.lengths.append(length)
        if self.first_ns is None:
            self.first_ns = time_ns
        self.last_ns = time_ns
        try:
            header = h265_rtp.parse_rtp_header(packet, prefix=length != len(packet))
        except h265_rtp.RtpError:
            self.invalid += 1
            return
        self.payload_types[header.payload_type] += 1
        self.ssrcs[header.ssrc] += 1
        self.markers += header.marker
        if self.first_timestamp is None:
            self.first_timestamp = header.timestamp
        self.last_timestamp = header.timestamp
        missing, out_of_order, new_last = rtp_pipeline.rtp_seq_step(self._last_seq, header.sequence)
        if self._last_seq is not None and new_last < self._last_seq:
            self.seq_wraps += 1
        self._last_seq = new_last
        self.seq_missing += missing
        self.seq_out_of_order += out_of_order
        self._add_payload_header(packet[header.payload_offset : header.payload_offset + 3])
        self._add_to_access_unit(header.timestamp, time_ns, length)

    def _add_payload_header(self, head: bytes) -> None:
        if len(head) < 2:
            self.kinds["invalid"] += 1
            return
        kind = h265_rtp.payload_type_of(head)
        if kind == h265_rtp.FU:
            self.kinds["fu"] += 1
            if len(head) == 3:
                self.fu_starts += bool(head[2] & 0x80)
                self.fu_ends += bool(head[2] & 0x40)
        elif kind == h265_rtp.AP:
            self.kinds["ap"] += 1
        elif kind == h265_rtp.PACI:
            self.kinds["paci"] += 1
        elif kind > h265_rtp.PACI:
            self.kinds["unknown"] += 1
        else:
            self.kinds["single"] += 1
            self.single_nal_types[h265_rtp.nal_type_name(kind)] += 1

    def _add_to_access_unit(self, timestamp: int, time_ns: int, length: int) -> None:
        if timestamp != self._au_timestamp:
            if self._au_timestamp is not None:
                self._finish_access_unit()
                step = (timestamp - self._au_timestamp) & 0xFFFFFFFF
                self.timestamp_steps[step - (1 << 32) if step >= 1 << 31 else step] += 1
            self._au_timestamp = timestamp
            self._au_first_ns = time_ns
            self._au_bytes = 0
            self._au_packets = 0
            if self._first_au_ns is None:
                self._first_au_ns = time_ns
            self._last_au_ns = time_ns
        self._au_bytes += length
        self._au_packets += 1
        self._au_last_ns = time_ns

    def _finish_access_unit(self) -> None:
        self.access_units += 1
        self.au_bytes.append(self._au_bytes)
        self.au_packets.append(self._au_packets)
        self.au_span_ns.append(self._au_last_ns - self._au_first_ns)

    def timestamp_step(self) -> int | None:
        """The median RTP timestamp step between consecutive access units (forward steps only)."""
        forward = sorted((step, count) for step, count in self.timestamp_steps.items() if step > 0)
        total = sum(count for _step, count in forward)
        seen = 0
        for step, count in forward:
            seen += count
            if 2 * seen >= total:
                return step
        return None

    def summary(self) -> dict:
        if self._au_timestamp is not None and self._au_packets:
            self._finish_access_unit()
            self._au_timestamp = None
            self._au_packets = 0
        duration_s = (self.last_ns - self.first_ns) / 1e9 if self.packets > 1 else 0.0
        step = self.timestamp_step()
        au_window_s = (self._last_au_ns - (self._first_au_ns or 0)) / 1e9
        known = sum(self.kinds[kind] for kind in KINDS)
        return {
            "packets": self.packets,
            "bytes": self.bytes,
            "invalid_rtp": self.invalid,
            "duration_s": round(duration_s, 6),
            "packets_per_s": round(self.packets / duration_s, 2) if duration_s else None,
            "bitrate_mbps": round(8 * self.bytes / duration_s / 1e6, 3) if duration_s else None,
            "payload_types": {str(pt): count for pt, count in sorted(self.payload_types.items())},
            "ssrcs": {f"0x{ssrc:08x}": count for ssrc, count in sorted(self.ssrcs.items())},
            "seq_missing": self.seq_missing,
            "seq_out_of_order": self.seq_out_of_order,
            "seq_wraps": self.seq_wraps,
            "markers": self.markers,
            "access_units": self.access_units,
            "timestamp_first": self.first_timestamp,
            "timestamp_last": self.last_timestamp,
            "timestamp_step_median": step,
            "timestamp_steps": {str(s): c for s, c in self.timestamp_steps.most_common(6)},
            "fps_from_timestamps": round(self.clock_rate / step, 3) if step else None,
            "fps_from_arrival": round((self.access_units - 1) / au_window_s, 3) if au_window_s > 0 else None,
            "packet_bytes": percentiles(self.lengths),
            "packetization": {kind: self.kinds[kind] for kind in (*KINDS, "invalid") if self.kinds[kind]},
            "packetization_share": {kind: round(self.kinds[kind] / known, 4) for kind in KINDS if known},
            "fu_starts": self.fu_starts,
            "fu_ends": self.fu_ends,
            "single_nal_types": dict(sorted(self.single_nal_types.items())),
            "au_bytes": percentiles(self.au_bytes),
            "au_packets": percentiles(self.au_packets),
            "au_span_ms": {k: round(v / 1e6, 3) for k, v in percentiles(self.au_span_ns).items()},
        }


def socket_drops(sock: socket.socket) -> int | None:
    """The kernel's drop counter of a UDP socket (receive buffer full), from /proc/net/udp of its network namespace."""
    inode = str(os.fstat(sock.fileno()).st_ino)
    for path in ("/proc/net/udp", "/proc/net/udp6"):
        try:
            with open(path) as table:
                lines = table.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) >= 13 and fields[9] == inode:
                return int(fields[12])
    return None


def read_rmem_max() -> int | None:
    try:
        return int(pathlib.Path("/proc/sys/net/core/rmem_max").read_text())
    except (OSError, ValueError):
        return None


class StopFlag:
    """Why a loop should stop, or None to go on; set_from_signal is the SIGINT/SIGTERM handler (reason "SIGTERM")."""

    def __init__(self):
        self.reason: str | None = None

    def set_from_signal(self, signum: int, _frame) -> None:
        self.reason = signal.Signals(signum).name


def _receive(sock, *, writer, deadline_ns, idle_ns, stop) -> tuple[array, array, bytearray]:
    """The receive loop: per packet only recv_into, monotonic_ns, two writes and three appends."""
    buf = bytearray(rtpcap.MAX_PACKET)
    view = memoryview(buf)
    times = array("Q")
    lengths = array("I")
    heads = bytearray()
    recv_into = sock.recv_into
    now = time.monotonic_ns
    add_time = times.append
    add_length = lengths.append
    write = writer.write if writer is not None else None
    last_ns = 0
    while stop.reason is None:
        try:
            size = recv_into(buf)
        except BlockingIOError:  # SO_RCVTIMEO expired: check the time limits
            current = now()
            if deadline_ns is not None and current >= deadline_ns:
                stop.reason = "duration"
            elif idle_ns and last_ns and current - last_ns >= idle_ns:
                stop.reason = "idle"
            continue
        last_ns = now()
        add_time(last_ns)
        add_length(size)
        heads += view[:HEAD_BYTES]
        if write is not None:
            write(last_ns, view[:size])
        if deadline_ns is not None and last_ns >= deadline_ns:
            stop.reason = "duration"
    return times, lengths, heads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bind", default="0.0.0.0", help="local address to bind (default all)")
    parser.add_argument("--duration", type=float, default=None, help="seconds to record (default: until a signal)")
    parser.add_argument("--idle-timeout", type=float, default=0.0, help="stop after this many seconds without packets")
    parser.add_argument("--out", default=None, help="the .rtpcap file to write (default: statistics only)")
    parser.add_argument("--rcvbuf", type=int, default=26_214_400, help="SO_RCVBUF to request, bytes")
    parser.add_argument("--stats-json", default=None, help="write the statistics here")
    args = parser.parse_args(argv)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf)
    rcvbuf_actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    # Wake up every 0.2 s without a packet to check the limits; a blocking recv costs nothing per packet.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("ll", 0, 200_000))
    sock.bind((args.bind, args.port))

    stop = StopFlag()
    signal.signal(signal.SIGINT, stop.set_from_signal)
    signal.signal(signal.SIGTERM, stop.set_from_signal)
    report = {
        "tool": "rtp_capture",
        "started_utc": rtpcap.utc_now(),
        "bind": args.bind,
        "port": args.port,
        "duration_requested_s": args.duration,
        "idle_timeout_s": args.idle_timeout,
        "rcvbuf_requested": args.rcvbuf,
        "rcvbuf_actual": rcvbuf_actual,
        "rmem_max": read_rmem_max(),
        "out": args.out,
    }
    writer = None
    if args.out:
        meta = {
            "format": "rtpcap",
            "version": 1,
            "clock": rtpcap.CLOCK_MONOTONIC,
            "source": "rtp_capture",
            "host": socket.gethostname(),
            **{key: report[key] for key in ("started_utc", "bind", "port", "rcvbuf_requested", "rcvbuf_actual")},
        }
        writer = rtpcap.Writer(args.out, meta)
    print(
        f"capturing udp {args.bind}:{args.port} (rcvbuf {rcvbuf_actual} bytes, rmem_max {report['rmem_max']})",
        flush=True,
    )
    start_ns = time.monotonic_ns()
    deadline_ns = None if args.duration is None else start_ns + int(args.duration * 1e9)
    try:
        times, lengths, heads = _receive(
            sock, writer=writer, deadline_ns=deadline_ns, idle_ns=int(args.idle_timeout * 1e9), stop=stop
        )
    finally:
        if writer is not None:
            writer.close()
        report["kernel_drops"] = socket_drops(sock)
        sock.close()
    report["stopped_by"] = stop.reason
    report["elapsed_s"] = round((time.monotonic_ns() - start_ns) / 1e9, 3)

    stats = PacketStats()
    for index, (time_ns, length) in enumerate(zip(times, lengths, strict=True)):
        head = heads[index * HEAD_BYTES : index * HEAD_BYTES + min(length, HEAD_BYTES)]
        stats.add(time_ns, bytes(head), length)
    report.update(stats.summary())
    if args.out:
        report.update(rtpcap.file_digest(args.out))
    _print_summary(report)
    if args.stats_json:
        with open(args.stats_json, "w") as out:
            json.dump(report, out, indent=2)
            out.write("\n")
    return 0


def _print_summary(report: dict) -> None:
    keys = [
        "stopped_by",
        "packets",
        "bytes",
        "duration_s",
        "packets_per_s",
        "bitrate_mbps",
        "access_units",
        "markers",
        "fps_from_timestamps",
        "fps_from_arrival",
        "seq_missing",
        "seq_out_of_order",
        "seq_wraps",
        "kernel_drops",
        "rcvbuf_actual",
        "rmem_max",
        "ssrcs",
        "payload_types",
        "packet_bytes",
        "packetization_share",
        "au_bytes",
        "au_span_ms",
        "packets_sha256",
    ]
    for key in keys:
        if key in report:
            print(f"  {key}: {report[key]}")


if __name__ == "__main__":
    sys.exit(main())
