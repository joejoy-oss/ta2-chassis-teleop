#!/usr/bin/env python3
"""Send the packets of a .rtpcap file to a UDP port with their recorded timing, as the camera would.

    python3 rtp_replay.py --file synthetic_nowpp_20mbps.rtpcap --port 18890 [--host 127.0.0.1] [--loop 3]
        [--speed 1.0] [--report-json replay.json]

Each packet is due at its recorded time (relative to the first packet) divided by --speed. The sender sleeps until
about 1 ms before a packet is due and busy-waits the rest, so it sends within microseconds of the plan unless it falls
behind (several hundred packets of one access unit sent back to back). The report gives the send-time error against
the plan (p50/p99/max), the packets later than --late-ms, send errors, and the file's packet count and packets_sha256
(rtpcap.packets_sha256), which a capture of the first pass must reproduce.

The first pass sends the packets unchanged, byte for byte. With --loop N, passes 2..N continue the stream: sequence
numbers carry on after the file's last one (mod 2^16), RTP timestamps after its last one plus one frame step (mod
2^32), the SSRC stays; each pass starts one frame interval after the last access unit of the previous pass. The
receiver thus sees one continuous, monotonic stream.

SIGINT or SIGTERM (e.g. `docker stop`) ends the replay at the next packet; the report is still written, with
"interrupted": true and the signal, packets_sent and wall_s up to the stop (planned_duration_s stays the plan). In a
container the replay must get the signal itself: make it the container's command (PID 1), or use `docker run --init`.

Use port 18890 for bench replays; 8890 belongs to the robot's real stream.
"""

from __future__ import annotations

import argparse
from array import array
import collections
import json
import pathlib
import signal
import socket
import struct
import sys
import time
from typing import NamedTuple

import rtp_capture
import rtpcap

_TELEAVATAR_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if _TELEAVATAR_DIR not in sys.path:
    sys.path.append(_TELEAVATAR_DIR)  # rtp_pipeline.py lives one directory up

import rtp_pipeline  # noqa: E402

SPIN_NS = 1_000_000  # sleep until this long before a packet is due, then busy-wait
_SEQ_TS = struct.Struct("!HI")  # RTP header bytes 2-7: sequence number, timestamp


class LoopPlan(NamedTuple):
    """How passes after the first continue the stream."""

    seq_advance: int  # added to every sequence number per pass (mod 2^16)
    ts_advance: int  # added to every RTP timestamp per pass (mod 2^32)
    period_ns: int  # time from the start of one pass to the start of the next, before --speed
    frame_step: int | None  # the RTP timestamp step between access units used for ts_advance


def plan_loops(packets: list[bytes], times_ns: list[int], clock_rate: int = 90_000) -> LoopPlan:
    """Sequence number, timestamp and time offsets per pass for `packets` sent at `times_ns`.

    The newest sequence number and timestamp are taken in the forward sense (late, reordered packets do not count),
    the frame step is the median step between consecutive access units (runs of packets with one timestamp).
    """
    first_seq, first_ts = _SEQ_TS.unpack_from(packets[0], 2)
    newest_seq = None
    newest_ts = first_ts
    au_starts_ns = [times_ns[0]]
    last_ts = first_ts
    steps: collections.Counter[int] = collections.Counter()
    for packet, time_ns in zip(packets, times_ns, strict=True):
        seq, ts = _SEQ_TS.unpack_from(packet, 2)
        _missing, _late, newest_seq = rtp_pipeline.rtp_seq_step(newest_seq, seq)
        if ts != last_ts:
            step = (ts - last_ts) & 0xFFFFFFFF
            if 0 < step < 1 << 31:
                steps[step] += 1
                au_starts_ns.append(time_ns)
            last_ts = ts
            if 0 < (ts - newest_ts) & 0xFFFFFFFF < 1 << 31:
                newest_ts = ts
    frame_step = _median(steps)
    if frame_step is None:
        return LoopPlan((newest_seq - first_seq + 1) & 0xFFFF, 0, au_starts_ns[-1] - times_ns[0], None)
    frame_ns = round(frame_step * 1e9 / clock_rate)
    return LoopPlan(
        seq_advance=(newest_seq - first_seq + 1) & 0xFFFF,
        ts_advance=(((newest_ts - first_ts) & 0xFFFFFFFF) + frame_step) & 0xFFFFFFFF,
        period_ns=au_starts_ns[-1] - times_ns[0] + frame_ns,
        frame_step=frame_step,
    )


def _median(counts: collections.Counter[int]) -> int | None:
    total = sum(counts.values())
    seen = 0
    for value, count in sorted(counts.items()):
        seen += count
        if 2 * seen >= total:
            return value
    return None


def rewrite(packet: bytes, seq_offset: int, ts_offset: int) -> bytes:
    """`packet` with seq_offset added to its RTP sequence number (mod 2^16) and ts_offset to its timestamp (mod 2^32);
    everything else unchanged."""
    out = bytearray(packet)
    seq, ts = _SEQ_TS.unpack_from(out, 2)
    _SEQ_TS.pack_into(out, 2, (seq + seq_offset) & 0xFFFF, (ts + ts_offset) & 0xFFFFFFFF)
    return bytes(out)


def replay(
    sock: socket.socket,
    address: tuple[str, int],
    packets: list[bytes],
    times_ns: list[int],
    *,
    passes: int,
    speed: float,
    plan: LoopPlan,
    start_delay_ns: int = 50_000_000,
    stop: rtp_capture.StopFlag | None = None,
) -> tuple[array, int, int]:
    """Send all passes, or until stop.reason is set (checked before each packet); returns (per-packet send error in
    ns, send errors, wall time in ns) for the packets sent."""
    if stop is None:
        stop = rtp_capture.StopFlag()
    errors = array("q")
    send_errors = 0
    now = time.monotonic_ns
    sleep = time.sleep
    sendto = sock.sendto
    record = errors.append
    first_ns = times_ns[0]
    offsets = [round((t - first_ns) / speed) for t in times_ns]
    start = now() + start_delay_ns
    for index in range(passes):
        pass_start = start + round(index * plan.period_ns / speed)
        seq_offset = index * plan.seq_advance
        ts_offset = index * plan.ts_advance
        for packet, offset in zip(packets, offsets, strict=True):
            data = packet if index == 0 else rewrite(packet, seq_offset, ts_offset)
            due = pass_start + offset
            remaining = due - now()
            if remaining > SPIN_NS:
                sleep((remaining - SPIN_NS) / 1e9)
            while (sent_at := now()) < due:
                pass
            if stop.reason is not None:  # a signal came while sending the last packet or waiting for this one
                return errors, send_errors, now() - start
            try:
                sendto(data, address)
            except OSError:
                send_errors += 1
            record(sent_at - due)
    return errors, send_errors, now() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, help="the .rtpcap file to send")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--loop", type=int, default=1, help="number of passes over the file (default 1)")
    parser.add_argument("--speed", type=float, default=1.0, help="2.0 sends twice as fast")
    parser.add_argument("--late-ms", type=float, default=1.0, help="count packets sent later than this")
    parser.add_argument("--sndbuf", type=int, default=4 << 20, help="SO_SNDBUF to request, bytes")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    if args.loop < 1 or args.speed <= 0:
        parser.error("--loop must be >= 1 and --speed > 0")
    stop = rtp_capture.StopFlag()
    handlers = {signum: signal.signal(signum, stop.set_from_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _replay_file(args, parser, stop)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _replay_file(args: argparse.Namespace, parser: argparse.ArgumentParser, stop: rtp_capture.StopFlag) -> int:
    packets: list[bytes] = []
    times_ns: list[int] = []
    with rtpcap.Reader(args.file) as reader:
        meta = reader.meta
        for time_ns, packet in reader:
            times_ns.append(time_ns)
            packets.append(packet)
    if not packets:
        parser.error(f"{args.file} has no packets")
    plan = plan_loops(packets, times_ns)
    if args.loop > 1 and plan.frame_step is None:
        parser.error("--loop needs at least two access units to know the frame step")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.sndbuf)
    address = (socket.gethostbyname(args.host), args.port)
    file_packets_sha256 = rtpcap.packets_sha256(packets)
    print(f"replaying {len(packets)} packets x {args.loop} to {address[0]}:{address[1]} at speed {args.speed}")
    started_utc = rtpcap.utc_now()
    errors, send_errors, wall_ns = replay(
        sock, address, packets, times_ns, passes=args.loop, speed=args.speed, plan=plan, stop=stop
    )
    sock.close()

    error_ms = {key: round(value / 1e6, 4) for key, value in rtp_capture.percentiles(errors).items()}
    late_ns = args.late_ms * 1e6
    report = {
        "tool": "rtp_replay",
        "started_utc": started_utc,
        "file": args.file,
        "file_source": meta.get("source"),
        "file_packets": len(packets),
        "file_packets_sha256": file_packets_sha256,
        "host": address[0],
        "port": args.port,
        "loop": args.loop,
        "speed": args.speed,
        "packets_planned": len(packets) * args.loop,
        "packets_sent": len(errors) - send_errors,
        "send_errors": send_errors,
        "interrupted": stop.reason is not None,
        "signal": stop.reason,
        "planned_duration_s": round(
            ((plan.period_ns * (args.loop - 1)) + times_ns[-1] - times_ns[0]) / args.speed / 1e9, 3
        ),
        "wall_s": round(wall_ns / 1e9, 3),
        "timing_error_ms": error_ms,
        "late_ms": args.late_ms,
        "late_packets": sum(1 for error in errors if error > late_ns),
        "seq_advance_per_pass": plan.seq_advance,
        "ts_advance_per_pass": plan.ts_advance,
        "pass_period_ms": round(plan.period_ns / 1e6, 3),
    }
    if stop.reason is not None:
        print(f"  interrupted by {stop.reason} after {report['packets_sent']} of {report['packets_planned']} packets")
    for key in ("packets_sent", "send_errors", "planned_duration_s", "wall_s", "timing_error_ms", "late_packets"):
        print(f"  {key}: {report[key]}")
    if args.report_json:
        with open(args.report_json, "w") as out:
            json.dump(report, out, indent=2)
            out.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
