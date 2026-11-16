"""Tests for rtp_capture.py: PacketStats (with the receiver's sequence counting, rtp_pipeline.rtp_seq_step), and a
loopback run of rtp_capture.py (subprocess) receiving what rtp_replay.py sends."""

from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys

import h265_rtp
from h265_rtp_test import packetize
from h265_rtp_test import rtp_packet
import pytest
import rtp_capture  # pytest puts this directory on sys.path
import rtp_pipeline  # rtp_capture put its directory (one up) on sys.path
import rtp_replay
import rtpcap

HERE = pathlib.Path(__file__).resolve().parent


def test_percentiles_nearest_rank():
    assert rtp_capture.percentiles([]) == {}
    assert rtp_capture.percentiles([5]) == {"min": 5, "p50": 5, "p99": 5, "max": 5}
    values = list(range(1, 101))  # 1..100
    assert rtp_capture.percentiles(reversed(values)) == {"min": 1, "p50": 50, "p99": 99, "max": 100}
    assert rtp_capture.percentiles(range(1, 1001), (50, 99.9))["p99.9"] == 999


def test_sequence_counting_matches_the_receiver_including_the_wrap():
    sequence_numbers = [65533, 65534, 65535, 0, 1, 3, 2, 4, 4, 9]
    stats = rtp_capture.PacketStats()
    for index, seq in enumerate(sequence_numbers):
        stats.add(index, rtp_packet(b"\x02\x01\x80", seq=seq, ts=0))
    # The same numbers through rtp_pipeline.rtp_seq_step directly, as RTPH265VideoInterface counts them.
    missing = out_of_order = 0
    last = None
    for seq in sequence_numbers:
        step_missing, step_out_of_order, last = rtp_pipeline.rtp_seq_step(last, seq)
        missing += step_missing
        out_of_order += step_out_of_order
    assert (stats.seq_missing, stats.seq_out_of_order) == (missing, out_of_order) == (1 + 4, 2)
    assert stats.seq_wraps == 1


def _stream(access_units: int, *, seq: int, ts: int, step: int = 2000) -> list[bytes]:
    """`access_units` access units: parameter sets in an AP before the first, then one 3000-byte slice in FUs."""
    packets = []
    for index in range(access_units):
        aggregate = bytes((h265_rtp.AP << 1, 1)) + b"".join(
            len(unit).to_bytes(2, "big") + unit for unit in (bytes((64, 1, 12)), bytes((66, 1, 1)))
        )
        if index == 0:
            packets.append(rtp_packet(aggregate, seq=seq, ts=ts))
            seq += 1
        slice_nal = bytes((2, 1, 0x80)) + b"\x55" * 3000
        units = packetize([slice_nal], ts=ts + index * step, seq=seq)
        packets += units
        seq += len(units)
    return packets


def test_packet_statistics():
    first_ts = 2**32 - 3000  # the RTP timestamp wraps after the second access unit
    packets = _stream(4, seq=65530, ts=first_ts)
    stats = rtp_capture.PacketStats()
    for index, packet in enumerate(packets):
        stats.add(1_000_000 + index * 10_000_000, packet)
    summary = stats.summary()
    assert summary["packets"] == len(packets) == 1 + 4 * 3
    assert summary["bytes"] == sum(map(len, packets))
    assert summary["access_units"] == summary["markers"] == 4
    assert (summary["timestamp_step_median"], summary["fps_from_timestamps"]) == (2000, 45.0)
    assert summary["timestamp_steps"] == {"2000": 3}
    assert (summary["timestamp_first"], summary["timestamp_last"]) == (first_ts, (first_ts + 6000) % 2**32)
    assert summary["packetization"] == {"ap": 1, "fu": 12}
    assert (summary["fu_starts"], summary["fu_ends"]) == (4, 4)
    assert (summary["seq_missing"], summary["seq_out_of_order"], summary["seq_wraps"]) == (0, 0, 1)
    assert summary["ssrcs"] == {"0x00001234": len(packets)}
    assert summary["payload_types"] == {"96": len(packets)}
    assert summary["packet_bytes"]["max"] == 1400
    assert summary["au_packets"] == {"min": 3, "p50": 3, "p99": 4, "max": 4}
    assert summary["duration_s"] == pytest.approx(0.12)


def test_statistics_from_packet_prefixes_equal_those_from_whole_packets():
    packets = _stream(3, seq=100, ts=0)
    whole = rtp_capture.PacketStats()
    prefixes = rtp_capture.PacketStats()
    for index, packet in enumerate(packets):
        whole.add(index, packet)
        prefixes.add(index, packet[: rtp_capture.HEAD_BYTES], len(packet))
    assert whole.summary() == prefixes.summary()


def test_invalid_packets_are_counted():
    stats = rtp_capture.PacketStats()
    stats.add(0, b"\x00" * 20)  # RTP version 0
    stats.add(1, rtp_packet(b"\x02", seq=1, ts=0))  # no payload header
    summary = stats.summary()
    assert (summary["invalid_rtp"], summary["packetization"]) == (1, {"invalid": 1})


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_capture_receives_a_replay_unchanged(tmp_path):
    """Two passes of a small stream over loopback: the first arrives byte for byte, the second continues it."""
    packets = _stream(5, seq=65530, ts=2**32 - 5000)
    source = tmp_path / "source.rtpcap"
    with rtpcap.Writer(source, {"source": "test"}) as writer:
        for send_ns, packet in rtpcap.synthesize_send_times(packets, fps=45, line_rate_bps=1e9):
            writer.write(send_ns, packet)
    port = _free_udp_port()
    capture = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(HERE / "rtp_capture.py"),
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--duration",
            "20",
            "--idle-timeout",
            "0.6",
            "--out",
            str(tmp_path / "captured.rtpcap"),
            "--stats-json",
            str(tmp_path / "capture.json"),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert capture.stdout.readline().startswith("capturing udp 127.0.0.1:")  # bound and receiving
        rtp_replay.main(
            ["--file", str(source), "--port", str(port), "--loop", "2", "--report-json", str(tmp_path / "replay.json")]
        )
        capture.wait(timeout=20)
    finally:
        capture.kill()
    assert capture.returncode == 0
    stats = json.loads((tmp_path / "capture.json").read_text())
    report = json.loads((tmp_path / "replay.json").read_text())
    received = [packet for _t, packet in rtpcap.iter_packets(tmp_path / "captured.rtpcap")]
    assert len(received) == stats["packets"] == report["packets_sent"] == 2 * len(packets)
    assert received[: len(packets)] == packets
    assert rtpcap.packets_sha256(received[: len(packets)]) == report["file_packets_sha256"]
    assert stats["packets_sha256"] == rtpcap.packets_sha256(received)
    assert (stats["seq_missing"], stats["seq_out_of_order"], stats["kernel_drops"]) == (0, 0, 0)
    assert (stats["access_units"], stats["markers"], stats["fps_from_timestamps"]) == (10, 10, 45.0)
    assert stats["stopped_by"] == "idle"
    assert report["send_errors"] == 0
