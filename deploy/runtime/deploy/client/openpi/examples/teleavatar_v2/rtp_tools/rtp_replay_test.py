"""Tests for rtp_replay.py: the per-pass sequence number and timestamp rewriting of --loop, sending, and stopping
on SIGINT/SIGTERM with the report still written."""

from __future__ import annotations

import contextlib
import itertools
import json
import pathlib
import signal
import socket
import struct
import subprocess
import sys

from h265_rtp_test import rtp_packet
import pytest
import rtp_replay  # pytest puts this directory on sys.path
import rtpcap

FRAME_NS = round(1e9 / 45)
HERE = pathlib.Path(__file__).resolve().parent


def _fields(packet: bytes) -> tuple[int, int]:
    return struct.unpack_from("!HI", packet, 2)


def test_rewrite_changes_only_sequence_number_and_timestamp():
    packet = rtp_packet(b"\x02\x01payload", seq=65535, ts=0xFFFFFFFF, marker=True)
    rewritten = rtp_replay.rewrite(packet, 3, 2)
    assert _fields(rewritten) == (2, 1)  # both wrap
    assert rewritten[:2] == packet[:2]
    assert rewritten[8:] == packet[8:]
    assert rtp_replay.rewrite(packet, 0, 0) == packet


def _stream() -> tuple[list[bytes], list[int]]:
    """Five access units of three packets, with GStreamer's 45 fps timestamp rounding (steps 1999/2000), wrapping
    sequence numbers and timestamps."""
    first_ts = 2**32 - 4000
    timestamps = [first_ts, first_ts + 1999, first_ts + 3999, first_ts + 5999, first_ts + 7999]
    packets, times = [], []
    seq = 65530
    for index, ts in enumerate(timestamps):
        for part in range(3):
            packets.append(rtp_packet(bytes((2, 1, part)), seq=seq, ts=ts, marker=part == 2))
            times.append(1_000_000 + index * FRAME_NS + part * 11_200)
            seq += 1
    return packets, times


def test_loop_plan():
    packets, times = _stream()
    plan = rtp_replay.plan_loops(packets, times)
    assert plan.frame_step == 2000  # the median step
    assert plan.seq_advance == 15
    assert plan.ts_advance == 7999 + 2000
    assert plan.period_ns == 4 * FRAME_NS + FRAME_NS  # last access unit start + one frame


def test_late_packet_at_the_end_does_not_change_the_plan():
    packets, times = _stream()
    late = packets[-4]  # a packet of the fourth access unit arriving again after the last one
    plan = rtp_replay.plan_loops([*packets, late], [*times, times[-1] + 1000])
    assert (plan.seq_advance, plan.ts_advance) == (15, 9999)


def test_passes_form_one_continuous_stream():
    packets, times = _stream()
    plan = rtp_replay.plan_loops(packets, times)
    stream = [
        rtp_replay.rewrite(packet, index * plan.seq_advance, index * plan.ts_advance)
        for index in range(3)
        for packet in packets
    ]
    missing = out_of_order = 0
    last = None
    for packet in stream:
        step_missing, step_out_of_order, last = rtp_replay.rtp_pipeline.rtp_seq_step(last, _fields(packet)[0])
        missing += step_missing
        out_of_order += step_out_of_order
    assert (missing, out_of_order) == (0, 0)
    access_unit_timestamps = [_fields(packet)[1] for packet in stream[::3]]
    steps = [(b - a) % 2**32 for a, b in itertools.pairwise(access_unit_timestamps)]
    assert steps == [1999, 2000, 2000, 2000, 2000] * 2 + [1999, 2000, 2000, 2000]  # one frame step between passes
    assert stream[: len(packets)] == packets  # the first pass is unchanged


def test_replay_sends_all_passes_on_time():
    packets, times = _stream()
    plan = rtp_replay.plan_loops(packets, times)
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
    ):
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(5)
        errors, send_errors, wall_ns = rtp_replay.replay(
            sender, receiver.getsockname(), packets, times, passes=2, speed=4.0, plan=plan, start_delay_ns=0
        )
        received = [receiver.recv(2048) for _ in range(2 * len(packets))]
    assert send_errors == 0
    assert len(errors) == 2 * len(packets)
    assert min(errors) >= 0  # never early
    assert received[: len(packets)] == packets
    assert [_fields(p)[0] for p in received[len(packets) :]] == [(65530 + 15 + i) % 2**16 for i in range(15)]
    assert wall_ns >= (plan.period_ns + times[-1] - times[0]) / 4.0


def test_replay_with_the_stop_flag_set_sends_nothing():
    packets, times = _stream()
    stop = rtp_replay.rtp_capture.StopFlag()
    stop.reason = "SIGTERM"
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
    ):
        receiver.bind(("127.0.0.1", 0))
        errors, send_errors, _wall_ns = rtp_replay.replay(
            sender,
            receiver.getsockname(),
            packets,
            times,
            passes=2,
            speed=1.0,
            plan=rtp_replay.plan_loops(packets, times),
            start_delay_ns=0,
            stop=stop,
        )
        receiver.settimeout(0)  # non-blocking
        with pytest.raises(BlockingIOError):
            receiver.recv(2048)
    assert (len(errors), send_errors) == (0, 0)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_a_signal_stops_the_replay_and_the_report_is_still_written(tmp_path, signum):
    """As under `docker stop`: rtp_replay.py stops at the next packet and still writes its report, marked interrupted."""
    packets = [rtp_packet(bytes((2, 1, i % 256)), seq=i, ts=2000 * i, marker=True) for i in range(180)]  # 4 s
    times = [send_ns for send_ns, _packet in rtpcap.synthesize_send_times(packets, fps=45, line_rate_bps=1e9)]
    with rtpcap.Writer(tmp_path / "x.rtpcap", {"source": "test"}) as writer:
        for send_ns, packet in zip(times, packets, strict=True):
            writer.write(send_ns, packet)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(20)
        command = [sys.executable, "-B", str(HERE / "rtp_replay.py"), "--file", str(tmp_path / "x.rtpcap")]
        command += ["--port", str(receiver.getsockname()[1]), "--loop", "2"]
        command += ["--report-json", str(tmp_path / "replay.json")]
        replay = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        try:
            received = [receiver.recv(2048)]  # the replay is sending
            replay.send_signal(signum)
            printed, _ = replay.communicate(timeout=20)
        finally:
            replay.kill()
        receiver.settimeout(0)  # non-blocking
        with contextlib.suppress(BlockingIOError):
            while True:
                received.append(receiver.recv(2048))
    assert replay.returncode == 0
    assert f"interrupted by {signum.name}" in printed
    report = json.loads((tmp_path / "replay.json").read_text())
    assert (report["interrupted"], report["signal"], report["send_errors"]) == (True, signum.name, 0)
    assert 0 < report["packets_sent"] == len(received) < report["packets_planned"] == 2 * len(packets)
    assert received == packets[: len(received)]
    plan = rtp_replay.plan_loops(packets, times)
    assert report["planned_duration_s"] == round((plan.period_ns + times[-1] - times[0]) / 1e9, 3)
    assert report["wall_s"] < report["planned_duration_s"] / 2
