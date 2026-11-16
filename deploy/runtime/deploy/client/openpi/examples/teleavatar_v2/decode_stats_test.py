"""Tests for decode_stats.py (test.py --stats-json), with a simulated clock and resource readings."""

import decode_stats  # pytest puts this directory on sys.path
import numpy as np
import pytest

FPS = 45.0


class _Resources:
    """Readings that tell which poll took them: rss = reading index, cpu = 2 s per reading."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return float(self.calls), 2.0 * self.calls


def _run(stats: decode_stats.DecodeStats, *, t0: float, seconds: float, hook_delay_s: float = 0.003) -> float:
    """Frames at 45 fps from t0, each reported hook_delay_s after it was decoded; polls every 0.1 s in between.
    Returns the time of the last frame."""
    n = round(seconds * FPS)
    decoded = [t0 + i / FPS for i in range(n)]
    reported = 0
    now = t0
    while reported < n:
        now += 0.1
        while reported < n and decoded[reported] + hook_delay_s <= now:
            stats.on_frame(reported, 5.0 + reported % 10, decoded[reported])
            reported += 1
        stats.poll(now)
    return decoded[-1]


def test_full_windows_only_and_resources_from_the_boundaries():
    resources = _Resources()
    stats = decode_stats.DecodeStats(window_s=10.0, resources=resources)
    last = _run(stats, t0=100.0, seconds=25.0)
    stats.poll(last + 0.2, grace_s=0.0, read_resources=False)
    assert [w["frames"] for w in stats.windows] == [450, 450]  # 20-25 s is partial
    assert [w["fps"] for w in stats.windows] == [45.0, 45.0]
    assert [w["start_s"] for w in stats.windows] == [0.0, 10.0]
    # reading 1 at the first frame, 2 at 10 s, 3 at 20 s: each window reports the reading at its end
    assert [w["rss_mb"] for w in stats.windows] == [2.0, 3.0]
    assert [w["cpu_s"] for w in stats.windows] == [2.0, 2.0]
    assert resources.calls == 3


def test_a_frame_reported_late_still_counts_in_its_window():
    stats = decode_stats.DecodeStats(window_s=1.0, resources=_Resources())
    stats.on_frame(0, 1.0, 10.0)
    stats.poll(10.0)
    stats.on_frame(1, 1.0, 10.5)
    stats.poll(11.05)  # window [10, 11) ended, resources read, frames not counted yet
    assert stats.windows == []
    stats.on_frame(2, 1.0, 10.999)  # decoded inside the window, reported after its end
    stats.poll(11.6)
    assert [w["frames"] for w in stats.windows] == [3]


def test_after_stopping_a_window_without_its_end_reading_is_dropped():
    resources = _Resources()
    stats = decode_stats.DecodeStats(window_s=1.0, resources=resources)
    stats.on_frame(0, 1.0, 10.0)
    stats.poll(10.5)
    stats.on_frame(1, 1.0, 10.9)
    # Decoding stopped before a poll saw 11.0: no reading at the boundary, so no window (not a post-stop reading).
    stats.poll(12.0, grace_s=0.0, read_resources=False)
    assert stats.windows == []
    assert resources.calls == 1


def test_summary():
    stats = decode_stats.DecodeStats(window_s=10.0, resources=_Resources())
    last = _run(stats, t0=0.0, seconds=21.0)
    stats.on_frame(999, None, last + 1 / FPS)  # a frame without a paired decoder input
    stats.poll(last + 1.0, grace_s=0.0, read_resources=False)
    counters = {"frames": 946, "h265_buffers": 947, "rtp_seq_gaps": 0, "queue_overruns": 0}
    summary = stats.summary(counters)
    latencies = np.array([5.0 + i % 10 for i in range(945)])
    assert summary["latency_ms"]["count"] == 945
    assert summary["latency_ms"]["unpaired"] == 1
    assert summary["latency_ms"]["p95"] == pytest.approx(np.percentile(latencies, 95))
    assert summary["latency_ms"]["max"] == 14.0
    assert summary["fps_overall"] == pytest.approx(FPS)
    assert summary["fps_min_window"] == 45.0
    assert summary["h265_buffers_minus_frames"] == 1
    rows = stats.sample_rows()
    assert rows[0] == [0, 5.0, 0.0]
    assert rows[-1][1] is None
    assert len(rows) == 946


def test_no_frames():
    stats = decode_stats.DecodeStats(window_s=10.0, resources=_Resources())
    stats.poll(5.0)
    stats.poll(50.0, grace_s=0.0, read_resources=False)
    summary = stats.summary({"frames": 0, "h265_buffers": 0})
    assert summary["fps_overall"] is None
    assert summary["fps_min_window"] is None
    assert summary["latency_ms"] == {"count": 0, "unpaired": 0}
    assert stats.sample_rows() == []


def test_process_resources_reads_this_process():
    rss_mb, cpu_s = decode_stats.process_resources()
    assert rss_mb is None or rss_mb > 0
    assert cpu_s > 0
