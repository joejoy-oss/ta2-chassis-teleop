"""Decode statistics for test.py --stats-json: per-frame decode latency plus per-window fps, RSS and CPU time.

No GStreamer import here (tested anywhere). Feed it from RTPH265VideoInterface's latency_hook (on_frame) and call
poll() from the main loop.
"""

from __future__ import annotations

from collections.abc import Callable
import resource

import numpy as np


def process_resources() -> tuple[float | None, float]:
    """(RSS in MiB or None, user + system CPU seconds) of this process, all threads."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    return rss_mb, usage.ru_utime + usage.ru_stime


class DecodeStats:
    """Windows start at the first decoded frame and are window_s long. A window's RSS and CPU time are read when its
    end passes (at the next poll), its frames are counted grace_s later because the hook runs a little after
    decoded_at. A last partial window is dropped."""

    def __init__(
        self, *, window_s: float, resources: Callable[[], tuple[float | None, float]] = process_resources
    ) -> None:
        self.window_s = window_s
        self.samples: list[tuple[int, float | None, float]] = []  # (pts, latency_ms or None, decoded_at)
        self.windows: list[dict] = []
        self._resources = resources
        self._t0: float | None = None
        self._marks: list[tuple[float | None, float]] = []  # resources at window boundaries 0, 1, 2, ...

    def on_frame(self, pts: int, latency_ms: float | None, decoded_at: float) -> None:
        self.samples.append((pts, latency_ms, decoded_at))

    def poll(self, now: float, *, grace_s: float = 0.5, read_resources: bool = True) -> None:
        """Read resources at window boundaries that have passed, and close the windows that are due. Once decoding
        has stopped, pass grace_s=0 and read_resources=False (a reading then would describe the stopped process)."""
        if self._t0 is None:
            if not self.samples or not read_resources:
                return
            self._t0 = self.samples[0][2]
            self._marks.append(self._resources())
        while read_resources and now >= self._t0 + len(self._marks) * self.window_s:
            self._marks.append(self._resources())
        while True:
            k = len(self.windows)
            start = self._t0 + k * self.window_s
            end = start + self.window_s
            if now < end + grace_s or len(self._marks) < k + 2:
                return
            frames = sum(1 for _pts, _lat, t in self.samples if start <= t < end)
            rss_mb, cpu_s = self._marks[k + 1]
            self.windows.append(
                {
                    "start_s": round(k * self.window_s, 3),
                    "frames": frames,
                    "fps": frames / self.window_s,
                    "rss_mb": rss_mb,
                    "cpu_s": round(cpu_s - self._marks[k][1], 3),
                }
            )

    def summary(self, counters: dict) -> dict:
        """Counters (RTPH265VideoInterface.get_stats()), fps overall and per window, latency percentiles."""
        latencies = np.array([lat for _pts, lat, _t in self.samples if lat is not None], dtype=np.float64)
        times = [t for _pts, _lat, t in self.samples]
        span = times[-1] - times[0] if len(times) > 1 else 0.0
        result = {
            "counters": counters,
            "h265_buffers_minus_frames": counters["h265_buffers"] - counters["frames"],
            "frames_timed": len(self.samples),
            "fps_overall": (len(times) - 1) / span if span > 0 else None,
            "fps_min_window": min((w["fps"] for w in self.windows), default=None),
            "window_s": self.window_s,
            "windows": self.windows,
            "latency_ms": {"count": int(latencies.size), "unpaired": len(self.samples) - int(latencies.size)},
        }
        if latencies.size:
            p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
            result["latency_ms"].update(
                p50=float(p50), p95=float(p95), p99=float(p99), max=float(latencies.max()), mean=float(latencies.mean())
            )
        return result

    def sample_rows(self) -> list[list]:
        """One row per decoded frame: [pts, latency_ms or None, seconds since the first decoded frame]."""
        if not self.samples:
            return []
        t0 = self.samples[0][2]
        return [[pts, None if lat is None else round(lat, 3), round(t - t0, 6)] for pts, lat, t in self.samples]
