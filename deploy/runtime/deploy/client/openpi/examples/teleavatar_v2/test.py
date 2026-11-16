#!/usr/bin/env python3
"""
Bring-up test for RTPH265VideoInterface.

This file records decoded RTP/H265 split frames to MP4 for validation, and/or
measures the decoder: --stats-json writes every frame's decode latency (the
interface's decode_latency), fps per window, process RSS/CPU, and the
interface counters; with --video-decode va also the GPU conversion's set-up
times, its per-frame steps and the decoder's own latency of the first
frames. Without --split-output-dir nothing is recorded, so the measurement
does not compete with MP4 encoding. The production interface stays in
rtp_video_interface.py and does not own file-writing logic.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np

try:
    from .decode_stats import DecodeStats
    from .rtp_video_interface import STARTUP_OVERRUN_WINDOW_S
    from .rtp_video_interface import RTPH265VideoInterface
except ImportError:
    from decode_stats import DecodeStats
    from rtp_video_interface import STARTUP_OVERRUN_WINDOW_S
    from rtp_video_interface import RTPH265VideoInterface


class SplitMp4Recorder:
    """Write split interface images to MP4 without coupling to the decoder."""

    def __init__(
        self,
        *,
        split_output_dir: Path,
        output_fps: float,
    ):
        self.split_output_dir = split_output_dir.expanduser()
        self.output_fps = output_fps
        self.split_output_dir.mkdir(parents=True, exist_ok=True)

        self._writers: dict[str, cv2.VideoWriter] = {}
        self._writer_paths: dict[str, Path] = {}
        self._writer_sizes: dict[str, tuple[int, int]] = {}
        self._frame_counts: dict[str, int] = {}
        self._last_timestamps: dict[str, float] = {}

    def write_latest(self, interface: RTPH265VideoInterface) -> None:
        images, timestamps = interface.get_latest_images_with_timestamps()
        for name in interface.split_image_names:
            self._write_if_new(name, self.split_output_dir / f"{name}.mp4", images, timestamps, name)

    def close(self) -> None:
        for key, writer in self._writers.items():
            writer.release()
            logging.info("Wrote %d frames to %s", self._frame_counts[key], self._writer_paths[key])
        self._writers.clear()

    def _write_if_new(
        self,
        key: str,
        path: Path,
        images: dict[str, np.ndarray],
        timestamps: dict[str, float],
        image_name: str,
    ) -> None:
        frame_rgb = images.get(image_name)
        timestamp = timestamps.get(image_name)
        if frame_rgb is None or timestamp is None:
            return
        if self._last_timestamps.get(key) == timestamp:
            return

        writer = self._get_writer(key, path, frame_rgb)
        size = (frame_rgb.shape[1], frame_rgb.shape[0])
        if size != self._writer_sizes[key]:
            logging.warning("Skipping %s frame with changed size: %s != %s", key, size, self._writer_sizes[key])
            return

        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        self._frame_counts[key] += 1
        self._last_timestamps[key] = timestamp

    def _get_writer(self, key: str, path: Path, frame_rgb: np.ndarray) -> cv2.VideoWriter:
        writer = self._writers.get(key)
        if writer is not None:
            return writer

        size = (frame_rgb.shape[1], frame_rgb.shape[0])
        writer = _open_mp4_writer(path, size, self.output_fps)
        self._writers[key] = writer
        self._writer_paths[key] = path
        self._writer_sizes[key] = size
        self._frame_counts[key] = 0
        logging.info("Recording %s to %s at %.2f fps", key, path, self.output_fps)
        return writer


def _open_mp4_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer: {path}")
    return writer


def _frame_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from None
    return width, height


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--payload", type=int, default=96)
    parser.add_argument("--udp-buffer-size", type=int, default=20_000_000)
    parser.add_argument("--decoder", default="nvh265dec max-display-delay=0", help="NVIDIA decoder element")
    parser.add_argument("--video-decode", choices=("nvidia", "cpu", "va"), default="nvidia")
    parser.add_argument(
        "--video-cpu-threads", type=int, default=1, help="cpu: avdec_h265 max-threads (0 = host CPUs, at most 16)"
    )
    parser.add_argument(
        "--video-cpu-thread-type", choices=("auto", "frame", "slice"), default="auto", help="slice needs a WPP stream"
    )
    parser.add_argument("--video-convert-threads", type=int, default=4, help="cpu: videoconvert n-threads")
    parser.add_argument("--video-va-decoder", default="vah265dec", help="va: the VA decoder element")
    parser.add_argument(
        "--video-va-frame-size",
        type=_frame_size,
        default="1280x2720",
        help="va: WIDTHxHEIGHT the GPU output is made before the first frame (another size costs one rebuild)",
    )
    parser.add_argument("--log-interval-s", type=float, default=2.0)
    parser.add_argument("--initial-timeout-s", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, help="Run for this many seconds after the first frame")
    # Default matches the ~45 fps RTP stream.
    parser.add_argument("--output-fps", type=float, default=45.0)
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        help="Write one MP4 per split crop to this directory (default: record nothing)",
    )
    parser.add_argument("--stats-json", type=Path, help="Write decode statistics to this JSON file on exit")
    parser.add_argument("--stats-window-s", type=float, default=10.0, help="Window for per-window fps/RSS/CPU")
    args = parser.parse_args()
    if args.split_output_dir is None and args.stats_json is None:
        parser.error("give --split-output-dir and/or --stats-json")
    return args


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    stats = DecodeStats(window_s=args.stats_window_s) if args.stats_json is not None else None
    interface = RTPH265VideoInterface(
        camera_name=args.camera_name,
        port=args.port,
        payload=args.payload,
        udp_buffer_size=args.udp_buffer_size,
        decoder=args.decoder,
        log_interval_s=args.log_interval_s,
        decode_backend=args.video_decode,
        cpu_decode_threads=args.video_cpu_threads,
        cpu_thread_type=args.video_cpu_thread_type,
        cpu_convert_threads=args.video_convert_threads,
        va_decoder=args.video_va_decoder,
        va_frame_size=args.video_va_frame_size,
        va_diagnostics=stats is not None and args.video_decode == "va",
        latency_hook=stats.on_frame if stats is not None else None,
    )
    recorder = None
    if args.split_output_dir is not None:
        recorder = SplitMp4Recorder(
            split_output_dir=args.split_output_dir,
            output_fps=args.output_fps,
        )

    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.set())

    if args.duration_s is not None:
        logging.info("Will run for %.1fs after the first decoded frame", args.duration_s)

    exit_code = 1
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    interface.start()
    try:
        if interface.wait_for_initial_data(timeout=args.initial_timeout_s):
            exit_code = _run(args, interface, recorder, stats, stop_requested)
    finally:
        if recorder is not None:
            recorder.close()
        interface.stop()
        if stats is not None:
            _write_stats(args, stats, interface, started_at, exit_code)

    return exit_code


def _run(
    args: argparse.Namespace,
    interface: RTPH265VideoInterface,
    recorder: SplitMp4Recorder | None,
    stats: DecodeStats | None,
    stop_requested: threading.Event,
) -> int:
    """Until --duration-s, a signal (0) or the end of the stream (1)."""
    start_time = time.monotonic()
    # Recording polls for new frames; a measurement only wakes up to close stats windows.
    poll_s = 0.002 if recorder is not None else 0.1
    while not stop_requested.is_set():
        if recorder is not None:
            recorder.write_latest(interface)
        if stats is not None:
            stats.poll(time.monotonic())

        if args.duration_s is not None and time.monotonic() - start_time >= args.duration_s:
            if stats is not None:
                stats.poll(time.monotonic())  # read resources at the final window boundary while still decoding
            break

        if interface.wait(timeout=poll_s):
            return 1
    return 0


def _write_stats(args, stats: DecodeStats, interface: RTPH265VideoInterface, started_at: str, exit_code: int) -> None:
    stats.poll(time.monotonic(), grace_s=0.0, read_resources=False)  # decoding has stopped: close the last window
    summary = stats.summary(interface.get_stats())
    report = {
        "tool": "examples/teleavatar_v2/test.py",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pipeline": interface.pipeline_description,
        "started_at": started_at,
        "ended_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "exit_code": exit_code,
        **summary,
    }
    one_line = {}
    va = interface.get_va_timings()
    if va is not None:
        steps = va.pop("steps_ms", {})
        va["steps_summary_ms"] = {name: _distribution(values) for name, values in steps.items()}
        report["va"] = va
        # Per-frame step times, in the order the frames were taken.
        one_line["va_steps_ms"] = steps
    # One row per decoded frame ([pts, latency_ms or null, seconds since the first frame]).
    one_line["samples"] = stats.sample_rows()
    body = json.dumps(report, indent=1)
    tail = "".join(f',\n "{key}": {json.dumps(value, separators=(",", ":"))}' for key, value in one_line.items())
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(body[: -len("\n}")] + tail + "\n}\n")
    lat = summary["latency_ms"]
    logging.info(
        "Stats: frames=%d fps_overall=%s fps_min_window=%s latency p50/p95/p99/max=%s/%s/%s/%s ms "
        "rtp_seq_gaps=%d queue_overruns=%d (%d after the first %g s) h265_buffers-frames=%d -> %s",
        summary["counters"]["frames"],
        _fmt(summary["fps_overall"]),
        _fmt(summary["fps_min_window"]),
        _fmt(lat.get("p50")),
        _fmt(lat.get("p95")),
        _fmt(lat.get("p99")),
        _fmt(lat.get("max")),
        summary["counters"]["rtp_seq_gaps"],
        summary["counters"]["queue_overruns"],
        summary["counters"]["queue_overruns_after_startup"],
        STARTUP_OVERRUN_WINDOW_S,
        summary["h265_buffers_minus_frames"],
        args.stats_json,
    )


def _distribution(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values)
    return {
        "count": len(values),
        **{f"p{q}": float(np.percentile(array, q)) for q in (50, 95, 99)},
        "max": float(array.max()),
        "first": float(array[0]),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
