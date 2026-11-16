#!/usr/bin/env python3
"""
Receive a Teleavatar H265 RTP video stream as RGB images.

This mirrors the image-facing part of TeleavatarROS2Interface: decoded frames
are stored in latest_images under a lock, and get_observation() returns image
copies. The interface keeps both the composite frame and six split camera views.
Test-only file writing lives in test.py.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import threading
import time
from typing import Dict
from typing import Optional

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib
    from gi.repository import Gst
except ImportError as exc:
    raise RuntimeError(
        "PyGObject/GStreamer Python bindings are required. Install python3-gi "
        "and gir1.2-gstreamer-1.0, or run with a Python that has gi available."
    ) from exc


try:
    from . import rtp_pipeline
except ImportError:  # run as a script from this directory (test.py)
    import rtp_pipeline

Gst.init(None)


# VA backend: consecutive conversion failures treated like a GStreamer error, and the frames whose decoder latency
# va_diagnostics records.
VA_MAX_FAILURES = 3
VA_DECODER_PROBE_FRAMES = 45
# Overruns of the leaky queue up to this long after the first decoded frame belong to the start of the stream: the
# decoder's first frame is slow and the next ones bunch up behind it. The acceptance criterion (owner, 11-03) allows one
# there and none after it; get_stats() counts both sides. Overrun times kept, and overruns logged one by one.
STARTUP_OVERRUN_WINDOW_S = 1.0
OVERRUN_TIMES_MAX = 1000
OVERRUN_LOGS_MAX = 10

TELEAVATAR_SPLIT_REGIONS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("head_right_eye", (0.1250, 0.0000, 0.8750, 0.3529411765)),
    ("head_left_eye", (0.1250, 0.3529411765, 0.8750, 0.7058823529)),
    ("right_wrist_left_eye", (0.0000, 0.7058823529, 0.5000, 0.8529411765)),
    ("right_wrist_right_eye", (0.5000, 0.7058823529, 1.0000, 0.8529411765)),
    ("left_wrist_left_eye", (0.0000, 0.8529411765, 0.5000, 1.0000)),
    ("left_wrist_right_eye", (0.5000, 0.8529411765, 1.0000, 1.0000)),
)


class RTPH265VideoInterface:
    """Thread-safe image interface backed by a GStreamer RTP/H265 decoder.

    decode_backend "nvidia" (default) decodes with `decoder` on the GPU; "cpu" uses avdec_h265 with
    cpu_decode_threads (max-threads) and cpu_thread_type, then videoconvert with cpu_convert_threads (see
    rtp_pipeline.py); "va" decodes with va_decoder on an Intel GPU and converts each frame to RGB on the GPU straight
    into this process's memory (va_frame.py), with output surfaces made va_frame_size (width, height) before the first
    frame. latency_hook, if given, is called for every decoded frame with (pts, latency_ms, decoded_at): latency_ms is
    None when the frame could not be paired with its decoder input, decoded_at is time.monotonic(). With
    va_diagnostics the VA path also keeps per-frame conversion step times and the decoder's own latency for the first
    frames (get_va_timings()).
    """

    def __init__(
        self,
        *,
        camera_name: str = "head_camera",
        port: int = 8890,
        payload: int = 96,
        udp_buffer_size: int = 20_000_000,
        decoder: str = rtp_pipeline.DEFAULT_NVIDIA_DECODER,
        log_interval_s: float = 2.0,
        decode_backend: str = rtp_pipeline.NVIDIA,
        cpu_decode_threads: int = 1,
        cpu_thread_type: str = "auto",
        cpu_convert_threads: int = 4,
        va_decoder: str = rtp_pipeline.DEFAULT_VA_DECODER,
        va_frame_size: tuple[int, int] = rtp_pipeline.DEFAULT_VA_FRAME_SIZE,
        va_diagnostics: bool = False,
        latency_hook: Optional[Callable[[int, Optional[float], float], None]] = None,
    ):
        self.camera_name = camera_name
        self.port = port
        self.payload = payload
        self.udp_buffer_size = udp_buffer_size
        self.decoder = decoder
        self.log_interval_s = log_interval_s
        self.decode_backend = decode_backend
        self.cpu_decode_threads = cpu_decode_threads
        self.cpu_thread_type = cpu_thread_type
        self.cpu_convert_threads = cpu_convert_threads
        self.va_decoder = va_decoder
        self.va_frame_size = va_frame_size
        self.va_diagnostics = va_diagnostics
        self.latency_hook = latency_hook
        self._build_pipeline()  # reject a bad backend or thread setting here rather than in start()
        self.split_regions = TELEAVATAR_SPLIT_REGIONS
        self.split_image_names = tuple(name for name, _box in self.split_regions)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.lock = threading.Lock()
        self.latest_images: Dict[str, np.ndarray] = {}
        self.image_timestamps: Dict[str, float] = {}

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._first_frame = threading.Event()
        self._eos_or_error = threading.Event()
        self._stop_requested = False

        self._frame_count = 0
        self._rtp_packet_count = 0
        self._h265_buffer_count = 0
        # Missing RTP sequence numbers (forward jumps), and packets that arrived behind the newest one.
        self._rtp_seq_gaps = 0
        self._rtp_seq_out_of_order = 0
        self._last_rtp_seq: Optional[int] = None
        # Decoded frames the leaky queue before the appsink dropped because the previous one was still being handled,
        # when (monotonic; the first OVERRUN_TIMES_MAX), and when the first frame was handed over (decoded_at).
        self._queue_overruns = 0
        self._queue_overrun_times: list[float] = []
        self._first_frame_at: Optional[float] = None
        # Longest time an RTP packet took from udpsrc to rtp_probe (ms; None before the first timestamped packet).
        self._receive_wait_max_ms: Optional[float] = None
        self._stats_start_time: Optional[float] = None
        self._last_log_time = time.monotonic()
        self._last_log_frame_count = 0
        # PTS-keyed (not a FIFO deque) so a frame dropped at the leaky appsink can't desync pairing.
        self._decode_start_times: dict[int, float] = {}
        self._latest_decode_latency_ms: Optional[float] = None
        # VA backend: the frame taker (open between start() and stop()), consecutive conversion failures, and what
        # get_va_timings() reports (kept after stop()).
        self._va = None
        self._va_failures = 0
        self._va_report: dict = {}
        self._decoder_out_ms: list[tuple[int, Optional[float]]] = []  # (pts, ms) in decoder output order

    def start(self) -> None:
        """Start receiving images in a background GStreamer thread."""
        if self._thread and self._thread.is_alive():
            return

        self._reset_runtime_state()
        pipeline_description = self._build_pipeline()
        self.logger.info("Starting GStreamer pipeline: %s", pipeline_description)

        try:
            pipeline = Gst.parse_launch(pipeline_description)
        except GLib.Error as exc:
            if self.decode_backend == rtp_pipeline.VA:
                raise RuntimeError(
                    f"Could not create the VA pipeline: {exc.message}. {self.va_decoder} only exists when a render "
                    "node is mapped into the container (docker run --device /dev/dri/renderD12x --group-add <render "
                    f"gid>) and the VA driver loads; check with gst-inspect-1.0 {self.va_decoder} and vainfo"
                ) from exc
            raise
        if not isinstance(pipeline, Gst.Pipeline):
            raise RuntimeError("GStreamer description did not create a Pipeline")

        appsink = pipeline.get_by_name("decoded_frames")
        if appsink is None:
            raise RuntimeError("Failed to find appsink named decoded_frames")
        appsink.connect("new-sample", self._image_callback)

        rtp_probe = pipeline.get_by_name("rtp_probe")
        if rtp_probe is not None:
            rtp_probe.connect("handoff", self._on_rtp_packet)

        h265_probe = pipeline.get_by_name("h265_probe")
        if h265_probe is not None:
            h265_probe.connect("handoff", self._on_h265_buffer)

        _leaky_queue(appsink).connect("overrun", self._on_queue_overrun)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        if self.decode_backend == rtp_pipeline.VA:
            self._va = self._open_va(pipeline)

        self._pipeline = pipeline
        self._loop = GLib.MainLoop()
        self._first_frame.clear()
        self._eos_or_error.clear()
        self._stop_requested = False

        self._thread = threading.Thread(target=self._run_loop, name="rtp-h265-video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop receiving images."""
        self._stop_requested = True

        if self._thread is not None:
            # quit() is a no-op on a loop that has not started running yet,
            # so retry until the loop thread actually exits (bounded).
            deadline = time.monotonic() + 2.0
            while self._thread.is_alive() and time.monotonic() < deadline:
                if self._loop is not None and self._loop.is_running():
                    self._loop.quit()
                self._thread.join(timeout=0.1)
            if self._thread.is_alive():
                self.logger.warning("RTP decoder thread did not exit within 2s")

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        # After NULL no streaming thread can be inside take(); libvapp keeps its own reference to the decoder's VA
        # display, so its surfaces outlive the pipeline safely.
        self._close_va()
        self._loop = None
        self.logger.info("Stopped decoder after %d frames", self._frame_count)

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for the first decoded image, matching TeleavatarROS2Interface."""
        self.logger.info("Waiting for initial video frame...")
        if self._first_frame.wait(timeout=timeout):
            self.logger.info("Initial video frame received")
            return True

        self.logger.error(
            "Timeout waiting for video after %.1fs: rtp_packets=%d h265_buffers=%d",
            timeout,
            self._rtp_packet_count,
            self._h265_buffer_count,
        )
        return False

    def get_observation(self) -> Optional[dict]:
        """Return current image observation, or None before the first frame."""
        with self.lock:
            if self.camera_name not in self.latest_images:
                return None
            return {"images": {name: image.copy() for name, image in self.latest_images.items()}}

    def get_latest_image(self) -> Optional[np.ndarray]:
        """Return the latest composite RGB image."""
        with self.lock:
            image = self.latest_images.get(self.camera_name)
            return None if image is None else image.copy()

    def get_latest_images(self) -> Dict[str, np.ndarray]:
        """Return all latest images, including the composite and split crops."""
        with self.lock:
            return {name: image.copy() for name, image in self.latest_images.items()}

    def get_latest_images_with_timestamps(self) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Return latest image copies and their receive timestamps."""
        with self.lock:
            images = {name: image.copy() for name, image in self.latest_images.items()}
            return images, dict(self.image_timestamps)

    def has_initial_frame(self) -> bool:
        return self._first_frame.is_set()

    def stream_ended(self) -> bool:
        """True once the pipeline hit EOS or an unrecoverable error (no new frames will arrive)."""
        return self._eos_or_error.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for EOS/error. Returns True if the stream ended."""
        return self._eos_or_error.wait(timeout=timeout)

    @property
    def pipeline_description(self) -> str:
        """The gst-launch description start() uses."""
        return self._build_pipeline()

    def get_stats(self) -> dict:
        """Counters since start(), the longest receive wait and the latest decode latency (ms, or None). Overruns also
        as times in seconds since the first frame was handed over (negative: before; the first OVERRUN_TIMES_MAX) and
        split at STARTUP_OVERRUN_WINDOW_S: at the start of the stream / after it (without a first frame, all after)."""
        first = self._first_frame_at
        times = [] if first is None else [round(t - first, 3) for t in self._queue_overrun_times]
        startup = sum(1 for t in times if t <= STARTUP_OVERRUN_WINDOW_S)
        return {
            "frames": self._frame_count,
            "rtp_packets": self._rtp_packet_count,
            "h265_buffers": self._h265_buffer_count,
            "rtp_seq_gaps": self._rtp_seq_gaps,
            "rtp_seq_out_of_order": self._rtp_seq_out_of_order,
            "queue_overruns": self._queue_overruns,
            "queue_overruns_startup": startup,
            "queue_overruns_after_startup": self._queue_overruns - startup,
            "queue_overrun_times_s": times,
            "receive_wait_ms_max": self._receive_wait_max_ms,
            "decode_latency_ms": self._latest_decode_latency_ms,
        }

    def get_va_timings(self) -> Optional[dict]:
        """VA backend only (else None): open() times of the frame taker (ms) and, with va_diagnostics, the per-frame
        conversion steps (ms, va_frame.STEPS) and the decoder's own latency (h265_probe -> decoder output, ms) of the
        first VA_DECODER_PROBE_FRAMES frames it outputs, as [pts, ms] (the pts pairs them with the frames taken; a
        frame the leaky queue drops has no sample). Available after stop()."""
        if self.decode_backend != rtp_pipeline.VA:
            return None
        report = dict(self._va_report)
        if self._va is not None:
            report.update(self._va_snapshot(self._va))
        report["decoder_first_frames_ms"] = [list(entry) for entry in self._decoder_out_ms]
        return report

    def _reset_runtime_state(self) -> None:
        self._frame_count = 0
        self._rtp_packet_count = 0
        self._h265_buffer_count = 0
        self._rtp_seq_gaps = 0
        self._rtp_seq_out_of_order = 0
        self._last_rtp_seq = None
        self._queue_overruns = 0
        self._queue_overrun_times = []
        self._first_frame_at = None
        self._receive_wait_max_ms = None
        self._stats_start_time = None
        self._last_log_time = time.monotonic()
        self._last_log_frame_count = 0
        self._decode_start_times.clear()
        self._latest_decode_latency_ms = None
        self._va_failures = 0
        self._va_report = {}
        self._decoder_out_ms = []

    def _build_pipeline(self) -> str:
        return rtp_pipeline.build_pipeline_description(
            port=self.port,
            payload=self.payload,
            udp_buffer_size=self.udp_buffer_size,
            decode_backend=self.decode_backend,
            decoder=self.decoder,
            cpu_decode_threads=self.cpu_decode_threads,
            cpu_thread_type=self.cpu_thread_type,
            cpu_convert_threads=self.cpu_convert_threads,
            va_decoder=self.va_decoder,
        )

    def _open_va(self, pipeline: Gst.Pipeline):
        """The frame taker's library, buffers and cv2 (prepare()), then READY (the decoder opens its VA display), then
        the taker's surfaces, context and warm-up, all before PLAYING: nothing of it is left for the first frame.
        udpsrc binds the port at READY and nobody reads the socket until PLAYING, so only what needs the VA display
        runs in between: the socket buffer (20 MB asked for, capped at rmem_max: 416 KiB in all on the B70) holds well
        under 0.1 s of the robot's stream. In B3's M4, while cv2 was still imported between READY and PLAYING, the
        first stats line of five of the six main.py runs already counted 910 missing packets (399 in the sixth); that
        most of them were lost in that stretch is inferred, not measured (the socket's drops were not read)."""
        try:
            from . import va_frame
        except ImportError:
            import va_frame

        taker = va_frame.VaFrameTaker(
            pipeline,
            element_name=rtp_pipeline.VA_DECODER_NAME,
            frame_size=self.va_frame_size,
            record_steps=self.va_diagnostics,
        )
        try:
            taker.prepare()
        except Exception as exc:
            raise RuntimeError(f"Could not set up the GPU conversion of the VA frames: {exc}") from exc
        if pipeline.set_state(Gst.State.READY) == Gst.StateChangeReturn.FAILURE:
            taker.close()
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                f"Could not open the VA decoder {self.va_decoder} (READY failed): is the render node mapped and "
                "readable by this user (docker run --device /dev/dri/renderD12x --group-add <render gid>)?"
            )
        try:
            open_ms = taker.open()
        except Exception as exc:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"Could not set up the GPU conversion of the VA frames: {exc}") from exc
        self.logger.info("VA frame taker ready (%s)", ", ".join(f"{k}={v:.1f}" for k, v in open_ms.items()))
        self._va_report = {"open_ms": open_ms}
        if self.va_diagnostics:
            decoder = pipeline.get_by_name(rtp_pipeline.VA_DECODER_NAME)
            decoder.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_decoder_output)
        return taker

    def _close_va(self) -> None:
        if self._va is None:
            return
        self._va_report.update(self._va_snapshot(self._va))
        self._va.close()
        self._va = None

    @staticmethod
    def _va_snapshot(taker) -> dict:
        steps = taker.steps()
        return {"resizes": taker.resizes, "steps_ms": steps if steps["strip"] else {}}

    def _on_decoder_output(self, _pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        """va_diagnostics: the decoder's own latency of the first frames (its input time is still in
        _decode_start_times: the appsink callback pairs it later)."""
        buffer = info.get_buffer()
        start = self._decode_start_times.get(buffer.pts) if buffer is not None else None
        pts = buffer.pts if buffer is not None else Gst.CLOCK_TIME_NONE
        self._decoder_out_ms.append((pts, None if start is None else (time.monotonic() - start) * 1000.0))
        if len(self._decoder_out_ms) >= VA_DECODER_PROBE_FRAMES:
            return Gst.PadProbeReturn.REMOVE
        return Gst.PadProbeReturn.OK

    def _run_loop(self) -> None:
        if self._pipeline is None or self._loop is None:
            raise RuntimeError("Pipeline was not initialized")

        pipeline = self._pipeline
        loop = self._loop
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.logger.error("Failed to set GStreamer pipeline to PLAYING")
            self._eos_or_error.set()
            return

        try:
            # stop() may have been requested before the loop started running;
            # quit() would have been lost, so check the flag first (stop()
            # also keeps retrying quit() until this thread exits).
            if not self._stop_requested:
                loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)

    def _image_callback(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        decoded_at = time.monotonic()
        out_buffer = sample.get_buffer()
        out_pts = out_buffer.pts if out_buffer is not None else Gst.CLOCK_TIME_NONE

        try:
            if self._va is not None:
                # VA memory: the frame is RGB in this process only once take() returns (GPU conversion, then the
                # RGBX -> RGB strip), so decode_latency ends here; the other backends hand over RGB at pull-sample.
                frame = self._va.take(sample)
                decoded_at = time.monotonic()
                self._va_failures = 0
            else:
                frame = _sample_to_rgb(sample)
        except Exception:
            self.logger.exception("Failed to convert decoded sample to RGB")
            if self._va is not None:
                self._va_failures += 1
                if self._va_failures >= VA_MAX_FAILURES:
                    self.logger.error("VA conversion failed %d times in a row; stopping the stream", self._va_failures)
                    self._eos_or_error.set()
                    if self._loop is not None:
                        self._loop.quit()
            return Gst.FlowReturn.OK

        if self._first_frame_at is None:
            self._first_frame_at = decoded_at
        split_frames = self._split_frame(frame)
        latency_ms = self._update_decode_latency(out_pts, decoded_at)
        if self.latency_hook is not None:
            try:
                self.latency_hook(out_pts, latency_ms, decoded_at)
            except Exception:
                self.logger.exception("latency_hook failed")

        timestamp = time.time()
        with self.lock:
            self.latest_images[self.camera_name] = frame
            self.image_timestamps[self.camera_name] = timestamp
            for name, split_frame in split_frames.items():
                self.latest_images[name] = split_frame
                self.image_timestamps[name] = timestamp

        now = time.monotonic()
        self._frame_count += 1
        self._first_frame.set()
        self._log_stats(frame, now)
        return Gst.FlowReturn.OK

    def _on_rtp_packet(self, identity: Gst.Element, buffer: Gst.Buffer) -> None:
        self._rtp_packet_count += 1
        clock = identity.get_clock()
        if clock is not None and buffer.pts != Gst.CLOCK_TIME_NONE:
            # udpsrc stamps each packet with the running time it read it; what is left is the wait before this probe
            # (the CPU description's receive queue), which decode_latency does not cover.
            wait_ms = (clock.get_time() - identity.get_base_time() - buffer.pts) / 1e6
            if self._receive_wait_max_ms is None or wait_ms > self._receive_wait_max_ms:
                self._receive_wait_max_ms = wait_ms
        seq_bytes = buffer.extract_dup(2, 2)  # RTP header bytes 2-3: sequence number
        if len(seq_bytes) != 2:
            return
        gaps, out_of_order, self._last_rtp_seq = rtp_pipeline.rtp_seq_step(
            self._last_rtp_seq, int.from_bytes(seq_bytes, "big")
        )
        self._rtp_seq_gaps += gaps
        self._rtp_seq_out_of_order += out_of_order

    def _on_queue_overrun(self, _queue: Gst.Element) -> None:
        self._queue_overruns += 1
        now = time.monotonic()
        if len(self._queue_overrun_times) < OVERRUN_TIMES_MAX:
            self._queue_overrun_times.append(now)
        if self._queue_overruns <= OVERRUN_LOGS_MAX:
            first = self._first_frame_at
            if first is None:
                when = "before the first frame was handed over"
            else:
                when = f"{now - first:.3f} s after the first frame"
            self.logger.warning(
                "The queue before the appsink dropped a decoded frame (overrun %d, %s)%s",
                self._queue_overruns,
                when,
                "; further overruns are only counted" if self._queue_overruns == OVERRUN_LOGS_MAX else "",
            )

    def _on_h265_buffer(self, _identity: Gst.Element, buffer: Gst.Buffer) -> None:
        self._h265_buffer_count += 1
        pts = buffer.pts
        if pts != Gst.CLOCK_TIME_NONE:
            self._decode_start_times[pts] = time.monotonic()
            # evict oldest; entries for dropped frames are never popped
            while len(self._decode_start_times) > 512:
                del self._decode_start_times[next(iter(self._decode_start_times))]

    def _update_decode_latency(self, pts: int, decoded_at: float) -> Optional[float]:
        start = self._decode_start_times.pop(pts, None)
        if start is None:
            return None
        self._latest_decode_latency_ms = max(0.0, (decoded_at - start) * 1000.0)
        return self._latest_decode_latency_ms

    def _split_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        height, width = frame.shape[:2]
        crops: Dict[str, np.ndarray] = {}
        for name, box in self.split_regions:
            x1, y1, x2, y2 = _normalized_box_to_pixels(box, width, height)
            if x2 > x1 and y2 > y1:
                crops[name] = frame[y1:y2, x1:x2].copy()
        return crops

    def _log_stats(self, frame: np.ndarray, now: float) -> None:
        if self._stats_start_time is None:
            self._stats_start_time = now
            self._last_log_time = now
            self._last_log_frame_count = self._frame_count
            return

        elapsed = now - self._last_log_time
        if elapsed < self.log_interval_s:
            return

        frames_since_last_log = self._frame_count - self._last_log_frame_count
        interval_fps = frames_since_last_log / elapsed if elapsed > 0.0 else 0.0
        overall_elapsed = now - self._stats_start_time
        overall_fps = (self._frame_count - 1) / overall_elapsed if overall_elapsed > 0.0 else 0.0
        self._last_log_time = now
        self._last_log_frame_count = self._frame_count

        latency = "n/a" if self._latest_decode_latency_ms is None else f"{self._latest_decode_latency_ms:.1f} ms"
        self.logger.info(
            "camera=%s frames=%d rtp_packets=%d h265_buffers=%d shape=%s fps=%.2f "
            "overall_fps=%.2f decode_latency=%s rtp_seq_gaps=%d queue_overruns=%d",
            self.camera_name,
            self._frame_count,
            self._rtp_packet_count,
            self._h265_buffer_count,
            tuple(frame.shape),
            interval_fps,
            overall_fps,
            latency,
            self._rtp_seq_gaps,
            self._queue_overruns,
        )

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self.logger.error("GStreamer error from %s: %s", message.src.get_name(), error)
            if debug:
                self.logger.error("GStreamer debug info: %s", debug)
            self._eos_or_error.set()
            if self._loop is not None:
                self._loop.quit()
        elif message.type == Gst.MessageType.EOS:
            self._eos_or_error.set()
            if self._loop is not None:
                self._loop.quit()


def _leaky_queue(appsink: Gst.Element) -> Gst.Element:
    """The queue feeding the appsink, which keeps only the newest frame. The description names it nowhere (the
    default description must stay byte-identical, and the CPU one has other queues), so find it as the upstream peer.
    """
    peer = appsink.get_static_pad("sink").get_peer()
    queue = peer.get_parent_element() if peer is not None else None
    if queue is None or queue.get_factory().get_name() != "queue":
        raise RuntimeError("Expected a queue right before the appsink")
    return queue


def _sample_to_rgb(sample: Gst.Sample) -> np.ndarray:
    caps = sample.get_caps()
    if caps is None or caps.get_size() == 0:
        raise RuntimeError("Decoded sample has no caps")

    structure = caps.get_structure(0)
    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))
    if structure.get_value("format") != "RGB":
        raise RuntimeError(f"Expected RGB sample, got {structure.get_value('format')}")

    buffer = sample.get_buffer()
    if buffer is None:
        raise RuntimeError("Decoded sample has no buffer")

    ok, map_info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Failed to map decoded frame buffer")

    try:
        row_bytes = width * 3
        raw = np.frombuffer(map_info.data, dtype=np.uint8)
        stride = raw.size // height if raw.size % height == 0 and raw.size // height >= row_bytes else row_bytes
        return raw[: height * stride].reshape((height, stride))[:, :row_bytes].reshape((height, width, 3)).copy()
    finally:
        buffer.unmap(map_info)


def _normalized_box_to_pixels(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, round(x1 * width))),
        max(0, min(height, round(y1 * height))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y2 * height))),
    )
