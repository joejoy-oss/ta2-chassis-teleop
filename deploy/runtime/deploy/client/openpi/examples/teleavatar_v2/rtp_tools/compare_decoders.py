#!/usr/bin/env python3
"""Decode one H.265 Annex-B file with two GStreamer decode chains and compare their RGB frames.

    python3 compare_decoders.py clip.h265 --frames 900 \\
        --a-backend cpu --a-decode-threads 1 --b-backend cpu --b-decode-threads 4 --b-thread-type slice \\
        --json compare.json
    python3 compare_decoders.py clip.h265 --frames 900 --a-backend va --b-backend cpu --dump-a va_dump --dump-every 30
    python3 compare_decoders.py clip.h265 --a-dump va_dump --b-backend nvidia      # on another machine

Each side is `filesrc ! h265parse ! <rtp_pipeline.decode_fragment(...)> ! appsink`, i.e. the receiver's own decode
and colour conversion elements, run offline: nothing is dropped, and frames are paired by their index in the stream,
not by PTS (in live reception the PTS is the arrival time). Reported for B - A: per channel signed mean (bias), mean
absolute difference, p99 and max of the absolute difference, for whole frames and for the three policy views, both as
cropped and after resize_with_pad to 224x224 (frame_compare.py). Defaults: A is the CPU backend with the receiver's
defaults, B the NVIDIA backend (needs an NVIDIA GPU and the nvcodec plugin). Get the Annex-B file of a .rtpcap with
`stream_info.py X.rtpcap --write-h265 X.h265` (with --drop-incomplete-aus for a recording with lost packets).

The VA backend (an Intel GPU, the render node mapped into the container) decodes into VA memory and converts each
frame with va_frame.VaFrameTaker, the receiver's own path: the pipeline goes to READY, the taker opens (surfaces sized
--X-va-frame-size), then PLAYING; it closes after NULL. VA and NVIDIA are not on the same machine, so one side can be
saved and compared elsewhere: --dump-a DIR saves A's frames (every --dump-every-th compared frame, as .npy with the
frame index; frame_compare.FrameDump), and --a-dump DIR makes A read them back, B then decodes the file and is compared
on those frame indices only.

Needs gi (python3-gi, GStreamer 1.0); runs in the client image. Pulls frames with appsink's action signals, so the
GstApp typelib is not needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import frame_compare
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

_TELEAVATAR_DIR = str(pathlib.Path(__file__).resolve().parents[1])
if _TELEAVATAR_DIR not in sys.path:
    sys.path.append(_TELEAVATAR_DIR)  # rtp_pipeline.py lives one directory up

import rtp_pipeline  # noqa: E402


def decode_description(path: str, fragment: str) -> str:
    return (
        f'filesrc location="{path}" ! h265parse ! video/x-h265,stream-format=byte-stream,alignment=au '
        f"! {fragment} ! appsink name=frames sync=false max-buffers=2 drop=false emit-signals=false"
    )


class Decoder:
    """One decode pipeline; pull() returns the next RGB frame and its PTS, or None at the end of the stream. With
    va_frame_size, a VA pipeline whose frames go through va_frame.VaFrameTaker (see the module docstring)."""

    def __init__(self, label: str, description: str, va_frame_size: tuple[int, int] | None = None):
        self.label = label
        self.description = description
        self.pipeline = Gst.parse_launch(description)
        self.sink = self.pipeline.get_by_name("frames")
        self.bus = self.pipeline.get_bus()
        self.taker = None
        if va_frame_size is not None:
            import va_frame  # only for the VA backend: needs libvapp.so

            if self.pipeline.set_state(Gst.State.READY) == Gst.StateChangeReturn.FAILURE:
                self._raise_bus_error()
                raise RuntimeError(f"{label}: VA pipeline did not reach READY (render node mapped?): {description}")
            self.taker = va_frame.VaFrameTaker(
                self.pipeline, element_name=rtp_pipeline.VA_DECODER_NAME, frame_size=va_frame_size
            )
            opened = self.taker.open()
            print(f"{label}: VA frame taker open ({', '.join(f'{k}={v:.1f}' for k, v in opened.items())})")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._raise_bus_error()
            raise RuntimeError(f"{label}: pipeline did not start: {description}")

    def pull(self, timeout_s: float = 60.0) -> tuple[np.ndarray, int] | None:
        sample = self.sink.emit("try-pull-sample", int(timeout_s * Gst.SECOND))
        if sample is None:
            self._raise_bus_error()
            if self.sink.get_property("eos"):
                return None
            raise RuntimeError(f"{self.label}: no frame within {timeout_s} s")
        frame = self.taker.take(sample) if self.taker is not None else sample_to_rgb(sample)
        return frame, sample.get_buffer().pts

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)
        if self.taker is not None:
            self.taker.close()  # after NULL, as the receiver does

    def _raise_bus_error(self) -> None:
        message = self.bus.pop_filtered(Gst.MessageType.ERROR)
        if message is not None:
            error, debug = message.parse_error()
            raise RuntimeError(f"{self.label}: {message.src.get_name()}: {error.message} ({debug})")


def sample_to_rgb(sample: Gst.Sample) -> np.ndarray:
    """An RGB sample as an H x W x 3 uint8 array (rows may be padded in the buffer)."""
    structure = sample.get_caps().get_structure(0)
    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))
    if structure.get_value("format") != "RGB":
        raise RuntimeError(f"expected RGB, got {structure.get_value('format')}")
    buffer = sample.get_buffer()
    ok, info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("could not map the frame buffer")
    try:
        raw = np.frombuffer(info.data, dtype=np.uint8)
        row_bytes = 3 * width
        stride = raw.size // height if raw.size % height == 0 and raw.size // height >= row_bytes else row_bytes
        return raw[: height * stride].reshape(height, stride)[:, :row_bytes].reshape(height, width, 3).copy()
    finally:
        buffer.unmap(info)


def _side_arguments(parser: argparse.ArgumentParser, side: str, backend: str) -> None:
    upper = side.upper()
    parser.add_argument(f"--{side}-backend", choices=rtp_pipeline.DECODE_BACKENDS, default=backend)
    parser.add_argument(
        f"--{side}-decoder", default=rtp_pipeline.DEFAULT_NVIDIA_DECODER, help=f"{upper}: NVIDIA element"
    )
    parser.add_argument(f"--{side}-decode-threads", type=int, default=1, help=f"{upper}: avdec_h265 max-threads")
    parser.add_argument(f"--{side}-thread-type", default="auto", choices=rtp_pipeline.CPU_THREAD_TYPES)
    parser.add_argument(f"--{side}-convert-threads", type=int, default=4, help=f"{upper}: videoconvert n-threads")
    parser.add_argument(
        f"--{side}-va-decoder", default=rtp_pipeline.DEFAULT_VA_DECODER, help=f"{upper}: VA decoder element"
    )
    parser.add_argument(
        f"--{side}-va-frame-size",
        default="{}x{}".format(*rtp_pipeline.DEFAULT_VA_FRAME_SIZE),
        help=f"{upper}: VA output surfaces, WxH",
    )


def _fragment(args: argparse.Namespace, side: str) -> str:
    return rtp_pipeline.decode_fragment(
        getattr(args, f"{side}_backend"),
        decoder=getattr(args, f"{side}_decoder"),
        cpu_decode_threads=getattr(args, f"{side}_decode_threads"),
        cpu_thread_type=getattr(args, f"{side}_thread_type"),
        cpu_convert_threads=getattr(args, f"{side}_convert_threads"),
        va_decoder=getattr(args, f"{side}_va_decoder"),
    )


def _decoder(args: argparse.Namespace, side: str, description: str) -> Decoder:
    va_frame_size = None
    if getattr(args, f"{side}_backend") == rtp_pipeline.VA:
        width, height = getattr(args, f"{side}_va_frame_size").lower().split("x")
        va_frame_size = (int(width), int(height))
    return Decoder(side.upper(), description, va_frame_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h265", help="Annex-B H.265 file")
    parser.add_argument("--frames", type=int, default=0, help="compare this many frames (default 0: all)")
    parser.add_argument("--json", default=None, help="write the results here")
    parser.add_argument("--dump-a", default=None, help="save A's frames in this directory (see above)")
    parser.add_argument("--dump-every", type=int, default=1, help="with --dump-a: every N-th compared frame")
    parser.add_argument("--a-dump", default=None, help="A reads the frames saved in this directory (see above)")
    _side_arguments(parser, "a", rtp_pipeline.CPU)
    _side_arguments(parser, "b", rtp_pipeline.NVIDIA)
    args = parser.parse_args(argv)
    if args.a_dump and args.dump_a:
        parser.error("--a-dump and --dump-a exclude each other")

    Gst.init(None)
    path = str(pathlib.Path(args.h265).resolve())
    sides = {side: decode_description(path, _fragment(args, side)) for side in ("a", "b")}
    dump_in = frame_compare.FrameDump(args.a_dump) if args.a_dump else None
    if dump_in is not None:
        sides["a"] = f"frames saved in {args.a_dump}: {dump_in.index.get('a')}"
    for side, description in sides.items():
        print(f"{side.upper()}: {description}")
    decoders = [_decoder(args, side, sides[side]) for side in ("a", "b") if not (side == "a" and dump_in is not None)]
    dump_out = None
    if args.dump_a:
        dump_out = frame_compare.FrameDump(args.dump_a, {"file": path, "a": sides["a"]})
    wanted = set(dump_in.frames()) if dump_in is not None else None
    comparison = frame_compare.FrameComparison()
    frames = 0
    index = -1
    pts_mismatches = 0
    ended = []
    start = time.monotonic()
    try:
        while args.frames <= 0 or frames < args.frames:
            if wanted is not None and not wanted:
                break  # every saved frame compared
            pulled = [decoder.pull() for decoder in decoders]
            if any(item is None for item in pulled):
                ended = [decoder.label for decoder, item in zip(decoders, pulled, strict=True) if item is None]
                break
            index += 1
            if wanted is not None:
                if index not in wanted:
                    continue
                wanted.discard(index)
                pulled.insert(0, dump_in.load(index))
            (frame_a, pts_a), (frame_b, pts_b) = pulled
            pts_mismatches += pts_a != pts_b
            comparison.add(frame_a, frame_b)
            if dump_out is not None and frames % args.dump_every == 0:
                dump_out.save(index, frame_a, pts_a)
            frames += 1
            if frames % 300 == 0:
                print(f"  {frames} frames, {time.monotonic() - start:.0f} s", flush=True)
    finally:
        for decoder in decoders:
            decoder.stop()
    result = {
        "file": path,
        "frames_compared": frames,
        "ended_first": ended,  # both, or the one that had fewer frames
        "pts_mismatches": pts_mismatches,
        "elapsed_s": round(time.monotonic() - start, 1),
        "a": sides["a"],
        "b": sides["b"],
        "dumped_frames": dump_out.frames() if dump_out is not None else None,
        "saved_frames_not_reached": sorted(wanted) if wanted else [],
        **comparison.summary(),
    }
    _print_result(result)
    if args.json:
        with open(args.json, "w") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
    return 0


def _print_result(result: dict) -> None:
    print(
        f"compared {result['frames_compared']} frames in {result['elapsed_s']} s; ended first: {result['ended_first']}"
    )
    print(f"PTS mismatches (informational; frames are paired by index): {result['pts_mismatches']}")
    keys = ("identical_frames", "bias", "mean_abs", "p99_abs", "max_abs", "overall_mean_abs", "overall_p99_abs")
    print("frame:", {key: result["frame"].get(key) for key in keys})
    for name, view in result["views"].items():
        for size in ("native", "resized_224"):
            print(f"{name} ({view['view']}) {size}:", {key: view[size].get(key) for key in keys})


if __name__ == "__main__":
    sys.exit(main())
