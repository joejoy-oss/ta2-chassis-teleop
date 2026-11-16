"""Pixel statistics for comparing two decoders' RGB output of the same stream, and the views the client gives the
policy, and FrameDump (frames saved to compare across machines). Used by compare_decoders.py; numpy and openpi_client
only, no gi.

DiffStats accumulates B - A over many frames: per channel the signed mean (a colour bias), the mean absolute
difference and the histogram of absolute differences, from which the p99 and the maximum come. The policy views are
cut out of the composite exactly as RTPH265VideoInterface does, and also compared after
openpi_client.image_tools.resize_with_pad to 224x224, roughly what the model sees.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from openpi_client import image_tools

# Same boxes as rtp_video_interface.TELEAVATAR_SPLIT_REGIONS (x1, y1, x2, y2 as fractions of width and height) and
# same policy views as ros2_interface._POLICY_TO_RTP_VIEW; frame_compare_test.py checks both against those files.
SPLIT_REGIONS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("head_right_eye", (0.1250, 0.0000, 0.8750, 0.3529411765)),
    ("head_left_eye", (0.1250, 0.3529411765, 0.8750, 0.7058823529)),
    ("right_wrist_left_eye", (0.0000, 0.7058823529, 0.5000, 0.8529411765)),
    ("right_wrist_right_eye", (0.5000, 0.7058823529, 1.0000, 0.8529411765)),
    ("left_wrist_left_eye", (0.0000, 0.8529411765, 0.5000, 1.0000)),
    ("left_wrist_right_eye", (0.5000, 0.8529411765, 1.0000, 1.0000)),
)
POLICY_VIEWS = {
    "head_camera": "head_left_eye",
    "left_color": "left_wrist_right_eye",
    "right_color": "right_wrist_left_eye",
}
POLICY_IMAGE_SIZE = 224


def box_to_pixels(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel corners of a fractional box, rounded and clamped like rtp_video_interface does."""
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, round(x1 * width))),
        max(0, min(height, round(y1 * height))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y2 * height))),
    )


def crop_view(frame: np.ndarray, view: str) -> np.ndarray:
    """The split view `view` (a SPLIT_REGIONS name) of an H x W x 3 composite frame."""
    box = dict(SPLIT_REGIONS)[view]
    x1, y1, x2, y2 = box_to_pixels(box, frame.shape[1], frame.shape[0])
    return frame[y1:y2, x1:x2]


def policy_image(frame: np.ndarray, view: str) -> np.ndarray:
    """The split view resized with padding to 224x224, as the policy receives it."""
    return image_tools.resize_with_pad(crop_view(frame, view), POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)


class DiffStats:
    """B - A statistics of uint8 images, accumulated over frames."""

    def __init__(self, channels: int = 3):
        self.channels = channels
        self.frames = 0
        self.identical_frames = 0
        self.pixels = 0  # per channel
        self.signed_sum = np.zeros(channels, np.int64)
        self.abs_sum = np.zeros(channels, np.int64)
        self.abs_hist = np.zeros((channels, 256), np.int64)

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError(f"need two uint8 images of one shape, got {a.shape} {a.dtype} and {b.shape} {b.dtype}")
        diff = (b.astype(np.int16) - a.astype(np.int16)).reshape(-1, self.channels)
        absolute = np.abs(diff)
        self.signed_sum += diff.sum(axis=0, dtype=np.int64)
        self.abs_sum += absolute.sum(axis=0, dtype=np.int64)
        # One bincount for all channels: channel c counts in bins 256*c ... 256*c + 255.
        shifted = absolute + (256 * np.arange(self.channels, dtype=np.int16))
        self.abs_hist += np.bincount(shifted.ravel(), minlength=256 * self.channels).reshape(self.channels, 256)
        self.pixels += diff.shape[0]
        self.frames += 1
        self.identical_frames += not absolute.any()

    def summary(self) -> dict:
        """Per channel (R, G, B) and overall: signed mean of B - A (bias), mean |B - A|, p99 and max of |B - A|."""
        if self.pixels == 0:
            return {"frames": 0}
        total_hist = self.abs_hist.sum(axis=0)
        return {
            "frames": self.frames,
            "identical_frames": self.identical_frames,
            "bias": [round(float(v), 4) for v in self.signed_sum / self.pixels],
            "mean_abs": [round(float(v), 4) for v in self.abs_sum / self.pixels],
            "p99_abs": [_hist_percentile(h, 99) for h in self.abs_hist],
            "max_abs": [_hist_max(h) for h in self.abs_hist],
            "overall_mean_abs": round(float(self.abs_sum.sum() / (self.pixels * self.channels)), 4),
            "overall_p99_abs": _hist_percentile(total_hist, 99),
            "overall_max_abs": _hist_max(total_hist),
        }


def _hist_percentile(hist: np.ndarray, point: float) -> int:
    """Smallest value v with at least point% of the samples <= v (nearest rank)."""
    total = int(hist.sum())
    if total == 0:
        return 0
    rank = -(-total * point // 100)
    return int(np.searchsorted(np.cumsum(hist), rank))


def _hist_max(hist: np.ndarray) -> int:
    nonzero = np.flatnonzero(hist)
    return int(nonzero[-1]) if nonzero.size else 0


class FrameComparison:
    """Statistics of two decoders over whole frames and over the policy views, native and at 224x224."""

    def __init__(self):
        self.frame = DiffStats()
        self.views = {key: {"native": DiffStats(), "resized_224": DiffStats()} for key in POLICY_VIEWS}

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        self.frame.add(a, b)
        for key, view in POLICY_VIEWS.items():
            self.views[key]["native"].add(crop_view(a, view), crop_view(b, view))
            self.views[key]["resized_224"].add(policy_image(a, view), policy_image(b, view))

    def summary(self) -> dict:
        return {
            "frame": self.frame.summary(),
            "views": {
                key: {"view": POLICY_VIEWS[key], **{size: stats.summary() for size, stats in sizes.items()}}
                for key, sizes in self.views.items()
            },
        }


class FrameDump:
    """RGB frames of one decoder saved for a comparison on another machine (compare_decoders.py --dump-a / --a-dump):
    DIR/frame_NNNNN.npy per frame (NNNNN: index in the stream), DIR/index.json with the indices, their PTS and what
    decoded them. Opened for writing with `meta`, for reading without."""

    def __init__(self, directory: str | pathlib.Path, meta: dict | None = None):
        self.directory = pathlib.Path(directory)
        if meta is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.index = {**meta, "frames": [], "pts": []}
        else:
            self.index = json.loads((self.directory / "index.json").read_text())

    def path(self, frame_index: int) -> pathlib.Path:
        return self.directory / f"frame_{frame_index:05d}.npy"

    def save(self, frame_index: int, frame: np.ndarray, pts: int | None) -> None:
        np.save(self.path(frame_index), frame)
        self.index["frames"].append(frame_index)
        self.index["pts"].append(pts)
        (self.directory / "index.json").write_text(json.dumps(self.index, indent=1) + "\n")

    def frames(self) -> list[int]:
        return list(self.index["frames"])

    def load(self, frame_index: int) -> tuple[np.ndarray, int | None]:
        position = self.index["frames"].index(frame_index)
        return np.load(self.path(frame_index)), self.index["pts"][position]
