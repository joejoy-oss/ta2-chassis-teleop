"""Tests for frame_compare.py: the difference statistics, and the policy views (checked against the receiver's own
crop boxes and view mapping, read from rtp_video_interface.py and ros2_interface.py without importing them)."""

from __future__ import annotations

import ast
import pathlib

import frame_compare  # pytest puts this directory on sys.path
import numpy as np
import pytest

TELEAVATAR_DIR = pathlib.Path(__file__).resolve().parents[1]


def _module_constant(path: pathlib.Path, name: str):
    """The literal value assigned to `name` at the top level of a Python file."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_views_match_the_receiver():
    regions = _module_constant(TELEAVATAR_DIR / "rtp_video_interface.py", "TELEAVATAR_SPLIT_REGIONS")
    views = _module_constant(TELEAVATAR_DIR / "ros2_interface.py", "_POLICY_TO_RTP_VIEW")
    assert regions == frame_compare.SPLIT_REGIONS
    assert views == frame_compare.POLICY_VIEWS


def test_composite_is_1280_wide_and_2720_high():
    """The README's view sizes (H x W: head 960x960, wrists 400x640) come out of a 1280-wide, 2720-high frame."""
    frame = np.zeros((2720, 1280, 3), np.uint8)
    shapes = {view: frame_compare.crop_view(frame, view).shape for _name, view in frame_compare.POLICY_VIEWS.items()}
    assert shapes == {
        "head_left_eye": (960, 960, 3),
        "left_wrist_right_eye": (400, 640, 3),
        "right_wrist_left_eye": (400, 640, 3),
    }
    landscape = np.zeros((1280, 2720, 3), np.uint8)
    assert frame_compare.crop_view(landscape, "head_left_eye").shape == (452, 2040, 3)  # not the README's views


def test_policy_image():
    frame = np.full((2720, 1280, 3), 200, np.uint8)
    image = frame_compare.policy_image(frame, "left_wrist_right_eye")
    assert (image.shape, image.dtype) == ((224, 224, 3), np.uint8)
    assert (image[0, 0, 0], image[112, 112, 0]) == (0, 200)  # padded above and below a 1.6:1 view


def test_diff_stats():
    a = np.zeros((2, 2, 3), np.uint8)
    b = a.copy()
    b[..., 0] = [[0, 1], [2, 3]]
    b[..., 1] = 2
    a[0, :, 1] = 4  # B - A in channel 1: -2, -2, 2, 2
    stats = frame_compare.DiffStats()
    stats.add(a, b)
    stats.add(a, a)
    summary = stats.summary()
    assert (summary["frames"], summary["identical_frames"]) == (2, 1)
    assert summary["bias"] == [0.75, 0.0, 0.0]
    assert summary["mean_abs"] == [0.75, 1.0, 0.0]
    assert summary["p99_abs"] == [3, 2, 0]
    assert summary["max_abs"] == [3, 2, 0]
    assert summary["overall_mean_abs"] == round(14 / 24, 4)
    assert (summary["overall_p99_abs"], summary["overall_max_abs"]) == (3, 3)


def test_diff_stats_percentile_is_nearest_rank():
    a = np.zeros((1, 200, 1), np.uint8)
    b = a.copy()
    b[0, :3, 0] = 9  # 3 of 200 values differ: 1.5% > 1%, so the p99 is 9
    stats = frame_compare.DiffStats(channels=1)
    stats.add(a, b)
    assert stats.summary()["p99_abs"] == [9]
    b[0, 2, 0] = 0  # 2 of 200: exactly 1% above, the p99 is 0
    stats = frame_compare.DiffStats(channels=1)
    stats.add(a, b)
    assert stats.summary()["p99_abs"] == [0]


def test_diff_stats_rejects_mismatched_images():
    with pytest.raises(ValueError, match="uint8"):
        frame_compare.DiffStats().add(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 3, 3), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        frame_compare.DiffStats().add(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.float32))


def test_frame_comparison_localizes_differences():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (2720, 1280, 3), dtype=np.uint8)
    b = a.copy()
    b[960:1920, 160:1120] = np.clip(a[960:1920, 160:1120].astype(int) + 1, 0, 255)  # only the head left eye
    comparison = frame_compare.FrameComparison()
    comparison.add(a, b)
    summary = comparison.summary()
    head = summary["views"]["head_camera"]
    assert head["view"] == "head_left_eye"
    assert 0.99 < head["native"]["bias"][0] <= 1.0
    assert head["native"]["max_abs"] == [1, 1, 1]
    assert head["resized_224"]["frames"] == 1
    assert head["resized_224"]["max_abs"][0] <= 1
    for key in ("left_color", "right_color"):
        assert summary["views"][key]["native"]["identical_frames"] == 1
        assert summary["views"][key]["resized_224"]["identical_frames"] == 1
    assert summary["frame"]["identical_frames"] == 0


def test_frame_dump_round_trip(tmp_path):
    dump = frame_compare.FrameDump(tmp_path / "dump", {"file": "x.h265", "side": "va"})
    frames = {i: np.full((4, 6, 3), i, dtype=np.uint8) for i in (0, 30, 60)}
    for i, frame in frames.items():
        dump.save(i, frame, 1000 * i)
    again = frame_compare.FrameDump(tmp_path / "dump")
    assert again.index["file"] == "x.h265"
    assert again.frames() == [0, 30, 60]
    frame, pts = again.load(30)
    np.testing.assert_array_equal(frame, frames[30])
    assert pts == 30000
