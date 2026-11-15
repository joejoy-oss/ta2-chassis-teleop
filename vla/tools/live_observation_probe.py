#!/usr/bin/env python3
"""Capture one fresh TA2 observation without publishing robot commands.

Run inside the existing openpi-xpu-client container. This imports only its
independent RTP decoder, never TeleavatarEnvironment or TeleavatarROS2Interface.
The only application ROS endpoints created here are subscriptions. There is no
enable heartbeat, reset, arm/gripper/chassis command or policy execution.

The NPZ keys match the deployed client exactly, including the slash-delimited
image keys and 48-dimensional state. Images are the already-cropped RGB inner
eyes selected by the deployed client, not raw stereo/composite camera frames.
Do not crop, flip, rotate or convert them to BGR again before policy inference.

Exit codes: 0 = fresh observation saved, 2 = readiness timeout / decoder ended,
3 = probe error, 130 = interrupted. Observation readiness is not motion approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
import traceback


CAMERAS = {
    "head_camera": ("head_left_eye", (960, 960, 3)),
    "left_color": ("left_wrist_right_eye", (400, 640, 3)),
    "right_color": ("right_wrist_left_eye", (400, 640, 3)),
}
JOINT_TOPICS = {
    "left_arm": "/left_arm/joint_states",
    "right_arm": "/right_arm/joint_states",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-dir", default="/app")
    parser.add_argument("--output-dir", required=True,
                        help="New directory for this capture; refuses to reuse an existing directory")
    parser.add_argument("--prompt", default="抓取红色沙包")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Maximum acquisition wait in seconds (default: 15)")
    parser.add_argument("--max-age", type=float, default=1.0,
                        help="Maximum receiver age for every required sensor in seconds (default: 1)")
    parser.add_argument("--rtp-port", type=int, default=8890)
    parser.add_argument("--video-decode", choices=("va", "cpu"), default="va")
    args = parser.parse_args()
    if not (0 < args.timeout <= 120):
        parser.error("--timeout must be between 0 and 120 seconds")
    if not (0 < args.max_age <= 5):
        parser.error("--max-age must be between 0 and 5 seconds")
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_montage(images, destination):
    from PIL import Image, ImageDraw, ImageOps

    panel_w, panel_h = 480, 510
    montage = Image.new("RGB", (panel_w * len(CAMERAS), panel_h), "#17202a")
    draw = ImageDraw.Draw(montage)
    for index, (policy_key, (rtp_view, _)) in enumerate(CAMERAS.items()):
        original = images[policy_key]
        panel = ImageOps.contain(Image.fromarray(original, mode="RGB"), (480, 460))
        left = index * panel_w
        montage.paste(panel, (left + (panel_w - panel.width) // 2, 50 + (460 - panel.height) // 2))
        draw.text((left + 10, 8), policy_key, fill="white")
        draw.text((left + 10, 27), f"{rtp_view}  {original.shape[1]}x{original.shape[0]}", fill="#c5d4e3")
    montage.save(destination, format="JPEG", quality=92)


def main():
    args = parse_args()
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "purpose": "read-only live observation capture; does not establish readiness for robot motion",
        "observation_ready": False,
        "readiness_scope": "all three fresh camera views, both fresh arm states, valid shapes and finite state",
        "mode": None,
        "mode_age_s": None,
        "mode_fresh": False,
        "api_mode_observed": None,
        "joint_mode_observed": None,
        "motion_mode_compatible": False,
        "mode_contract_source": "https://www.dexteleop.com/docs/teleavatar-2",
        "timeout_s": args.timeout,
        "max_receiver_age_s": args.max_age,
        "prompt": args.prompt,
        "camera_format": "RGB uint8 HWC, already cropped to deployed policy eyes; no extra rotation/flip/crop",
        "camera_mapping": {
            key: {"rtp_view": view, "shape": list(shape)}
            for key, (view, shape) in CAMERAS.items()
        },
        "state_format": {
            "shape": [48], "dtype": "float32",
            "layout": "positions[0:16], velocities[16:32], efforts[32:48]",
            "left_arm_positions": "0:7", "right_arm_positions": "8:15",
            "gripper_slots": "7,15,23,31,39,47 are zero, matching deployed client",
        },
        "artifacts": {},
        "cleanup_errors": [],
    }
    interrupted = {"signal": None}

    def request_stop(signum, _frame):
        interrupted["signal"] = signal.Signals(signum).name

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    node = video = rclpy = None
    status = 3
    try:
        sys.path.insert(0, str(Path(args.openpi_dir).resolve()))
        import numpy as np
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from examples.teleavatar_v2.rtp_video_interface import RTPH265VideoInterface

        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node(
            f"sandbag_readonly_observation_probe_{os.getpid()}",
            enable_rosout=False,
            start_parameter_services=False,
        )
        samples = {}
        sample_times = {}
        received_counts = {}
        latest_mode = {"value": None, "received_at": None}

        def on_joints(name, message):
            samples[name] = message
            sample_times[name] = time.time()
            received_counts[name] = received_counts.get(name, 0) + 1

        def on_mode(message):
            try:
                latest_mode["value"] = json.loads(message.data)
            except (TypeError, ValueError):
                latest_mode["value"] = {"unparsed": message.data}
            latest_mode["received_at"] = time.time()

        subscriptions = [
            node.create_subscription(JointState, topic, lambda msg, key=key: on_joints(key, msg), 10)
            for key, topic in JOINT_TOPICS.items()
        ]
        subscriptions.append(node.create_subscription(String, "/api/current_mode", on_mode, 10))
        video = RTPH265VideoInterface(
            port=args.rtp_port, decode_backend=args.video_decode,
            cpu_decode_threads=8, cpu_thread_type="slice", cpu_convert_threads=4,
        )
        video.start()
        deadline = started + args.timeout
        selected_images = {}
        state = None
        reasons = ["waiting for sensor data"]
        while not interrupted["signal"] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.03, max(0.0, deadline - time.monotonic())))
            images, image_times = video.get_latest_images_with_timestamps()
            # Read the clock after copying the decoder's snapshot: its worker
            # may publish a new frame while we acquire the image lock.
            now = time.time()
            reasons = []
            selected_images = {}
            freshness = {}
            for key, (view, shape) in CAMERAS.items():
                stamp = image_times.get(view)
                age = None if stamp is None else now - stamp
                value = images.get(view)
                valid_shape = value is not None and value.shape == shape and value.dtype == np.uint8
                fresh = age is not None and 0 <= age <= args.max_age
                freshness[key] = {"received": value is not None, "age_s": age,
                                  "fresh": fresh, "valid_shape_dtype": bool(valid_shape)}
                if not valid_shape or not fresh:
                    reasons.append(f"camera {key}: missing, stale or wrong shape/dtype")
                else:
                    selected_images[key] = value

            state = np.zeros(48, dtype=np.float32)
            for key, offset, prefix in (("left_arm", 0, "l"), ("right_arm", 8, "r")):
                message = samples.get(key)
                stamp = sample_times.get(key)
                age = None if stamp is None else now - stamp
                fresh = age is not None and 0 <= age <= args.max_age
                valid = message is not None and len(message.position) >= 7
                expected_names = [f"{prefix}_joint{i}" for i in range(1, 8)]
                names_match = message is not None and list(message.name[:7]) == expected_names
                freshness[key] = {"received": message is not None, "age_s": age,
                                  "fresh": fresh, "positions_present": bool(valid),
                                  "joint_names_match": bool(names_match),
                                  "messages_received": received_counts.get(key, 0)}
                if not valid or not names_match or not fresh:
                    reasons.append(f"{key}: missing, stale or unexpected joint order")
                    continue
                for field, block in (("position", 0), ("velocity", 16), ("effort", 32)):
                    values = list(getattr(message, field))[:7]
                    state[block + offset:block + offset + len(values)] = values
            if not np.isfinite(state).all():
                reasons.append("state contains non-finite values")
            if video.stream_ended():
                reasons.append("RTP decoder ended or failed")
            report["freshness"] = freshness
            report["mode"] = latest_mode["value"]
            mode_age = None if latest_mode["received_at"] is None else now - latest_mode["received_at"]
            report["mode_age_s"] = mode_age
            report["mode_fresh"] = mode_age is not None and 0 <= mode_age <= args.max_age
            report["api_mode_observed"] = (
                latest_mode["value"].get("meta_mode") == 1
                if report["mode_fresh"] and isinstance(latest_mode["value"], dict) else None
            )
            mode = latest_mode["value"]
            mode_known = report["mode_fresh"] and isinstance(mode, dict)
            report["joint_mode_observed"] = (
                mode.get("left_arm_control_mode") == 1 and mode.get("right_arm_control_mode") == 1
                if mode_known else None
            )
            report["motion_mode_compatible"] = bool(
                mode_known and report["api_mode_observed"] and report["joint_mode_observed"]
                and mode.get("enable_left_arm") is True and mode.get("enable_right_arm") is True
            )
            report["unready_reasons"] = reasons
            if not reasons:
                report["captured_at_utc"] = utc_now()
                observation = {"observation/state": state, "prompt": np.asarray(args.prompt)}
                observation.update({f"observation/images/{key}": value for key, value in selected_images.items()})
                np.savez_compressed(destination / "observation.npz", **observation)
                write_montage(selected_images, destination / "camera_montage.jpg")
                report["artifacts"] = {"observation_npz": "observation.npz", "camera_montage": "camera_montage.jpg"}
                report["observation_ready"] = True
                status = 0
                break
            if video.stream_ended():
                status = 2
                break
        else:
            status = 130 if interrupted["signal"] else 2
        if status == 2:
            report["result"] = "no complete fresh observation before timeout or decoder termination"
        elif status == 0:
            report["result"] = "fresh observation saved for offline inference"
        elif status == 130:
            report["result"] = f"interrupted by {interrupted['signal']}"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        logging.exception("Read-only observation probe failed")
        status = 3
    finally:
        if video is not None:
            report["video_stats"] = video.get_stats()
            try:
                video.stop()
            except Exception as exc:
                report["cleanup_errors"].append(f"video.stop: {exc}")
        if node is not None:
            try:
                node.destroy_node()
            except Exception as exc:
                report["cleanup_errors"].append(f"node.destroy_node: {exc}")
        if rclpy is not None and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as exc:
                report["cleanup_errors"].append(f"rclpy.shutdown: {exc}")
        if report["cleanup_errors"]:
            status = 3
        report["finished_at_utc"] = utc_now()
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        report["exit_code"] = status
        (destination / "readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output_dir": str(destination), "observation_ready": report["observation_ready"],
                          "exit_code": status, "readiness": str(destination / "readiness.json")}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
