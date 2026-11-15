#!/usr/bin/env python3
"""Bounded TA2 arm-only grasp. Default is stored-observation inference, no ROS.

Run in the client container with /app as the deployed client source. --execute
is required for any robot command. A saved-observation preflight must pass
before the robot environment is even imported or constructed. Live data and
mode are then checked again before every command, including after inference.

Mode enums: dexteleop.com/docs/teleavatar-2, section 4.2: meta_mode 1=API;
left/right_arm_control_mode 1=joint, 0=end-effector. No override is provided.
The 19-dimensional policy remains unchanged; only its first 16 values are
sent to an environment with chassis disabled. No reset/zero/release is sent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import concurrent.futures
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import numpy as np

JOINTS = np.array([*range(7), *range(8, 15)])
GRIPPERS = np.array([7, 15])
CAMERAS = {"head_camera": (960, 960, 3), "left_color": (400, 640, 3), "right_color": (400, 640, 3)}
COMMAND_TOPICS = ("/api/left_arm/joint_cmd", "/api/right_arm/joint_cmd", "/api/left_gripper/cmd", "/api/right_gripper/cmd")


class GuardError(RuntimeError):
    pass


STARTUP_SENSOR_FAILURE = 3   # exit code: no live data at startup, nothing was commanded; vla_run.sh retries once


def bound_startup_wait(args, report, stopped):
    """The deployed client waits up to 30 s for the first video frame and both arm joint states and ignores STOP
    meanwhile. Bound it to --startup-timeout, end it at once on STOP, print progress for the web log, and record
    what was missing (report["startup_missing"]). Returns a function that restores the original method."""
    try:
        from examples.teleavatar_v2 import ros2_interface as _ri
        cls = _ri.TeleavatarROS2Interface
    except (ImportError, AttributeError):
        return lambda: None
    original = cls.wait_for_initial_data
    limit = float(getattr(args, "startup_timeout", 8.0))

    def bounded(self, timeout=10.0):
        started = time.monotonic()
        next_note = started + 2.0
        while True:
            video = bool(self._video.has_initial_frame())
            with self.lock:
                joints = [j for j in ("left_arm", "right_arm") if j in self.latest_joint_states]
            if video and len(joints) == 2:
                report["startup_wait_s"] = round(time.monotonic() - started, 3)
                return True
            now = time.monotonic()
            if stopped.is_set() or now - started >= min(timeout, limit):
                report["startup_missing"] = {"video": not video,
                                             "joints": [j for j in ("left_arm", "right_arm") if j not in joints]}
                print(f"startup: no live data after {now - started:.1f} s: video={'ok' if video else 'MISSING'}, "
                      f"joints={joints or 'MISSING'}", flush=True)
                return False
            if now >= next_note:
                print(f"startup: waiting for live data ({now - started:.0f} s): video={'ok' if video else 'missing'}, "
                      f"joints={joints or 'missing'}", flush=True)
                next_note += 2.0
            stopped.wait(0.05)

    cls.wait_for_initial_data = bounded

    def restore():
        cls.wait_for_initial_data = original
    return restore


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_mode(mode, age, max_age):
    if not isinstance(mode, dict) or age is None or not 0 <= age <= max_age:
        raise GuardError("Current mode is missing or stale")
    expected = {"meta_mode": 1, "left_arm_control_mode": 1, "right_arm_control_mode": 1,
                "enable_left_arm": True, "enable_right_arm": True}
    wrong = {k: mode.get(k) for k, value in expected.items() if mode.get(k) != value}
    if wrong:
        raise GuardError(f"Require API + both arms enabled in JOINT mode (arm enum 1); got {wrong}")


def validate_observation(obs):
    state = np.asarray(obs["observation/state"])
    if state.shape != (48,) or not np.isfinite(state).all():
        raise GuardError(f"Expected finite live 48D state, got {state.shape}")
    for name, shape in CAMERAS.items():
        image = np.asarray(obs[f"observation/images/{name}"])
        if image.shape != shape or image.dtype != np.uint8:
            raise GuardError(f"Invalid RGB image {name}: {image.shape}/{image.dtype}")
    return state


def validate_chunk(result):
    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.shape != (30, 19) or not np.isfinite(actions).all():
        raise GuardError(f"Expected finite (30,19) policy actions, got {actions.shape}")
    return actions


def guarded_action(target, measured, previous, *, period, speed, max_tracking, max_initial, lower=None, upper=None):
    target = np.asarray(target, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if target.shape != (19,) or measured.shape != (14,) or not np.isfinite(target).all() or not np.isfinite(measured).all():
        raise GuardError("Invalid action or measured joint shape/value")
    if not (0 < speed <= 1.0 and 0 < period <= 1 and 0 < max_tracking <= 0.20 and 0 < max_initial <= 0.60):
        raise GuardError("Invalid guard limits")
    desired = target[JOINTS]
    initial_gap = float(np.max(np.abs(desired - measured)))
    if previous is None:
        if initial_gap > max_initial:
            raise GuardError(f"Initial predicted target gap {initial_gap:.6f} rad exceeds {max_initial:.3f}; no reset or movement allowed")
        previous = measured.copy()
    else:
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != (14,) or not np.isfinite(previous).all():
            raise GuardError("Invalid previous joint command")
        tracking = float(np.max(np.abs(previous - measured)))
        if tracking > max_tracking:
            raise GuardError(f"Joint tracking error {tracking:.6f} rad exceeds {max_tracking:.3f}")
    # Always use ONE nominal period. A slow inference must not permit a larger
    # next step, because the robot would receive that whole position jump.
    command = target[:16].copy()
    command[JOINTS] = previous + np.clip(desired - previous, -speed * period, speed * period)
    command[GRIPPERS] = np.clip(command[GRIPPERS], -1.6, 2.0)
    if float(np.max(np.abs(command[JOINTS] - measured))) > max_tracking:
        raise GuardError("Next command would exceed the maximum measured tracking gap")
    limit_clamped = []
    if lower is not None and upper is not None:
        # Clamp to the physical limits like the official client (_clamp_arm_cmd) instead of aborting.
        joints = command[JOINTS]
        limit_clamped = [int(i) for i in np.flatnonzero((joints < np.asarray(lower)) | (joints > np.asarray(upper)))]
        command[JOINTS] = np.clip(joints, lower, upper)
    return command, {"raw_target_gap_rad": initial_gap,
                     "command_step_rad": float(np.max(np.abs(command[JOINTS] - previous))),
                     "tracking_error_rad": float(np.max(np.abs(previous - measured))),
                     "gripper_clamped": bool(np.any(command[GRIPPERS] != target[GRIPPERS])),
                     "limit_clamped_joints": limit_clamped}


def load_observation(path, prompt):
    with np.load(path, allow_pickle=False) as data:
        obs = {key: data[key].copy() for key in data.files if key.startswith("observation/")}
    obs["prompt"] = prompt
    validate_observation(obs)
    return obs


class BoundedPolicy:
    """Official wire format, with bounded connect and receive operations."""
    def __init__(self, host, port, timeout=2.0):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.timeout = timeout
        self.ws = connect(f"ws://{host}:{port}", compression=None, max_size=None,
                          open_timeout=timeout, close_timeout=1.0)
        try:
            self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=timeout))
        except BaseException:
            self.ws.close()
            raise

    def infer(self, observation, timeout=None):
        request = dict(observation)
        # Synchronous mode: always an unconditioned chunk. --rtc mode: the RTC broker sets the
        # rtc/* fields (previous chunk, prefix start, delay, horizon); keep them.
        request.setdefault("rtc/prev_chunk_id", 0)
        budget = min(self.timeout, timeout) if timeout else self.timeout
        if threading.current_thread() is not threading.main_thread():
            started = time.monotonic()
            self.ws.send(self.packer.pack(request))
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise GuardError("Policy send used the full inference deadline")
            response = self.ws.recv(timeout=remaining)
            if isinstance(response, str):
                raise GuardError(f"Policy server error: {response}")
            return self.codec.unpackb(response)
        # recv has its own deadline; SIGALRM additionally covers serialization
        # and a blocked socket send, so the complete call is bounded on Linux.
        def expired(_signum, _frame):
            raise GuardError(f"Policy round trip exceeded {budget:.3f} seconds")
        old_handler = signal.signal(signal.SIGALRM, expired)
        old_timer = signal.setitimer(signal.ITIMER_REAL, budget)
        try:
            started = time.monotonic()
            self.ws.send(self.packer.pack(request))
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise GuardError("Policy send used the full inference deadline")
            response = self.ws.recv(timeout=remaining)
        finally:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)
            signal.signal(signal.SIGALRM, old_handler)
        if isinstance(response, str):
            raise GuardError(f"Policy server error: {response}")
        return self.codec.unpackb(response)

    def close(self):
        self.ws.close()


def dump_observation(path, observation):
    np.savez_compressed(path, **observation)


class RecordingPolicy:
    """--rtc only: sits between RTCActionBroker and the policy (runs on the broker's worker thread) and keeps,
    per returned chunk id, the observation sent, round-trip seconds and the 19D chunk, so the main loop can log
    observation_NNN.npz / seconds / actions19 exactly like the synchronous mode when that chunk is installed."""

    def __init__(self, policy):
        self.policy = policy
        self.lock = threading.Lock()
        self.records = {}

    def infer(self, request, timeout=None):
        started = time.monotonic()
        result = self.policy.infer(request) if timeout is None else self.policy.infer(request, timeout=timeout)
        seconds = time.monotonic() - started
        chunk_id = result.get("rtc/chunk_id")
        with self.lock:
            self.records[chunk_id] = {
                "obs": {k: v for k, v in request.items() if not str(k).startswith("rtc/")},
                "seconds": seconds, "actions19": np.asarray(result["actions"]),
                "prev_chunk_id": request.get("rtc/prev_chunk_id"), "prefix_start": request.get("rtc/prefix_start")}
        return result

    def take(self, chunk_id):
        """Return the record of chunk_id and drop it and every older one (bootstrap / superseded chunks)."""
        with self.lock:
            record = self.records.pop(chunk_id, None)
            for key in [k for k in self.records if isinstance(k, int) and isinstance(chunk_id, int) and k < chunk_id]:
                del self.records[key]
        return record

    def __getattr__(self, name):
        return getattr(self.policy, name)


def execute(args, policy, destination, report, stopped):
    # No ROS imports, node, publishers or heartbeat exist before this point.
    from examples.teleavatar_v2.env import TeleavatarEnvironment
    from std_msgs.msg import String

    environment = None
    commands = None
    plans = None
    mode_lock = threading.Lock()
    mode_state = {"value": None, "time": None}
    count = 0
    attempted = 0
    started = None
    previous = None
    last_command = None
    last_obs = None
    status = "aborted"
    broker = None
    recorder = None
    writer = None

    def on_mode(message):
        try:
            mode = json.loads(message.data)
        except (ValueError, TypeError):
            mode = None
        with mode_lock:
            mode_state.update(value=mode, time=time.monotonic())

    try:
        # No timer publishes during inference or cleanup. Native noninterpolated
        # publishing is supported by the deployed interface; every setpoint is
        # constrained to <= speed / frequency radians before it reaches it.
        restore_wait = bound_startup_wait(args, report, stopped)
        try:
            environment = TeleavatarEnvironment(prompt=args.prompt, control_frequency=args.frequency,
                                                interpolate=False, enable_chassis=False, video_decode="va")
        finally:
            restore_wait()
        interface = environment._ros_interface
        interface.sensor_timeout = args.max_age
        assert "chassis" not in interface.action_publishers
        subscription = interface.create_subscription(String, "/api/current_mode", on_mode, 10)
        setup_deadline = time.monotonic() + 3.0
        while mode_state["value"] is None and not stopped.is_set() and time.monotonic() < setup_deadline:
            stopped.wait(0.02)
        if stopped.is_set():
            raise GuardError("Interrupted before initial live observation")

        fresh_calls = [0]

        def fresh():
            with mode_lock:
                value, received = mode_state["value"], mode_state["time"]
            age = None if received is None else time.monotonic() - received
            # /api/current_mode is published at exactly 1 Hz: a 1.0 s limit aborted healthy runs whenever one
            # message arrived a few ms late. Mode gets its own limit (one missed message); sensors keep max_age.
            validate_mode(value, age, getattr(args, "max_mode_age", args.max_age))
            obs = environment.get_observation()
            validate_observation(obs)
            # The deployed extractor takes the first seven joints. Check their
            # order explicitly so an unexpected message layout cannot move us.
            with interface.lock:
                for arm, prefix in (("left_arm", "l"), ("right_arm", "r")):
                    msg = interface.latest_joint_states.get(arm)
                    if msg is None or list(msg.name[:7]) != [f"{prefix}_joint{i}" for i in range(1, 8)]:
                        raise GuardError(f"Unexpected {arm} joint names/order")
            fresh_calls[0] += 1
            graph_check = not getattr(args, "rtc", False) or fresh_calls[0] % 9 == 1
            for topic in (COMMAND_TOPICS if graph_check else ()):
                publishers = interface.get_publishers_info_by_topic(topic)
                if len(publishers) != 1 or publishers[0].node_name != interface.get_name() or publishers[0].node_namespace != interface.get_namespace():
                    raise GuardError(f"Require this interface as the only arm/gripper publisher: {topic}, found {[(p.node_name, p.node_namespace) for p in publishers]}")
                if not interface.get_subscriptions_info_by_topic(topic):
                    raise GuardError(f"No matched subscriber for {topic}; command route is unavailable")
            # Other enable heartbeats do not themselves contain an arm target;
            # record them, while enforcing exclusive arm/gripper command sources.
            report["enable_publishers"] = [{"node": p.node_name, "namespace": p.node_namespace}
                for p in interface.get_publishers_info_by_topic("/api/fsm/enable")]
            report["last_mode"] = value
            report["last_mode_age_s"] = age
            return obs

        lower = np.concatenate([interface.joint_lower["left_arm"], interface.joint_lower["right_arm"]])
        upper = np.concatenate([interface.joint_upper["left_arm"], interface.joint_upper["right_arm"]])
        commands = (destination / "commands.jsonl").open("w", buffering=1)
        plans = (destination / "inferences.jsonl").open("w", buffering=1)
        report["live_inferences"] = 0
        actions = None
        cursor = args.replan_steps
        if getattr(args, "rtc", False):
            from openpi_client import rtc as _rtc
            recorder = RecordingPolicy(policy)
            writer = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="obs-writer")
            broker = _rtc.RTCActionBroker(policy=recorder, config=_rtc.RTCBrokerConfig(
                control_frequency=args.frequency, inference_delay=getattr(args, "rtc_delay", None), execution_horizon=getattr(args, "rtc_horizon", None)))
            report["rtc_mode"] = "async RTCActionBroker (official openpi_client.rtc)"
        next_command_at = time.monotonic()
        while count < args.max_actions and not stopped.is_set():
            now = time.monotonic()
            if started is not None and now >= started + args.max_seconds:
                status = "time_limit"
                break
            if now < next_command_at:
                stopped.wait(next_command_at - now)
                continue
            last_obs = fresh()
            if broker is not None:
                first = report["live_inferences"] == 0
                step_out = broker.infer(last_obs)        # first call: warm-up + delay calibration (arms idle)
                target = np.asarray(step_out["actions"], dtype=np.float64)
                if target.shape != (19,) or not np.isfinite(target).all():
                    raise GuardError(f"Expected one finite 19D action from the RTC broker, got {target.shape}")
                chunk_id = step_out.get("rtc/chunk_id")
                if first or chunk_id != report.get("rtc_last_chunk"):
                    report["live_inferences"] += 1
                    report["rtc_last_chunk"] = chunk_id
                    report["rtc_inference_delay"] = broker.inference_delay
                    report["rtc_execution_horizon"] = broker.execution_horizon
                    index = report["live_inferences"] - 1
                    entry = {"inference": index, "at": utc_now(), "rtc_chunk_id": chunk_id,
                             "delay_steps": broker.inference_delay, "horizon_steps": broker.execution_horizon}
                    record = recorder.take(chunk_id)
                    if record is not None:
                        # npz compression runs on a writer thread so the 45 Hz command loop never waits on disk.
                        writer.submit(dump_observation, destination / f"observation_{index:03d}.npz", record["obs"])
                        entry.update({"seconds": record["seconds"], "observation": f"observation_{index:03d}.npz",
                                      "rtc_prev_chunk_id": record["prev_chunk_id"], "rtc_prefix_start": record["prefix_start"],
                                      "actions19": record["actions19"].tolist()})
                    plans.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if first:
                        last_obs = fresh()           # calibration took ~1-2 s: re-read everything before moving
            elif actions is None or cursor >= args.replan_steps:
                infer_started = time.monotonic()
                remaining = 2.0 if started is None else started + args.max_seconds - infer_started
                if remaining <= 0:
                    status = "time_limit"
                    break
                result = policy.infer(last_obs, timeout=min(2.0, remaining))
                actions = validate_chunk(result)
                infer_seconds = time.monotonic() - infer_started
                if infer_seconds > 2.0:
                    raise GuardError(f"Inference exceeded2 seconds: {infer_seconds:.3f}")
                index = report["live_inferences"]
                dump_observation(destination / f"observation_{index:03d}.npz", last_obs)
                plans.write(json.dumps({"inference": index, "at": utc_now(), "seconds": infer_seconds,
                                        "observation": f"observation_{index:03d}.npz",
                                        "rtc_chunk_id": result.get("rtc/chunk_id"), "actions19": actions.tolist()}, ensure_ascii=False) + "\n")
                report["live_inferences"] += 1
                cursor = 0
                # Re-read ALL sensors and mode after inference AND disk logging.
                last_obs = fresh()
            measured = last_obs["observation/state"][JOINTS]
            command, diagnostics = guarded_action(target if broker is not None else actions[cursor], measured, previous,
                period=1.0 / args.frequency, speed=args.max_speed, max_tracking=args.max_tracking,
                max_initial=args.max_initial_gap, lower=lower, upper=upper)
            if stopped.is_set():
                break
            now = time.monotonic()
            if started is not None and now >= started + args.max_seconds:
                status = "time_limit"
                break
            # Calling this once sends one arm/gripper frame. There is no background
            # interpolation timer and no chassis publisher. Never send a final release.
            if started is None:
                started = now
            attempted += 1
            environment.apply_action({"actions": command})
            count += 1
            previous = command[JOINTS].copy()
            last_command = command.copy()
            commands.write(json.dumps({"actuation": count, "elapsed_s": now - started,
                                       "state48": last_obs["observation/state"].tolist(),
                                       "command16": command.tolist(), **diagnostics}, ensure_ascii=False) + "\n")
            cursor += 1
            next_command_at = time.monotonic() + 1.0 / args.frequency
        if count >= args.max_actions:
            status = "action_limit"
        elif stopped.is_set():
            status = "interrupted"
        report["status"] = status
    finally:
        try:
            if broker is not None:
                broker._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        if writer is not None:
            writer.shutdown(wait=True)              # finish the queued observation_NNN.npz files
        report["actuations"] = count
        report["attempted_actuations"] = attempted
        report["actuation_count_scope"] = "Completed apply_action calls; each attempts two arm and two gripper publishes. A failed call may partially publish."
        report["motion_elapsed_s"] = 0.0 if started is None else time.monotonic() - started
        report["last_command16"] = None if last_command is None else last_command.tolist()
        report["extra_final_release_command_sent"] = False
        if commands is not None:
            commands.close()
        if plans is not None:
            plans.close()
        if environment is not None:
            report["interface_publish_started"] = bool(environment._ros_interface._published_once)
            try:
                final = environment.get_observation()
                validate_observation(final)
                report["final_state48"] = final["observation/state"].tolist()
                dump_observation(destination / "final_observation.npz", final)
            except Exception as exc:
                report["final_observation_error"] = str(exc)
            report["environment_close_ok"] = bool(environment.close())
            if not report["environment_close_ok"]:
                raise GuardError("Environment teardown failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, help="Saved live NPZ used for the mandatory no-motion preflight")
    parser.add_argument("--output-dir", required=True, help="New directory; existing paths are refused")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--openpi-dir", default="/app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="抓取红色沙包")
    parser.add_argument("--frequency", type=float, default=45.0)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--max-actions", type=int, default=675)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--max-tracking", type=float, default=0.20)
    parser.add_argument("--max-initial-gap", type=float, default=0.60)
    parser.add_argument("--max-age", type=float, default=1.0, help="joint state / camera freshness limit (s)")
    parser.add_argument("--max-mode-age", type=float, default=2.5,
                        help="/api/current_mode freshness limit (s); the topic is 1 Hz, so 2.5 tolerates one missed message")
    parser.add_argument("--startup-timeout", type=float, default=8.0,
                        help="seconds to wait for the first video frame and both arm joint states (exit 3 if missing)")
    parser.add_argument("--rtc", action="store_true",
                        help="async Real-Time Chunking (official RTCActionBroker): infer in the background while executing")
    parser.add_argument("--rtc-delay", type=int, default=None, help="RTC: pin the inference delay in steps (default: measured)")
    parser.add_argument("--rtc-horizon", type=int, default=None, help="RTC: pin the execution horizon in steps (default: 2x delay)")
    args = parser.parse_args()
    limits = ((args.frequency, 45), (args.max_seconds, 120), (args.max_speed, 1.0),
              (args.max_tracking, 0.20), (args.max_initial_gap, 0.60), (args.max_age, 1), (args.max_mode_age, 3.0), (args.startup_timeout, 30.0))
    if any(not math.isfinite(value) or not 0 < value <= cap for value, cap in limits):
        parser.error("Guard values must be positive and cannot exceed their defaults")
    if not 1 <= args.max_actions <= 5400 or not 6 <= args.replan_steps <= 10 or args.frequency < 1:
        parser.error("Require max-actions1..5400, replan-steps6..10, frequency1..45")
    return args


def main():
    args = parse_args()
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, args.openpi_dir)
    sys.path.insert(0, str(Path(args.openpi_dir) / "packages/openpi-client/src"))
    report = {"started_at_utc": utc_now(), "execute_requested": args.execute, "actuations": 0,
              "chassis_publisher_created": False, "chassis_commands_sent": 0,
              "extra_final_release_command_sent": False, "task_success": None,
              "status": "preflight", "limits": vars(args),
              "rtc": "Unconditioned19D chunks; explicit prev_chunk_id0. No continuity asserted after rate limiting.",
              "joint_mode_enum_source": "https://www.dexteleop.com/docs/teleavatar-2 section4.2"}
    stopped = threading.Event()
    def stop(signum, _frame):
        report["signal"] = signal.Signals(signum).name
        stopped.set()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    policy = None
    exit_code = 2
    try:
        observation = load_observation(args.observation, args.prompt)
        policy = BoundedPolicy(args.host, args.port)
        result = policy.infer(observation)
        actions = validate_chunk(result)
        np.save(destination / "preflight_actions19.npy", actions)
        command, diagnostics = guarded_action(actions[0], observation["observation/state"][JOINTS], None,
            period=1 / args.frequency, speed=args.max_speed, max_tracking=args.max_tracking,
            max_initial=args.max_initial_gap)
        report["preflight"] = {"passed": True, "first_command16": command.tolist(), **diagnostics}
        if stopped.is_set():
            raise GuardError("Interrupted before execution")
        if args.execute:
            execute(args, policy, destination, report, stopped)
        else:
            report["status"] = "dry_run_passed"
        exit_code = 130 if stopped.is_set() else 0
    except BaseException as exc:
        report["status"] = "aborted"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if stopped.is_set() or isinstance(exc, KeyboardInterrupt):
            exit_code = 130
        elif report.get("startup_missing") and report.get("actuations", 0) == 0:
            exit_code = STARTUP_SENSOR_FAILURE
        else:
            exit_code = 2
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception as exc:
                report["policy_close_error"] = str(exc)
        report["finished_at_utc"] = utc_now()
        report["exit_code"] = exit_code
        (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "actuations": report["actuations"],
                          "error": report.get("error"), "report": str(destination / "report.json")}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
