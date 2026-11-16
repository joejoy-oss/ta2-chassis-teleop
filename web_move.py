#!/usr/bin/env python3
"""TA2 web remote (chassis, arms, VLA grasp). Runs on B70.

Page (Chinese, single column, phone friendly): top bar with connection / API mode / current activity
and the only red button, 急停 (E-STOP, also the Esc key); then 全流程, VLA 抓取, 路线, 底盘, and folded
路径与轨迹 and 日志 sections. The chassis centre 停 is the soft STOP described below.

Start:  nohup python3 /root/web_move.py > /root/web_move.log 2>&1 &
Open:   http://192.168.31.228:8088  (any browser on the robot's router network)

Click a direction (release within 0.3 s): move 10 cm, or turn 15 deg for CCW/CW.
Press and hold: keep moving at 0.10 m/s (turning at 0.2 rad/s) until release (dead-man: stops
0.4 s after the browser stops sending heartbeats; a single hold is capped at 10 s).

Trajectory: Start / Stop recording, Save (CSV download + copy in /root/ta2_trails/), Reset
(odometry origin = current pose, trail cleared). CSV columns: t_s, x_m (right), y_m (forward),
heading_deg (CCW), all relative to the origin.

Paths: CSV files of that format in /root/ta2_paths/ are listed on the page (Load adds more from
the browser). Run follows one path from the current pose; Run all follows them in name order,
each starting where the previous one ended. Corners are driven through without stopping; in Run all
the robot stands still for PATH_PAUSE seconds after each path before starting the next one. Paths do
not depend on the page: once started they run to the end even if the browser closes or the phone
loses Wi-Fi; only STOP or E-STOP ends them early (plus the robot-side checks in teleop.py).
Routes: green to red (001 -> 002) and red to green (003 -> 004) use the same sequence runner.
Once a route starts, only E-STOP is accepted; all other web control commands are refused,
including STOP and release. Page polling is optional; the server supplies the path heartbeat.

Arms: "Raise arms" / "Lower arms" / "Start pose" run arm/arm_pose.py (standalone, /outputs/arm/ in the container)
as a separate process; its output goes to the log. Arms and chassis never move at the same time.
STOP also interrupts the arm script, which then holds where it is and pauses the FSM.
VLA grasp: "VLA grasp 60 s" runs /root/vla_run.sh (Start pose, then the step-4999 policy for 60 s at
max 1.0 rad/s, prompt 抓取沙包). Needs the policy container running. STOP or E-STOP interrupts it.
E-STOP (red header button, or the Esc key): stops the chassis and any path, interrupts every arm or
VLA script in the container (arm_pose.py, guarded_grasp.py; force-killed after 3 s), then publishes
/api/fsm/enable 0.0 for 2 s so the robot PAUSEs (zero-velocity hold). It is a software stop: keep
the hardware emergency stop within reach.

Motion is done by teleop.py, a long-running ROS node inside the openpi-xpu-client
container, fed over stdin. If this server exits, teleop.py sees EOF, sends zero and exits.
"""
import atexit
import collections
import copy
import datetime
import html
import json
import os
import shlex
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mission as mission_module
import queue
import subprocess
import threading
import time
import uuid
from urllib.parse import parse_qs, unquote, urlsplit
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8088"))
DEPLOY = os.environ.get("DEPLOY", "/data/ta2/software/deploy")
ROBOT = "192.168.31.218"
CONTAINER = "openpi-xpu-client"
TELEOP = ["docker", "exec", "-i", CONTAINER, "/usr/local/bin/openpi-client-entrypoint.sh",
          "python3", "-u", "/outputs/tools/teleop.py"]
if os.environ.get("TELEOP_CMD"):              # for offline testing
    TELEOP = shlex.split(os.environ["TELEOP_CMD"])
DIRS = ("forward", "back", "left", "right", "ccw", "cw")
TRAIL_DIR = os.environ.get("TRAIL_DIR", "/root/ta2_trails")
PATH_DIR = os.environ.get("PATH_DIR", "/root/ta2_paths")
PATH_PAUSE = float(os.environ.get("PATH_PAUSE", "10"))   # Run all: seconds to stand still after each path
ROUTES = {"g2r": ("绿 → 红", ["001", "002"]), "r2g": ("红 → 绿", ["003", "004"])}
ARM_CMD = shlex.split(os.environ.get("ARM_CMD", f"docker exec {CONTAINER} /usr/local/bin/openpi-client-entrypoint.sh "
                                    "python3 -u /outputs/arm/arm_pose.py"))
ARM_STOP = shlex.split(os.environ.get("ARM_STOP", f"docker exec {CONTAINER} pkill -INT -f arm_pose.py"))
# E-STOP: every process in the container whose command line matches this is stopped
ESTOP_PROCS = "guarded_grasp.py|arm_pose.py|grasp_monitor.py|live_observation_probe.py"
DEXEC = shlex.split(os.environ.get("ESTOP_DEXEC", f"docker exec {CONTAINER}"))   # "echo" for offline tests
FSM_PAUSE = ["/usr/local/bin/openpi-client-entrypoint.sh", "timeout", "8", "ros2", "topic", "pub",
             "-r", "20", "-t", "40", "/api/fsm/enable", "std_msgs/msg/Float32", "{data: 0.0}"]
VLA_CMD = shlex.split(os.environ.get("VLA_CMD", "/root/vla_run.sh"))
VLA_SECONDS, VLA_SPEED = "60", "1.0"
POLICY_CONTAINER = os.environ.get("POLICY_CONTAINER", "")
with open(os.path.join(os.path.dirname(__file__), "vla_models.json")) as model_file:
    MODEL_CONFIG = json.load(model_file)
VLA_MODELS = {model["id"]: model for model in MODEL_CONFIG["models"]}
DEFAULT_MODEL = MODEL_CONFIG["default"]
assert DEFAULT_MODEL in VLA_MODELS, "default VLA model must be registered"
VLA_SELECTION_FILE = os.environ.get("VLA_SELECTION_FILE", "/root/ta2_vla_selected.json")
try:
    with open(VLA_SELECTION_FILE) as selection_file:
        saved_model = json.load(selection_file)["model"]
except (OSError, ValueError, KeyError, TypeError):
    saved_model = DEFAULT_MODEL
if not isinstance(saved_model, str) or saved_model not in VLA_MODELS:
    saved_model = DEFAULT_MODEL
ARM_TIME = {"raise": 8, "lower": 15, "start": 10, "start_hold": 10, "down5": 8, "fwd5_left": 8, "fwd5_right": 8, "open_right": 3, "close_right": 3, "open_left": 3, "close_left": 3}
ARM_DOWN_CMD = shlex.split(os.environ.get("ARM_DOWN_CMD", "/root/arm_down.sh"))   # base command; callers append side and cm
ARM_MOVE_CMD = shlex.split(os.environ.get("ARM_MOVE_CMD", "/root/arm_move.sh"))   # + side forward_cm left_cm up_cm
# start = VLA training start pose (arm/train_start_pose.json, first frame of a training episode)
paths = {}                  # name -> {"points": [[x, y, th], ...], "length": m, "end": [dx, dy]}
seq = {"names": [], "i": -1, "abort": False, "route": None, "pause_until": None}
seq_lock = threading.RLock()  # serialize web commands and sequence starts, including nested dispatch
path_done = threading.Event()
path_result = {"why": None}
last_poll = [0.0]

state = {"connected": False, "moving": None, "mode_ok": None, "last": "starting", "clicks": 0, "holds": 0,
         "pose": [0.0, 0.0, 0.0], "epoch": int(time.time()),    # epoch changes on restart and on trail reset
         "recording": False, "rec_t0": None, "active_path": None, "path_anchor": [0.0, 0.0, 0.0],
         "arm": None, "arm_last": "arms: not moved yet", "estop": False, "estop_last": None,
         "vla": None, "vla_model": None, "vla_selected": saved_model,
         "vla_last": "VLA: not run yet", "route_last": None}
trail = []                  # [[x, y, th, t], ...]; metres / rad / s; x = right, y = forward of the origin
TRAIL_MAX = 20000
log = collections.deque(maxlen=60)
lock = threading.Lock()
proc = {"p": None}
ACTION_LOG = os.environ.get("ACTION_LOG", "/root/ta2_actions.jsonl")
actions = collections.deque(maxlen=200)   # API records only, newest first
current_action = [None]                  # worker reservation, retained until motion really ends
pending_motion = [None]                  # includes web commands awaiting teleop acknowledgement
terminal_claims = {}                     # atomic pop chooses exactly one terminal status
log_claims = {}                          # append once, after the worker supplies its result
action_logs = queue.SimpleQueue()
estop_generation = [0]  # catches an E-STOP that finishes during API preflight


class APIError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def busy():
    return bool(mission.active or current_action[0] or pending_motion[0] or state["moving"] or
                seq["names"] or state["arm"] or state["vla"])


def action_log_loop():
    while True:
        record = action_logs.get()
        try:
            with open(ACTION_LOG, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            note(f"action log: {e}")


def end_action(action, result, success=False, code=None, release=True):
    # Normally called under seq_lock. E-STOP may call without it: a single dict
    # update publishes the terminal status. Logging is queued when the worker exits.
    if action is None:
        return
    if terminal_claims.pop(action["id"], None) is action:
        why = str(result.get("why") or result.get("error") or result.get("result") or result.get("status") or result)
        if code is None and not success:
            text = str(result.get("path_why") or why).lower()
            if "not in api mode" in text:
                code = "NOT_API_MODE"
            elif "feedback" in text or "connection lost" in text:
                code = "FEEDBACK_LOST"
            elif "disconnected" in text:
                code = "TELEOP_DISCONNECTED"
            elif "timeout" in text or "timed out" in text:
                code = "TIMEOUT"
            elif "stuck" in text:
                code = "STUCK"
            elif text in ("stopped", "released") or text.startswith("stopped before ") or "cancel" in text:
                code = "STOPPED"
            else:
                code = "SCRIPT_ERROR" if action["type"] in ("arm.pose", "arm.down", "arm.move", "vla.grasp") else "ACTION_FAILED"
        action.update(status="succeeded" if success else "aborted" if code in ("ESTOP", "STOPPED") else "failed",
                      ended_at=timestamp(), result=result,
                      error=None if success else {"code": code, "message": why})
    if release:
        if action["status"] == "aborted":
            action["result"] = {**result, "interrupted_by": action["error"]["code"]}
        if log_claims.pop(action["id"], None) is action:
            action_logs.put(copy.deepcopy(action))
        if current_action[0] is action:
            current_action[0] = None


def interrupt_action(code):
    # No seq_lock, subprocess, file write or network call on the E-STOP request path.
    action = current_action[0]
    if action:
        why = "emergency stop" if code == "ESTOP" else "stopped"
        result = {"why": why}
        if action["type"] == "vla.grasp":
            tag = state["vla"]
            model_id = action["params"].get("model", state["vla_selected"])
            result.update(model=model_id, rtc=model_rtc(VLA_MODELS[model_id]),
                          seconds=action["params"].get("seconds", VLA_MODELS[model_id]["seconds"]),
                          grasped=False if action["params"].get("stop_on_grasp") else None, hand=None,
                          status="aborted", actuations=None, error=why, tag=tag,
                          report=f"/root/openpi-xpu/client/outputs/{tag}/report.json")
        end_action(action, result, code=code, release=False)


def start_sequence(names, route=None, action=None):
    if state["estop"] or (action and action["status"] != "running"):
        end_action(action, {"why": "emergency stop"}, code="ESTOP")
        return
    last_poll[0] = time.monotonic()
    seq.update(names=names, i=-1, abort=False, route=route, pause_until=None)
    threading.Thread(target=run_sequence, args=(names, action), daemon=True).start()


def start_arm(pose, action=None, down=None, move=None):
    state["arm"] = pose
    detail = (f"{move[0]}, forward {move[1]:g} / left {move[2]:g} / up {move[3]:g} cm" if move else
              f"{down[0]}, {down[1]} cm" if down else f"{ARM_TIME[pose]} s")
    note(f"arms: {pose} ({detail})")
    threading.Thread(target=run_arm, args=(pose, action, down, move), daemon=True).start()


def model_rtc(model):
    """RTC entries run through a *_rtc.sh wrapper (RTC=1); an explicit "rtc" key wins."""
    return bool(model.get("rtc", any(str(c).endswith("_rtc.sh") for c in model["command"])))


def public_model(model_id):
    model = VLA_MODELS[model_id]
    out = {key: model[key] for key in ("id", "name", "training", "prompt", "seconds", "speed")}
    out["rtc"] = model_rtc(model)
    return out


def model_services():
    """Read service states only; never start, stop or reload a policy container."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.Name}} {{.State.Running}}"]
            + [model["container"] for model in VLA_MODELS.values()],
            capture_output=True, text=True, timeout=3)
        running = {fields[0].lstrip("/"): fields[1] == "true"
                   for line in result.stdout.splitlines() if len(fields := line.split()) == 2}
    except (OSError, subprocess.TimeoutExpired):
        running = {}
    return [{**public_model(model_id), "running": running.get(model["container"], False)}
            for model_id, model in VLA_MODELS.items()]


def select_model(model_id):
    """Called under seq_lock, so changing selection and dispatching motion are serialized."""
    temporary = VLA_SELECTION_FILE + ".tmp"
    try:
        with open(temporary, "w") as selection_file:
            json.dump({"model": model_id}, selection_file)
        os.replace(temporary, VLA_SELECTION_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    state["vla_selected"] = model_id
    note("VLA model selected: " + VLA_MODELS[model_id]["name"])


def check_policy(port=None, model_id=None):
    if os.environ.get("VLA_CMD"):  # custom/offline script supplies its own policy
        return
    model = VLA_MODELS.get(model_id)
    if model is None and port is not None:
        model = next((m for m in VLA_MODELS.values() if m.get("port") == port), None)
    if model is None:
        model = VLA_MODELS[state["vla_selected"]]
    container = POLICY_CONTAINER or model["container"]
    try:
        up = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container],
                            capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        up = ""
    if up != "true":
        raise APIError(503, "POLICY_DOWN", f"{model['name']} 的模型服务未运行")


class GripperMonitor:
    """One lazy read-only stream; per-action listeners never survive an attempt."""
    def __init__(self):
        self.lock = threading.RLock()
        self.start_lock = threading.Lock()
        self.monitor = None
        self.listeners = set()

    def observe(self, frame):
        with self.lock:
            for listener in self.listeners:
                listener.observe(frame)

    def get(self, require_fresh=True):
        # Never ensure_container here: read-only requests must not start services.
        with self.start_lock:
            if self.monitor is None or self.monitor.p.poll() is not None:
                if self.monitor:
                    self.monitor.close()
                self.monitor = None
                try:
                    self.monitor = mission_module.Monitor(mission._monitor_cmd(), self.observe)
                except OSError as exc:
                    raise APIError(503, "MONITOR_DOWN", str(exc)) from exc
            monitor = self.monitor
        deadline = time.monotonic() + 3
        while True:
            frame = monitor.frame
            has_data = frame is not None and all(mission_module.finite(frame[side].get(key))
                       for side in mission_module.SIDES for key in ("grip_pos", "grip_eff", "ee_z"))
            if monitor.p.poll() is None and (monitor.fresh() if require_fresh else has_data):
                return monitor
            if monitor.p.poll() is not None or time.monotonic() >= deadline:
                raise APIError(503, "MONITOR_DOWN", "no usable arm/gripper/EE feedback within 3 seconds")
            time.sleep(.02)

    def readings(self):
        monitor = self.get(require_fresh=False)
        frame, received = monitor.frame, monitor.received
        age = max(0, time.monotonic() - received)
        out = {"age_s": age, "source": "monitor"}
        for side in mission_module.SIDES:
            sample = frame[side]
            out[side] = {"pos": sample.get("grip_pos"), "eff": sample.get("grip_eff"), "ee_z": sample.get("ee_z"),
                         "fresh": bool(monitor.p.poll() is None and age < mission_module.MONITOR_TIMEOUT
                                       and sample.get("fresh") is True
                                       and all(mission_module.finite(sample.get(k)) for k in ("grip_pos", "grip_eff", "ee_z"))),
                         "holding": mission_module.holding(sample, side)}
        return out

    def close(self):
        if self.monitor:
            self.monitor.close()


class GraspDetection:
    def __init__(self, monitor):
        self.monitor = monitor
        self.detectors = {}
        self.hand = None
        self.fault = None
        self.no_grasp = None          # empty grasp / held without lifting: end the run early, grasped false
        self.done = threading.Event()

    def observe(self, frame):
        if self.hand or self.fault or self.no_grasp:
            return
        if not self.monitor.fresh():
            self.fault = "arm feedback lost"
            return
        for side in mission_module.SIDES:
            self.detectors[side] = mission_module.grasp_update(self.detectors.get(side), frame[side], frame["t"], side)
            if self.detectors[side]["success"] and self.hand is None:
                self.hand = side
            elif self.hand is None and self.no_grasp is None:
                other = self.detectors.get("left" if side == "right" else "right") or {}
                if self.detectors[side]["empty"] and not other.get("held"):
                    self.no_grasp = f"{side} gripper closed on nothing"
                elif self.detectors[side]["no_lift"]:
                    self.no_grasp = f"{side} gripper holding but not lifted within {mission_module.NO_LIFT_S:g} s"

    def watch(self, record):
        while not self.done.wait(.02):
            with gripper_monitor.lock:
                if not self.hand and not self.monitor.fresh():
                    self.fault = "arm feedback lost"
                should_stop = self.hand is not None or self.fault is not None or self.no_grasp is not None
            if record["status"] != "running":
                return  # external STOP/E-STOP owns cancellation
            if should_stop:
                # Do not interrupt_action: a detected grasp is a successful completion.
                stop_vla_motion()
                return


gripper_monitor = GripperMonitor()
atexit.register(gripper_monitor.close)


def resolve_model(model_id):
    if not isinstance(model_id, str) or model_id not in VLA_MODELS:
        raise APIError(400, "UNKNOWN_MODEL", "unknown model: " + str(model_id))
    return model_id


GRIP_LOG = os.environ.get("GRIP_LOG", "/root/ta2_gripper_calib.jsonl")   # gripper calibration records


def record_gripper(body):
    """Read-only calibration snapshot: sample one gripper for `seconds` at ~10 Hz, append stats to GRIP_LOG."""
    if not isinstance(body, dict) or set(body) - {"label", "hand", "seconds"}:
        raise APIError(400, "BAD_REQUEST", "parameters: label (required), hand left/right, seconds 0.5-10")
    label, hand, seconds = body.get("label"), body.get("hand", "right"), body.get("seconds", 2)
    if not isinstance(label, str) or not 1 <= len(label.strip()) <= 40:
        raise APIError(400, "BAD_REQUEST", "label must be 1-40 characters")
    if hand not in ("left", "right") or type(seconds) not in (int, float) or not .5 <= seconds <= 10:
        raise APIError(400, "BAD_REQUEST", "hand must be left/right and seconds 0.5-10")
    samples, end = [], time.monotonic() + seconds
    while True:
        samples.append(gripper_monitor.readings())
        if time.monotonic() >= end:
            break
        time.sleep(.1)

    def stats(side, key):
        vals = [x[side][key] for x in samples if mission_module.finite(x[side].get(key))]
        return {"mean": round(sum(vals) / len(vals), 4), "min": round(min(vals), 4), "max": round(max(vals), 4)} if vals else None
    other = "left" if hand == "right" else "right"
    record = {"time": timestamp(), "label": label.strip(), "hand": hand, "seconds": seconds, "samples": len(samples),
              "pos_rad": stats(hand, "pos"), "eff_nm": stats(hand, "eff"), "ee_z_m": stats(hand, "ee_z"),
              "fresh": all(x[hand]["fresh"] for x in samples),
              "holding_now_rule": sum(bool(x[hand]["holding"]) for x in samples),
              "other_hand": {"hand": other, "pos_rad": stats(other, "pos"), "eff_nm": stats(other, "eff")}}
    try:
        with open(GRIP_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        record["log_error"] = str(exc)
    note(f"gripper record [{record['label']}] {hand}: pos {record['pos_rad']}, eff {record['eff_nm']}")
    return record


def classify_gripper(hand="right", seconds=1.0):
    """Read-only: sample one gripper for `seconds` and say what it is doing, with the numbers behind it."""
    if hand not in ("left", "right") or not .2 <= seconds <= 5:
        raise APIError(400, "BAD_REQUEST", "hand must be left/right and seconds 0.2-5")
    samples, end = [], time.monotonic() + seconds
    while True:
        samples.append(gripper_monitor.readings()[hand])
        if time.monotonic() >= end:
            break
        time.sleep(.1)
    classes = [mission_module.classify_grip({"grip_pos": x["pos"], "grip_eff": x["eff"]}, hand) for x in samples]
    counts = {c: classes.count(c) for c in dict.fromkeys(classes)}
    verdict = max(counts, key=counts.get)
    pos = [x["pos"] for x in samples if mission_module.finite(x["pos"])]
    eff = [x["eff"] for x in samples if mission_module.finite(x["eff"])]
    p, e = (sum(pos) / len(pos) if pos else None), (sum(eff) / len(eff) if eff else None)
    threshold = mission_module.hold_threshold(hand)
    why = {"open": f"位置 {p:.3f} rad > {mission_module.HOLD_POS_MAX:g}（张开）",
           "empty": f"在用力（力矩 {e:.2f} ≤ {mission_module.HOLD_EFF_MAX:g}），位置 {p:.3f} < 空夹门槛 {threshold:.4f}",
           "thin": f"在用力（力矩 {e:.2f}），位置 {p:.3f} ≥ 空夹门槛 {threshold:.4f}，且 < 实体分界 {mission_module.THIN_SOLID_SPLIT:g}",
           "solid": f"在用力（力矩 {e:.2f}），位置 {p:.3f} ≥ 实体分界 {mission_module.THIN_SOLID_SPLIT:g}",
           "closing": f"位置 {p:.3f} 在张开和夹紧之间，但力矩 {e:.2f} > {mission_module.HOLD_EFF_MAX:g}（没在用力）",
           "unknown": "没有有效读数"}[verdict] if p is not None and e is not None else "没有有效读数"
    return {"hand": hand, "verdict": verdict, "label": mission_module.GRIP_CLASSES[verdict],
            "holding": verdict in ("thin", "solid"), "why": why, "samples": len(samples), "counts": counts,
            "pos_rad": None if p is None else round(p, 4), "eff_nm": None if e is None else round(e, 3),
            "fresh": all(x["fresh"] for x in samples),
            "rule": {"empty_closed": mission_module.EMPTY_CLOSED[hand], "hold_threshold": round(threshold, 4),
                     "thin_solid_split": mission_module.THIN_SOLID_SPLIT, "open_above": mission_module.HOLD_POS_MAX,
                     "pushing_effort_at_most": mission_module.HOLD_EFF_MAX}}


def gripper_records(limit=100):
    try:
        with open(GRIP_LOG) as f:
            lines = f.read().splitlines()[-limit:]
    except FileNotFoundError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def api_preflight(model_id):
    resolve_model(model_id)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        check_policy(model_id=model_id)
        add("policy", True, model_id)
    except APIError as exc:
        add("policy", False, exc.code + ": " + exc.message)
    try:
        readings = gripper_monitor.readings()
        for side in mission_module.SIDES:
            sample = readings[side]
            add(side + "_gripper", sample["fresh"] and not sample["holding"],
                f"fresh={sample['fresh']}, holding={sample['holding']}")
    except APIError as exc:
        for side in mission_module.SIDES:
            add(side + "_gripper", False, exc.code + ": " + exc.message)
    with seq_lock:
        add("connected", state["connected"], "chassis connection")
        add("api_mode", state["mode_ok"] is True, "robot API mode")
        add("estop_clear", not state["estop"], "emergency stop inactive")
        add("idle", not busy(), "no web or API action running")
        missing = [name for name in ("001", "002", "003", "004") if name not in paths]
        add("paths", not missing, "missing: " + ", ".join(missing) if missing else "001–004 loaded")
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def start_vla(action=None, model_id=None, monitor=None):
    model_id = model_id or state["vla_selected"]
    model = VLA_MODELS[model_id]
    tag = time.strftime(f"vla_{model_id}_%Y%m%d_%H%M%S_") + str(time.time_ns() % 1000000)
    state["vla"], state["vla_model"] = tag, model_id
    seconds = action["params"].get("seconds", model["seconds"]) if action else model["seconds"]
    note(f"VLA grasp started: {model['name']} / {tag} ({seconds} s, max {model['speed']} rad/s)")
    args = (tag, model_id) if action is None else (tag, model_id, action, monitor)
    threading.Thread(target=run_vla, args=args, daemon=True).start()
    return tag


def note(msg):
    log.append(time.strftime("%H:%M:%S ") + msg)


def parse_path_csv(text):
    pts = []
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    head = [h.strip() for h in lines[0].split(",")]
    ix, iy = head.index("x_m"), head.index("y_m")
    ih = head.index("heading_deg") if "heading_deg" in head else None
    for ln in lines[1:]:
        c = ln.split(",")
        th = float(c[ih]) * 0.0174532925 if ih is not None else 0.0
        pts.append([float(c[ix]), float(c[iy]), th])
    if len(pts) < 2:
        raise ValueError("needs at least 2 points")
    length = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:]))
    return {"points": pts, "length": round(length, 3),
            "end": [round(pts[-1][0] - pts[0][0], 3), round(pts[-1][1] - pts[0][1], 3)]}


def clean_name(n):
    n = os.path.basename(n).rsplit(".", 1)[0]
    n = "".join(ch for ch in n if ch.isalnum() or ch in "-_")[:40]
    return n or "path"


def load_path_dir():
    if not os.path.isdir(PATH_DIR):
        return
    for f in sorted(os.listdir(PATH_DIR)):
        if f.lower().endswith(".csv"):
            try:
                with open(os.path.join(PATH_DIR, f)) as fh:
                    paths[clean_name(f)] = parse_path_csv(fh.read())
            except (ValueError, OSError, IndexError) as e:
                note(f"skip {f}: {e}")
    note(f"loaded {len(paths)} paths from {PATH_DIR}")


def run_sequence(names, action=None, guard=None, boundary=None):
    # The request handler reserves names and clears abort before starting this worker.
    # Never clear abort here: STOP may already have arrived after the request.
    route = seq["route"]
    n, result = names[0], "finished"
    note("run all: " + " -> ".join(names))
    try:
        for i, n in enumerate(names):
            with seq_lock:
                if seq["abort"] or (action and action["status"] != "running") or (guard and not guard()) or (boundary and not boundary()):
                    result = f"stopped before {n}"
                    break
                seq["i"] = i
                path_done.clear()
                path_result["why"] = None
                if not send("path_cmd", n):
                    raise RuntimeError("teleop disconnected")
            deadline = time.monotonic() + 3 * paths[n]["length"] / 0.10 + 30
            while not path_done.wait(timeout=0.1) and not seq["abort"]:
                if guard and not guard():
                    break
                if time.monotonic() >= deadline:
                    send("stop")
                    path_result["why"] = "timeout"
                    break
            why = "stopped" if seq["abort"] else path_result["why"]
            if why != "path done":
                result = f"stopped at {n} ({why})"
                break
            if boundary and not boundary():
                result = f"stopped after {n} (object dropped)"
                break
            if i < len(names) - 1 and PATH_PAUSE > 0:
                note(f"run all: {n} done, pausing {PATH_PAUSE:.0f} s before {names[i + 1]}")
                seq["pause_until"] = time.time() + PATH_PAUSE
                end = time.monotonic() + PATH_PAUSE
                while time.monotonic() < end and not seq["abort"]:
                    if guard and not guard():
                        break
                    if boundary and not boundary():
                        break
                    time.sleep(0.1)
                seq["pause_until"] = None
            elif not route:     # retain the ordinary Run all settling delay
                time.sleep(1.0)
        if seq["abort"] and result == "finished":
            result = f"stopped at {n} (stopped)"
    except Exception as e:  # noqa: BLE001
        result = f"stopped at {n} (error: {e})"
        send("stop")
    finally:
        with seq_lock:
            note("run all: " + result)
            if route:
                state["route_last"] = ROUTES[route][0] + ": " + result
                note(state["route_last"])
            seq.update(names=[], i=-1, route=None, pause_until=None)
            end_action(action, {"why": result, "path_why": path_result["why"],
                                "route_last": state["route_last"] if route else None}, result == "finished")

    return result


arm_proc = [None]  # stop host-side descent preparation before it can launch a later step


def end_arm_runner():
    p = arm_proc[0]
    if p is not None and p.poll() is None:
        try:
            p.terminate()
        except ProcessLookupError:
            pass


def run_arm(action, record=None, down=None, move=None):
    rc, last = None, ""
    tail = collections.deque(maxlen=8)
    try:
        ensure_container()
        if state["estop"] or (record and record["status"] != "running"):
            return
        if action == "down5":
            down = ("right", 5)
        if action in ("fwd5_left", "fwd5_right"):        # page buttons: one click = 5 cm forward
            move = (action[5:], 5, 0, 0)
        if move:
            cmd = ARM_MOVE_CMD + [move[0]] + [f"{v:g}" for v in move[1:]]
        else:
            cmd = ARM_DOWN_CMD + [down[0], str(down[1])] if down else ARM_CMD + [action, "--time", str(ARM_TIME[action])]
        with seq_lock:
            if state["estop"] or (record and record["status"] != "running"):
                return
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            arm_proc[0] = p
            if state["estop"] or (record and record["status"] != "running"):
                end_arm_runner()
                subprocess.run(ARM_STOP, capture_output=True)
        last = ""
        for line in p.stdout:
            line = line.rstrip()
            if line and "multicast" not in line and not line.startswith("Current mode"):
                note("arm: " + line)
                last = line
                tail.append(line)
        rc = p.wait()
        state["arm_last"] = f"arms {action}: " + ("done" if rc == 0 else f"stopped ({last})")
    except Exception as e:  # noqa: BLE001
        state["arm_last"] = f"arms {action}: error {e}"
        note(state["arm_last"])
    finally:
        with seq_lock:
            arm_proc[0] = None
            state["arm"] = None
            end_action(record, {"why": state["arm_last"], "returncode": rc,
                                "output": "\n".join(tail) if record and record["type"] in ("arm.down", "arm.move") else last}, rc == 0)


vla_proc = [None]   # host-side VLA runner (vla_run.sh); STOP / E-STOP end it so it cannot start a later step


def end_vla_runner():
    """Terminate the host-side runner so nothing it would start next (guarded_grasp after the probe, a startup
    retry) can run. The container processes are stopped separately with pkill."""
    p = vla_proc[0]
    if p is not None and p.poll() is None:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass


def stop_vla_motion():
    end_vla_runner()
    subprocess.run(DEXEC + ["pkill", "-INT", "-f", ESTOP_PROCS], capture_output=True)


def run_vla(tag, model_id, record=None, monitor=None):
    model = VLA_MODELS[model_id]
    label = model["name"]
    result, rc = None, None
    seconds = record["params"].get("seconds", model["seconds"]) if record else model["seconds"]
    detection = GraspDetection(monitor) if monitor else None
    watcher = None
    try:
        ensure_container()
        if state["estop"] or (record and record["status"] != "running"):
            return
        command = shlex.split(os.environ["VLA_CMD"]) if os.environ.get("VLA_CMD") else model["command"]
        with seq_lock:
            if state["estop"] or (record and record["status"] != "running"):
                return
            if monitor and not monitor.fresh():
                raise RuntimeError("arm feedback lost before VLA launch")
            p = subprocess.Popen(command + [tag, str(seconds), str(model["speed"])], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            vla_proc[0] = p
            if detection:
                with gripper_monitor.lock:
                    gripper_monitor.listeners.add(detection)
                watcher = threading.Thread(target=detection.watch, args=(record,), daemon=True)
                watcher.start()
            # E-STOP bypasses seq_lock; catch it even if it raced with Popen.
            if state["estop"] or (record and record["status"] != "running"):
                stop_vla_motion()
        for line in p.stdout:
            line = line.rstrip()
            if not line or "multicast" in line or " INFO " in line or line.startswith("Current mode"):
                continue
            note("vla: " + line[:200])
            if line.startswith("{") and '"status"' in line:
                try:
                    result = json.loads(line)
                except ValueError:
                    pass
        rc = p.wait()
        if result and "actuations" in result:
            why = result.get("error") or result["status"]
            state["vla_last"] = f"{label} / {tag}: {result['actuations']} steps, {why}"
        else:
            state["vla_last"] = f"{label} / {tag}: exited rc={rc} before running the policy (see log)"
    except Exception as e:  # noqa: BLE001
        state["vla_last"] = f"{label} / {tag}: error {e}"
    finally:
        if detection:
            detection.done.set()
            with gripper_monitor.lock:
                gripper_monitor.listeners.discard(detection)
            if watcher:
                watcher.join()  # keep the reservation until its stop command has finished
        vla_proc[0] = None
        with seq_lock:
            state["vla"] = None
            state["vla_model"] = None
            note(state["vla_last"])
            detail = dict(result) if isinstance(result, dict) else {}
            detail.setdefault("status", "script_error")
            detail.setdefault("actuations", None)
            detail.setdefault("error", None if rc == 0 and result else state["vla_last"])
            detail.setdefault("report", f"/root/openpi-xpu/client/outputs/{tag}/report.json")
            detail.update(tag=tag, model=model_id, rtc=model_rtc(model), seconds=seconds, returncode=rc,
                          grasped=bool(detection.hand) if detection else None,
                          hand=detection.hand if detection else None)
            ok = rc == 0 and not detail["error"] and detail["status"] in ("time_limit", "done", "finished", "succeeded")
            code = None
            if detection and detection.hand:
                detail.update(status="succeeded", error=None)
                state["vla_last"] = f"{label} / {tag}: grasped ({detection.hand})"
                ok = True
            elif detection and detection.no_grasp:
                detail.update(status="no_grasp", error=detection.no_grasp)
                state["vla_last"] = f"{label} / {tag}: no grasp ({detection.no_grasp})"
                ok, code = False, "NO_GRASP"
            elif detection and (detection.fault or (monitor and not monitor.fresh())):
                detail.update(status="feedback_lost", error=detection.fault or "arm feedback lost")
                ok, code = False, "FEEDBACK_LOST"
            if record and record["status"] == "aborted":
                detail.update(status="aborted", error=record["error"]["message"], grasped=False if detection else None, hand=None)
            end_action(record, detail, ok, code=code)


def estop():
    """Software emergency stop: chassis and paths, arm and VLA scripts, then FSM pause."""
    try:
        seq["abort"] = True
        send("stop")
        end_arm_runner()
        end_vla_runner()
        subprocess.run(DEXEC + ["pkill", "-INT", "-f", ESTOP_PROCS], capture_output=True)
        for _ in range(30):            # up to 3 s for a clean exit (scripts hold their pose)
            if subprocess.run(DEXEC + ["pgrep", "-f", ESTOP_PROCS], capture_output=True).returncode != 0:
                break
            time.sleep(0.1)
        else:
            subprocess.run(DEXEC + ["pkill", "-KILL", "-f", ESTOP_PROCS], capture_output=True)
            note("E-STOP: scripts did not exit in 3 s, force-killed")
        r = subprocess.run(DEXEC + FSM_PAUSE, capture_output=True, text=True)
        ok = r.returncode in (0, 124)
        state["estop_last"] = time.strftime("%H:%M:%S ") + ("FSM paused" if ok else f"FSM pause failed rc={r.returncode}")
    except Exception as e:  # noqa: BLE001
        state["estop_last"] = time.strftime("%H:%M:%S ") + f"error {e}"
    finally:
        state["estop"] = False
        note("E-STOP done: " + str(state["estop_last"]))


def alive_loop():
    while True:
        time.sleep(0.25)
        if state["active_path"]:      # paths do not depend on the page; STOP / E-STOP end them
            send("alive")


def add_point(x, y, th, force=False):
    moved = not trail or (x - trail[-1][0]) ** 2 + (y - trail[-1][1]) ** 2 > 0.005 ** 2 \
        or abs(th - trail[-1][2]) > 0.035
    if (force or moved) and len(trail) < TRAIL_MAX:
        trail.append([x, y, th, round(time.monotonic() - state["rec_t0"], 2)])


def trail_csv():
    rows = ["t_s,x_m,y_m,heading_deg"]
    rows += [f"{t:.2f},{x:.4f},{y:.4f},{th * 57.29578:.2f}" for x, y, th, t in trail]
    return "\n".join(rows) + "\n"


def ensure_container():
    if os.environ.get("TELEOP_CMD"):     # offline/custom teleop does not need Docker
        return
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    if CONTAINER in r.stdout.split():
        return
    note("starting client container")
    env = {**os.environ, "OPENPI_DIR": f"{DEPLOY}/client/openpi", "ZENOH_ENDPOINT": f"tcp/{ROBOT}:9000"}
    r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True)
    if CONTAINER in r.stdout.split():          # left in Exited state (e.g. after a B70 reboot): start refuses it
        subprocess.run([f"{DEPLOY}/client/run_client.sh", "stop"], cwd=DEPLOY, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run([f"{DEPLOY}/client/run_client.sh", "start"], cwd=DEPLOY, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(8)


def reader(p):
    for line in p.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        with seq_lock:
            e = ev.get("event")
            if e == "ready":
                state["connected"] = True
                state["last"] = "ready"
                note("teleop ready")
            elif e == "mode":
                state["mode_ok"] = ev.get("ok")
                note("robot mode " + ("OK (API, chassis enabled)" if ev.get("ok") else "NOT OK"))
            elif e == "moving":
                state["moving"] = f"{ev['kind']} {ev['dir']}"
                if ev["kind"] == "path":
                    state["active_path"] = ev["dir"]
                    state["path_anchor"] = list(state["pose"])
                    note(f"path {ev['dir']}: {ev.get('length')} m, {ev.get('corners')} corner(s), "
                         f"{ev.get('dwell', 0):.0f} s pause at each")
                else:
                    note(f"moving: {ev['kind']} {ev['dir']}")
            elif e == "path":
                if ev.get("dwell"):
                    state["moving"] = f"path {ev['name']} - corner pause {ev['dwell']:.0f} s"
                    note(f"path {ev['name']}: at corner, pausing {ev['dwell']:.0f} s")
                elif ev.get("resume"):
                    state["moving"] = f"path {ev['name']}"
                    note(f"path {ev['name']}: pause over, continuing")
                else:
                    note(f"path {ev['name']}: waypoint {ev['waypoint']}/{ev['of']}")
            elif e == "idle":
                state["moving"] = None
                if ev.get("kind") == "path":
                    state["active_path"] = None
                    path_result["why"] = ev.get("why")
                    path_done.set()
                state["last"] = f"{ev.get('kind')} {ev.get('dir')}: {ev.get('why')}"
                note(state["last"])
            elif e == "error":
                state["last"] = "error: " + ev.get("msg", "")
                note(state["last"])
                if seq["names"]:
                    path_result["why"] = state["last"]
                    path_done.set()
            elif e == "pose":
                x, y, th = ev["x"], ev["y"], ev["th"]
                state["pose"] = [x, y, th]
                if ev.get("reset"):
                    trail.clear()
                    state["epoch"] += 1
                    if state["recording"]:
                        state["rec_t0"] = time.monotonic()
                if state["recording"]:
                    add_point(x, y, th)
            elif e == "exit":
                note("teleop exited")
            motion_event(ev)
    with seq_lock:
        state["connected"] = False
        state["moving"] = None
        if seq["names"]:
            path_result["why"] = "teleop disconnected"
            path_done.set()
        pending_motion[0] = None
        action = current_action[0]
        if action and action["type"] in ("chassis.step", "path.run"):
            end_action(action, {"why": "teleop disconnected"})


def teleop():
    """Return a running teleop process, starting it if needed."""
    with lock:
        p = proc["p"]
        if p is not None and p.poll() is None:
            return p
        ensure_container()
        note("starting teleop")
        p = subprocess.Popen(TELEOP, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=reader, args=(p,), daemon=True).start()
        proc["p"] = p
        return p


def send(cmd, d=None):
    if cmd in ("step", "hold", "path_cmd") and state["estop"]:
        return False
    p = teleop()
    if cmd == "path_cmd":
        msg = {"cmd": "path", "name": d, "points": paths[d]["points"], "dwell": 0}   # no stops at corners
    else:
        msg = {"cmd": cmd} if d is None else {"cmd": cmd, "dir": d}
    if cmd in ("step", "hold", "path_cmd"):
        pending_motion[0] = (cmd, d)
    try:
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        return True
    except (BrokenPipeError, OSError):
        pending_motion[0] = None
        return False


def motion_event(ev):
    event = ev.get("event")
    if event == "moving":
        pending_motion[0] = None
    if event not in ("idle", "error"):
        return
    pending_motion[0] = None
    action = current_action[0]
    if not action or action["type"] not in ("chassis.step", "path.run"):
        return
    if event == "idle":
        expected = "step" if action["type"] == "chassis.step" else "path"
        if ev.get("kind") != expected:
            return
    why = ev.get("why") if event == "idle" else ev.get("msg", "teleop error")
    end_action(action, {"why": why, "teleop": ev}, why in ("done", "step done", "path done"))


def motion_watchdog(action, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with seq_lock:
            if current_action[0] is not action:
                return
        time.sleep(0.1)
    with seq_lock:
        if current_action[0] is action:
            send("stop")
            end_action(action, {"why": "motion timeout"}, code="TIMEOUT")


ACTION_PARAMS = {
    "chassis.step": {"direction": list(DIRS)},
    "path.run": {"name": "loaded path name"},
    "path.run_all": {},
    "route.run": {"route": list(ROUTES)},
    "arm.pose": {"pose": list(ARM_TIME)},
    "arm.down": {"hand": ["left", "right"], "cm": "number from 0.5 to 15"},
    "arm.move": {"hand": ["left", "right"], "forward_cm": "optional number -15..15 (shoulder frame x, + = forward)",
                 "left_cm": "optional number -15..15 (y, + = robot's left)", "up_cm": "optional number -15..15 (z, + = up)"},
    "vla.grasp": {"model": "optional registered model id (default: selected model)",
                  "seconds": "optional number from 1 to 120 (default: model seconds)",
                  "stop_on_grasp": "optional boolean (default: false)"},
    "odom.reset": {},
    "mission.run": {"attempts": "optional integer", "prompt": "optional string", "port": "optional integer",
                    "flow": "optional: v1 (steps 1-6, default) | v2 (adds step 7 voice interaction module) | voice (step 7 only) | grasp (step 3 only: retried grasp, stops holding) | handoff (7.4-7.7 only: reach out, lower, release with the hand holding the object)",
                    "rtc": "optional boolean (default: the selected model's mode)"},
}


def start_action(body):
    if not isinstance(body, dict) or set(body) - {"type", "params"}:
        raise APIError(400, "BAD_REQUEST", "expected {type, params}")
    kind, params = body.get("type"), body.get("params", {})
    if not isinstance(kind, str) or kind not in ACTION_PARAMS or not isinstance(params, dict):
        raise APIError(400, "BAD_REQUEST", "unknown type or invalid params")
    spec = ACTION_PARAMS[kind]
    generation = estop_generation[0]
    monitor = None
    if kind == "arm.down":
        if (set(params) != {"hand", "cm"} or params["hand"] not in ("left", "right")
                or type(params["cm"]) not in (int, float) or not .5 <= params["cm"] <= 15):
            raise APIError(400, "BAD_REQUEST", "arm.down requires hand left/right and cm from 0.5 to 15")
    elif kind == "arm.move":
        dist = {k: params.get(k, 0) for k in ("forward_cm", "left_cm", "up_cm")}
        if (set(params) - {"hand", "forward_cm", "left_cm", "up_cm"} or params.get("hand") not in ("left", "right")
                or any(type(v) not in (int, float) or not -15 <= v <= 15 for v in dist.values())
                or max(abs(v) for v in dist.values()) < .1):
            raise APIError(400, "BAD_REQUEST", "arm.move requires hand left/right and forward_cm/left_cm/up_cm "
                           "from -15 to 15 (at least one of them 0.1 cm or more)")
        params = {"hand": params["hand"], **dist}
    elif kind == "vla.grasp":
        if set(params) - set(spec):
            raise APIError(400, "BAD_REQUEST", "vla.grasp parameters: model, seconds, stop_on_grasp")
        model_id = resolve_model(params.get("model", state["vla_selected"]))
        seconds = params.get("seconds", VLA_MODELS[model_id]["seconds"])
        if type(seconds) not in (int, float) or not 1 <= seconds <= 120:
            raise APIError(400, "BAD_REQUEST", "seconds must be a number from 1 to 120")
        if type(params.get("stop_on_grasp", False)) is not bool:
            raise APIError(400, "BAD_REQUEST", "stop_on_grasp must be true or false")
        params = {"model": model_id, "seconds": seconds, "stop_on_grasp": params.get("stop_on_grasp", False)}
    elif kind == "mission.run":
        merged = dict(params)
        sel = VLA_MODELS[state["vla_selected"]]
        if "prompt" not in merged and "PROMPT" not in os.environ:
            merged["prompt"] = sel["prompt"]
        if "port" not in merged and "MISSION_PORT" not in os.environ and "POLICY_PORT" not in os.environ:
            merged["port"] = sel["port"]
        if "rtc" not in merged and "MISSION_RTC" not in os.environ:
            merged["rtc"] = model_rtc(sel)
        try:
            cfg = mission_module.options(merged)
        except ValueError as exc:
            raise APIError(400, "BAD_REQUEST", str(exc)) from exc
        cfg.update(model=state["vla_selected"], model_name=sel["name"])
    elif set(params) != set(spec) or any(not isinstance(v, str) for v in params.values()):
        raise APIError(400, "BAD_REQUEST", "required parameters: " + ", ".join(spec))
    for key, allowed in spec.items():
        if isinstance(allowed, list) and params[key] not in allowed:
            code = "UNKNOWN_ROUTE" if key == "route" else "BAD_REQUEST"
            raise APIError(400, code, f"invalid {key}: {params[key]}")
    names = None
    if kind == "path.run":
        if params["name"] not in paths:
            raise APIError(400, "UNKNOWN_PATH", "unknown path: " + params["name"])
    elif kind in ("route.run", "path.run_all"):
        names = ROUTES[params["route"]][1][:] if kind == "route.run" else sorted(paths)
        missing = [n for n in names if n not in paths]
        if not names or missing:
            raise APIError(400, "UNKNOWN_PATH", "missing paths: " + ", ".join(missing or ["no paths loaded"]))
    if state["estop"]:
        raise APIError(409, "ESTOP_ACTIVE", "emergency stop is in progress")
    if busy():
        raise APIError(409, "BUSY", "wait until the robot is still")
    if not state["connected"]:
        raise APIError(503, "NOT_CONNECTED", "robot teleop is not connected")
    if kind == "mission.run":
        try:
            mission.prepare(cfg)
        except mission_module.MissionAborted as exc:
            raise APIError(409, "ESTOP_ACTIVE", "emergency stop during preflight") from exc
        except mission_module.MissionFailure as exc:
            raise APIError(409, "MISSION_PREFLIGHT_FAILED", str(exc)) from exc
    if kind == "vla.grasp":
        check_policy(model_id=params["model"])
        if params["stop_on_grasp"]:
            monitor = gripper_monitor.get()
    # E-STOP bypasses seq_lock and may arrive during the policy check.
    if state["estop"] or (kind == "vla.grasp" and generation != estop_generation[0]) or (kind == "mission.run" and mission.cancel.is_set()):
        if kind == "mission.run" and mission.monitor:
            mission.monitor.close()
        raise APIError(409, "ESTOP_ACTIVE", "emergency stop is in progress")
    action = {"id": uuid.uuid4().hex[:12], "type": kind, "params": dict(params),
              "status": "running", "created_at": timestamp(), "ended_at": None,
              "result": None, "error": None}
    terminal_claims[action["id"]] = action
    log_claims[action["id"]] = action
    actions.appendleft(action)
    current_action[0] = action
    try:
        if kind == "mission.run":
            mission.start(cfg, action)
        elif kind in ("route.run", "path.run_all"):
            start_sequence(names, params.get("route"), action)
        elif kind == "arm.pose":
            start_arm(params["pose"], action)
        elif kind == "arm.down":
            start_arm("down", action, (params["hand"], params["cm"]))
        elif kind == "arm.move":
            start_arm("move", action, move=(params["hand"], params["forward_cm"], params["left_cm"], params["up_cm"]))
        elif kind == "vla.grasp":
            start_vla(action, params["model"], monitor)
        else:
            cmd, arg = ("step", params["direction"]) if kind == "chassis.step" else (
                ("path_cmd", params["name"]) if kind == "path.run" else ("reset_pose", None))
            if not send(cmd, arg):
                end_action(action, {"why": "teleop disconnected"})
            elif kind == "odom.reset":
                end_action(action, {"why": "reset_pose sent"}, True)
            else:
                deadline = 3 * paths[arg]["length"] / 0.10 + 30 if kind == "path.run" else 15
                threading.Thread(target=motion_watchdog, args=(action, deadline), daemon=True).start()
    except Exception as e:  # a failed launch still has an auditable action record
        end_action(action, {"why": str(e)}, code="SCRIPT_ERROR")
    return {"action_id": action["id"], "action": copy.deepcopy(action)}


def api_state():
    x, y, th = state["pose"]
    pause = seq["pause_until"]
    return {"robot": {"connected": state["connected"], "api_mode": state["mode_ok"]},
            "chassis": {"pose": {"x_m": x, "y_m": y, "heading_deg": th * 57.295779513},
                        "moving": state["moving"]},
            "arms": {"busy": bool(state["arm"]), "last": state["arm_last"]},
            "vla": {"running": bool(state["vla"]), "last": state["vla_last"]},
            "sequence": {"names": list(seq["names"]), "index": seq["i"], "route": seq["route"],
                         "pause_left_s": max(0, pause - time.time()) if pause else 0},
            "estop": {"active": state["estop"], "last": state["estop_last"]},
            "mission": mission.snapshot(),
            "current_action": copy.deepcopy(current_action[0])}


class _Host:
    def __getattr__(self, name):
        return globals()[name]


mission = mission_module.Mission(sys.modules.get(__name__) or _Host())

# ---- robot status check (/robot-check): read-only, one click ---------------------------------------------
# Runs inside the client container (stdin script): listens ~2 s to the robot topics and reports rates + values.
ROBOT_TOPICS_SCRIPT = r'''
import json, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
TOPICS = %s
rclpy.init(); node = Node("ta2_robot_check"); types = {}
end = time.monotonic() + 1.5
while time.monotonic() < end and len(types) < len(TOPICS):
    for name, ts in node.get_topic_names_and_types():
        if name in TOPICS and ts:
            types[name] = ts[0]
    rclpy.spin_once(node, timeout_sec=0.1)
counts, last, errors = {t: 0 for t in TOPICS}, {}, {}
def got(t, m):
    counts[t] += 1; last[t] = m
for t, ty in types.items():
    try:
        node.create_subscription(get_message(ty), t, lambda m, t=t: got(t, m), qos_profile_sensor_data)
    except Exception as exc:
        errors[t] = str(exc)[:120]
WINDOW = 2.0; t0 = time.monotonic()
while time.monotonic() - t0 < WINDOW:
    rclpy.spin_once(node, timeout_sec=0.05)
def summary(m):
    d = message_to_ordereddict(m)
    if "position" in d and isinstance(d["position"], list):
        return {"position": [round(v, 3) for v in d["position"][:7]], "effort": [round(v, 2) for v in (d.get("effort") or [])[:1]]}
    if "position" in d and isinstance(d["position"], dict):
        return {"z": round(d["position"]["z"], 3)}
    if "pose" in d:
        p = d["pose"]["pose"]["position"]; return {"x": round(p["x"], 3), "y": round(p["y"], 3)}
    return json.loads(json.dumps(d, default=str))
out = {t: {"declared": t in types, "hz": round(counts[t] / WINDOW, 1),
           "value": summary(last[t]) if t in last else None, "error": errors.get(t)} for t in TOPICS}
print(json.dumps(out))
'''
ROBOT_TOPICS = [("/left_arm/joint_states", "左臂关节"), ("/right_arm/joint_states", "右臂关节"),
                ("/left_gripper/joint_states", "左夹爪"), ("/right_gripper/joint_states", "右夹爪"),
                ("/left_arm/current_ee_pose", "左臂末端位姿"), ("/right_arm/current_ee_pose", "右臂末端位姿"),
                ("/api/current_mode", "控制模式"), ("/fsm_state", "状态机"), ("/chassis/odom", "底盘里程计")]


def _check_cmd(cmd, timeout, stdin=None):
    try:
        r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "", str(exc)


def _container_up(name):
    rc, out, _ = _check_cmd(["docker", "inspect", "-f", "{{.State.Running}}", name], 3)
    return rc == 0 and out == "true"


def robot_check():
    """Read-only snapshot of everything the robot needs; no command is sent to the robot."""
    items = []

    def add(group, name, level, value, hint=""):          # level: ok / warn / bad
        items.append({"group": group, "name": name, "level": level, "value": value, "hint": hint})

    ping = shlex.split(os.environ.get("ROBOT_PING_CMD", f"ping -c 1 -W 1 {ROBOT}"))
    rc, out, _ = _check_cmd(ping, 4)
    rtt = next((p.split("=")[-1].split()[0] for p in out.split() if p.startswith("time=")), None)
    add("网络", f"B70 → 机器人 {ROBOT}", "ok" if rc == 0 else "bad", f"通（{rtt} ms）" if rc == 0 else "不通",
        "" if rc == 0 else "检查机器人电源和网线 / Wi-Fi")
    add("网络", "遥控连接（8088 记录）", "ok" if state["connected"] else "bad", "已连接" if state["connected"] else "未连接",
        "这是 8088 的缓存状态；以下面的实时数据为准")
    add("网络", "API 模式（8088 记录）", "ok" if state["mode_ok"] is True else "bad",
        {True: "是", False: "否"}.get(state["mode_ok"], "未知"), "" if state["mode_ok"] is True else "在机器人上切到 API 模式")

    cmd = shlex.split(os.environ.get("ROBOT_CHECK_CMD", f"docker exec -i {CONTAINER} "
                                     "/usr/local/bin/openpi-client-entrypoint.sh python3 -u -"))
    script = ROBOT_TOPICS_SCRIPT % json.dumps([t for t, _ in ROBOT_TOPICS])
    rc, out, err = _check_cmd(cmd, 15, script)
    try:
        topics = json.loads(out.splitlines()[-1]) if rc == 0 and out else None
    except ValueError:
        topics = None
    if topics is None:
        add("机器人数据（实时 2 秒）", "读取话题", "bad", "检测脚本失败", (err or out)[-200:])
    else:
        for t, label in ROBOT_TOPICS:
            d = topics.get(t) or {}
            hz, val = d.get("hz", 0), d.get("value")
            if hz > 0:
                level, text = "ok", f"{hz:g} Hz"
            elif d.get("declared"):
                level, text = "bad", "通道在，但 2 秒内没有数据"
            else:
                level, text = "bad", "没有这个话题"
            if t == "/fsm_state" and hz == 0 and val is None:
                level = "warn"                      # published only on change on some firmware
            if val is not None:
                text += " · " + json.dumps(val, ensure_ascii=False)[:160]
            hint = "" if level == "ok" else ("机器人本体没发这个数据：查急停 / 报错，必要时重启本体控制程序"
                                             if t != "/fsm_state" else "状态机话题可能只在变化时发布")
            add("机器人数据（实时 2 秒）", f"{label}  {t}", level, text, d.get("error") or hint)
        mode = (topics.get("/api/current_mode") or {}).get("value")
        if isinstance(mode, dict):
            for side, cn in (("left", "左臂"), ("right", "右臂")):
                en, cm = mode.get(f"enable_{side}_arm"), mode.get(f"{side}_arm_control_mode")
                add("机器人配置", f"{cn}使能 / 控制模式", "ok" if en is True and cm == 1 else "warn",
                    f"使能 {en} · 控制模式 {cm}", "" if en is True and cm == 1 else "手臂需要使能且为关节模式（1）")

    with seq_lock:
        add("8088 服务", "急停", "bad" if state["estop"] else "ok", "已触发" if state["estop"] else "未触发",
            "在页面上解除急停" if state["estop"] else "")
        b = busy()
        act = current_action[0]
        add("8088 服务", "当前任务", "warn" if b else "ok",
            (act["type"] if act else "运行中") if b else "空闲", "")
    model = VLA_MODELS.get(state["vla_selected"]) or {}
    container = POLICY_CONTAINER or model.get("container", "")
    up = _container_up(container) if container else False
    add("模型与工具", "当前选中模型", "ok" if up else "bad", f"{model.get('name', state['vla_selected'])} · 容器 {container} "
        + ("运行中" if up else "未运行"), "" if up else "在「选择 / 切换模型」页启动它的容器")
    for name, label, hint in ((CONTAINER, "ROS 客户端容器", "手臂、VLA、夹爪读取都依赖它"),
                              ("ta2-sim-pipeline", "逆运动学容器", "前伸 / 下降需要它")):
        ok = _container_up(name)
        add("模型与工具", label, "ok" if ok else "bad", f"{name} " + ("运行中" if ok else "未运行"), "" if ok else hint)
    try:
        req = urllib.request.Request(mission_module.VOICE_BRIDGE + "/status", headers={"Host": mission_module.VOICE_BRIDGE_HOST})
        with urllib.request.urlopen(req, timeout=3) as r:
            v = json.load(r)
        mics = v.get("mics") or {}
        bad_mics = [k for k, x in mics.items() if not isinstance(x, (int, float))]
        add("语音（Mac）", "语音服务", "ok", {"idle": "空闲", "listening": "正在听", "recognized": "已识别", "playing": "正在播报",
                                         "done": "播报完成", "failed": "上次：" + str(v.get("detail", ""))}.get(v.get("state"), v.get("state")))
        if mics:
            add("语音（Mac）", "麦克风", "warn" if bad_mics else "ok",
                " · ".join(f"{k} {x:g} dB" if isinstance(x, (int, float)) else f"{k} 不可用" for k, x in mics.items()),
                "iPhone：同一 Apple ID、开 Wi-Fi 和蓝牙、靠近 Mac" if bad_mics else "")
    except (OSError, ValueError) as exc:
        add("语音（Mac）", "语音服务", "bad", "连不上", f"在 MacBook 终端运行 ~/ta2-voice/start_mac_voice.sh（{exc}）")
    counts = {k: sum(i["level"] == k for i in items) for k in ("ok", "warn", "bad")}
    return {"checked_at": timestamp(), "summary": counts, "items": items}


ROBOT_CHECK_PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>机器人状态检测</title><style>
:root{--bg:#f6f7f9;--fg:#1f2933;--muted:#616e7c;--panel:#fff;--line:#d9dee5;--ok:#2f855a;--ok-f:#ebf8f0;--warn:#b7791f;--warn-f:#fffaeb;--bad:#c53030;--bad-f:#fdeeee;--accent:#2b6cb0}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e4e7eb;--muted:#9aa5b1;--panel:#1b1f24;--line:#323a43;--ok:#68d391;--ok-f:#16261d;--warn:#d69e2e;--warn-f:#2a2416;--bad:#fc8181;--bad-f:#2b1818;--accent:#63b3ed}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}a{color:var(--accent)}h1{font-size:20px;margin:6px 0 2px}.sub{color:var(--muted);margin:0 0 12px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
button{font:inherit;font-weight:600;padding:10px 22px;border-radius:8px;border:0;background:var(--accent);color:#fff;cursor:pointer}button:disabled{opacity:.5}
#verdict{padding:8px 14px;border-radius:8px;font-weight:700;border:1px solid var(--line);background:var(--panel)}
#verdict.ok{color:var(--ok);background:var(--ok-f);border-color:var(--ok)}#verdict.warn{color:var(--warn);background:var(--warn-f);border-color:var(--warn)}#verdict.bad{color:var(--bad);background:var(--bad-f);border-color:var(--bad)}
.group{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:10px;overflow:hidden}
.group h2{font-size:14px;margin:0;padding:8px 14px;border-bottom:1px solid var(--line)}
.row{display:grid;grid-template-columns:22px minmax(140px,1.1fr) 2fr;gap:8px;padding:7px 14px;border-top:1px solid var(--line);align-items:start}
.row:first-of-type{border-top:0}.ic{font-weight:700}.ok .ic{color:var(--ok)}.warn .ic{color:var(--warn)}.bad .ic{color:var(--bad)}
.bad{background:var(--bad-f)}.warn{background:var(--warn-f)}
.val{word-break:break-all}.hint{display:block;color:var(--muted);font-size:12px}
@media(max-width:560px){.row{grid-template-columns:22px 1fr}.val{grid-column:2}}
</style></head><body><main>
<a href="/">← 返回 TA2 遥控</a>
<h1>机器人状态检测</h1>
<p class="sub">只读检测，不向机器人发送任何指令。实时数据项会监听机器人话题约 2 秒，整次检测约 5 秒。</p>
<div class="bar"><button id="run">一键检测</button><span id="verdict">尚未检测</span><span id="at" class="sub" style="margin:0"></span></div>
<div id="out"></div>
</main><script>
const IC={ok:'✓',warn:'!',bad:'✗'};
function esc(v){return String(v==null?'':v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function run(){
  const b=document.getElementById('run'),v=document.getElementById('verdict');b.disabled=true;v.className='';v.textContent='检测中…';
  try{
    const r=await fetch('/api/v1/robot-check',{cache:'no-store'});const d=await r.json();if(!r.ok)throw Error((d.error&&d.error.message)||r.status);
    const s=d.summary;v.className=s.bad?'bad':s.warn?'warn':'ok';
    v.textContent=s.bad?'有 '+s.bad+' 项异常'+(s.warn?'，'+s.warn+' 项警告':''):s.warn?'正常，'+s.warn+' 项警告':'全部正常（'+s.ok+' 项）';
    document.getElementById('at').textContent='检测时间 '+d.checked_at;
    const groups={};d.items.forEach(i=>(groups[i.group]=groups[i.group]||[]).push(i));
    document.getElementById('out').innerHTML=Object.entries(groups).map(([g,list])=>'<div class="group"><h2>'+esc(g)+'</h2>'+
      list.map(i=>'<div class="row '+i.level+'"><span class="ic">'+IC[i.level]+'</span><span>'+esc(i.name)+'</span><span class="val">'+esc(i.value)+
        (i.hint?'<span class="hint">'+esc(i.hint)+'</span>':'')+'</span></div>').join('')+'</div>').join('');
  }catch(e){v.className='bad';v.textContent='检测失败：'+e.message}
  finally{b.disabled=false}
}
document.getElementById('run').onclick=run;run();
</script></body></html>"""


def _gf_text(cx, cy, lines, size=14, lh=19):
    y0 = cy - (len(lines) - 1) * lh / 2 + size * 0.35
    spans = "".join('<tspan x="{}" y="{:.1f}"{}>{}</tspan>'.format(
        cx, y0 + i * lh, ' class="sub"' if line.startswith("~") else "", html.escape(line.lstrip("~")))
        for i, line in enumerate(lines))
    return f'<text text-anchor="middle" font-size="{size}">{spans}</text>'


def _gf_box(x, y, w, h, lines, cls="proc", phase=None, rx=6):
    ph = f' data-phase="{phase}"' if phase else ""
    return (f'<g class="node {cls}"{ph}><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'
            + _gf_text(x + w / 2, y + h / 2, lines) + "</g>")


def _gf_dec(cx, cy, w, h, lines, phase="judge"):
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    return f'<g class="node dec" data-phase="{phase}"><polygon points="{pts}"/>' + _gf_text(cx, cy, lines, 13, 17) + "</g>"


def _gf_edge(points, label=None, at=None, anchor="middle"):
    pts = " ".join(f"{x},{y}" for x, y in points)
    out = f'<polyline class="edge" points="{pts}" marker-end="url(#ah)"/>'
    if label:
        lx, ly = at
        out += f'<text class="lbl" x="{lx}" y="{ly}" text-anchor="{anchor}">{html.escape(label)}</text>'
    return out


def grasp_flow_svg():
    m = mission_module
    g = lambda v: f"{v:g}"
    N, LIM, LIFT, SUCC = g(m.ATTEMPTS), g(m.ATTEMPT_S), g(m.LIFT_MIN * 100), g(m.HOLD_PERSIST)
    EMP, NOL = g(m.EMPTY_PERSIST), g(m.NO_LIFT_S)
    cx, bx, bw = 330, 180, 300            # main column
    dw, dh = 300, 100                      # decisions
    p = []
    p.append(_gf_box(205, 16, 250, 46, ["开始抓取", "~大按钮 · 全流程第 3 步 · 语音 7.2"], "term", rx=23))
    p.append(_gf_edge([(cx, 62), (cx, 92)]))
    p.append(_gf_box(bx, 92, bw, 44, [f"第 i 次尝试，i = 1（最多 {N} 次）"], "proc"))
    p.append(_gf_edge([(cx, 136), (cx, 166)]))
    p.append(_gf_box(bx, 166, bw, 54, ["① 复位", "~回起始姿态，两只夹爪张开（约 6 s）"], "proc", "reset"))
    p.append(_gf_edge([(cx, 220), (cx, 250)]))
    p.append(_gf_box(bx, 250, bw, 54, ["② 取帧", "~读取一帧相机图像与关节状态"], "proc", "probe"))
    p.append(_gf_edge([(cx, 304), (cx, 334)]))
    p.append(_gf_box(bx, 334, bw, 66, ["③ VLA 推理执行", f"~最多 {LIM} s（从第一个动作计时）", "~限速 1.0 rad/s，每帧判定左右手"], "proc", "policy"))
    p.append(_gf_edge([(cx, 400), (cx, 432)]))
    ys = [482, 622, 762, 902]
    texts = [[f"夹住且抬高 ≥ {LIFT} cm", f"连续保持 {SUCC} s？"],
             ["空夹后 {} s 仍不张开".format(EMP), "（卡住）？"],
             ["夹住超过 {} s".format(NOL), "仍未抬起？"],
             [f"{LIM} s 到时，", "或 VLA 脚本退出？"]]
    for i, (y, t) in enumerate(zip(ys, texts)):
        p.append(_gf_dec(cx, y, dw, dh, t))
        if i < 3:
            p.append(_gf_edge([(cx, y + dh / 2), (cx, ys[i + 1] - dh / 2)], "否", (cx + 10, y + dh / 2 + 22), "start"))
    # an empty close is not a failure: the policy keeps running and may reopen to try again
    p.append(f'<polyline class="edge dash" points="{cx + dw / 2},{ys[1]} 560,{ys[1]}"/>')
    p.append(_gf_box(560, ys[1] - 46, 250, 92, ["空夹 ≠ 失败", "~VLA 不停，继续推理", "~模型张开重抓 → 计时清零", f"~{EMP} s 内一直闭合不动才算卡住"],
                     "info"))
    # success
    p.append(_gf_edge([(cx + dw / 2, ys[0]), (575, ys[0])], "是", (cx + dw / 2 + 14, ys[0] - 8), "start"))
    p.append(_gf_box(575, ys[0] - 33, 225, 66, ["停止 VLA，原地保持夹紧", "~抓取成功"], "ok", "success", rx=33))
    # "no" at the last check: keep inferring (loop back to ③)
    p.append(_gf_edge([(cx + dw / 2, ys[3]), (835, ys[3]), (835, 367), (bx + bw, 367)], "否：继续推理",
                      (cx + dw / 2 + 14, ys[3] - 10), "start"))
    # "yes" at checks 2-4 -> attempt failed
    for y in ys[1:]:
        p.append(f'<polyline class="edge" points="{cx - dw / 2},{y} 120,{y}"/>')
        p.append(f'<text class="lbl" x="{cx - dw / 2 - 12}" y="{y - 8}" text-anchor="end">是</text>')
    yf = 1000
    p.append(_gf_edge([(120, ys[1]), (120, yf + 27), (bx, yf + 27)]))
    p.append(_gf_box(bx, yf, bw, 54, ["停止 VLA，本次失败", "~记录原因：卡住 / 不抬 / 到时"], "bad", "fail"))
    p.append(_gf_edge([(cx, yf + 54), (cx, 1090)]))
    yd = 1140
    p.append(_gf_dec(cx, yd, 260, 96, ["还有剩余次数？", "（i < N）"], "retry"))
    p.append(_gf_edge([(cx - 130, yd), (60, yd), (60, 193), (bx, 193)], "是：i = i + 1，重新复位", (66, yd - 10), "start"))
    p.append(_gf_edge([(cx, yd + 48), (cx, 1222)], "否", (cx + 10, yd + 70), "start"))
    p.append(_gf_box(190, 1222, 280, 54, ["回起始姿态，张开夹爪", "~任务失败"], "bad term-bad", rx=27))
    return ('<svg viewBox="0 0 860 1290" role="img" aria-label="抓取流程图" xmlns="http://www.w3.org/2000/svg">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahead"/></marker></defs>'
            + "".join(p) + "</svg>")


GRASP_FLOW_PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>抓取流程图</title><style>
:root{--bg:#f6f7f9;--fg:#1f2933;--muted:#616e7c;--panel:#fff;--line:#d9dee5;--edge:#7b8794;
--proc-f:#fff;--proc-s:#52606d;--dec-f:#fffaeb;--dec-s:#b7791f;--ok-f:#ebf8f0;--ok-s:#2f855a;--bad-f:#fdeeee;--bad-s:#c53030;
--term-f:#eef2f7;--term-s:#3e4c59;--hi:#2b6cb0;--hi-f:#e6f0fb}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e4e7eb;--muted:#9aa5b1;--panel:#1b1f24;--line:#323a43;--edge:#8a96a3;
--proc-f:#1b1f24;--proc-s:#9aa5b1;--dec-f:#2a2416;--dec-s:#d69e2e;--ok-f:#16261d;--ok-s:#68d391;--bad-f:#2b1818;--bad-s:#fc8181;
--term-f:#1f262e;--term-s:#cbd2d9;--hi:#63b3ed;--hi-f:#162636}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}a{color:var(--hi)}h1{font-size:20px;margin:6px 0 2px}
.sub-h{color:var(--muted);margin:0 0 12px}
#now{display:flex;gap:10px;align-items:center;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--panel);margin-bottom:12px}
#now .dot{width:9px;height:9px;border-radius:50%;background:var(--edge)}#now.live .dot{background:var(--hi);box-shadow:0 0 0 4px var(--hi-f)}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
.chart svg{display:block;width:100%;min-width:560px;height:auto;font-family:inherit}
.node rect,.node polygon{stroke-width:1.4}
.node text{fill:var(--fg);font-weight:600}.node .sub{fill:var(--muted);font-weight:400;font-size:12px}
.proc rect{fill:var(--proc-f);stroke:var(--proc-s)}.term rect{fill:var(--term-f);stroke:var(--term-s)}
.dec polygon{fill:var(--dec-f);stroke:var(--dec-s)}
.ok rect{fill:var(--ok-f);stroke:var(--ok-s)}.ok text{fill:var(--ok-s)}
.bad rect{fill:var(--bad-f);stroke:var(--bad-s)}.bad text{fill:var(--bad-s)}
.node.on rect,.node.on polygon{stroke:var(--hi);stroke-width:3;fill:var(--hi-f)}
.edge{fill:none;stroke:var(--edge);stroke-width:1.5}.edge.dash{stroke-dasharray:4 4}
.info rect{fill:var(--hi-f);stroke:var(--hi);stroke-dasharray:5 4}.info text{fill:var(--hi)}.info .sub{fill:var(--fg)}.ahead{fill:var(--edge)}
.lbl{fill:var(--muted);font-size:12px;font-weight:600}
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin-top:12px}
.note{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.note h2{font-size:14px;margin:0 0 4px}.note ul{margin:0;padding-left:18px}.note li{margin:2px 0}
</style></head><body><main>
<a href="/">← 返回 TA2 遥控</a>
<h1>抓取流程图</h1>
<p class="sub-h">大按钮「开始抓取」、全流程第 3 步、语音模块 7.2 共用这一套流程。图中参数取自当前运行的代码。</p>
<div id="now" role="status"><span class="dot"></span><span id="nowtext">当前：未在抓取</span></div>
<div class="chart">__SVG__</div>
<div class="notes">
<div class="note"><h2>判定定义（左右手各自独立，每帧一次）</h2><ul>
<li><b>夹住</b>：夹爪用力（力 ≤ __EFF__）且停在空夹位置以上（右 ≥ __HR__ rad，左 ≥ __HL__ rad），实体、薄片都算</li>
<li><b>空夹</b>：夹爪用力且合拢到空夹位置以下</li>
<li>夹爪张开（&gt; __OPEN__ rad）后才开始判定；每次重新张开，计时全部清零</li>
<li>空夹后模型自己张开重抓：继续推理，不打断</li>
<li>另一只手正夹着东西时，这只手空夹不算失败</li></ul></div>
<div class="note"><h2>抓住之后</h2><ul>
<li>大按钮：结束，原地保持夹紧</li><li>全流程：第 4 步收臂（保持夹紧）</li><li>语音模块：7.3 收臂</li></ul></div>
<div class="note"><h2>任何时候</h2><ul>
<li>急停 / 停止：立即中止</li><li>手臂数据丢失、底盘断开、退出 API 模式：任务失败</li>
<li>启动后 __START__ s 没收到相机 / 关节数据：不动作，自动重来一次</li></ul></div>
</div>
</main><script>
const PH={starting:'准备中',reset:'① 复位中',probe:'② 取帧中',policy:'③ VLA 推理执行中',policy_retry:'③ 启动没收到数据，重来一次',stopping:'判定已出，正在停止 VLA',ended:'本次已结束'};
async function poll(){
  let L=null,m={};
  try{m=(await (await fetch('/api/v1/state',{cache:'no-store'})).json()).mission||{};L=m.live}catch(e){document.getElementById('nowtext').textContent='读取状态失败';return}
  const run=!!(L&&m.active&&L.phase!=='ended'),ok=L&&L.phase==='stopping'&&(L.note||'').startsWith('抓住');
  const on=run?{reset:['reset'],probe:['probe'],policy:['policy','judge'],policy_retry:['policy'],stopping:[ok?'success':'fail']}[L.phase]||[]:[];
  document.querySelectorAll('[data-phase]').forEach(n=>n.classList.toggle('on',on.includes(n.dataset.phase)));
  document.getElementById('now').classList.toggle('live',run);
  document.getElementById('nowtext').textContent=run?'当前：第 '+L.attempt+'/'+L.attempts+' 次 · '+(PH[L.phase]||L.phase)+' · 本次 '+Math.round(L.elapsed_s||0)+' s':
    L?'最近一次：第 '+L.attempt+'/'+L.attempts+' 次 · '+(L.verdict==='succeeded'?'抓住（'+(L.hand==='left'?'左手':'右手')+'）':'未抓住')+(m.result?' · 任务结果：'+m.result:''):'当前：未在抓取';
}
poll();setInterval(poll,700);
</script></body></html>"""


def grasp_flow_page():
    m = mission_module
    vals = {"__OPEN__": m.HOLD_POS_MAX, "__EFF__": m.HOLD_EFF_MAX, "__HR__": round(m.hold_threshold("right"), 4),
            "__HL__": round(m.hold_threshold("left"), 4), "__START__": 8}
    page = GRASP_FLOW_PAGE.replace("__SVG__", grasp_flow_svg())
    for k, v in vals.items():
        page = page.replace(k, f"{v:g}")
    return page


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>TA2 遥控</title>
<style>
:root{--bg:#f5f5f4;--fg:#252729;--muted:#64686c;--panel:#fff;--line:#dcdedc;--grid:#eceeec;--accent:#455c68;--solid:#303c43;--on-solid:#fff;--stop:#c62828;--trail:#687e71;--ok:#4f7260}
@media (prefers-color-scheme:dark){:root{--bg:#17191a;--fg:#e5e7e8;--muted:#a3a9ad;--panel:#202426;--line:#3e4548;--grid:#2c3235;--accent:#a7bdc8;--solid:#cfdbdf;--on-solid:#1c2529;--stop:#c62828;--trail:#a4b9aa;--ok:#9abca8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.bar{max-width:720px;margin:auto;padding:14px 20px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px 16px;align-items:center}
.identity{display:flex;align-items:center;flex-wrap:wrap;gap:6px 12px;min-width:0}
h1{font-size:19px;line-height:1.4;margin:0;white-space:nowrap}
.connection{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:12px;white-space:nowrap}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor;flex-shrink:0}
.connection.ok{color:var(--ok)}
.badge{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);white-space:nowrap;color:var(--muted)}
.activity{grid-column:1/-1;font-size:12px;color:var(--muted);display:flex;gap:8px;min-width:0}
.activity>span:first-child{white-space:nowrap}
#status{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--fg)}
.app{max-width:720px;margin:auto;padding:24px 20px 36px;display:flex;flex-direction:column;gap:20px}
.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}
h2{font-size:16px;font-weight:600;margin:0 0 18px}
.body{padding:22px}
.hint,.muted,.status-line{color:var(--muted);font-size:12px}
.hint{margin:16px 0 0}
.status-line{margin:10px 0 0;overflow-wrap:anywhere}
button{font:inherit;min-height:44px;min-width:44px;padding:9px 14px;border-radius:9px;border:1px solid var(--line);background:var(--panel);color:var(--fg);cursor:pointer;touch-action:manipulation;user-select:none;-webkit-user-select:none}
button:hover:not(:disabled){border-color:var(--accent);background:var(--grid)}
button:focus-visible,summary:focus-visible,.prow:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
button:disabled{opacity:.45;cursor:not-allowed}
button.on,.primary{background:var(--solid);border-color:var(--solid);color:var(--on-solid)}
.primary:hover:not(:disabled){background:var(--accent);color:var(--on-solid)}
#vlarun{width:100%;min-height:64px;font-size:18px;font-weight:600}
.route-buttons{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.route-buttons button{min-height:88px;padding:12px 8px}
.route-title{display:flex;align-items:center;justify-content:center;gap:6px;font-size:18px;font-weight:600}
.route-buttons small{display:block;color:var(--muted);font-size:12px;margin-top:5px}
.route-dot{display:inline-block;width:8px;height:8px;border-radius:50%;flex-shrink:0}
.route-dot.green{background:#388e3c}.route-dot.red{background:#d84343}
.estop{background:var(--stop)!important;border-color:var(--stop)!important;color:#fff;font-weight:700;font-size:16px;padding:10px 18px;grid-column:2;grid-row:1}
#vla-progress{margin-top:16px}
.progress-row{display:flex;gap:12px;align-items:center;justify-content:space-between;margin-bottom:10px}
#vlaelapsed{font-size:13px;font-variant-numeric:tabular-nums}
progress{display:block;width:100%;height:8px;border:0;border-radius:99px;overflow:hidden;background:var(--grid);accent-color:var(--accent)}
progress::-webkit-progress-bar{background:var(--grid)}
progress::-webkit-progress-value{background:var(--accent);border-radius:99px}
progress::-moz-progress-bar{background:var(--accent);border-radius:99px}
.arm-section{border-top:1px solid var(--line);padding-top:18px;margin-top:22px}
.arm-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
h3{font-size:13px;font-weight:600;margin:0 8px 0 0}
.small{font-size:13px;padding:8px 12px}
.pad{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.pad button,.rot button{min-height:56px;font-size:15px;touch-action:none}
#stop{font-weight:600;border-color:var(--accent)}
.rot{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:8px}
summary{min-height:56px;padding:16px 22px;cursor:pointer;font-size:15px;font-weight:600;touch-action:manipulation}
details[open]>summary{border-bottom:1px solid var(--line)}
.subhead{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.tools{display:flex;flex-wrap:wrap;gap:8px}
.plist{max-height:240px;overflow:auto;border:1px solid var(--line);border-radius:9px}
.prow{display:flex;align-items:center;flex-wrap:wrap;gap:4px 10px;padding:10px 12px;border-top:1px solid var(--line);font-size:13px;cursor:pointer;min-height:44px}
.prow:first-child{border-top:none}
.prow.sel{background:var(--grid)}
.prow.run{color:var(--accent);font-weight:700}
.prow .nm{font-weight:600;min-width:0;flex:1;overflow-wrap:anywhere}
.prow .info{color:var(--muted);font-variant-numeric:tabular-nums;flex-basis:100%;font-size:12px;overflow-wrap:anywhere}
.prow button{flex-shrink:0}
.trajectory{border-top:1px solid var(--line);margin-top:22px;padding-top:18px}
#rec{font-size:12px;color:var(--accent);font-weight:600;visibility:hidden}
#pose{font-size:12px;font-variant-numeric:tabular-nums;color:var(--muted);margin:8px 0 12px;overflow-wrap:anywhere}
.canvas-wrap{position:relative;aspect-ratio:4/3;max-height:480px;width:100%;margin-bottom:14px}
canvas{position:absolute;inset:0;width:100%;height:100%}
pre{margin:0;padding:16px 22px;font:12px/1.7 -apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;height:200px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
[hidden]{display:none!important}
@media(max-width:420px){.bar{padding:12px 16px;gap:8px 10px}.app{padding:20px 12px 28px;gap:16px}.body{padding:18px 16px}.identity{gap:6px 8px}summary{padding:14px 16px}.estop{padding:10px 14px}.hint{font-size:11px}}
.mission-current{font-weight:700;color:var(--accent);background:var(--grid);border-radius:6px}
#voicesub{margin:4px 0 0;padding-left:8px;list-style:none;font-weight:400;background:none}
#voicesub li{margin:2px 0}#voicesub small{color:var(--muted,#888)}
#voicesub .sub-running{color:var(--accent);font-weight:700}#voicesub .sub-failed{color:#c62828;font-weight:700}
.glive{margin:8px 0;padding:10px 12px;border:1px solid var(--line,#ccc);border-radius:8px;font-size:14px;line-height:1.55}
.glive .gh{font-weight:700}.glive .gp{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}
.glive .gp span{padding:1px 8px;border-radius:10px;border:1px solid var(--line,#ccc);opacity:.6}
.glive .gp .on{opacity:1;border-color:var(--accent);color:var(--accent);font-weight:700}.glive .gp .ok{opacity:1;color:#2e7d32;border-color:#2e7d32}
.glive .gp .bad{opacity:1;color:#c62828;border-color:#c62828;font-weight:700}
.glive .hand{margin:2px 0}.glive .hand b{display:inline-block;min-width:3em}.glive .warn{color:#c62828;font-weight:700}.glive .good{color:#2e7d32;font-weight:700}
.glive .hist{margin-top:6px;color:var(--muted,#888)}.glive .hist .ok{color:#2e7d32}.glive .hist .bad{color:#c62828}
#missionsteps .mk{font-weight:700}#missionsteps .mk-done{color:#2e7d32}#missionsteps .mk-failed{color:#c62828}#missionsteps .mk-running{color:var(--accent)}
#voicesub .sub-succeeded{color:#2e7d32}#voicesub .sub-skipped{opacity:.55}
.listen{display:flex;align-items:center;gap:10px;margin:6px 0;padding:8px 10px;border-radius:8px;background:rgba(229,57,53,.1);color:#c62828;font-weight:700}
.listen .mic{position:relative;flex:none;width:28px;height:28px;border-radius:50%;background:#e53935;color:#fff;display:grid;place-items:center;font-size:15px}
.listen .mic::before,.listen .mic::after{content:'';position:absolute;inset:0;border-radius:50%;border:2px solid #e53935;animation:ring 1.6s ease-out infinite}
.listen .mic::after{animation-delay:.8s}
.listen .bars{display:flex;gap:3px;align-items:center;height:18px}.listen .bars i{width:3px;height:100%;background:currentColor;border-radius:2px;animation:bar 1s ease-in-out infinite}
.listen .bars i:nth-child(2){animation-delay:.15s}.listen .bars i:nth-child(3){animation-delay:.3s}.listen .bars i:nth-child(4){animation-delay:.45s}
@keyframes ring{from{transform:scale(1);opacity:.8}to{transform:scale(2.2);opacity:0}}
@keyframes bar{0%,100%{transform:scaleY(.25)}50%{transform:scaleY(1)}}
@media (prefers-reduced-motion:reduce){.listen .mic::before,.listen .mic::after,.listen .bars i{animation:none}}
</style></head><body>
<header><div class="bar">
  <div class="identity"><h1>TA2 遥控</h1><a href="/api-docs" target="_blank" rel="noopener">接口文档 / 测试</a><a href="/grasp-flow" target="_blank" rel="noopener">抓取流程图</a><a href="/robot-check" target="_blank" rel="noopener">机器人状态检测</a><span id="connection" class="connection" role="status"><span class="dot" aria-hidden="true"></span><span id="connection-text">连接中…</span></span><span id="mode" class="badge">模式未知</span></div>
  <button id="estop" class="estop" title="急停底盘、手臂与抓取任务（Esc）">急停</button>
  <div class="activity"><span>当前活动</span><span id="status" role="status">正在连接机器人…</span></div>
</div></header>
<main class="app">
  <section class="card"><div class="body">
    <h2>全流程</h2>
    <p id="missionmodel" class="status-line">当前模型：读取中…</p>
    <button id="missionrun" class="primary" disabled>开始全流程（1–6）</button>
    <button id="missionrun2" class="primary" disabled>开始全流程 v2（1–7，含语音交互）</button>
    <button id="missionvoice" disabled>单独运行第 7 步</button>
    <button id="missionhandoff" disabled title="与第 7 步的 7.4–7.7 完全相同：用当前夹着东西的那只手前伸 10 cm → 下降 10 cm → 松开 → 确认已松开。需要正好一只手夹着东西；全流程里这之前会先收臂（起始姿态·保持夹爪）">单独放下（前伸+下降+松开）</button>
    <ol id="missionsteps"><li data-step="preflight">预检</li><li data-step="g2r">绿→红</li><li data-step="grasp">抓取 0/5</li><li data-step="stow">收臂</li><li data-step="r2g">红→绿</li><li data-step="place">放下（前伸 10 cm → 下降 10 cm → 松开）</li><li data-step="voice"><span>语音交互模块（全流程 v2）</span><ol id="voicesub"></ol></li></ol>
    <div id="missionlive" class="glive" role="status" hidden></div>
    <p id="missionstatus" class="status-line" role="status">尚未运行</p>
  </div></section>
  <section class="card"><div class="body">
    <h2>VLA 抓取</h2>
    <p class="status-line">当前模型：<strong id="vlamodel">读取中…</strong></p>
    <p><button id="modelsbtn" type="button" class="small" onclick="location.href='/models'">选择 / 切换模型 →</button></p>
    <p id="vlaconfig" class="status-line">读取模型配置中…</p>
    <button id="vlarun" class="primary" title="与全流程第 3 步完全相同：每次先复位张爪再抓；抓住（夹着并抬高 4 cm 保持 1.5 秒）立即停，原地保持夹紧；空夹后 5 秒内不张开重抓（卡住）、夹着 10 秒不抬或到时 → 下一次；空夹后模型自己张开重抓则继续推理；全部失败回起始姿态" disabled>开始抓取（同全流程第 3 步）</button>
    <label class="status-line">最多尝试 <select id="vlaattempts"><option value="5">5 次（同全流程）</option><option value="3">3 次</option><option value="1">1 次</option></select> · 每次最多 __ATTEMPT_S__ 秒</label>
    <div id="vla-progress" hidden>
      <div class="progress-row"><span id="vlaelapsed" role="status">已运行 0 / 60 s</span><button id="vlacancel" class="small">停止</button></div>
      <progress id="vlaprogress" max="60" value="0" aria-label="抓取运行进度" aria-describedby="vlaelapsed"></progress>
    </div>
    <p class="status-line">上次：<span id="vlast">尚未运行</span></p>
    <div id="vlalive" class="glive" role="status" hidden></div>
    <div class="arm-section">
      <div class="arm-controls"><h3>手臂</h3><button id="armstart" class="small" title="回到训练起始姿态，夹爪张开，约 10 秒" disabled>起始姿态</button><button id="armstarthold" class="small" title="回到训练起始姿态，正在夹着的夹爪保持夹紧（不松开），约 10 秒" disabled>起始姿态·保持夹爪</button><button id="armdown5" class="small" title="右臂末端竖直下降 5 cm，朝向不变，夹着的夹爪保持夹紧；每次按当前位置重新计算，最多降到末端 z = −0.40 m（不足 5 cm 时只降到下限），约 8 秒" disabled>右臂下降 5 cm</button><button id="armfwdleft" class="small" title="左臂末端沿直线向前 5 cm，朝向和夹爪不变；每次按当前位置重新计算，够不到或路径不安全时拒绝，约 8 秒" disabled>左臂前送 5 cm</button><button id="armfwdright" class="small" title="右臂末端沿直线向前 5 cm，朝向和夹爪不变；每次按当前位置重新计算，够不到或路径不安全时拒绝，约 8 秒" disabled>右臂前送 5 cm</button><button id="armopenright" class="small" title="张开右夹爪（全力张开），手臂保持当前姿态不动；夹着的东西会掉下来，约 3 秒" disabled>松开右爪</button><button id="armcloseright" class="small" title="闭合右夹爪（全力夹紧），手臂保持当前姿态不动，约 3 秒；之后保持夹紧" disabled>闭合右爪</button><button id="armraise" class="small" title="抬起手臂，约 8 秒" disabled>抬起</button><button id="armlower" class="small" title="返回抬起前的姿态，约 15 秒" disabled>放下</button></div>
      <p id="armst" class="status-line">尚未移动</p>
    </div>
  </div></section>
  <section class="card"><div class="body">
    <h2>底盘</h2>
    <div class="pad">
      <span></span><button data-d="forward" disabled>▲ 前进</button><span></span>
      <button data-d="left" disabled>◀ 左</button><button id="stop" title="停止当前底盘、路径、手臂或抓取任务">停</button><button data-d="right" disabled>▶ 右</button>
      <span></span><button data-d="back" disabled>▼ 后退</button><span></span>
    </div>
    <div class="rot"><button data-d="ccw" disabled>⟲ 逆时针</button><button data-d="cw" disabled>⟳ 顺时针</button></div>
    <p class="hint">点按 10 cm / 15°，长按连续移动（最长 10 秒）</p>
  </div></section>
  <details id="grip-details" class="card"><summary>夹爪标定记录</summary><div class="body">
    <p class="status-line">只读：记录所选夹爪约 2 秒内的位置、力矩、手腕高度（不发任何指令）。先让夹爪夹好，再点对应按钮。</p>
    <div class="grip-rec"><select id="griphand"><option value="right">右爪</option><option value="left">左爪</option></select>
      <button class="small" data-glabel="实体">记录：实体</button><button class="small" data-glabel="薄片">记录：薄片</button><button class="small" data-glabel="空夹">记录：空夹</button>
      <input id="griplabel" placeholder="其他标签" maxlength="40" size="10"><button class="small" id="gripcustom">记录</button></div>
    <p id="gripst" class="status-line" role="status"></p>
    <div class="grip-rec"><button class="small" id="gripclassify">判定当前状态</button>
      <span class="status-line" style="display:inline">用上面选的夹爪：张开 / 空夹 / 夹住（薄片） / 夹住（实体）</span></div>
    <p id="gripverdict" class="status-line" role="status" style="font-size:16px;font-weight:700"></p>
    <ol id="gripverdicts" class="status-line" style="font-size:12px;margin:0"></ol>
    <div style="overflow-x:auto"><table id="griptable" style="font-size:12px;border-collapse:collapse;width:100%"></table></div>
  </div></details>
  <details id="route-details" class="card"><summary>路线</summary><div class="body">
    <div class="route-buttons">
      <button data-route="g2r" disabled><span class="route-title"><span class="route-dot green" aria-hidden="true"></span>绿 → 红<span class="route-dot red" aria-hidden="true"></span></span><small>001 → 停 10 秒 → 002</small></button>
      <button data-route="r2g" disabled><span class="route-title"><span class="route-dot red" aria-hidden="true"></span>红 → 绿<span class="route-dot green" aria-hidden="true"></span></span><small>003 → 停 10 秒 → 004</small></button>
    </div>
    <p id="routest" class="status-line" role="status">上次：尚未运行</p>
    <p class="hint">路径从当前位置出发：先把机器人停在起点、朝向与录制时一致</p>
    <p class="hint">开始后仅接受急停；关闭页面或手机断网仍会继续执行</p>
  </div></details>
  <details id="paths-details" class="card"><summary>路径与轨迹</summary><div class="body">
    <div class="subhead"><h3>路径</h3><div class="tools"><button class="small" id="load" title="加载 CSV 路径文件">加载</button><button class="small" id="runall" title="按名称顺序运行全部路径，每条从上一条终点开始">全部运行</button></div></div>
    <input type="file" id="file" accept=".csv,text/csv" multiple hidden>
    <div class="plist" id="plist" title="点击路径行预览；路径从当前位置开始；全部运行时每条路径跑完停 10 秒"><div class="prow muted">尚未加载路径</div></div>
    <div class="trajectory">
      <div class="subhead"><h3>轨迹</h3><span id="rec" title="正在录制轨迹">● 录制中</span></div>
      <div id="pose"></div>
      <div class="canvas-wrap" title="轮式里程计；网格 10 厘米，上方为原点处的前进方向"><canvas id="cv" aria-label="路径预览与已录制轨迹"></canvas></div>
      <div class="tools"><button class="small" id="recstart">开始录制</button><button class="small" id="recstop" disabled>停止录制</button><button class="small" id="save" title="下载轨迹 CSV 并在服务端保留副本" disabled>保存</button><button class="small" id="reset" title="以当前位姿重置里程计原点并清空轨迹">重置</button></div>
    </div>
  </div></details>
  <details id="log-details" class="card"><summary>日志</summary><pre id="log" aria-label="运行日志"></pre></details>
</main><script>

const st=document.getElementById('status'),lg=document.getElementById('log'),poseEl=document.getElementById('pose');
const moves=[...document.querySelectorAll('button[data-d]')];
let press=null;
function post(u){return fetch(u,{method:'POST',keepalive:true}).then(r=>r.json()).catch(e=>({error:String(e)}))}
function begin(b){
  if(press||b.disabled)return;
  press={d:b.dataset.d,btn:b,hb:null};
  b.classList.add('on');
  press.timer=setTimeout(()=>{press.timer=null;post('/hold/'+press.d);
    press.hb=setInterval(()=>post('/hold/'+press.d),100)},300);
}
function end(){
  if(!press)return;
  const p=press;press=null;p.btn.classList.remove('on');
  if(p.timer){clearTimeout(p.timer);post('/step/'+p.d)}else{clearInterval(p.hb);post('/release')}
}
moves.forEach(b=>{
  b.addEventListener('pointerdown',e=>{e.preventDefault();b.setPointerCapture(e.pointerId);begin(b)});
  b.addEventListener('pointerup',end);b.addEventListener('pointercancel',end);
  b.addEventListener('lostpointercapture',end);b.addEventListener('contextmenu',e=>e.preventDefault());
});
window.addEventListener('blur',end);
document.addEventListener('visibilitychange',()=>{if(document.hidden)end()});
document.getElementById('stop').onclick=document.getElementById('vlacancel').onclick=()=>{end();post('/stop')};
const estopBtn=document.getElementById('estop');
const doEstop=()=>{if(estopBtn.disabled)return;end();post('/estop')};estopBtn.onclick=doEstop;
document.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();doEstop()}});
const modeEl=document.getElementById('mode'),connectionEl=document.getElementById('connection'),connectionText=document.getElementById('connection-text');
const directions={forward:'前进',back:'后退',left:'向左',right:'向右',ccw:'逆时针',cw:'顺时针'};
const armActions={start:'起始姿态',start_hold:'起始姿态（保持夹爪）',down5:'右臂下降 5 cm',fwd5_left:'左臂前送 5 cm',fwd5_right:'右臂前送 5 cm',open_right:'松开右爪',close_right:'闭合右爪',open_left:'松开左爪',close_left:'闭合左爪',raise:'抬起',lower:'放下'};
function uiText(value){
  let t=String(value??'');
  const phrases=[['route running: only E-STOP is accepted','路线执行中，仅接受急停'],
    ['arms: not moved yet','尚未移动'],['VLA: not run yet','尚未运行'],
    ['busy: wait until the robot is still','忙碌中，请等待机器人静止'],['arms are moving','手臂或抓取任务正在运行'],
    ['not in API mode or chassis disabled (or mode unknown yet)','非 API 模式、底盘未启用或模式未知'],
    ['no fresh wheel feedback from the robot','尚未收到机器人最新轮速反馈'],['busy, path not started','忙碌中，路径未启动'],
    ['exited rc=','退出码：'],['time limit without a grasp','到时限仍未抓住'],['all grasp attempts failed; arms reset','全部尝试都未抓住，手臂已复位'],['grasped (left)','已抓住（左手），已停止'],['grasped (right)','已抓住（右手），已停止'],
    ['no grasp (','未抓住，已提前结束（'],['left gripper closed on nothing','左爪空夹'],['without a grasp','，未抓住'],['VLA script exited','VLA 脚本退出码 '],['right gripper closed on nothing','右爪空夹'],
    ['left gripper holding but not lifted within','左爪夹着但未按时抬起：'],['right gripper holding but not lifted within','右爪夹着但未按时抬起：'],['before running the policy (see log)','，策略尚未运行（详见日志）'],
    ['FSM pause failed rc=','暂停失败，退出码：'],['FSM paused','已暂停'],['E-STOP','急停'],
    ['time_limit','达到时限'],['path done','路径已完成'],['corner pause','转角停留'],[' done, pausing ',' 已完成，停留 '],[' s before ',' 秒后开始 '],['run all: ','全部运行：'],['heartbeat timeout','心跳超时'],
    ['dead-man timeout','心跳超时'],['step done','点按移动已完成'],['hold timeout','长按已达时限'],
    ['page closed or connection lost','页面已关闭或连接中断'],['wheel feedback lost','轮速反馈中断'],
    ['hold limit','长按达到时限'],['stuck: no progress for','移动受阻，未取得进展已持续'],
    ['unknown route','未知路线'],['missing paths: ','未加载路径：'],['teleop disconnected','机器人连接已断开'],
    ['stopped before ','已中止，未开始路径 '],['stopped at ','已中止于路径 '],['finished','已完成'],
    ['trail is empty','轨迹为空'],['starting','正在启动'],['ready','就绪'],['stopped','已停止'],
    ['released','已松开'],['release','松开'],['timeout','超时'],['busy','忙碌中'],['done','已完成'],['error','错误']];
  for(const [from,to] of phrases)t=t.split(from).join(to);
  t=t.replace(/\barms (start_hold|start|down5|fwd5_left|fwd5_right|open_right|close_right|open_left|close_left|raise|lower):/g,(_,a)=>'手臂'+armActions[a]+'：')
    .replace(/\b(step|hold) (forward|back|left|right|ccw|cw)\b/g,(_,k,d)=>(k==='step'?'点按':'持续')+directions[d])
    .replace(/\bpath /g,'路径 ').replace(/\bsteps\b/g,'步').replace(/\bNone\b/g,'无');
  return t;
}
document.getElementById('reset').onclick=()=>post('/reset');
document.getElementById('recstart').onclick=()=>post('/rec/start');
document.getElementById('recstop').onclick=()=>post('/rec/stop');
document.getElementById('save').onclick=async()=>{
  const r=await post('/rec/save');if(r.error){st.textContent=uiText(r.error);return}
  const a=document.createElement('a');a.href='/trail.csv';a.download='';document.body.appendChild(a);a.click();a.remove();
};
const recEl=document.getElementById('rec'),recStart=document.getElementById('recstart'),recStop=document.getElementById('recstop'),saveBtn=document.getElementById('save');
let recording=false;

// paths
let paths={},sel=null,active=null,anchor=[0,0,0],seqNames=[],seqI=-1;
const plist=document.getElementById('plist'),fileIn=document.getElementById('file');
async function loadPaths(){try{const r=await (await fetch('/paths')).json();paths={};r.paths.forEach(p=>paths[p.name]=p);renderPaths()}catch(e){}}
function renderPaths(){
  const names=Object.keys(paths).sort();
  if(!names.length){plist.innerHTML='<div class="prow muted">尚未加载路径</div>';return}
  plist.innerHTML='';
  names.forEach(n=>{const p=paths[n],row=document.createElement('div');
    row.className='prow'+(n===sel?' sel':'')+(n===active?' run':'');
    const q=seqNames.length?(seqNames.indexOf(n)<seqI?' 已完成':''):'';
    row.innerHTML='<span class="nm"></span><span class="info"></span><button class="small">运行</button>';
    row.querySelector('.nm').textContent=n;
    row.querySelector('.info').textContent=p.length.toFixed(2)+' m → ('+(p.end[0]*100).toFixed(0)+', '+(p.end[1]*100).toFixed(0)+') cm'+q;
    row.title='点击预览路径';row.tabIndex=0;
    row.setAttribute('aria-label',n+'，点击或按回车预览路径');
    row.onclick=()=>{sel=(sel===n?null:n);renderPaths();draw()};
    row.onkeydown=e=>{if(e.target===row&&(e.key==='Enter'||e.key===' ')){e.preventDefault();row.click()}};
    const b=row.querySelector('button');b.disabled=!!active||seqNames.length>0;
    b.onclick=async e=>{e.stopPropagation();const r=await post('/paths/run/'+encodeURIComponent(n));if(r.error)st.textContent=uiText(r.error)};
    plist.appendChild(row)});
  document.getElementById('runall').disabled=!!active||seqNames.length>0;
}
document.getElementById('load').onclick=()=>fileIn.click();
fileIn.onchange=async()=>{
  const fs=[...fileIn.files].sort((a,b)=>a.name.localeCompare(b.name));
  for(const f of fs){const r=await fetch('/paths/upload',{method:'POST',body:JSON.stringify({name:f.name,csv:await f.text()})}).then(r=>r.json()).catch(e=>({error:String(e)}));
    if(r.error)st.textContent=f.name+'：'+uiText(r.error)}
  fileIn.value='';loadPaths();
};
const armRaise=document.getElementById('armraise'),armLower=document.getElementById('armlower'),armStart=document.getElementById('armstart'),armSt=document.getElementById('armst');
armRaise.onclick=async()=>{const r=await post('/arm/raise');if(r.error)st.textContent=uiText(r.error)};
armLower.onclick=async()=>{const r=await post('/arm/lower');if(r.error)st.textContent=uiText(r.error)};
armStart.onclick=async()=>{const r=await post('/arm/start');if(r.error)st.textContent=uiText(r.error)};
const armCloseRight=document.getElementById('armcloseright');armCloseRight.onclick=async()=>{const r=await post('/arm/close_right');if(r.error)st.textContent=r.error};
const armOpenRight=document.getElementById('armopenright');armOpenRight.onclick=async()=>{const r=await post('/arm/open_right');if(r.error)st.textContent=r.error};
async function gripRecord(label){
  const st2=document.getElementById('gripst');if(!label.trim()){st2.textContent='请先填写标签';return}
  st2.textContent='记录中（约 2 秒）…';document.querySelectorAll('[data-glabel],#gripcustom').forEach(b=>b.disabled=true);
  try{const r=await fetch('/api/v1/grippers/record',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({label,hand:document.getElementById('griphand').value,seconds:2})});
    const d=await r.json();if(!r.ok)throw Error((d.error&&d.error.message)||r.status);
    st2.textContent='已记录：'+d.label+' · '+(d.hand==='right'?'右爪':'左爪')+' · 位置 '+d.pos_rad.mean+' rad · 力矩 '+d.eff_nm.mean+' N·m';
  }catch(e){st2.textContent='记录失败：'+e.message}
  finally{document.querySelectorAll('[data-glabel],#gripcustom').forEach(b=>b.disabled=false);loadGripRecords()}
}
async function loadGripRecords(){
  try{const d=await (await fetch('/api/v1/grippers/records?limit=50')).json();const t=document.getElementById('griptable');
    const f=x=>x?x.mean+' ('+x.min+'～'+x.max+')':'—';
    t.innerHTML='<tr><th>时间</th><th>标签</th><th>手</th><th>位置 rad 均值(最小～最大)</th><th>力矩 N·m 均值(最小～最大)</th><th>手腕高度 m</th><th>现规则判夹住</th></tr>'+
      d.records.map(r=>'<tr><td>'+r.time.slice(11,19)+'</td><td>'+r.label.replace(/[<&]/g,'')+'</td><td>'+(r.hand==='right'?'右':'左')+'</td><td>'+f(r.pos_rad)+'</td><td>'+f(r.eff_nm)+'</td><td>'+(r.ee_z_m?r.ee_z_m.mean:'—')+'</td><td>'+r.holding_now_rule+'/'+r.samples+'</td></tr>').join('');
    t.querySelectorAll('th,td').forEach(c=>{c.style.border='1px solid var(--line,#ccc)';c.style.padding='2px 6px'});
  }catch(e){}
}
document.querySelectorAll('[data-glabel]').forEach(b=>b.onclick=()=>gripRecord(b.dataset.glabel));
document.getElementById('gripclassify').onclick=async()=>{
  const out=document.getElementById('gripverdict'),btn=document.getElementById('gripclassify'),hand=document.getElementById('griphand').value;
  btn.disabled=true;out.textContent='判定中（约 1 秒）…';
  try{const r=await fetch('/api/v1/grippers/classify?hand='+hand+'&seconds=1');const d=await r.json();
    if(!r.ok)throw Error((d.error&&d.error.message)||r.status);
    const text=(hand==='right'?'右爪':'左爪')+'：'+d.label+(d.fresh?'':'（读数不新鲜）');
    out.textContent=text;const li=document.createElement('li');
    li.textContent=new Date().toTimeString().slice(0,8)+' '+text+' · '+d.why+' · 帧 '+JSON.stringify(d.counts);
    const ol=document.getElementById('gripverdicts');ol.prepend(li);while(ol.children.length>10)ol.lastChild.remove();
  }catch(e){out.textContent='判定失败：'+e.message}
  finally{btn.disabled=false}
};
document.getElementById('gripcustom').onclick=()=>gripRecord(document.getElementById('griplabel').value);
loadGripRecords();
const armDown5=document.getElementById('armdown5');armDown5.onclick=async()=>{const r=await post('/arm/down5');if(r.error)st.textContent=r.error};
const armFwdLeft=document.getElementById('armfwdleft');armFwdLeft.onclick=async()=>{const r=await post('/arm/fwd5_left');if(r.error)st.textContent=r.error};
const armFwdRight=document.getElementById('armfwdright');armFwdRight.onclick=async()=>{const r=await post('/arm/fwd5_right');if(r.error)st.textContent=r.error};
const armStartHold=document.getElementById('armstarthold');armStartHold.onclick=async()=>{const r=await post('/arm/start_hold');if(r.error)st.textContent=r.error};
const vlaBtn=document.getElementById('vlarun'),vlaSt=document.getElementById('vlast');
const routeBtns=[...document.querySelectorAll('button[data-route]')],routeSt=document.getElementById('routest');
const routeLabels={g2r:'绿 → 红',r2g:'红 → 绿'};
routeBtns.forEach(b=>b.onclick=async()=>{const r=await post('/route/'+b.dataset.route);if(r.error)routeSt.textContent=uiText(r.error)});
const missionBtn=document.getElementById('missionrun'),missionSt=document.getElementById('missionstatus');
async function startMission(path,btn){btn.disabled=true;try{const r=await post(path);if(r.error)missionSt.textContent=uiText(typeof r.error==='string'?r.error:r.error.message)}catch(e){missionSt.textContent='无法开始任务'}}
missionBtn.onclick=()=>startMission('/mission/run',missionBtn);
const missionBtn2=document.getElementById('missionrun2'),missionVoiceBtn=document.getElementById('missionvoice');
missionBtn2.onclick=()=>startMission('/mission/run/v2',missionBtn2);
missionVoiceBtn.onclick=()=>startMission('/mission/run/voice',missionVoiceBtn);
const missionHandoffBtn=document.getElementById('missionhandoff');
missionHandoffBtn.onclick=()=>startMission('/mission/run/handoff',missionHandoffBtn);
const VOICE_STEPS=__VOICE_STEPS__;
function renderVoice(m){
  const steps=(m.voice&&m.voice.steps)||VOICE_STEPS.map(([id,name,uses])=>({id,name,uses,status:'pending',detail:''}));
  const mark={pending:'',running:' …',succeeded:' ✓',failed:' ✗',skipped:' –'};
  const ol=document.getElementById('voicesub');ol.textContent='';
  steps.forEach(st=>{const li=document.createElement('li');li.className='sub-'+st.status;
    li.textContent=st.id+' '+st.name+(mark[st.status]||'')+(st.detail?' · '+st.detail:'');
    const u=document.createElement('small');u.textContent='　调用：'+st.uses;li.appendChild(u);ol.appendChild(li);
    if(st.id==='7.1'&&st.status==='running'){const d=document.createElement('div');d.className='listen';d.setAttribute('role','status');
      d.innerHTML='<span class="mic">🎤</span><span>正在听，请说「小度小度，请把沙包给我」</span><span class="bars"><i></i><i></i><i></i><i></i></span>';li.appendChild(d)}});
}
// Grasp attempt internals (mission state "live", same for the big button, step 3 and step 7.2).
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
const PHASES=[['reset','复位：回起始姿态、张开夹爪'],['probe','取帧：相机 + 关节'],['policy','VLA 推理执行'],['judge','判定结果']];
const PHASE_NOW={starting:'准备中',reset:'复位中（回起始姿态、张开夹爪）',probe:'取帧中（相机 + 关节）',policy:'VLA 推理执行中',
  policy_retry:'启动时没收到数据，自动重来一次',stopping:'判定已出，正在停止 VLA',ended:'本次已结束'};
function handLine(name,h,r,side){
  if(!h)return '<div class="hand"><b>'+name+'</b> 无读数</div>';
  const n=x=>x==null?'–':Number(x).toFixed(2);
  let t='<div class="hand"><b>'+name+'</b> 夹爪 '+n(h.pos)+' rad · 力 '+n(h.eff)+' · ';
  if(!h.armed)t+='等待张开（张开前不判定）';
  else if(h.grip==='open')t+='张开，等待合拢';
  else if(h.grip==='empty')t+='<span class="warn">空夹 '+h.empty_s.toFixed(1)+' / '+r.empty_s+' s → 到时本次失败</span>';
  else if(h.grip==='thin'||h.grip==='solid'){
    t+='<span class="good">夹住（'+(h.grip==='solid'?'实体':'薄片')+'）</span> · 抬高 '+(h.lift_cm==null?'–':h.lift_cm.toFixed(1))+' / '+r.lift_cm+' cm';
    t+=h.good_s>0?' · <span class="good">成功计时 '+h.good_s.toFixed(1)+' / '+r.success_s+' s</span>':' · 还没抬够';
    t+=' · 已夹 '+h.hold_s.toFixed(1)+' / '+r.no_lift_s+' s（超时不抬判失败）';
  }else if(h.grip==='closing')t+='合拢中（还没用力夹紧）';
  else t+='无有效读数';
  return t+'</div>';
}
function renderLive(el,L,show){
  el.hidden=!(show&&L);if(el.hidden)return;
  const r=L.rules||{},ended=L.phase==='ended',idx={starting:0,reset:0,probe:1,policy:2,policy_retry:2,stopping:3,ended:3}[L.phase]??0;
  let h='<div class="gh">抓取内部状态 · 第 '+L.attempt+'/'+L.attempts+' 次 · '+esc(PHASE_NOW[L.phase]||L.phase)+
    ' · 本次 '+(L.elapsed_s||0).toFixed(0)+' / '+r.limit_s+' s</div><div class="gp">';
  PHASES.forEach(([k,label],i)=>{let c=i<idx?'ok':i===idx?'on':'';
    if(i===3&&ended)c=L.verdict==='succeeded'?'ok':'bad';
    let txt=label;if(i===3&&ended)txt=L.verdict==='succeeded'?'✓ 抓住（'+(L.hand==='left'?'左手':'右手')+'）':'✗ '+uiText(L.note||'失败');
    h+='<span class="'+c+'">'+esc(txt)+'</span>'});
  h+='</div>';
  if(L.note&&!ended)h+='<div class="hand">'+esc(uiText(L.note))+'</div>';
  if(!ended||L.hands.left){h+=handLine('左手',L.hands.left,r,'left')+handLine('右手',L.hands.right,r,'right')}
  h+='<div class="hand"><small>规则：夹住 = 力 ≤ '+r.effort_max+' 且夹爪高于空夹位 · 成功 = 夹住并抬高 ≥ '+r.lift_cm+' cm 保持 '+r.success_s+' s · 空夹 '+r.empty_s+' s 或夹着 '+r.no_lift_s+' s 不抬 → 本次失败，进入下一次</small></div>';
  el.innerHTML=h;
}
function renderHistory(el,m){
  const list=m.attempt_results&&m.attempt_results.length?m.attempt_results:(m.voice&&m.voice.attempt_results)||[];
  if(el.hidden||!list.length)return;
  const d=document.createElement('div');d.className='hist';
  d.innerHTML='历次尝试：'+list.map(a=>a.result==='succeeded'?'<span class="ok">#'+a.attempt+' ✓ '+(a.hand==='left'?'左手':'右手')+(a.seconds!=null?'（'+a.seconds.toFixed(0)+' s）':'')+'</span>':
    a.result==='running'?'#'+a.attempt+' …':'<span class="bad">#'+a.attempt+' ✗ '+esc(uiText(a.reason||a.result))+(a.seconds!=null?'（'+a.seconds.toFixed(0)+' s）':'')+'</span>').join('　');
  el.appendChild(d);
}
function updateMission(s){
  const m=s.mission||{},active=!!m.active;
  const mc=m.config,mm=document.getElementById('missionmodel');
  if(active&&mc)mm.textContent='本次使用：'+(mc.model_name||('端口 '+mc.port))+'（'+mc.prompt+'）· '+(mc.rtc?'RTC 流畅执行':'同步执行');
  else if(s.vla_selection)mm.textContent='将使用：'+s.vla_selection.name+'（'+s.vla_selection.prompt+'）· '+(s.vla_selection.rtc?'RTC 流畅执行':'同步执行')+' · 在上方「抓取模型」里切换';
  missionBtn.disabled=active||!s.connected||s.mode_ok!==true||!!s.estop||!!s.moving||s.seq.length>0||!!s.arm||!!s.vla;
  missionBtn2.disabled=missionVoiceBtn.disabled=missionHandoffBtn.disabled=missionBtn.disabled;
  renderVoice(m);
  const ml=document.getElementById('missionlive');renderLive(ml,m.live,!!m.live);renderHistory(ml,m);
  const vl=document.getElementById('vlalive');renderLive(vl,m.live,!!m.live&&m.flow==='grasp');renderHistory(vl,m);
  document.querySelectorAll('#missionsteps li').forEach(el=>{
    el.classList.toggle('mission-current',el.dataset.step===m.step||(m.step==='handoff'&&el.dataset.step==='voice'));
    if(el.dataset.step==='grasp')el.textContent='抓取 '+(m.attempt||0)+'/'+(m.attempts||5);
  });
  // ✓ steps already passed, … the running step, ✗ the step where the run failed or was stopped
  document.querySelectorAll('#missionsteps > li').forEach(el=>{
    const key=el.dataset.step==='voice'&&m.step==='handoff'?'handoff':el.dataset.step;
    const cur=m.step===key&&m.status&&m.status!=='idle';
    const st=cur&&m.status==='running'?'running':cur&&(m.status==='failed'||m.status==='aborted')?'failed':
      (m.durations&&key in m.durations)?'done':'';
    let mk=el.querySelector(':scope > .mk');
    if(!mk){mk=document.createElement('span');el.insertBefore(mk,el.querySelector(':scope > ol'))}
    mk.className='mk'+(st?' mk-'+st:'');mk.textContent={done:' ✓',failed:' ✗',running:' …'}[st]||'';
  });
  const results={idle:'尚未运行',running:'运行中',succeeded:'成功',failed:'失败',aborted:'已急停中止'};
  missionSt.textContent=({v1:'全流程 · ',v2:'全流程 v2 · ',voice:'仅第 7 步 · ',grasp:'仅抓取（同第 3 步） · ',handoff:'仅放下（7.4–7.7） · '}[m.flow]||'')+(results[m.status]||'尚未运行')+(m.hand?' · 抓取手：'+(m.hand==='left'?'左手':'右手'):'')+(m.pause_left_s>0?' · 停留 '+Math.ceil(m.pause_left_s)+' 秒':'')+(m.result?' · '+m.result:'')+(active?' · 仅接受急停':'');
  if(active)document.querySelectorAll('button').forEach(b=>{if(b.id!=='estop')b.disabled=true});
}
function updateRoute(s){
  routeBtns.forEach(b=>b.disabled=!!s.moving||s.seq.length>0||!!s.arm||!!s.vla||!!s.estop||!s.connected);
  document.getElementById('stop').disabled=!!s.route;
  if(!s.route){routeSt.textContent='上次：'+(s.route_last?uiText(s.route_last):'尚未运行');return}
  const title=routeLabels[s.route]+'：',i=s.seq_i;
  routeSt.textContent=s.pause_left>0?title+'停留中，还剩 '+Math.ceil(s.pause_left)+' 秒，然后执行 '+s.seq[i+1]:
    i<0?title+'准备中…':title+'执行 '+s.seq[i]+'（'+(i+1)+'/'+s.seq.length+'）';
}
let selectedVlaId=null,vlaSeconds=60;
// The big button runs exactly full-flow step 3 (mission flow 'grasp'): same attempts, reset, detector and ending.
const vlaAttempts=document.getElementById('vlaattempts');
try{const saved=localStorage.getItem('ta2_vla_attempts_step3');if(saved)vlaAttempts.value=saved}catch(e){}
vlaAttempts.onchange=()=>{try{localStorage.setItem('ta2_vla_attempts_step3',vlaAttempts.value)}catch(e){}};
vlaBtn.onclick=async()=>{
  if(!selectedVlaId)return;
  try{const r=await fetch('/api/v1/actions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({type:'mission.run',params:{flow:'grasp',attempts:Number(vlaAttempts.value)}})});
    const d=await r.json();if(!r.ok)st.textContent=uiText((d.error&&d.error.message)||String(r.status));
  }catch(e){st.textContent=uiText(String(e))}
};
const vlaProgress=document.getElementById('vla-progress'),vlaMeter=document.getElementById('vlaprogress'),vlaElapsed=document.getElementById('vlaelapsed');
let vlaStarted=null;
function updateVlaProgress(){
  if(vlaStarted===null)return;
  const elapsed=Math.floor((performance.now()-vlaStarted)/1000);
  vlaMeter.value=Math.min(vlaSeconds,elapsed);
  vlaElapsed.textContent=elapsed>=vlaSeconds?'收尾中…':'已运行 '+elapsed+' / '+vlaSeconds+' s';
}
setInterval(updateVlaProgress,200);
document.getElementById('runall').onclick=async()=>{const r=await post('/paths/runall');if(r.error)st.textContent=uiText(r.error)};
loadPaths();

// trajectory
const cv=document.getElementById('cv');let trail=[],epoch=-1,pose=[0,0,0];
function anchored(p,a){  // path points in the path's own frame -> odometry frame, first point at pose a
  const [x0,y0,t0]=p.points[0],r=a[2]-t0,c=Math.cos(r),s=Math.sin(r);
  return p.points.map(q=>[a[0]+(q[0]-x0)*c-(q[1]-y0)*s,a[1]+(q[0]-x0)*s+(q[1]-y0)*c]);
}
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function draw(){
  const dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
  if(!W||!H)return;
  cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,W,H);
  const show=active&&paths[active]?anchored(paths[active],anchor):(sel&&paths[sel]?anchored(paths[sel],pose):null);
  const pts=trail.concat([[pose[0],pose[1]]]).concat(show||[]);
  let x0=Math.min(...pts.map(p=>p[0])),x1=Math.max(...pts.map(p=>p[0]));
  let y0=Math.min(...pts.map(p=>p[1])),y1=Math.max(...pts.map(p=>p[1]));
  const cx=(x0+x1)/2,cy=(y0+y1)/2,half=Math.max(0.6,(x1-x0)/2+0.45,(y1-y0)/2+0.45);
  const s=Math.min(W,H)/(2*half);
  const X=x=>W/2+(x-cx)*s,Y=y=>H/2-(y-cy)*s;
  // grid every 10 cm, stronger every 50 cm
  g.lineWidth=1;
  const gx0=Math.floor((cx-W/2/s)*10),gx1=Math.ceil((cx+W/2/s)*10),gy0=Math.floor((cy-H/2/s)*10),gy1=Math.ceil((cy+H/2/s)*10);
  for(let i=gx0;i<=gx1;i++){g.strokeStyle=i%5?css('--grid'):css('--line');g.beginPath();g.moveTo(X(i/10),0);g.lineTo(X(i/10),H);g.stroke()}
  for(let i=gy0;i<=gy1;i++){g.strokeStyle=i%5?css('--grid'):css('--line');g.beginPath();g.moveTo(0,Y(i/10));g.lineTo(W,Y(i/10));g.stroke()}
  // origin
  g.fillStyle=css('--muted');g.beginPath();g.arc(X(0),Y(0),4,0,7);g.fill();
  g.font='11px -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';g.fillText('原点',X(0)+6,Y(0)-6);
  // trail
  if(trail.length>0){g.strokeStyle=css('--trail');g.lineWidth=2.5;g.lineJoin='round';g.beginPath();
    trail.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));
    if(recording)g.lineTo(X(pose[0]),Y(pose[1]));g.stroke()}
  // planned path (selected preview, or the one running)
  if(show){g.setLineDash([6,5]);g.strokeStyle=css('--accent');g.lineWidth=2;g.beginPath();
    show.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();g.setLineDash([]);
    const e=show[show.length-1];g.fillStyle=css('--accent');g.beginPath();g.arc(X(e[0]),Y(e[1]),4,0,7);g.fill();
    g.fillText(active||sel,X(e[0])+6,Y(e[1])-6)}
  // robot footprint 0.62 x 0.625 m, forward = +y rotated by th (CCW)
  const [px,py,th]=pose,c=Math.cos(th),sn=Math.sin(th);
  const R=(dx,dy)=>[X(px+dx*c-dy*sn),Y(py+dx*sn+dy*c)];
  const hw=0.31,hl=0.3125;
  g.strokeStyle=css('--accent');g.lineWidth=2;g.fillStyle=css('--accent')+'22';g.beginPath();
  [[-hw,-hl],[hw,-hl],[hw,hl],[-hw,hl]].forEach((q,i)=>{const r=R(q[0],q[1]);i?g.lineTo(r[0],r[1]):g.moveTo(r[0],r[1])});
  g.closePath();g.fill();g.stroke();
  const a=R(0,0),b=R(0,hl+0.08),l=R(-0.06,hl-0.02),r=R(0.06,hl-0.02);
  g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke();
  g.fillStyle=css('--accent');g.beginPath();g.moveTo(b[0],b[1]);g.lineTo(l[0],l[1]);g.lineTo(r[0],r[1]);g.closePath();g.fill();
}
window.addEventListener('resize',draw);
document.getElementById('paths-details').addEventListener('toggle',e=>{if(e.currentTarget.open)draw()});
document.getElementById('log-details').addEventListener('toggle',e=>{if(e.currentTarget.open)lg.scrollTop=lg.scrollHeight});
window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);

async function poll(){try{
  const s=await (await fetch('/status?epoch='+epoch+'&since='+trail.length)).json();
  if(s.epoch!==epoch||s.trail_from===0){trail=[];epoch=s.epoch}
  trail=trail.concat(s.trail);pose=s.pose;recording=s.recording;
  const pn=(s.path_names||[]).join(',');if(pn!==Object.keys(paths).sort().join(','))loadPaths();
  if(active!==s.active_path||seqI!==s.seq_i||seqNames.length!==s.seq.length){active=s.active_path;anchor=s.path_anchor;seqNames=s.seq;seqI=s.seq_i;renderPaths()}
  recEl.style.visibility=recording?'visible':'hidden';recStart.disabled=recording;recStop.disabled=!recording;saveBtn.disabled=!trail.length;
  poseEl.textContent='横向 '+(pose[0]*100).toFixed(1)+' cm，前向 '+(pose[1]*100).toFixed(1)+' cm，朝向 '+(pose[2]*180/Math.PI).toFixed(1)+'°';
  connectionText.textContent=s.connected?'已连接':'连接中…';connectionEl.className='connection'+(s.connected?' ok':'');
  let t=!s.connected?'正在连接机器人…':s.moving?('移动中：'+uiText(s.moving)):uiText(s.last);
  if(s.seq.length)t='全部运行：'+s.seq.map((n,i)=>i===s.seq_i?'['+n+']':n).join(' → ')+(s.moving?'（'+uiText(s.moving)+'）':'');
  if(s.arm)t='手臂'+(armActions[s.arm]||s.arm)+'进行中（停止后保持当前姿态）';
  if(s.vla)t='VLA 抓取进行中（可停止或按 Esc 急停）';
  if(s.estop)t='急停中…';else if(s.estop_last&&s.last.startsWith('E-STOP'))t='急停：'+uiText(s.estop_last);
  modeEl.textContent=s.mode_ok===true?'API 模式':s.mode_ok===false?'非 API 模式':'模式未知';
  estopBtn.disabled=!!s.estop;estopBtn.textContent=s.estop?'急停中…':'急停';
  armSt.textContent=s.arm?((armActions[s.arm]||s.arm)+'进行中…'):uiText(s.arm_last);
  const still=!s.moving&&!s.seq.length&&!s.arm&&!s.vla;
  const model=s.vla_running||s.vla_selection;
  if(s.vla_selection){
    selectedVlaId=s.vla_selection.id;vlaSeconds=model.seconds;vlaMeter.max=vlaSeconds;
    document.getElementById('vlamodel').textContent=model.name;
    document.getElementById('vlaconfig').textContent=model.prompt+' · 速度上限 1 rad/s';
    document.getElementById('modelsbtn').disabled=!still||!!s.estop;
  }
  const gm=s.mission||{};
  if(gm.flow==='grasp'&&gm.status&&gm.status!=='idle'){
    const res={running:'进行中',succeeded:'已抓住',failed:'未抓住',aborted:'已急停中止'};
    const last=(gm.attempt_results||[]).slice(-1)[0];
    vlaSt.textContent='抓取（同第 3 步）：第 '+(gm.attempt||0)+'/'+(gm.attempts||0)+' 次 · '+(res[gm.status]||gm.status||'')+
      (gm.hand?'（'+(gm.hand==='left'?'左手':'右手')+'，原地保持夹紧）':'')+
      (last&&last.reason?' · 上一次：'+uiText(last.reason):'')+(gm.status!=='running'&&gm.result?' · '+uiText(gm.result):'');
  }else vlaSt.textContent=uiText(s.vla_last);
  if(s.vla){if(vlaStarted===null)vlaStarted=performance.now()}else vlaStarted=null;
  vlaProgress.hidden=!s.vla;updateVlaProgress();
  updateRoute(s);
  armRaise.disabled=!still;armLower.disabled=!still;armStart.disabled=!still;armStartHold.disabled=!still;armDown5.disabled=!still;armFwdLeft.disabled=!still;armFwdRight.disabled=!still;armOpenRight.disabled=!still;armCloseRight.disabled=!still;vlaBtn.disabled=!still||!!s.estop||!s.connected;
  if(s.mission&&s.mission.active)t="全流程运行中，仅接受急停";
  st.textContent=t;st.title=t;
  const busyStep=s.moving&&(s.moving.startsWith('step')||s.moving.startsWith('path'))||s.seq.length>0||!!s.arm;
  const pathBusy=!!s.active_path||s.seq.length>0||!!s.arm;
  document.querySelectorAll('.prow button').forEach(b=>b.disabled=pathBusy);
  document.getElementById('runall').disabled=pathBusy;
  moves.forEach(x=>x.disabled=!press&&(busyStep||!s.connected));
  updateMission(s);
  lg.textContent=s.log.join('\n');lg.scrollTop=lg.scrollHeight;draw();
}catch(e){connectionText.textContent='B70 不可达';connectionEl.className='connection';st.textContent='无法获取机器人状态';st.title=st.textContent;routeBtns.forEach(b=>b.disabled=true);if(!press)moves.forEach(x=>x.disabled=true)}}
poll();setInterval(poll,300);
</script></body></html>""".replace("__VOICE_STEPS__", json.dumps(mission_module.VOICE_STEPS, ensure_ascii=False)).replace("__ATTEMPT_S__", f"{mission_module.ATTEMPT_S:g}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if self.path.startswith("/api/v1/") and code == 503 and body == {"ok": False}:
            body = {"error": {"code": "TELEOP_DISCONNECTED", "message": "teleop disconnected"}}
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def api_error(self, error):
        return self._send(error.status, {"error": {"code": error.code, "message": error.message}})

    def api_GET(self, path, query):
        # Monitor startup/service inspection must not block STOP or action polling.
        if path == "/api/v1/grippers":
            return self._send(200, gripper_monitor.readings())
        if path == "/api/v1/robot-check":
            return self._send(200, robot_check())
        if path == "/api/v1/grippers/classify":
            q = parse_qs(query)
            try:
                seconds = float(q.get("seconds", ["1"])[0])
            except ValueError:
                raise APIError(400, "BAD_REQUEST", "seconds must be a number")
            return self._send(200, classify_gripper(q.get("hand", ["right"])[0], seconds))
        if path == "/api/v1/grippers/records":
            raw = parse_qs(query).get("limit", ["100"])[0]
            limit = int(raw) if raw.isdigit() and 1 <= int(raw) <= 1000 else 100
            return self._send(200, {"file": GRIP_LOG, "records": gripper_records(limit)})
        if path == "/api/v1/preflight":
            model_id = parse_qs(query, keep_blank_values=True).get("model", [state["vla_selected"]])[0]
            return self._send(200, api_preflight(model_id))
        if path == "/api/v1/models":
            services = model_services()
            with seq_lock:
                selected = state["vla_selected"]
            return self._send(200, {"models": [{**m, "selected": m["id"] == selected} for m in services],
                                    "selected": selected})
        with seq_lock:
            if path == "/api/v1/health":
                action = current_action[0]
                return self._send(200, {"ok": True, "connected": state["connected"],
                                       "api_mode": state["mode_ok"], "estop_active": state["estop"],
                                       "busy": busy(), "current_action": action["id"] if action else None})
            if path == "/api/v1/state":
                return self._send(200, api_state())
            if path == "/api/v1/capabilities":
                sel = VLA_MODELS[state["vla_selected"]]
                return self._send(200, {"actions": ACTION_PARAMS, "paths": sorted(paths),
                                       "routes": [{"id": key, "name": name, "paths": names}
                                                  for key, (name, names) in ROUTES.items()],
                                       "arm_poses": list(ARM_TIME),
                                       "vla": {"seconds": sel["seconds"], "max_rad_s": sel["speed"],
                                               "prompt": sel["prompt"], "model": sel["id"]}})
            if path == "/api/v1/actions":
                try:
                    limit = int(parse_qs(query).get("limit", ["20"])[0])
                    if not 1 <= limit <= 200:
                        raise ValueError()
                except ValueError:
                    raise APIError(400, "BAD_REQUEST", "limit must be an integer from 1 to 200")
                return self._send(200, {"actions": copy.deepcopy(list(actions)[:limit])})
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:3] == ["api", "v1", "actions"]:
                return self._send(200, copy.deepcopy(self.find_action(parts[3])))
            raise APIError(404, "NOT_FOUND", "endpoint not found")

    def find_action(self, aid):
        for action in actions:
            if action["id"] == aid:
                return action
        raise APIError(404, "NOT_FOUND", "action not found: " + aid)

    def api_POST(self, path):
        if path == "/api/v1/actions":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("body must contain 1 to 65536 bytes")
                body = json.loads(self.rfile.read(size))
            except (ValueError, UnicodeError) as e:
                raise APIError(400, "BAD_REQUEST", "invalid JSON body: " + str(e))
            return self._send(202, start_action(body))
        if path in ("/api/v1/stop", "/api/v1/estop"):
            return self._do_POST(path.rsplit("/", 1)[1])
        parts = path.strip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["api", "v1", "actions"] and parts[4] == "cancel":
            action = self.find_action(parts[3])
            if action["status"] != "running" or current_action[0] is not action:
                raise APIError(409, "NOT_RUNNING", "action is no longer running")
            return self._do_POST("stop")
        raise APIError(404, "NOT_FOUND", "endpoint not found")

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/api/v1" or path.startswith("/api/v1/"):
            try:
                return self.api_GET(unquote(path), query)
            except APIError as e:
                return self.api_error(e)
            except Exception as e:
                note(f"API request error: {e}")
                return self.api_error(APIError(500, "INTERNAL_ERROR", str(e)))
        if path == "/":
            self._send(200, PAGE, "text/html")
        elif path == "/robot-check":
            self._send(200, ROBOT_CHECK_PAGE, "text/html")
        elif path == "/grasp-flow":
            self._send(200, grasp_flow_page(), "text/html")
        elif path == "/api-docs":
            with open(os.path.join(os.path.dirname(__file__), "api_docs.html"), encoding="utf-8") as api_page:
                self._send(200, api_page.read(), "text/html")
        elif path in ("/models", "/models/"):
            with open(os.path.join(os.path.dirname(__file__), "web_models.html")) as model_page:
                self._send(200, model_page.read(), "text/html")
        elif path == "/api/models":
            services = model_services()
            with seq_lock:
                self._send(200, {"models": services, "selected": state["vla_selected"],
                                 "busy": bool(busy() or state["estop"])})
        elif path == "/trail.csv":
            data = trail_csv().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition",
                             f'attachment; filename="ta2_trail_{time.strftime("%Y%m%d_%H%M%S")}.csv"')
            self.end_headers()
            self.wfile.write(data)
        elif path == "/paths":
            self._send(200, {"paths": [{"name": n, **paths[n]} for n in sorted(paths)]})
        elif path == "/status":
            last_poll[0] = time.monotonic()
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            try:
                since = int(q.get("since", 0)) if int(q.get("epoch", -1)) == state["epoch"] else 0
            except ValueError:
                since = 0
            since = min(since, len(trail))
            with seq_lock:
                pause_until = seq["pause_until"]
                sequence = {"seq": seq["names"], "seq_i": seq["i"], "route": seq["route"],
                            "pause_left": max(0, pause_until - time.time()) if pause_until else 0,
                            "route_last": state["route_last"],
                            "vla_selection": public_model(state["vla_selected"]),
                            "vla_running": public_model(state["vla_model"]) if state["vla_model"] else None}
            self._send(200, {**state, "log": list(log), "trail_from": since, "trail": trail[since:],
                             **sequence, "mission": mission.snapshot(), "path_names": sorted(paths)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = unquote(urlsplit(self.path).path).rstrip("/")
        api = path == "/api/v1" or path.startswith("/api/v1/")
        try:
            # Both E-STOP endpoints bypass command locks and route gates.
            if path in ("/estop", "/api/v1/estop"):
                return self._do_POST("estop")
            if path == "/api/v1/grippers/record":   # read-only sampling: no command lock, allowed at any time
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(size)) if size else {}
                except ValueError as exc:
                    raise APIError(400, "BAD_REQUEST", "invalid JSON body: " + str(exc)) from exc
                return self._send(200, record_gripper(body))
            with seq_lock:
                if mission.active:
                    if api:
                        raise APIError(409, "MISSION_ESTOP_ONLY", "mission running: only E-STOP is accepted")
                    return self._send(409, {"error": "mission running: only E-STOP is accepted", "code": "MISSION_ESTOP_ONLY"})
                if seq["route"]:
                    if api:
                        raise APIError(409, "ROUTE_ESTOP_ONLY", "route running: only E-STOP is accepted")
                    return self._send(409, {"error": "route running: only E-STOP is accepted"})
                if api:
                    return self.api_POST(path)
                if current_action[0] and path not in ("/stop", "/release"):
                    return self._send(409, {"error": "busy: API action running"})
                return self._do_POST()
        except APIError as e:
            return self.api_error(e)
        except Exception as e:
            if not api:
                raise
            note(f"API request error: {e}")
            return self.api_error(APIError(500, "INTERNAL_ERROR", str(e)))

    def _do_POST(self, command=None):
        parts = (command or self.path.strip("/")).split("/")
        if parts == ["mission", "run"]:
            return self._send(202, start_action({"type": "mission.run"}))
        if len(parts) == 3 and parts[:2] == ["mission", "run"] and parts[2] in ("v2", "voice", "grasp", "handoff"):
            return self._send(202, start_action({"type": "mission.run", "params": {"flow": parts[2]}}))
        if parts[:2] == ["models", "select"]:
            if len(parts) != 3 or parts[2] not in VLA_MODELS:
                return self._send(400, {"error": "未知模型，请刷新模型列表"})
            if busy() or state["estop"]:
                return self._send(409, {"error": "机器人正在运行，请结束当前任务后切换模型"})
            try:
                select_model(parts[2])
            except OSError as exc:
                return self._send(500, {"error": f"保存模型选择失败：{exc}"})
            return self._send(200, {"ok": True, "selected": parts[2]})
        if parts[0] == "route":
            if len(parts) != 2 or parts[1] not in ROUTES:
                return self._send(400, {"error": "unknown route"})
            route = parts[1]
            names = ROUTES[route][1][:]
            missing = [n for n in names if n not in paths]
            if missing:
                return self._send(400, {"error": "missing paths: " + ", ".join(missing)})
            with seq_lock:
                if pending_motion[0] or state["moving"] or seq["names"] or state["arm"] or state["vla"] or state["estop"]:
                    return self._send(409, {"error": "busy: wait until the robot is still"})
                start_sequence(names, route)
            return self._send(200, {"ok": True})
        if parts[0] in ("step", "hold", "paths") and (state["arm"] or state["vla"]) and parts[:2] != ["paths", "upload"]:
            return self._send(409, {"error": "arms are moving"})
        if parts[0] == "vla":
            if len(parts) > 2 or (len(parts) == 2 and parts[1] not in VLA_MODELS):
                return self._send(400, {"error": "未知模型，请刷新模型列表"})
            model_id = parts[1] if len(parts) == 2 else state["vla_selected"]
            if model_id != state["vla_selected"]:
                return self._send(409, {"error": "选用模型已在其他页面更改，请刷新后再开始"})
            if busy() or state["estop"]:
                return self._send(409, {"error": "busy: wait until the robot is still"})
            try:
                check_policy(model_id=model_id)
            except APIError as e:
                return self._send(e.status, {"error": e.message})
            tag = start_vla(model_id=model_id)
            return self._send(200, {"ok": True, "tag": tag, "model": model_id})
        if parts[0] == "arm" and len(parts) == 2 and parts[1] in ARM_TIME:
            if pending_motion[0] or state["arm"] or state["moving"] or seq["names"] or state["vla"]:
                return self._send(409, {"error": "busy: wait until the robot is still"})
            start_arm(parts[1])
            return self._send(200, {"ok": True})
        if parts[0] in ("step", "hold") and len(parts) == 2:
            d = parts[1]
            if d not in DIRS:
                return self._send(400, {"error": "bad direction"})
            if pending_motion[0] and pending_motion[0] != ("hold", d):
                return self._send(409, {"error": "busy"})
            if parts[0] == "step":
                state["clicks"] += 1
            elif state["moving"] != f"hold {d}":
                state["holds"] += 1
            ok = send(parts[0], d)
            self._send(200 if ok else 503, {"ok": ok})
        elif parts[0] == "paths" and len(parts) >= 2:
            if parts[1] == "upload":
                n = int(self.headers.get("Content-Length", 0))
                if n > 2_000_000:
                    return self._send(413, {"error": "file too large"})
                try:
                    body = json.loads(self.rfile.read(n))
                    name = clean_name(body["name"])
                    paths[name] = parse_path_csv(body["csv"])
                    os.makedirs(PATH_DIR, exist_ok=True)
                    with open(os.path.join(PATH_DIR, name + ".csv"), "w") as f:
                        f.write(body["csv"])
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    return self._send(400, {"error": f"cannot read file: {e}"})
                note(f"loaded path {name}: {paths[name]['length']} m")
                return self._send(200, {"ok": True, "name": name})
            if pending_motion[0] or state["moving"] or seq["names"]:
                return self._send(409, {"error": "busy"})
            if parts[1] == "run" and len(parts) == 3 and parts[2] in paths:
                last_poll[0] = time.monotonic()
                ok = send("path_cmd", parts[2])
                return self._send(200 if ok else 503, {"ok": ok})
            if parts[1] == "runall" and paths:
                with seq_lock:
                    if pending_motion[0] or state["moving"] or seq["names"]:
                        return self._send(409, {"error": "busy"})
                    start_sequence(sorted(paths))
                return self._send(200, {"ok": True})
            self._send(400, {"error": "unknown path"})
        elif parts[0] == "reset":
            ok = send("reset_pose")
            note("trail reset, origin = current pose")
            self._send(200 if ok else 503, {"ok": ok})
        elif parts[0] == "rec" and len(parts) == 2 and parts[1] in ("start", "stop", "save"):
            if parts[1] == "start" and not state["recording"]:
                if not trail:
                    state["rec_t0"] = time.monotonic()
                state["recording"] = True
                add_point(*state["pose"], force=True)
                note("recording started")
            elif parts[1] == "stop" and state["recording"]:
                state["recording"] = False
                note(f"recording stopped ({len(trail)} points)")
            elif parts[1] == "save":
                if not trail:
                    return self._send(400, {"error": "trail is empty"})
                os.makedirs(TRAIL_DIR, exist_ok=True)
                path = os.path.join(TRAIL_DIR, time.strftime("trail_%Y%m%d_%H%M%S.csv"))
                with open(path, "w") as f:
                    f.write(trail_csv())
                note(f"saved {len(trail)} points to {path}")
                return self._send(200, {"ok": True, "file": path})
            self._send(200, {"ok": True})
        elif parts[0] == "estop":
            already_stopping = state["estop"]
            estop_generation[0] += 1
            state["estop"] = True
            seq["abort"] = True
            mission.abort()
            interrupt_action("ESTOP")
            note("E-STOP pressed")
            if not already_stopping:
                seq["abort"] = True               # plain assignment, no lock: E-STOP must not block
                threading.Thread(target=estop, daemon=True).start()
            self._send(200, {"ok": True})
        elif parts[0] in ("release", "stop"):
            if parts[0] == "stop":
                interrupt_action("STOPPED")
                with seq_lock:
                    seq["abort"] = True
                    ok = send("stop")
                if state["arm"]:
                    end_arm_runner()
                    subprocess.run(ARM_STOP, capture_output=True)
                    note("STOP: arm script interrupted")
                if state["vla"]:
                    stop_vla_motion()
                    note("STOP: VLA interrupted (arms hold where they are)")
            else:
                ok = send(parts[0])
            if parts[0] == "stop":
                note("STOP pressed")
            self._send(200 if ok else 503, {"ok": ok})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"TA2 chassis web remote on http://0.0.0.0:{PORT}", flush=True)
    load_path_dir()
    threading.Thread(target=action_log_loop, daemon=True).start()
    threading.Thread(target=teleop, daemon=True).start()   # connect at startup
    threading.Thread(target=alive_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
