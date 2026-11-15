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
max 0.5 rad/s, prompt 抓取沙包). Needs the policy container running. STOP or E-STOP interrupts it.
E-STOP (red header button, or the Esc key): stops the chassis and any path, interrupts every arm or
VLA script in the container (arm_pose.py, guarded_grasp.py; force-killed after 3 s), then publishes
/api/fsm/enable 0.0 for 2 s so the robot PAUSEs (zero-velocity hold). It is a software stop: keep
the hardware emergency stop within reach.

Motion is done by teleop.py, a long-running ROS node inside the openpi-xpu-client
container, fed over stdin. If this server exits, teleop.py sees EOF, sends zero and exits.
"""
import collections
import copy
import datetime
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
ESTOP_PROCS = "guarded_grasp.py|arm_pose.py|grasp_monitor.py"
DEXEC = shlex.split(os.environ.get("ESTOP_DEXEC", f"docker exec {CONTAINER}"))   # "echo" for offline tests
FSM_PAUSE = ["/usr/local/bin/openpi-client-entrypoint.sh", "timeout", "8", "ros2", "topic", "pub",
             "-r", "20", "-t", "40", "/api/fsm/enable", "std_msgs/msg/Float32", "{data: 0.0}"]
VLA_CMD = shlex.split(os.environ.get("VLA_CMD", "/root/vla_run.sh"))
VLA_SECONDS, VLA_SPEED = "60", "0.5"
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
ARM_TIME = {"raise": 8, "lower": 15, "start": 10, "start_hold": 10, "down5": 8, "open_right": 3, "close_right": 3, "open_left": 3, "close_left": 3}
ARM_DOWN_CMD = shlex.split(os.environ.get("ARM_DOWN_CMD", "/root/arm_down.sh"))   # base command; callers append side and cm
ARM_UP_CMD = shlex.split(os.environ.get("ARM_UP_CMD", "/root/arm_up.sh"))         # base command; callers append side and cm
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
vla_proc = {"p": None}
ACTION_LOG = os.environ.get("ACTION_LOG", "/root/ta2_actions.jsonl")
actions = collections.deque(maxlen=200)   # API records only, newest first
current_action = [None]                  # worker reservation, retained until motion really ends
pending_motion = [None]                  # includes web commands awaiting teleop acknowledgement
terminal_claims = {}                     # atomic pop chooses exactly one terminal status
log_claims = {}                          # append once, after the worker supplies its result
action_logs = queue.SimpleQueue()


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
                code = "SCRIPT_ERROR" if action["type"] in ("arm.pose", "vla.grasp") else "ACTION_FAILED"
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
            result.update(status="aborted", actuations=None, error=why, tag=tag,
                          report=f"/root/openpi-xpu/client/outputs/{tag}/report.json")
        end_action(action, result, code=code, release=False)


def start_sequence(names, route=None, action=None):
    if state["estop"] or (action and action["status"] != "running"):
        end_action(action, {"why": "emergency stop"}, code="ESTOP")
        return
    last_poll[0] = time.monotonic()
    seq.update(names=names, i=-1, abort=False, route=route, pause_until=None)
    threading.Thread(target=run_sequence, args=(names, action), daemon=True).start()


def start_arm(pose, action=None):
    state["arm"] = pose
    note(f"arms: {pose} ({ARM_TIME[pose]} s)")
    threading.Thread(target=run_arm, args=(pose, action), daemon=True).start()


def public_model(model_id):
    model = VLA_MODELS[model_id]
    return {key: model[key] for key in ("id", "name", "training", "prompt", "seconds", "speed")}


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


def start_vla(action=None, model_id=None):
    model_id = model_id or state["vla_selected"]
    model = VLA_MODELS[model_id]
    tag = time.strftime(f"vla_{model_id}_%Y%m%d_%H%M%S_") + str(time.time_ns() % 1000000)
    state["vla"], state["vla_model"] = tag, model_id
    note(f"VLA grasp started: {model['name']} / {tag} ({model['seconds']} s, max {model['speed']} rad/s)")
    args = (tag, model_id) if action is None else (tag, model_id, action)
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


def run_arm(action, record=None):
    rc, last = None, ""
    try:
        ensure_container()
        if state["estop"] or (record and record["status"] != "running"):
            return
        cmd = ARM_DOWN_CMD + ["right", "5"] if action == "down5" else ARM_CMD + [action, "--time", str(ARM_TIME[action])]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        last = ""
        for line in p.stdout:
            line = line.rstrip()
            if line and "multicast" not in line and not line.startswith("Current mode"):
                note("arm: " + line)
                last = line
        rc = p.wait()
        state["arm_last"] = f"arms {action}: " + ("done" if rc == 0 else f"stopped ({last})")
    except Exception as e:  # noqa: BLE001
        state["arm_last"] = f"arms {action}: error {e}"
        note(state["arm_last"])
    finally:
        with seq_lock:
            state["arm"] = None
            end_action(record, {"why": state["arm_last"], "returncode": rc, "output": last}, rc == 0)


def _signal_vla_host(sig):
    vp = vla_proc["p"]
    if vp is not None and isinstance(getattr(vp, "pid", None), int) and vp.poll() is None:
        try:
            os.killpg(vp.pid, sig)
        except ProcessLookupError:
            pass


def run_vla(tag, model_id, record=None):
    model = VLA_MODELS[model_id]
    label = model["name"]
    result, rc = None, None
    try:
        ensure_container()
        if state["estop"] or (record and record["status"] != "running"):
            return
        command = shlex.split(os.environ["VLA_CMD"]) if os.environ.get("VLA_CMD") else model["command"]
        p = subprocess.Popen(command + [tag, str(model["seconds"]), str(model["speed"])], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
        vla_proc["p"] = p
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
            if result.get("status") == "succeeded" and result.get("hand"):
                why = f"grasped ({result['hand']}) + lifted + stowed"
            state["vla_last"] = f"{label} / {tag}: {result['actuations']} steps, {why}"
        else:
            state["vla_last"] = f"{label} / {tag}: exited rc={rc} before running the policy (see log)"
    except Exception as e:  # noqa: BLE001
        state["vla_last"] = f"{label} / {tag}: error {e}"
    finally:
        vla_proc["p"] = None
        with seq_lock:
            state["vla"] = None
            state["vla_model"] = None
            note(state["vla_last"])
            detail = dict(result) if isinstance(result, dict) else {}
            detail.setdefault("status", "script_error")
            detail.setdefault("actuations", None)
            detail.setdefault("error", None if rc == 0 and result else state["vla_last"])
            detail.setdefault("report", f"/root/openpi-xpu/client/outputs/{tag}/report.json")
            detail.update(tag=tag, model=model_id, returncode=rc)
            ok = rc == 0 and not detail["error"] and detail["status"] in ("time_limit", "done", "finished", "succeeded")
            end_action(record, detail, ok)


def estop():
    """Software emergency stop: chassis and paths, arm and VLA scripts, then FSM pause."""
    try:
        seq["abort"] = True
        send("stop")
        _signal_vla_host(mission_module.signal.SIGINT)
        subprocess.run(DEXEC + ["pkill", "-INT", "-f", ESTOP_PROCS], capture_output=True)
        for _ in range(30):            # up to 3 s for a clean exit (scripts hold their pose)
            vp = vla_proc["p"]
            host_alive = vp is not None and isinstance(getattr(vp, "pid", None), int) and vp.poll() is None
            if not host_alive and subprocess.run(DEXEC + ["pgrep", "-f", ESTOP_PROCS], capture_output=True).returncode != 0:
                break
            time.sleep(0.1)
        else:
            _signal_vla_host(mission_module.signal.SIGKILL)
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
    "vla.grasp": {},
    "odom.reset": {},
    "mission.run": {"attempts": "optional integer", "prompt": "optional string", "port": "optional integer"},
}


def start_action(body):
    if not isinstance(body, dict) or set(body) - {"type", "params"}:
        raise APIError(400, "BAD_REQUEST", "expected {type, params}")
    kind, params = body.get("type"), body.get("params", {})
    if not isinstance(kind, str) or kind not in ACTION_PARAMS or not isinstance(params, dict):
        raise APIError(400, "BAD_REQUEST", "unknown type or invalid params")
    spec = ACTION_PARAMS[kind]
    if kind == "mission.run":
        merged = dict(params)
        sel = VLA_MODELS[state["vla_selected"]]
        if "prompt" not in merged and "PROMPT" not in os.environ:
            merged["prompt"] = sel["prompt"]
        if "port" not in merged and "MISSION_PORT" not in os.environ and "POLICY_PORT" not in os.environ:
            merged["port"] = sel["port"]
        try:
            cfg = mission_module.options(merged)
        except ValueError as exc:
            raise APIError(400, "BAD_REQUEST", str(exc)) from exc
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
        check_policy()
    # E-STOP bypasses seq_lock and may arrive during the policy check.
    if state["estop"] or (kind == "mission.run" and mission.cancel.is_set()):
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
        elif kind == "vla.grasp":
            start_vla(action)
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
</style></head><body>
<header><div class="bar">
  <div class="identity"><h1>TA2 遥控</h1><span id="connection" class="connection" role="status"><span class="dot" aria-hidden="true"></span><span id="connection-text">连接中…</span></span><span id="mode" class="badge">模式未知</span></div>
  <button id="estop" class="estop" title="急停底盘、手臂与抓取任务（Esc）">急停</button>
  <div class="activity"><span>当前活动</span><span id="status" role="status">正在连接机器人…</span></div>
</div></header>
<main class="app">
  <section class="card"><div class="body">
    <h2>全流程</h2>
    <p id="missionmodel" class="status-line">当前模型：读取中…</p>
    <button id="missionrun" class="primary" disabled>开始全流程</button>
    <ol id="missionsteps"><li data-step="preflight">预检</li><li data-step="g2r">绿→红</li><li data-step="grasp">抓取 0/5</li><li data-step="stow">收臂</li><li data-step="r2g">红→绿</li><li data-step="place">放下</li></ol>
    <p id="missionstatus" class="status-line" role="status">尚未运行</p>
  </div></section>
  <section class="card"><div class="body">
    <h2>VLA 抓取</h2>
    <p class="status-line">当前模型：<strong id="vlamodel">读取中…</strong></p>
    <p><button id="modelsbtn" type="button" class="small" onclick="location.href='/models'">选择 / 切换模型 →</button></p>
    <p id="vlaconfig" class="status-line">读取模型配置中…</p>
    <button id="vlarun" class="primary" title="回到起始姿态并张开夹爪，取帧后使用当前选定模型抓取" disabled>开始抓取（60 秒）</button>
    <div id="vla-progress" hidden>
      <div class="progress-row"><span id="vlaelapsed" role="status">已运行 0 / 60 s</span><button id="vlastop" class="small">停止</button></div>
      <progress id="vlaprogress" max="60" value="0" aria-label="抓取运行进度" aria-describedby="vlaelapsed"></progress>
    </div>
    <p class="status-line">上次：<span id="vlast">尚未运行</span></p>
    <div class="arm-section">
      <div class="arm-controls"><h3>手臂</h3><button id="armstart" class="small" title="回到训练起始姿态，夹爪张开，约 10 秒" disabled>起始姿态</button><button id="armstarthold" class="small" title="回到训练起始姿态，正在夹着的夹爪保持夹紧（不松开），约 10 秒" disabled>起始姿态·保持夹爪</button><button id="armdown5" class="small" title="右臂末端竖直下降 5 cm，朝向不变，夹着的夹爪保持夹紧；每次按当前位置重新计算，最多降到末端 z = −0.40 m（不足 5 cm 时只降到下限），约 8 秒" disabled>右臂下降 5 cm</button><button id="armopenright" class="small" title="张开右夹爪（全力张开），手臂保持当前姿态不动；夹着的东西会掉下来，约 3 秒" disabled>松开右爪</button><button id="armcloseright" class="small" title="闭合右夹爪（全力夹紧），手臂保持当前姿态不动，约 3 秒；之后保持夹紧" disabled>闭合右爪</button><button id="armraise" class="small" title="抬起手臂，约 8 秒" disabled>抬起</button><button id="armlower" class="small" title="返回抬起前的姿态，约 15 秒" disabled>放下</button></div>
      <p id="armst" class="status-line">尚未移动</p>
    </div>
  </div></section>
  <section class="card"><div class="body">
    <h2>路线</h2>
    <div class="route-buttons">
      <button data-route="g2r" disabled><span class="route-title"><span class="route-dot green" aria-hidden="true"></span>绿 → 红<span class="route-dot red" aria-hidden="true"></span></span><small>001 → 停 10 秒 → 002</small></button>
      <button data-route="r2g" disabled><span class="route-title"><span class="route-dot red" aria-hidden="true"></span>红 → 绿<span class="route-dot green" aria-hidden="true"></span></span><small>003 → 停 10 秒 → 004</small></button>
    </div>
    <p id="routest" class="status-line" role="status">上次：尚未运行</p>
    <p class="hint">路径从当前位置出发：先把机器人停在起点、朝向与录制时一致</p>
    <p class="hint">开始后仅接受急停；关闭页面或手机断网仍会继续执行</p>
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
document.getElementById('stop').onclick=document.getElementById('vlastop').onclick=()=>{end();post('/stop')};
const estopBtn=document.getElementById('estop');
const doEstop=()=>{if(estopBtn.disabled)return;end();post('/estop')};estopBtn.onclick=doEstop;
document.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();doEstop()}});
const modeEl=document.getElementById('mode'),connectionEl=document.getElementById('connection'),connectionText=document.getElementById('connection-text');
const directions={forward:'前进',back:'后退',left:'向左',right:'向右',ccw:'逆时针',cw:'顺时针'};
const armActions={start:'起始姿态',start_hold:'起始姿态（保持夹爪）',down5:'右臂下降 5 cm',open_right:'松开右爪',close_right:'闭合右爪',open_left:'松开左爪',close_left:'闭合左爪',raise:'抬起',lower:'放下'};
function uiText(value){
  let t=String(value??'');
  const phrases=[['route running: only E-STOP is accepted','路线执行中，仅接受急停'],
    ['arms: not moved yet','尚未移动'],['VLA: not run yet','尚未运行'],
    ['busy: wait until the robot is still','忙碌中，请等待机器人静止'],['arms are moving','手臂或抓取任务正在运行'],
    ['not in API mode or chassis disabled (or mode unknown yet)','非 API 模式、底盘未启用或模式未知'],
    ['no fresh wheel feedback from the robot','尚未收到机器人最新轮速反馈'],['busy, path not started','忙碌中，路径未启动'],
    ['exited rc=','退出码：'],['before running the policy (see log)','，策略尚未运行（详见日志）'],
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
  t=t.replace(/\barms (start_hold|start|down5|open_right|close_right|open_left|close_left|raise|lower):/g,(_,a)=>'手臂'+armActions[a]+'：')
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
const armDown5=document.getElementById('armdown5');armDown5.onclick=async()=>{const r=await post('/arm/down5');if(r.error)st.textContent=r.error};
const armStartHold=document.getElementById('armstarthold');armStartHold.onclick=async()=>{const r=await post('/arm/start_hold');if(r.error)st.textContent=r.error};
const vlaBtn=document.getElementById('vlarun'),vlaSt=document.getElementById('vlast');
const routeBtns=[...document.querySelectorAll('button[data-route]')],routeSt=document.getElementById('routest');
const routeLabels={g2r:'绿 → 红',r2g:'红 → 绿'};
routeBtns.forEach(b=>b.onclick=async()=>{const r=await post('/route/'+b.dataset.route);if(r.error)routeSt.textContent=uiText(r.error)});
const missionBtn=document.getElementById('missionrun'),missionSt=document.getElementById('missionstatus');
missionBtn.onclick=async()=>{missionBtn.disabled=true;try{const r=await post('/mission/run');if(r.error)missionSt.textContent=uiText(typeof r.error==='string'?r.error:r.error.message)}catch(e){missionSt.textContent='无法开始任务'}};
function updateMission(s){
  const m=s.mission||{},active=!!m.active;
  if(s.vla_selection)document.getElementById('missionmodel').textContent='当前模型：'+s.vla_selection.name+'（'+s.vla_selection.prompt+'）';
  missionBtn.disabled=active||!s.connected||s.mode_ok!==true||!!s.estop||!!s.moving||s.seq.length>0||!!s.arm||!!s.vla;
  document.querySelectorAll('#missionsteps li').forEach(el=>{
    el.classList.toggle('mission-current',el.dataset.step===m.step);
    if(el.dataset.step==='grasp')el.textContent='抓取 '+(m.attempt||0)+'/'+(m.attempts||5);
  });
  const results={idle:'尚未运行',running:'运行中',succeeded:'成功',failed:'失败',aborted:'已急停中止'};
  missionSt.textContent=(results[m.status]||'尚未运行')+(m.hand?' · 抓取手：'+(m.hand==='left'?'左手':'右手'):'')+(m.pause_left_s>0?' · 停留 '+Math.ceil(m.pause_left_s)+' 秒':'')+(m.result?' · '+m.result:'')+(active?' · 仅接受急停':'');
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
vlaBtn.onclick=async()=>{if(!selectedVlaId)return;const r=await post('/vla/'+encodeURIComponent(selectedVlaId));if(r.error)st.textContent=uiText(r.error)};
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
    document.getElementById('vlaconfig').textContent=model.prompt+' · '+model.seconds+' 秒 · 速度上限 '+model.speed+' rad/s';
    vlaBtn.textContent='开始抓取（'+s.vla_selection.seconds+' 秒）';
    document.getElementById('modelsbtn').disabled=!still||!!s.estop;
  }
  vlaSt.textContent=uiText(s.vla_last);
  if(s.vla){if(vlaStarted===null)vlaStarted=performance.now()}else vlaStarted=null;
  vlaProgress.hidden=!s.vla;updateVlaProgress();
  updateRoute(s);
  armRaise.disabled=!still;armLower.disabled=!still;armStart.disabled=!still;armStartHold.disabled=!still;armDown5.disabled=!still;armOpenRight.disabled=!still;armCloseRight.disabled=!still;vlaBtn.disabled=!still||!!s.estop||!s.connected;
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
</script></body></html>"""


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
        if path == "/":
            self._send(200, PAGE, "text/html")
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
                    subprocess.run(ARM_STOP, capture_output=True)
                    note("STOP: arm script interrupted")
                if state["vla"]:
                    _signal_vla_host(mission_module.signal.SIGINT)
                    subprocess.run(DEXEC + ["pkill", "-INT", "-f", ESTOP_PROCS], capture_output=True)
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
