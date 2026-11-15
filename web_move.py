#!/usr/bin/env python3
"""TA2 web remote (chassis, arms, VLA grasp). Runs on B70.

Page (Chinese, single column, phone friendly): top bar with connection / API mode / current activity
and the only red button, 急停 (E-STOP, also the Esc key); then VLA 抓取 (main), 底盘, and folded
路径与轨迹 and 日志 sections. The chassis centre 停 is the soft STOP described below.

Start:  nohup python3 /root/web_move.py > /root/web_move.log 2>&1 &
Open:   http://192.168.31.228:8088  (any browser on the robot's router network)

Click a direction (release within 0.3 s): move 10 cm, or turn 15 deg for CCW/CW.
Press and hold: keep moving at 0.05 m/s (turning at 0.2 rad/s) until release (dead-man: stops
0.4 s after the browser stops sending heartbeats; a single hold is capped at 10 s).

Trajectory: Start / Stop recording, Save (CSV download + copy in /root/ta2_trails/), Reset
(odometry origin = current pose, trail cleared). CSV columns: t_s, x_m (right), y_m (forward),
heading_deg (CCW), all relative to the origin.

Paths: CSV files of that format in /root/ta2_paths/ are listed on the page (Load adds more from
the browser). Run follows one path from the current pose; Run all follows them in name order,
each starting where the previous one ended. At every corner of a path (turn > 45 deg) the robot
stands still for CORNER_PAUSE seconds. A path stops if the page stops polling for 1 s.

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
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8088"))
DEPLOY = os.environ.get("DEPLOY", "/data/ta2/software/deploy")
ROBOT = "192.168.31.218"
CONTAINER = "openpi-xpu-client"
TELEOP = ["docker", "exec", "-i", CONTAINER, "/usr/local/bin/openpi-client-entrypoint.sh",
          "python3", "-u", "/outputs/tools/teleop.py"]
if os.environ.get("TELEOP_CMD"):              # for offline testing
    TELEOP = os.environ["TELEOP_CMD"].split()
DIRS = ("forward", "back", "left", "right", "ccw", "cw")
TRAIL_DIR = os.environ.get("TRAIL_DIR", "/root/ta2_trails")
PATH_DIR = os.environ.get("PATH_DIR", "/root/ta2_paths")
CORNER_PAUSE = float(os.environ.get("CORNER_PAUSE", "10"))   # seconds to stand still at each corner
ARM_CMD = os.environ.get("ARM_CMD", f"docker exec {CONTAINER} /usr/local/bin/openpi-client-entrypoint.sh "
                                    "python3 -u /outputs/arm/arm_pose.py").split()
ARM_STOP = os.environ.get("ARM_STOP", f"docker exec {CONTAINER} pkill -INT -f arm_pose.py").split()
# E-STOP: every process in the container whose command line matches this is stopped
ESTOP_PROCS = "guarded_grasp.py|arm_pose.py"
DEXEC = os.environ.get("ESTOP_DEXEC", f"docker exec {CONTAINER}").split()   # "echo" for offline tests
FSM_PAUSE = ["/usr/local/bin/openpi-client-entrypoint.sh", "timeout", "8", "ros2", "topic", "pub",
             "-r", "20", "-t", "40", "/api/fsm/enable", "std_msgs/msg/Float32", "{data: 0.0}"]
VLA_CMD = os.environ.get("VLA_CMD", "/root/vla_run.sh").split()
VLA_SECONDS, VLA_SPEED = "60", "0.5"
POLICY_CONTAINER = os.environ.get("POLICY_CONTAINER", "sandbag-step4999-policy")
ARM_TIME = {"raise": 8, "lower": 15, "start": 10}    # seconds; lower is slower on purpose
# start = VLA training start pose (arm/train_start_pose.json, first frame of a training episode)
paths = {}                  # name -> {"points": [[x, y, th], ...], "length": m, "end": [dx, dy]}
seq = {"names": [], "i": -1, "abort": False}
path_done = threading.Event()
path_result = {"why": None}
last_poll = [0.0]

state = {"connected": False, "moving": None, "mode_ok": None, "last": "starting", "clicks": 0, "holds": 0,
         "pose": [0.0, 0.0, 0.0], "epoch": int(time.time()),    # epoch changes on restart and on trail reset
         "recording": False, "rec_t0": None, "active_path": None, "path_anchor": [0.0, 0.0, 0.0],
         "arm": None, "arm_last": "arms: not moved yet", "estop": False, "estop_last": None,
         "vla": None, "vla_last": "VLA: not run yet"}
trail = []                  # [[x, y, th, t], ...]; metres / rad / s; x = right, y = forward of the origin
TRAIL_MAX = 20000
log = collections.deque(maxlen=60)
lock = threading.Lock()
proc = {"p": None}


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


def run_sequence(names):
    seq.update(names=names, i=-1, abort=False)
    note("run all: " + " -> ".join(names))
    for i, n in enumerate(names):
        if seq["abort"]:
            break
        seq["i"] = i
        path_done.clear()
        path_result["why"] = None
        send("path_cmd", n)
        if not path_done.wait(timeout=3 * paths[n]["length"] / 0.05 + 30 + 10 * CORNER_PAUSE):
            send("stop")
            note(f"run all: {n} did not finish, stopped")
            break
        if path_result["why"] != "path done":
            note(f"run all: stopped at {n} ({path_result['why']})")
            break
        time.sleep(1.0)
    else:
        note("run all: finished")
    seq.update(names=[], i=-1)


def run_arm(action):
    try:
        ensure_container()
        p = subprocess.Popen(ARM_CMD + [action, "--time", str(ARM_TIME[action])], stdout=subprocess.PIPE,
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
        state["arm"] = None


def run_vla():
    tag = state["vla"]
    result = None
    try:
        ensure_container()
        p = subprocess.Popen(VLA_CMD + [tag, VLA_SECONDS, VLA_SPEED], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
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
            state["vla_last"] = f"VLA {tag}: {result['actuations']} steps, {why}"
        else:
            state["vla_last"] = f"VLA {tag}: exited rc={rc} before running the policy (see log)"
    except Exception as e:  # noqa: BLE001
        state["vla_last"] = f"VLA {tag}: error {e}"
    finally:
        state["vla"] = None
        note(state["vla_last"])


def estop():
    """Software emergency stop: chassis and paths, arm and VLA scripts, then FSM pause."""
    try:
        seq["abort"] = True
        send("stop")
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
        if state["active_path"] and time.monotonic() - last_poll[0] < 1.0:
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
    state["connected"] = False
    state["moving"] = None


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
    p = teleop()
    if cmd == "path_cmd":
        msg = {"cmd": "path", "name": d, "points": paths[d]["points"], "dwell": CORNER_PAUSE}
    else:
        msg = {"cmd": cmd} if d is None else {"cmd": cmd, "dir": d}
    try:
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        return True
    except (BrokenPipeError, OSError):
        return False


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
</style></head><body>
<header><div class="bar">
  <div class="identity"><h1>TA2 遥控</h1><span id="connection" class="connection" role="status"><span class="dot" aria-hidden="true"></span><span id="connection-text">连接中…</span></span><span id="mode" class="badge">模式未知</span></div>
  <button id="estop" class="estop" title="急停底盘、手臂与抓取任务（Esc）">急停</button>
  <div class="activity"><span>当前活动</span><span id="status" role="status">正在连接机器人…</span></div>
</div></header>
<main class="app">
  <section class="card"><div class="body">
    <h2>VLA 抓取</h2>
    <button id="vlarun" class="primary" title="复位、取帧后执行沙包抓取" disabled>开始抓取（60 秒）</button>
    <div id="vla-progress" hidden>
      <div class="progress-row"><span id="vlaelapsed" role="status">已运行 0 / 60 s</span><button id="vlastop" class="small">停止</button></div>
      <progress id="vlaprogress" max="60" value="0" aria-label="抓取运行进度" aria-describedby="vlaelapsed"></progress>
    </div>
    <p class="status-line">上次：<span id="vlast">尚未运行</span></p>
    <div class="arm-section">
      <div class="arm-controls"><h3>手臂</h3><button id="armstart" class="small" title="回到训练起始姿态，夹爪张开，约 10 秒" disabled>起始姿态</button><button id="armraise" class="small" title="抬起手臂，约 8 秒" disabled>抬起</button><button id="armlower" class="small" title="返回抬起前的姿态，约 15 秒" disabled>放下</button></div>
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
  <details id="paths-details" class="card"><summary>路径与轨迹</summary><div class="body">
    <div class="subhead"><h3>路径</h3><div class="tools"><button class="small" id="load" title="加载 CSV 路径文件">加载</button><button class="small" id="runall" title="按名称顺序运行全部路径，每条从上一条终点开始">全部运行</button></div></div>
    <input type="file" id="file" accept=".csv,text/csv" multiple hidden>
    <div class="plist" id="plist" title="点击路径行预览；路径从当前位置开始，转角停留 10 秒"><div class="prow muted">尚未加载路径</div></div>
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
const armActions={start:'起始姿态',raise:'抬起',lower:'放下'};
function uiText(value){
  let t=String(value??'');
  const phrases=[['arms: not moved yet','尚未移动'],['VLA: not run yet','尚未运行'],
    ['busy: wait until the robot is still','忙碌中，请等待机器人静止'],['arms are moving','手臂或抓取任务正在运行'],
    ['not in API mode or chassis disabled (or mode unknown yet)','非 API 模式、底盘未启用或模式未知'],
    ['no fresh wheel feedback from the robot','尚未收到机器人最新轮速反馈'],['busy, path not started','忙碌中，路径未启动'],
    ['exited rc=','退出码：'],['before running the policy (see log)','，策略尚未运行（详见日志）'],
    ['FSM pause failed rc=','暂停失败，退出码：'],['FSM paused','已暂停'],['E-STOP','急停'],
    ['time_limit','达到时限'],['path done','路径已完成'],['corner pause','转角停留'],['heartbeat timeout','心跳超时'],
    ['dead-man timeout','心跳超时'],['step done','点按移动已完成'],['hold timeout','长按已达时限'],
    ['page closed or connection lost','页面已关闭或连接中断'],['wheel feedback lost','轮速反馈中断'],
    ['hold limit','长按达到时限'],['stuck: no progress for','移动受阻，未取得进展已持续'],
    ['trail is empty','轨迹为空'],['starting','正在启动'],['ready','就绪'],['stopped','已停止'],
    ['released','已松开'],['release','松开'],['timeout','超时'],['busy','忙碌中'],['done','已完成'],['error','错误']];
  for(const [from,to] of phrases)t=t.split(from).join(to);
  t=t.replace(/\barms (start|raise|lower):/g,(_,a)=>'手臂'+armActions[a]+'：')
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
const vlaBtn=document.getElementById('vlarun'),vlaSt=document.getElementById('vlast');
vlaBtn.onclick=async()=>{const r=await post('/vla');if(r.error)st.textContent=uiText(r.error)};
const vlaProgress=document.getElementById('vla-progress'),vlaMeter=document.getElementById('vlaprogress'),vlaElapsed=document.getElementById('vlaelapsed');
let vlaStarted=null;
function updateVlaProgress(){
  if(vlaStarted===null)return;
  const elapsed=Math.floor((performance.now()-vlaStarted)/1000);
  vlaMeter.value=Math.min(60,elapsed);
  vlaElapsed.textContent=elapsed>=60?'收尾中…':'已运行 '+elapsed+' / 60 s';
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
  vlaSt.textContent=uiText(s.vla_last);
  if(s.vla){if(vlaStarted===null)vlaStarted=performance.now()}else vlaStarted=null;
  vlaProgress.hidden=!s.vla;updateVlaProgress();
  armRaise.disabled=!still;armLower.disabled=!still;armStart.disabled=!still;vlaBtn.disabled=!still||!!s.estop;
  st.textContent=t;st.title=t;
  const busyStep=s.moving&&(s.moving.startsWith('step')||s.moving.startsWith('path'))||s.seq.length>0||!!s.arm;
  const pathBusy=!!s.active_path||s.seq.length>0||!!s.arm;
  document.querySelectorAll('.prow button').forEach(b=>b.disabled=pathBusy);
  document.getElementById('runall').disabled=pathBusy;
  moves.forEach(x=>x.disabled=!press&&(busyStep||!s.connected));
  lg.textContent=s.log.join('\n');lg.scrollTop=lg.scrollHeight;draw();
}catch(e){connectionText.textContent='B70 不可达';connectionEl.className='connection';st.textContent='无法获取机器人状态';st.title=st.textContent;if(!press)moves.forEach(x=>x.disabled=true)}}
poll();setInterval(poll,300);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/":
            self._send(200, PAGE, "text/html")
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
            self._send(200, {**state, "log": list(log), "trail_from": since, "trail": trail[since:],
                             "seq": seq["names"], "seq_i": seq["i"], "path_names": sorted(paths)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if parts[0] in ("step", "hold", "paths") and (state["arm"] or state["vla"]) and parts[:2] != ["paths", "upload"]:
            return self._send(409, {"error": "arms are moving"})
        if parts[0] == "vla":
            if state["arm"] or state["moving"] or seq["names"] or state["vla"] or state["estop"]:
                return self._send(409, {"error": "busy: wait until the robot is still"})
            up = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", POLICY_CONTAINER],
                                capture_output=True, text=True).stdout.strip()
            if up != "true" and not os.environ.get("VLA_CMD"):
                return self._send(503, {"error": f"policy server {POLICY_CONTAINER} is not running"})
            state["vla"] = time.strftime("vla_web_%Y%m%d_%H%M%S")
            note(f"VLA grasp started: {state['vla']} ({VLA_SECONDS} s, max {VLA_SPEED} rad/s)")
            threading.Thread(target=run_vla, daemon=True).start()
            return self._send(200, {"ok": True, "tag": state["vla"]})
        if parts[0] == "arm" and len(parts) == 2 and parts[1] in ARM_TIME:
            if state["arm"] or state["moving"] or seq["names"] or state["vla"]:
                return self._send(409, {"error": "busy: wait until the robot is still"})
            state["arm"] = parts[1]
            note(f"arms: {parts[1]} ({ARM_TIME[parts[1]]} s)")
            threading.Thread(target=run_arm, args=(parts[1],), daemon=True).start()
            return self._send(200, {"ok": True})
        if parts[0] in ("step", "hold") and len(parts) == 2:
            d = parts[1]
            if d not in DIRS:
                return self._send(400, {"error": "bad direction"})
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
            if state["moving"] or seq["names"]:
                return self._send(409, {"error": "busy"})
            if parts[1] == "run" and len(parts) == 3 and parts[2] in paths:
                last_poll[0] = time.monotonic()
                ok = send("path_cmd", parts[2])
                return self._send(200 if ok else 503, {"ok": ok})
            if parts[1] == "runall" and paths:
                last_poll[0] = time.monotonic()
                threading.Thread(target=run_sequence, args=(sorted(paths),), daemon=True).start()
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
            note("E-STOP pressed")
            if not state["estop"]:
                state["estop"] = True
                threading.Thread(target=estop, daemon=True).start()
            self._send(200, {"ok": True})
        elif parts[0] in ("release", "stop"):
            if parts[0] == "stop":
                seq["abort"] = True
                if state["arm"]:
                    subprocess.run(ARM_STOP, capture_output=True)
                    note("STOP: arm script interrupted")
                if state["vla"]:
                    subprocess.run(DEXEC + ["pkill", "-INT", "-f", ESTOP_PROCS], capture_output=True)
                    note("STOP: VLA interrupted (arms hold where they are)")
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
    threading.Thread(target=teleop, daemon=True).start()   # connect at startup
    threading.Thread(target=alive_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
