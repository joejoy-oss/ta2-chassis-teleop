#!/usr/bin/env python3
"""TA2 mission orchestration. All motion is delegated to existing web/arm/VLA tools."""
import copy
import json
import math
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ATTEMPTS = int(os.environ.get("ATTEMPTS", "5"))
ATTEMPT_S = float(os.environ.get("ATTEMPT_S", "60"))   # s of policy execution per attempt (reset + probe come before it)
PROMPT = os.environ.get("PROMPT", "抓取沙包")
PORT = int(os.environ.get("MISSION_PORT", os.environ.get("POLICY_PORT", "8000")))
RTC = os.environ.get("MISSION_RTC", "0") == "1"   # async Real-Time Chunking for the grasp attempts (vla_run.sh RTC=1)
ARRIVE_PAUSE = float(os.environ.get("ARRIVE_PAUSE", "3"))   # s standing still at the route end before grasp / place
PLACE_CM = float(os.environ.get("PLACE_CM", "10"))
PLACE_FORWARD_CM = float(os.environ.get("PLACE_FORWARD_CM", "10"))   # step 6: reach out before lowering; 0 skips
# "Holding something" = gripper pushing closed (effort) but stopped clearly above its empty-closed position.
# Empty-closed is very repeatable (right: -0.0485 +/- 0.0004 rad over 5 trials, 2026-11-15 17:17; left: -0.059,
# one earlier reading -- recalibrate with the page's 夹爪标定记录). Solid and thin objects both count as held.
EMPTY_CLOSED = {"right": float(os.environ.get("HOLD_EMPTY_RIGHT", "-0.0485")),
                "left": float(os.environ.get("HOLD_EMPTY_LEFT", "-0.059"))}
HOLD_MARGIN = float(os.environ.get("HOLD_MARGIN", "0.01"))     # rad above empty-closed (25x its noise)
EMPTY_PERSIST = float(os.environ.get("EMPTY_PERSIST", "2.0"))   # s closed on nothing without reopening -> attempt ended
# (2 s: a policy that reopens to try again restarts the timer; real runs lift the empty gripper instead, 2026-11-16)
NO_LIFT_S = float(os.environ.get("NO_LIFT_S", "10"))            # s holding without the lift -> attempt failed
THIN_SOLID_SPLIT = float(os.environ.get("THIN_SOLID_SPLIT", "0.25"))  # rad; right: thin <= 0.13, solid >= 0.37
HOLD_POS_MAX = float(os.environ.get("HOLD_POS_MAX", "0.75"))
HOLD_EFF_MAX = float(os.environ.get("HOLD_EFF_MAX", "-1.0"))
LIFT_MIN = float(os.environ.get("LIFT_MIN", "0.04"))
HOLD_PERSIST = float(os.environ.get("HOLD_PERSIST", "1.5"))
DROP_PERSIST = float(os.environ.get("DROP_PERSIST", "0.25"))
MISSION_LOG_DIR = os.environ.get("MISSION_LOG_DIR", "/root/ta2_mission")
MONITOR_TIMEOUT = float(os.environ.get("MONITOR_TIMEOUT", "0.6"))
SCRIPT_TIMEOUT = float(os.environ.get("SCRIPT_TIMEOUT", "90"))
SIDES = ("left", "right")
# Full flow v2 = steps 1-6 + step 7 "voice interaction module" (flow "voice" runs only step 7, for debugging).
FLOWS = ("v1", "v2", "voice", "grasp", "handoff")   # grasp = step 3 alone (stops holding); handoff = 7.4-7.7 alone
VOICE_BRIDGE = os.environ.get("VOICE_BRIDGE", "http://127.0.0.1:18096")   # Mac voice service via ssh -R
VOICE_BRIDGE_HOST = os.environ.get("VOICE_BRIDGE_HOST", "127.0.0.1:8096")
VOICE_LISTEN_S = float(os.environ.get("VOICE_LISTEN_S", "120"))
VOICE_ANNOUNCE_S = float(os.environ.get("VOICE_ANNOUNCE_S", "60"))
VOICE_RELEASE_S = float(os.environ.get("VOICE_RELEASE_S", "5"))
HANDOFF_CM = float(os.environ.get("HANDOFF_CM", "10"))
HANDOFF_FORWARD_CM = float(os.environ.get("HANDOFF_FORWARD_CM", "10"))   # reach out before lowering; 0 skips
OPEN_MIN = float(os.environ.get("OPEN_MIN", "0.9"))   # rad; open is about 1.05, empty-closed about -0.05
VOICE_STEPS = (
    ("7.1", "识别口令", "Mac 语音端 /rehearse → /status"),
    ("7.2", "真机抓取（抓住即停）", "VLA + 抓取成功判定（同第 3 步）"),
    ("7.3", "收臂（保持夹紧）", "arm start_hold（同第 4 步）"),
    ("7.4", f"前伸 {HANDOFF_FORWARD_CM:g} cm", "arm_move.sh（同手臂「前送」）"),
    ("7.5", f"下降 {HANDOFF_CM:g} cm", "arm_down.sh（同第 6 步）"),
    ("7.6", "松开夹爪", "arm open_<手>（同第 6 步）"),
    ("7.7", "确认交接", "夹爪读数：张开且未夹持"),
    ("7.8", "完成播报", "Mac 语音端 /confirm"),
)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def hold_threshold(side=None):
    """Lowest gripper position that counts as holding; without a side, the stricter of the two."""
    base = EMPTY_CLOSED[side] if side in EMPTY_CLOSED else max(EMPTY_CLOSED.values())
    return base + HOLD_MARGIN


def holding(sample, side=None):
    """Something (solid or thin) is between the fingers: the gripper pushes closed (effort) but stopped clearly
    above its empty-closed position. Pure predicate; the caller must also check feedback freshness."""
    pos, eff = sample.get("grip_pos"), sample.get("grip_eff")
    return (finite(pos) and finite(eff) and hold_threshold(side) <= pos <= HOLD_POS_MAX and eff <= HOLD_EFF_MAX)


def closed_empty(sample, side=None):
    """The gripper pushes closed and reached its empty-closed position: nothing between the fingers."""
    pos, eff = sample.get("grip_pos"), sample.get("grip_eff")
    return finite(pos) and finite(eff) and eff <= HOLD_EFF_MAX and pos < hold_threshold(side)


GRIP_CLASSES = {"open": "张开", "empty": "空夹", "thin": "夹住（薄片）", "solid": "夹住（实体）",
                "closing": "合拢中 / 没用力", "unknown": "无有效读数"}


def classify_grip(sample, side=None):
    """One reading -> open / empty / thin / solid / closing / unknown (the same rules as the grasp detector;
    thin vs solid only splits 'holding' by position for display)."""
    pos, eff = sample.get("grip_pos"), sample.get("grip_eff")
    if not (finite(pos) and finite(eff)):
        return "unknown"
    if pos > HOLD_POS_MAX:
        return "open"
    if closed_empty(sample, side):
        return "empty"
    if holding(sample, side):
        return "solid" if pos >= THIN_SOLID_SPLIT else "thin"
    return "closing"


def grasp_update(previous, sample, t, side=None):
    """Pure state transition for one hand during one attempt.

    success: holding something (solid or thin) AND lifted >= LIFT_MIN above the first close height, for
             HOLD_PERSIST seconds without a break.
    empty:   closed on nothing for EMPTY_PERSIST seconds without reopening (stuck) -> the attempt can be ended.
    no_lift: holding for NO_LIFT_S seconds without reaching success -> the attempt can be ended.
    held:    holding right now (lets the caller ignore an empty other hand while this one holds).
    A fresh detector is used for each attempt. Nothing is judged until the gripper has been seen open in
    this attempt (the previous attempt may leave it closed until the reset opens it). Every reopening
    clears the close height and all timers; invalid/stale/gapped samples break every persistence interval.
    """
    out = dict(previous or {"armed": False, "z_at_close": None, "since": None, "last_t": None,
                            "empty_since": None, "hold_since": None})
    out.update(success=False, empty=False, no_lift=False, held=False)
    if not finite(t) or (out["last_t"] is not None and t <= out["last_t"]):
        out.update(since=None, empty_since=None, hold_since=None)
        return out
    gap = out["last_t"] is not None and t - out["last_t"] > MONITOR_TIMEOUT
    out["last_t"] = t
    pos, z = sample.get("grip_pos"), sample.get("ee_z")
    if sample.get("fresh") is not True or not finite(pos) or not finite(z):
        out.update(since=None, empty_since=None, hold_since=None)
        return out
    if gap:
        out.update(since=None, empty_since=None, hold_since=None)
    if pos > HOLD_POS_MAX:                 # open: arm the detector, forget any earlier close
        out.update(armed=True, z_at_close=None, since=None, empty_since=None, hold_since=None)
        return out
    if not out["armed"]:                   # still closed from before this attempt: judge nothing yet
        return out
    if closed_empty(sample, side):
        out["empty_since"] = out["empty_since"] if out["empty_since"] is not None else t
        out["empty"] = t - out["empty_since"] >= EMPTY_PERSIST
    else:
        out["empty_since"] = None
    if out["z_at_close"] is None:
        out["z_at_close"] = z
    held = out["held"] = holding(sample, side)
    if held:
        out["hold_since"] = out["hold_since"] if out["hold_since"] is not None else t
    else:
        out["hold_since"] = None
    good = held and z >= out["z_at_close"] + LIFT_MIN
    if not good or gap:
        out["since"] = None
    if good:
        if out["since"] is None:
            out["since"] = t
        out["success"] = t - out["since"] >= HOLD_PERSIST
    if out["hold_since"] is not None and not out["success"] and t - out["hold_since"] >= NO_LIFT_S:
        out["no_lift"] = True
    return out


def options(params):
    if set(params) - {"attempts", "prompt", "port", "rtc", "flow"}:
        raise ValueError("mission parameters: attempts, prompt, port, rtc, flow (all optional)")
    cfg = {"attempts": ATTEMPTS, "prompt": PROMPT, "port": PORT, "rtc": RTC, "flow": "v1", **params}
    if cfg["flow"] not in FLOWS:
        raise ValueError("flow must be v1, v2, voice, grasp or handoff")
    if type(cfg["rtc"]) is not bool:
        raise ValueError("rtc must be true or false")
    if type(cfg["attempts"]) is not int or not 1 <= cfg["attempts"] <= ATTEMPTS:
        raise ValueError(f"attempts must be an integer from 1 to {ATTEMPTS}")
    if type(cfg["port"]) is not int or not 1 <= cfg["port"] <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    if not isinstance(cfg["prompt"], str) or not cfg["prompt"].strip() or len(cfg["prompt"]) > 500:
        raise ValueError("prompt must contain 1 to 500 characters")
    # vla_run.sh embeds PROMPT in a single-quoted shell argument.
    if any(c in cfg["prompt"] for c in ("'", "\n", "\r", "\0")):
        raise ValueError("prompt cannot contain single quotes, newlines or NUL")
    return cfg


class MissionFailure(Exception):
    pass


class MissionAborted(Exception):
    pass


class Monitor:
    def __init__(self, cmd, callback, stop_cmd=None):
        self.frame = None
        self.received = 0.0
        self.callback = callback
        self.stop_cmd = stop_cmd
        self.closed = False
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.p.stdout:
                try:
                    frame = json.loads(line)
                    if not isinstance(frame, dict) or not finite(frame.get("t")):
                        continue
                    if self.frame is not None and frame["t"] <= self.frame["t"]:
                        continue
                    if not all(isinstance(frame.get(s), dict) for s in SIDES):
                        continue
                except (ValueError, TypeError):
                    continue
                if self.closed:
                    break
                self.frame, self.received = frame, time.monotonic()
                self.callback(frame)
        finally:
            self.p.stdout.close()

    def fresh(self):
        f = self.frame
        return (self.p.poll() is None and time.monotonic() - self.received < MONITOR_TIMEOUT
                and f is not None and f.get("arms_fresh") is True
                and all(f[s].get("fresh") is True and all(finite(f[s].get(k))
                        for k in ("grip_pos", "grip_eff", "ee_z")) for s in SIDES))

    def close(self):
        self.closed = True
        try:
            if self.stop_cmd:
                subprocess.run(self.stop_cmd, capture_output=True, timeout=5)
        finally:
            stop_process(self.p)


def stop_process(p):
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=3)
        except ProcessLookupError:
            pass


class Mission:
    """Uses the host's seq_lock, run_sequence, stop and subprocess commands."""
    def __init__(self, host):
        self.h = host
        self.state = {"id": None, "status": "idle", "step": None, "hand": None,
                      "attempt": 0, "attempts": ATTEMPTS, "pause_until": None,
                      "result": None, "attempt_results": [], "durations": {}}
        self.active = False
        self.cancel = threading.Event()
        self.monitor = None
        self.fault = None
        self.dropped = False
        self.drop_since = None
        self.detecting = False
        self.watching = False
        self.carrying = False
        self.detectors = {}
        self.success_hand = None
        self.attempt_fail = None
        self.command = None
        self.logfile = None
        self.active_step = None
        self.step_started = None
        self.preflight_seconds = 0.0
        self.handoff_hand = None
        self._vcur = 0

    @property
    def running(self):
        return self.state["status"] == "running"

    def snapshot(self):
        out = copy.deepcopy(self.state)
        out["active"] = self.active
        pause = out["pause_until"]
        out["pause_left_s"] = max(0, pause - time.time()) if pause else 0
        live = out.get("live")
        if live:
            now = time.time()
            live["elapsed_s"] = round((live["ended"] or now) - live["started"], 1)
            live["phase_s"] = round((live["ended"] or now) - live["phase_since"], 1)
        return out

    def abort(self):
        # Called directly by the E-STOP request, bypassing seq_lock.
        self.cancel.set()
        if self.running:
            self.state.update(status="aborted", result="emergency stop", pause_until=None)

    def _monitor_cmd(self):
        return shlex.split(os.environ.get("MONITOR_CMD", f"docker exec {self.h.CONTAINER} "
                           "/usr/local/bin/openpi-client-entrypoint.sh python3 -u /outputs/arm/grasp_monitor.py"))

    def _observe(self, frame):
        valid = frame.get("arms_fresh") is True and all(
            frame[s].get("fresh") is True and all(finite(frame[s].get(k))
            for k in ("grip_pos", "grip_eff", "ee_z")) for s in SIDES)
        if self.active and self.watching and not valid:
            self.fault = "arm feedback lost"
        if self.detecting:
            if self.logfile:      # gripper telemetry for every attempt (~10 Hz) to re-tune the detector later
                try:
                    self._event("grip", ft=frame["t"], **{s[0]: [frame[s].get(k) for k in ("grip_pos", "grip_eff", "ee_z")]
                                                          for s in SIDES})
                except (OSError, ValueError):
                    pass
            hands = {}
            for side in SIDES:
                self.detectors[side] = grasp_update(self.detectors.get(side), frame[side], frame["t"], side)
                hands[side] = self._hand_view(self.detectors[side], frame[side], frame["t"], side)
                if self.detectors[side]["success"] and self.success_hand is None:
                    self.success_hand = side
                elif self.attempt_fail is None and self.success_hand is None:
                    other = self.detectors.get("left" if side == "right" else "right") or {}
                    if self.detectors[side]["empty"] and not other.get("held"):   # the other hand may still win
                        self.attempt_fail = f"{side} gripper closed on nothing"
                    elif self.detectors[side]["no_lift"]:
                        self.attempt_fail = f"{side} gripper holding but not lifted within {NO_LIFT_S:g} s"
            live = self.state.get("live")
            if live:
                live["hands"] = hands          # replaced whole: snapshot() may be copying the old one
        if self.carrying:
            if holding(frame[self.state["hand"]], self.state["hand"]):
                self.drop_since = None
            elif self.drop_since is None:
                self.drop_since = frame["t"]
            elif frame["t"] - self.drop_since >= DROP_PERSIST:
                self.dropped = True

    @staticmethod
    def _hand_view(d, sample, t, side):
        """What the grasp detector currently sees for one hand (for the page; no decisions here)."""
        pos, eff, z = sample.get("grip_pos"), sample.get("grip_eff"), sample.get("ee_z")
        lift = (z - d["z_at_close"]) * 100 if d.get("z_at_close") is not None and finite(z) else None
        age = lambda k: round(t - d[k], 1) if d.get(k) is not None else 0
        return {"pos": round(pos, 3) if finite(pos) else None, "eff": round(eff, 2) if finite(eff) else None,
                "grip": classify_grip(sample, side), "armed": bool(d.get("armed")),
                "lift_cm": round(lift, 1) if lift is not None else None,
                "hold_s": age("hold_since"), "good_s": age("since"), "empty_s": age("empty_since")}

    def _new_monitor(self):
        self.watching = False
        if self.monitor:
            self.monitor.close()
        stop_cmd = None if os.environ.get("MONITOR_CMD") else self.h.DEXEC + ["pkill", "-INT", "-f", "grasp_monitor.py"]
        self.monitor = Monitor(self._monitor_cmd(), self._observe, stop_cmd)
        end = time.monotonic() + 3
        while not self.monitor.fresh():
            self._check(health=False)
            if time.monotonic() >= end or self.monitor.p.poll() is not None:
                raise MissionFailure("no fresh arm/gripper/EE feedback")
            self.cancel.wait(0.02)
        self.watching = True

    def prepare(self, cfg):
        """Preflight before accepting an action; no motion commands or resets."""
        began = time.monotonic()
        self.cancel.clear()
        self.fault = None
        try:
            if self.h.state["estop"]:
                raise MissionAborted()
            if not self.h.state["connected"]:
                raise MissionFailure("chassis is not connected")
            if self.h.state["mode_ok"] is not True:
                raise MissionFailure("robot is not in API mode")
            if cfg.get("flow") != "handoff":          # handoff alone does not run the VLA
                self.h.check_policy(port=cfg["port"])
            missing = [n for n in ("001", "002", "003", "004") if n not in self.h.paths]
            if cfg.get("flow") in ("voice", "grasp", "handoff"):
                missing = []        # step 7 alone does not drive the chassis
            if cfg.get("flow") in ("v2", "voice"):
                try:
                    self._bridge("GET", "/status")
                except MissionFailure as exc:
                    raise MissionFailure("voice module unavailable: " + str(exc)) from exc
            if missing:
                raise MissionFailure("missing paths: " + ", ".join(missing))
            self.detecting = self.carrying = False
            self._new_monitor()
            held = [s for s in SIDES if holding(self.monitor.frame[s], s)]
            if cfg.get("flow") == "handoff":
                if len(held) != 1:      # handoff alone starts from a grasp: exactly one hand holds the object
                    raise MissionFailure("handoff needs exactly one gripper holding an object (holding: "
                                         + (", ".join(held) or "none") + ")")
                self.handoff_hand = held[0]
            elif held:
                raise MissionFailure("a gripper is already holding an object")
            self._check()
            self.preflight_seconds = time.monotonic() - began
        except Exception:
            if self.monitor:
                self.monitor.close()
                self.monitor = None
            raise

    def start(self, cfg, action):
        self.active = True
        mid = uuid.uuid4().hex[:12]
        self.state = {"id": mid, "status": "running", "step": "preflight", "hand": None,
                      "attempt": 0, "attempts": cfg["attempts"], "pause_until": None,
                      "result": None, "attempt_results": [], "durations": {},
                      "config": {k: cfg.get(k) for k in ("model", "model_name", "port", "prompt", "rtc", "flow")},
                      "flow": cfg.get("flow", "v1"), "live": None,
                      "voice": None if cfg.get("flow", "v1") in ("v1", "grasp") else {
                          "hand": None, "attempt_results": [],
                          "steps": [{"id": i, "name": n, "uses": u, "status": "pending", "detail": ""}
                                    for i, n, u in VOICE_STEPS]}}
        self.active_step, self.step_started = None, None
        self.dropped = False
        self.drop_since = None
        threading.Thread(target=self._run, args=(cfg, action), daemon=True).start()

    def _event(self, event, **data):
        self.logfile.write(json.dumps({"t": time.time(), "event": event, **data}, ensure_ascii=False) + "\n")
        self.logfile.flush()

    def _finish_step(self):
        if self.active_step:
            dt = time.monotonic() - self.step_started
            if self.active_step == "preflight":
                dt += self.preflight_seconds
            self.state["durations"][self.active_step] = self.state["durations"].get(self.active_step, 0) + dt
            self._event("step_end", step=self.active_step, seconds=dt)
        self.active_step = None

    def _step(self, name):
        self._check()
        self._finish_step()
        self.state.update(step=name, pause_until=None)
        self.active_step, self.step_started = name, time.monotonic()
        self._event("step", step=name)

    def _check(self, health=True):
        if self.cancel.is_set() or self.h.state["estop"]:
            raise MissionAborted("emergency stop")
        if self.fault:
            raise MissionFailure(self.fault)
        if health:
            if not self.monitor or not self.monitor.fresh():
                raise MissionFailure("arm feedback lost or monitor stopped")
            if not self.h.state["connected"] or self.h.state["mode_ok"] is not True:
                raise MissionFailure("chassis disconnected or API mode lost")

    def _pause(self):
        self.state["pause_until"] = time.time() + ARRIVE_PAUSE
        end = time.monotonic() + ARRIVE_PAUSE
        while time.monotonic() < end:
            self._check()
            if self.carrying and self.dropped:
                raise MissionFailure("object dropped")
            self.cancel.wait(0.02)
        self.state["pause_until"] = None

    def _route(self, route):
        h = self.h
        names = h.ROUTES[route][1][:]
        with h.seq_lock:
            self._check()
            h.seq.update(names=names, i=-1, abort=False, route=route, pause_until=None)
        # The existing runner calls the guard while waiting; a drop is checked
        # only at path boundaries, before the next path is dispatched.
        result = h.run_sequence(names, guard=self._route_guard,
                                boundary=lambda: not self.dropped if self.carrying else True)
        self._check()
        if self.carrying and self.dropped:
            raise MissionFailure("object dropped during return route")
        if result != "finished":
            raise MissionFailure("route incomplete: " + result)

    def _route_guard(self):
        try:
            self._check()
            return True
        except (MissionFailure, MissionAborted) as exc:
            if not self.cancel.is_set():
                self.fault = str(exc)
            self.h.seq["abort"] = True
            self.h.send("stop")
            return False

    def _output(self, p):
        for line in p.stdout:
            self._event("output", text=line.rstrip())
            if self.detecting:
                self._phase_from(line)
        p.stdout.close()

    # vla_run.sh output -> attempt phase shown on the page (reset -> probe -> policy)
    PHASE_MARKS = (("Arms: start", "reset"), ("Starting GStreamer pipeline", "probe"),
                   ('"observation_ready"', "policy"), ("retrying once with a fresh process", "policy_retry"))

    def _phase(self, name, note=None):
        live = self.state.get("live")
        if live and live["phase"] != name:
            live.update(phase=name, phase_since=time.time(), note=note if note is not None else live["note"])

    def _phase_from(self, line):
        live = self.state.get("live")
        if not live or live["phase"] in ("stopping", "ended"):
            return
        for mark, name in self.PHASE_MARKS:
            if mark in line and not (name == "probe" and live["phase"] not in ("reset", "starting")):
                self._phase(name)
        if line.startswith("startup: waiting for live data"):
            live["note"] = line.strip()[:160]

    def _stop_command(self, kind):
        # Killing docker-exec locally does not stop its container-side process.
        # vla_run.sh = arm.sh start (arm_pose.py) -> live_observation_probe.py -> guarded_grasp.py: stop all of it,
        # or a reset still moving in the container would race the next attempt's reset.
        pattern = "guarded_grasp.py|arm_pose.py|live_observation_probe.py" if kind == "vla" else "arm_pose.py"
        override = {"vla": "VLA_CMD", "down": "ARM_DOWN_CMD", "move": "ARM_MOVE_CMD", "arm": "ARM_CMD"}[kind]
        if not os.environ.get(override):
            subprocess.run(self.h.DEXEC + ["pkill", "-INT", "-f", pattern],
                           capture_output=True, timeout=5)
        if self.command:
            stop_process(self.command)
        if not os.environ.get(override):
            # Ensure the next step cannot race an old writer still holding a pose.
            for _ in range(30):
                rc = subprocess.run(self.h.DEXEC + ["pgrep", "-f", pattern],
                                    capture_output=True, timeout=5).returncode
                if rc == 1:
                    return
                if rc != 0:
                    raise MissionFailure("cannot confirm script stopped")
                time.sleep(0.1)
            subprocess.run(self.h.DEXEC + ["pkill", "-KILL", "-f", pattern],
                           capture_output=True, timeout=5)
            raise MissionFailure("script did not stop cleanly")

    def _execute(self, cmd, kind="arm", env=None, attempt=False):
        self._check()
        # A reader wraps output as JSON events; the safety loop never waits on readline.
        self._event("command", argv=cmd)
        with self.h.seq_lock:
            self._check()
            self.command = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            env=env, text=True, start_new_session=True)
        output = threading.Thread(target=self._output, args=(self.command,), daemon=True)
        output.start()
        deadline = time.monotonic() + (ATTEMPT_S + SCRIPT_TIMEOUT if attempt else SCRIPT_TIMEOUT)
        try:
            while True:
                self._check()
                rc = self.command.poll()
                if rc is not None:
                    # A grasp confirmed just as the run ended (its 1.5 s hold straddling the time limit) still counts.
                    return rc, (self.success_hand if attempt else None)
                if attempt and self.success_hand:
                    hand = self.success_hand
                    self._phase("stopping", "抓住了，正在停止 VLA")
                    self.detecting = False
                    self._stop_command("vla")
                    return self.command.returncode, hand
                if attempt and self.attempt_fail:          # empty grasp / no lift: do not wait for the time limit
                    self._phase("stopping", self.attempt_fail)
                    self.detecting = False
                    self._stop_command("vla")
                    return self.command.returncode, None
                if self.carrying and self.dropped:
                    raise MissionFailure("object dropped")
                if time.monotonic() >= deadline:
                    self._stop_command(kind)
                    if attempt:
                        return None, None
                    raise MissionFailure("script timeout")
                self.cancel.wait(0.02)
        finally:
            if self.command and self.command.poll() is None:
                self._stop_command(kind)
            output.join(timeout=3)
            self.command = None

    def _arm(self, action):
        rc, _ = self._execute(self.h.ARM_CMD + [action])
        if rc != 0:
            raise MissionFailure(f"arm {action} exited {rc}")

    def _grasp_attempts(self, cfg, prefix, results):
        """Up to cfg["attempts"] VLA runs; each stops as soon as the grasp detector reports success.
        Returns the grasping hand or None. Shared by step 3 and step 7.2."""
        for i in range(1, cfg["attempts"] + 1):
            self._check()
            self.state["attempt"] = i
            self.detecting = False
            self._new_monitor()
            self.detectors, self.success_hand, self.attempt_fail = {}, None, None
            self.detecting = True
            tag = f"{prefix}_{i}"
            entry = {"attempt": i, "tag": tag, "result": "running", "hand": None}
            began = time.time()
            self.state["live"] = {"attempt": i, "attempts": cfg["attempts"], "phase": "starting", "phase_since": began,
                                  "started": began, "ended": None, "note": "", "verdict": None, "hand": None,
                                  "hands": {}, "elapsed_s": 0, "phase_s": 0,
                                  "rules": {"limit_s": ATTEMPT_S, "lift_cm": LIFT_MIN * 100, "success_s": HOLD_PERSIST,
                                            "empty_s": EMPTY_PERSIST, "no_lift_s": NO_LIFT_S,
                                            "hold_min": {s: round(hold_threshold(s), 4) for s in SIDES},
                                            "open_above": HOLD_POS_MAX, "effort_max": HOLD_EFF_MAX}}
            results.append(entry)
            self._event("attempt_start", **entry)
            try:
                rc, hand = self._execute(self.h.VLA_CMD + [tag, str(ATTEMPT_S), "1.0"],
                                         kind="vla", attempt=True,
                                         env={**os.environ, "PROMPT": cfg["prompt"], "PORT": str(cfg["port"]), "RESET": "1",
                                              "RTC": "1" if cfg.get("rtc") else "0"})
                reason = None if hand else (self.attempt_fail or ("time limit without a grasp" if rc in (0, None)
                                                                  else f"VLA script exited {rc} without a grasp"))
                entry.update(result="succeeded" if hand else "failed", hand=hand, returncode=rc, reason=reason)
            except Exception:
                entry.update(result="aborted" if self.cancel.is_set() else "failed")
                raise
            finally:
                self.detecting = False
                entry["seconds"] = round(time.time() - began, 1)
                live = self.state["live"]
                live.update(ended=time.time(), verdict=entry["result"], hand=entry["hand"],
                            note=entry.get("reason") or live["note"])
                self._phase("ended")
                self._event("attempt_end", **entry)
            if hand:
                return hand
        return None

    def _bridge(self, method, path, body=None, timeout=4):
        """Mac voice service (microphone + speaker) through the B70 reverse tunnel."""
        data = None if body is None else json.dumps(body).encode()
        req = Request(VOICE_BRIDGE + path, data=data, method=method,
                      headers={"Content-Type": "application/json", "Host": VOICE_BRIDGE_HOST})
        try:
            with urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except (URLError, OSError, ValueError) as exc:
            raise MissionFailure(f"Mac voice service: {exc}") from exc

    def _voice_sub(self, index, status, detail=None):
        sub = self.state["voice"]["steps"][index]
        sub["status"] = status
        if detail is not None:
            sub["detail"] = detail
        self._event("voice_step", id=sub["id"], status=status, detail=sub["detail"])

    def _voice(self, cfg):
        """Step 7, voice interaction module: 7.1 listen ... 7.8 announce. Each sub-step must succeed."""
        self._step("voice")
        v, self._vcur = self.state["voice"], 0
        try:
            # 7.1 the recipient asks for the bag
            self._voice_sub(0, "running", "Mac 正在听")
            seen = self._bridge("POST", "/rehearse", {}).get("state") == "listening"
            deadline = time.monotonic() + VOICE_LISTEN_S
            while True:
                self._check()
                state = self._bridge("GET", "/status").get("state")
                seen = seen or state == "listening"
                if state == "recognized" and seen:
                    break
                if state == "failed":
                    raise MissionFailure("7.1 voice recognition failed")
                if time.monotonic() >= deadline:
                    raise MissionFailure(f"7.1 no voice command within {VOICE_LISTEN_S:g} s")
                self.cancel.wait(0.2)
            self._voice_sub(0, "succeeded", "已识别口令")
            # 7.2 grasp the bag from the table (same attempts and detector as step 3)
            self._vcur = 1
            self._voice_sub(1, "running", "抓住即停")
            hand = self._grasp_attempts(cfg, f"mission_{self.state['id']}_v", v["attempt_results"])
            if hand is None:
                self._arm("start")
                raise MissionFailure("7.2 all grasp attempts failed; arms reset")
            v["hand"] = self.state["hand"] = hand
            self._voice_sub(1, "succeeded", "已抓住（" + ("左手" if hand == "left" else "右手") + "）")
            # 7.3 stow, gripper kept closed
            self._vcur = 2
            self._voice_sub(2, "running")
            self._arm("start_hold")
            self._check()
            if not holding(self.monitor.frame[hand], hand):
                raise MissionFailure("7.3 object dropped after stow")
            self.drop_since, self.dropped, self.carrying = None, False, True
            self._voice_sub(2, "succeeded")
            self._handoff(hand)
            # 7.8 completion announcement on the Mac speaker
            self._vcur = 7
            self._voice_sub(7, "running")
            self._bridge("POST", "/confirm", {})
            deadline = time.monotonic() + VOICE_ANNOUNCE_S
            while True:
                self._check()
                state = self._bridge("GET", "/status").get("state")
                if state == "done":
                    break
                if state == "failed":
                    raise MissionFailure("7.8 announcement failed")
                if time.monotonic() >= deadline:
                    raise MissionFailure("7.8 announcement timeout")
                self.cancel.wait(0.2)
            self._voice_sub(7, "succeeded", "播报完成")
        except BaseException:
            self._voice_sub(self._vcur, "failed")
            raise

    def _handoff(self, hand):
        """7.4 reach out, 7.5 lower, 7.6 release, 7.7 confirm. Shared by step 7 and flow "handoff" (alone).
        The caller has checked that `hand` holds the object and set carrying."""
        # 7.4 reach out toward the recipient (straight line, gripper kept closed)
        self._vcur = 3
        self._voice_sub(3, "running")
        if HANDOFF_FORWARD_CM > 0:
            rc, _ = self._execute(self.h.ARM_MOVE_CMD + [hand, f"{HANDOFF_FORWARD_CM:g}", "0", "0"], kind="move")
            if rc != 0:
                raise MissionFailure(f"7.4 arm_move exited {rc}")
            self._check()
            if self.dropped or not holding(self.monitor.frame[hand], hand):
                raise MissionFailure("7.4 object dropped while reaching out")
            self._voice_sub(3, "succeeded")
        else:
            self._voice_sub(3, "succeeded", "已跳过（HANDOFF_FORWARD_CM=0）")
        # 7.5 lower toward the recipient's hand
        self._vcur = 4
        self._voice_sub(4, "running")
        rc, _ = self._execute(self.h.ARM_DOWN_CMD + [hand, f"{HANDOFF_CM:g}"], kind="down")
        if rc != 0:
            raise MissionFailure(f"7.5 arm_down exited {rc}")
        self._check()
        if self.dropped or not holding(self.monitor.frame[hand], hand):
            raise MissionFailure("7.5 object dropped before release")
        self._voice_sub(4, "succeeded")
        # 7.6 release (only the grasping hand)
        self._vcur = 5
        self._voice_sub(5, "running")
        self.carrying = False      # opening is intentional from here on
        self._arm("open_" + hand)
        self._check()
        self._voice_sub(5, "succeeded")
        # 7.7 handover confirmed: that gripper is open and holds nothing
        self._vcur = 6
        self._voice_sub(6, "running")
        deadline = time.monotonic() + VOICE_RELEASE_S
        while True:
            self._check()
            f = self.monitor.frame[hand]
            if finite(f.get("grip_pos")) and f["grip_pos"] >= OPEN_MIN and not holding(f, hand):
                break
            if time.monotonic() >= deadline:
                raise MissionFailure(f"7.7 gripper not confirmed open/empty (pos {f.get('grip_pos')})")
            self.cancel.wait(0.05)
        self._voice_sub(6, "succeeded", f"夹爪 {f['grip_pos']:.2f} rad，未夹持")

    def _run(self, cfg, action):
        status, reason = "failed", "mission did not finish"
        try:
            Path(MISSION_LOG_DIR).mkdir(parents=True, exist_ok=True)
            self.logfile = open(Path(MISSION_LOG_DIR) / (self.state["id"] + ".jsonl"), "a", buffering=1)
            self._event("start", id=self.state["id"], config=cfg)
            self._step("preflight")
            flow = cfg.get("flow", "v1")
            if flow == "grasp":
                # Step 3 alone: the same attempts/detector as the full flow; on success the arm stops where it
                # is with the gripper still closed (no stow), like the page's single grasp.
                self._step("grasp")
                hand = self._grasp_attempts(cfg, f"mission_{self.state['id']}", self.state["attempt_results"])
                if hand is None:
                    self._arm("start")
                    raise MissionFailure("all grasp attempts failed; arms reset")
                self.state["hand"] = hand
            elif flow == "handoff":
                # 7.4-7.7 alone (reach out, lower, release, confirm) with the hand that holds the object now.
                self._step("handoff")
                hand = self.state["hand"] = self.state["voice"]["hand"] = self.handoff_hand
                for k in range(3):
                    self._voice_sub(k, "skipped", "单独放下：未执行")
                self.drop_since, self.dropped, self.carrying = None, False, True
                self._vcur = 3
                try:
                    self._handoff(hand)
                except BaseException:
                    self._voice_sub(self._vcur, "failed")
                    raise
                for k in range(7, len(VOICE_STEPS)):
                    self._voice_sub(k, "skipped", "单独放下：未执行")
            elif flow != "voice":
                self._steps_1_to_6(cfg)
            if flow in ("v2", "voice"):
                self._voice(cfg)
            status, reason = "succeeded", {"v1": "placed", "grasp": "grasped", "handoff": "released"}.get(flow, "handed over")
        except MissionAborted:
            status, reason = "aborted", "emergency stop"
        except Exception as exc:
            reason = str(exc)
        finally:
            self._finish(status, reason, action)

    def _steps_1_to_6(self, cfg):
        self._step("g2r")
        self._route("g2r")
        self._pause()
        self._step("grasp")
        hand = self._grasp_attempts(cfg, f"mission_{self.state['id']}", self.state["attempt_results"])
        if hand is None:
            self._arm("start")
            raise MissionFailure("all grasp attempts failed; arms reset")
        self.state["hand"] = hand
        self._step("stow")
        self._arm("start_hold")
        self._check()
        if not holding(self.monitor.frame[self.state["hand"]], self.state["hand"]):
            raise MissionFailure("object dropped after stow")
        self.drop_since = None
        self.carrying = True
        self._step("r2g")
        self._route("r2g")
        self._pause()
        self._step("place")
        if PLACE_FORWARD_CM > 0:      # same as the handoff (7.4): reach out first, then lower, then release
            rc, _ = self._execute(self.h.ARM_MOVE_CMD + [self.state["hand"], f"{PLACE_FORWARD_CM:g}", "0", "0"], kind="move")
            if rc != 0:
                raise MissionFailure(f"arm_move exited {rc}")
            self._check()
            if self.dropped or not holding(self.monitor.frame[self.state["hand"]], self.state["hand"]):
                raise MissionFailure("object dropped while reaching out")
        rc, _ = self._execute(self.h.ARM_DOWN_CMD + [self.state["hand"], f"{PLACE_CM:g}"], kind="down")
        if rc != 0:
            raise MissionFailure(f"arm_down exited {rc}")
        self._check()
        if self.dropped or not holding(self.monitor.frame[self.state["hand"]], self.state["hand"]):
            raise MissionFailure("object dropped before release")
        self.carrying = False  # opening is intentional from this point on
        self._arm("open_" + self.state["hand"])
        self._check()

    def _finish(self, status, reason, action):
        self.detecting = self.carrying = False
        try:
            if status != "succeeded":
                self.h.seq["abort"] = True
                self.h.send("stop")
        except Exception as exc:
            reason += f"; stop failed: {exc}"
        try:
            if self.monitor:
                self.monitor.close()
        except Exception as exc:
            status = "failed"
            reason += f"; monitor cleanup failed: {exc}"
        finally:
            self.monitor = None
        with self.h.seq_lock:
            if self.cancel.is_set():
                status, reason = "aborted", "emergency stop"
            self.state.update(status=status, result=reason, pause_until=None)
            self.active = False
            try:
                if self.logfile:
                    self._finish_step()
                    self._event("summary", **self.snapshot())
            except OSError as exc:
                self.h.note(f"mission final log failed: {exc}")
            finally:
                try:
                    if self.logfile:
                        self.logfile.close()
                finally:
                    self.logfile = None
                    self.h.end_action(action, self.snapshot(), status == "succeeded",
                                      code="ESTOP" if status == "aborted" else None)
            self.h.note(f"mission {self.state['id']}: {status}: {reason}")
