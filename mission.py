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

ATTEMPTS = int(os.environ.get("ATTEMPTS", "5"))
ATTEMPT_S = float(os.environ.get("ATTEMPT_S", "45"))
PROMPT = os.environ.get("PROMPT", "抓取沙包")
PORT = int(os.environ.get("MISSION_PORT", os.environ.get("POLICY_PORT", "8000")))
RTC = os.environ.get("MISSION_RTC", "0") == "1"   # async Real-Time Chunking for the grasp attempts (vla_run.sh RTC=1)
ARRIVE_PAUSE = float(os.environ.get("ARRIVE_PAUSE", "10"))
PLACE_CM = float(os.environ.get("PLACE_CM", "10"))
HOLD_POS_MIN = float(os.environ.get("HOLD_POS_MIN", "0.06"))  # calibrated: empty-closed is -0.047 (R) / -0.059 (L) rad
HOLD_POS_MAX = float(os.environ.get("HOLD_POS_MAX", "0.75"))
HOLD_EFF_MAX = float(os.environ.get("HOLD_EFF_MAX", "-1.0"))
LIFT_MIN = float(os.environ.get("LIFT_MIN", "0.04"))
HOLD_PERSIST = float(os.environ.get("HOLD_PERSIST", "1.5"))
DROP_PERSIST = float(os.environ.get("DROP_PERSIST", "0.25"))
MISSION_LOG_DIR = os.environ.get("MISSION_LOG_DIR", "/root/ta2_mission")
MONITOR_TIMEOUT = float(os.environ.get("MONITOR_TIMEOUT", "0.6"))
SCRIPT_TIMEOUT = float(os.environ.get("SCRIPT_TIMEOUT", "90"))
SIDES = ("left", "right")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def holding(sample):
    """Pure position/effort predicate. Caller must also check feedback freshness."""
    pos, eff = sample.get("grip_pos"), sample.get("grip_eff")
    return (finite(pos) and finite(eff) and HOLD_POS_MIN <= pos <= HOLD_POS_MAX and eff <= HOLD_EFF_MAX)


def grasp_update(previous, sample, t):
    """Pure state transition: close height + uninterrupted loaded, lifted hold.

    A fresh detector is used for each attempt. The first close height remains
    fixed; opening/invalid/stale/gapped samples break the persistence interval.
    """
    out = dict(previous or {"z_at_close": None, "since": None, "last_t": None})
    out["success"] = False
    if not finite(t) or (out["last_t"] is not None and t <= out["last_t"]):
        out["since"] = None
        return out
    gap = out["last_t"] is not None and t - out["last_t"] > MONITOR_TIMEOUT
    out["last_t"] = t
    pos, z = sample.get("grip_pos"), sample.get("ee_z")
    if sample.get("fresh") is not True or not finite(pos) or not finite(z):
        out["since"] = None
        return out
    if pos > HOLD_POS_MAX:
        out["since"] = None
        return out
    if out["z_at_close"] is None:
        out["z_at_close"] = z
    good = holding(sample) and z >= out["z_at_close"] + LIFT_MIN
    if not good or gap:
        out["since"] = None
    if good:
        if out["since"] is None:
            out["since"] = t
        out["success"] = t - out["since"] >= HOLD_PERSIST
    return out


def options(params):
    if set(params) - {"attempts", "prompt", "port", "rtc"}:
        raise ValueError("mission parameters: attempts, prompt, port, rtc (all optional)")
    cfg = {"attempts": ATTEMPTS, "prompt": PROMPT, "port": PORT, "rtc": RTC, **params}
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
        self.command = None
        self.logfile = None
        self.active_step = None
        self.step_started = None
        self.preflight_seconds = 0.0

    @property
    def running(self):
        return self.state["status"] == "running"

    def snapshot(self):
        out = copy.deepcopy(self.state)
        out["active"] = self.active
        pause = out["pause_until"]
        out["pause_left_s"] = max(0, pause - time.time()) if pause else 0
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
            for side in SIDES:
                self.detectors[side] = grasp_update(self.detectors.get(side), frame[side], frame["t"])
                if self.detectors[side]["success"] and self.success_hand is None:
                    self.success_hand = side
        if self.carrying:
            if holding(frame[self.state["hand"]]):
                self.drop_since = None
            elif self.drop_since is None:
                self.drop_since = frame["t"]
            elif frame["t"] - self.drop_since >= DROP_PERSIST:
                self.dropped = True

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
            self.h.check_policy(port=cfg["port"])
            missing = [n for n in ("001", "002", "003", "004") if n not in self.h.paths]
            if missing:
                raise MissionFailure("missing paths: " + ", ".join(missing))
            self.detecting = self.carrying = False
            self._new_monitor()
            if any(holding(self.monitor.frame[s]) for s in SIDES):
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
                      "config": {k: cfg.get(k) for k in ("model", "model_name", "port", "prompt", "rtc")}}
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
        p.stdout.close()

    def _stop_command(self, kind):
        # Killing docker-exec locally does not stop its container-side process.
        pattern = "guarded_grasp.py" if kind == "vla" else "arm_pose.py"
        override = {"vla": "VLA_CMD", "down": "ARM_DOWN_CMD", "arm": "ARM_CMD"}[kind]
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
                    return rc, None
                if attempt and self.success_hand:
                    hand = self.success_hand
                    self.detecting = False
                    self._stop_command("vla")
                    return self.command.returncode, hand
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

    def _run(self, cfg, action):
        status, reason = "failed", "mission did not finish"
        try:
            Path(MISSION_LOG_DIR).mkdir(parents=True, exist_ok=True)
            self.logfile = open(Path(MISSION_LOG_DIR) / (self.state["id"] + ".jsonl"), "a", buffering=1)
            self._event("start", id=self.state["id"], config=cfg)
            self._step("preflight")
            self._step("g2r")
            self._route("g2r")
            self._pause()
            self._step("grasp")
            for i in range(1, cfg["attempts"] + 1):
                self._check()
                self.state["attempt"] = i
                self.detecting = False
                self._new_monitor()
                self.detectors, self.success_hand = {}, None
                self.detecting = True
                tag = f"mission_{self.state['id']}_{i}"
                entry = {"attempt": i, "tag": tag, "result": "running", "hand": None}
                self.state["attempt_results"].append(entry)
                self._event("attempt_start", **entry)
                try:
                    rc, hand = self._execute(self.h.VLA_CMD + [tag, str(ATTEMPT_S), "1.0"],
                                             kind="vla", attempt=True,
                                             env={**os.environ, "PROMPT": cfg["prompt"], "PORT": str(cfg["port"]), "RESET": "1",
                                                  "RTC": "1" if cfg.get("rtc") else "0"})
                    entry.update(result="succeeded" if hand else "failed", hand=hand, returncode=rc)
                except Exception:
                    entry.update(result="aborted" if self.cancel.is_set() else "failed")
                    raise
                finally:
                    self.detecting = False
                    self._event("attempt_end", **entry)
                if hand:
                    self.state["hand"] = hand
                    break
            if self.state["hand"] is None:
                self._arm("start")
                raise MissionFailure("all grasp attempts failed; arms reset")
            self._step("stow")
            self._arm("start_hold")
            self._check()
            if not holding(self.monitor.frame[self.state["hand"]]):
                raise MissionFailure("object dropped after stow")
            self.drop_since = None
            self.carrying = True
            self._step("r2g")
            self._route("r2g")
            self._pause()
            self._step("place")
            rc, _ = self._execute(self.h.ARM_DOWN_CMD + [self.state["hand"], f"{PLACE_CM:g}"], kind="down")
            if rc != 0:
                raise MissionFailure(f"arm_down exited {rc}")
            self._check()
            if self.dropped or not holding(self.monitor.frame[self.state["hand"]]):
                raise MissionFailure("object dropped before release")
            self.carrying = False  # opening is intentional from this point on
            self._arm("open_" + self.state["hand"])
            self._check()
            status, reason = "succeeded", "placed"
        except MissionAborted:
            status, reason = "aborted", "emergency stop"
        except Exception as exc:
            reason = str(exc)
        finally:
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
