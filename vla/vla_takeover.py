#!/usr/bin/env python3
"""Guarded single-grasp runner: VLA policy -> tabletop grasp detect -> 8 cm vertical IK lift -> mid-air check -> start_hold."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):
    if (candidate / "mission.py").exists():
        sys.path.insert(0, str(candidate))
        break

from mission import LIFT_CM, SIDES, Monitor, finite, grasp_update, holding, overload_update, stop_process  # noqa: E402

CONTAINER = os.environ.get("CONTAINER", "openpi-xpu-client")
DEXEC = shlex.split(os.environ.get("ESTOP_DEXEC", f"docker exec {CONTAINER}"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/root/openpi-xpu/client/outputs"))
ARM_SH = shlex.split(os.environ.get("ARM_SH_CMD", "/root/arm.sh"))
ARM_UP = shlex.split(os.environ.get("ARM_UP_CMD", "/root/arm_up.sh"))


def read_report(tag):
    path = OUTPUTS_DIR / tag / "report.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def stop_container_proc(pattern):
    if os.environ.get("GUARDED_CMD"):
        return
    subprocess.run(DEXEC + ["pkill", "-INT", "-f", pattern], capture_output=True, timeout=5)
    for _ in range(30):
        rc = subprocess.run(DEXEC + ["pgrep", "-f", pattern], capture_output=True, timeout=5).returncode
        if rc == 1:
            return
        time.sleep(0.1)
    subprocess.run(DEXEC + ["pkill", "-KILL", "-f", pattern], capture_output=True, timeout=5)


def run_step(cmd, interrupted):
    if interrupted.is_set():
        return 130
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    lines = []

    def pump():
        for ln in p.stdout:
            s = ln.rstrip()
            if s and "multicast" not in s and not s.startswith("Current mode"):
                print(s, flush=True)
                lines.append(s)
        p.stdout.close()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    while p.poll() is None:
        if interrupted.is_set():
            stop_container_proc("arm_pose.py")
            stop_process(p)
            t.join(timeout=2)
            return 130
        interrupted.wait(0.05)
    t.join(timeout=2)
    return p.returncode


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: vla_takeover.py <tag> [seconds=60] [max_speed=0.5]")
    tag = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    prompt = os.environ.get("PROMPT", "抓取沙包")
    port = int(os.environ.get("PORT", "8000"))

    interrupted = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: interrupted.set())

    probe_cmd = (shlex.split(os.environ["PROBE_CMD"]) + [tag] if os.environ.get("PROBE_CMD")
                 else ["docker", "exec", "-i", CONTAINER, "bash", "-lc",
                       f"source /opt/ros/humble/setup.bash; cd /outputs/tools && "
                       f"python3 live_observation_probe.py --output-dir /outputs/{tag}_probe | tail -1"])
    if run_step(probe_cmd, interrupted) != 0:
        print(json.dumps({"status": "probe_failed", "actuations": 0, "error": "live observation probe failed"}), flush=True)
        return 1

    state = {"detectors": {}, "overloads": {}, "hand": None, "overload": None}

    def observe(frame):
        for side in SIDES:
            state["detectors"][side] = grasp_update(state["detectors"].get(side), frame[side], frame["t"])
            if state["detectors"][side]["success"] and state["hand"] is None:
                state["hand"] = side
            state["overloads"][side] = overload_update(state["overloads"].get(side), frame[side], frame["t"])
            if state["overloads"][side]["overloaded"] and state["overload"] is None:
                state["overload"] = side

    mon_cmd = shlex.split(os.environ.get(
        "MONITOR_CMD",
        f"docker exec {CONTAINER} /usr/local/bin/openpi-client-entrypoint.sh python3 -u /outputs/arm/grasp_monitor.py"))
    mon_stop = None if os.environ.get("MONITOR_CMD") else DEXEC + ["pkill", "-INT", "-f", "grasp_monitor.py"]
    monitor = Monitor(mon_cmd, observe, mon_stop)
    grasp_proc = None
    try:
        deadline = time.monotonic() + 3.0
        while not monitor.fresh():
            if interrupted.is_set():
                print(json.dumps({"status": "interrupted", "actuations": 0, "error": "interrupted"}), flush=True)
                return 130
            if time.monotonic() >= deadline or monitor.p.poll() is not None:
                print(json.dumps({"status": "no_feedback", "actuations": 0, "error": "no fresh arm/gripper/EE feedback"}), flush=True)
                return 1
            interrupted.wait(0.02)

        state["detectors"].clear()
        state["overloads"].clear()
        state["hand"] = None
        state["overload"] = None

        vla_cmd = (shlex.split(os.environ["GUARDED_CMD"]) + [tag, str(seconds), str(speed)]
                   if os.environ.get("GUARDED_CMD")
                   else ["docker", "exec", "-i", CONTAINER, "bash", "-lc",
                         f"source /opt/ros/humble/setup.bash; cd /outputs/tools && "
                         f"python3 guarded_grasp.py --observation /outputs/{tag}_probe/observation.npz "
                         f"--output-dir /outputs/{tag} --execute --prompt {shlex.quote(prompt)} --port {port} "
                         f"--max-seconds {seconds} --max-actions 5400 --max-speed {speed}"])
        grasp_proc = subprocess.Popen(vla_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, start_new_session=True)

        def pump_vla():
            for ln in grasp_proc.stdout:
                s = ln.rstrip()
                if s and " INFO " not in s and "multicast" not in s:
                    print(s, flush=True)
            grasp_proc.stdout.close()

        vla_out = threading.Thread(target=pump_vla, daemon=True)
        vla_out.start()

        stop_reason = None
        hard_deadline = time.monotonic() + seconds + 30.0
        while grasp_proc.poll() is None:
            if interrupted.is_set():
                stop_reason = "interrupted"
                stop_container_proc("guarded_grasp.py")
                stop_process(grasp_proc)
                break
            if not monitor.fresh():
                stop_reason = "feedback_lost"
                stop_container_proc("guarded_grasp.py")
                stop_process(grasp_proc)
                break
            if state["hand"] is not None:
                stop_reason = "grasped"
                print(f"[takeover] {state['hand']} gripper grasp confirmed; stopping VLA policy...", flush=True)
                stop_container_proc("guarded_grasp.py")
                stop_process(grasp_proc)
                break
            if state["overload"] is not None:
                stop_reason = "overload"
                print(f"[takeover] {state['overload']} arm joint torque overload; stopping VLA to protect motors...", flush=True)
                stop_container_proc("guarded_grasp.py")
                stop_process(grasp_proc)
                break
            if time.monotonic() >= hard_deadline:
                stop_reason = "timeout"
                stop_container_proc("guarded_grasp.py")
                stop_process(grasp_proc)
                break
            interrupted.wait(0.02)

        vla_out.join(timeout=3)
        rc = grasp_proc.returncode
        grasp_proc = None
        rep = read_report(tag)
        actuations = rep.get("actuations")

        if stop_reason == "interrupted" or interrupted.is_set():
            print(json.dumps({"status": "interrupted", "actuations": actuations, "error": "interrupted"}), flush=True)
            return 130
        if stop_reason == "feedback_lost":
            print(json.dumps({"status": "feedback_lost", "actuations": actuations, "error": "arm/gripper feedback lost"}), flush=True)
            return 1
        if stop_reason == "overload":
            side = state["overload"]
            print(json.dumps({"status": "overload_stopped", "hand": side, "actuations": actuations,
                              "error": f"{side} arm torque overload stopped"}), flush=True)
            return 1
        if stop_reason == "timeout":
            print(json.dumps({"status": "timeout", "actuations": actuations, "error": "VLA script timeout"}), flush=True)
            return 1

        if stop_reason != "grasped":
            status = rep.get("status") or ("done" if rc == 0 else "aborted")
            err = rep.get("error")
            print(json.dumps({"status": status, "actuations": actuations, "error": err}), flush=True)
            return 0 if rc == 0 and not err else (rc or 1)

        hand = state["hand"]
        interrupted.wait(0.25)
        if interrupted.is_set():
            print(json.dumps({"status": "interrupted", "actuations": actuations, "error": "interrupted"}), flush=True)
            return 130

        print(f"[takeover] Lifting {hand} arm vertically by {LIFT_CM:g} cm...", flush=True)
        rc_up = run_step(ARM_UP + [hand, f"{LIFT_CM:g}"], interrupted)
        if interrupted.is_set() or rc_up == 130:
            print(json.dumps({"status": "interrupted", "actuations": actuations, "error": "interrupted"}), flush=True)
            return 130
        if rc_up != 0:
            print(json.dumps({"status": "lift_failed", "hand": hand, "actuations": actuations,
                              "error": f"arm_up exited {rc_up}"}), flush=True)
            return 1

        interrupted.wait(0.20)
        if not monitor.fresh():
            print(json.dumps({"status": "feedback_lost", "hand": hand, "actuations": actuations,
                              "error": "feedback lost after lift"}), flush=True)
            return 1
        snap = monitor.frame[hand]
        if not holding(snap):
            print(f"[takeover] {hand} gripper not holding in mid-air (pos={snap.get('grip_pos')}, eff={snap.get('grip_eff')}); resetting arms to start...", flush=True)
            run_step(ARM_SH + ["start", "--time", "8"], interrupted)
            print(json.dumps({"status": "slip_reset", "hand": hand, "actuations": actuations,
                              "error": "object slipped during lift; arms reset to start"}), flush=True)
            return 1

        print(f"[takeover] Mid-air hold verified on {hand} (pos={snap['grip_pos']:.3f} rad, eff={snap['grip_eff']:.2f} N·m); returning to start_hold...", flush=True)
        rc_stow = run_step(ARM_SH + ["start_hold", "--time", "8"], interrupted)
        if interrupted.is_set() or rc_stow == 130:
            print(json.dumps({"status": "interrupted", "actuations": actuations, "error": "interrupted"}), flush=True)
            return 130
        if rc_stow != 0:
            print(json.dumps({"status": "stow_failed", "hand": hand, "actuations": actuations,
                              "error": f"start_hold exited {rc_stow}"}), flush=True)
            return 1

        interrupted.wait(0.20)
        if not monitor.fresh() or not holding(monitor.frame[hand]):
            print(json.dumps({"status": "dropped_in_stow", "hand": hand, "actuations": actuations,
                              "error": "object dropped during start_hold"}), flush=True)
            return 1

        print(f"[takeover] Grasp, {LIFT_CM:g} cm lift, and start_hold stow succeeded on {hand} arm.", flush=True)
        print(json.dumps({"status": "succeeded", "hand": hand, "actuations": actuations, "error": None}), flush=True)
        return 0
    finally:
        if grasp_proc is not None and grasp_proc.poll() is None:
            stop_container_proc("guarded_grasp.py")
            stop_process(grasp_proc)
        monitor.close()


if __name__ == "__main__":
    sys.exit(main())
