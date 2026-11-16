#!/usr/bin/env python3
"""Offline subprocess fixtures for the mission HTTP integration tests."""
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(os.environ["FAKE_ROOT"])
SCENARIO = os.environ.get("MISSION_SCENARIO", "success_left")


def read():
    try:
        return json.loads((ROOT / "state.json").read_text())
    except FileNotFoundError:
        return {"attempt": 0, "stage": "idle"}


def write(data):
    tmp = ROOT / "state.tmp"
    tmp.write_text(json.dumps(data))
    tmp.replace(ROOT / "state.json")


def event(kind, argv):
    with (ROOT / "commands.jsonl").open("a") as f:
        f.write(json.dumps({"kind": kind, "argv": argv, "t": time.monotonic()}) + "\n")


def main():
    kind, args = sys.argv[1], sys.argv[2:]
    if kind == "dex":
        event("dex", args)
        if SCENARIO.startswith("api_") and args[0] == "pkill" and "grasp_monitor.py" in args[-1]:
            for pidfile in ROOT.glob("monitor-*.pid"):
                try:
                    os.kill(int(pidfile.read_text()), signal.SIGINT)
                except ProcessLookupError:
                    pass
        sys.exit(1 if args[0] == "pgrep" else 0)
    if kind == "monitor":
        (ROOT / f"monitor-{os.getpid()}.pid").write_text(str(os.getpid()))
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        while True:
            if (ROOT / "monitor_exit").exists():
                return
            if (ROOT / "monitor_silent").exists():
                time.sleep(0.02)
                continue
            st = read()
            now = time.monotonic()
            fresh = SCENARIO != "stale" and not (ROOT / "stale").exists()
            frame = {"t": now, "fsm": 2, "arms_fresh": fresh}
            for side in ("left", "right"):
                frame[side] = {"grip_pos": 1.05, "grip_eff": 0.2, "ee_z": -0.30, "fresh": fresh}
            side = "right" if SCENARIO in ("success_right", "api_success_right", "holding_right") else "left"
            success_at = 3 if SCENARIO == "success_left" else 1
            loaded = (st["attempt"] >= success_at and SCENARIO not in ("all_fail", "api_no_grasp") and
                      st["stage"] not in ("idle", "start", "open_left", "open_right"))
            if loaded and now - st.get("began", now) > 0.10:
                frame[side].update(grip_pos=0.39, grip_eff=-2.5,
                                   ee_z=-0.30 if now - st["began"] < 0.30 else -0.23)
            if SCENARIO == "empty_fast" and st["stage"] == "idle":
                frame["left"].update(grip_pos=-0.06, grip_eff=-1.71)        # left closed (empty) before the mission
            if SCENARIO == "empty_fast" and st["stage"] == "vla":
                # leftover closed-empty gripper until the reset opens it (1.2 s), open 0.3 s, then closed on nothing
                age = now - st.get("began", now)
                if age < 1.2 or age > 1.5:
                    frame["left"].update(grip_pos=-0.06, grip_eff=-1.71)
                else:
                    frame["left"].update(grip_pos=1.05, grip_eff=0.2, ee_z=-0.30)
            if SCENARIO == "holding_right" and st["stage"] != "open_right":
                frame["right"].update(grip_pos=0.39, grip_eff=-2.5)          # holding the bag until released
            if SCENARIO == "occupied":
                frame["right"].update(grip_pos=0.4, grip_eff=-2.0)
            if (ROOT / "drop").exists():
                frame[side].update(grip_pos=-0.06, grip_eff=-1.71)
            print(json.dumps(frame), flush=True)
            time.sleep(0.05)
    else:
        event(kind, args)
        st = read()
        if kind == "vla":
            st.update(attempt=st["attempt"] + 1, stage="vla", began=time.monotonic())
            write(st)
            if SCENARIO.startswith("api_"):
                def api_interrupted(*_):
                    event("vla_interrupted", [])
                    print(json.dumps({"status": "stopped", "actuations": 8, "error": "interrupted"}), flush=True)
                    sys.exit(130)
                signal.signal(signal.SIGTERM, api_interrupted)
                signal.signal(signal.SIGINT, api_interrupted)
                time.sleep(float(args[1]))
                print(json.dumps({"status": "time_limit", "actuations": 12, "error": None}), flush=True)
                return
            event("vla_env", [os.environ["PROMPT"], os.environ["PORT"], os.environ["RESET"]])
            for line in ("Arms: start. Checking robot state...", "INFO Starting GStreamer pipeline: udpsrc port=8890",
                         '{"output_dir": "/outputs/x_probe", "observation_ready": true, "exit_code": 0}'):
                print(line, flush=True)
            success_at = 3 if SCENARIO == "success_left" else 1
            if SCENARIO == "empty_fast":            # runs to its time limit unless the mission stops it
                def stopped(*_):
                    event("vla_interrupted", [])
                    sys.exit(130)
                signal.signal(signal.SIGINT, stopped)
                time.sleep(float(args[1]))
                return
            if SCENARIO == "all_fail" or st["attempt"] < success_at:
                time.sleep(0.15)
                sys.exit(7 if st["attempt"] == 1 else 0)
            def interrupted(*_):
                event("vla_interrupted", [])
                sys.exit(130)
            signal.signal(signal.SIGINT, interrupted)
            time.sleep(float(args[1]))
        elif kind == "arm":
            st["stage"] = args[0]
            write(st)
            time.sleep(0.15)
            if SCENARIO == "arm_fail" and args[0] == "start_hold":
                sys.exit(4)
        elif kind == "down":
            while (ROOT / "down_hold").exists():
                time.sleep(.02)
            for i in range(10):
                print(f"descent output {i}", flush=True)
            time.sleep(0.1)
            if SCENARIO == "down_fail":
                sys.exit(6)
        print(json.dumps({"fake": kind, "done": True}), flush=True)


if __name__ == "__main__":
    main()
