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
        sys.exit(1 if args[0] == "pgrep" else 0)
    if kind == "monitor":
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
            side = "right" if SCENARIO == "success_right" else "left"
            success_at = 3 if SCENARIO == "success_left" else 1
            loaded = (st["attempt"] >= success_at and SCENARIO != "all_fail" and
                      st["stage"] not in ("idle", "start", "open_left", "open_right"))
            if loaded and now - st.get("began", now) > 0.10:
                frame[side].update(grip_pos=0.39, grip_eff=-2.5,
                                   ee_z=-0.30 if now - st["began"] < 0.30 else -0.23)
            if SCENARIO == "occupied":
                frame["right"].update(grip_pos=0.4, grip_eff=-2.0)
            if (ROOT / "drop").exists():
                frame[side].update(grip_pos=0.02, grip_eff=-2.0)
            print(json.dumps(frame), flush=True)
            time.sleep(0.05)
    else:
        event(kind, args)
        st = read()
        if kind == "vla":
            st.update(attempt=st["attempt"] + 1, stage="vla", began=time.monotonic())
            write(st)
            event("vla_env", [os.environ["PROMPT"], os.environ["PORT"], os.environ["RESET"]])
            success_at = 3 if SCENARIO == "success_left" else 1
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
            time.sleep(0.1)
            if SCENARIO == "down_fail":
                sys.exit(6)
        print(json.dumps({"fake": kind, "done": True}), flush=True)


if __name__ == "__main__":
    main()
