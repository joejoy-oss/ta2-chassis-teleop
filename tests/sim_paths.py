#!/usr/bin/env python3
"""Offline check: run teleop.py against the fake robot and follow paths/001-004.csv in order.

    python3 tests/sim_paths.py            # 0.15 m/s to keep it short, 3 s corner pause
Prints, for each path, how far the robot moved versus the recorded displacement.
"""
import csv, json, math, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
env = {**os.environ, "PYTHONPATH": os.path.join(HERE, "fakeros"),
       "SC": json.dumps({"noise": 0.08}), "TELEOP_SPEED": "0.15"}
p = subprocess.Popen([sys.executable, os.path.join(ROOT, "teleop.py")], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, text=True, env=env)
events, pose = [], [0.0, 0.0, 0.0]


def reader():
    for line in p.stdout:
        d = json.loads(line)
        if d["event"] == "pose":
            pose[:] = [d["x"], d["y"], d["th"]]
        else:
            events.append(d)


def send(c):
    p.stdin.write(json.dumps(c) + "\n")
    p.stdin.flush()


running = [True]


def heartbeat():
    while running[0]:
        try:
            send({"cmd": "alive"})
        except (ValueError, BrokenPipeError):
            return
        time.sleep(0.25)


threading.Thread(target=reader, daemon=True).start()
time.sleep(1.5)
threading.Thread(target=heartbeat, daemon=True).start()
ok = True
for name in ("001", "002", "003", "004"):
    with open(os.path.join(ROOT, "paths", f"{name}.csv")) as f:
        pts = [(float(r["x_m"]), float(r["y_m"]), math.radians(float(r["heading_deg"]))) for r in csv.DictReader(f)]
    start, k = list(pose), len(events)
    send({"cmd": "path", "name": name, "points": pts, "dwell": 3})
    while not any(e["event"] in ("idle", "error") for e in events[k:]):
        time.sleep(0.1)
    end = [e for e in events[k:] if e["event"] in ("idle", "error")][0]
    dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
    mx, my = pose[0] - start[0], pose[1] - start[1]
    err = math.hypot(mx - dx, my - dy)
    ok &= end.get("why") == "path done" and err < 0.05
    print(f"{name}: {end.get('why', end)}  moved ({mx:+.3f},{my:+.3f})  recorded ({dx:+.3f},{dy:+.3f})  error {err*100:.1f} cm")
running[0] = False
time.sleep(0.3)
p.stdin.close()
p.wait(timeout=5)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
