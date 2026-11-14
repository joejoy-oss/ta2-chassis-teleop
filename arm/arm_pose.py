#!/usr/bin/env python3
"""TA2 arms: move both arms to the official "raised" pose (same as zero.py), or back down.

Usage (inside the openpi-xpu-client container):
  python3 arm_pose.py raise [--time 8] [--dry-run]   both arms to the official zero.py pose
  python3 arm_pose.py lower [--time 8] [--dry-run]   back to the pose saved by the last "raise"

Interfaces (developer doc 4.1, 4.2, 4.11, 4.13):
  publish   /api/{left,right}_arm/joint_cmd  sensor_msgs/JointState  position = 7 joint angles (rad)
  publish   /api/fsm/enable                  std_msgs/Float32        1.0 heartbeat, 0.0 = pause
  subscribe /{left,right}_arm/joint_states, /fsm_state (Int32: 0 PAUSE, 1 SLOW_START, 2 READY),
            /api/current_mode (needs meta_mode 1, both arms enabled, both arm control modes 1 = joint)

Safety:
  - refuses unless API mode, both arms enabled and in joint mode, and joint feedback is fresh
  - first commands the current pose and waits for FSM READY, then moves with a smooth (min-jerk)
    profile; peak joint speed is kept under 0.35 rad/s by stretching the time if needed
  - targets are clipped to the joint limits of arm_config.yml / doc 4.13
  - if any joint lags its command by > 0.3 rad for 0.5 s (possible contact): hold where it is, pause
  - Ctrl+C / SIGTERM / SSH hangup: hold where it is, then publish fsm/enable 0.0 (PAUSE)
  - when done: hold 1 s, then fsm/enable 0.0 (PAUSE: zero-velocity hold, motors stay enabled)
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32, String

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None

ARMS = ("left", "right")
# official zero.py target pose (openpi feat/teleavatar-v2-client-b70, examples/teleavatar_v2/zero.py)
RAISED = {"left": [0.41, 1.16, -0.47, 0.90, 0.23, -0.15, 0.60],
          "right": [-0.50, -0.92, 0.52, -1.28, 0.32, 0.55, -0.52]}
LIMITS = {"left": ([-1.8, 0.0, -2.6, 0.25, -1.8, -1.4, -0.7], [1.8, 1.9, 1.3, 2.2, 1.8, 1.4, 0.7]),
          "right": ([-1.8, -1.9, -1.3, -2.2, -1.8, -1.4, -0.7], [1.8, 0.0, 2.6, -0.25, 1.8, 1.4, 0.7])}
SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_start_pose.json")
CMD_RATE = 100.0          # joint_cmd, Hz
HB_RATE = 20.0            # fsm/enable heartbeat, Hz
MAX_SPEED = 0.35          # rad/s peak joint speed
FRESH = 0.3               # s
LAG_LIMIT, LAG_TIME = 0.3, 0.5
DONE_TOL, DONE_HOLD = 0.03, 0.5
READY_TIMEOUT = 6.0
FSM = {0: "PAUSE", 1: "SLOW_START", 2: "READY"}


class Abort(Exception):
    pass


def clip(arm, q):
    lo, hi = LIMITS[arm]
    return [min(max(v, lo[i]), hi[i]) for i, v in enumerate(q)]


def minjerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["raise", "lower"])
    ap.add_argument("--time", type=float, default=8.0, help="motion time in seconds (stretched if too fast)")
    ap.add_argument("--dry-run", action="store_true", help="check and print the plan, send nothing")
    a = ap.parse_args()

    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (TypeError, AttributeError):
        rclpy.init()

    def to_interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, to_interrupt)
    signal.signal(signal.SIGHUP, to_interrupt)

    n = Node("ta2_arm_pose")
    st = {"q": {}, "names": {}, "t": {}, "fsm": None, "mode": None}
    pubs = {arm: n.create_publisher(JointState, f"/api/{arm}_arm/joint_cmd", 10) for arm in ARMS}
    hb = n.create_publisher(Float32, "/api/fsm/enable", 10)

    def on_js(arm):
        def cb(m):
            if len(m.position) >= 7 and all(math.isfinite(v) for v in m.position[:7]):
                st["q"][arm] = list(m.position[:7])
                st["names"][arm] = list(m.name[:7])
                st["t"][arm] = time.monotonic()
        return cb
    for arm in ARMS:
        n.create_subscription(JointState, f"/{arm}_arm/joint_states", on_js(arm), 10)
    n.create_subscription(Int32, "/fsm_state", lambda m: st.__setitem__("fsm", m.data), 10)

    def on_mode(m):
        try:
            st["mode"] = json.loads(m.data)
        except (ValueError, TypeError):
            st["mode"] = {}
    n.create_subscription(String, "/api/current_mode", on_mode, 10)

    sending = {"on": False}
    nxt = {"cmd": 0.0, "hb": 0.0}

    def publish(cmd):
        now = time.monotonic()
        if now >= nxt["hb"]:
            hb.publish(Float32(data=1.0))
            nxt["hb"] = now + 1.0 / HB_RATE
        for arm in ARMS:
            m = JointState()
            m.header.stamp = n.get_clock().now().to_msg()
            m.name = st["names"][arm]
            m.position = clip(arm, cmd[arm])
            m.velocity = [0.0] * 7
            m.effort = [0.0] * 7
            pubs[arm].publish(m)

    def spin_for(sec, cmd_fn=None, check=None):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(n, timeout_sec=0.002)
            now = time.monotonic()
            for arm in ARMS:
                if sending["on"] and now - st["t"].get(arm, -1e9) > FRESH:
                    raise Abort(f"{arm} arm joint feedback stalled")
            if cmd_fn is not None and now >= nxt["cmd"]:
                publish(cmd_fn(now))
                nxt["cmd"] = now + 1.0 / CMD_RATE
            if check is not None and check(now):
                return True
        return False

    exit_code = 0
    hold = None
    try:
        print(f"Arms: {a.action}. Checking robot state...", flush=True)
        spin_for(2.5, check=lambda now: st["mode"] is not None and len(st["q"]) == 2)
        if st["mode"] is None:
            raise Abort("no /api/current_mode: is the zenoh bridge running?")
        m = st["mode"]
        print("Current mode:", m)
        problems = []
        if m.get("meta_mode") != 1:
            problems.append("run mode is not API")
        for arm in ARMS:
            if m.get(f"enable_{arm}_arm") is not True:
                problems.append(f"{arm} arm not enabled")
            if m.get(f"{arm}_arm_control_mode") != 1:
                problems.append(f"{arm} arm control mode is not joint (it is end-effector)")
        if problems:
            raise Abort("; ".join(problems) + ". Fix in the dev config: run mode API, both arms enabled "
                        "and set to joint -> Save config -> Apply -> reboot the robot")
        for arm in ARMS:
            if time.monotonic() - st["t"].get(arm, -1e9) > FRESH:
                raise Abort(f"no fresh /{arm}_arm/joint_states")
        start = {arm: list(st["q"][arm]) for arm in ARMS}

        if a.action == "raise":
            target = {arm: clip(arm, RAISED[arm]) for arm in ARMS}
        else:
            try:
                with open(SAVE) as f:
                    saved = json.load(f)
                target = {arm: clip(arm, saved[arm]) for arm in ARMS}
            except (OSError, ValueError, KeyError):
                raise Abort(f"no saved start pose ({SAVE}); run 'raise' first")
        delta = max(abs(target[arm][i] - start[arm][i]) for arm in ARMS for i in range(7))
        dur = max(a.time, 1.875 * delta / MAX_SPEED, 2.0)    # min-jerk peak speed = 1.875 * delta / T
        print(f"FSM state: {FSM.get(st['fsm'], st['fsm'])}")
        for arm in ARMS:
            print(f"{arm:5s} now   ", [round(v, 2) for v in start[arm]])
            print(f"{arm:5s} target", [round(v, 2) for v in target[arm]])
        print(f"Largest joint change {delta:.2f} rad, motion time {dur:.1f} s, peak {1.875 * delta / dur:.2f} rad/s")
        if a.dry_run:
            print("Dry run: nothing sent.")
            return
        if a.action == "raise":
            with open(SAVE, "w") as f:
                json.dump(start, f)

        # 1. hold the current pose with heartbeat until READY
        sending["on"] = True
        hold = start
        print("Enabling (holding current pose) and waiting for READY...", flush=True)
        if not spin_for(READY_TIMEOUT, cmd_fn=lambda now: start, check=lambda now: st["fsm"] == 2):
            raise Abort(f"FSM did not reach READY within {READY_TIMEOUT:.0f} s "
                        f"(state {FSM.get(st['fsm'], st['fsm'])})")
        # 2. smooth move
        print(f">> moving ({dur:.1f} s). Ctrl+C holds and pauses.", flush=True)
        t0 = time.monotonic()
        lag_since = [None]
        cur_cmd = {}

        def traj(now):
            s = minjerk((now - t0) / dur)
            for arm in ARMS:
                cur_cmd[arm] = [start[arm][i] + (target[arm][i] - start[arm][i]) * s for i in range(7)]
            return cur_cmd

        def watch(now):
            if not cur_cmd:
                return False
            lag = max(abs(clip(arm, cur_cmd[arm])[i] - st["q"][arm][i]) for arm in ARMS for i in range(7))
            if lag > LAG_LIMIT:
                lag_since[0] = lag_since[0] or now
                if now - lag_since[0] > LAG_TIME:
                    raise Abort(f"joint lags its command by {lag:.2f} rad: possible contact, stopping")
            else:
                lag_since[0] = None
            hold.update({arm: list(st["q"][arm]) for arm in ARMS})
            if now - t0 >= dur:
                err = max(abs(target[arm][i] - st["q"][arm][i]) for arm in ARMS for i in range(7))
                return err < DONE_TOL
            return False
        hold = {arm: list(start[arm]) for arm in ARMS}
        done_at = [None]

        def finished(now):
            if watch(now):
                done_at[0] = done_at[0] or now
                return now - done_at[0] > DONE_HOLD
            done_at[0] = None
            return False
        if not spin_for(dur + 5.0, cmd_fn=traj, check=finished):
            err = max(abs(target[arm][i] - st["q"][arm][i]) for arm in ARMS for i in range(7))
            print(f"Note: not within {DONE_TOL} rad of the target after {dur + 5:.0f} s (max error {err:.3f} rad)")
            exit_code = 2
        else:
            print("Reached the target.")
        spin_for(1.0, cmd_fn=lambda now: target)
        hold = None
    except Abort as e:
        print(f"Aborted: {e}", flush=True)
        exit_code = 1
    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
        exit_code = 130
    finally:
        if sending["on"]:
            try:
                if hold is not None and len(st["q"]) == 2:
                    here = {arm: list(st["q"][arm]) for arm in ARMS}
                    end = time.monotonic() + 0.3
                    while time.monotonic() < end:
                        publish(here)
                        time.sleep(0.01)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: could not send the final hold ({e})")
            try:
                for _ in range(3):
                    hb.publish(Float32(data=0.0))
                    time.sleep(0.05)
                print("FSM paused (fsm/enable 0.0): arms hold with zero velocity.")
            except Exception as e:  # noqa: BLE001
                print(f"Warning: could not send the pause ({e}); FSM pauses ~1 s after heartbeats stop")
        try:
            n.destroy_node()
        except Exception:
            pass
        (getattr(rclpy, "try_shutdown", None) or rclpy.shutdown)()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
