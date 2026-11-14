#!/usr/bin/env python3
"""TA2 chassis test: drive at a constant velocity, stop, and log the three chassis motors.

Usage:
  python3 chassis_test.py VX VY WZ [--dur SEC] [--stop zero|cut] [--vmax M/S] [--skip-mode-check]
  --stop zero : send an all-zero velocity at the end (normal stop)
  --stop cut  : just stop publishing and watch the chassis stop by itself (doc: after ~1 s)

Measured axes on the real robot (2026-11-14): +vx = right, +vy = forward, +wz = CCW (seen from above).
The developer doc says x forward / y left, which is wrong for this robot.

Interfaces (developer doc 4.1 / 4.5 / 4.8 / 4.11.2 / 6.5):
  publish   /api/chassis/velocity  std_msgs/Float32MultiArray  data=[vx, vy, wz]
  subscribe /chassis/joint_states  sensor_msgs/JointState      chassis_joint1-3, 200 Hz
  subscribe /api/current_mode      std_msgs/String (JSON)      meta_mode / enable_chassis

Safety:
  - limits: planar speed <= 0.15 m/s (--vmax up to 0.3), |wz| <= 0.3 rad/s, duration <= 12 s
  - before moving: API mode + chassis enabled, wheel feedback fresh (< 0.2 s) and steady
  - wheel feedback lost for > 0.3 s while moving: stop publishing motion and send zero
  - Ctrl+C / kill / SSH hangup: send zero while the ROS context is still valid, then exit
"""
import argparse
import json
import math
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:          # older rclpy without this option
    SignalHandlerOptions = None

VMAX, WMAX, DMAX = 0.15, 0.3, 12.0   # test safety limits
RATE = 20.0                          # command publish rate, Hz
NAMES = ("chassis_joint1", "chassis_joint2", "chassis_joint3")
COUNTDOWN = 3.0      # countdown: listen only, also measures noise at rest
MIN_FRAMES = 20      # minimum wheel frames during the countdown
FRESH_START = 0.2    # newest wheel frame must be younger than this before moving
FRESH_RUN = 0.3      # abort if wheel feedback stalls this long while moving
STILL_HOLD = 0.15    # all wheels below threshold this long = stopped
OBSERVE = 3.0        # keep watching this long after stopping
CUT_FALLBACK = 1.8   # cut mode: send zero if not slowing down this long after the last command


class Abort(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vx", type=float)
    ap.add_argument("vy", type=float)
    ap.add_argument("wz", type=float)
    ap.add_argument("--dur", type=float, default=2.0)
    ap.add_argument("--stop", choices=["zero", "cut"], default="zero")
    ap.add_argument("--vmax", type=float, default=VMAX,
                    help=f"planar speed limit, default {VMAX}, capped at 0.3 m/s")
    ap.add_argument("--countdown", type=float, default=COUNTDOWN,
                    help=f"listen-only wait before moving, default {COUNTDOWN} s, minimum 1 s")
    ap.add_argument("--observe", type=float, default=OBSERVE,
                    help=f"keep watching after stopping, default {OBSERVE} s, minimum 1 s")
    ap.add_argument("--skip-mode-check", action="store_true",
                    help="continue without /api/current_mode (not recommended)")
    a = ap.parse_args()
    cmd = [a.vx, a.vy, a.wz]
    vmax = min(a.vmax, 0.3)
    countdown = max(1.0, a.countdown)
    observe = max(1.0, a.observe)
    if (not all(math.isfinite(x) for x in cmd + [a.dur]) or math.hypot(a.vx, a.vy) > vmax
            or abs(a.wz) > WMAX or not 0 < a.dur <= DMAX):
        sys.exit(f"Refused: planar speed <= {vmax} m/s, |wz| <= {WMAX} rad/s, duration 0-{DMAX} s")

    # Ctrl+C only raises KeyboardInterrupt; don't let rclpy shut the context first (zero would not get out)
    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (TypeError, AttributeError):   # rclpy without this option
        rclpy.init()

    def to_interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, to_interrupt)
    signal.signal(signal.SIGHUP, to_interrupt)

    n = Node("chassis_test")
    pub = n.create_publisher(Float32MultiArray, "/api/chassis/velocity", 10)
    log = []                 # (t, [w1,w2,w3], [p1,p2,p3]), valid frames only
    st = {"last_fb": -1e9, "bad": 0, "mode": None, "last_nonzero": None}

    def on_js(m):
        t = time.monotonic()
        names = list(m.name)
        if not all(x in names for x in NAMES):
            st["bad"] += 1
            return
        idx = [names.index(x) for x in NAMES]
        if len(m.velocity) <= max(idx) or len(m.position) <= max(idx):
            st["bad"] += 1
            return
        w = [float(m.velocity[i]) for i in idx]
        p = [float(m.position[i]) for i in idx]
        if not all(math.isfinite(x) for x in w + p):
            st["bad"] += 1
            return
        log.append((t, w, p))
        st["last_fb"] = t

    def on_mode(m):
        try:
            st["mode"] = json.loads(m.data)
        except (ValueError, TypeError):
            st["mode"] = {"_raw": m.data}

    n.create_subscription(JointState, "/chassis/joint_states", on_js, 10)
    n.create_subscription(String, "/api/current_mode", on_mode, 10)

    def send(v):
        pub.publish(Float32MultiArray(data=[float(x) for x in v]))
        if any(x != 0.0 for x in v):
            st["last_nonzero"] = time.monotonic()

    def spin(sec, v=None, watchdog=False, stop_when=None):
        """Run for sec seconds, publishing v at RATE (None = publish nothing); with watchdog, abort on stalled feedback."""
        end, nxt = time.monotonic() + sec, 0.0
        while time.monotonic() < end:
            now = time.monotonic()
            if watchdog and now - st["last_fb"] > FRESH_RUN:
                raise Abort(f"wheel feedback stalled for more than {FRESH_RUN} s while moving")
            if stop_when is not None and stop_when():
                return
            if v is not None and now >= nxt:
                send(v)
                nxt = now + 1.0 / RATE
            rclpy.spin_once(n, timeout_sec=0.005)

    def speed(w):
        return max(abs(x) for x in w)

    def smooth(rows, win=0.1):
        """Replace each frame by the mean wheel speed over the preceding win seconds to cancel noise. Returns [(t, w_mean)]."""
        out, j, acc = [], 0, [0.0, 0.0, 0.0]
        for k, (t, w, _) in enumerate(rows):
            acc = [acc[i] + w[i] for i in range(3)]
            while rows[j][0] < t - win:
                acc = [acc[i] - rows[j][1][i] for i in range(3)]
                j += 1
            m = k - j + 1
            out.append((t, [a / m for a in acc]))
        return out

    exit_code = 0
    try:
        # 1. Self-check: listen only during the countdown and measure noise at rest
        print(f"Command {cmd} for {a.dur} s, stop mode {a.stop}. Starting in {countdown:.0f} s, Ctrl+C to cancel...", flush=True)
        spin(countdown)
        if st["mode"] is None:
            spin(2.0, stop_when=lambda: st["mode"] is not None)   # 1 Hz topic, wait a bit longer
        frames = len(log)
        age = time.monotonic() - st["last_fb"]
        if frames < MIN_FRAMES or age > FRESH_START:
            if frames == 0 and st["bad"]:
                raise Abort(f"got {st['bad']} wheel messages but none had chassis_joint1-3 with complete data. "
                            "Check with: ros2 topic echo --once /chassis/joint_states")
            last = "none received" if frames == 0 else f"newest {age:.2f} s ago"
            raise Abort(f"wheel feedback not OK: {frames} frames in {countdown:.0f} s, {last}. "
                        "Check that the zenoh bridge is running and ROS_DOMAIN_ID=29")
        if st["bad"]:
            print(f"Note: ignored {st['bad']} malformed wheel messages")
        if pub.get_subscription_count() == 0:
            raise Abort("/api/chassis/velocity has no subscriber: check the bridge, API mode and chassis enable")
        mode = st["mode"]
        if mode is None:
            if not a.skip_mode_check:
                raise Abort("no /api/current_mode, cannot confirm API mode and chassis enable (use --skip-mode-check if you are sure)")
            print("Note: no /api/current_mode, continuing because of --skip-mode-check")
        else:
            print("Current mode:", mode)
            if mode.get("meta_mode") != 1 or mode.get("enable_chassis") is not True:
                raise Abort("robot not in API mode or chassis disabled (need meta_mode=1, enable_chassis=true). "
                            "Fix in the dev config: Save config -> Apply -> reboot the robot")
        # Raw wheel-speed noise is large (peaks ~0.3 rad/s measured): judge motion by position, stopping by 0.1 s means
        drift = max(abs(log[-1][2][i] - log[0][2][i]) for i in range(3))
        if drift > 0.05:
            raise Abort(f"motors turned {drift:.3f} rad during the countdown: the chassis is moving or another program controls it. Stop other controllers first")
        noise = max((speed(w) for t, w in smooth(log) if t >= log[0][0] + 0.1), default=0.0)
        thr = max(0.05, 2.0 * noise)
        print(f"Noise at rest (0.1 s mean) max {noise:.3f} rad/s, stop threshold {thr:.3f} rad/s")

        # 2. Move
        t0 = time.monotonic()
        print(">> moving", flush=True)
        spin(a.dur, cmd, watchdog=True)
        t_cmd_end = st["last_nonzero"] or time.monotonic()
        print(f"[] stop: last non-zero command at t={t_cmd_end - t0:.2f} s ({a.stop})", flush=True)

        # 3. Stop and keep watching
        fallback = False
        if a.stop == "zero":
            spin(observe, [0.0, 0.0, 0.0])
        else:
            spin(max(0.0, t_cmd_end + CUT_FALLBACK - time.monotonic()))
            steady_w = [speed(w) for t, w in smooth(log) if t0 + min(1.0, a.dur * 0.5) <= t <= t_cmd_end]
            sm = smooth(log)
            recent = speed(sm[-1][1]) if sm else 0.0
            if steady_w and recent > 0.7 * (sum(steady_w) / len(steady_w)):
                fallback = True
                print(f"! still not slowing down {CUT_FALLBACK} s after the last command, sending zero", flush=True)
                spin(max(0.0, t_cmd_end + observe - time.monotonic()), [0.0, 0.0, 0.0])
            else:
                spin(max(0.0, t_cmd_end + observe - time.monotonic()))
        t_end = time.monotonic()

        # 4. Report
        steady_from = t0 + min(1.0, a.dur * 0.5)
        steady = [w for t, w, _ in log if steady_from <= t <= t_cmd_end]
        if steady:
            mean = [sum(w[i] for w in steady) / len(steady) for i in range(3)]
            print("Steady-state mean motor speeds (rad/s):", [round(x, 3) for x in mean])

        seg = [(t, w, p) for t, w, p in log if t0 <= t <= t_end]
        ang = [0.0, 0.0, 0.0]
        unwrapped = [0.0, 0.0, 0.0]
        for (ta, wa, pa), (tb, wb, pb) in zip(seg, seg[1:]):
            for i in range(3):
                ang[i] += 0.5 * (wa[i] + wb[i]) * (tb - ta)
                d = pb[i] - pa[i]
                unwrapped[i] += (d + math.pi) % (2 * math.pi) - math.pi
        print("Total motor angles, integrated speed (rad):", [round(x, 2) for x in ang])
        print("Total motor angles, unwrapped position (rad):", [round(x, 2) for x in unwrapped])

        after = [(t, w) for t, w in smooth(log) if t >= t_cmd_end]
        gaps = [b[0] - a_[0] for a_, b in zip(after, after[1:])]
        if not after or (t_end - after[-1][0]) > FRESH_RUN or (gaps and max(gaps) > FRESH_RUN):
            print("Stop result: wheel feedback lost while stopping, cannot tell whether it stopped")
            exit_code = 2
        else:
            stopped_at = None
            run_start = None
            for t, w in after:
                if speed(w) < thr:
                    run_start = t if run_start is None else run_start
                    if t - run_start >= STILL_HOLD:
                        stopped_at = run_start
                        break
                else:
                    run_start = None
            if stopped_at is None:
                print(f"Stop result: not stopped within {observe:.0f} s")
                exit_code = 2
            else:
                print(f"Stop result: stopped {stopped_at - t_cmd_end:.2f} s after the last non-zero command")
        if fallback:
            print("Note: in cut mode the chassis did not slow down by itself after ~1 s - record this")
    except Abort as e:
        print(f"Aborted: {e}", flush=True)
        exit_code = 1
    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
        exit_code = 130
    finally:
        sent = 0
        for _ in range(5):
            try:
                pub.publish(Float32MultiArray(data=[0.0, 0.0, 0.0]))
                sent += 1
            except Exception:
                pass
            time.sleep(0.05)
        if sent == 0:
            print("Warning: zero velocity could not be sent; the chassis stops by itself ~1 s after commands stop")
        try:
            n.destroy_node()
        except Exception:
            pass
        (getattr(rclpy, "try_shutdown", None) or rclpy.shutdown)()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
