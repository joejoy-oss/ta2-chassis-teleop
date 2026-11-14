#!/usr/bin/env python3
"""Long-running TA2 chassis teleop process, driven by web_move.py over stdin/stdout.

Runs inside the openpi-xpu-client container. Reads one JSON command per line on stdin:
  {"cmd": "step", "dir": "forward"}   move 10 cm (0.05 m/s for 2.04 s); "ccw"/"cw" turn 15 deg in place
  {"cmd": "hold", "dir": "forward"}   keep moving; must be repeated at least every 0.4 s
  {"cmd": "release"} / {"cmd": "stop"} stop now
  {"cmd": "reset_pose"}                set the odometry origin to the current pose
  {"cmd": "path", "name": "001", "points": [[x, y, th], ...], "dwell": 10}
                                       follow a recorded trail (metres / rad, the trail's own frame);
                                       its first point is anchored to the current pose, so the robot
                                       repeats the same relative motion from where it stands now;
                                       "dwell" = seconds to stand still at every corner (turn > 45 deg)
  {"cmd": "alive"}                     page heartbeat; a path stops if none arrives for 1 s
Writes one JSON status line per state change on stdout, plus {"event": "pose", ...} every 0.1 s
while the odometry changes (x = right, y = forward of the starting pose, metres; th = CCW, rad).
Odometry comes from the three motor positions through J_INV, measured on the real robot from
command -> motor speed ratios (not yet checked against a tape measure).

Measured axes (2026-11-14): +vx = right, +vy = forward, +wz = CCW.

Safety:
  - moves only when /api/current_mode says API mode + chassis enabled
  - hold is a dead-man switch: no heartbeat for 0.4 s -> stop
  - a single hold is capped at 10 s; release and press again to continue
  - wheel feedback older than 0.3 s while moving -> stop and report
  - stdin closed (web server gone), SIGINT, SIGTERM -> send zero and exit
"""
import json
import math
import os
import signal
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None

SPEED = float(os.environ.get("TELEOP_SPEED", "0.05"))   # m/s (env override for offline tests only)
ROT_SPEED = 0.2               # rad/s, in place
STEP_DUR = 0.10 / SPEED + 0.04  # 10 cm incl. acceleration/braking loss
# 15 deg: accel limit 0.3 rad/s^2 and decel 0.6 rad/s^2 lose 0.033 rad -> add 0.167 s
ROT_STEP_DUR = math.radians(15) / ROT_SPEED + 0.167
HOLD_TIMEOUT = 0.4            # dead-man: stop if no heartbeat for this long
HOLD_MAX = 10.0               # longest single hold
FRESH = 0.3                   # wheel feedback must be younger than this while moving
RATE = 20.0                   # publish rate while moving, Hz
ZERO_TAIL = 0.5               # keep publishing zero this long after stopping, then go quiet
DIRS = {"forward": (0.0, SPEED, 0.0), "back": (0.0, -SPEED, 0.0),
        "left": (-SPEED, 0.0, 0.0), "right": (SPEED, 0.0, 0.0),
        "ccw": (0.0, 0.0, ROT_SPEED), "cw": (0.0, 0.0, -ROT_SPEED)}
NAMES = ("chassis_joint1", "chassis_joint2", "chassis_joint3")
# motor angle deltas (rad) -> body motion [right m, forward m, CCW rad]; from 10 real runs on 2026-11-14
J_INV = ((-0.032365, -0.033224, 0.066149),
         (0.058231, -0.057545, 0.000625),
         (0.078921, 0.07905, 0.07994))
POSE_PERIOD = 0.1
# path following
MIN_SPEED = 0.03              # m/s floor while still away from a waypoint
WP_TOL = 0.04                 # m, pass an intermediate waypoint
END_TOL = 0.02                # m, final waypoint
HEAD_TOL = math.radians(2)
PATH_MAX_LEN = 6.0            # m
ALIVE_TIMEOUT = 1.0           # s without page heartbeat -> stop
STUCK_TIME = 5.0              # s without 1 cm of progress -> stop
CORNER_ANGLE = math.radians(45)   # a waypoint is a corner if the path turns more than this there


def angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def simplify(pts, eps=0.02):
    """Ramer-Douglas-Peucker on (x, y); keeps the heading of every kept point."""
    if len(pts) < 3:
        return list(pts)
    (x1, y1), (x2, y2) = pts[0][:2], pts[-1][:2]
    L = math.hypot(x2 - x1, y2 - y1)
    best, bi = -1.0, 0
    for i in range(1, len(pts) - 1):
        x, y = pts[i][:2]
        d = (abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / L) if L > 1e-9 else math.hypot(x - x1, y - y1)
        if d > best:
            best, bi = d, i
    if best > eps:
        return simplify(pts[:bi + 1], eps)[:-1] + simplify(pts[bi:], eps)
    return [pts[0], pts[-1]]


def emit(**kw):
    print(json.dumps(kw), flush=True)


class Teleop(Node):
    def __init__(self):
        super().__init__("ta2_web_teleop")
        self.pub = self.create_publisher(Float32MultiArray, "/api/chassis/velocity", 10)
        self.create_subscription(JointState, "/chassis/joint_states", self.on_js, 10)
        self.create_subscription(String, "/api/current_mode", self.on_mode, 10)
        self.lock = threading.Lock()
        self.last_fb = -1e9
        self.mode_ok = None
        self.kind = None          # None | "step" | "hold"
        self.dir = None
        self.until = 0.0          # step end time, or hold dead-man deadline
        self.hold_start = 0.0
        self.zero_until = 0.0
        self.next_pub = 0.0
        self.eof = False
        self.last_p = None
        self.x = self.y = self.th = 0.0
        self.next_pose = 0.0
        self.pose_sent = None
        self.path = None
        self.v_path = (0.0, 0.0, 0.0)
        self.alive_t = -1e9

    def on_js(self, m):
        names = list(m.name)
        if all(x in names for x in NAMES):
            idx = [names.index(x) for x in NAMES]
            if len(m.velocity) > max(idx) and all(math.isfinite(m.velocity[i]) for i in idx):
                self.last_fb = time.monotonic()
            if len(m.position) > max(idx) and all(math.isfinite(m.position[i]) for i in idx):
                self.odometry([m.position[i] for i in idx])

    def odometry(self, p):
        if self.last_p is not None:
            d = [((p[i] - self.last_p[i]) + math.pi) % (2 * math.pi) - math.pi for i in range(3)]
            right, fwd, dth = (sum(J_INV[r][c] * d[c] for c in range(3)) for r in range(3))
            mid = self.th + dth / 2
            self.x += right * math.cos(mid) - fwd * math.sin(mid)
            self.y += right * math.sin(mid) + fwd * math.cos(mid)
            self.th += dth
        self.last_p = p

    def on_mode(self, m):
        try:
            d = json.loads(m.data)
            ok = d.get("meta_mode") == 1 and d.get("enable_chassis") is True
        except (ValueError, TypeError, AttributeError):
            ok = False
        if ok != self.mode_ok:
            self.mode_ok = ok
            emit(event="mode", ok=ok)

    def command(self, c):
        now = time.monotonic()
        cmd, d = c.get("cmd"), c.get("dir")
        if cmd == "reset_pose":
            self.x = self.y = self.th = 0.0
            self.pose_sent = None
            emit(event="pose", x=0.0, y=0.0, th=0.0, reset=True)
            return
        if cmd == "alive":
            self.alive_t = now
            return
        with self.lock:
            if cmd == "path":
                self._start_path(now, c)
                return
            if cmd in ("stop", "release"):
                self._halt(now, "released" if cmd == "release" else "stopped")
                return
            if d not in DIRS:
                emit(event="error", msg=f"bad direction {d!r}")
                return
            if cmd == "hold" and self.kind == "hold" and self.dir == d:
                if now - self.hold_start < HOLD_MAX:
                    self.until = now + HOLD_TIMEOUT          # heartbeat
                return
            if self.kind is not None:
                return                                         # busy with another move
            if self.mode_ok is not True:
                emit(event="error", msg="not in API mode or chassis disabled (or mode unknown yet)")
                return
            if now - self.last_fb > FRESH:
                emit(event="error", msg="no fresh wheel feedback from the robot")
                return
            self.kind, self.dir = cmd, d
            if cmd == "step":
                self.until = now + (ROT_STEP_DUR if d in ("ccw", "cw") else STEP_DUR)
            elif cmd == "hold":
                self.hold_start = now
                self.until = now + HOLD_TIMEOUT
            else:
                self.kind = self.dir = None
                emit(event="error", msg=f"bad command {cmd!r}")
                return
            self.next_pub = 0.0
            emit(event="moving", kind=cmd, dir=d)

    def _start_path(self, now, c):
        name = str(c.get("name", "path"))[:40]
        if self.kind is not None:
            emit(event="error", msg="busy, path not started")
            return
        if self.mode_ok is not True:
            emit(event="error", msg="not in API mode or chassis disabled (or mode unknown yet)")
            return
        if now - self.last_fb > FRESH:
            emit(event="error", msg="no fresh wheel feedback from the robot")
            return
        try:
            pts = [(float(p[0]), float(p[1]), float(p[2])) for p in c.get("points", [])]
        except (TypeError, ValueError, IndexError):
            pts = []
        if len(pts) < 2 or len(pts) > 20000 or not all(math.isfinite(v) for p in pts for v in p):
            emit(event="error", msg=f"path {name}: bad or empty point list")
            return
        pts = simplify(pts)
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        if length > PATH_MAX_LEN:
            emit(event="error", msg=f"path {name}: {length:.2f} m is longer than {PATH_MAX_LEN} m")
            return
        # anchor the trail's first point to the current pose
        x0, y0, th0 = pts[0]
        rot = self.th - th0
        cr, sr = math.cos(rot), math.sin(rot)
        wps = []
        for px, py, pth in pts[1:]:
            dx, dy = px - x0, py - y0
            wps.append((self.x + dx * cr - dy * sr, self.y + dx * sr + dy * cr, pth + rot))
        try:
            dwell = max(0.0, min(60.0, float(c.get("dwell", 0))))
        except (TypeError, ValueError):
            dwell = 0.0
        legs = [(wps[0][0] - self.x, wps[0][1] - self.y)] + \
               [(b[0] - a[0], b[1] - a[1]) for a, b in zip(wps, wps[1:])]
        corner = []
        for i in range(len(wps)):                # corner at waypoint i: angle between leg i and leg i+1
            if i + 1 >= len(legs):
                corner.append(False)
                continue
            a1, a2 = math.atan2(legs[i][1], legs[i][0]), math.atan2(legs[i + 1][1], legs[i + 1][0])
            corner.append(abs(angdiff(a2, a1)) > CORNER_ANGLE)
        n_corners = sum(corner) if dwell > 0 else 0
        rest = [0.0] * len(wps)                 # path length still to go after waypoint i
        for i in range(len(wps) - 2, -1, -1):
            rest[i] = rest[i + 1] + math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
        self.path = {"wps": wps, "rest": rest, "i": 0, "best": 1e9, "best_t": now,
                     "corner": corner, "dwell": dwell, "dwell_until": None,
                     "deadline": now + 3 * length / SPEED + 10 + n_corners * dwell}
        self.kind, self.dir = "path", name
        self.alive_t = now
        self.next_pub = 0.0
        emit(event="moving", kind="path", dir=name, waypoints=len(wps), length=round(length, 3),
             corners=sum(corner), dwell=dwell)

    def _path_control(self, now):
        P = self.path
        if P["dwell_until"] is not None:          # standing still at a corner
            if now - self.alive_t > ALIVE_TIMEOUT:
                self._halt(now, "page closed or connection lost")
                return
            if now < P["dwell_until"]:
                P["best_t"] = now
                self.v_path = (0.0, 0.0, 0.0)
                return
            P["dwell_until"] = None
            P["i"] += 1
            P["best"], P["best_t"] = 1e9, now
            emit(event="path", name=self.dir, waypoint=P["i"], of=len(P["wps"]), resume=True)
        tx, ty, tth = P["wps"][P["i"]]
        ex, ey = tx - self.x, ty - self.y
        dist = math.hypot(ex, ey)
        last = P["i"] == len(P["wps"]) - 1
        eth = angdiff(tth, self.th)
        dwell_here = not last and P["corner"][P["i"]] and P["dwell"] > 0
        if dwell_here and dist < END_TOL:
            P["dwell_until"] = now + P["dwell"]
            self.v_path = (0.0, 0.0, 0.0)
            emit(event="path", name=self.dir, waypoint=P["i"] + 1, of=len(P["wps"]), dwell=P["dwell"])
            return
        if not last and not dwell_here and dist < WP_TOL:
            P["i"] += 1
            emit(event="path", name=self.dir, waypoint=P["i"], of=len(P["wps"]))
            return self._path_control(now)
        if last and dist < END_TOL and abs(eth) < HEAD_TOL:
            self._halt(now, "path done")
            return
        remaining = dist + P["rest"][P["i"]] + abs(eth) * 0.3
        if remaining < P["best"] - 0.01:
            P["best"], P["best_t"] = remaining, now
        if now - P["best_t"] > STUCK_TIME:
            self._halt(now, f"stuck: no progress for {STUCK_TIME:.0f} s")
        elif now > P["deadline"]:
            self._halt(now, "path timeout")
        elif now - self.alive_t > ALIVE_TIMEOUT:
            self._halt(now, "page closed or connection lost")
        if self.kind != "path":
            return
        speed = 0.0 if (last and dist < END_TOL) else min(SPEED, max(MIN_SPEED, 1.0 * dist))
        vxw, vyw = (ex / dist * speed, ey / dist * speed) if dist > 1e-6 else (0.0, 0.0)
        c, s = math.cos(self.th), math.sin(self.th)
        right, fwd = vxw * c + vyw * s, -vxw * s + vyw * c
        wz = max(-ROT_SPEED, min(ROT_SPEED, 1.5 * eth)) if abs(eth) > math.radians(1) else 0.0
        self.v_path = (right, fwd, wz)

    def _halt(self, now, why):
        if self.kind is not None:
            emit(event="idle", why=why, dir=self.dir, kind=self.kind)
        self.kind = self.dir = None
        self.path = None
        self.v_path = (0.0, 0.0, 0.0)
        self.zero_until = now + ZERO_TAIL
        self.next_pub = 0.0

    def tick(self):
        now = time.monotonic()
        if now >= self.next_pose:
            self.next_pose = now + POSE_PERIOD
            pose = (round(self.x, 4), round(self.y, 4), round(self.th, 4))
            if pose != self.pose_sent:
                self.pose_sent = pose
                emit(event="pose", x=pose[0], y=pose[1], th=pose[2])
        with self.lock:
            if self.kind is not None:
                if self.kind == "step" and now >= self.until:
                    self._halt(now, "done")
                elif self.kind == "hold" and now >= self.until:
                    self._halt(now, "released")
                elif self.kind == "hold" and now - self.hold_start >= HOLD_MAX:
                    self._halt(now, f"hold limit {HOLD_MAX:.0f} s")
                elif now - self.last_fb > FRESH:
                    self._halt(now, "wheel feedback lost")
                elif self.kind == "path":
                    self._path_control(now)
            if self.kind == "path":
                v = self.v_path
            else:
                v = DIRS[self.dir] if self.kind is not None else (0.0, 0.0, 0.0)
            moving_or_tail = self.kind is not None or now < self.zero_until
            if moving_or_tail and now >= self.next_pub:
                self.pub.publish(Float32MultiArray(data=list(v)))
                self.next_pub = now + 1.0 / RATE


def main():
    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (TypeError, AttributeError):
        rclpy.init()

    def to_interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, to_interrupt)
    signal.signal(signal.SIGHUP, to_interrupt)

    n = Teleop()

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                n.command(json.loads(line))
            except (ValueError, TypeError) as e:
                emit(event="error", msg=f"bad input: {e}")
        n.eof = True
    threading.Thread(target=reader, daemon=True).start()

    emit(event="ready")
    try:
        while not n.eof:
            rclpy.spin_once(n, timeout_sec=0.01)
            n.tick()
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            try:
                n.pub.publish(Float32MultiArray(data=[0.0, 0.0, 0.0]))
            except Exception:
                pass
            time.sleep(0.05)
        emit(event="exit")
        try:
            n.destroy_node()
        except Exception:
            pass
        (getattr(rclpy, "try_shutdown", None) or rclpy.shutdown)()


if __name__ == "__main__":
    main()
