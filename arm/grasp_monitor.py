#!/usr/bin/env python3
"""Read-only ROS grasp telemetry; run with python3 -u inside the client container."""
import json
import math
import time

FRESH_S = 0.3
SIDES = ("left", "right")


class Feedback:
    """Track each topic separately: fresh joints cannot mask a stale gripper or EE."""
    def __init__(self):
        self.values = {s: {"grip_pos": None, "grip_eff": None, "ee_z": None, "arm_max_eff": None} for s in SIDES}
        self.times = {s: {} for s in SIDES}
        self.fsm = None

    def gripper(self, side, msg, now):
        if msg.position and msg.effort and all(math.isfinite(v) for v in (msg.position[0], msg.effort[0])):
            self.values[side].update(grip_pos=float(msg.position[0]), grip_eff=float(msg.effort[0]))
            self.times[side]["gripper"] = now

    def ee(self, side, msg, now):
        if math.isfinite(msg.position.z):
            self.values[side]["ee_z"] = float(msg.position.z)
            self.times[side]["ee"] = now

    def joints(self, side, msg, now):
        if len(msg.position) >= 7 and all(math.isfinite(v) for v in msg.position[:7]):
            self.times[side]["joints"] = now
            eff = getattr(msg, "effort", None)
            if eff and len(eff) >= 7 and all(math.isfinite(v) for v in eff[:7]):
                self.values[side]["arm_max_eff"] = float(max(abs(v) for v in eff[:7]))

    def snapshot(self, now):
        out = {"t": now, "fsm": self.fsm}
        for side in SIDES:
            ts = self.times[side]
            out[side] = {**self.values[side], "fresh": all(now - ts.get(k, -1e9) <= FRESH_S
                                                        for k in ("gripper", "ee", "joints"))}
        out["arms_fresh"] = all(out[s]["fresh"] for s in SIDES)
        return out


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Pose
    from std_msgs.msg import Int32

    rclpy.init()
    node = Node("ta2_grasp_monitor")
    feedback = Feedback()
    for side in SIDES:
        node.create_subscription(JointState, f"/{side}_gripper/joint_states",
                                 lambda m, s=side: feedback.gripper(s, m, time.monotonic()), 10)
        node.create_subscription(Pose, f"/{side}_arm/current_ee_pose",
                                 lambda m, s=side: feedback.ee(s, m, time.monotonic()), 10)
        node.create_subscription(JointState, f"/{side}_arm/joint_states",
                                 lambda m, s=side: feedback.joints(s, m, time.monotonic()), 10)
    node.create_subscription(Int32, "/fsm_state", lambda m: setattr(feedback, "fsm", int(m.data)), 10)
    try:
        next_emit = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            if now >= next_emit:
                print(json.dumps(feedback.snapshot(now), allow_nan=False), flush=True)
                next_emit = now + 0.1
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
