#!/usr/bin/env python3
"""Read-only: print one JSON line with an arm's joint state, EE pose and gripper position.
usage (inside openpi-xpu-client): python3 read_arm_now.py right"""
import json, sys, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

arm = sys.argv[1] if len(sys.argv) > 1 else "right"
rclpy.init(); n = Node(f"read_{arm}_arm_now"); out = {}
n.create_subscription(JointState, f"/{arm}_arm/joint_states", lambda m: out.setdefault("q", [list(m.name), list(m.position)]), 10)
n.create_subscription(Pose, f"/{arm}_arm/current_ee_pose", lambda m: out.setdefault(
    "ee", [m.position.x, m.position.y, m.position.z, m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w]), 10)
n.create_subscription(JointState, f"/{arm}_gripper/joint_states", lambda m: out.setdefault("g", list(m.position)), 10)
t = time.time()
while not ("q" in out and "ee" in out) and time.time() - t < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
tg = time.time()
while "q" in out and "ee" in out and "g" not in out and time.time() - tg < 0.5:
    rclpy.spin_once(n, timeout_sec=0.05)
n.destroy_node(); rclpy.shutdown()
if "q" not in out or "ee" not in out:
    print(f"no fresh {arm} arm state"); sys.exit(1)
print(json.dumps(out))
