"""Exercise the real ROS-facing monitor and new left gripper actions on fakeros."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArmTools(unittest.TestCase):
    def env(self, scenario):
        return {**os.environ, "PYTHONPATH": str(ROOT / "tests/fakeros"), "SC": json.dumps(scenario)}

    def test_left_open_and_close_leave_right_unchanged(self):
        mode = {"meta_mode": 1, "enable_left_arm": True, "enable_right_arm": True,
                "left_arm_control_mode": 1, "right_arm_control_mode": 1}
        for action, position in (("open_left", 1.05), ("close_left", .2)):
            with self.subTest(action=action):
                result = subprocess.run([sys.executable, str(ROOT / "arm/arm_pose.py"), action],
                                        env=self.env({"mode": mode}), capture_output=True, text=True, timeout=12)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                lines = [line for line in result.stdout.splitlines() if line.startswith("Grippers now")]
                self.assertIn(f"'left': {position}", lines[-1])
                self.assertIn("'right': 0.6", lines[-1])
                self.assertIn("other gripper untouched", result.stdout)

    def test_real_monitor_fresh_then_lost(self):
        p = subprocess.Popen([sys.executable, "-u", str(ROOT / "arm/grasp_monitor.py")],
                             env=self.env({"lost_after": .3}), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        frames = []
        try:
            for _ in range(8):
                ready, _, _ = select.select([p.stdout], [], [], 3)
                self.assertTrue(ready, "monitor output timeout")
                frames.append(json.loads(p.stdout.readline()))
        finally:
            p.send_signal(signal.SIGINT)
            _, stderr = p.communicate(timeout=5)
        self.assertEqual(p.returncode, 0, stderr)
        self.assertTrue(frames[0]["arms_fresh"])
        self.assertFalse(frames[-1]["arms_fresh"])
        for side in ("left", "right"):
            self.assertEqual(frames[0][side]["grip_pos"], .6)
            self.assertEqual(frames[0][side]["grip_eff"], -2.5)
            self.assertEqual(frames[0][side]["ee_z"], -.3)
        intervals = [b["t"] - a["t"] for a, b in zip(frames, frames[1:])]
        self.assertTrue(all(.08 <= dt <= .2 for dt in intervals), intervals)


if __name__ == "__main__":
    unittest.main()
