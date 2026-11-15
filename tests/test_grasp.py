import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission import grasp_update, holding, options, overload_update
from arm.grasp_monitor import Feedback


def sample(pos=0.39, eff=-2.5, z=-0.39, fresh=True, arm_max_eff=8.0):
    return dict(grip_pos=pos, grip_eff=eff, ee_z=z, fresh=fresh, arm_max_eff=arm_max_eff)


class GraspTests(unittest.TestCase):
    def sequence(self, frames):
        state = None
        for t, f in frames:
            state = grasp_update(state, f, t)
        return state

    def test_open(self):
        self.assertFalse(holding(sample(pos=1.05, eff=0.2)))
        self.assertFalse(self.sequence([(i / 10, sample(pos=1.05)) for i in range(20)])["success"])

    def test_empty_closed_and_closing_sweep(self):
        self.assertFalse(holding(sample(pos=0.02)))
        # Fast empty-closing sweep from 0.75 rad down to -0.05 rad never triggers
        sweep = [(i * 0.05, sample(pos=0.75 - i * 0.12)) for i in range(8)]
        for t, f in sweep:
            st = grasp_update(st if t > 0 else None, f, t)
            self.assertFalse(st["success"])

    def test_tabletop_settled_hold_triggers_without_waiting_for_vla_lift(self):
        self.assertTrue(holding(sample()))
        # Staying at tabletop z=-0.39 with stable grip_pos=0.39 for 0.6s triggers takeover
        frames = [(i / 10, sample(z=-0.39)) for i in range(8)]
        result = self.sequence(frames)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["since"], 0.0)

    def test_soft_sandbag_compression_settles_and_triggers(self):
        # Compresses from 0.68 -> 0.50 -> 0.40 -> 0.39 and holds
        positions = [0.68, 0.50, 0.40, 0.39, 0.39, 0.38, 0.39, 0.39, 0.39]
        result = self.sequence([(i / 10, sample(pos=p)) for i, p in enumerate(positions)])
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["since"], 0.1)

    def test_short_hold_does_not_trigger(self):
        self.assertFalse(self.sequence([(i / 10, sample()) for i in range(6)])["success"])

    def test_jitter_breaks_persistence(self):
        frames = [(i / 10, sample(eff=0 if i == 4 else -2.5)) for i in range(9)]
        self.assertFalse(self.sequence(frames)["success"])

    def test_reopen_breaks_hold(self):
        result = self.sequence([(0, sample()), (.2, sample()), (.3, sample(pos=1.05)), (.5, sample())])
        self.assertFalse(result["success"])

    def test_stale_gap_and_nonmonotonic(self):
        for t, f in [(0.6, sample(fresh=False)), (2.0, sample()), (0.3, sample())]:
            frames = [(i / 10, sample()) for i in range(5)] + [(t, f)]
            self.assertFalse(self.sequence(frames)["success"])

    def test_invalid_measurements(self):
        for bad in (None, math.nan, math.inf, "0.3", True):
            self.assertFalse(holding(sample(pos=bad)))
            self.assertFalse(holding(sample(eff=bad)))
            self.assertFalse(grasp_update(None, sample(z=bad), 1)["success"])

    def test_boundaries(self):
        self.assertTrue(holding(sample(pos=.06, eff=-1)))
        self.assertFalse(holding(sample(pos=.059, eff=-1)))
        self.assertTrue(holding(sample(pos=.75, eff=-1)))
        self.assertFalse(holding(sample(eff=-.99)))

    def test_overload_detector(self):
        st = None
        for i in range(9):
            st = overload_update(st, sample(arm_max_eff=21.5), i / 10)
            self.assertFalse(st["overloaded"])
        st = overload_update(st, sample(arm_max_eff=21.5), 1.0)
        self.assertTrue(st["overloaded"])
        st = overload_update(st, sample(arm_max_eff=10.0), 1.1)
        self.assertFalse(st["overloaded"])

    def test_options(self):
        self.assertEqual(options({"attempts": 3, "port": 8001})["attempts"], 3)
        for params in ({"attempts": True}, {"attempts": 6}, {"attempts": 0}, {"port": "8000"}, {"extra": 1}, {"prompt": "bad'prompt"}):
            with self.assertRaises(ValueError):
                options(params)

    def test_monitor_individual_topic_freshness_and_effort(self):
        f = Feedback()
        for side in ("left", "right"):
            f.joints(side, SimpleNamespace(position=[0.] * 7, effort=[-3., 12.5, 0., 0., 0., 0., 0.]), 1)
            f.gripper(side, SimpleNamespace(position=[.4], effort=[-2.]), 1)
            f.ee(side, SimpleNamespace(position=SimpleNamespace(z=-.2)), 1)
        snap = f.snapshot(1.1)
        self.assertTrue(snap["arms_fresh"])
        self.assertEqual(snap["left"]["arm_max_eff"], 12.5)
        for side in ("left", "right"):
            f.joints(side, SimpleNamespace(position=[0.] * 7), 1.4)
        self.assertFalse(f.snapshot(1.4)["arms_fresh"])
        self.assertFalse(f.snapshot(1.4)["left"]["fresh"])


if __name__ == "__main__":
    unittest.main()
