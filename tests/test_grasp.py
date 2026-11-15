import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission import EMPTY_CLOSED, EMPTY_PERSIST, HOLD_MARGIN, NO_LIFT_S, closed_empty, grasp_update, holding, options
from arm.grasp_monitor import Feedback


def sample(pos=0.39, eff=-2.5, z=-0.3, fresh=True):
    return dict(grip_pos=pos, grip_eff=eff, ee_z=z, fresh=fresh)


class GraspTests(unittest.TestCase):
    def sequence(self, frames):
        state = None
        for t, f in frames:
            state = grasp_update(state, f, t)
        return state

    def test_open(self):
        self.assertFalse(holding(sample(pos=1.05, eff=0.2)))
        self.assertFalse(self.sequence([(i / 10, sample(pos=1.05)) for i in range(30)])["success"])

    def test_empty_closed(self):
        self.assertFalse(holding(sample(pos=-0.0485, eff=-1.71), "right"))
        self.assertFalse(self.sequence([(i / 10, sample(pos=-0.0485, eff=-1.71, z=-0.3 + i / 100)) for i in range(30)])["success"])

    def test_loaded_without_lift(self):
        self.assertTrue(holding(sample()))
        self.assertFalse(self.sequence([(i / 10, sample()) for i in range(30)])["success"])

    def test_lifted_persistent(self):
        frames = [(0, sample())] + [(i / 10, sample(z=-0.24)) for i in range(1, 18)]
        result = self.sequence(frames)
        self.assertTrue(result["success"])
        self.assertEqual(result["z_at_close"], -0.3)

    def test_short_lift(self):
        self.assertFalse(self.sequence([(0, sample())] + [(i / 10, sample(z=-0.24)) for i in range(1, 15)])["success"])

    def test_jitter_breaks_persistence(self):
        frames = [(0, sample())] + [(i / 10, sample(z=-0.24, eff=0 if i == 12 else -2.5)) for i in range(1, 26)]
        self.assertFalse(self.sequence(frames)["success"])

    def test_reopen_keeps_first_close_height_and_breaks_hold(self):
        result = self.sequence([(0, sample()), (.1, sample(z=-.2)), (.2, sample(pos=1.05)), (.3, sample(z=-.1))])
        self.assertEqual(result["z_at_close"], -.3)
        self.assertFalse(result["success"])

    def test_stale_gap_and_nonmonotonic(self):
        for t, f in [(1.5, sample(z=-.2, fresh=False)), (3, sample(z=-.2)), (.9, sample(z=-.2))]:
            frames = [(0, sample())] + [(i / 10, sample(z=-.2)) for i in range(1, 15)] + [(t, f)]
            self.assertFalse(self.sequence(frames)["success"])

    def test_invalid_measurements(self):
        for bad in (None, math.nan, math.inf, "0.3", True):
            self.assertFalse(holding(sample(pos=bad)))
            self.assertFalse(holding(sample(eff=bad)))
            self.assertFalse(grasp_update(None, sample(z=bad), 1)["success"])

    def test_boundaries(self):
        for side in ("left", "right"):
            edge = EMPTY_CLOSED[side] + HOLD_MARGIN
            self.assertTrue(holding(sample(pos=edge, eff=-1), side))
            self.assertFalse(holding(sample(pos=edge - 1e-4, eff=-1), side))
            self.assertTrue(closed_empty(sample(pos=edge - 1e-4, eff=-1), side))
        self.assertTrue(holding(sample(pos=.75, eff=-1)))
        self.assertFalse(holding(sample(eff=-.99)))
        self.assertFalse(closed_empty(sample(pos=-.05, eff=-.5)))         # not pushing closed: not "empty"

    def test_calibration_records_right(self):
        """Right gripper records of 2026-11-15 17:14-17:18 (实体 / 薄片 / 空夹)."""
        solid = [(.5169, -2.74), (.5189, -2.74), (.5604, -2.81), (.4218, -2.57), (.3658, -2.47)]
        thin = [(.1285, -2.04), (.0542, -1.90), (-.0032, -1.80), (-.0341, -1.73)]
        empty = [(-.0485, -1.71), (-.0485, -1.71), (-.0486, -1.71), (-.0487, -1.71), (-.0487, -1.71), (-.047, -1.71)]
        for pos, eff in solid + thin:
            self.assertTrue(holding(sample(pos=pos, eff=eff), "right"), (pos, eff))
        for pos, eff in empty:
            self.assertFalse(holding(sample(pos=pos, eff=eff), "right"), (pos, eff))
            self.assertTrue(closed_empty(sample(pos=pos, eff=eff), "right"), (pos, eff))

    def test_empty_ends_attempt_after_persist(self):
        frames = [(i / 10, sample(pos=-0.0485, eff=-1.71)) for i in range(int(EMPTY_PERSIST * 10))]
        self.assertFalse(self.sequence(frames + [(EMPTY_PERSIST - .05, sample(pos=-0.0485, eff=-1.71))])["empty"])
        state = self.sequence(frames + [(EMPTY_PERSIST + .01, sample(pos=-0.0485, eff=-1.71))])
        self.assertTrue(state["empty"])
        self.assertFalse(state["success"])
        # reopening (or holding something) breaks the empty interval
        self.assertFalse(self.sequence([(0, sample(pos=-0.0485, eff=-1.71)), (.5, sample(pos=1.05, eff=.2)),
                                        (1.2, sample(pos=-0.0485, eff=-1.71))])["empty"])

    def test_thin_grasp_lifted_succeeds(self):
        frames = [(0, sample(pos=-.03, eff=-1.73))] + [(i / 10, sample(pos=-.03, eff=-1.73, z=-.24)) for i in range(1, 18)]
        state = None
        for t, f in frames:
            state = grasp_update(state, f, t, "right")
        self.assertTrue(state["success"])
        self.assertFalse(state["empty"])

    def test_no_lift_flag(self):
        state = None
        for i in range(int(NO_LIFT_S * 10) + 2):
            state = grasp_update(state, sample(), i / 10)
        self.assertTrue(state["no_lift"])
        self.assertFalse(state["success"])

    def test_options(self):
        self.assertEqual(options({"attempts": 3, "port": 8001})["attempts"], 3)
        for params in ({"attempts": True}, {"attempts": 6}, {"attempts": 0}, {"port": "8000"}, {"extra": 1}, {"prompt": "bad'prompt"}):
            with self.assertRaises(ValueError):
                options(params)

    def test_monitor_individual_topic_freshness(self):
        f = Feedback()
        for side in ("left", "right"):
            f.joints(side, SimpleNamespace(position=[0.] * 7), 1)
            f.gripper(side, SimpleNamespace(position=[.4], effort=[-2.]), 1)
            f.ee(side, SimpleNamespace(position=SimpleNamespace(z=-.2)), 1)
        self.assertTrue(f.snapshot(1.1)["arms_fresh"])
        for side in ("left", "right"):
            f.joints(side, SimpleNamespace(position=[0.] * 7), 1.4)
        self.assertFalse(f.snapshot(1.4)["arms_fresh"])
        self.assertFalse(f.snapshot(1.4)["left"]["fresh"])


if __name__ == "__main__":
    unittest.main()
