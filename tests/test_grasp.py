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
    def sequence(self, frames, side=None, opened=True):
        """By default the gripper is seen open just before the frames, as after a start-pose reset."""
        state = grasp_update(None, sample(pos=1.05, eff=0.2), frames[0][0] - 0.05, side) if opened else None
        for t, f in frames:
            state = grasp_update(state, f, t, side)
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

    def test_reopen_clears_close_height_and_breaks_hold(self):
        result = self.sequence([(0, sample()), (.1, sample(z=-.2)), (.2, sample(pos=1.05)), (.3, sample(z=-.1))])
        self.assertEqual(result["z_at_close"], -.1)            # the close after the reopening counts
        self.assertFalse(result["success"])

    def test_nothing_judged_until_seen_open(self):
        """Previous attempt left the gripper closed on nothing (or holding) until the reset opens it."""
        leftover = [(i / 10, sample(pos=-0.0485, eff=-1.71)) for i in range(30)]
        state = self.sequence(leftover, "right", opened=False)
        self.assertFalse(state["empty"] or state["success"] or state["no_lift"])
        self.assertIsNone(state["z_at_close"])
        lifted_leftover = [(0, sample())] + [(i / 10, sample(z=-.2)) for i in range(1, 30)]
        self.assertFalse(self.sequence(lifted_leftover, opened=False)["success"])
        # reset opens it, then the policy closes on nothing: empty after EMPTY_PERSIST
        frames = leftover + [(3.1, sample(pos=1.05, eff=.2)), (3.5, sample(pos=1.0, eff=.2))]
        frames += [(3.6 + i / 10, sample(pos=-0.0485, eff=-1.71)) for i in range(int(EMPTY_PERSIST * 10) + 2)]
        self.assertTrue(self.sequence(frames, "right", opened=False)["empty"])

    def test_stale_gap_and_nonmonotonic(self):
        for t, f in [(1.5, sample(z=-.2, fresh=False)), (3, sample(z=-.2)), (.9, sample(z=-.2))]:
            frames = [(0, sample())] + [(i / 10, sample(z=-.2)) for i in range(1, 15)] + [(t, f)]
            self.assertFalse(self.sequence(frames)["success"])

    def test_invalid_measurements(self):
        for bad in (None, math.nan, math.inf, "0.3", True):
            self.assertFalse(holding(sample(pos=bad)))
            self.assertFalse(holding(sample(eff=bad)))
            self.assertFalse(self.sequence([(1, sample(z=bad))])["success"])

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

    def test_classify_calibration_records(self):
        from mission import classify_grip
        c = lambda p, e: classify_grip(sample(pos=p, eff=e), "right")
        for p, e in [(.5169, -2.74), (.5604, -2.81), (.3658, -2.47)]:
            self.assertEqual(c(p, e), "solid")
        for p, e in [(.1285, -2.04), (.0542, -1.90), (-.0032, -1.80), (-.0341, -1.73)]:
            self.assertEqual(c(p, e), "thin")
        for p, e in [(-.0485, -1.71), (-.047, -1.71)]:
            self.assertEqual(c(p, e), "empty")
        self.assertEqual(c(.99, .24), "open")
        self.assertEqual(c(.4, -.5), "closing")
        self.assertEqual(c(None, -1), "unknown")

    def test_empty_ends_attempt_after_persist(self):
        frames = [(i / 10, sample(pos=-0.0485, eff=-1.71)) for i in range(int(EMPTY_PERSIST * 10))]
        self.assertFalse(self.sequence(frames + [(EMPTY_PERSIST - .05, sample(pos=-0.0485, eff=-1.71))])["empty"])
        state = self.sequence(frames + [(EMPTY_PERSIST + .01, sample(pos=-0.0485, eff=-1.71))])
        self.assertTrue(state["empty"])
        self.assertFalse(state["success"])
        # reopening (or holding something) breaks the empty interval
        self.assertFalse(self.sequence([(0, sample(pos=-0.0485, eff=-1.71)), (.5, sample(pos=1.05, eff=.2)),
                                        (1.2, sample(pos=-0.0485, eff=-1.71))])["empty"])

    def test_empty_close_then_reopen_keeps_the_attempt(self):
        # default 2 s: a policy that closes on nothing and reopens within that time keeps running (no stop)
        self.assertEqual(EMPTY_PERSIST, 2.0)
        frames = [(0, sample(pos=1.05, eff=.2))]
        frames += [(.1 + i / 10, sample(pos=-0.0485, eff=-1.71)) for i in range(15)]        # 1.5 s closed on nothing
        frames += [(1.7, sample(pos=1.05, eff=.2))]                                         # reopens: timers reset
        frames += [(1.8 + i / 10, sample(pos=-0.0485, eff=-1.71)) for i in range(15)]       # another 1.5 s empty
        self.assertFalse(self.sequence(frames)["empty"])
        frames += [(3.3 + i / 10, sample(pos=-0.0485, eff=-1.71)) for i in range(9)]       # stays closed > 2 s: stuck
        self.assertTrue(self.sequence(frames)["empty"])

    def test_thin_grasp_lifted_succeeds(self):
        frames = [(0, sample(pos=-.03, eff=-1.73))] + [(i / 10, sample(pos=-.03, eff=-1.73, z=-.24)) for i in range(1, 18)]
        state = self.sequence(frames, "right")
        self.assertTrue(state["success"])
        self.assertFalse(state["empty"])

    def test_no_lift_flag(self):
        state = self.sequence([(i / 10, sample()) for i in range(int(NO_LIFT_S * 10) + 2)])
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
