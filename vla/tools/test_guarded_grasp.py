"""Pure guards and stubbed execution; these tests never import ROS or connect."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

import guarded_grasp as runner


def observation():
    data = {"observation/state": np.zeros(48, dtype=np.float32), "prompt": "抓取红色沙包"}
    data.update({f"observation/images/{name}": np.zeros(shape, np.uint8) for name, shape in runner.CAMERAS.items()})
    return data


class GuardTests(unittest.TestCase):
    def action(self, target=None, measured=None, previous=None, **overrides):
        params = dict(period=1 / 45, speed=0.5, max_tracking=0.2, max_initial=0.2)
        params.update(overrides)
        return runner.guarded_action(np.zeros(19) if target is None else target,
            np.zeros(14) if measured is None else measured, previous, **params)

    def test_initial_jump_refused(self):
        target = np.zeros(19); target[0] = 0.201
        with self.assertRaisesRegex(runner.GuardError, "Initial predicted"):
            self.action(target)

    def test_all_joint_speed_and_gripper_range(self):
        target = np.zeros(19); target[runner.JOINTS] = np.linspace(-0.19, 0.19, 14)
        target[runner.GRIPPERS] = [-9, 9]; target[16:] = [4, 5, 6]
        command, detail = self.action(target)
        self.assertEqual(command.shape, (16,))
        self.assertLessEqual(np.max(np.abs(command[runner.JOINTS])), 0.5 / 45 + 1e-12)
        np.testing.assert_array_equal(command[runner.GRIPPERS], [-1.6, 2.0])
        self.assertTrue(detail["gripper_clamped"])

    def test_tracking_error_and_next_gap(self):
        with self.assertRaisesRegex(runner.GuardError, "tracking error"):
            self.action(previous=np.full(14, 0.201))
        target = np.zeros(19); target[runner.JOINTS] = 0.8
        with self.assertRaisesRegex(runner.GuardError, "Next command"):
            self.action(target, previous=np.full(14, 0.199))

    def test_nonfinite_and_shape_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            target = np.zeros(19); target[18] = value
            with self.assertRaises(runner.GuardError): self.action(target)
        with self.assertRaises(runner.GuardError): runner.validate_chunk({"actions": np.zeros((30, 16))})
        with self.assertRaises(runner.GuardError): self.action(np.zeros(16))

    def test_joint_limits_clamp_and_report(self):
        target = np.zeros(19); target[0] = 0.1
        command, diag = self.action(target, lower=np.full(14, -0.005), upper=np.full(14, 0.005))
        self.assertAlmostEqual(command[0], 0.005)
        self.assertEqual(diag["limit_clamped_joints"], [0])

    def test_mode_joint_enum_is_one(self):
        mode = dict(meta_mode=1, left_arm_control_mode=1, right_arm_control_mode=1,
                    enable_left_arm=True, enable_right_arm=True)
        runner.validate_mode(mode, 0.1, 1.0)
        for key in ["meta_mode", "left_arm_control_mode", "right_arm_control_mode", "enable_left_arm"]:
            with self.assertRaises(runner.GuardError): runner.validate_mode({**mode, key: 0}, 0.1, 1)
        with self.assertRaises(runner.GuardError): runner.validate_mode(mode, 1.1, 1)
        with self.assertRaises(runner.GuardError): runner.validate_mode(mode, -1, 1)


class StubExecutionTests(unittest.TestCase):
    def run_stub(self, *, lose_mode=False, max_actions=3, max_seconds=15, inference_delay=0,
                 subscribers=True, duplicate=False, partial_failure=False, rtc=False):
        envs = []
        clock = [10.0]
        mode = dict(meta_mode=1, left_arm_control_mode=1, right_arm_control_mode=1,
                    enable_left_arm=True, enable_right_arm=True)
        class Event:
            def is_set(self): return False
            def wait(self, seconds): clock[0] += seconds
        class Interface:
            def __init__(self):
                self.action_publishers = {x: object() for x in ("left_arm", "right_arm", "left_gripper", "right_gripper")}
                self._published_once = False
                self.joint_lower = {arm: np.full(7, -3.0) for arm in ("left_arm", "right_arm")}
                self.joint_upper = {arm: np.full(7, 3.0) for arm in ("left_arm", "right_arm")}
                self.lock = threading.Lock()
                self.latest_joint_states = {arm: types.SimpleNamespace(name=[f"{prefix}_joint{i}" for i in range(1, 8)])
                                           for arm, prefix in (("left_arm", "l"), ("right_arm", "r"))}
            def create_subscription(self, cls, topic, cb, qos):
                self.callback = cb; cb(types.SimpleNamespace(data=json.dumps(mode))); return object()
            def get_name(self): return "guard_stub"
            def get_namespace(self): return "/"
            def get_publishers_info_by_topic(self, topic):
                result = [types.SimpleNamespace(node_name="guard_stub", node_namespace="/")]
                if duplicate: result.append(types.SimpleNamespace(node_name="other_controller", node_namespace="/"))
                return result
            def get_subscriptions_info_by_topic(self, topic):
                return [types.SimpleNamespace(node_name="zenoh_bridge_ros2dds", node_namespace="/")] if subscribers else []
        class Environment:
            def __init__(self, **kwargs):
                assert kwargs["enable_chassis"] is False
                assert kwargs["interpolate"] is False
                self._ros_interface = Interface(); self.sent = []; self.closed = False; self.obs = observation(); envs.append(self)
            def get_observation(self): return self.obs
            def apply_action(self, action):
                self._ros_interface._published_once = True
                if partial_failure: raise runner.GuardError("Second publisher failed after the first published")
                command = action["actions"].copy(); self.sent.append(command)
                self.obs["observation/state"][runner.JOINTS] = command[runner.JOINTS]
                if lose_mode:
                    self._ros_interface.callback(types.SimpleNamespace(data=json.dumps({**mode, "meta_mode": 0})))
            def close(self): self.closed = True; return True
        class Policy:
            chunk = 0
            def infer(self, obs, timeout=None):
                clock[0] += inference_delay
                actions = np.zeros((30, 19)); actions[:, runner.GRIPPERS] = -1.0
                actions[:, 16:] = 1.0
                Policy.chunk += 1
                return {"actions": actions, "rtc/chunk_id": Policy.chunk if rtc else 1}
        modules = {"examples": types.ModuleType("examples"), "examples.teleavatar_v2": types.ModuleType("teleavatar_v2"),
                   "examples.teleavatar_v2.env": types.ModuleType("env"), "std_msgs": types.ModuleType("std_msgs"),
                   "std_msgs.msg": types.ModuleType("msg")}
        modules["examples.teleavatar_v2.env"].TeleavatarEnvironment = Environment
        modules["std_msgs.msg"].String = type("String", (), {})
        args = types.SimpleNamespace(prompt="grasp", frequency=45, max_age=1, max_actions=max_actions,
            max_seconds=max_seconds, replan_steps=8, max_speed=0.5, max_tracking=0.2, max_initial_gap=0.2,
            rtc=rtc, rtc_delay=2 if rtc else None, rtc_horizon=5 if rtc else None)
        report = {}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch.object(runner.time, "monotonic", lambda: clock[0]), patch.object(runner, "dump_observation") as dump:
            try: runner.execute(args, Policy(), Path(tmp), report, Event())
            except runner.GuardError:
                if not (lose_mode or not subscribers or duplicate or partial_failure or inference_delay > 2): raise
            plans = Path(tmp) / "inferences.jsonl"
            report["_plans"] = [json.loads(l) for l in plans.read_text().splitlines()] if plans.exists() else []
            report["_dumped"] = [c.args[0].name for c in dump.call_args_list]
        return envs[0], report

    def test_exact_count_no_chassis_no_final_release(self):
        env, report = self.run_stub()
        self.assertEqual(len(env.sent), 3)
        self.assertEqual(report["actuations"], 3)
        self.assertTrue(env.closed)
        self.assertTrue(all(command.shape == (16,) for command in env.sent))
        np.testing.assert_array_equal(env.sent[-1][runner.GRIPPERS], [-1, -1])
        self.assertFalse(report["extra_final_release_command_sent"])
        self.assertEqual(report["attempted_actuations"], 3)
        self.assertTrue(report["interface_publish_started"])

    def test_rtc_logs_observation_seconds_actions_per_chunk(self):
        env, report = self.run_stub(rtc=True, max_actions=40)
        self.assertEqual(report["actuations"], 40)
        plans = report["_plans"]
        self.assertGreaterEqual(len(plans), 2)
        for i, entry in enumerate(plans):
            self.assertEqual(entry["inference"], i)
            self.assertEqual(entry["observation"], f"observation_{i:03d}.npz")
            self.assertIsInstance(entry["seconds"], float)
            self.assertEqual(np.asarray(entry["actions19"]).shape, (30, 19))
        self.assertEqual(report["_dumped"], [e["observation"] for e in plans] + ["final_observation.npz"])

    def test_mode_loss_stops_before_second_actuation(self):
        env, report = self.run_stub(lose_mode=True)
        self.assertEqual(len(env.sent), 1)
        self.assertEqual(report["actuations"], 1)
        self.assertTrue(env.closed)

    def test_runtime_bound_exact(self):
        env, report = self.run_stub(max_actions=675, max_seconds=0.05)
        self.assertEqual(len(env.sent), 3)
        self.assertEqual(report["status"], "time_limit")

    def test_missing_subscriber_and_other_controller_prevent_all_motion(self):
        for kwargs in [{"subscribers": False}, {"duplicate": True}]:
            env, report = self.run_stub(**kwargs)
            self.assertEqual(report["attempted_actuations"], 0)
            self.assertEqual(len(env.sent), 0)
            self.assertTrue(env.closed)

    def test_partial_publish_count_is_not_falsely_zero(self):
        env, report = self.run_stub(partial_failure=True)
        self.assertEqual(report["actuations"], 0)
        self.assertEqual(report["attempted_actuations"], 1)
        self.assertTrue(report["interface_publish_started"])
        self.assertTrue(env.closed)

    def test_inference_overrun_prevents_first_motion(self):
        env, report = self.run_stub(inference_delay=2.1)
        self.assertEqual(report["attempted_actuations"], 0)
        self.assertEqual(len(env.sent), 0)
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
