"""Module API contracts through TA2Client, real localhost HTTP and fake ROS/scripts."""
import concurrent.futures
import json
import os
import signal
import time
import unittest
from unittest.mock import patch

from test_mission_e2e import _MissionServer
from ta2_client import TA2Client, TA2Error


class _ModuleServer(_MissionServer):
    scenario = "api_success_left"

    def setUp(self):
        super().setUp()
        self.client = TA2Client(self.url, timeout=8)

    def tearDown(self):
        # Monitor deliberately uses a separate process group, like production.
        for pidfile in self.dir.glob("monitor-*.pid"):
            try:
                os.kill(int(pidfile.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass
        super().tearDown()

    def error(self, code, status, call, *args, **kwargs):
        with self.assertRaises(TA2Error) as ctx:
            call(*args, **kwargs)
        self.assertEqual((ctx.exception.code, ctx.exception.http_status), (code, status))

    def idle(self):
        self.wait_for(lambda: not self.client.health()["busy"])


class ModuleAPI(_ModuleServer, unittest.TestCase):
    def test_arm_move_forward_and_validation(self):
        result = self.client.arm_move("right", forward_cm=5, wait=True, poll=.03)
        self.assertEqual(result["status"], "succeeded", result)
        self.client.arm_move("left", forward_cm=-2.5, left_cm=1, up_cm=15, wait=True, poll=.03)
        self.assertEqual(self.commands("move"), [["right", "5", "0", "0"], ["left", "-2.5", "1", "15"]])
        for kw in ({"forward_cm": 15.01}, {"up_cm": -16}, {"forward_cm": 0}, {"left_cm": "5"}, {"forward_cm": True}):
            self.error("BAD_REQUEST", 400, self.client.arm_move, "left", **kw)
        self.error("BAD_REQUEST", 400, self.client.arm_move, "both", forward_cm=5)
        self.assertIn("arm.move", self.client.capabilities()["actions"])
        for side in ("left", "right"):                       # page buttons: one click = 5 cm forward
            self.idle()
            self.assertEqual(self.request("POST", f"/arm/fwd5_{side}")[0], 200)
            self.wait_for(lambda: len(self.commands("move")) == 3 + (side == "right"))
        self.assertEqual(self.commands("move")[2:], [["left", "5", "0", "0"], ["right", "5", "0", "0"]])

    def test_arm_down_both_sides_and_validation(self):
        for side, cm in (("left", .5), ("right", 15)):
            result = self.client.arm_down(side, cm, wait=True, poll=.03)
            self.assertEqual(result["status"], "succeeded", result)
            self.assertEqual(result["result"]["returncode"], 0)
            self.assertEqual(result["result"]["output"].splitlines()[-2], "descent output 9")
            self.assertLessEqual(len(result["result"]["output"].splitlines()), 8)
        self.assertEqual(self.commands("down"), [["left", "0.5"], ["right", "15"]])
        for cm in (0, .49, 15.01, True, "5", None, float("nan"), float("inf")):
            self.error("BAD_REQUEST", 400, self.client.arm_down, "left", cm)
        self.error("BAD_REQUEST", 400, self.client.arm_down, "both", 5)
        self.assertIn("arm.down", self.client.capabilities()["actions"])

    def test_busy_including_web_action(self):
        (self.dir / "down_hold").touch()
        try:
            self.assertEqual(self.request("POST", "/arm/down5")[0], 200)
            self.error("BUSY", 409, self.client.arm_down, "left", 3)
            self.assertFalse(self.client.preflight()["ok"])
        finally:
            (self.dir / "down_hold").unlink()
        self.idle()
        aid = self.client.vla_grasp(seconds=3)
        self.error("BUSY", 409, self.client.arm_down, "right", 4)
        self.client.stop()
        self.assertEqual(self.client.wait(aid, poll=.03)["error"]["code"], "STOPPED")
        self.idle()

    def test_arm_descent_preparation_is_stopped(self):
        (self.dir / "down_hold").touch()
        for index, (code, stop) in enumerate((("STOPPED", self.client.stop), ("ESTOP", self.client.estop)), 1):
            aid = self.client.arm_down("left", 4)
            self.wait_for(lambda: len(self.commands("down")) == index)
            stop()
            result = self.client.wait(aid, timeout=4, poll=.03)
            self.assertEqual((result["status"], result["error"]["code"]), ("aborted", code))
            self.idle()  # host script actually exited; reservation released
            self.wait_for(lambda: not self.client.health()["estop_active"])
            self.assertNotEqual(self.client.action(aid)["result"]["returncode"], 0)
        (self.dir / "down_hold").unlink()

    def test_detected_grasp_success_and_per_call_model(self):
        selected = self.status()["vla_selected"]
        model = next(m for m in self.client.models()["models"] if m["rtc"] and m["id"] != selected)
        # A second call proves that the detector does not retain the first hold/lift.
        for _ in range(2):
            # Simulate removing the object before the next independent grasp.
            (self.dir / "state.json").write_text(json.dumps({"attempt": 0, "stage": "idle"}))
            self.wait_for(lambda: not self.client.grippers()["left"]["holding"])
            began = time.monotonic()
            result = self.client.vla_grasp(model=model["id"], seconds=6, stop_on_grasp=True,
                                           wait=True, poll=.03)
            self.assertEqual(result["status"], "succeeded", result)
            detail = result["result"]
            self.assertTrue(detail["grasped"], result)
            self.assertGreater(time.monotonic() - began, 1.5)
            self.assertEqual(detail["hand"], "left")
            self.assertEqual((detail["model"], detail["rtc"], detail["seconds"]), (model["id"], True, 6))
            self.assertEqual(detail["status"], "succeeded")
            self.assertIsNone(detail["error"])
            self.assertEqual(detail["returncode"], 130)
            self.assertEqual(self.status()["vla_selected"], selected)
            self.assertTrue({"actuations", "report", "tag"} <= detail.keys())
            self.assertFalse(self.client.preflight()["ok"])  # fresh but occupied hand
        self.assertEqual(len(self.commands("vla_interrupted")), 2)
        self.assertTrue(any(cmd[:3] == ["pkill", "-INT", "-f"] for cmd in self.commands("dex")))
        self.assertEqual(self.commands("arm"), [])  # no reset/stow on self-stop
        self.assertEqual(self.commands("down"), [])

    def test_unmonitored_grasp_and_parameter_errors(self):
        (self.dir / "monitor_exit").touch()  # unmonitored calls must ignore monitor availability
        result = self.client.vla_grasp(seconds=1, stop_on_grasp=False, wait=True, poll=.03)
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["result"]["status"], "time_limit")
        self.assertIsNone(result["result"]["grasped"])
        self.assertIsNone(result["result"]["hand"])
        self.assertFalse(list(self.dir.glob("monitor-*.pid")))
        self.assertEqual(self.commands("vla_interrupted"), [])
        self.error("UNKNOWN_MODEL", 400, self.client.vla_grasp, model="no-such-model")
        self.error("UNKNOWN_MODEL", 400, self.client.vla_grasp, model=[])
        for seconds in (0, 121, True, "1", float("nan")):
            self.error("BAD_REQUEST", 400, self.client.vla_grasp, seconds=seconds)
        self.error("BAD_REQUEST", 400, self.client.vla_grasp, stop_on_grasp=1)
        self.error("BAD_REQUEST", 400, self.client.start, "vla.grasp", extra=1)
        aid = self.client.vla_grasp()  # unchanged default call
        self.wait_for(lambda: len(self.commands("vla")) == 2)
        self.assertEqual(self.commands("vla")[-1][1], "60")
        self.client.stop()
        self.idle()
        self.assertIsNone(self.client.action(aid)["result"]["grasped"])

    def test_monitored_time_limit_without_confirmed_grasp(self):
        result = self.client.vla_grasp(seconds=1, stop_on_grasp=True, wait=True, poll=.03)
        self.assertEqual(result["status"], "succeeded", result)
        self.assertFalse(result["result"]["grasped"])
        self.assertIsNone(result["result"]["hand"])
        self.assertEqual(result["result"]["status"], "time_limit")
        self.assertEqual(self.commands("vla_interrupted"), [])

    def test_external_stop_and_estop(self):
        for index, (code, stop) in enumerate((("STOPPED", self.client.stop), ("ESTOP", self.client.estop)), 1):
            aid = self.client.vla_grasp(seconds=10, stop_on_grasp=True)
            self.wait_for(lambda: len(self.commands("vla")) == index)
            time.sleep(.15)
            start = time.monotonic()
            stop()
            self.assertLess(time.monotonic() - start, .6)
            result = self.client.wait(aid, poll=.03)
            self.assertEqual((result["status"], result["error"]["code"]), ("aborted", code))
            self.assertTrue({"grasped", "hand", "model", "rtc", "seconds", "status", "actuations", "error", "report", "tag"} <= result["result"].keys())
            self.idle()
            self.wait_for(lambda: not self.client.health()["estop_active"])
            final = self.client.action(aid)
            self.assertEqual((final["status"], final["error"]["code"]), ("aborted", code))
            self.assertFalse(final["result"]["grasped"])

    def test_grippers_restart_and_readonly_preflight(self):
        readings = self.client.grippers()
        self.assertEqual(readings["source"], "monitor")
        for side in ("left", "right"):
            self.assertEqual(set(readings[side]), {"pos", "eff", "ee_z", "fresh", "holding"})
            self.assertTrue(readings[side]["fresh"])
            self.assertFalse(readings[side]["holding"])
        pid = int(next(self.dir.glob("monitor-*.pid")).read_text())
        os.kill(pid, signal.SIGKILL)
        time.sleep(.05)
        self.assertTrue(self.client.grippers()["left"]["fresh"])
        self.assertEqual(len(list(self.dir.glob("monitor-*.pid"))), 2)
        preflight = self.client.preflight()
        self.assertTrue(preflight["ok"], preflight)
        self.assertEqual(len(preflight["checks"]), 8)
        models = self.client.models()
        self.assertEqual(sum(m["selected"] for m in models["models"]), 1)
        for model in models["models"]:
            self.assertIsInstance(model["rtc"], bool)
            self.assertIsInstance(model["running"], bool)
            self.assertEqual(model["selected"], model["id"] == models["selected"])
        self.assertTrue(self.client.preflight(model=models["models"][-1]["id"])["ok"])
        self.error("UNKNOWN_MODEL", 400, self.client.preflight, model="absent")
        (self.dir / "stale").touch()
        time.sleep(.1)
        preflight = self.client.preflight()
        self.assertFalse(preflight["ok"])
        self.assertFalse(next(c for c in preflight["checks"] if c["name"] == "left_gripper")["ok"])
        self.assertEqual(self.commands("vla") + self.commands("arm") + self.commands("down") + self.commands("dex"), [])

    def test_monitor_unavailable_blocks_grasp(self):
        for marker in ("monitor_exit", "stale", "monitor_silent"):
            flag = self.dir / marker
            flag.touch()
            time.sleep(.7)  # let the latest valid frame expire / the fake observe the marker
            try:
                if marker == "monitor_exit":
                    self.error("MONITOR_DOWN", 503, self.client.grippers)
                else:
                    self.assertFalse(self.client.grippers()["left"]["fresh"])
                self.error("MONITOR_DOWN", 503, self.client.vla_grasp, stop_on_grasp=True)
                self.assertEqual(self.commands("vla"), [])
            finally:
                flag.unlink()
            self.wait_for(lambda: self.client.grippers()["left"]["fresh"])

    def test_monitor_never_produces_data(self):
        (self.dir / "monitor_silent").touch()
        self.error("MONITOR_DOWN", 503, self.client.grippers)
        self.assertEqual(self.commands("vla"), [])

    def test_feedback_loss_during_grasp_stops_runner(self):
        aid = self.client.vla_grasp(seconds=10, stop_on_grasp=True)
        self.wait_for(lambda: self.commands("vla"))
        (self.dir / "monitor_silent").touch()
        result = self.client.wait(aid, timeout=4, poll=.03)
        self.assertEqual((result["status"], result["error"]["code"]), ("failed", "FEEDBACK_LOST"), result)
        self.assertFalse(result["result"]["grasped"])
        self.assertEqual(len(self.commands("vla_interrupted")), 1)

    def test_estop_during_monitor_preflight_never_launches(self):
        (self.dir / "stale").touch()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(self.client.vla_grasp, stop_on_grasp=True)
            self.wait_for(lambda: list(self.dir.glob("monitor-*.pid")))
            start = time.monotonic()
            self.client.estop()
            self.assertLess(time.monotonic() - start, .5)
            with self.assertRaises(TA2Error) as ctx:
                future.result(timeout=5)
            self.assertIn(ctx.exception.code, ("MONITOR_DOWN", "ESTOP_ACTIVE"))
        self.assertEqual(self.commands("vla"), [])


class RightGrasp(_ModuleServer, unittest.TestCase):
    scenario = "api_success_right"

    def test_right_hand_grasp(self):
        result = self.client.vla_grasp(seconds=6, stop_on_grasp=True, wait=True, poll=.03)
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["result"]["hand"], "right")


class ModuleUnitTests(unittest.TestCase):
    def test_policy_down_and_monitor_launch_failure(self):
        import web_move as web
        from types import SimpleNamespace
        with patch.dict(os.environ), patch.object(web, "state", {**web.state, "connected": True}), \
                patch.object(web.subprocess, "run", return_value=SimpleNamespace(stdout="false")):
            os.environ.pop("VLA_CMD", None)
            with self.assertRaises(web.APIError) as ctx:
                web.start_action({"type": "vla.grasp", "params": {"model": web.DEFAULT_MODEL}})
            self.assertEqual((ctx.exception.status, ctx.exception.code), (503, "POLICY_DOWN"))
        monitor = web.GripperMonitor()
        with patch.object(web.mission_module, "Monitor", side_effect=OSError("missing executable")):
            with self.assertRaises(web.APIError) as ctx:
                monitor.get()
            self.assertEqual((ctx.exception.status, ctx.exception.code), (503, "MONITOR_DOWN"))

    def test_client_grasp_parameters_and_default_compatibility(self):
        client = TA2Client("http://unused")
        with patch.object(client, "_run", return_value="ok") as run:
            self.assertEqual(client.vla_grasp(), "ok")
            self.assertEqual(run.call_args.args, ("vla.grasp", {"stop_on_grasp": False}, False, 300, .5, False))
            self.assertEqual(client.vla_grasp("model", 10, True, True), "ok")
            self.assertEqual(run.call_args.args, ("vla.grasp", {"model": "model", "seconds": 10, "stop_on_grasp": True}, True, 300, .5, False))
