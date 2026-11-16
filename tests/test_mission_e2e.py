"""Real HTTP server + real teleop against tests/fakeros; no Docker or network beyond localhost."""
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ta2_client import TA2Client


class _MissionServer:
    scenario = "success_left"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ta2-mission-test-")
        self.dir = Path(self.tmp.name)
        paths = self.dir / "paths"
        paths.mkdir()
        for name in ("001", "002", "003", "004"):
            (paths / (name + ".csv")).write_text("t_s,x_m,y_m,heading_deg\n0,0,0,0\n1,0,0.08,0\n")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        fake = [sys.executable, str(ROOT / "tests/mission_fakes.py")]
        env = {**os.environ, "PORT": str(port), "FAKE_ROOT": str(self.dir),
               "MISSION_SCENARIO": self.scenario, "PYTHONPATH": str(ROOT / "tests/fakeros"),
               "TELEOP_CMD": shlex.join([sys.executable, "-u", str(ROOT / "teleop.py")]),
               "VLA_CMD": shlex.join(fake + ["vla"]), "ARM_CMD": shlex.join(fake + ["arm"]),
               "ARM_DOWN_CMD": shlex.join(fake + ["down"]), "ARM_MOVE_CMD": shlex.join(fake + ["move"]), "MONITOR_CMD": shlex.join(fake + ["monitor"]),
               "ESTOP_DEXEC": shlex.join(fake + ["dex"]), "ARM_STOP": shlex.join(fake + ["dex", "pkill"]),
               "PATH_DIR": str(paths), "TRAIL_DIR": str(self.dir / "trails"),
               "MISSION_LOG_DIR": str(self.dir / "missions"), "ACTION_LOG": str(self.dir / "actions.jsonl"), "GRIP_LOG": str(self.dir / "grip.jsonl"),
               "PATH_PAUSE": "0.05", "ARRIVE_PAUSE": "0.05", "ATTEMPT_S": "4", "SC": '{"noise":0}',
               **getattr(self, "extra_env", {})}
        self.server_log = (self.dir / "server.log").open("w")
        self.server = subprocess.Popen([sys.executable, "-u", str(ROOT / "web_move.py")], cwd=ROOT,
                                       env=env, stdout=self.server_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        try:
            self.wait_for(lambda: self.request("GET", "/status")[1]["mode_ok"] is True, timeout=8)
        except Exception:
            self.tearDown()
            raise

    def tearDown(self):
        try:
            self.request("POST", "/estop")
            self.wait_for(lambda: not self.status()["mission"]["active"], timeout=8)
        finally:
            os.killpg(self.server.pid, signal.SIGTERM)
            self.server.wait(timeout=5)
            self.server_log.close()
            self.tmp.cleanup()

    def request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else b"" if method == "POST" else None
        req = Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=8) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def wait_for(self, predicate, timeout=18):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                value = predicate()
                if value:
                    return value
            except (URLError, ConnectionError):
                pass
            time.sleep(.03)
        self.fail("condition timed out; " + (self.dir / "server.log").read_text())

    def status(self):
        return self.request("GET", "/status")[1]

    def start(self, **params):
        code, body = self.request("POST", "/api/v1/actions", {"type": "mission.run", "params": params})
        self.assertEqual(code, 202, body)
        return body["action_id"]

    def finish(self, aid, expected):
        self.wait_for(lambda: not self.status()["mission"]["active"])
        m = self.status()["mission"]
        self.assertEqual(m["status"], expected, m)
        action = self.request("GET", "/api/v1/actions/" + aid)[1]
        self.assertEqual(action["status"], expected, action)
        records = [json.loads(line) for line in (self.dir / "missions" / (m["id"] + ".jsonl")).read_text().splitlines()]
        self.assertEqual(records[-1]["event"], "summary")
        self.assertEqual(records[-1]["status"], expected)
        self.assertEqual(records[-1]["attempt_results"], m["attempt_results"])
        self.assertIn("g2r", records[-1]["durations"])
        self.wait_for(lambda: (self.dir / "actions.jsonl").exists())
        logs = [json.loads(line) for line in (self.dir / "actions.jsonl").read_text().splitlines()]
        self.assertEqual(len([a for a in logs if a["id"] == aid]), 1)
        return m, records

    def commands(self, kind):
        p = self.dir / "commands.jsonl"
        return [r["argv"] for r in map(json.loads, p.read_text().splitlines()) if r["kind"] == kind] if p.exists() else []


class LeftSuccess(_MissionServer, unittest.TestCase):
    scenario = "success_left"

    def test_third_attempt_left_success(self):
        aid = self.start(prompt="测试抓取", port=8001)
        m, records = self.finish(aid, "succeeded")
        self.assertEqual(m["hand"], "left")
        self.assertEqual([r["result"] for r in m["attempt_results"]], ["failed", "failed", "succeeded"])
        self.assertEqual(self.commands("arm"), [["start_hold"], ["open_left"]])
        self.assertEqual(self.commands("down"), [["left", "10"]])
        self.assertEqual(self.commands("move"), [["left", "10", "0", "0"]])        # step 6: reach out, then lower
        order = [r["kind"] for r in map(json.loads, (self.dir / "commands.jsonl").read_text().splitlines())
                 if r["kind"] in ("move", "down")]
        self.assertEqual(order, ["move", "down"])
        self.assertEqual(self.commands("vla_env"), [["测试抓取", "8001", "1"]] * 3)
        self.assertEqual(len(self.commands("vla_interrupted")), 1)
        self.assertEqual(list(m["durations"]), ["preflight", "g2r", "grasp", "stow", "r2g", "place"])
        self.assertEqual(self.request("GET", "/api/v1/state")[1]["mission"]["id"], m["id"])



class AllFail(_MissionServer, unittest.TestCase):
    scenario = "all_fail"

    def test_all_five_fail(self):
        # Set scenario before launching a separate server via a helper subclass below.
        self.assertEqual(self.scenario, "all_fail")
        aid = self.start()
        m, records = self.finish(aid, "failed")
        self.assertEqual(len(m["attempt_results"]), 5)
        self.assertEqual(self.commands("arm"), [["start"]])
        self.assertEqual(self.commands("down"), [])
        self.assertNotIn("r2g", m["durations"])
        self.assertFalse(any("path 003" in line for line in self.status()["log"]))



class ReturnDrop(_MissionServer, unittest.TestCase):
    scenario = "drop"

    def test_return_drop(self):
        aid = self.start()
        self.wait_for(lambda: self.status()["active_path"] == "003")
        (self.dir / "drop").touch()
        m, _ = self.finish(aid, "failed")
        self.assertIn("dropped", m["result"])
        self.assertEqual(self.commands("down"), [])
        self.assertEqual(self.commands("arm"), [["start_hold"]])
        log = "\n".join(self.status()["log"])
        self.assertIn("path 003: path done", log)
        self.assertNotIn("path 004:", log)



class EstopGate(_MissionServer, unittest.TestCase):
    scenario = "estop"

    def test_estop_and_post_gate(self):
        aid = self.start()
        self.wait_for(lambda: self.status()["active_path"] == "001")
        posts = [("/stop", None), ("/release", None), ("/arm/start_hold", None), ("/vla", None),
                 ("/step/forward", None), ("/paths/upload", {}), ("/route/r2g", None), ("/rec/start", None),
                 ("/api/v1/stop", None), ("/api/v1/actions", {"type": "arm.pose", "params": {"pose": "start"}}),
                 (f"/api/v1/actions/{aid}/cancel", None)]
        for path, body in posts:
            code, result = self.request("POST", path, body)
            self.assertEqual(code, 409, (path, result))
            self.assertEqual(result["error"]["code"] if isinstance(result["error"], dict) else result["code"], "MISSION_ESTOP_ONLY")
        t0 = time.monotonic()
        self.assertEqual(self.request("POST", "/api/v1/estop")[0], 200)
        self.assertLess(time.monotonic() - t0, .5)
        m, _ = self.finish(aid, "aborted")
        self.assertEqual(self.commands("vla"), [])
        self.assertEqual(self.commands("arm"), [])
        self.assertNotIn("grasp", m["durations"])



class StalePreflight(_MissionServer, unittest.TestCase):
    scenario = "stale"

    def test_preflight_stale(self):
        code, body = self.request("POST", "/api/v1/actions", {"type": "mission.run"})
        self.assertEqual(code, 409, body)
        self.assertEqual(body["error"]["code"], "MISSION_PREFLIGHT_FAILED")
        self.assertFalse(self.status()["mission"]["active"])
        self.assertEqual(self.commands("vla"), [])
        self.assertFalse(list((self.dir / "missions").glob("*.jsonl")))

class EmptyGraspFailsFast(_MissionServer, unittest.TestCase):
    scenario = "empty_fast"
    extra_env = {"EMPTY_PERSIST": "1"}          # the fake attempt lasts 4 s; this tests the early stop itself

    def test_empty_attempts_end_after_about_a_second(self):
        aid = self.start(attempts=2)
        m, records = self.finish(aid, "failed")
        self.assertIn("all grasp attempts failed", m["result"])
        self.assertEqual([r["reason"] for r in m["attempt_results"]], ["left gripper closed on nothing"] * 2)
        self.assertEqual(len(self.commands("vla_interrupted")), 2)            # stopped, not run to ATTEMPT_S
        ends = {r["tag"]: r["t"] for r in records if r["event"] == "attempt_end"}
        starts = {r["tag"]: r["t"] for r in records if r["event"] == "attempt_start"}
        for tag in ends:
            # not during the leftover-closed reset phase (< 1.2 s), and well before ATTEMPT_S (4 s)
            self.assertTrue(2.0 < ends[tag] - starts[tag] < 3.6, ends[tag] - starts[tag])
        self.assertTrue(any(r["event"] == "grip" for r in records))           # telemetry is logged


class GraspOnlyRetry(_MissionServer, unittest.TestCase):
    scenario = "success_left"            # attempts 1-2 fail, attempt 3 grasps with the left hand

    def test_grasp_flow_retries_and_stops_holding(self):
        aid = TA2Client(self.url).run_mission(attempts=5, flow="grasp")
        self.wait_for(lambda: not self.status()["mission"]["active"])
        m = self.status()["mission"]
        self.assertEqual((m["status"], m["result"], m["hand"], m["flow"]), ("succeeded", "grasped", "left", "grasp"), m)
        self.assertEqual([r["result"] for r in m["attempt_results"]], ["failed", "failed", "succeeded"])
        self.assertEqual(list(m["durations"]), ["preflight", "grasp"])
        live = m["live"]                                   # internals shown on the page for the last attempt
        self.assertEqual((live["attempt"], live["attempts"], live["phase"], live["verdict"], live["hand"]),
                         (3, m["attempts"], "ended", "succeeded", "left"))
        self.assertEqual(set(live["hands"]), {"left", "right"})
        self.assertEqual(live["hands"]["left"]["grip"], "solid")
        self.assertTrue(live["hands"]["left"]["armed"])
        self.assertGreaterEqual(live["hands"]["left"]["lift_cm"], 4)
        self.assertGreaterEqual(live["hands"]["left"]["good_s"], live["rules"]["success_s"])
        self.assertTrue(all(isinstance(r["seconds"], float) for r in m["attempt_results"]))
        self.assertEqual(self.commands("arm"), [])          # no stow, no release: stops holding
        self.assertEqual(self.commands("down"), [])
        self.assertEqual(self.request("GET", "/api/v1/actions/" + aid)[1]["status"], "succeeded")
        from urllib.request import urlopen
        html = urlopen(self.url + "/", timeout=5).read().decode()
        self.assertIn('id="vlaattempts"', html)
        self.assertIn('id="vlalive"', html)
        self.assertIn('id="missionlive"', html)
        flow = urlopen(self.url + "/grasp-flow", timeout=5).read().decode()
        self.assertIn('href="/grasp-flow"', html)
        for text in ("抓取流程图", "<svg", "① 复位", "② 取帧", "③ VLA 推理执行", "夹住且抬高 ≥ 4 cm", "连续保持 1.5 s？",
                     "没有再张开？", "否：继续推理", "是：i = i + 1，重新复位", "最多 4 s（从第一个动作计时）",
                     "右 ≥ -0.0385 rad"):                                   # ATTEMPT_S=4 in this test env
            self.assertIn(text, flow)
        self.assertNotIn("__", flow.split("<script>")[0])
        self.assertIn("flow:'grasp'", html)


class GraspOnlyAllFail(_MissionServer, unittest.TestCase):
    scenario = "empty_fast"
    extra_env = {"EMPTY_PERSIST": "1"}

    def test_grasp_flow_all_fail_resets_arms(self):
        aid = TA2Client(self.url).run_mission(attempts=2, flow="grasp")
        self.wait_for(lambda: not self.status()["mission"]["active"])
        m = self.status()["mission"]
        self.assertEqual(m["status"], "failed", m)
        self.assertIn("all grasp attempts failed", m["result"])
        self.assertEqual(self.commands("arm"), [["start"]])


class RightSuccess(_MissionServer, unittest.TestCase):
    scenario = "success_right"

    def test_first_attempt_right(self):
        client = TA2Client(self.url)
        aid = client.run_mission(attempts=1)
        m, _ = self.finish(aid, "succeeded")
        self.assertEqual(m["hand"], "right")
        self.assertEqual(self.commands("arm"), [["start_hold"], ["open_right"]])
        self.assertEqual(self.commands("down"), [["right", "10"]])
        self.assertEqual(self.commands("move"), [["right", "10", "0", "0"]])


class ScriptFailures(_MissionServer, unittest.TestCase):
    scenario = "arm_fail"

    def test_stow_nonzero(self):
        aid = self.start()
        m, _ = self.finish(aid, "failed")
        self.assertIn("exited 4", m["result"])
        self.assertNotIn("r2g", m["durations"])
        self.assertEqual(self.commands("down"), [])


class DescentFailure(_MissionServer, unittest.TestCase):
    scenario = "down_fail"

    def test_no_release_on_descent_failure(self):
        aid = self.start()
        m, _ = self.finish(aid, "failed")
        self.assertIn("exited 6", m["result"])
        self.assertEqual(self.commands("arm"), [["start_hold"]])


class FeedbackLoss(_MissionServer, unittest.TestCase):
    scenario = "feedback_loss"

    def test_stale_during_outbound(self):
        aid = self.start()
        self.wait_for(lambda: self.status()["active_path"] == "001")
        (self.dir / "stale").touch()
        m, _ = self.finish(aid, "failed")
        self.assertIn("feedback", m["result"])
        self.assertEqual(self.commands("vla"), [])
        self.assertNotIn("path 002:", "\n".join(self.status()["log"]))

    def test_monitor_exits_during_outbound(self):
        aid = self.start()
        self.wait_for(lambda: self.status()["active_path"] == "001")
        (self.dir / "monitor_exit").touch()
        self.finish(aid, "failed")
        self.assertEqual(self.commands("vla"), [])

    def test_monitor_silent_during_outbound(self):
        aid = self.start()
        self.wait_for(lambda: self.status()["active_path"] == "001")
        (self.dir / "monitor_silent").touch()
        self.finish(aid, "failed")
        self.assertEqual(self.commands("vla"), [])


class OccupiedPreflight(_MissionServer, unittest.TestCase):
    scenario = "occupied"

    def test_refuses_loaded_gripper(self):
        code, body = self.request("POST", "/api/v1/actions", {"type": "mission.run"})
        self.assertEqual(code, 409, body)
        self.assertEqual(body["error"]["code"], "MISSION_PREFLIGHT_FAILED")
        self.assertEqual(self.commands("arm"), [])


class ApiRegression(_MissionServer, unittest.TestCase):
    scenario = "api"

    def test_all_arm_poses_and_down5(self):
        client = TA2Client(self.url)
        poses = client.capabilities()["arm_poses"]
        self.assertEqual(set(poses), {"raise", "lower", "start", "start_hold", "down5", "fwd5_left", "fwd5_right",
                                      "open_right", "close_right", "open_left", "close_left"})
        for pose in poses:
            record = client.arm_pose(pose, wait=True, poll=.03)
            self.assertEqual(record["status"], "succeeded", record)
        self.assertEqual(self.commands("down"), [["right", "5"]])
        self.assertEqual(self.commands("move"), [["left", "5", "0", "0"], ["right", "5", "0", "0"]])

    def test_route_only_accepts_estop(self):
        client = TA2Client(self.url)
        aid = client.run_route("g2r")
        self.wait_for(lambda: self.status()["active_path"] == "001")
        code, body = self.request("POST", "/api/v1/stop")
        self.assertEqual(code, 409)
        self.assertEqual(body["error"]["code"], "ROUTE_ESTOP_ONLY")
        client.estop()
        record = client.wait(aid, poll=.03)
        self.assertEqual(record["status"], "aborted")

    def test_web_start_and_legacy_estop(self):
        code, body = self.request("POST", "/mission/run")
        self.assertEqual(code, 202, body)
        self.wait_for(lambda: self.status()["active_path"] == "001")
        self.assertEqual(self.request("POST", "/estop")[0], 200)
        self.finish(body["action_id"], "aborted")

    def test_estop_bypasses_preflight_lock(self):
        import concurrent.futures
        (self.dir / "stale").touch()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            start = pool.submit(self.request, "POST", "/api/v1/actions", {"type": "mission.run"})
            time.sleep(.15)
            t0 = time.monotonic()
            self.assertEqual(self.request("POST", "/estop")[0], 200)
            self.assertLess(time.monotonic() - t0, .5)
            code, body = start.result(timeout=5)
        self.assertEqual(code, 409, body)
        self.assertEqual(body["error"]["code"], "ESTOP_ACTIVE")
        self.assertEqual(self.commands("arm"), [])


if __name__ == "__main__":
    unittest.main()
