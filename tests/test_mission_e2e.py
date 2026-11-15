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
               "ARM_UP_CMD": shlex.join(fake + ["up"]), "ARM_DOWN_CMD": shlex.join(fake + ["down"]),
               "MONITOR_CMD": shlex.join(fake + ["monitor"]),
               "ESTOP_DEXEC": shlex.join(fake + ["dex"]), "ARM_STOP": shlex.join(fake + ["dex", "pkill"]),
               "PATH_DIR": str(paths), "TRAIL_DIR": str(self.dir / "trails"),
               "MISSION_LOG_DIR": str(self.dir / "missions"), "ACTION_LOG": str(self.dir / "actions.jsonl"),
               "PATH_PAUSE": "0.05", "ARRIVE_PAUSE": "0.05", "ATTEMPT_S": "4", "SC": '{"noise":0}'}
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
        self.assertEqual(self.commands("up"), [["left", "8"]])
        self.assertEqual(self.commands("arm"), [["start_hold"], ["open_left"]])
        self.assertEqual(self.commands("down"), [["left", "10"]])
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

class RightSuccess(_MissionServer, unittest.TestCase):
    scenario = "success_right"

    def test_first_attempt_right(self):
        client = TA2Client(self.url)
        aid = client.run_mission(attempts=1)
        m, _ = self.finish(aid, "succeeded")
        self.assertEqual(m["hand"], "right")
        self.assertEqual(self.commands("up"), [["right", "8"]])
        self.assertEqual(self.commands("arm"), [["start_hold"], ["open_right"]])
        self.assertEqual(self.commands("down"), [["right", "10"]])


class SlipRetry(_MissionServer, unittest.TestCase):
    scenario = "slip_once"

    def test_midair_slip_retries_and_succeeds_second_attempt(self):
        aid = self.start()
        m, _ = self.finish(aid, "succeeded")
        self.assertEqual([r["result"] for r in m["attempt_results"]], ["failed", "succeeded"])
        self.assertEqual(m["attempt_results"][0].get("reason"), "slipped_on_lift")
        self.assertEqual(self.commands("up"), [["left", "8"], ["left", "8"]])
        self.assertEqual(self.commands("arm"), [["start_hold"], ["open_left"]])


class TakeoverStandaloneTests(unittest.TestCase):
    def test_standalone_takeover_lifts_verifies_and_stows(self):
        with tempfile.TemporaryDirectory(prefix="ta2-takeover-") as tmp:
            d = Path(tmp)
            out = d / "outputs" / "trial1"
            out.mkdir(parents=True)
            (out / "report.json").write_text(json.dumps({"status": "interrupted", "actuations": 142}))
            fake = [sys.executable, str(ROOT / "tests/mission_fakes.py")]
            env = {**os.environ, "FAKE_ROOT": str(d), "MISSION_SCENARIO": "success_right",
                   "OUTPUTS_DIR": str(d / "outputs"),
                   "PROBE_CMD": shlex.join([sys.executable, "-c", "print('probe ok')"]),
                   "MONITOR_CMD": shlex.join(fake + ["monitor"]),
                   "GUARDED_CMD": shlex.join(fake + ["vla"]),
                   "ARM_UP_CMD": shlex.join(fake + ["up"]),
                   "ARM_SH_CMD": shlex.join(fake + ["arm"]),
                   "ESTOP_DEXEC": shlex.join(fake + ["dex"]),
                   "PROMPT": "抓取沙包", "PORT": "8000", "RESET": "1"}
            res = subprocess.run([sys.executable, str(ROOT / "vla/vla_takeover.py"), "trial1", "10", "0.5"],
                                 env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            summary = json.loads(res.stdout.strip().splitlines()[-1])
            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["hand"], "right")
            self.assertEqual(summary["actuations"], 142)
            cmds = [json.loads(ln) for ln in (d / "commands.jsonl").read_text().splitlines()]
            self.assertEqual([c["kind"] for c in cmds], ["vla", "vla_env", "vla_interrupted", "up", "arm"])
            self.assertEqual(cmds[3]["argv"], ["right", "8"])
            self.assertEqual(cmds[4]["argv"][0], "start_hold")


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
        self.assertEqual(set(poses), {"raise", "lower", "start", "start_hold", "down5", "open_right", "close_right", "open_left", "close_left"})
        for pose in poses:
            record = client.arm_pose(pose, wait=True, poll=.03)
            self.assertEqual(record["status"], "succeeded", record)
        self.assertEqual(self.commands("down"), [["right", "5"]])

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
