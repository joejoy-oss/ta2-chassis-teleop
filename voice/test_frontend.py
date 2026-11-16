"""Offline test of the 8099 voice-module page against a fake 8088 /api/v1 and a fake Mac voice service."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [{"id": f"7.{i}", "name": f"step {i}", "uses": f"uses {i}", "status": "succeeded" if i < 3 else "running" if i == 3 else "pending",
          "detail": "已抓住（右手）" if i == 2 else ""} for i in range(1, 8)]


class Fake:
    def __init__(self):
        self.calls, self.mission = [], {"flow": "voice", "status": "running", "active": True, "step": "voice",
                                        "voice": {"hand": "right", "steps": STEPS}}
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, data):
                raw = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                fake.calls.append(("GET", self.path, None))
                if self.path == "/api/v1/state":
                    return self.reply(200, {"mission": fake.mission})
                if self.path == "/api/v1/models":
                    return self.reply(200, {"selected": "m_rtc", "models": [{"id": "m_rtc", "name": "v3 13000 RTC"}]})
                if self.path == "/status":
                    return self.reply(200, {"state": "listening"})
                self.reply(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fake.calls.append(("POST", self.path, body))
                if self.path == "/api/v1/estop":
                    return self.reply(200, {"ok": True})
                if self.path == "/api/v1/actions":
                    if fake.mission.get("active"):
                        return self.reply(409, {"error": {"code": "MISSION_ESTOP_ONLY", "message": "busy"}})
                    return self.reply(202, {"action_id": "a1", "action": {"id": "a1"}})
                self.reply(404, {})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"


class PageTests(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.host = f"127.0.0.1:{self.port}"
        env = {**os.environ, "TA2_VOICE_HOST": self.host, "TA2_CONTROL": self.fake.base + "/api/v1",
               "TA2_MAC_BRIDGE": self.fake.base, "TA2_ALLOWED_CLIENT": "127.0.0.1"}
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "b70_frontend.py")], env=env, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.1).close()
                break
            except OSError:
                time.sleep(0.05)

    def tearDown(self):
        self.proc.terminate()
        self.proc.wait(5)
        self.fake.server.shutdown()
        self.fake.server.server_close()

    def call(self, method, path, host=None):
        req = Request(f"http://127.0.0.1:{self.port}{path}", method=method, data=b"" if method == "POST" else None,
                      headers={"Host": host or self.host})
        try:
            with urlopen(req, timeout=10) as r:
                raw = r.read()
                return r.status, json.loads(raw) if "json" in r.headers.get("Content-Type", "") else raw.decode()
        except HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_page_and_access(self):
        code, page = self.call("GET", "/")
        self.assertEqual(code, 200)
        for text in ("语音交互模块", "单独运行第 7 步", "急停", "/api/voice/status"):
            self.assertIn(text, page)
        self.assertEqual(self.call("GET", "/", host="evil:1")[0], 403)
        self.assertEqual(self.call("GET", "/voice")[0], 404)          # old endpoints are gone

    def test_status_mirrors_step_7(self):
        code, d = self.call("GET", "/api/voice/status")
        self.assertEqual(code, 200)
        self.assertEqual([s["id"] for s in d["steps"]], [f"7.{i}" for i in range(1, 8)])
        self.assertEqual(d["steps"][1]["detail"], "已抓住（右手）")
        self.assertEqual((d["hand"], d["model"], d["voice_service"]["state"]), ("right", "v3 13000 RTC", "listening"))
        self.assertEqual(d["mission"]["flow"], "voice")

    def test_fallback_steps_before_first_run(self):
        self.fake.mission = {"flow": "v1", "status": "idle", "active": False, "voice": None}
        d = self.call("GET", "/api/voice/status")[1]
        self.assertEqual([s["status"] for s in d["steps"]], ["pending"] * 8)

    def test_run_and_estop(self):
        self.assertEqual(self.call("POST", "/api/voice/run")[0], 409)   # mission running -> 8088 refuses
        self.fake.mission = {"flow": "v1", "status": "idle", "active": False, "voice": None}
        code, d = self.call("POST", "/api/voice/run")
        self.assertEqual((code, d["action_id"]), (202, "a1"))
        self.assertIn(("POST", "/api/v1/actions", {"type": "mission.run", "params": {"flow": "voice"}}), self.fake.calls)
        self.assertEqual(self.call("POST", "/api/voice/estop")[0], 200)
        self.assertIn(("POST", "/api/v1/estop", {}), self.fake.calls)


if __name__ == "__main__":
    unittest.main()
