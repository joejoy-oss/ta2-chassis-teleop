"""Full flow v2 (steps 1-6 + step 7 voice interaction module) against fakes and a fake Mac voice service."""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from test_mission_e2e import TA2Client, _MissionServer


class FakeVoice:
    """Mac voice service: /rehearse -> listening -> (listen_result) ; /confirm -> playing -> done."""
    def __init__(self, listen_result="recognized"):
        self.state, self.polls, self.calls, self.listen_result = "idle", 0, [], listen_result
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, data):
                raw = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                fake.calls.append(("GET", self.path, self.headers.get("Host")))
                fake.polls += 1
                if fake.state == "listening" and fake.polls >= 3 and fake.listen_result != "never":
                    fake.state = fake.listen_result
                elif fake.state == "playing" and fake.polls >= 3:
                    fake.state = "done"
                self.reply({"state": fake.state, "detail": ""})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fake.calls.append(("POST", self.path, self.headers.get("Host")))
                fake.state, fake.polls = {"/rehearse": "listening", "/confirm": "playing"}.get(self.path, fake.state), 0
                self.reply({"state": fake.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def posts(self):
        return [p for m, p, _ in self.calls if m == "POST"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _VoiceServer(_MissionServer):
    listen_result = "recognized"

    def setUp(self):
        self.voice = FakeVoice(self.listen_result)
        self.extra_env = {"VOICE_BRIDGE": self.voice.url, "VOICE_LISTEN_S": "3", "VOICE_ANNOUNCE_S": "3",
                          "VOICE_RELEASE_S": "2"}
        super().setUp()

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            self.voice.close()

    def end(self, aid, expected):
        self.wait_for(lambda: not self.status()["mission"]["active"], timeout=40)
        m = self.status()["mission"]
        self.assertEqual(m["status"], expected, m)
        self.assertEqual(self.request("GET", "/api/v1/actions/" + aid)[1]["status"], expected)
        return m


class FlowV2(_VoiceServer, unittest.TestCase):
    scenario = "success_right"

    def test_v2_runs_steps_1_to_7(self):
        aid = TA2Client(self.url).run_mission(attempts=1, flow="v2")
        m = self.end(aid, "succeeded")
        self.assertEqual(m["flow"], "v2")
        self.assertEqual(m["result"], "handed over")
        self.assertEqual(list(m["durations"]), ["preflight", "g2r", "grasp", "stow", "r2g", "place", "voice"])
        self.assertEqual([s["id"] for s in m["voice"]["steps"]], ["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8"])
        self.assertEqual({s["status"] for s in m["voice"]["steps"]}, {"succeeded"})
        self.assertEqual(m["voice"]["hand"], "right")
        self.assertEqual(self.commands("arm"), [["start_hold"], ["open_right"], ["start_hold"], ["open_right"]])
        self.assertEqual(self.commands("down"), [["right", "10"], ["right", "10"]])
        self.assertEqual(self.commands("move"), [["right", "10", "0", "0"]])   # step 7 only: reach out, then lower
        self.assertEqual(len(self.commands("vla")), 2)
        self.assertEqual(self.voice.posts(), ["/rehearse", "/confirm"])
        self.assertTrue(all(h == "127.0.0.1:8096" for _, _, h in self.voice.calls))

    def test_page_lists_step_7(self):
        from urllib.request import urlopen
        page = urlopen(self.url + "/", timeout=5).read().decode()
        for text in ("开始全流程 v2", "单独运行第 7 步", "语音交互模块", "7.1", "识别口令", "7.8", "完成播报", "7.4", "前伸", "mk-done", "单独放下"):
            self.assertIn(text, page)
        caps = self.request("GET", "/api/v1/capabilities")[1]
        self.assertIn("flow", caps["actions"]["mission.run"])


class VoiceOnly(_VoiceServer, unittest.TestCase):
    scenario = "success_right"

    def test_step_7_alone(self):
        code, body = self.request("POST", "/mission/run/voice")
        self.assertEqual(code, 202, body)
        m = self.end(body["action_id"], "succeeded")
        self.assertEqual(list(m["durations"]), ["preflight", "voice"])
        self.assertEqual(self.commands("arm"), [["start_hold"], ["open_right"]])
        self.assertEqual(self.commands("down"), [["right", "10"]])
        self.assertEqual(self.commands("move"), [["right", "10", "0", "0"]])
        order = [r["kind"] for r in map(json.loads, (self.dir / "commands.jsonl").read_text().splitlines())
                 if r["kind"] in ("move", "down")]
        self.assertEqual(order, ["move", "down"])                        # reach out first, then lower
        self.assertEqual(m["voice"]["steps"][6]["status"], "succeeded")
        self.assertIn("未夹持", m["voice"]["steps"][6]["detail"])

    def test_bad_flow_rejected(self):
        code, body = self.request("POST", "/api/v1/actions", {"type": "mission.run", "params": {"flow": "v9"}})
        self.assertEqual(code, 400, body)


class VoiceRecognitionFails(_VoiceServer, unittest.TestCase):
    scenario = "success_right"
    listen_result = "failed"

    def test_stops_at_7_1(self):
        code, body = self.request("POST", "/mission/run/voice")
        self.assertEqual(code, 202, body)
        m = self.end(body["action_id"], "failed")
        self.assertIn("7.1", m["result"])
        self.assertEqual([s["status"] for s in m["voice"]["steps"]][:2], ["failed", "pending"])
        self.assertEqual(self.commands("vla"), [])
        self.assertNotIn("/confirm", self.voice.posts())


class VoiceEstop(_VoiceServer, unittest.TestCase):
    scenario = "success_right"
    listen_result = "never"

    def test_estop_while_listening(self):
        code, body = self.request("POST", "/mission/run/voice")
        self.assertEqual(code, 202, body)
        self.wait_for(lambda: self.status()["mission"]["voice"]["steps"][0]["status"] == "running")
        self.request("POST", "/api/v1/estop")
        m = self.end(body["action_id"], "aborted")
        self.assertEqual(m["voice"]["steps"][0]["status"], "failed")
        self.assertEqual(self.commands("vla"), [])


if __name__ == "__main__":
    unittest.main()


class HandoffAlone(_VoiceServer, unittest.TestCase):
    scenario = "holding_right"

    def test_reach_lower_release_with_holding_hand(self):
        code, body = self.request("POST", "/mission/run/handoff")
        self.assertEqual(code, 202, body)
        m = self.end(body["action_id"], "succeeded")
        self.assertEqual((m["flow"], m["result"], m["hand"]), ("handoff", "released", "right"))
        self.assertEqual(list(m["durations"]), ["preflight", "handoff"])
        order = [(r["kind"], r["argv"]) for r in map(json.loads, (self.dir / "commands.jsonl").read_text().splitlines())
                 if r["kind"] in ("move", "down", "arm", "vla")]
        self.assertEqual(order, [("move", ["right", "10", "0", "0"]), ("down", ["right", "10"]), ("arm", ["open_right"])])
        self.assertEqual([s["status"] for s in m["voice"]["steps"]],
                         ["skipped"] * 3 + ["succeeded"] * 4 + ["skipped"])
        self.assertEqual(self.voice.posts(), [])          # no listening, no announcement


class HandoffNeedsHeldObject(_VoiceServer, unittest.TestCase):
    scenario = "success_right"            # nothing held at preflight

    def test_refused_without_object(self):
        code, body = self.request("POST", "/api/v1/actions", {"type": "mission.run", "params": {"flow": "handoff"}})
        self.assertEqual(code, 409, body)
        self.assertIn("exactly one gripper", json.dumps(body))
        self.assertEqual(self.commands("move"), [])
