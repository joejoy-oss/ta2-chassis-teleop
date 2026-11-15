"""No ROS or robot processes: import only; mock all subprocess and worker dispatch."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
NEW = "grasp_v3_step5000"
OLD = "sandbag_v2_step4999"


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.selection = str(Path(self.temp.name) / "selected.json")
        self.web = self.load_module()
        self.web.state.update(moving=None, arm=None, vla=None, vla_model=None, estop=False)
        self.web.seq.update(names=[], route=None)
        self.subprocess = patch.object(self.web.subprocess, "run", return_value=SimpleNamespace(stdout="true\n"))
        self.check = self.subprocess.start()
        self.addCleanup(self.subprocess.stop)
        self.threads = patch.object(self.web.threading, "Thread")
        self.worker = self.threads.start()
        self.addCleanup(self.threads.stop)

    def load_module(self):
        spec = importlib.util.spec_from_file_location("web_models_under_test", ROOT / "web_move.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"VLA_SELECTION_FILE": self.selection}):
            spec.loader.exec_module(module)
        return module

    def request(self, path, method="POST"):
        handler = self.web.Handler.__new__(self.web.Handler)
        handler.path = path
        replies = []
        handler._send = lambda status, body, *args: replies.append((status, body))
        getattr(handler, "do_" + method)()
        return replies[-1]

    def test_selection_persists_without_starting_processes(self):
        self.assertEqual(self.request("/models/select/" + OLD)[0], 200)
        self.assertEqual(self.web.state["vla_selected"], OLD)
        self.assertEqual(json.loads(Path(self.selection).read_text())["model"], OLD)
        self.assertEqual(self.load_module().state["vla_selected"], OLD)
        self.check.assert_not_called()
        self.worker.assert_not_called()

    def test_busy_selection_is_refused_and_not_saved(self):
        for key, value in [("arm", "start"), ("moving", "hold cw"), ("vla", "trial"), ("estop", True)]:
            with self.subTest(key=key):
                self.web.state[key] = value
                self.assertEqual(self.request("/models/select/" + OLD)[0], 409)
                self.web.state[key] = None
        self.web.seq["names"] = ["001"]
        self.assertEqual(self.request("/models/select/" + OLD)[0], 409)
        self.assertFalse(Path(self.selection).exists())

    def test_invalid_and_stale_models_cannot_start_motion(self):
        self.assertEqual(self.request("/models/select/missing")[0], 400)
        self.assertEqual(self.request("/vla/missing")[0], 400)
        self.assertEqual(self.request("/vla/" + OLD)[0], 409)
        self.worker.assert_not_called()
        self.check.assert_not_called()

    def test_selected_models_bind_correct_service_and_worker(self):
        for model_id in (NEW, OLD):
            with self.subTest(model=model_id):
                self.web.state.update(vla=None, vla_model=None)
                self.web.select_model(model_id)
                status, result = self.request("/vla/" + model_id)
                self.assertEqual(status, 200)
                self.assertEqual(result["model"], model_id)
                self.assertIn(model_id, result["tag"])
                self.assertEqual(self.check.call_args.args[0][-1], self.web.VLA_MODELS[model_id]["container"])
                self.assertEqual(self.worker.call_args.kwargs["args"], (result["tag"], model_id))
                self.assertEqual(self.web.state["vla_model"], model_id)

    def test_worker_uses_bound_profile_and_reports_model(self):
        for model_id in (NEW, OLD):
            with self.subTest(model=model_id):
                model = self.web.VLA_MODELS[model_id]
                fake = MagicMock()
                fake.stdout = iter(['{"status":"time_limit","actuations":12}\n'])
                fake.wait.return_value = 0
                self.web.state["vla_selected"] = OLD if model_id == NEW else NEW
                with patch.object(self.web, "ensure_container"), patch.object(self.web.subprocess, "Popen", return_value=fake) as launch, patch.dict(os.environ):
                    os.environ.pop("VLA_CMD", None)
                    self.web.run_vla("bound_trial", model_id)
                self.assertEqual(launch.call_args.args[0], model["command"] + ["bound_trial", "60", "0.25"])
                self.assertIn(model["name"], self.web.state["vla_last"])
                self.assertIsNone(self.web.state["vla"])
                self.assertIsNone(self.web.state["vla_model"])

    def test_stopped_service_refuses_start(self):
        self.check.return_value.stdout = "false\n"
        with patch.dict(os.environ):
            os.environ.pop("VLA_CMD", None)
            self.assertEqual(self.request("/vla/" + NEW)[0], 503)
        self.worker.assert_not_called()

    def test_models_and_status_expose_current_selection(self):
        self.check.return_value.stdout = "/grasp-v3-step5000-policy true\n/sandbag-step4999-policy false\n"
        self.web.select_model(OLD)
        status, result = self.request("/api/models", "GET")
        self.assertEqual(status, 200)
        self.assertEqual(result["selected"], OLD)
        self.assertEqual([model["running"] for model in result["models"]], [True, False])
        status, result = self.request("/status", "GET")
        self.assertEqual(result["vla_selection"]["id"], OLD)
        self.assertIsNone(result["vla_running"])

    def test_failed_save_keeps_previous_selection(self):
        with patch.object(self.web.os, "replace", side_effect=OSError("disk unavailable")):
            self.assertEqual(self.request("/models/select/" + OLD)[0], 500)
        self.assertEqual(self.web.state["vla_selected"], NEW)

    def test_invalid_saved_selection_falls_back_to_default(self):
        for value in ("removed_model", [], None):
            Path(self.selection).write_text(json.dumps({"model": value}))
            self.assertEqual(self.load_module().state["vla_selected"], NEW)


if __name__ == "__main__":
    unittest.main()
