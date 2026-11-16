"""B70 head-camera capture, annotation, training and detection web service."""

import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from dataset import LABELS, Dataset, write_json
from infer import Camera, Detector


APP = Path(__file__).resolve().parent
IDENTITY = re.compile(r"[0-9a-f]{32}")
TRAIN_OUTPUT = "detector"
LOG = logging.getLogger(__name__)


class Workbench:
    def __init__(self, args):
        self.root = args.data.absolute()
        self.runs = self.root / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.dataset = Dataset(self.root)
        self.camera = Camera(args.source)
        self.device = args.device
        self.train_python = os.environ.get("TA2_TRAIN_PYTHON", sys.executable)
        self.weights = args.weights.resolve()
        self.lock = threading.Lock()
        self.active_path = self.root / "active-model.json"
        self.active_model = None
        self.model_error = None
        for path in self.runs.glob("*/job.json"):
            job = self.load_job(path)
            if job and job["status"] == "running":
                write_json(path, job | {"status": "failed", "message": "服务重启，训练中断"})
        if self.active_path.is_file():
            try:
                record = json.loads(self.active_path.read_text(encoding="utf-8"))
                self.activate(record["run_id"], record["confidence"])
            except Exception as error:
                self.model_error = str(error)
                LOG.exception("Could not restore model")

    @staticmethod
    def load_job(path):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def train(self, epochs):
        if type(epochs) is not int or not 1 <= epochs <= 80:
            raise ValueError("训练轮数必须为 1–80")
        with self.lock:
            for path in self.runs.glob("*/job.json"):
                if self.load_job(path)["status"] == "running":
                    raise ValueError("已有训练正在运行")
            if not self.weights.is_file():
                raise ValueError("B70 本地基础权重缺失")
            identity = uuid.uuid4().hex
            run = self.runs / identity
            counts = self.dataset.export(run / "dataset")
            job = {"id": identity, "started_at": time.time(), "status": "running", "device": self.device,
                   "epochs": epochs, "counts": counts, "message": None}
            path = run / "job.json"
            write_json(path, job)
            try:
                with (run / "train.log").open("wb") as log:
                    process = subprocess.Popen([
                        self.train_python, "-u", str(APP / "train.py"), "--dataset", str(run / "dataset" / "data.yaml"),
                        "--weights", str(self.weights), "--run", str(run / TRAIN_OUTPUT),
                        "--epochs", str(epochs), "--device", self.device,
                    ], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            except OSError as error:
                write_json(path, job | {"status": "failed", "message": str(error)})
                raise ValueError("训练脚本启动失败") from error
            threading.Thread(target=self.finish, args=(process, path, job), daemon=True).start()
            return job

    def finish(self, process, path, job):
        code = process.wait()
        with self.lock:
            finished = code == 0 and (path.parent / TRAIN_OUTPUT / "weights" / "best.pt").is_file()
            write_json(path, job | {"status": "complete" if finished else "failed", "finished_at": time.time(),
                                    "message": None if finished else f"训练退出（{code}），请查看日志"})

    def job(self, identity):
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            raise ValueError("无效的训练任务")
        run = self.runs / identity
        job = self.load_job(run / "job.json")
        if job is None:
            raise ValueError("找不到训练任务")
        csv_path = run / TRAIN_OUTPUT / "results.csv"
        progress = None
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    try:
                        candidate = {"epoch": int(row["epoch"]), "map50": float(row["metrics/mAP50(B)"])}
                    except (ValueError, KeyError, TypeError):
                        continue
                    if math.isfinite(candidate["map50"]):
                        progress = candidate
        log_path = run / "train.log"
        tail = ""
        if log_path.is_file():
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 6000))
                tail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log.read(12000).decode("utf-8", errors="replace"))
                tail = "\n".join(tail.splitlines()[-28:])
        return job | {"progress": progress, "log": tail}

    def activate(self, identity, confidence):
        if type(confidence) not in (int, float) or not 0.05 <= confidence <= 0.99:
            raise ValueError("置信度须在 0.05–0.99 内")
        job = self.job(identity)
        if job["status"] != "complete":
            raise ValueError("选择一个已完成的训练任务")
        weights = self.runs / identity / TRAIN_OUTPUT / "weights" / "best.pt"
        if not weights.is_file():
            raise ValueError("训练权重不存在")
        with self.lock:
            try:
                model = Detector(weights)
            except Exception as error:
                LOG.exception("Model could not be loaded")
                raise ValueError(f"模型加载失败：{error}") from error
            record = {"run_id": identity, "confidence": confidence}
            write_json(self.active_path, record)
            self.camera.set_model(model, identity, confidence)
            self.active_model, self.model_error = record, None
            return record

    def state(self):
        jobs = [job for path in self.runs.glob("*/job.json") if (job := self.load_job(path))]
        jobs.sort(key=lambda job: job["started_at"], reverse=True)
        return {"classes": LABELS, "groups": self.dataset.groups(), "jobs": jobs,
                "device": self.device, "model": self.active_model, "model_error": self.model_error}


class Handler(BaseHTTPRequestHandler):
    def send(self, value, code=200, content_type="application/json; charset=utf-8"):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass

    def do_GET(self):
        try:
            self.get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as error:
            self.send({"error": str(error)}, 400)

    def get(self):
        request = urlsplit(self.path)
        work = self.server.work
        assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                  "/style.css": ("style.css", "text/css")}
        if request.path in assets:
            file, kind = assets[request.path]
            self.send((APP / file).read_bytes(), content_type=kind + "; charset=utf-8")
        elif request.path == "/api/live":
            self.send(work.camera.snapshot(parse_qs(request.query).get("since", [None])[0]))
        elif request.path == "/api/state":
            self.send(work.state())
        elif request.path.startswith("/api/jobs/"):
            self.send(work.job(request.path.rsplit("/", 1)[1]))
        elif request.path.startswith("/api/images/"):
            image = work.dataset.image(request.path.rsplit("/", 1)[1])
            if image is None:
                self.send({"error": "找不到图片"}, 404)
            else:
                self.send(image, content_type="image/jpeg")
        else:
            self.send({"error": "找不到页面"}, 404)

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send({"error": "请从工作台网页操作"}, 403)
                return
            length = int(self.headers.get("Content-Length", 0))
            if self.headers.get_content_type() != "application/json" or not 0 < length <= 65_536:
                raise ValueError("请求格式无效")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("请求格式无效")
            self.post(urlsplit(self.path).path, payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as error:
            self.send({"error": str(error)}, 400)
        except Exception:
            LOG.exception("Workbench operation failed")
            self.send({"error": "操作失败，请查看 B70 服务日志"}, 500)

    def post(self, path, data):
        work = self.server.work
        if path == "/api/groups":
            self.send({"id": work.dataset.new_group(data.get("name"), data.get("split"), data.get("negative"))}, 201)
        elif path == "/api/groups/prepare":
            work.dataset.prepare_groups()
            self.send({"ok": True})
        elif path == "/api/capture":
            frame = work.camera.frame(data.get("frame_id"))
            if frame is None:
                raise ValueError("画面已经过期，请等待头顶画面更新")
            self.send({"id": work.dataset.capture(data.get("group_id"), frame)}, 201)
        elif path == "/api/label":
            work.dataset.label(data.get("id"), data.get("boxes"), data.get("negative"))
            self.send({"ok": True})
        elif path == "/api/train":
            self.send(work.train(data.get("epochs", 80)), 202)
        elif path == "/api/activate":
            self.send(work.activate(data.get("id"), data.get("confidence", 0.5)))
        else:
            self.send({"error": "找不到操作"}, 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", default="http://127.0.0.1:8097")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8104)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with ThreadingHTTPServer((args.bind, args.port), Handler) as server:
        server.work = Workbench(args)
        LOG.info("Head detection listening on %s:%s", args.bind, args.port)
        try:
            server.serve_forever()
        finally:
            server.work.camera.stop.set()


if __name__ == "__main__":
    main()
