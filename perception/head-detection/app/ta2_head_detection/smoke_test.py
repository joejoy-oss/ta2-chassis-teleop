"""Check the B70 web and YOLO26 workflow with an isolated, synthetic camera."""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


BOXES = [{"class_id": index, "xyxy": [x, y, x + .18, y + .18]}
         for index, (x, y) in enumerate(((.12, .1), (.65, .12), (.13, .65), (.67, .61)))]


class Fixture(BaseHTTPRequestHandler):
    sequence = 0
    paused = False
    previous = None

    def log_message(self, *_):
        pass

    def do_GET(self):
        cls = type(self)
        if not cls.paused:
            cls.sequence += 1
            image = np.full((640, 640, 3), 26 + cls.sequence % 15, dtype=np.uint8)
            if cls.sequence % 2:
                for box, color in zip(BOXES, ((0, 0, 225), (0, 225, 0), (230, 0, 0), (0, 230, 230))):
                    x1, y1, x2, y2 = (round(v * 640) for v in box["xyxy"])
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            ok, jpg = cv2.imencode('.jpg', image)
            assert ok
            cls.previous = json.dumps({"session_id": "smoke", "seq": cls.sequence, "received_at": time.time(),
                                       "width": 640, "height": 640, "regions": {"head_left_eye": [0, 0, 1, 1]},
                                       "jpeg_b64": base64.b64encode(jpg).decode()}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(cls.previous)))
        self.end_headers()
        self.wfile.write(cls.previous)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(callback, timeout=20):
    error = None
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if (result := callback()):
                return result
        except (OSError, ValueError) as exc:
            error = exc
        time.sleep(.10)
    raise TimeoutError(f"Timed out: {error or 'condition did not become true'}")


def verify(args, source, folder):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ | {"YOLO_CONFIG_DIR": str(folder / "ultralytics"), "OMP_NUM_THREADS": "4",
                       "XDG_CACHE_HOME": str(folder / "cache"), "YOLO_AUTOINSTALL": "false"}
    log = folder / "web.log"

    def start():
        with log.open("ab") as output:
            return subprocess.Popen([
                sys.executable, "-u", str(Path(__file__).with_name("app.py")), "--data", str(folder),
                "--source", source, "--weights", str(args.weights), "--port", str(port),
                "--bind", "127.0.0.1", "--device", args.device,
            ], env=env, stdout=output, stderr=subprocess.STDOUT)

    def stop(process):
        process.terminate()
        process.wait(10)

    def api(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def live(state=None, parity=None, previous=None):
        value = api("/api/live")
        identity = value["frame_id"]
        if state and value["state"] != state or identity == previous:
            return None
        if parity is not None and (not identity or int(identity.split(":")[1]) % 2 != parity):
            return None
        return value

    process = start()
    try:
        wait_for(lambda: live("model_missing", 1))
        api("/api/groups/prepare", {})
        state = api("/api/state")
        assert len(state["groups"]) == 10
        previous = None
        for split, negative, count in (("train", False, 4), ("train", True, 2), ("val", False, 2),
                                       ("val", True, 1), ("test", False, 2), ("test", True, 1)):
            group = next(g for g in state["groups"] if g["split"] == split and bool(g["negative"]) == negative)
            for _ in range(count):
                frame = wait_for(lambda: live("model_missing", int(not negative), previous))
                previous = frame["frame_id"]
                result = api("/api/capture", {"group_id": group["id"], "frame_id": previous})
                api("/api/label", {"id": result["id"], "boxes": [] if negative else BOXES, "negative": negative})
        run = api("/api/train", {"epochs": 1})
        assert run["counts"] == {"train": 6, "val": 3, "test": 3}
        print(f"Started {args.device} training via HTTP", flush=True)

        def completed():
            job = api(f'/api/jobs/{run["id"]}')
            return job if job["status"] != "running" else None

        job = wait_for(completed, timeout=180)
        assert job["status"] == "complete", job["log"]
        assert job["progress"]["epoch"] == 1, job["progress"]
        api("/api/activate", {"id": run["id"], "confidence": .5})
        assert api("/api/state")["model"]["run_id"] == run["id"]
        assert "jpeg" in wait_for(lambda: live("ready", 1))
        distinct = set()
        until = time.monotonic() + 4
        while time.monotonic() < until:
            if value := live("ready"):
                distinct.add(value["frame_id"])
            time.sleep(.12)
        print(f"Inference updates: {len(distinct) / 4:.2f} fps on fixture", flush=True)

        Fixture.paused = True
        assert wait_for(lambda: api("/api/live")["state"] == "camera_offline", timeout=3)
        assert api("/api/live")["boxes"] == []
        Fixture.paused = False
        wait_for(lambda: live("ready", 1))
        stop(process)
        process = start()
        wait_for(lambda: live("ready", 1))
        assert len(api("/api/state")["groups"]) == 10
        assert api("/api/state")["model"]["run_id"] == run["id"]
        print("PASS: capture, positive/negative labels, group split, training, model activation, feed loss and restart", flush=True)
    except Exception:
        print(log.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
        raise
    finally:
        stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", 0), Fixture) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory(prefix="ta2-head-smoke-") as folder:
            verify(args, f"http://127.0.0.1:{server.server_port}", Path(folder))
        server.shutdown()


if __name__ == "__main__":
    main()
