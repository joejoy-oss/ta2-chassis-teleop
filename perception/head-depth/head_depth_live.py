"""Read-only live TA2 head stereo preview. No RTP receiver or ROS publishers."""

import argparse
import base64
import json
import logging
import sys
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import openvino as ov

from foundation_depth_check import metric_depth


class Preview:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.status = {"state": "compiling", "device": args.device, "precision": "FP32"}
        self.frame = None
        self.ticks = deque(maxlen=24)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def update(self, **fields):
        with self.lock:
            self.status.update(fields)

    def packet(self, since):
        with self.lock:
            status = dict(self.status)
            frame = self.frame
        if frame:
            identity, received, jpeg, _ = frame
            age = time.time() - received
            status.update(frame_id=identity, frame_age_ms=round(age * 1000))
            if age > 2 and status["state"] == "live":
                status["state"] = "stale"
            if identity != since:
                status["jpeg_b64"] = jpeg
        return status

    def depth_png(self):
        with self.lock:
            frame = self.frame if self.status["state"] == "live" else None
        if frame is None or time.time() - frame[1] > 2:
            return None, None
        return frame[0], frame[3]

    def fetch(self):
        with self.opener.open(self.args.source.rstrip("/") + "/api/frame", timeout=3) as response:
            packet = json.load(response)
        age = time.time() - packet["received_at"]
        if not 0 <= age < 1.5:
            raise ValueError(f"视频已过期（{age:.1f} 秒）")
        return packet

    def run(self):
        try:
            self.infer()
        except Exception as exc:
            logging.exception("Depth model initialization failed")
            self.update(state="error", error=str(exc))

    def infer(self):
        from core import PAIRS, Stereo, crop

        args = self.args
        calibration = json.loads(args.calibration.read_text())
        size = tuple(calibration["size"])
        stereo = Stereo(calibration, calibration["regions_version"], "head", [size, size])
        self.update(calibration=calibration["version"], input_width=args.size, input_height=args.size)
        core = ov.Core()
        start = time.monotonic()
        graph = core.read_model(str(args.model))
        names = [port.any_name for port in graph.inputs]
        if set(names) != {"left_image", "right_image"}:
            raise ValueError(f"Unexpected model inputs {names}")
        graph.reshape({name: [1, 3, args.size, args.size] for name in names})
        options = {
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32",
            "CACHE_DIR": str(args.cache),
        }
        if args.device == "CPU":
            options.update(INFERENCE_NUM_THREADS=8, ENABLE_CPU_PINNING=False)
        compiled = core.compile_model(graph, args.device, options)
        device = core.get_property(args.device, "FULL_DEVICE_NAME")
        self.update(state="waiting", device=device,
                    execution_devices=compiled.get_property("EXECUTION_DEVICES"),
                    compile_seconds=round(time.monotonic() - start, 2))
        logging.info("Compiled on %s (%s)", device, self.status["compile_seconds"])
        baseline = abs(float(stereo.P2[0, 3] / stereo.P2[0, 0]))
        view_size = (args.size, args.size)
        K = stereo.P1[:3, :3].copy()
        # OpenCV resize uses half-pixel centres; scale principal point accordingly.
        for axis, scale in enumerate([args.size / size[0], args.size / size[1]]):
            K[axis, axis] *= scale
            K[axis, 2] = (K[axis, 2] + 0.5) * scale - 0.5

        previous = None
        while True:
            loop_start = time.monotonic()
            try:
                packet = self.fetch()
                identity = f"{packet['session_id']}:{packet['seq']}"
                if packet["regions_version"] != calibration["regions_version"]:
                    raise ValueError("视频裁剪版本与头部标定不一致")
                if identity != previous:
                    previous = identity
                    im = cv2.imdecode(np.frombuffer(base64.b64decode(packet["jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
                    if im is None or im.shape[:2] != (packet["height"], packet["width"]):
                        raise ValueError("拼接图解码失败或尺寸不一致")
                    eyes = [crop(im, packet["regions"][name]) for name in PAIRS["head"]]
                    if any(eye.shape[1::-1] != size for eye in eyes):
                        raise ValueError("头部双目分辨率与标定不一致")
                    left, right = [cv2.resize(img, view_size, interpolation=cv2.INTER_AREA) for img in stereo.rectify(*eyes)]
                    self.render(compiled, identity, packet["received_at"], left, right, K, baseline, loop_start)
            except Exception as exc:
                if "CL_OUT_OF_RESOURCES" in str(exc):
                    raise
                self.ticks.clear()
                self.update(state="waiting", error=str(exc), fps=0)
                time.sleep(0.25)
            time.sleep(max(0.03, 1 / args.max_fps - (time.monotonic() - loop_start)))

    @staticmethod
    def tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        rgb = (rgb - np.float32([.485, .456, .406])) / np.float32([.229, .224, .225])
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    def render(self, compiled, identity, received, left, right, K, baseline, start):
        inputs = {"left_image": self.tensor(left), "right_image": self.tensor(right)}
        infer_start = time.monotonic()
        disparity = np.asarray(compiled(inputs)[0], dtype=np.float32).squeeze()
        infer_ms = (time.monotonic() - infer_start) * 1000
        if disparity.shape != left.shape[:2]:
            raise ValueError(f"Unexpected disparity size {disparity.shape}")
        depth, _, valid = metric_depth(disparity, K, baseline, np.any(left > 0, axis=2))
        near, far = self.args.near, self.args.far
        palette = np.clip((depth - near) / (far - near) * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(palette, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        joined = cv2.hconcat([left, colored])
        ok_jpg, jpg = cv2.imencode(".jpg", joined, [cv2.IMWRITE_JPEG_QUALITY, 83])
        ok_png, png = cv2.imencode(".png", np.rint(depth * 1000).astype(np.uint16))
        if not (ok_jpg and ok_png):
            raise ValueError("预览图编码失败")
        clock = time.monotonic()
        self.ticks.append(clock)
        fps = (len(self.ticks) - 1) / (clock - self.ticks[0]) if len(self.ticks) > 1 else 0
        with self.lock:
            self.frame = (identity, received, base64.b64encode(jpg).decode(), png.tobytes())
            self.status.update(state="live", error=None, fps=round(fps, 2),
                               inference_ms=round(infer_ms, 1), processing_ms=round((clock-start)*1000, 1),
                               valid_percent=round(float(valid.mean())*100, 1), near_m=near, far_m=far)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, body, mime, status=200, frame_id=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if frame_id:
            self.send_header("X-Frame-Id", frame_id)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        req = urlsplit(self.path)
        if req.path == "/":
            return self.reply(self.server.page, "text/html; charset=utf-8")
        if req.path in ("/api/depth", "/api/status"):
            since = parse_qs(req.query).get("since", [None])[0]
            status = self.server.preview.packet(since)
            if req.path == "/api/status":
                status.pop("jpeg_b64", None)
            return self.reply(json.dumps(status, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        if req.path == "/depth-mm.png":
            frame_id, body = self.server.preview.depth_png()
            if body:
                return self.reply(body, "image/png", frame_id=frame_id)
            return self.reply(b"No fresh depth frame", "text/plain", 503)
        self.send_error(404)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="http://127.0.0.1:8097")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--vision-tools", default="/data/ta2/tools/ta2_perception")
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--device", default="GPU")
    p.add_argument("--size", default=480, type=int)
    p.add_argument("--max-fps", default=5, type=float)
    p.add_argument("--near", default=.25, type=float)
    p.add_argument("--far", default=2, type=float)
    args = p.parse_args()
    if args.size < 224 or args.size % 32 or not 0 < args.near < args.far or args.max_fps <= 0:
        p.error("invalid input resolution, palette range or fps limit")
    sys.path.insert(0, args.vision_tools)
    logging.basicConfig(level=logging.INFO)
    cv2.setNumThreads(2)
    args.cache.mkdir(exist_ok=True, parents=True)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.page = Path(__file__).with_name("head_depth_live.html").read_bytes()
    server.preview = Preview(args)
    threading.Thread(target=server.preview.run, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
