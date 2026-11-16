"""TA2 腕部抓取判定台（只读服务）。

* 帧源：B70 视觉查看器 :8097 /api/frame（旁路消费，不占 RTP 8890、不发机器人指令）。
* 输出：/api/grasp 判决 JSON、/api/overlay.jpg 叠加图、attempts/ 成绩单、captures/ 采集包。
* 页面：http://<B70>:8101/（判定 / 采集 / 标定 / 参数）。

安全边界：本进程只读视频帧、只写自己的 state_dir；没有 ROS、没有任何控制发布。
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import threading
import time
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

import frames
from grasp_core import (DEFAULTS, STATE_SUCCESS, GraspJudge, draw_overlay, jpeg,
                        jaw_geometry, merge, red_mask)

LOG = logging.getLogger("grasp")

SCENARIOS = {
    "A": {"name": "正常抓住", "hint": "右爪对准沙包正中，闭合，抬起 5–10cm，停 1s，再放下"},
    "B": {"name": "局部夹持", "hint": "只夹住沙包的一个角或一条边，抬起 5–10cm，停 1s"},
    "C": {"name": "在视野内没夹住", "hint": "夹爪下到沙包上方空合，或贴沙包侧面闭合；沙包要留在腕部画面里；随后抬起夹爪"},
    "D": {"name": "滑落（可选）", "hint": "先局部夹住，再缓慢抬升/晃动，让沙包在爪里下滑"},
    "E": {"name": "空场景（负样本）", "hint": "腕部画面里没有红色沙包"},
}
LABELS = {"held": "抓住", "partial": "局部夹住", "not_held": "没夹住", "ambiguous": "不确定"}


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def default_config(state_dir: str) -> dict:
    return {
        "source": "http://127.0.0.1:8097",
        "port": 8101,
        "side": "right",
        "state_dir": state_dir,
        "max_frame_age_s": 0.6,
        "ring_seconds": 20.0,
        "ring_max": 240,
        "record_max_s": 20.0,
        "save_frames": 60,
        "max_attempts": 150,
        "max_captures": 120,
        "corroborate": True,
        "operator": "",
    }


class HeadLift:
    """头部视角里的红沙包竖直位移，作为旁证（只加置信度，不参与主判）。"""

    def __init__(self, red_cfg):
        self.red = dict(red_cfg)
        self.hist = deque(maxlen=64)
        self.last = None

    def update(self, image, ts):
        if image is None:
            return None
        try:
            bag = red_mask(image, self.red)
        except Exception:
            return self.last
        if bag["ok"]:
            self.hist.append((ts, float(bag["centroid"][1]), float(bag["area_ratio"])))
        else:
            self.hist.append((ts, None, None))
        while self.hist and ts - self.hist[0][0] > 2.5:
            self.hist.popleft()
        ys = [y for _, y, _ in self.hist if y is not None]
        lift = None
        if len(ys) >= 4:
            first = float(np.median(ys[:3]))
            last = float(np.median(ys[-3:]))
            lift = first - last
        self.last = {
            "ok": bool(bag["ok"]),
            "centroid": bag["centroid"] if bag["ok"] else None,
            "area_ratio": float(bag["area_ratio"]) if bag["ok"] else None,
            "lift_px": None if lift is None else round(lift, 1),
            "bag": bag if bag["ok"] else None,
        }
        return self.last


class Worker:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = read_json(config_path, {}) or {}
        state_dir = self.config.get("state_dir", "/data/ta2/grasp")
        self.config = merge(default_config(state_dir), self.config)
        self.state_dir = Path(self.config["state_dir"])
        for sub in ("attempts", "captures", "logs"):
            (self.state_dir / sub).mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.judge = GraspJudge(self.config, jaw=self.load_jaw())
        head_red = merge(DEFAULTS["red"], self.config.get("red", {}))
        head_red = merge(head_red, DEFAULTS["red_head"])
        head_red = merge(head_red, self.config.get("red_head", {}))
        self.red_head = head_red
        self.head = HeadLift(head_red)
        self.ring = deque(maxlen=int(self.config["ring_max"]))
        self.latest = {
            "state": "starting", "success": None, "partial": None, "confidence": 0.0,
            "reasons": ["服务启动中"], "features": {}, "scores": {}, "frame": {},
            "side": self.config["side"], "updated_at": time.time(),
        }
        self.wrist_jpg = self.head_jpg = self.overlay_jpg = None
        self.stream = {"state": "starting", "age_ms": None, "fps": 0.0, "seq": None,
                       "frames": 0, "last_error": None, "session_id": None}
        self._ticks = deque(maxlen=120)
        self._last_saved = 0.0
        self._last_state = "starting"
        self._previous_seq = None
        self._previous_ts = None
        self._jaw_mtime = 0.0
        self._jaw_check = 0.0
        self._record = None
        self._record_lock = threading.Lock()

    # ---- 夹爪标定 ----
    def load_jaw(self):
        path = self.state_dir / ("jaw_%s.json" % self.config.get("side", "right"))
        data = read_json(path)
        if not data or not data.get("corridor"):
            return None
        try:
            self._jaw_mtime = path.stat().st_mtime
        except OSError:
            pass
        return data

    def watch_jaw(self, now):
        if now - self._jaw_check < 1.5:
            return
        self._jaw_check = now
        path = self.state_dir / ("jaw_%s.json" % self.config.get("side", "right"))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime != self._jaw_mtime:
            self.judge.jaw = self.load_jaw()
            LOG.info("夹爪标定已重新加载：%s", bool(self.judge.jaw))

    # ---- 主循环 ----
    def run(self):
        while not self.stop.is_set():
            now = time.time()
            self.watch_jaw(now)
            self._auto_stop_record(now)
            try:
                packet, frame = frames.fetch(self.config["source"], timeout=2.5)
            except Exception as exc:
                self._mark_stream("error", str(exc))
                self.stop.wait(0.3)
                continue
            age = frames.age_s(packet)
            seq = packet.get("seq")
            session = packet.get("session_id")
            if age > float(self.config["max_frame_age_s"]):
                self._mark_stream("stale", "画面 %.0f ms 未更新" % (age * 1000.0))
                self._set_stale(age)
                self.stop.wait(0.05)
                continue
            if seq == self._previous_seq:
                self.stop.wait(0.02)
                continue
            if session != self.stream["session_id"] or (
                    self._previous_ts is not None and now - self._previous_ts > 1.5):
                self.judge.reset()
                self.head = HeadLift(self.red_head)
            self._previous_seq = seq
            self._previous_ts = now
            try:
                wrist = frames.extract_wrist(packet, frame, self.config["side"])
                head = frames.extract_head(packet, frame) if self.config.get("corroborate", True) else None
                stamp = float(packet.get("received_at") or now)
                head_info = self.head.update(head, stamp) if head is not None else None
                head_lift = None if head_info is None else head_info.get("lift_px")
                result = self.judge.update(wrist, ts=stamp, seq=seq, head_lift=head_lift)
                result["frame"]["age_ms"] = round(age * 1000.0, 1)
                result["frame"]["regions_version"] = packet.get("regions_version")
                result["frame"]["source"] = self.config["source"]
                self._corroborate(result, head_info)
                self._publish(wrist, head, result, packet)
                self._maybe_attempt(result, now, seq)
            except Exception as exc:
                LOG.exception("process")
                self._mark_stream("error", str(exc))
            self.stop.wait(0.01)

    def _corroborate(self, result, info):
        """头部旁证：升是一个独立证据（软包摆动时补足腕图），这里只负责展示与打分。"""
        lift = None if info is None else info.get("lift_px")
        score = None
        if lift is not None:
            score = 1.0 if lift >= 8 else (0.5 if lift <= 2 else 0.75)
        result["corroboration"] = {"lift_px": lift, "score": score,
                                   "head_bag": bool(info and info.get("ok"))}
        if score is not None:
            result["scores"]["corroboration"] = round(score, 2)

    def _publish(self, wrist, head, result, packet):
        bag = getattr(self.judge, "last_bag", None)
        jaw = getattr(self.judge, "last_jaw", None)
        analysis = dict(result)
        analysis["_bag"] = bag
        analysis["_jaw"] = jaw
        overlay = draw_overlay(wrist, analysis, tracks={
            "world": getattr(self.judge, "tracks_world", None),
            "bag": getattr(self.judge, "tracks_bag", None),
        })
        head_overlay = head
        if head is not None and self.head.last and self.head.last.get("bag"):
            try:
                hb = self.head.last["bag"]
                head_overlay = head.copy()
                contours, _ = cv2.findContours(hb["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(head_overlay, contours, -1, (80, 255, 80), 2)
                lift = self.head.last.get("lift_px")
                if lift is not None:
                    cv2.putText(head_overlay, "LIFT %+.0fpx" % lift, (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
            except Exception:
                head_overlay = head
        wrist_raw = jpeg(wrist, 85)
        head_raw = jpeg(head, 80) if head is not None else None
        overlay_jpg = jpeg(overlay, 85)
        head_jpg = jpeg(head_overlay, 80) if head_overlay is not None else None
        entry = {"ts": float(packet.get("received_at") or time.time()), "seq": packet.get("seq"),
                 "wrist": wrist_raw, "head": head_raw, "overlay": overlay_jpg, "result": result}
        with self.lock:
            self.wrist_jpg, self.head_jpg, self.overlay_jpg = wrist_raw, head_jpg, overlay_jpg
            self.latest = result
            self.ring.append(entry)
            tick = time.time()
            self._ticks.append(tick)
            span = tick - self._ticks[0]
            self.stream = {
                "state": "live", "age_ms": result["frame"].get("age_ms"), "seq": packet.get("seq"),
                "fps": round((len(self._ticks) - 1) / span, 1) if span > 0.5 else 0.0,
                "frames": self.stream.get("frames", 0) + 1, "last_error": None,
                "session_id": packet.get("session_id"),
            }

    def _mark_stream(self, state, error):
        with self.lock:
            self.stream.update({"state": state, "last_error": error})

    def _set_stale(self, age):
        with self.lock:
            stale = dict(self.latest)
            stale.update({
                "state": "stale", "success": None, "partial": None, "confidence": 0.0,
                "reasons": ["画面过期（%.0f ms 未更新），不沿用旧结论" % (age * 1000.0)],
                "frame": {**stale.get("frame", {}), "age_ms": round(age * 1000.0, 1)},
                "updated_at": time.time(),
            })
            self.latest = stale

    # ---- 成绩单 ----
    def _maybe_attempt(self, result, now, seq):
        """只在判决状态发生变化时落一张成绩单，2 秒内不重复落。"""
        state = result.get("state")
        previous = self._last_state
        if state == previous:
            return
        self._last_state = state
        if state not in ("held", "held_partial", "not_held", "slipping"):
            return
        if now - self._last_saved < 2.0:
            return
        self._last_saved = now
        with self.lock:
            entries = list(self.ring)[-int(self.config["save_frames"]):]
        if len(entries) < 8:
            return
        ident = time.strftime("%Y%m%dT%H%M%S", time.localtime(now)) + "-%s" % seq
        directory = self.state_dir / "attempts" / ident
        try:
            self._write_attempt(directory, ident, entries, result)
        except Exception:
            LOG.exception("attempt save")
            return
        self._prune("attempts", int(self.config["max_attempts"]))
        LOG.info("成绩单 %s %s conf=%.2f", ident, state, result.get("confidence", 0))

    def _write_attempt(self, directory, ident, entries, result):
        (directory / "frames").mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate(entries):
            (directory / "frames" / ("%03d.jpg" % i)).write_bytes(entry["overlay"])
        cv2.imwrite(str(directory / "before.jpg"),
                    cv2.imdecode(np.frombuffer(entries[0]["overlay"], np.uint8), cv2.IMREAD_COLOR))
        cv2.imwrite(str(directory / "after.jpg"),
                    cv2.imdecode(np.frombuffer(entries[-1]["overlay"], np.uint8), cv2.IMREAD_COLOR))
        report = {
            "id": ident, "kind": "attempt", "side": self.config["side"],
            "ts": time.time(), "state": result.get("state"), "success": result.get("success"),
            "partial": result.get("partial"), "confidence": result.get("confidence"),
            "scores": result.get("scores"), "features": result.get("features"),
            "reasons": result.get("reasons"), "corroboration": result.get("corroboration"),
            "frames": len(entries),
            "frame_times": [round(e["ts"], 3) for e in entries],
            "verdicts": [e["result"].get("state") for e in entries],
            "human_label": None, "note": "",
        }
        atomic_json(directory / "report.json", report)

    # ---- 采集包 ----
    def start_record(self, scenario, label, note, operator):
        with self._record_lock:
            if self._record:
                raise ValueError("已经在录制了")
            self._record = {
                "start": time.time(), "scenario": scenario, "label": label,
                "note": note, "operator": operator, "side": self.config["side"],
            }
        LOG.info("开始录制：%s %s", scenario, label)
        return dict(self._record)

    def record_status(self):
        with self._record_lock:
            record = dict(self._record) if self._record else None
        if not record:
            return {"recording": False, "max_s": float(self.config.get("record_max_s", 20.0))}
        return {
            "recording": True, "elapsed": round(time.time() - record["start"], 1),
            "scenario": record["scenario"], "label": record["label"],
            "max_s": float(self.config.get("record_max_s", 20.0)),
        }

    def stop_record(self):
        with self._record_lock:
            record, self._record = self._record, None
        if not record:
            raise ValueError("当前没有在录制")
        with self.lock:
            entries = [e for e in self.ring if e["ts"] >= record["start"] - 0.5]
        if len(entries) < 8:
            raise ValueError("录制太短（%d 帧）" % len(entries))
        meta = self.save_capture(record["scenario"], record["label"], record["note"],
                                 record["operator"], entries=entries)
        meta["trigger"] = "record"
        meta["duration_s"] = round(entries[-1]["ts"] - entries[0]["ts"], 2)
        atomic_json(self.state_dir / "captures" / meta["id"] / "meta.json", meta)
        LOG.info("结束录制：%s %d 帧 %.1fs", meta["id"], len(entries), meta["duration_s"])
        return meta

    def _auto_stop_record(self, now):
        with self._record_lock:
            record = self._record
        if not record:
            return
        if now - record["start"] < float(self.config.get("record_max_s", 20.0)):
            return
        try:
            self.stop_record()
        except Exception:
            LOG.exception("自动结束录制失败")
            with self._record_lock:
                self._record = None

    def save_capture(self, scenario, label, note, operator, entries=None):
        if entries is None:
            with self.lock:
                entries = list(self.ring)
            if entries:
                cutoff = entries[-1]["ts"] - float(self.config["ring_seconds"])
                entries = [e for e in entries if e["ts"] >= cutoff]
        if len(entries) < 5:
            raise ValueError("还没有足够画面（%d 帧）" % len(entries))
        now = time.time()
        ident = time.strftime("%Y%m%dT%H%M%S", time.localtime(now)) + "-%s" % entries[-1]["seq"]
        directory = self.state_dir / "captures" / ("%s_%s" % (scenario, ident))
        (directory / "wrist").mkdir(parents=True, exist_ok=True)
        (directory / "head").mkdir(parents=True, exist_ok=True)
        with (directory / "results.jsonl").open("w", encoding="utf-8") as fh:
            for i, entry in enumerate(entries):
                (directory / "wrist" / ("%03d.jpg" % i)).write_bytes(entry["wrist"])
                if entry.get("head"):
                    (directory / "head" / ("%03d.jpg" % i)).write_bytes(entry["head"])
                small = {k: v for k, v in entry["result"].items()}
                small["frame"] = {k: v for k, v in small.get("frame", {}).items() if k != "source"}
                fh.write(json.dumps({"i": i, "ts": round(entry["ts"], 3), **small},
                                    ensure_ascii=False, allow_nan=False) + "\n")
        meta = {
            "id": directory.name, "scenario": scenario,
            "scenario_name": SCENARIOS.get(scenario, {}).get("name", scenario),
            "human_label": label, "note": note, "operator": operator,
            "side": self.config["side"], "ts": now, "frames": len(entries),
            "duration_s": round(entries[-1]["ts"] - entries[0]["ts"], 2),
            "judged": {"state": entries[-1]["result"].get("state"),
                       "confidence": entries[-1]["result"].get("confidence")},
        }
        atomic_json(directory / "meta.json", meta)
        self._prune("captures", int(self.config["max_captures"]))
        LOG.info("采集包 %s %s frames=%d", meta["id"], label, len(entries))
        return meta

    def _prune(self, sub, limit):
        root = self.state_dir / sub
        items = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
        for old in items[:-limit]:
            for child in sorted(old.rglob("*"), reverse=True):
                try:
                    child.unlink() if child.is_file() else child.rmdir()
                except OSError:
                    pass
            try:
                old.rmdir()
            except OSError:
                pass

    def snapshot(self):
        with self.lock:
            return dict(self.latest), dict(self.stream), self.overlay_jpg, self.head_jpg


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    # ---- 基础 ----
    def _send(self, body, code=200, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= 2_000_000:
            raise ValueError("请求体长度非法")
        return json.loads(self.rfile.read(length))

    # ---- GET ----
    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        worker = self.server.worker
        if path in ("/", "/index.html"):
            page = Path(__file__).with_name("page.html")
            return self._send(page.read_bytes(), ctype="text/html; charset=utf-8")
        if path == "/api/grasp":
            result, stream, _, _ = worker.snapshot()
            return self._send({"result": result, "stream": stream})
        if path == "/api/overlay.jpg":
            _, _, overlay, head = worker.snapshot()
            blob = head if (qs.get("view") or ["wrist"])[0] == "head" else overlay
            if not blob:
                return self._send({"error": "还没有画面"}, 503)
            return self._send(blob, ctype="image/jpeg")
        if path == "/api/raw.jpg":
            view = (qs.get("view") or ["wrist"])[0]
            blob = worker.head_jpg if view == "head" else worker.wrist_jpg
            if not blob:
                return self._send({"error": "还没有画面"}, 503)
            return self._send(blob, ctype="image/jpeg")
        if path == "/api/health":
            _, stream, _, _ = worker.snapshot()
            return self._send({"ok": stream.get("state") == "live", "stream": stream,
                               "side": worker.config["side"], "jaw": bool(worker.judge.jaw),
                               "state_dir": str(worker.state_dir)})
        if path == "/api/attempts":
            return self._send({"items": self._list("attempts", (qs.get("limit") or ["30"])[0])})
        if path.startswith("/api/attempts/"):
            parts = path[len("/api/attempts/"):].strip("/").split("/")
            ident = parts[0]
            directory = (worker.state_dir / "attempts" / ident).resolve()
            if worker.state_dir.resolve() not in directory.parents:
                return self._send({"error": "bad id"}, 400)
            if len(parts) == 1:
                data = read_json(directory / "report.json")
                return self._send(data if data else {"error": "not found"}, 200 if data else 404)
            if len(parts) == 3 and parts[1] == "frames" and re.fullmatch(r"\d{1,3}\.jpg", parts[2]):
                target = directory / "frames" / parts[2]
                if target.exists():
                    return self._send(target.read_bytes(), ctype="image/jpeg")
                return self._send({"error": "not found"}, 404)
            return self._send({"error": "not found"}, 404)
        if path == "/api/captures":
            return self._send({"items": self._list("captures", "200"),
                               "scenarios": SCENARIOS})
        if path == "/api/capture/status":
            return self._send(worker.record_status())
        if path == "/api/captures.zip":
            return self._zip_all()
        if path.startswith("/api/captures/") and path.endswith(".zip"):
            return self._zip_one(path[len("/api/captures/"):-4])
        if path == "/api/jaw":
            pathj = worker.state_dir / ("jaw_%s.json" % worker.config["side"])
            return self._send(read_json(pathj, None) or {"calibrated": False})
        if path == "/api/config":
            return self._send(worker.config)
        if path == "/api/demo":
            return self._send({"scenarios": SCENARIOS, "labels": LABELS})
        return self._send({"error": "not found", "path": path}, 404)

    def _list(self, sub, limit):
        worker = self.server.worker
        root = worker.state_dir / sub
        items = []
        try:
            directories = sorted([p for p in root.iterdir() if p.is_dir()],
                                 key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            directories = []
        for directory in directories[:int(limit)]:
            data = read_json(directory / ("report.json" if sub == "attempts" else "meta.json"))
            if not data:
                continue
            data["dir"] = directory.name
            items.append(data)
        return items

    def _zip_all(self):
        worker = self.server.worker
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            root = worker.state_dir / "captures"
            if root.exists():
                for directory in sorted(root.iterdir()):
                    if not directory.is_dir():
                        continue
                    for item in directory.rglob("*"):
                        if item.is_file():
                            zf.write(item, item.relative_to(root))
        data = buffer.getvalue()
        return self._send(data, ctype="application/zip",
                          extra={"Content-Disposition": 'attachment; filename="ta2_grasp_captures.zip"'})

    def _zip_one(self, name):
        worker = self.server.worker
        directory = (worker.state_dir / "captures" / name).resolve()
        if worker.state_dir.resolve() not in directory.parents or not directory.is_dir():
            return self._send({"error": "bad id"}, 400)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in directory.rglob("*"):
                if item.is_file():
                    zf.write(item, item.relative_to(directory))
        data = buffer.getvalue()
        return self._send(data, ctype="application/zip",
                          extra={"Content-Disposition": 'attachment; filename="%s.zip"' % name})

    # ---- POST ----
    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._send({"error": "cross-origin write rejected"}, 403)
        worker = self.server.worker
        path = urlparse(self.path).path
        try:
            data = self._json_body()
            if path == "/api/capture/start":
                side = str(data.get("side") or worker.config["side"])
                if side != worker.config["side"]:
                    worker.config = merge(worker.config, {"side": side})
                    worker.judge.side = side
                    worker.judge.jaw = worker.load_jaw()
                    atomic_json(worker.config_path, worker.config)
                record = worker.start_record(
                    str(data.get("scenario", "A"))[:1].upper() or "A",
                    str(data.get("label", "ambiguous")),
                    str(data.get("note", ""))[:500],
                    str(data.get("operator") or worker.config.get("operator") or "")[:40])
                return self._send({"ok": True, "record": record,
                                   "max_s": float(worker.config.get("record_max_s", 20.0))})
            if path == "/api/capture/stop":
                meta = worker.stop_record()
                return self._send({"ok": True, "capture": meta})
            if path == "/api/capture":
                meta = worker.save_capture(
                    str(data.get("scenario", "A"))[:1].upper() or "A",
                    str(data.get("label", "ambiguous")),
                    str(data.get("note", ""))[:500],
                    str(data.get("operator") or worker.config.get("operator") or "")[:40])
                return self._send({"ok": True, "capture": meta})
            if path == "/api/jaw":
                side = data.get("side") or worker.config["side"]
                open_pts = data.get("open") or []
                closed_pts = data.get("closed") or []
                image_size = data.get("image_size") or [640, 400]
                geo = jaw_geometry(closed_pts, image_size)
                max_gap = 0.25 * float(image_size[0])
                if geo["closed_gap_px"] > max_gap:
                    raise ValueError(
                        "空合指距 %.0f px 太大（画面宽 %.0f px 的 %.0f%%），夹爪看起来没有完全闭合；"
                        "请先把两指收到相触、再冻结画面点选" %
                        (geo["closed_gap_px"], image_size[0], geo["closed_gap_px"] / image_size[0] * 100.0))
                geo.update({"side": side, "open": open_pts, "calibrated": True,
                            "updated_at": time.time(), "note": data.get("note", "")})
                atomic_json(worker.state_dir / ("jaw_%s.json" % side), geo)
                if side == worker.config["side"]:
                    worker.judge.jaw = geo
                return self._send({"ok": True, "jaw": geo})
            if path == "/api/config":
                allowed = {"source", "side", "max_frame_age_s", "ring_seconds", "corroborate",
                           "operator", "red", "judge", "flow"}
                update = {k: v for k, v in data.items() if k in allowed}
                if update.get("side") not in (None, "right", "left"):
                    raise ValueError("side 只能是 right/left")
                worker.config = merge(worker.config, update)
                if "side" in update:
                    worker.judge.side = worker.config["side"]
                    worker.judge.jaw = worker.load_jaw()
                atomic_json(worker.config_path, worker.config)
                return self._send({"ok": True, "config": worker.config})
            if path.startswith("/api/attempts/") and path.endswith("/label"):
                ident = path[len("/api/attempts/"):-len("/label")].strip("/")
                directory = (worker.state_dir / "attempts" / ident).resolve()
                if worker.state_dir.resolve() not in directory.parents:
                    raise ValueError("bad id")
                report = read_json(directory / "report.json")
                if not report:
                    raise ValueError("没有这张成绩单")
                report["human_label"] = data.get("label")
                report["note"] = str(data.get("note", ""))[:500]
                atomic_json(directory / "report.json", report)
                return self._send({"ok": True, "report": report})
            return self._send({"error": "not found"}, 404)
        except Exception as exc:
            return self._send({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/data/ta2/grasp/config.json")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cv2.setNumThreads(2)
    config_path = Path(args.config)
    if not config_path.exists():
        state_dir = str(config_path.parent)
        atomic_json(config_path, default_config(state_dir))
    worker = Worker(config_path)
    if args.port:
        worker.config["port"] = args.port
    server = ThreadingHTTPServer(("0.0.0.0", int(worker.config["port"])), Handler)
    server.worker = worker
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    LOG.info("抓取判定台: http://0.0.0.0:%d  帧源=%s", worker.config["port"], worker.config["source"])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
