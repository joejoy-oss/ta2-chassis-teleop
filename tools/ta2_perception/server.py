"""TA2 perception worker and review UI. Never opens RTP or publishes robot commands."""
import argparse
import base64
import collections
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

from calibrate import atomic_json, fetch, capture
from core import (Board, CLASSES, PAIRS, Detector, Stereo, crop, digest, expire,
                  fuse, rigid, sandbag_geometry, transform, bottle_possible_region)

LOG = logging.getLogger("perception")


def jpg(image, quality=85):
    return base64.b64encode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode()


class Worker:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.root = Path(self.config["data_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.result = {"state": "starting", "received_at": 0, "objects": [], "grasp_candidates": []}
        self.raw, self.packet = None, None
        self.board = Board(self.config["board"])
        self.preview_board = Board(self.config["board"])
        self.corner_cache = {}
        self.preview_corner_interval_s = 1.0
        self.detector = Detector(self.config["model"], self.config["device"])
        self.stereos, self.history = {}, {}
        self.previous = {}
        self.target_history = collections.deque(maxlen=5)
        self.last_sample = 0
        self.stop = threading.Event()
        self.latencies = collections.deque(maxlen=600)
        self.session = None

    def snapshot(self):
        with self.lock:
            return expire(self.result, max_age=self.config["max_age_s"])

    def preview(self, pair):
        """Live crop of one stereo pair for the calibration tab.

        Reads the camera source directly instead of the result cache, so the operator
        sees the board even while the perception loop is stale or missing a model.
        Falls back to the last processed frame when the source is unreachable.
        """
        source, note = "live", ""
        try:
            packet, frame = fetch(self.config["source"])
        except Exception as exc:
            source, note = "cached", f"上游取帧失败（{exc}），显示缓存画面"
            with self.lock:
                packet, frame = self.packet, self.raw
            if frame is None or packet is None:
                return {"pair": pair, "views": {}, "error": f"取不到画面：{exc}"}
        now = time.time()
        age = now - packet["received_at"]
        regions = packet.get("regions") or {}
        views = {}
        for eye, name in zip(("left", "right"), PAIRS[pair]):
            box = regions.get(name)
            if box is None:
                continue
            try:
                image = crop(frame, box)
            except Exception as exc:
                LOG.warning("preview crop %s: %s", name, exc)
                continue
            views[eye] = {"name": name, "width": int(image.shape[1]), "height": int(image.shape[0]),
                          "jpeg_b64": jpg(image, 80), **self.corner_count(pair, name, image, now)}
        return {"pair": pair, "source": source, "note": note, "age_s": round(age, 3),
                "fresh": age <= self.config["max_age_s"], "views": views}

    def corner_count(self, pair, name, image, now):
        """Board corners visible in one eye, throttled so preview polling cannot starve inference."""
        key = (pair, name)
        cached = self.corner_cache.get(key)
        if cached and now - cached[0] < self.preview_corner_interval_s:
            return {"corners": cached[1], "corners_age_s": round(now - cached[0], 2)}
        try:
            found = self.preview_board.corners(image)
            count = 0 if found is None else int(len(found[1]))
        except Exception as exc:
            LOG.warning("preview corners %s: %s", name, exc)
            count = None
        self.corner_cache[key] = (now, count)
        return {"corners": count, "corners_age_s": 0.0}

    def stereo(self, pair, packet, images):
        path = self.root / "calibration" / f"{pair}.json"
        if not path.exists():
            return None, "缺少双目标定"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("board") != self.config["board"]:
                raise ValueError("标定板规格改变")
            key = digest(data) + packet["regions_version"] + str([i.shape for i in images])
            if self.stereos.get(pair, (None,))[0] != key:
                self.stereos[pair] = (key, Stereo(data, packet["regions_version"], pair, [i.shape[1::-1] for i in images]))
            return self.stereos[pair][1], None
        except Exception as exc:
            return None, str(exc)

    def stationary(self, pair, image, pose, timestamp):
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 100))
        previous = self.previous.get(pair)
        stable = False
        if previous:
            old, old_pose, old_ts = previous
            change = np.mean(cv2.absdiff(gray, old))
            delta = np.linalg.inv(old_pose) @ pose if pose is not None and old_pose is not None else None
            stable = (0 < timestamp-old_ts < .5 and change < 2.5 and delta is not None
                      and np.linalg.norm(delta[:3, 3]) < .003
                      and np.linalg.norm(cv2.Rodrigues(delta[:3, :3])[0]) < np.deg2rad(.5))
        self.previous[pair] = (gray, pose, timestamp)
        if not stable:
            self.history[pair] = timestamp
        return stable and timestamp - self.history.get(pair, timestamp) >= self.config["stationary_window_s"]

    def robot_transform(self, head_pose):
        record = self.config.get("robot_transform")
        if not record or head_pose is None:
            return None
        if (record.get("validation", {}).get("passed") is not True or not record.get("axes_verified")
                or not record.get("robot_frame") or record.get("setup_id") != self.config.get("setup_id")):
            return None
        T, reference = record.get("T_robot_table"), record.get("reference_T_head_table")
        if not rigid(T) or not rigid(reference):
            return None
        delta = np.linalg.inv(np.array(reference)) @ head_pose
        if np.linalg.norm(delta[:3, 3]) > .005 or np.linalg.norm(cv2.Rodrigues(delta[:3, :3])[0]) > np.deg2rad(.5):
            return None
        return np.array(T)

    def process(self, packet, frame):
        started = time.perf_counter()
        session = (packet["session_id"], packet["regions_version"], packet["width"], packet["height"])
        if session != self.session:
            self.history.clear(); self.previous.clear(); self.target_history.clear()
            self.session = session
        result = {"state": "2d_only", "frame_id": f'{packet["session_id"]}:{packet["seq"]}',
                  "seq": packet["seq"], "received_at": packet["received_at"],
                  "timestamp_kind": packet["timestamp_kind"], "capture_time": None,
                  "regions_version": packet["regions_version"], "frame_id_3d": "table",
                  "units": "m", "objects": [], "views": {}, "pairs": {}, "target": None,
                  "grasp_candidates": [], "unknown_obstacles": [], "requirements": [],
                  "detector_error": self.detector.error, "robot_frame": None}
        if self.detector.error:
            result["requirements"].append("现场四类分割模型尚未就绪；当前不进行物品识别")
        table_spec = self.config["table_board"]
        table_ok = (table_spec.get("verified_flush_with_table") is True and
                    abs((table_spec.get("measured_square_m") or 0)-self.config["board"]["square_m"]) < .0003)
        if not table_ok:
            result["requirements"].append("请实测标定板尺寸，并确认标记面与桌面重合")
        images, names, contexts = [], [], {}
        head_pose = None
        for pair, eyes in PAIRS.items():
            raw = [crop(frame, packet["regions"][name]) for name in eyes]
            stereo, reason = self.stereo(pair, packet, raw)
            pose, pose_error = None, None
            if stereo:
                rectified = stereo.rectify(*raw)
                # Detect board on rectified image, so pose uses rectified optical axes.
                solved = self.board.pose(rectified[0], stereo.P1[:3, :3], np.zeros(5))
                if solved:
                    Tr, pose_error = solved
                    undo = np.eye(4); undo[:3, :3] = stereo.R1.T
                    pose = undo @ Tr
                    if pair == "head": head_pose = pose
                else:
                    reason = "本帧看不到足够标定板角点；不复用旧外参"
            else:
                rectified = raw
            stationary = self.stationary(pair, rectified[0], pose, packet["received_at"])
            if stereo and pose is not None and not stationary:
                reason = "等待画面和相机位姿停稳 1 秒"
            result["pairs"][pair] = {"calibrated": stereo is not None, "table_pose": pose is not None,
                                     "stationary": bool(stationary), "reason": reason,
                                     "calibration_version": stereo.data.get("version") if stereo else None,
                                     "pose_error_px": pose_error}
            contexts[pair] = (stereo, pose, stationary, rectified)
            for name, im in zip(eyes, rectified):
                names.append(name); images.append(im)
                result["views"][name] = {"width": im.shape[1], "height": im.shape[0],
                                         "rectified": stereo is not None, "jpeg_b64": jpg(im)}
        predictions = dict(zip(names, self.detector.predict(images)))
        measures, ambiguous = [], False
        for pair, eyes in PAIRS.items():
            stereo, pose, stationary, rectified = contexts[pair]
            geometry_ready = stereo is not None and pose is not None and stationary and table_ok
            depth = None
            bags = [d for d in predictions[eyes[0]] if d["class_id"] == 0 and d["confidence"] >= .7]
            if len(bags) > 1:
                ambiguous = True
            if bags and geometry_ready:
                depth = stereo.depth(*rectified)
            for eye_index, eye in enumerate(eyes):
                for idx, detection in enumerate(predictions[eye]):
                    cls, score = detection["class_id"], detection["confidence"]
                    obj = {"id": f'{eye}:{idx}', "view": eye, "class_id": cls,
                           "class_name": CLASSES[cls] if cls == 0 or score >= .7 else "bottle_unknown",
                           "confidence": score, "mask_polygon_px": detection["polygon"],
                           "position_table_m": None, "position_robot_m": None,
                           "sigma_m": None, "obstacle": None, "reason": "仅二维识别"}
                    if cls == 0 and eye_index == 0 and depth is not None and score >= .7:
                        geo = sandbag_geometry(detection["mask"], depth[0], depth[1], np.linalg.inv(pose), stereo)
                        if geo:
                            obj.update(position_table_m=geo["position"], sigma_m=geo["sigma"], reason=None)
                            measures.append({**geo, "source": pair})
                        else:
                            obj["reason"] = "沙包内部可靠视差不足"
                    elif cls != 0:
                        # Silhouette does not prove that its bottom touches the table.
                        # Preserve an explicit unknown obstacle until contact evidence exists.
                        obj["reason"] = "瓶底接触/尺寸证据未验证；不可当作空闲空间"
                        if geometry_ready and eye_index == 0:
                            T_table_rect = np.linalg.inv(pose).copy()
                            T_table_rect[:3, :3] = T_table_rect[:3, :3] @ stereo.R1.T
                            obj["obstacle"] = bottle_possible_region(detection["polygon"], stereo.P1[:3, :3],
                                T_table_rect, self.config["bottle_dimensions"].get(CLASSES[cls]))
                        result["unknown_obstacles"].append(obj["id"])
                    result["objects"].append(obj)
        fused, reason = fuse(measures) if not ambiguous else (None, "multiple_targets")
        robot_T = self.robot_transform(head_pose)
        if fused:
            moving = bool(self.target_history and np.linalg.norm(np.array(fused["position"])-self.target_history[-1]) > .01)
            if moving: self.target_history.clear()
            self.target_history.append(np.array(fused["position"]))
            stable_target = len(self.target_history) >= 3 and max(np.linalg.norm(p-self.target_history[-1]) for p in self.target_history) <= .01
            result["target"] = {"id": "sandbag-1", "position_table_m": fused["position"], "sigma_m": fused["sigma"],
                                "sources": fused["sources"], "position_robot_m": None}
            result["state"] = "table_coordinates"
            if robot_T is not None:
                result["robot_frame"] = self.config["robot_transform"]["robot_frame"]
                result["target"]["position_robot_m"] = transform([fused["position"]], robot_T)[0].tolist()
                result["state"] = "robot_coordinates"
                for obj in result["objects"]:
                    if obj["position_table_m"] is not None:
                        obj["position_robot_m"] = transform([obj["position_table_m"]], robot_T)[0].tolist()
            blockers = []
            if not stable_target: blockers.append("target_not_stable")
            if result["unknown_obstacles"]: blockers.append("obstacles_unknown")
            if max(fused["sigma"]) > .015: blockers.append("uncertainty_too_large")
            # Unknown obstacle regions gate all candidate publication, including table-only.
            if not blockers:
                result["grasp_candidates"] = [{"id": "sandbag-1", "point_table_m": fused["point"],
                    "point_robot_m": transform([fused["point"]], robot_T)[0].tolist() if robot_T is not None else None,
                    "yaw_rad": fused["yaw_rad"], "surface_height_m": fused["point"][2],
                    "kind": "visible_surface_candidate", "collision_checked": False,
                    "note": "候选表面点；不是已验证机械臂轨迹或夹爪闭合位姿"}]
            result["candidate_blockers"] = blockers
        else:
            self.target_history.clear()
            result["candidate_blockers"] = [reason]
        if self.detector.model is None: result["state"] = "model_missing"
        result["processing_ms"] = (time.perf_counter()-started)*1000
        result["receive_to_result_ms"] = (time.time()-packet["received_at"])*1000
        self.latencies.append(result["receive_to_result_ms"])
        result["latency_p95_ms"] = float(np.percentile(self.latencies, 95))
        return result

    def run(self):
        previous = None
        while not self.stop.is_set():
            start = time.monotonic()
            try:
                packet, frame = fetch(self.config["source"])
                token = (packet["session_id"], packet["seq"])
                if token != previous:
                    previous = token
                    result = self.process(packet, frame)
                    with self.lock:
                        self.result, self.raw, self.packet = result, frame, packet
                    self.sample(packet, frame, result)
            except Exception as exc:
                LOG.warning("perception: %s", exc)
                with self.lock:
                    self.result = {"state": "error", "received_at": 0, "objects": [], "views": {},
                                   "grasp_candidates": [], "error": str(exc)}
            self.stop.wait(max(.01, .2-(time.monotonic()-start)))

    def sample(self, packet, image, result):
        if time.time()-self.last_sample < self.config["sample_interval_s"]:
            return
        self.last_sample = time.time()
        root = self.root / "replay"; root.mkdir(exist_ok=True)
        name = f'{time.time_ns()}'
        cv2.imwrite(str(root / (name + ".jpg")), image)
        small = {k: v for k, v in result.items() if k != "views"}
        atomic_json(root / (name + ".json"), {"packet": packet, "result": small})
        # Bounded rolling diagnostics (~4 hours at 10 sec); training captures are never pruned.
        for old in sorted(root.glob("*.json"))[:-1440]:
            old.with_suffix(".jpg").unlink(missing_ok=True); old.unlink()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, value, code=200, ctype="application/json; charset=utf-8"):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if isinstance(value, (dict, list)) else value
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError): pass

    def do_GET(self):
        path = urlparse(self.path).path
        worker = self.server.worker
        if path in ("/", "/perception"):
            return self.send(Path(__file__).with_name("index.html").read_bytes(), ctype="text/html; charset=utf-8")
        if path == "/api/perception": return self.send(worker.snapshot())
        if path == "/api/info":
            return self.send({"classes": CLASSES, "board": worker.config["board"],
                              "calibration_counts": {p: {s: len(list((worker.root/"calibration_captures"/p/s).glob("*.json")))
                                                         for s in ("train", "validation")} for p in PAIRS},
                              "labeled_images": len(list((worker.root/"dataset"/"labels").glob("*.txt")))})
        if path == "/api/dataset/pending":
            dataset = worker.root / "dataset"
            items = []
            for metadata in sorted((dataset/"metadata").glob("*.json")):
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                if meta.get("labeled"): continue
                image = cv2.imread(str(dataset/"images"/(metadata.stem+".png")))
                if image is not None:
                    items.append({"id": metadata.stem, "view": meta["view"], "jpeg_b64": jpg(image,95)})
                if len(items) == 6: break
            return self.send({"items": items})
        if path == "/api/calibration/preview":
            pair = (parse_qs(urlparse(self.path).query).get("pair") or ["head"])[0]
            if pair not in PAIRS: return self.send({"error": "unknown pair"}, 400)
            return self.send(worker.preview(pair))
        if path == "/api/raw":
            with worker.lock:
                im, packet = worker.raw, worker.packet
            if im is None: return self.send({"error": "没有有效帧"}, 503)
            return self.send({"packet": packet, "views": {n: jpg(crop(im, b), 95) for n, b in packet["regions"].items()}})
        if path == "/board.svg":
            target = worker.root/"board"/"charuco-print.svg"
            if target.exists(): return self.send(target.read_bytes(), ctype="image/svg+xml")
        return self.send({"error": "not found"}, 404)

    def do_POST(self):
        # Reject cross-site browser writes. Local CLI requests may omit Origin.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.send({"error": "cross-origin write rejected"}, 403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 2_000_000: raise ValueError("invalid length")
            data = json.loads(self.rfile.read(length))
            worker = self.server.worker
            path = urlparse(self.path).path
            if path == "/api/calibration/capture":
                if data.get("pair") not in PAIRS or data.get("split") not in ("train", "validation"):
                    raise ValueError("invalid pair/split")
                if not data.get("scene"): raise ValueError("请填写姿态编号")
                return self.send({"id": capture(worker.config, data["pair"], data["split"], data["scene"])})
            if path == "/api/dataset/capture":
                scene = data.get("scene", "")
                if not re.fullmatch(r"[\w-]{1,64}", scene): raise ValueError("场次名称只允许文字、数字、下划线和连字符")
                with worker.lock:
                    frame, packet = worker.raw, worker.packet
                if frame is None or time.time()-packet["received_at"] > .5: raise ValueError("没有新鲜图像")
                dataset = worker.root/"dataset"
                for directory in ("images", "labels", "metadata"): (dataset/directory).mkdir(parents=True, exist_ok=True)
                items = []
                for name in packet["regions"]:
                    identity = f'{scene}_{time.time_ns()}_{name}'
                    im = crop(frame, packet["regions"][name])
                    cv2.imwrite(str(dataset/"images"/(identity+".png")), im)
                    atomic_json(dataset/"metadata"/(identity+".json"), {"scene": scene, "view": name,
                                "frame_id": f'{packet["session_id"]}:{packet["seq"]}', "width": im.shape[1], "height": im.shape[0],
                                "labeled": False, "regions_version": packet["regions_version"]})
                    items.append({"id": identity, "view": name, "jpeg_b64": jpg(im, 95)})
                return self.send({"items": items})
            if path == "/api/dataset/label":
                identity = data["id"]
                if not re.fullmatch(r"[\w-]{1,180}", identity): raise ValueError("invalid id")
                dataset = worker.root/"dataset"
                meta_path = dataset/"metadata"/(identity+".json")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                lines = []
                for item in data["polygons"]:
                    cls, points = item["class_id"], np.array(item["points"], float)
                    if cls not in range(4) or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                        raise ValueError("invalid polygon")
                    if not np.isfinite(points).all() or np.min(points) < 0 or np.max(points) > 1:
                        raise ValueError("坐标应归一化至 0~1")
                    if cv2.contourArea(points.astype(np.float32)) < 1e-5: raise ValueError("polygon area too small")
                    lines.append(str(cls) + " " + " ".join(f"{n:.7f}" for n in points.ravel()))
                (dataset/"labels"/(identity+".txt")).write_text("\n".join(lines), encoding="utf-8")
                meta["labeled"] = True; atomic_json(meta_path, meta)
                return self.send({"ok": True, "objects": len(lines)})
            return self.send({"error": "not found"}, 404)
        except Exception as exc:
            self.send({"error": str(exc)}, 400)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    cv2.setNumThreads(4)
    worker = Worker(args.config)
    server = ThreadingHTTPServer(("0.0.0.0", worker.config["port"]), Handler)
    server.worker = worker
    thread = threading.Thread(target=worker.run, daemon=True); thread.start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: worker.stop.set(); server.server_close()


if __name__ == "__main__": main()
