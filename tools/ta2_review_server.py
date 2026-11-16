#!/usr/bin/env python3
"""
ta2_review_server.py — 筛选工作台（三路同屏 / 三态标注 / 采集时间 / prompt 编辑 / 一键导子集）

纯标准库实现，不依赖 venv 也能起服务；只有导出时才调 venv 里的 python（要 pyarrow）。

启动
    python3 ta2_review_server.py --scan-root /data/ta2/converted --port 8092
    /data/ta2/tools/venv/bin/python ta2_review_server.py ...   # 想用 venv 也行

设计约定
    * 数据集自动发现：扫 <scan-root>/**/meta/info.json
    * 标注与源数据集分离：<review-root>/annotations/<key>.json，绝不改原数据集
    * 视频走 Range 分片，进度条可拖动
    * 导出是后台任务，页面轮询
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_SCAN_ROOT = "/data/ta2/converted"
DEFAULT_REVIEW_ROOT = "/data/ta2/review"
DEFAULT_VENV_PYTHON = "/data/ta2/tools/venv/bin/python"
HERE = Path(__file__).resolve().parent

MARK_VALUES = ("valid", "invalid", "unmarked")
# 左右手标注：只用于统计训练数据左右手是否平衡，不影响「有效/无效」判定
HAND_VALUES = ("left", "right")
EXPORT_JOBS = {}
MERGE_JOBS = {}
EXPORT_LOCK = threading.Lock()


# ================================================================ 工具

def parse_episode_from_name(fname):
    r"""从 '12.mp4' / '12.png' 里取 episode 号。

    只取扩展名之前的数字：不能用 re.sub(r"\D","",fname)，
    因为 'mp4' 里的 '4' 会被拼进去（'12.mp4' -> 124）——踩过这个坑。
    """
    stem = fname.rsplit(".", 1)[0]
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def dataset_key(path, scan_root):
    """相对扫描根的路径做 key，稳定且可读：a__b"""
    try:
        rel = Path(path).resolve().relative_to(Path(scan_root).resolve())
    except ValueError:
        rel = Path(path).resolve().name
    key = str(rel).replace(os.sep, "__")
    return re.sub(r"[^0-9A-Za-z_.\-]", "-", key)


def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ================================================================ 数据集

class Dataset:
    def __init__(self, path, scan_root):
        self.path = Path(path).resolve()
        self.key = dataset_key(self.path, scan_root)
        self.meta = self.path / "meta"
        self.info = json.loads((self.meta / "info.json").read_text(encoding="utf-8"))
        self.episodes = read_jsonl(self.meta / "episodes.jsonl")
        self.tasks = read_jsonl(self.meta / "tasks.jsonl")
        cams = [k for k in (self.info.get("features") or {}) if "images" in k]
        self.cameras = sorted(cams)
        self.fps = float(self.info.get("fps") or 45)

    def camera_short(self, idx):
        name = self.cameras[idx]
        for suffix, short in (("head_camera", "head"), ("left_color", "left"), ("right_color", "right")):
            if name.endswith(suffix):
                return short
        return f"cam{idx}"

    def video_path(self, cam_idx, ep):
        chunk = ep // int(self.info.get("chunks_size") or 1000)
        return (self.path / "videos" / f"chunk-{chunk:03d}" / self.cameras[cam_idx]
                / f"episode_{ep:06d}.mp4")

    def parquet_path(self, ep):
        chunk = ep // int(self.info.get("chunks_size") or 1000)
        return self.path / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"

    def default_task(self):
        if self.tasks:
            return self.tasks[0].get("task") or ""
        if self.episodes:
            t = self.episodes[0].get("tasks")
            if isinstance(t, list) and t:
                return t[0]
        return ""

    def summary(self):
        return {
            "key": self.key,
            "path": str(self.path),
            "name": self.path.name,
            "fps": self.fps,
            "total_episodes": int(self.info.get("total_episodes") or len(self.episodes)),
            "total_frames": int(self.info.get("total_frames") or 0),
            "cameras": [{"index": i, "key": c, "short": self.camera_short(i)}
                        for i, c in enumerate(self.cameras)],
            "default_task": self.default_task(),
        }


NOISE_DIRS = {"reports", "backup", "backups", "old", "archive", "archives",
              "superseded", "tmp", "temp", "__pycache__"}


def is_noise(ds_path, scan_root):
    """过滤掉备份/报告/临时目录里的数据集，避免下拉列表被噪音塞满。"""
    try:
        rel = Path(ds_path).resolve().relative_to(Path(scan_root).resolve())
    except ValueError:
        return False
    for part in rel.parts[:-1]:
        if part.lower() in NOISE_DIRS or part.startswith("_"):
            return True
    return False


def discover_datasets(scan_root):
    root = Path(scan_root)
    out = []
    if not root.exists():
        return out
    for info in root.rglob("meta/info.json"):
        ds_path = info.parent.parent
        if is_noise(ds_path, scan_root):
            continue
        try:
            out.append(Dataset(ds_path, scan_root).summary())
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    out.sort(key=lambda d: d["path"])
    return out


# ================================================================ 标注

class AnnotationStore:
    def __init__(self, review_root):
        self.dir = Path(review_root) / "annotations"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def path(self, key):
        return self.dir / f"{key}.json"

    def load(self, key, dataset):
        p = self.path(key)
        with self.lock:
            data = None
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = None
            if data is None:
                data = {
                    "dataset": str(dataset.path),
                    "key": key,
                    "fingerprint": "",
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                    "default_prompt": dataset.default_task(),
                    "episodes": {},
                }
            # 数据集变化检测
            fp = "%s/%s" % (dataset.info.get("total_episodes"), dataset.info.get("total_frames"))
            if data.get("fingerprint") and data["fingerprint"] != fp:
                data["fingerprint_changed"] = True
                data["previous_fingerprint"] = data["fingerprint"]
            data["fingerprint"] = fp
            data.setdefault("default_prompt", dataset.default_task())
            data.setdefault("episodes", {})
            return data

    def save(self, key, data):
        data["updated_at"] = now_iso()
        with self.lock:
            atomic_write_json(self.path(key), data)

    def update_mark(self, key, dataset, idx, mark, note=None, hand=None):
        if mark not in MARK_VALUES:
            raise ValueError(f"mark 必须是 {MARK_VALUES}")
        if hand is not None and hand not in HAND_VALUES:
            raise ValueError(f"hand 必须是 {HAND_VALUES}")
        data = self.load(key, dataset)
        rec = data["episodes"].setdefault(str(idx), {})
        rec["mark"] = mark
        if note is not None:
            rec["note"] = note
        if hand is not None:
            rec["hand"] = hand
        rec["updated_at"] = now_iso()
        self.save(key, data)
        return rec

    def set_hand(self, key, dataset, idx, hand):
        """标注左右手。hand: 'left' | 'right' | None（清掉标注）。"""
        if hand is not None and hand not in HAND_VALUES:
            raise ValueError(f"hand 必须是 {HAND_VALUES} 或 null")
        data = self.load(key, dataset)
        rec = data["episodes"].setdefault(str(idx), {})
        if hand is None:
            rec.pop("hand", None)
        else:
            rec["hand"] = hand
        rec["updated_at"] = now_iso()
        self.save(key, data)
        return rec


# ================================================================ 来源映射

def load_sources(review_root, key, dataset=None):
    """来源映射：先按数据集 key 找，再按目录名找（兼容两种命名）。"""
    base = Path(review_root) / "episode_sources"
    cands = [base / f"{key}.jsonl"]
    if dataset is not None:
        cands.append(base / f"{dataset.path.name}.jsonl")
    for p in cands:
        if p.exists():
            rows = read_jsonl(p)
            out = {}
            for r in rows:
                if "episode_index" in r:
                    out[int(r["episode_index"])] = r
            return out
    return {}


# ================================================================ 缩略图

def ensure_sheet(dataset, cam_idx, ep, review_root):
    out_dir = Path(review_root) / "sheets" / dataset.key / dataset.camera_short(cam_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"ep{ep:06d}.png"
    if png.exists() and png.stat().st_size > 0:
        return png
    video = dataset.video_path(cam_idx, ep)
    if not video.exists():
        return None
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        dur_out = subprocess.run(
            [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, timeout=30)
        dur = float(dur_out.stdout.strip() or 0)
    except (ValueError, subprocess.SubprocessError):
        dur = 0.0
    if dur <= 0:
        return None
    rate = 9.0 / dur
    vf = f"fps={rate:.6f},scale=320:-2,tile=3x3"
    tmp = png.with_suffix(".tmp.png")
    r = subprocess.run([exe, "-y", "-v", "error", "-i", str(video), "-vf", vf,
                        "-frames:v", "1", str(tmp)],
                       capture_output=True, timeout=120)
    if r.returncode == 0 and tmp.exists():
        os.replace(tmp, png)
        return png
    return None


def ensure_thumb(dataset, cam_idx, ep, t, review_root):
    """侧栏缩略图：抽视频中间附近一帧，缩到 200 宽，缓存复用。"""
    out_dir = Path(review_root) / "thumbs" / dataset.key / dataset.camera_short(cam_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"ep{ep:06d}.png"
    if png.exists() and png.stat().st_size > 0:
        return png
    video = dataset.video_path(cam_idx, ep)
    if not video.exists():
        return None
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    tmp = png.with_suffix(".tmp.png")
    cmd = [exe, "-y", "-v", "error"]
    if t and t > 0:
        cmd += ["-ss", f"{t:.3f}"]
    cmd += ["-i", str(video), "-frames:v", "1", "-vf", "scale=200:-2", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        os.replace(tmp, png)
        return png
    return None


# ================================================================ 导出任务

def start_export(dataset, review_root, venv_python, name=None):
    with EXPORT_LOCK:
        running = [j for j in EXPORT_JOBS.values() if j["status"] == "running"]
        if running:
            raise RuntimeError(f"已有导出任务在跑：{running[0]['id']}")
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        script = HERE / "ta2_export_subset.py"
        if not script.exists():
            script = Path("/data/ta2/tools/ta2_export_subset.py")
        cmd = [venv_python, str(script), "--dataset", str(dataset.path),
               "--annotations", str(Path(review_root) / "annotations" / f"{dataset.key}.json"),
               "--review-root", str(review_root), "--json"]
        if name:
            cmd += ["--name", name]
        log_path = Path(review_root) / "exports" / f"export_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        job = {"id": job_id, "status": "running", "started_at": now_iso(),
               "dataset": str(dataset.path), "proc": proc, "log": log_path, "_fh": fh,
               "output": None, "error": None}
        EXPORT_JOBS[job_id] = job
        return job


def refresh_export_job(job):
    proc = job.get("proc")
    if proc is not None and job["status"] == "running" and proc.poll() is not None:
        try:
            job["_fh"].close()
        except OSError:
            pass
        job["finished_at"] = now_iso()
        tail = ""
        try:
            tail = job["log"].read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            pass
        job["log_tail"] = tail
        for line in reversed(tail.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    job["output"] = parsed.get("output")
                    job["result"] = parsed
                    break
                except json.JSONDecodeError:
                    continue
        job["status"] = "done" if proc.returncode == 0 else "failed"
        job["returncode"] = proc.returncode
    return job


def job_public(job):
    return {k: v for k, v in job.items() if not k.startswith("_") and k not in ("proc", "log")}


def start_merge(datasets, name, prompt, review_root, convert_root, venv_python):
    with EXPORT_LOCK:
        running = [j for j in MERGE_JOBS.values() if j["status"] == "running"]
        if running:
            raise RuntimeError(f"已有合并任务在跑：{running[0]['id']}")
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        script = HERE / "ta2_merge_train.py"
        if not script.exists():
            script = Path("/data/ta2/tools/ta2_merge_train.py")
        cmd = [venv_python, str(script), "--datasets", *datasets,
               "--name", name, "--prompt", prompt,
               "--review-root", str(review_root), "--convert-root", str(convert_root),
               "--json"]
        log_path = Path(review_root) / "train_sets" / f"merge_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        job = {"id": job_id, "status": "running", "started_at": now_iso(),
               "datasets": datasets, "name": name, "prompt": prompt,
               "proc": proc, "log": log_path, "_fh": fh, "output": None}
        MERGE_JOBS[job_id] = job
        return job


def refresh_merge_job(job):
    proc = job.get("proc")
    if proc is not None and job["status"] == "running" and proc.poll() is not None:
        try:
            job["_fh"].close()
        except OSError:
            pass
        job["finished_at"] = now_iso()
        try:
            tail = job["log"].read_text(encoding="utf-8", errors="replace")[-5000:]
        except OSError:
            tail = ""
        job["log_tail"] = tail
        for line in reversed(tail.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    job["output"] = parsed.get("output")
                    job["result"] = parsed
                    break
                except json.JSONDecodeError:
                    continue
        job["status"] = "done" if proc.returncode == 0 else "failed"
    return job


# ================================================================ HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "ta2review/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------- 基础 ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _cfg(self):
        return self.server.cfg

    def _dataset(self, key):
        cfg = self._cfg()
        for d in cfg["datasets"]():
            if d["key"] == key:
                return Dataset(d["path"], cfg["scan_root"])
        raise KeyError(key)

    def _send_file(self, path, ctype):
        path = Path(path)
        if not path.exists() or not path.is_file():
            return self._send(404, {"error": "not found"})
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng and rng.startswith("bytes="):
            spec = rng[6:].split(",")[0]
            a, _, b = spec.partition("-")
            try:
                if a:
                    start = int(a)
                    end = int(b) if b else size - 1
                elif b:
                    start = max(0, size - int(b))
                    end = size - 1
                partial = True
            except ValueError:
                start, end, partial = 0, size - 1, False
        end = min(end, size - 1)
        length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    chunk = fh.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---------- 路由 ----------
    def do_GET(self):
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        cfg = self._cfg()

        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")

        if path == "/api/datasets":
            rows = cfg["datasets"]()
            store = cfg["store"]
            for r in rows:
                try:
                    ann = store.load(r["key"], Dataset(r["path"], cfg["scan_root"]))
                    stats = {"valid": 0, "invalid": 0, "unmarked": 0,
                             "hand_left": 0, "hand_right": 0, "valid_need_hand": 0}
                    for i in range(r["total_episodes"]):
                        rec_i = ann["episodes"].get(str(i)) or {}
                        m = rec_i.get("mark") or "unmarked"
                        stats[m] = stats.get(m, 0) + 1
                        h = rec_i.get("hand")
                        if h == "left":
                            stats["hand_left"] += 1
                        elif h == "right":
                            stats["hand_right"] += 1
                        if m == "valid" and not h:
                            stats["valid_need_hand"] += 1
                    r["stats"] = stats
                    r["default_prompt"] = ann.get("default_prompt")
                except (OSError, KeyError):
                    r["stats"] = {"valid": 0, "invalid": 0, "unmarked": 0}
            return self._send(200, {"datasets": rows})

        if path == "/api/episodes":
            key = (q.get("ds") or [""])[0]
            try:
                ds = self._dataset(key)
            except KeyError:
                return self._send(404, {"error": f"数据集不存在: {key}"})
            ann = cfg["store"].load(key, ds)
            src = load_sources(cfg["review_root"], key, ds)
            episodes = []
            for e in ds.episodes:
                i = int(e["episode_index"])
                rec = ann["episodes"].get(str(i)) or {}
                s = src.get(i) or {}
                length = int(e.get("length") or 0)
                episodes.append({
                    "i": i,
                    "length": length,
                    "duration_s": round(length / ds.fps, 2) if ds.fps else None,
                    "task": (e.get("tasks") or [""])[0] if e.get("tasks") else "",
                    "mark": rec.get("mark") or "unmarked",
                    "hand": rec.get("hand") or None,
                    "prompt": rec.get("prompt") or None,
                    "note": rec.get("note") or "",
                    "record_id": s.get("record_id"),
                    "bag_started_at": s.get("bag_started_at"),
                    "quality_status": s.get("quality_status"),
                })
            return self._send(200, {
                "dataset": ds.summary(),
                "default_prompt": ann.get("default_prompt") or ds.default_task(),
                "fingerprint_changed": bool(ann.get("fingerprint_changed")),
                "sources_available": bool(src),
                "episodes": episodes,
            })

        if path.startswith("/media/"):
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) != 4:
                return self._send(400, {"error": "bad media path"})
            _, key, ci, fname = parts
            try:
                ds = self._dataset(key)
                ep = parse_episode_from_name(fname)
                if ep is None:
                    return self._send(404, {"error": "bad episode name"})
                vp = ds.video_path(int(ci), ep)
            except (KeyError, ValueError):
                return self._send(404, {"error": "not found"})
            return self._send_file(vp, "video/mp4")

        if path.startswith("/sheet/"):
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) != 4:
                return self._send(400, {"error": "bad sheet path"})
            _, key, ci, fname = parts
            try:
                ds = self._dataset(key)
                ep = parse_episode_from_name(fname)
                if ep is None:
                    return self._send(404, {"error": "bad episode name"})
                png = ensure_sheet(ds, int(ci), ep, cfg["review_root"])
            except (KeyError, ValueError):
                png = None
            if not png:
                return self._send(404, {"error": "sheet unavailable"})
            return self._send_file(png, "image/png")

        if path.startswith("/thumb/"):
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) != 4:
                return self._send(400, {"error": "bad thumb path"})
            _, key, ci, fname = parts
            try:
                t = float((q.get("t") or ["0"])[0] or 0)
            except ValueError:
                t = 0.0
            try:
                ds = self._dataset(key)
                ep = parse_episode_from_name(fname)
                if ep is None:
                    return self._send(404, {"error": "bad episode name"})
                thumb = ensure_thumb(ds, int(ci), ep, t, cfg["review_root"])
            except (KeyError, ValueError):
                thumb = None
            if not thumb:
                return self._send(404, {"error": "thumb unavailable"})
            return self._send_file(thumb, "image/png")

        if path == "/api/export":
            jobs = []
            for j in list(EXPORT_JOBS.values()):
                refresh_export_job(j)
                jobs.append(job_public(j))
            return self._send(200, {"jobs": jobs})

        if path.startswith("/api/merge/"):
            jid = path.rsplit("/", 1)[-1]
            job = MERGE_JOBS.get(jid)
            if not job:
                return self._send(404, {"error": "no such job"})
            refresh_merge_job(job)
            return self._send(200, {k: v for k, v in job.items()
                                    if not k.startswith("_") and k not in ("proc", "log")})

        if path.startswith("/api/export/"):
            jid = path.rsplit("/", 1)[-1]
            job = EXPORT_JOBS.get(jid)
            if not job:
                return self._send(404, {"error": "no such job"})
            refresh_export_job(job)
            return self._send(200, job_public(job))

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        cfg = self._cfg()
        body = self._json_body()

        if path == "/api/mark":
            key, i, mark = body.get("ds"), body.get("i"), body.get("mark")
            try:
                ds = self._dataset(key)
                rec = cfg["store"].update_mark(key, ds, int(i), mark,
                                               body.get("note"), body.get("hand"))
            except (KeyError, ValueError) as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(200, {"ok": True, "record": rec})

        if path == "/api/hand":
            key, i, hand = body.get("ds"), body.get("i"), body.get("hand")
            try:
                ds = self._dataset(key)
                rec = cfg["store"].set_hand(key, ds, int(i), hand)
            except (KeyError, ValueError) as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(200, {"ok": True, "record": rec})

        if path == "/api/mark_bulk":
            key, mark = body.get("ds"), body.get("mark")
            indices = body.get("indices")
            try:
                ds = self._dataset(key)
                if indices is None:
                    indices = list(range(len(ds.episodes)))
                data = cfg["store"].load(key, ds)
                for i in indices:
                    rec = data["episodes"].setdefault(str(int(i)), {})
                    rec["mark"] = mark
                    rec["updated_at"] = now_iso()
                cfg["store"].save(key, data)
            except (KeyError, ValueError) as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(200, {"ok": True, "modified": len(indices)})

        if path == "/api/note":
            key, i = body.get("ds"), body.get("i")
            try:
                ds = self._dataset(key)
                data = cfg["store"].load(key, ds)
                rec = data["episodes"].setdefault(str(int(i)), {})
                rec["note"] = body.get("note") or ""
                rec["updated_at"] = now_iso()
                cfg["store"].save(key, data)
            except KeyError as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(200, {"ok": True})

        if path == "/api/prompt":
            key = body.get("ds")
            try:
                ds = self._dataset(key)
                data = cfg["store"].load(key, ds)
                if "default_prompt" in body:
                    data["default_prompt"] = body["default_prompt"] or ""
                if "i" in body and body["i"] is not None:
                    rec = data["episodes"].setdefault(str(int(body["i"])), {})
                    p = body.get("prompt")
                    if p:
                        rec["prompt"] = p
                    else:
                        rec.pop("prompt", None)
                cfg["store"].save(key, data)
            except KeyError as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(200, {"ok": True, "default_prompt": data.get("default_prompt")})

        if path == "/api/build_sources":
            key, log = body.get("ds"), body.get("log")
            try:
                ds = self._dataset(key)
            except KeyError as exc:
                return self._send(400, {"error": str(exc)})
            script = HERE / "ta2_episode_sources.py"
            if not script.exists():
                script = Path("/data/ta2/tools/ta2_episode_sources.py")
            r = subprocess.run([cfg["venv_python"], str(script),
                                "--dataset", str(ds.path), "--log", log,
                                "--review-root", str(cfg["review_root"])],
                               capture_output=True, text=True, timeout=300)
            return self._send(200 if r.returncode == 0 else 500,
                              {"ok": r.returncode == 0,
                               "stdout": r.stdout[-3000:], "stderr": r.stderr[-2000:]})

        if path == "/api/merge":
            datasets = [str(x) for x in (body.get("datasets") or [])]
            name = (body.get("name") or "").strip()
            prompt = (body.get("prompt") or "").strip()
            if not datasets:
                return self._send(400, {"error": "没有选择数据集"})
            if not name:
                return self._send(400, {"error": "训练集名字必填"})
            if not prompt:
                return self._send(400, {"error": "统一 prompt 必填"})
            try:
                job = start_merge(datasets, name, prompt, cfg["review_root"],
                                  cfg["scan_root"], cfg["venv_python"])
            except RuntimeError as exc:
                return self._send(409, {"error": str(exc)})
            return self._send(200, {"id": job["id"], "status": job["status"]})

        if path == "/api/export":
            key = body.get("ds")
            try:
                ds = self._dataset(key)
                job = start_export(ds, cfg["review_root"], cfg["venv_python"], body.get("name"))
            except (KeyError, RuntimeError) as exc:
                return self._send(409 if isinstance(exc, RuntimeError) else 400, {"error": str(exc)})
            return self._send(200, job_public(job))

        return self._send(404, {"error": "not found"})


# ================================================================ 页面

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>筛选工作台 · TA2</title>
<style>
 :root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;
       --ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#4c8dff}
 *{box-sizing:border-box}
 html,body{height:100%}
 body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;overflow:hidden}
 header{height:52px;background:#12151b;border-bottom:1px solid var(--line);padding:0 16px;
        display:flex;gap:12px;align-items:center;flex-wrap:nowrap;overflow-x:auto}
 h1{font-size:15px;margin:0;font-weight:600;white-space:nowrap}
 .pill{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;color:var(--dim);white-space:nowrap}
 .pill b{color:var(--fg)}
 #s-valid b{color:var(--ok)} #s-invalid b{color:var(--bad)} #s-unmarked b{color:var(--warn)}
 button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:6px 12px;font-size:13px;cursor:pointer;white-space:nowrap}
 button:disabled{opacity:.45;cursor:not-allowed}
 button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
 button.ok{background:#238636}button.bad{background:#a4262c}button.un{background:#3a3f4b}
 button.tiny{padding:3px 9px;font-size:12px}
 input,select{background:var(--card);border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:5px 8px;font-size:13px;font-family:inherit}
 .spacer{flex:1}

 #app{display:flex;height:calc(100vh - 52px)}
 #side{width:290px;min-width:290px;border-right:1px solid var(--line);display:flex;flex-direction:column;background:#12151b}
 .side-head{padding:8px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:6px}
 .seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;font-size:12px}
 .seg button{flex:1;background:transparent;color:var(--dim);border-radius:0;padding:5px 0}
 .seg button.on{background:#22304a;color:#fff}
 #thumbs{flex:1;overflow-y:auto;padding:6px}
 .item{display:flex;gap:8px;padding:6px;border-radius:8px;cursor:pointer;border:1px solid transparent;margin-bottom:4px;align-items:center}
 .item:hover{background:#1b1f28}
 .item.cur{background:#1e2a44;border-color:#3b5bdb}
 .item img{width:96px;height:48px;object-fit:cover;border-radius:5px;background:#0b0d11;flex:0 0 auto}
 .item .t{font-size:12px;line-height:1.35;min-width:0}
 .item .t .e{font-weight:600}
 .item .t .m{color:var(--dim);font-size:11px}
 .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:middle}
 .dot.valid{background:var(--ok)}.dot.invalid{background:var(--bad)}.dot.unmarked{background:var(--warn)}
 /* 左右手标注：与 有效/无效 的颜色体系区分开 */
 .hand{display:inline-block;font-size:11px;line-height:1;padding:2px 7px;border-radius:999px;border:1px solid;white-space:nowrap}
 .hand.left{color:#4c8dff;border-color:#4c8dff;background:rgba(76,141,255,.14)}
 .hand.right{color:#ff9f1a;border-color:#ff9f1a;background:rgba(255,159,26,.14)}
 .hand.none{color:var(--warn);border-color:var(--warn);background:rgba(210,153,34,.12)}
 #s-hand b{color:#ff9f1a}
 .t .h{font-size:11px;margin-top:2px}
 .t .h .left{color:#4c8dff}.t .h .right{color:#ff9f1a}.t .h .none{color:var(--dim)}
 .ok[data-hand="left"].on,.ok[data-hand="right"].on{outline:2px solid currentColor}
 .ghost.tiny.on{outline:2px solid var(--accent);color:var(--fg)}
 .item.valid{box-shadow:inset 3px 0 0 var(--ok)}
 .item.invalid{box-shadow:inset 3px 0 0 var(--bad);opacity:.72}
 .item.unmarked{box-shadow:inset 3px 0 0 var(--warn)}

 #main{flex:1;overflow-y:auto;padding:14px 18px 30px}
 .empty{color:var(--dim);text-align:center;padding:80px 20px}
 .ep-head{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
 .ep-head .big{font-size:20px;font-weight:700}
 .meta{color:var(--dim);font-size:12px}
 .meta b{color:var(--fg);font-weight:500}
 .vids{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
 .vbox{background:#0b0d11;border:1px solid var(--line);border-radius:8px;overflow:hidden}
 .vbox .cap{font-size:11px;color:var(--dim);padding:4px 8px;background:#141821;display:flex;justify-content:space-between}
 video{width:100%;display:block;background:#000;aspect-ratio:2/1;object-fit:contain}
 .ep-body{display:grid;grid-template-columns:220px 1fr;gap:14px;margin-top:12px}
 .ep-body img{width:220px;border:1px solid var(--line);border-radius:6px;cursor:zoom-in;background:#0b0d11}
 .fields{display:flex;flex-direction:column;gap:7px}
 .fields label{font-size:11px;color:var(--dim)}
 .row{display:flex;gap:8px;align-items:center}
 .nav{display:flex;gap:10px;align-items:center;margin-top:14px}
 .kbd{font:11px ui-monospace,monospace;background:#0b0d11;border:1px solid var(--line);border-radius:4px;padding:1px 5px;color:var(--dim)}
 .nav a{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:12px;color:var(--fg);text-decoration:none;white-space:nowrap}
 .nav a:hover{border-color:var(--accent)}
 #toast{position:fixed;right:18px;bottom:18px;background:#1f2937;border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:13px;display:none;max-width:420px;z-index:60}
 #modal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:50}
 #modal .box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;width:520px;max-width:92vw}
#modal-merge{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:50}
#modal-merge .box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;max-width:92vw}
#modal-merge label.meta{color:var(--dim);font-size:12px}
 #lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:70;cursor:zoom-out}
 #lightbox img{max-width:94vw;max-height:94vh}
</style>
</head>
<body>
<header>
  <h1>筛选工作台</h1>
  <span class="nav"><a href="http://192.168.31.228:8095" target="_blank">转化台</a>
    <a href="http://192.168.31.228:8090" target="_blank">下载台</a></span>
  <select id="ds"></select>
  <span class="pill">条数 <b id="s-total">-</b></span>
  <span class="pill" id="s-valid">有效 <b>0</b></span>
  <span class="pill" id="s-invalid">无效 <b>0</b></span>
  <span class="pill" id="s-unmarked">未标记 <b>0</b></span>
  <span class="pill" id="s-hand" title="点击看左右手平衡详情，可筛选「待标左右」">左手 <b>0</b> / 右手 <b>0</b></span>
  <span class="pill" id="s-pos">第 - / - 条</span>
  <span class="spacer"></span>
  <input id="default-prompt" placeholder="统一 prompt" size="20">
  <button class="ghost tiny" id="save-prompt">保存</button>
  <button id="btn-export">导出有效数据</button>
  <button class="ghost" id="btn-merge">合并训练集</button>
</header>

<div id="app">
  <aside id="side">
    <div class="side-head">
      <input id="search" placeholder="搜 episode 号 / record_id">
      <div class="seg" id="filter">
        <button data-f="all" class="on">全部</button>
        <button data-f="unmarked">未标记</button>
        <button data-f="valid">有效</button>
        <button data-f="invalid">无效</button>
        <button data-f="need_hand">待标左右</button>
      </div>
      <label class="meta"><input type="checkbox" id="auto-next" checked> 标完后自动下一条</label>
    </div>
    <div id="thumbs"></div>
  </aside>

  <section id="main">
    <div id="empty" class="empty">左侧点选一条开始审阅<br><br>
      <span class="kbd">j</span>/<span class="kbd">k</span> 上下条　
      <span class="kbd">1</span>有效 <span class="kbd">2</span>无效 <span class="kbd">3</span>未标记　
      <span class="kbd">Q</span>左手 <span class="kbd">E</span>右手　
      <span class="kbd">空格</span>播放</label></div>
    <div id="view" style="display:none"></div>
  </section>
</div>

<div id="modal-merge"><div class="box" style="width:640px">
  <h3 style="margin-top:0">合并多个数据集为训练集</h3>
  <p class="meta">每个数据集只取标记为「有效」的条目，合并后统一成一个 prompt。产出到 <code>/data/ta2/review/train_sets/&lt;名字&gt;/</code>。</p>
  <div style="max-height:300px;overflow:auto;border:1px solid var(--line);border-radius:8px;padding:8px;margin:10px 0">
    <div id="merge-list"></div>
  </div>
  <div class="row" style="margin-bottom:8px">
    <label class="meta" style="width:80px">训练集名</label>
    <input id="merge-name" size="28" placeholder="train_all">
  </div>
  <div class="row">
    <label class="meta" style="width:80px">统一 prompt</label>
    <input id="merge-prompt" size="40" placeholder="抓取红色沙包">
  </div>
  <div class="row" style="justify-content:flex-end;margin-top:14px">
    <button class="ghost" id="mg-cancel">取消</button>
    <button id="mg-go">开始合并</button>
  </div>
  <div id="mg-progress" class="meta" style="margin-top:10px"></div>
</div></div>

<div id="toast"></div>
<div id="lightbox"><img id="lb-img" alt=""></div>
<div id="modal"><div class="box">
  <h3 style="margin-top:0">导出训练子集</h3>
  <p class="meta" id="m-text"></p>
  <p class="meta">只包含「有效」条目；无效与未标记都会排除。产出为新目录（视频硬链接，几乎不占空间），不改动源数据集。</p>
  <div class="row" style="justify-content:flex-end;margin-top:14px">
    <button class="ghost" id="m-cancel">取消</button>
    <button id="m-go">开始导出</button>
  </div>
  <div id="m-progress" class="meta" style="margin-top:12px"></div>
</div></div>

<script>
const $ = s => document.querySelector(s);
let DS = [], CURINFO = null, EPS = [], VIEW = [], CURSOR = -1;
let FILTER = 'all', SEARCH = '';
let HAND_STATS = {left:0, right:0, need:0};

function toast(m, ms=2400){ const t=$('#toast'); t.textContent=m; t.style.display='block';
  clearTimeout(t._h); t._h=setTimeout(()=>t.style.display='none', ms); }
function fmtDur(s){ return s==null? '-' : (s<60? s.toFixed(1)+'s' : Math.floor(s/60)+'m'+(s%60).toFixed(0)+'s'); }
function esc(x){ return String(x==null?'':x).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;'); }

async function api(path, body){
  const opt = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  const d = await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(d.error || r.status);
  return d;
}
function dsKey(){ return CURINFO.dataset.key; }
function videoURL(ci, ep){ return `/media/${encodeURIComponent(dsKey())}/${ci}/${ep}.mp4`; }
function sheetURL(ci, ep){ return `/sheet/${encodeURIComponent(dsKey())}/${ci}/${ep}.png`; }
function thumbURL(ci, ep, dur){ return `/thumb/${encodeURIComponent(dsKey())}/${ci}/${ep}.png?t=${(dur||0)*0.5}`; }

/* ---------------- 数据加载 ---------------- */
async function loadDatasets(){
  const d = await api('/api/datasets');
  DS = d.datasets || [];
  const sel = $('#ds'); sel.innerHTML = '';
  DS.forEach(x => { const o=document.createElement('option'); o.value=x.key;
    o.textContent = `${x.name} (${x.total_episodes})`; sel.appendChild(o); });
  if(!DS.length){ $('#empty').textContent='没有发现数据集'; return; }
  const want = new URLSearchParams(location.search).get('ds');
  const prefer = DS.find(x => x.key === want)
              || DS.find(x => x.key.includes('grasp_red_sandbag'))
              || DS[DS.length-1];
  sel.value = prefer.key;
  await selectDataset(prefer.key);
}

async function selectDataset(key){
  const d = await api('/api/episodes?ds='+encodeURIComponent(key));
  CURINFO = d; EPS = d.episodes;
  $('#s-total').textContent = EPS.length;
  $('#default-prompt').value = d.default_prompt || '';
  if(d.fingerprint_changed) toast('数据集有更新，新增条目默认未标记', 4000);
  if(!d.sources_available) toast('没有采集时间映射，时间显示为 —', 4000);
  CURSOR = -1;
  applyFilter(true);
  updateStats();
}

function updateStats(){
  const c = {valid:0, invalid:0, unmarked:0};
  EPS.forEach(e => c[e.mark] = (c[e.mark]||0)+1);
  $('#s-valid').innerHTML = '有效 <b>'+c.valid+'</b>';
  $('#s-invalid').innerHTML = '无效 <b>'+c.invalid+'</b>';
  $('#s-unmarked').innerHTML = '未标记 <b>'+c.unmarked+'</b>';

  // 左右手统计：只看「有效」条目（无效的不会进训练集）
  const L = EPS.filter(e => e.hand==='left').length;
  const R = EPS.filter(e => e.hand==='right').length;
  const need = EPS.filter(e => e.mark==='valid' && !e.hand).length;
  $('#s-hand').innerHTML = '左手 <b>'+L+'</b> / 右手 <b>'+R+'</b>'
      + (need? ' <span style="color:var(--warn)">待标 '+need+'</span>' : '');
  HAND_STATS = {left:L, right:R, need};
}

function balanceText(){
  const {left:L, right:R, need} = HAND_STATS;
  const tagged = L + R;
  if(!tagged) return `还没有任何左右手标注（有效 ${EPS.filter(e=>e.mark==='valid').length} 条）`;
  const pctL = L/tagged*100, pctR = R/tagged*100;
  const ratio = Math.min(L,R) / Math.max(L,R);      // 1.0 = 完全平衡
  let verdict = '基本平衡';
  if(ratio < 0.5) verdict = '⚠ 明显不平衡';
  else if(ratio < 0.75) verdict = '轻微不平衡';
  return `已标注 ${tagged} 条：左手 ${L} (${pctL.toFixed(1)}%) / 右手 ${R} (${pctR.toFixed(1)}%)`
       + `　平衡度 ${(ratio*100).toFixed(0)}% → ${verdict}`
       + (need? `　｜　另有 ${need} 条「有效」还没标左右` : '');
}

function matches(e){
  // need_hand 是组合条件：已判有效、但还没标左右手
  if(FILTER==='need_hand'){ if(!(e.mark==='valid' && !e.hand)) return false; }
  else if(FILTER!=='all' && e.mark!==FILTER) return false;
  if(SEARCH){
    const s = SEARCH.toLowerCase();
    if(!String(e.i).includes(s) && !(e.record_id||'').toLowerCase().includes(s)) return false;
  }
  return true;
}

function applyFilter(resetCursor){
  const prevEp = (VIEW[CURSOR]||{}).i;
  VIEW = EPS.filter(matches);
  renderThumbs();
  if(resetCursor || VIEW.length===0){ CURSOR = VIEW.length? 0 : -1; }
  else {
    const idx = VIEW.findIndex(e => e.i === prevEp);
    CURSOR = idx >= 0 ? idx : Math.min(CURSOR, VIEW.length-1);
  }
  showCurrent();
}

/* ---------------- 侧栏 ---------------- */
function handText(e){ return e.hand==='left' ? '左手' : e.hand==='right' ? '右手' : '未标左右'; }
function handTag(e){
  const cls = e.hand || 'none';
  return `<span class="hand ${cls}">${handText(e)}</span>`;
}

function renderThumbs(){
  const box = $('#thumbs');
  const frag = document.createDocumentFragment();
  VIEW.forEach((e, idx) => {
    const div = document.createElement('div');
    div.className = 'item ' + e.mark + (idx===CURSOR? ' cur':'');
    div.dataset.idx = idx;
    div.innerHTML = `
      <img loading="lazy" src="${thumbURL(0, e.i, e.duration_s)}" onerror="this.style.opacity=.15">
      <div class="t">
        <div class="e"><span class="dot ${e.mark}"></span>ep ${e.i}</div>
        <div class="h ${e.hand||'none'}">${e.hand==='left'?'左手':e.hand==='right'?'右手':'—'}</div>
        <div class="m">${fmtDur(e.duration_s)} · ${e.length} 帧</div>
        <div class="m">${(e.bag_started_at||'—').slice(5)}</div>
      </div>`;
    div.onclick = () => { CURSOR = idx; showCurrent(); };
    frag.appendChild(div);
  });
  box.replaceChildren(frag);
}

/* ---------------- 主视图 ---------------- */
function showCurrent(){
  const e = VIEW[CURSOR];
  if(!e){ $('#view').style.display='none'; $('#empty').style.display='block'; 
          $('#s-pos').textContent='第 - / - 条'; return; }
  $('#empty').style.display='none';
  $('#view').style.display='block';
  $('#s-pos').textContent = `第 ${CURSOR+1} / ${VIEW.length} 条`;
  document.querySelectorAll('#thumbs .item').forEach((el,i)=>el.classList.toggle('cur', i===CURSOR));
  const cur = document.querySelector('#thumbs .item.cur');
  if(cur) cur.scrollIntoView({block:'nearest'});

  const cams = CURINFO.dataset.cameras;
  const vids = cams.slice(0,3).map((c,ci) => `
    <div class="vbox">
      <div class="cap"><span>${c.short}</span><span>${c.key.split('.').pop()}</span></div>
      <video preload="metadata" playsinline muted src="${videoURL(ci, e.i)}"></video>
    </div>`).join('');

  $('#view').innerHTML = `
    <div class="ep-head">
      <span class="big">ep ${e.i}</span>
      <span class="meta">时长 <b>${fmtDur(e.duration_s)}</b>　帧 <b>${e.length}</b></span>
      <span class="meta">采集 <b>${e.bag_started_at||'—'}</b></span>
      <span class="meta">记录 <b>${e.record_id||'—'}</b></span>
      <span class="spacer"></span>
      <button class="ok" data-mark="valid">有效 <span class="kbd">1</span></button>
      <button class="bad" data-mark="invalid">无效 <span class="kbd">2</span></button>
      <button class="un" data-mark="unmarked">未标记 <span class="kbd">3</span></button>
      <span class="spacer" style="flex:0 0 14px"></span>
      <span id="hand-tag" class="hand ${e.hand||'none'}">${handText(e)}</span>
      <button class="ghost tiny" data-hand="left">左手 <span class="kbd">Q</span></button>
      <button class="ghost tiny" data-hand="right">右手 <span class="kbd">E</span></button>
    </div>
    <div class="vids">${vids}</div>
    <div class="ep-body">
      <img id="sheet" src="${sheetURL(cams[0]?cams[0].index:0, e.i)}" title="点击看大图">
      <div class="fields">
        <label>本条 prompt 覆盖（留空 = 用顶部统一 prompt）</label>
        <div class="row"><input id="ep-prompt" size="52" value="${esc(e.prompt)}">
          <button class="ghost tiny" id="save-ep-prompt">保存</button></div>
        <label>备注</label>
        <div class="row"><input id="ep-note" size="52" value="${esc(e.note)}">
          <button class="ghost tiny" id="save-note">保存</button></div>
        <div class="meta">${e.quality_status? '质检 '+e.quality_status+'　':''}时间精度：bag 级（同一条记录切出的多条共享）</div>
      </div>
    </div>
    <div class="nav">
      <button class="ghost" id="prev">← 上一条</button>
      <button class="ghost" id="next">下一条 →</button>
      <span class="meta">${CURSOR+1} / ${VIEW.length}</span>
      <span class="spacer"></span>
      <span class="meta"><span class="kbd">j</span>/<span class="kbd">k</span> 切换　<span class="kbd">1</span>/<span class="kbd">2</span>/<span class="kbd">3</span> 标记　<span class="kbd">Q</span> 左手 <span class="kbd">E</span> 右手　<span class="kbd">空格</span> 播放</span>
    </div>`;

  bindSync([...$('#view').querySelectorAll('video')]);
  $('#view').querySelectorAll('[data-mark]').forEach(b => b.onclick = () => setMark(e, b.dataset.mark));
  $('#view').querySelectorAll('[data-hand]').forEach(b => {
    b.classList.toggle('on', e.hand === b.dataset.hand);   // 当前已标的那个高亮
    b.onclick = () => setHand(e, b.dataset.hand);
  });
  $('#prev').onclick = () => move(-1);
  $('#next').onclick = () => move(1);
  $('#sheet').onclick = () => { $('#lb-img').src = $('#sheet').src; $('#lightbox').style.display='flex'; };
  $('#save-ep-prompt').onclick = async () => {
    const v = $('#ep-prompt').value.trim();
    await api('/api/prompt', {ds:dsKey(), i:e.i, prompt:v});
    e.prompt = v || null; toast('ep '+e.i+' prompt 已保存');
  };
  $('#save-note').onclick = async () => {
    const v = $('#ep-note').value.trim();
    await api('/api/note', {ds:dsKey(), i:e.i, note:v});
    e.note = v; toast('ep '+e.i+' 备注已保存');
  };
}

function bindSync(vids){
  let master = null;
  const align = src => { vids.forEach(v => { if(v!==src && v.readyState>0 && Math.abs(v.currentTime-src.currentTime)>0.15) v.currentTime = src.currentTime; }); };
  vids.forEach(v => {
    v.addEventListener('play', () => { master=v; vids.forEach(o => { if(o!==v){ o.currentTime=v.currentTime; o.play().catch(()=>{}); } }); });
    v.addEventListener('pause', () => { vids.forEach(o => { if(o!==v && !o.paused) o.pause(); }); });
    v.addEventListener('seeking', () => align(v));
    v.addEventListener('timeupdate', () => { if(master===v) align(v); });
  });
}

function move(delta){
  if(!VIEW.length) return;
  const n = CURSOR + delta;
  if(n < 0){ toast('已经是第一条'); return; }
  if(n >= VIEW.length){ toast('已经是最后一条'); return; }
  CURSOR = n; showCurrent();
}

/* 标记/标注之后要不要前进：抽成公共逻辑，setMark 与 setHand 都用 */
function afterMark(e){
  if($('#auto-next').checked && FILTER==='all'){
    if(CURSOR < VIEW.length-1){ setTimeout(()=>move(1), 120); }
    return;
  }
  if(FILTER!=='all' && !matches(e)){
    // 当前过滤条件下已不符合，移出列表
    const idxInView = VIEW.findIndex(x => x.i === e.i);
    if(idxInView >= 0) VIEW.splice(idxInView,1);
    renderThumbs();
    CURSOR = Math.min(CURSOR, VIEW.length-1);
    showCurrent();
  }
}

/* 只就地刷新当前这条在侧栏的样子，不重排列表（重排会打断连按手感） */
function refreshThumb(e){
  const item = document.querySelectorAll('#thumbs .item')[CURSOR];
  if(!item) return;
  item.className = 'item ' + e.mark + ' cur';
  const dot = item.querySelector('.dot');
  if(dot) dot.className = 'dot ' + e.mark;
  const h = item.querySelector('.h');
  if(h){
    h.className = 'h ' + (e.hand||'none');
    h.textContent = e.hand==='left' ? '左手' : e.hand==='right' ? '右手' : '—';
  }
}

async function setMark(e, mark){
  await api('/api/mark', {ds:dsKey(), i:e.i, mark});
  e.mark = mark;
  updateStats();
  refreshThumb(e);

  // 关键：判「有效」但还没标左右 → 停住等 Q/E，不自动跳下一条
  if(mark==='valid' && !e.hand){
    toast('已判有效：按 Q 左手 / E 右手，标完自动下一条', 5000);
    return;
  }
  afterMark(e);
}

async function setHand(e, hand){
  await api('/api/hand', {ds:dsKey(), i:e.i, hand});
  e.hand = hand;
  updateStats();
  refreshThumb(e);
  const tag = $('#hand-tag');
  if(tag){ tag.className = 'hand ' + hand; tag.textContent = hand==='left' ? '左手' : '右手'; }
  toast((hand==='left' ? '左手' : '右手') + ' 已标注');

  // 只有「有效」条目在补完左右后才自动前进
  if(e.mark==='valid') afterMark(e);
}

/* ---------------- 事件 ---------------- */
$('#ds').onchange = ev => selectDataset(ev.target.value);
$('#filter').onclick = ev => {
  const b = ev.target.closest('button'); if(!b) return;
  FILTER = b.dataset.f;
  $('#filter').querySelectorAll('button').forEach(x => x.classList.toggle('on', x===b));
  applyFilter(true);
};
$('#search').oninput = ev => { SEARCH = ev.target.value.trim(); applyFilter(true); };
// 点药丸看左右手平衡详情（要筛选「待标左右」请用左侧过滤按钮）
$('#s-hand').onclick = () => toast(balanceText(), 8000);
$('#save-prompt').onclick = async () => {
  const v = $('#default-prompt').value.trim();
  await api('/api/prompt', {ds:dsKey(), default_prompt:v});
  toast('统一 prompt 已保存：' + v);
};
$('#lightbox').onclick = () => $('#lightbox').style.display='none';

document.addEventListener('keydown', ev => {
  if(['INPUT','TEXTAREA','SELECT'].includes(ev.target.tagName)) return;
  if(ev.key==='j' || ev.key==='ArrowDown'){ move(1); ev.preventDefault(); return; }
  if(ev.key==='k' || ev.key==='ArrowUp'){ move(-1); ev.preventDefault(); return; }
  const e = VIEW[CURSOR]; if(!e) return;
  const k = ev.key.length===1 ? ev.key.toLowerCase() : ev.key;   // 大小写都认
  if(k==='1'){ setMark(e,'valid'); ev.preventDefault(); }
  else if(k==='2'){ setMark(e,'invalid'); ev.preventDefault(); }
  else if(k==='3'){ setMark(e,'unmarked'); ev.preventDefault(); }
  else if(k==='q'){ setHand(e,'left'); ev.preventDefault(); }    // Q = 左手
  else if(k==='e'){ setHand(e,'right'); ev.preventDefault(); }   // E = 右手
  else if(k===' '){
    const v = $('#view').querySelector('video');
    if(v){ if(v.paused) v.play().catch(()=>{}); else v.pause(); }
    ev.preventDefault();
  }
});

/* ---------------- 导出 ---------------- */
$('#btn-export').onclick = () => {
  const n = EPS.filter(x=>x.mark==='valid').length;
  const un = EPS.filter(x=>x.mark==='unmarked').length;
  const bad = EPS.filter(x=>x.mark==='invalid').length;
  $('#m-text').innerHTML = `将导出 <b>${n}</b> 条有效；排除 ${bad} 条无效 + ${un} 条未标记。`;
  $('#m-progress').textContent = '';
  $('#m-go').disabled = n===0;
  $('#modal').style.display='flex';
};
$('#m-cancel').onclick = () => $('#modal').style.display='none';
$('#m-go').onclick = async () => {
  $('#m-go').disabled = true;
  $('#m-progress').textContent = '正在启动导出…';
  try{
    const job = await api('/api/export', {ds:dsKey()});
    const poll = async () => {
      const st = await api('/api/export/' + job.id);
      if(st.status === 'running'){ $('#m-progress').textContent = '导出中…（' + st.id + '）'; setTimeout(poll, 1500); return; }
      if(st.status === 'done'){
        $('#m-progress').innerHTML = '✅ 完成：<b>' + (st.output||'') + '</b>';
        toast('导出完成：' + (st.output||''), 6000);
      } else {
        $('#m-progress').textContent = '❌ 失败，见 ' + (st.log || '');
      }
      $('#m-go').disabled = false;
    };
    setTimeout(poll, 1200);
  }catch(e){ $('#m-progress').textContent='启动失败：'+e.message; $('#m-go').disabled=false; }
};

$('#btn-merge').onclick = async () => {
  const d = await api('/api/datasets');
  const ds = d.datasets || [];
  $('#merge-list').innerHTML = ds.map(x => {
    const st = x.stats || {};
    return `<label style="display:flex;gap:8px;align-items:center;padding:4px 2px">
      <input type="checkbox" class="mgpick" value="${x.path}">
      <span style="flex:1">${x.name}</span>
      <span class="meta">${x.total_episodes} 条　有效 <b>${st.valid||0}</b></span>
    </label>`;
  }).join('') || '<div class="meta">没有数据集</div>';
  $('#merge-name').value = 'train_' + new Date().toISOString().slice(0,10).replace(/-/g,'');
  $('#merge-prompt').value = $('#default-prompt').value || '抓取红色沙包';
  $('#mg-progress').textContent = '';
  $('#modal-merge').style.display = 'flex';
};
$('#mg-cancel').onclick = () => $('#modal-merge').style.display = 'none';
$('#mg-go').onclick = async () => {
  const datasets = [...document.querySelectorAll('.mgpick:checked')].map(c=>c.value);
  if(!datasets.length){ toast('至少勾一个数据集'); return; }
  $('#mg-go').disabled = true;
  $('#mg-progress').textContent = '正在启动合并…';
  try{
    const job = await api('/api/merge', {datasets,
      name: $('#merge-name').value.trim(), prompt: $('#merge-prompt').value.trim()});
    const poll = async () => {
      const st = await api('/api/merge/' + job.id);
      if(st.status === 'running'){ $('#mg-progress').textContent = '合并中…（' + st.id + '）'; setTimeout(poll, 2000); return; }
      if(st.status === 'done'){
        $('#mg-progress').innerHTML = '✅ 完成：<b>' + (st.output || '') + '</b>';
        toast('训练集已生成：' + (st.output||''), 6000);
      } else {
        $('#mg-progress').textContent = '❌ 合并失败，见 ' + (st.log || '');
      }
      $('#mg-go').disabled = false;
    };
    setTimeout(poll, 1500);
  }catch(e){ $('#mg-progress').textContent = '启动失败：' + e.message; $('#mg-go').disabled = false; }
};

loadDatasets().catch(e => { $('#empty').textContent = '加载失败：'+e.message; });
</script>
</body>
</html>
"""


# ================================================================ 启动

def main(argv=None):
    ap = argparse.ArgumentParser(description="TA2 筛选工作台")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--scan-root", default=DEFAULT_SCAN_ROOT)
    ap.add_argument("--review-root", default=DEFAULT_REVIEW_ROOT)
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON)
    args = ap.parse_args(argv)

    store = AnnotationStore(args.review_root)
    cache = {"t": 0.0, "rows": []}

    def datasets():
        now = time.time()
        if now - cache["t"] > 5:
            cache["rows"] = discover_datasets(args.scan_root)
            cache["t"] = now
        return cache["rows"]

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.cfg = {
        "scan_root": args.scan_root,
        "review_root": args.review_root,
        "venv_python": args.venv_python,
        "store": store,
        "datasets": datasets,
    }
    print(f"筛选工作台: http://{args.host}:{args.port}  scan={args.scan_root}  review={args.review_root}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n退出", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

