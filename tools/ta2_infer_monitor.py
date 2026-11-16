#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ta2_infer_monitor.py — TA2 π0.5 推理监控台（:8096，只读）

只读取 client 已经落盘的 run 目录（guarded_grasp.py 产物）：
  inferences.jsonl   每次推理：延迟、30x19 动作块、rtc_chunk_id、观测文件名
  commands.jsonl     每次 actuation（45 Hz）：command16 / state48 / 跟踪误差 / 限位夹紧
  observation_NNN.npz 每次推理模型真正看到的三路图 + 48 维状态
  report.json        运行结果、限值、preflight
  <run>_probe/readiness.json  起飞检查（模式、新鲜度、prompt）

原则：
  * 只读：不发布 ROS、不启停 client、不写 run 目录
  * 零依赖：系统 python3（只用标准库）+ ffmpeg（可选，帧编码）
  * 零干扰：nice 由 ops 脚本设置；单线程 tail；无 viewer 时不解码帧

用法：
  python3 ta2_infer_monitor.py --port 8096
  python3 ta2_infer_monitor.py --selftest --no-frames
  python3 ta2_infer_monitor.py --simulate /path/to/run --speed 1
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import logging
import math
import os
import queue
import re
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VERSION = "1.0.0"
LOG = logging.getLogger("ta2_infer_monitor")

DEFAULT_CONFIG = {
    "port": 8096,
    "host": "0.0.0.0",
    "timezone": "Asia/Shanghai",
    "run_roots": ["/root/openpi-xpu/client/outputs"],
    "cache_dir": "/data/ta2/infer",
    "log": "/data/ta2/infer/monitor.log",
    "ui": "",  # 空 = 与本文件同目录的 infer_monitor_ui.html
    "window_s": 120,
    "command_hz": 10,
    "max_series": 4000,
    "max_commands": 20000,
    "frame": {
        "enabled": True,
        "width": 320,
        "quality": 6,
        "cache_frames": 48,
    },
    "budget": {
        "frequency_fallback": 45.0,
        "replan_steps_fallback": 8,
        "latency_warn_ms": 250.0,
    },
    "limits": {
        "max_tracking_rad": 0.2,
        "max_initial_gap_rad": 0.6,
        "max_stale_s": 1.0,
        "disk_low_gb": 20.0,
    },
    "expected_prompts": [],
}


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def now() -> float:
    return time.time()


def iso_to_epoch(value) -> float | None:
    """解析 ISO8601（支持 Z），返回 epoch 秒；失败返回 None。"""
    if not isinstance(value, str) or not value:
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def get_tz(name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return timezone(timedelta(hours=8))


def fmt_local(epoch: float | None, tz) -> str:
    if epoch is None:
        return "—"
    try:
        return datetime.fromtimestamp(epoch, tz).strftime("%H:%M:%S")
    except Exception:
        return "—"


def safe_float(value, default=None):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def inference_window(obj: dict, replan: int):
    """RTC 的每次推理窗口 = horizon - delay 控制步；同步模式退回 replan。"""
    delay = safe_int(obj.get("delay_steps"))
    horizon = safe_int(obj.get("horizon_steps"))
    if delay is not None and horizon is not None and horizon > delay >= 0:
        return max(1, horizon - delay), delay, horizon
    return max(1, replan), delay, horizon


def percentile(values, pct: float):
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def median(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(vals) if vals else None


def l2(a, b) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def clamp_round(value, ndigits=5):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, val in (extra or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def file_stat(path: Path):
    try:
        st = path.stat()
        return {"size": st.st_size, "mtime": st.st_mtime}
    except OSError:
        return None


def free_gb(path: Path) -> float | None:
    try:
        target = path
        while not target.exists() and target != target.parent:
            target = target.parent
        return shutil.disk_usage(str(target)).free / (1024 ** 3)
    except Exception:
        return None


# --------------------------------------------------------------------------
# .npy / .npz 读取（只用标准库）
# --------------------------------------------------------------------------

NPY_MAGIC = b"\x93NUMPY"


def parse_npy(raw: bytes):
    """返回 (shape, descr, data_bytes)。仅支持 C 顺序；不解码 dtype。"""
    if not raw.startswith(NPY_MAGIC):
        raise ValueError("not a npy payload")
    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    else:
        raise ValueError("unsupported npy version %r" % (major,))
    header = ast.literal_eval(raw[offset: offset + header_len].decode("latin1"))
    shape = tuple(int(v) for v in (header.get("shape") or ()))
    descr = str(header.get("descr") or "")
    if header.get("fortran_order"):
        raise ValueError("fortran_order not supported")
    data = raw[offset + header_len:]
    return shape, descr, data


CAMERA_CANDIDATES = {
    "head": (
        "observation/images/head_camera",
        "observation/images/head",
        "head_camera",
        "head",
    ),
    "left": (
        "observation/images/left_color",
        "observation/images/left_wrist",
        "left_color",
        "left",
    ),
    "right": (
        "observation/images/right_color",
        "observation/images/right_wrist",
        "right_color",
        "right",
    ),
}


def extract_cameras(npz_path: Path):
    """从 observation npz 提取三路 uint8 HWC RGB；返回 {cam: (w, h, bytes)}。"""
    result = {}
    try:
        with zipfile.ZipFile(npz_path) as zf:
            names = zf.namelist()
            lookup = {name[:-4] if name.endswith(".npy") else name: name for name in names}
            for cam, candidates in CAMERA_CANDIDATES.items():
                member = None
                for cand in candidates:
                    if cand in lookup:
                        member = lookup[cand]
                        break
                if member is None:
                    # 兜底：按关键字匹配
                    for key, name in lookup.items():
                        low = key.lower()
                        if cam == "head" and "head" in low and "image" in low:
                            member = name
                            break
                        if cam == "left" and "left" in low and "image" in low and "wrist" not in low:
                            member = name
                            break
                        if cam == "right" and "right" in low and "image" in low and "wrist" not in low:
                            member = name
                            break
                if member is None:
                    continue
                raw = zf.read(member)
                shape, descr, data = parse_npy(raw)
                if len(shape) != 3 or shape[2] != 3 or "u1" not in descr:
                    continue
                h, w = shape[0], shape[1]
                need = h * w * 3
                if len(data) < need:
                    continue
                result[cam] = (w, h, data[:need])
    except Exception as exc:  # 坏文件、半写、非 npz
        LOG.debug("extract_cameras failed for %s: %s", npz_path, exc)
    return result


# --------------------------------------------------------------------------
# 配置与运行目录
# --------------------------------------------------------------------------


class Config:
    def __init__(self, data: dict):
        self.data = deep_merge(DEFAULT_CONFIG, data or {})

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def tz(self):
        return get_tz(str(self.data.get("timezone") or "Asia/Shanghai"))

    @property
    def run_roots(self):
        return [Path(p) for p in (self.data.get("run_roots") or [])]

    @property
    def cache_dir(self) -> Path:
        return Path(self.data.get("cache_dir") or "/data/ta2/infer")

    def save(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            LOG.warning("save config failed: %s", exc)


def load_config(path: str | None) -> Config:
    data = {}
    if path:
        p = Path(path)
        if p.exists():
            loaded = read_json(p)
            if isinstance(loaded, dict):
                data = loaded
            else:
                LOG.warning("配置文件不是 JSON 对象：%s", p)
        else:
            LOG.warning("配置文件不存在，使用默认值：%s", p)
    return Config(data)


class RunInfo:
    def __init__(self, path: Path):
        self.id = path.name
        self.path = path
        inf = path / "inferences.jsonl"
        cmd = path / "commands.jsonl"
        rep = path / "report.json"
        stats = [s for s in (file_stat(inf), file_stat(cmd), file_stat(rep)) if s]
        self.mtime = max((s["mtime"] for s in stats), default=0.0)
        self.size = sum(s["size"] for s in stats)
        self.live = False
        self.prompt = None
        self.status = None
        self.refresh_live(rep, inf, cmd)

    def refresh_live(self, rep: Path, inf: Path, cmd: Path):
        rep_stat = file_stat(rep)
        inf_stat = file_stat(inf) or {"mtime": 0.0}
        cmd_stat = file_stat(cmd) or {"mtime": 0.0}
        newest = max(inf_stat["mtime"], cmd_stat["mtime"])
        if rep_stat is None:
            self.live = newest > 0 and (now() - newest) < 90.0
        else:
            # report 已落盘但 JSONL 还在增长 → 仍视为 live（防止 report 提前写）
            self.live = newest > rep_stat["mtime"] + 0.5 and (now() - newest) < 90.0
        self.refresh_meta(rep)

    def refresh_meta(self, rep: Path):
        before = getattr(self, "_meta_mtime", None)
        rep_stat = file_stat(rep)
        if rep_stat and rep_stat["mtime"] == before:
            return
        self._meta_mtime = rep_stat["mtime"] if rep_stat else None
        report = read_json(rep) if rep_stat else None
        if isinstance(report, dict):
            limits = report.get("limits") or {}
            self.prompt = limits.get("prompt") or report.get("prompt")
            self.status = report.get("status")
        readiness = read_json(self.path.parent / (self.id + "_probe") / "readiness.json")
        if not isinstance(readiness, dict):
            readiness = read_json(self.path / "readiness.json")
        if isinstance(readiness, dict) and not self.prompt:
            self.prompt = readiness.get("prompt")

    def as_dict(self):
        return {
            "id": self.id,
            "path": str(self.path),
            "mtime": self.mtime,
            "size": self.size,
            "live": self.live,
            "prompt": self.prompt,
            "status": self.status,
        }


class RunIndex:
    """扫 run 根目录；5 秒缓存。"""

    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()
        self._runs: list[RunInfo] = []
        self._scanned_at = 0.0
        self._ttl = 5.0

    def scan(self, force=False) -> list[RunInfo]:
        with self._lock:
            if not force and (now() - self._scanned_at) < self._ttl:
                return list(self._runs)
            runs: list[RunInfo] = []
            for root in self.config.run_roots:
                try:
                    entries = list(os.scandir(root))
                except OSError:
                    continue
                for entry in entries:
                    try:
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    p = Path(entry.path)
                    if not ((p / "inferences.jsonl").exists() or (p / "commands.jsonl").exists()):
                        continue
                    try:
                        runs.append(RunInfo(p))
                    except Exception as exc:
                        LOG.debug("skip run %s: %s", p, exc)
            runs.sort(key=lambda r: r.mtime, reverse=True)
            self._runs = runs[:80]
            self._scanned_at = now()
            return list(self._runs)

    def get(self, run_id: str) -> RunInfo | None:
        for run in self.scan():
            if run.id == run_id:
                return run
        # 缓存里没有：强制扫一次
        for run in self.scan(force=True):
            if run.id == run_id:
                return run
        return None

    def pick_live(self, runs: list[RunInfo]) -> RunInfo | None:
        for run in runs:
            if run.live:
                return run
        return None


# --------------------------------------------------------------------------
# JSONL 增量读取
# --------------------------------------------------------------------------


class Tailer:
    """按字节偏移增量读，处理半行与文件截断。"""

    def __init__(self, path: Path, on_line, initial_bytes: int = 8_000_000):
        self.path = path
        self.on_line = on_line
        self.initial_bytes = initial_bytes
        self.offset = 0
        self.buffer = b""
        self.started = False
        self.lines = 0
        self.errors = 0
        self._fh = None

    def start(self):
        try:
            self._fh = self.path.open("rb")
        except OSError:
            self._fh = None
            return
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        start = max(0, size - self.initial_bytes)
        if start > 0:
            self._fh.seek(start)
            # 丢弃第一段半行
            discarded = self._fh.readline()
            start += len(discarded)
        else:
            self._fh.seek(0)
        self.offset = start
        self.started = True
        self._drain(initial=True)

    def _drain(self, initial=False):
        if not self._fh:
            return
        try:
            chunk = self._fh.read(1_000_000)
        except OSError:
            return
        while chunk:
            self.buffer += chunk
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                self.offset += len(line) + 1
                self._handle(line)
            chunk = self._fh.read(1_000_000)
            if initial and self.lines > 20_000:
                break
        self.offset = self._fh.tell()

    def _handle(self, raw: bytes):
        if not raw.strip():
            return
        try:
            obj = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            self.errors += 1
            if self.errors <= 5:
                LOG.debug("bad json line in %s: %r", self.path, raw[:160])
            return
        self.lines += 1
        try:
            self.on_line(obj)
        except Exception:
            self.errors += 1
            if self.errors <= 5:
                LOG.debug("on_line failed in %s\n%s", self.path, traceback.format_exc())

    def poll(self):
        if not self._fh:
            if self.path.exists():
                self.start()
            return
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:
            # 文件被截断/轮转
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
            self.buffer = b""
            self.offset = 0
            self.start()
            return
        if size > self.offset:
            self._drain()

    def close(self):
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
        self._fh = None


# --------------------------------------------------------------------------
# 帧编码（ffmpeg 常驻进程）
# --------------------------------------------------------------------------


class JpegEncoder:
    """把原始 RGB 帧喂给常驻 ffmpeg，线程读回 JPEG。"""

    def __init__(self, width: int, height: int, out_width: int, quality: int):
        self.width = width
        self.height = height
        self.out_width = out_width
        self.quality = quality
        self.latest: bytes | None = None
        self.count = 0
        self.errors = 0
        self._proc = None
        self._reader = None
        self._lock = threading.Lock()
        self._stop = False
        self._spawn()

    def _spawn(self):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (self.width, self.height),
            "-i", "pipe:0",
            "-vf", "scale=%d:-2" % self.out_width,
            "-f", "mjpeg", "-q:v", str(self.quality),
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception:
            self._proc = None
            self.errors += 1
            LOG.warning("ffmpeg 启动失败（帧编码停用）")
            return
        self._reader = threading.Thread(target=self._read_loop, name="jpeg-reader", daemon=True)
        self._reader.start()

    def _read_loop(self):
        proc = self._proc
        if not proc or not proc.stdout:
            return
        buf = b""
        try:
            while not self._stop:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    start = buf.find(b"\xff\xd8")
                    if start < 0:
                        buf = buf[-2:] if len(buf) > 2 else buf
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start > 0:
                            buf = buf[start:]
                        break
                    jpeg = buf[start:end + 2]
                    buf = buf[end + 2:]
                    with self._lock:
                        self.latest = jpeg
                        self.count += 1
                if len(buf) > 8_000_000:
                    buf = buf[-2_000_000:]
        except Exception:
            self.errors += 1
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    def feed(self, raw: bytes):
        proc = self._proc
        if not proc or not proc.stdin:
            return
        try:
            proc.stdin.write(raw)
            proc.stdin.flush()
        except Exception:
            self.errors += 1
            try:
                proc.kill()
            except Exception:
                pass
            self._proc = None
            self._spawn()

    def snapshot(self):
        with self._lock:
            return self.latest

    def close(self):
        self._stop = True
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass


class FrameHub:
    """按 run 维护三路最新 JPEG；回放帧按需编码 + LRU。"""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(config["frame"].get("enabled", True))
        self.out_width = int(config["frame"].get("width") or 320)
        self.quality = int(config["frame"].get("quality") or 6)
        self.cache_size = int(config["frame"].get("cache_frames") or 48)
        self._encoders: dict[str, JpegEncoder] = {}
        self._lock = threading.Lock()
        self._last_obs = None
        self._cache: OrderedDict = OrderedDict()
        self._viewers = 0
        self.frames_fed = 0
        self._pending = None
        self._cv = threading.Condition()
        self._stop = False
        self._worker = threading.Thread(target=self._run_worker, name="frame-hub", daemon=True)
        self._worker.start()

    # ---- live ----
    def feed_observation(self, run_path: Path, obs_name: str, max_width: int | None = None):
        """只登记最新一帧，解码/编码在 worker 线程做，避免阻塞 tail。"""
        if not self.enabled or max_width == 0:
            return 0
        key = (str(run_path), obs_name)
        if key == self._last_obs:
            return 0
        self._last_obs = key
        with self._cv:
            self._pending = (run_path, obs_name)
            self._cv.notify()
        return 0

    def _run_worker(self):
        while not self._stop:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait(timeout=1.0)
                if self._stop:
                    return
                run_path, obs_name = self._pending
                self._pending = None
            if self.viewer_count() <= 0:
                continue
            try:
                cameras = extract_cameras(run_path / obs_name)
                fed = 0
                for cam, (w, h, data) in cameras.items():
                    with self._lock:
                        enc = self._encoders.get(cam)
                        if enc is None or enc.width != w or enc.height != h:
                            if enc:
                                enc.close()
                            enc = JpegEncoder(w, h, self.out_width, self.quality)
                            self._encoders[cam] = enc
                    enc.feed(data)
                    fed += 1
                self.frames_fed += fed
            except Exception:
                LOG.debug("frame worker failed", exc_info=True)

    def latest(self, cam: str):
        with self._lock:
            enc = self._encoders.get(cam)
        return enc.snapshot() if enc else None

    def viewer_count(self):
        with self._lock:
            return self._viewers

    def add_viewer(self, delta: int):
        with self._lock:
            self._viewers += delta
            self._viewers = max(0, self._viewers)

    # ---- replay（按需）----
    def frame_for_path(self, obs_path: Path, cam: str):
        if not self.enabled:
            return None
        key = (str(obs_path), cam, self.out_width)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        cameras = extract_cameras(obs_path)
        if cam not in cameras:
            return None
        w, h, data = cameras[cam]
        jpeg = self._encode_once(w, h, data)
        if jpeg:
            with self._lock:
                self._cache[key] = jpeg
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return jpeg

    def frame_for(self, run_path: Path, obs_name: str, cam: str):
        return self.frame_for_path(run_path / obs_name, cam)

    def _encode_once(self, w: int, h: int, raw: bytes):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
            "-i", "pipe:0", "-vf", "scale=%d:-2" % self.out_width,
            "-f", "mjpeg", "-q:v", str(self.quality), "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, timeout=10)
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except Exception:
            pass
        return None

    def close(self):
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        try:
            self._worker.join(timeout=2.0)
        except Exception:
            pass
        with self._lock:
            for enc in self._encoders.values():
                enc.close()
            self._encoders.clear()

    def status(self):
        with self._lock:
            encoders = {cam: {"count": e.count, "errors": e.errors}
                        for cam, e in self._encoders.items()}
        return {
            "enabled": self.enabled,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "out_width": self.out_width,
            "viewers": self.viewer_count(),
            "frames_fed": self.frames_fed,
            "encoders": encoders,
        }


# --------------------------------------------------------------------------
# SSE 广播
# --------------------------------------------------------------------------


class Broadcaster:
    def __init__(self):
        self._lock = threading.Lock()
        self._subs: list[tuple[str | None, queue.Queue]] = []

    def subscribe(self, run_id: str | None):
        q: queue.Queue = queue.Queue(maxsize=300)
        with self._lock:
            self._subs.append((run_id, q))
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs = [(rid, qq) for rid, qq in self._subs if qq is not q]

    def publish(self, run_id: str | None, event: str, data):
        payload = (event, data)
        with self._lock:
            targets = list(self._subs)
        for rid, q in targets:
            if rid is not None and run_id is not None and rid != run_id:
                continue
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def count(self):
        with self._lock:
            return len(self._subs)


# --------------------------------------------------------------------------
# 单次 run 的实时状态
# --------------------------------------------------------------------------


class RunState:
    def __init__(self, run: RunInfo, config: Config, broadcaster: Broadcaster,
                 marks: "MarkStore", frames: FrameHub):
        self.run = run
        self.id = run.id
        self.path = run.path
        self.config = config
        self.broadcaster = broadcaster
        self.marks = marks
        self.frames = frames
        self.lock = threading.RLock()
        self.infs: deque = deque(maxlen=int(config["max_series"]))
        self.cmds: deque = deque(maxlen=int(config["max_commands"]))
        self.events: deque = deque(maxlen=800)
        self.alerts_seen: dict[str, float] = {}
        self.last_inf_wall = None
        self.last_cmd_wall = None
        self.last_inf = None
        self.last_cmd = None
        self.last_q16 = None
        self.prev_chunk16 = None
        self.last_chunk16 = None
        self.prev_seam = None
        self.start_utc = None
        self.started_wall = now()
        self.limits = dict(config["limits"])
        self.frequency = float(config["budget"]["frequency_fallback"])
        self.replan = int(config["budget"]["replan_steps_fallback"])
        self.prompt = run.prompt
        self.port = None
        self.inference_count = 0
        self.actuation_count = 0
        self.pending_cmds: list = []
        self._last_flush = now()
        self._last_periodic = 0.0
        self._last_disk_check = 0.0
        self._ended_reported = False
        self._tail_inf = Tailer(self.path / "inferences.jsonl", self._on_inference)
        self._tail_cmd = Tailer(self.path / "commands.jsonl", self._on_command)
        self._load_context()
        self._tail_cmd.start()
        self._tail_inf.start()

    # ---- 上下文（report / readiness）----
    def _load_context(self):
        report = read_json(self.path / "report.json")
        readiness = read_json(self.path.parent / (self.id + "_probe") / "readiness.json")
        if not isinstance(readiness, dict):
            readiness = read_json(self.path / "readiness.json")
        if isinstance(report, dict):
            limits = report.get("limits") or {}
            freq = safe_float(limits.get("frequency"))
            replan = safe_int(limits.get("replan_steps"))
            if freq and freq > 0:
                self.frequency = freq
            if replan and replan > 0:
                self.replan = replan
            self.prompt = limits.get("prompt") or self.prompt
            self.port = limits.get("port")
            for key in ("max_tracking_rad", "max_initial_gap_rad"):
                if safe_float(limits.get(key)) is not None:
                    self.limits[key] = float(limits[key])
            start = iso_to_epoch(report.get("started_at_utc"))
            if start:
                self.start_utc = start
        if isinstance(readiness, dict):
            self.prompt = self.prompt or readiness.get("prompt")
            self.readiness = readiness
        else:
            self.readiness = None
        self.capabilities = run_capabilities(self.run)

    # ---- 解析回调 ----
    def _on_inference(self, obj: dict):
        at = iso_to_epoch(obj.get("at"))
        sec = safe_float(obj.get("seconds"))
        if at is None:
            at = now()
        if self.start_utc is None:
            # 用最近一条 command 的 elapsed_s 反推 run 起点
            if self.last_cmd and self.last_cmd.get("rel") is not None:
                self.start_utc = at - (self.last_cmd["rel"] + (sec or 0.0))
            else:
                self.start_utc = at
        rel = at - self.start_utc
        exec_steps, delay_steps, horizon_steps = inference_window(obj, self.replan)
        is_rtc = delay_steps is not None and horizon_steps is not None
        chunk19 = obj.get("actions19")
        chunk16 = None
        mean_abs = None
        seam = None
        if isinstance(chunk19, list) and chunk19:
            chunk16 = []
            for row in chunk19[:80]:
                if not isinstance(row, list):
                    continue
                chunk16.append([clamp_round(v, 5) for v in row[:16]])
            if chunk16:
                flat = [abs(v) for row in chunk16 for v in row[:16] if v is not None]
                mean_abs = sum(flat) / len(flat) if flat else None
                # RTC 的块有前缀重叠，"接缝"不能沿用同步模式的 8 步口径；先留空
                if self.prev_chunk16 and not is_rtc:
                    idx = min(self.replan, len(self.prev_chunk16)) - 1
                    seam = l2(self.prev_chunk16[idx], chunk16[0])
                self.prev_chunk16 = chunk16
        plan_gap = None
        # 只在实时解析时用最新 state 算「计划首步 vs 当前状态」；补历史时不硬凑
        if chunk16 and self.last_q16 and self.last_cmd_wall and (now() - self.last_cmd_wall) < 1.0:
            plan_gap = l2(chunk16[0][:14], self.last_q16[:14])
        budget = None
        if sec is not None and self.frequency > 0:
            budget = sec / (exec_steps / self.frequency)
        abs_i = safe_int(obj.get("inference"))
        if abs_i is None:
            abs_i = self.inference_count
        record = {
            "i": abs_i,
            "rel": clamp_round(rel, 3),
            "at": at,
            "sec": clamp_round(sec, 5),
            "budget": clamp_round(budget, 4),
            "exec_steps": exec_steps,
            "delay_steps": delay_steps,
            "horizon_steps": horizon_steps,
            "seam": clamp_round(seam, 5),
            "plan_gap": clamp_round(plan_gap, 5),
            "obs": obj.get("observation"),
            "chunk_id": obj.get("rtc_chunk_id"),
            "prefix_start": safe_int(obj.get("rtc_prefix_start")),
            "has_chunk": bool(chunk16),
            "mean_abs": clamp_round(mean_abs, 5),
        }
        with self.lock:
            self.inference_count += 1
            self.infs.append(record)
            self.last_inf = record
            self.last_inf_wall = now()
            self.last_chunk16 = chunk16
        self.broadcaster.publish(self.id, "inference", {
            "i": record["i"], "rel": record["rel"], "at": record["at"],
            "sec": record["sec"], "budget": record["budget"],
            "exec_steps": record["exec_steps"],
            "delay_steps": record["delay_steps"], "horizon_steps": record["horizon_steps"],
            "seam": record["seam"], "gap": record["plan_gap"],
            "obs": record["obs"], "chunk_id": record["chunk_id"],
            "chunk16": chunk16,
        })
        # 帧只在有人看时解
        if self.frames.viewer_count() > 0 and record["obs"]:
            try:
                self.frames.feed_observation(self.path, record["obs"])
            except Exception:
                LOG.debug("feed frame failed", exc_info=True)
        self._check_inference(record)
        self.maybe_auto_mark(record)

    def _on_command(self, obj: dict):
        rel = safe_float(obj.get("elapsed_s"))
        if rel is None:
            rel = 0.0
        state48 = obj.get("state48") if isinstance(obj.get("state48"), list) else None
        cmd16 = obj.get("command16") if isinstance(obj.get("command16"), list) else None
        q16 = None
        if state48 and len(state48) >= 16:
            q16 = [clamp_round(v, 5) for v in state48[:16]]
        cmd = [clamp_round(v, 5) for v in (cmd16 or [])][:16]
        err = safe_float(obj.get("tracking_error_rad"))
        gap = safe_float(obj.get("raw_target_gap_rad"))
        step = safe_float(obj.get("command_step_rad"))
        clamped_joints = obj.get("limit_clamped_joints")
        if not isinstance(clamped_joints, list):
            clamped_joints = []
        gripper_clamped = bool(obj.get("gripper_clamped"))
        joint_clamp = 1 if clamped_joints else 0
        clamp = 1 if (clamped_joints or gripper_clamped) else 0
        row = {
            "rel": clamp_round(rel, 3),
            "act": safe_int(obj.get("actuation")),
            "cmd": cmd,
            "q": q16,
            "err": clamp_round(err, 5),
            "gap": clamp_round(gap, 5),
            "step": clamp_round(step, 6),
            "clamp": clamp,
            "joint_clamp": joint_clamp,
            "gripper_clamp": 1 if gripper_clamped else 0,
            "clamped_joints": clamped_joints,
            "gripper_clamped": gripper_clamped,
        }
        with self.lock:
            self.actuation_count += 1
            self.cmds.append(row)
            self.last_cmd = row
            self.last_cmd_wall = now()
            if q16:
                self.last_q16 = q16
            self.pending_cmds.append([
                row["rel"], row["act"],
                row["cmd"][:16],
                row["q"][:16] if row["q"] else [],
                row["err"], row["gap"], row["clamp"],
                row["joint_clamp"], row["gripper_clamp"],
            ])
            if len(self.pending_cmds) > 500:
                self.pending_cmds = self.pending_cmds[-500:]
        self._check_command(row)

    # ---- 统计 ----
    def _window_infs(self, seconds: float | None = None):
        if not self.infs:
            return []
        last = self.last_inf["rel"] if self.last_inf else None
        if last is None or seconds is None:
            return list(self.infs)
        return [r for r in self.infs if r["rel"] is not None and last - r["rel"] <= seconds]

    def stats(self):
        with self.lock:
            win = self._window_infs(float(self.config["window_s"]))
            secs = [r["sec"] for r in win if r["sec"] is not None]
            intervals = []
            prev = None
            for r in win:
                if r["rel"] is None:
                    continue
                if prev is not None and r["rel"] > prev:
                    intervals.append(r["rel"] - prev)
                prev = r["rel"]
            cycle = median(intervals)
            rate = (1.0 / cycle) if cycle else None
            errs = [c["err"] for c in self.cmds if c["err"] is not None]
            errs_win = errs[-400:] if errs else []
            joint_clamps = sum(1 for c in self.cmds if c.get("joint_clamp"))
            gripper_clamps = sum(1 for c in self.cmds if c.get("gripper_clamp"))
            clamps = joint_clamps + gripper_clamps
            seams = [r["seam"] for r in self.infs if r["seam"] is not None]
            last = self.last_inf or {}
            last_cmd = self.last_cmd or {}
            budget = last.get("budget")
            # 运行中 capabilities 可能在建 state 时还没写第一行；按已解析数据刷新
            caps = dict(getattr(self, "capabilities", {}) or {})
            if self.infs:
                recent_infs = list(self.infs)[-80:]
                caps["per_inference_obs"] = any(r.get("obs") for r in recent_infs)
                caps["latency"] = any(r.get("sec") is not None for r in recent_infs)
                caps["chunk"] = any(r.get("has_chunk") for r in recent_infs)
                caps["rtc"] = any(r.get("horizon_steps") is not None for r in recent_infs)
                caps["final_obs"] = (self.path / "final_observation.npz").exists()
                caps["probe_obs"] = (self.path.parent / (self.id + "_probe") / "observation.npz").exists()
                self.capabilities = caps
            sec_p50 = percentile(secs, 50)
            exec_vals = [r.get("exec_steps") for r in win if r.get("exec_steps")]
            exec_steps_med = median(exec_vals) if exec_vals else self.replan
            exec_steps_med = max(1, int(round(exec_steps_med)))
            exec_plan = (exec_steps_med / self.frequency) if self.frequency > 0 else None
            delay_p50 = percentile([r.get("delay_steps") for r in win if r.get("delay_steps") is not None], 50)
            horizon_p50 = percentile([r.get("horizon_steps") for r in win if r.get("horizon_steps") is not None], 50)
            cmd_rate = None
            if len(self.cmds) >= 2:
                first_rel = self.cmds[0].get("rel")
                last_rel = self.cmds[-1].get("rel")
                if first_rel is not None and last_rel is not None and last_rel > first_rel:
                    cmd_rate = (len(self.cmds) - 1) / (last_rel - first_rel)
            rtc_mode = any(
                r.get("horizon_steps") is not None and r.get("delay_steps") is not None
                for r in win
            )
            if rtc_mode:
                # RTC：推理在后台与执行并行，循环不是「推理 + 执行」；窗口是 H-D 步
                exec_actual = None
                slowdown = None
                share = (sec_p50 / exec_plan) if (exec_plan and sec_p50 is not None) else None
            else:
                exec_actual = (cycle - sec_p50) if (cycle and sec_p50 is not None) else None
                slowdown = ((exec_actual - exec_plan)
                            if (exec_actual is not None and exec_plan) else None)
                share = (sec_p50 / cycle) if (cycle and sec_p50 is not None) else None
            loop = {
                "mode": "rtc" if rtc_mode else "sync",
                "cycle_s": clamp_round(cycle, 4),
                "exec_plan_s": clamp_round(exec_plan, 4),
                "inference_s": clamp_round(sec_p50, 4),
                "exec_actual_s": clamp_round(exec_actual, 4),
                "exec_slowdown_s": clamp_round(slowdown, 4),
                "inference_share": clamp_round(share, 3),
                "cmd_rate_hz": clamp_round(cmd_rate, 2),
                "target_cmd_rate_hz": clamp_round(self.frequency, 2),
                "target_inference_rate_hz": clamp_round(
                    (self.frequency / exec_steps_med) if exec_steps_med else None, 2),
                "exec_steps": exec_steps_med,
                "delay_steps_p50": clamp_round(delay_p50, 2),
                "horizon_steps_p50": clamp_round(horizon_p50, 2),
            }
            return {
                "id": self.id,
                "live": self.run.live,
                "prompt": self.prompt,
                "port": self.port,
                "frequency": self.frequency,
                "replan_steps": self.replan,
                "start_utc": self.start_utc,
                "counts": {
                    "inferences": self.inference_count,
                    "actuations": self.actuation_count,
                },
                "last": {
                    "i": last.get("i"),
                    "age": (now() - self.last_inf_wall) if self.last_inf_wall else None,
                    "sec": last.get("sec"),
                    "budget": budget,
                    "exec_steps": last.get("exec_steps"),
                    "delay_steps": last.get("delay_steps"),
                    "horizon_steps": last.get("horizon_steps"),
                    "seam": last.get("seam"),
                    "plan_gap": last.get("plan_gap"),
                    "obs": last.get("obs"),
                    "rel": last.get("rel"),
                },
                "window": {
                    "sec_p50": percentile(secs, 50),
                    "sec_p95": percentile(secs, 95),
                    "sec_max": max(secs) if secs else None,
                    "rate_hz": rate,
                    "budget_p50": percentile([r["budget"] for r in win if r["budget"] is not None], 50),
                    "budget_p95": percentile([r["budget"] for r in win if r["budget"] is not None], 95),
                    "seam_p95": percentile(seams, 95),
                    "seam_max": max(seams) if seams else None,
                    "tracking_max": max(errs_win) if errs_win else None,
                    "clamps": clamps,
                    "joint_clamps": joint_clamps,
                    "gripper_clamps": gripper_clamps,
                    "cmd_age": (now() - self.last_cmd_wall) if self.last_cmd_wall else None,
                    "cmd_rate": cmd_rate,
                },
                "loop": loop,
                "state": {
                    "len": len(self.last_q16) if self.last_q16 else 0,
                    "finite": all(v is not None for v in (self.last_q16 or [])),
                },
                "limits": dict(self.limits),
                "capabilities": caps,
                "framework": {
                    "cmd_step": last_cmd.get("step"),
                    "gap": last_cmd.get("gap"),
                    "err": last_cmd.get("err"),
                },
            }

    # ---- 规则 ----
    def _alert(self, kind: str, sev: str, msg: str, rel=None, evidence=None, auto_mark=True):
        key = kind
        last = self.alerts_seen.get(key, 0.0)
        cooldown = 30.0 if kind in ("stale_obs", "stale_cmd", "disk_low", "gripper_clamp") else 4.0
        if now() - last < cooldown:
            return
        self.alerts_seen[key] = now()
        event = {
            "kind": kind,
            "sev": sev,
            "msg": msg,
            "rel": clamp_round(rel, 3) if rel is not None else None,
            "t": now(),
            "evidence": evidence or {},
        }
        with self.lock:
            self.events.append(event)
        self.broadcaster.publish(self.id, "alert", event)
        if auto_mark and rel is not None:
            try:
                self.marks.add(self.id, rel, "[" + sev + "] " + msg, source="alert")
            except Exception:
                LOG.debug("auto mark failed", exc_info=True)

    def _check_inference(self, record: dict):
        sec = record.get("sec")
        recent = list(self.infs)[-300:]
        p95 = percentile([r["sec"] for r in recent if r["sec"] is not None], 95)
        if sec is not None and sec > float(self.config["budget"]["latency_warn_ms"]) / 1000.0:
            if p95 is None or sec > max(0.25, 2.0 * p95):
                self._alert("latency_spike", "warn",
                            "推理延迟 %.0f ms（p95 %.0f ms）" % (sec * 1000, (p95 or 0) * 1000),
                            rel=record["rel"], evidence={"i": record["i"], "obs": record["obs"]})
        if record.get("budget") is not None and record["budget"] > 1.0:
            self._alert("budget_over", "warn",
                        "推理预算占用 %.0f%%（一次推理比计划间隔更久）" % (record["budget"] * 100),
                        rel=record["rel"], evidence={"i": record["i"]})
        seam = record.get("seam")
        seams = [r["seam"] for r in recent if r["seam"] is not None]
        if seam is not None and len(seams) > 8:
            med = median(seams[:-1]) or 0.0
            if seam > max(0.05, 3.0 * med):
                self._alert("seam_jump", "warn",
                            "块接缝跳变 %.4f rad（历史中位 %.4f）" % (seam, med),
                            rel=record["rel"], evidence={"i": record["i"]})
        obs = record.get("obs")
        if obs and self.frames.viewer_count() > 0:
            pass  # 黑屏检测在帧线程里做，避免阻塞解析

    def _check_command(self, row: dict):
        limit = float(self.limits.get("max_tracking_rad") or 0.2)
        err = row.get("err")
        if err is not None and err > limit:
            self._alert("tracking_over", "warn",
                        "跟踪误差 %.3f > %.2f" % (err, limit),
                        rel=row["rel"], evidence={"act": row["act"], "err": err})
        gap = row.get("gap")
        gap_limit = float(self.limits.get("max_initial_gap_rad") or 0.6)
        if gap is not None and gap > gap_limit:
            self._alert("gap_over", "warn",
                        "指令跟不上计划：目标差 %.3f > %.2f" % (gap, gap_limit),
                        rel=row["rel"], evidence={"act": row["act"]})
        if row.get("joint_clamp"):
            self._alert("joint_clamp", "warn", "关节限位介入：%s" % (row.get("clamped_joints") or "?"),
                        rel=row["rel"], evidence={"act": row["act"]})
        elif row.get("gripper_clamp"):
            self._alert("gripper_clamp", "info", "夹爪夹紧（力控上限）",
                        rel=row["rel"], evidence={"act": row["act"]})
        q16 = row.get("q") or []
        if q16 and any(v is None for v in q16):
            self._alert("state_bad", "err", "state 出现 NaN/Inf", rel=row["rel"])

    def periodic(self):
        t = now()
        if t - self._last_periodic < 1.0:
            return
        self._last_periodic = t
        stale = float(self.limits.get("max_stale_s") or 1.0)
        if self.run.live:
            if self.last_inf_wall and (t - self.last_inf_wall) > max(stale * 2.0, 2.0):
                self._alert("stale_obs", "warn",
                            "已 %.1f s 没有新推理" % (t - self.last_inf_wall))
            if self.last_cmd_wall and (t - self.last_cmd_wall) > stale:
                self._alert("stale_cmd", "warn",
                            "已 %.1f s 没有新指令" % (t - self.last_cmd_wall))
        # 频率骤降
        rels = [r["rel"] for r in list(self.infs)[-6:] if r["rel"] is not None]
        if len(rels) >= 5:
            gaps = [b - a for a, b in zip(rels, rels[1:]) if b > a]
            med = median(gaps)
            if med and gaps[-1] > med * 2.5:
                self._alert("rate_drop", "warn",
                            "推理间隔 %.2f s（中位 %.2f s）" % (gaps[-1], med),
                            rel=rels[-1])
        if t - self._last_disk_check > 60:
            self._last_disk_check = t
            free = free_gb(self.config.cache_dir)
            if free is not None and free < float(self.config["limits"]["disk_low_gb"]):
                self._alert("disk_low", "err", "磁盘剩余 %.1f GB" % free)
        if not self._ended_reported:
            report = read_json(self.path / "report.json")
            if isinstance(report, dict) and report.get("status"):
                self._ended_reported = True
                self._alert("run_end", "info",
                            "run 结束：%s" % report.get("status"),
                            rel=self.last_inf["rel"] if self.last_inf else None)

    def maybe_auto_mark(self, record: dict):
        if self.last_inf is None:
            return
        if record["i"] == 0:
            self.marks.add(self.id, record["rel"], "首条推理", source="monitor")

    # ---- 周期刷 commands ----
    def flush_commands(self):
        with self.lock:
            if not self.pending_cmds:
                return
            if now() - self._last_flush < 0.2 and len(self.pending_cmds) < 30:
                return
            rows = self.pending_cmds
            self.pending_cmds = []
            self._last_flush = now()
        self.broadcaster.publish(self.id, "commands", {"run": self.id, "rows": rows})

    def poll(self):
        self._tail_cmd.poll()
        self._tail_inf.poll()
        self.flush_commands()
        self.periodic()

    def close(self):
        self._tail_inf.close()
        self._tail_cmd.close()

    def snapshot_series(self):
        with self.lock:
            infs = [dict(r) for r in self.infs]
            cmds = [dict(c) for c in self.cmds]
            events = list(self.events)
        return infs, cmds, events

    def close_events(self):
        return list(self.events)


# --------------------------------------------------------------------------
# 历史 run 读取（回放）
# --------------------------------------------------------------------------


def _iter_jsonl(path: Path, limit: int = 200_000):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                if idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except OSError:
        return


def resolve_obs_path(run: RunInfo, obs: str):
    """只允许 run 目录内的观测文件；兼容 RTC client 的首/末观测。"""
    if re.match(r"^observation_\d{3,6}\.npz$", obs):
        return run.path / obs
    if obs == "final_observation.npz":
        return run.path / obs
    if obs == "observation.npz":
        return run.path.parent / (run.id + "_probe") / obs
    return None


def run_capabilities(run: RunInfo):
    """判断这类 run 能提供什么：逐步图 / 延迟 / 动作块 / RTC 字段 / 首末观测。"""
    caps = {
        "per_inference_obs": False,
        "latency": False,
        "chunk": False,
        "rtc": False,
        "final_obs": (run.path / "final_observation.npz").exists(),
        "probe_obs": (run.path.parent / (run.id + "_probe") / "observation.npz").exists(),
    }
    try:
        with (run.path / "inferences.jsonl").open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                if idx >= 5:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("observation"):
                    caps["per_inference_obs"] = True
                if obj.get("seconds") is not None:
                    caps["latency"] = True
                if obj.get("actions19"):
                    caps["chunk"] = True
                if obj.get("delay_steps") is not None or obj.get("horizon_steps") is not None:
                    caps["rtc"] = True
    except OSError:
        pass
    return caps


def run_meta(run: RunInfo):
    report = read_json(run.path / "report.json") or {}
    readiness = read_json(run.path.parent / (run.id + "_probe") / "readiness.json")
    if not isinstance(readiness, dict):
        readiness = read_json(run.path / "readiness.json")
    limits = report.get("limits") if isinstance(report.get("limits"), dict) else {}
    preflight = report.get("preflight") if isinstance(report.get("preflight"), dict) else {}
    pick = lambda d, keys: {k: d.get(k) for k in keys if k in d}  # noqa: E731
    return {
        "id": run.id,
        "path": str(run.path),
        "mtime": run.mtime,
        "size": run.size,
        "live": run.live,
        "prompt": (limits or {}).get("prompt") or (readiness or {}).get("prompt"),
        "port": (limits or {}).get("port"),
        "status": report.get("status"),
        "exit_code": report.get("exit_code"),
        "started_at_utc": report.get("started_at_utc") or (readiness or {}).get("started_at_utc"),
        "finished_at_utc": report.get("finished_at_utc"),
        "actuations": report.get("actuations"),
        "live_inferences": report.get("live_inferences"),
        "limits": pick(limits, [
            "frequency", "replan_steps", "max_seconds", "max_actions", "max_speed",
            "max_tracking", "max_initial_gap", "max_age", "prompt", "port", "host", "execute",
        ]),
        "preflight": preflight,
        "report": pick(report, [
            "status", "exit_code", "signal", "task_success", "error",
            "attempted_actuations", "motion_elapsed_s", "rtc",
        ]),
        "readiness": None if not isinstance(readiness, dict) else pick(readiness, [
            "prompt", "observation_ready", "mode", "mode_fresh", "mode_age_s",
            "api_mode_observed", "joint_mode_observed", "motion_mode_compatible",
            "camera_format", "camera_mapping", "readiness_scope", "timeout_s",
            "max_receiver_age_s", "started_at_utc",
        ]),
        "capabilities": run_capabilities(run),
    }


def run_stats(run: RunInfo):
    """全量统计（只在 /summary 调一次）：计数不能用降采样后的 series。"""
    joint_clamps = 0
    gripper_clamps = 0
    tracking_max = None
    actuations = 0
    for obj in _iter_jsonl(run.path / "commands.jsonl"):
        actuations += 1
        if obj.get("limit_clamped_joints"):
            joint_clamps += 1
        if obj.get("gripper_clamped"):
            gripper_clamps += 1
        err = safe_float(obj.get("tracking_error_rad"))
        if err is not None and (tracking_max is None or err > tracking_max):
            tracking_max = err
    inferences = 0
    latency_available = False
    delays, horizons = [], []
    for obj in _iter_jsonl(run.path / "inferences.jsonl"):
        inferences += 1
        if obj.get("seconds") is not None:
            latency_available = True
        d = safe_int(obj.get("delay_steps"))
        h = safe_int(obj.get("horizon_steps"))
        if d is not None:
            delays.append(d)
        if h is not None:
            horizons.append(h)
    return {
        "inferences": inferences,
        "actuations": actuations,
        "joint_clamps": joint_clamps,
        "gripper_clamps": gripper_clamps,
        "tracking_max": clamp_round(tracking_max, 5),
        "latency_available": latency_available,
        "rtc": {
            "delay_min": min(delays) if delays else None,
            "delay_max": max(delays) if delays else None,
            "horizon_min": min(horizons) if horizons else None,
            "horizon_max": max(horizons) if horizons else None,
        },
    }


def load_run_series(run: RunInfo, window_s: float, max_points: int = 700):
    inf_path = run.path / "inferences.jsonl"
    cmd_path = run.path / "commands.jsonl"
    meta = run_meta(run)
    start = iso_to_epoch(meta.get("started_at_utc"))
    limits = meta.get("limits") or {}
    replan = safe_int(limits.get("replan_steps"), 8) or 8
    frequency = safe_float(limits.get("frequency"), 45.0) or 45.0
    infs = []
    last_chunk = None
    prev_chunk = None
    for obj in _iter_jsonl(inf_path):
        at = iso_to_epoch(obj.get("at"))
        if at is None:
            continue
        if start is None:
            start = at
        exec_steps, delay, horizon = inference_window(obj, replan)
        is_rtc = delay is not None and horizon is not None
        chunk = _crop_chunk(obj.get("actions19"))
        seam = None
        if chunk and prev_chunk and not is_rtc:
            idx = min(replan, len(prev_chunk)) - 1
            seam = clamp_round(l2(prev_chunk[idx], chunk[0]), 5)
        sec = safe_float(obj.get("seconds"))
        budget = (sec / (exec_steps / frequency)) if (sec is not None and frequency > 0) else None
        infs.append({
            "i": safe_int(obj.get("inference")) if safe_int(obj.get("inference")) is not None else len(infs),
            "rel": at - start,
            "at": at,
            "sec": sec,
            "budget": clamp_round(budget, 4),
            "exec_steps": exec_steps,
            "delay_steps": delay,
            "horizon_steps": horizon,
            "prefix_start": safe_int(obj.get("rtc_prefix_start")),
            "obs": obj.get("observation"),
            "chunk_id": obj.get("rtc_chunk_id"),
            "seam": seam,
        })
        if chunk:
            last_chunk = chunk
            prev_chunk = chunk
    if len(infs) > 4000:
        infs = infs[:: max(1, len(infs) // 4000)]
    if infs:
        infs[-1]["chunk16"] = last_chunk
    rows = []
    for obj in _iter_jsonl(cmd_path):
        rel = safe_float(obj.get("elapsed_s"))
        if rel is None:
            continue
        state48 = obj.get("state48") if isinstance(obj.get("state48"), list) else []
        cmd16 = obj.get("command16") if isinstance(obj.get("command16"), list) else []
        joint_clamped = obj.get("limit_clamped_joints") if isinstance(obj.get("limit_clamped_joints"), list) else []
        gripper_clamped = bool(obj.get("gripper_clamped"))
        rows.append([
            round(rel, 3),
            safe_int(obj.get("actuation")),
            [clamp_round(v, 4) for v in cmd16[:16]],
            [clamp_round(v, 4) for v in state48[:16]],
            clamp_round(safe_float(obj.get("tracking_error_rad")), 5),
            clamp_round(safe_float(obj.get("raw_target_gap_rad")), 5),
            1 if (joint_clamped or gripper_clamped) else 0,
            1 if joint_clamped else 0,
            1 if gripper_clamped else 0,
        ])
    stride = max(1, int(math.ceil(len(rows) / float(max_points)))) if rows else 1
    rows = rows[::stride]
    duration = 0.0
    if infs:
        duration = max(duration, infs[-1]["rel"])
    if rows:
        duration = max(duration, rows[-1][0])
    return {
        "run": run.id,
        "start_utc": start,
        "duration_s": clamp_round(duration, 2),
        "stride": stride,
        "inferences": [
            {k: r[k] for k in ("i", "rel", "sec", "budget", "exec_steps", "delay_steps",
                               "horizon_steps", "prefix_start", "obs", "chunk_id", "seam")}
            for r in infs
        ],
        "last_chunk16": last_chunk,
        "commands": [[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows],
        "meta": meta,
    }


def load_chunk(run: RunInfo, index: int):
    """按 inference 编号取单块 30×16，附带与前一块的接缝。"""
    meta = run_meta(run)
    replan = safe_int((meta.get("limits") or {}).get("replan_steps"), 8) or 8
    prev_chunk = None
    for obj in _iter_jsonl(run.path / "inferences.jsonl"):
        i = safe_int(obj.get("inference"))
        exec_steps, delay, horizon = inference_window(obj, replan)
        is_rtc = delay is not None and horizon is not None
        chunk = _crop_chunk(obj.get("actions19"))
        seam = None
        if chunk and prev_chunk and not is_rtc:
            pos = min(replan, len(prev_chunk)) - 1
            seam = clamp_round(l2(prev_chunk[pos], chunk[0]), 5)
        if i == index:
            if not chunk:
                return None
            return {
                "i": i,
                "seam": seam,
                "chunk16": chunk,
                "exec_steps": exec_steps,
                "delay_steps": delay,
                "horizon_steps": horizon,
                "prefix_start": safe_int(obj.get("rtc_prefix_start")),
                "obs": obj.get("observation"),
                "chunk_id": obj.get("rtc_chunk_id"),
            }
        if chunk:
            prev_chunk = chunk
    return None


def _crop_chunk(chunk):
    if not isinstance(chunk, list):
        return None
    out = []
    for row in chunk[:30]:
        if isinstance(row, list):
            out.append([clamp_round(v, 4) for v in row[:16]])
    return out or None


# --------------------------------------------------------------------------
# 标记与告警存储
# --------------------------------------------------------------------------


class MarkStore:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _path(self, run_id: str) -> Path:
        safe = "".join(ch for ch in run_id if ch.isalnum() or ch in "-_.") or "run"
        return self.root / (safe + ".jsonl")

    def add(self, run_id: str, rel: float | None, note: str, source: str = "user"):
        record = {
            "run": run_id,
            "rel": clamp_round(rel, 3),
            "note": (note or "")[:500],
            "source": source,
            "t": now(),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            try:
                with self._path(run_id).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                LOG.warning("写 mark 失败", exc_info=True)
        return record

    def list(self, run_id: str):
        out = []
        try:
            with self._path(run_id).open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except OSError:
            pass
        return out[-200:]


# --------------------------------------------------------------------------
# 监控主体
# --------------------------------------------------------------------------


class Monitor:
    def __init__(self, config: Config, force_run: str | None = None):
        self.config = config
        self.started_at = now()
        self.cache_dir = self._ensure_cache_dir()
        self.index = RunIndex(config)
        self.broadcaster = Broadcaster()
        self.frames = FrameHub(config)
        self.marks = MarkStore(self.cache_dir / "marks")
        self.state: RunState | None = None
        self.force_run = force_run
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._warnings: list[str] = []
        self._chunk_cache: OrderedDict = OrderedDict()
        self._ui_path = self._resolve_ui()
        self._ui_html = None

    def _ensure_cache_dir(self) -> Path:
        """缓存/标记目录：Linux 用 /data/ta2/infer，Windows 本地调试退回当前目录。"""
        candidates = [self.config.cache_dir]
        if os.name == "nt":
            candidates.append(Path(os.getcwd()) / ".ta2_infer_monitor")
        candidates.append(Path(os.getcwd()) / ".ta2_infer_monitor")
        for cand in candidates:
            try:
                cand.mkdir(parents=True, exist_ok=True)
                self.config.data["cache_dir"] = str(cand)
                return cand
            except OSError:
                continue
        return Path(".")

    def _resolve_ui(self) -> Path:
        configured = self.config.get("ui")
        if configured:
            return Path(configured)
        return Path(__file__).with_name("infer_monitor_ui.html")

    def ui_html(self) -> str:
        if self._ui_html is None:
            try:
                self._ui_html = self._ui_path.read_text(encoding="utf-8")
            except OSError:
                self._ui_html = (
                    "<!doctype html><meta charset='utf-8'><h1>TA2 推理监控台</h1>"
                    "<p>缺少 UI 文件：%s</p>" % self._ui_path
                )
        return self._ui_html

    def start(self):
        self._thread = threading.Thread(target=self._run, name="monitor-watcher", daemon=True)
        self._thread.start()

    def _run(self):
        LOG.info("watcher 启动，run_roots=%s", [str(p) for p in self.config.run_roots])
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                LOG.warning("watcher tick 失败\n%s", traceback.format_exc())
            time.sleep(0.12)

    def _tick(self):
        runs = self.index.scan()
        target = None
        if self.force_run:
            for run in runs:
                if run.id == self.force_run:
                    target = run
                    break
        if target is None:
            target = self.index.pick_live(runs)
        if target is None and self.state is None and runs:
            # 没有 live：默认展示最近一次跑完的 run（可回放）
            target = runs[0]
        if target is None:
            # 当前 run 已结束：继续 tail 一次，保证 report/run_end 被看到
            with self._lock:
                state = self.state
            if state:
                for run in runs:
                    if run.id == state.id:
                        state.run.live = run.live
                        break
                state.poll()
            return
        with self._lock:
            if self.state is None or self.state.id != target.id:
                old = self.state
                self.state = RunState(target, self.config, self.broadcaster,
                                      self.marks, self.frames)
                LOG.info("切换到 run：%s", target.id)
                self.broadcaster.publish(None, "run_changed", {
                    "run": target.id,
                    "prompt": target.prompt,
                })
                if old:
                    old.close()
            state = self.state
        if state:
            for run in runs:
                if run.id == state.id:
                    state.run.live = run.live
                    break
            state.poll()

    def stop(self):
        self._stop.set()
        with self._lock:
            if self.state:
                self.state.close()
                self.state = None
        self.frames.close()

    def current(self) -> RunState | None:
        with self._lock:
            return self.state

    def status(self):
        runs = self.index.scan()
        state = self.current()
        current = state.stats() if state else None
        free = free_gb(self.config.cache_dir)
        warnings = list(self._warnings)
        if not runs:
            warnings.append("没有找到任何 run：检查 run_roots 或 client 是否在跑")
        if not self.frames.enabled:
            warnings.append("帧编码已关闭（--no-frames）")
        elif not shutil.which("ffmpeg"):
            warnings.append("找不到 ffmpeg，帧不可用")
        caps = ((current or {}).get("capabilities") or {}) if current else {}
        if caps:
            if not caps.get("per_inference_obs"):
                warnings.append("当前 run 未逐步落图：视频只能显示静态首/末观测")
            if not caps.get("latency"):
                warnings.append("当前 client 未记录推理耗时：延迟/预算不可用")
        return {
            "ok": True,
            "name": "TA2 推理监控台",
            "version": VERSION,
            "port": self.config["port"],
            "uptime_s": clamp_round(now() - self.started_at, 1),
            "now": now(),
            "clock": datetime.now(self.config.tz).strftime("%H:%M:%S"),
            "tz": self.config.get("timezone"),
            "run_roots": [str(p) for p in self.config.run_roots],
            "cache_dir": str(self.cache_dir),
            "marks_root": str(self.marks.root),
            "marks_count": len(self.marks.list(state.id)) if state else 0,
            "current_run": state.id if state else None,
            "current": current,
            "runs": [r.as_dict() for r in runs[:20]],
            "data_source": "jsonl" if runs else "unsupported",
            "frames": self.frames.status(),
            "viewers": self.broadcaster.count(),
            "disk_free_gb": clamp_round(free, 1),
            "warnings": warnings,
            "expected_prompts": self.config.get("expected_prompts") or [],
        }

    def summary(self, run: RunInfo):
        meta = run_meta(run)
        meta["stats"] = run_stats(run)
        return meta

    def series(self, run: RunInfo, window_s: float):
        state = self.current()
        if state and state.id == run.id:
            infs, cmds, events = state.snapshot_series()
            last = infs[-1]["rel"] if infs else 0.0
            infs = [r for r in infs if r["rel"] is None or last - r["rel"] <= window_s]
            cmds = [c for c in cmds if c["rel"] is None or last - c["rel"] <= window_s]
            stride = max(1, int(math.ceil(len(cmds) / 700.0))) if cmds else 1
            cmds = cmds[::stride]
            rows = [[c["rel"], c["act"], c["cmd"], c["q"], c["err"], c["gap"], c["clamp"],
                     c.get("joint_clamp", 0), c.get("gripper_clamp", 0)]
                    for c in cmds]
            duration = last or 0.0
            infs_out = [
                {"i": r["i"], "rel": r["rel"], "sec": r["sec"], "budget": r.get("budget"),
                 "exec_steps": r.get("exec_steps"), "delay_steps": r.get("delay_steps"),
                 "horizon_steps": r.get("horizon_steps"), "prefix_start": r.get("prefix_start"),
                 "obs": r["obs"], "chunk_id": r["chunk_id"], "seam": r.get("seam")}
                for r in infs
            ]
            if infs_out and state.last_chunk16:
                infs_out[-1]["chunk16"] = state.last_chunk16
            return {
                "run": run.id,
                "start_utc": state.start_utc,
                "duration_s": clamp_round(duration, 2),
                "stride": stride,
                "inferences": infs_out,
                "last_chunk16": state.last_chunk16,
                "commands": rows,
                "events": events,
                "marks": self.marks.list(run.id),
            }
        data = load_run_series(run, window_s)
        data["events"] = []
        data["marks"] = self.marks.list(run.id)
        return data

    def chunk(self, run: RunInfo, index: int):
        state = self.current()
        if (state and state.id == run.id and state.last_inf
                and state.last_inf.get("i") == index and state.last_chunk16):
            return {
                "i": index,
                "seam": state.last_inf.get("seam"),
                "chunk16": state.last_chunk16,
                "exec_steps": state.last_inf.get("exec_steps"),
                "delay_steps": state.last_inf.get("delay_steps"),
                "horizon_steps": state.last_inf.get("horizon_steps"),
                "prefix_start": state.last_inf.get("prefix_start"),
                "obs": state.last_inf.get("obs"),
                "chunk_id": state.last_inf.get("chunk_id"),
            }
        key = (run.id, run.mtime, index)
        with self._lock:
            hit = self._chunk_cache.get(key)
            if hit is not None:
                self._chunk_cache.move_to_end(key)
                return hit
        data = load_chunk(run, index)
        if data:
            with self._lock:
                self._chunk_cache[key] = data
                while len(self._chunk_cache) > 64:
                    self._chunk_cache.popitem(last=False)
        return data


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def json_bytes(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "ta2-infer-monitor/" + VERSION
    protocol_version = "HTTP/1.1"

    @property
    def monitor(self) -> Monitor:
        return self.server.monitor  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # 静音：访问日志太吵
        LOG.debug("%s - %s", self.address_string(), fmt % args)

    # ---- 基础 ----
    def _send(self, code: int, body: bytes, content_type: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, val in (extra or {}).items():
            self.send_header(key, val)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code: int, payload):
        self._send(code, json_bytes(payload), "application/json; charset=utf-8")

    def _error(self, code: int, message: str):
        self._json(code, {"ok": False, "error": message})

    def do_GET(self):
        try:
            self._route_get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            LOG.warning("GET %s 失败\n%s", self.path, traceback.format_exc())
            try:
                self._error(500, "internal error")
            except Exception:
                pass

    def do_POST(self):
        try:
            self._route_post()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            LOG.warning("POST %s 失败\n%s", self.path, traceback.format_exc())
            try:
                self._error(500, "internal error")
            except Exception:
                pass

    # ---- 路由 ----
    def _route_get(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            html = self.monitor.ui_html().encode("utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        if path == "/healthz":
            return self._send(200, b"ok", "text/plain; charset=utf-8")
        if path == "/api/status":
            return self._json(200, self.monitor.status())
        if path == "/api/runs":
            runs = self.monitor.index.scan()
            return self._json(200, {"ok": True, "runs": [r.as_dict() for r in runs[:50]]})
        if path == "/api/discover":
            return self._json(200, {
                "ok": True,
                "run_roots": [{"path": str(p), "exists": p.exists()}
                              for p in self.monitor.config.run_roots],
                "ffmpeg": shutil.which("ffmpeg"),
                "frames": self.monitor.frames.status(),
                "ui": str(self.monitor._ui_path),
            })
        if path == "/api/export":
            run_id = (query.get("run") or [""])[0]
            return self._export(run_id)
        if path.startswith("/api/run/"):
            return self._route_run(path, query)
        return self._error(404, "not found")

    def _route_run(self, path: str, query):
        rest = path[len("/api/run/"):]
        parts = rest.split("/")
        if not parts or not parts[0]:
            return self._error(404, "missing run id")
        run_id = parts[0]
        action = parts[1] if len(parts) > 1 else ""
        run = self.monitor.index.get(run_id)
        if run is None:
            return self._error(404, "run not found: " + run_id)
        if action == "summary":
            return self._json(200, {"ok": True, "summary": self.monitor.summary(run)})
        if action == "series":
            window = safe_float((query.get("window") or ["120"])[0], 120.0) or 120.0
            window = max(5.0, min(window, 3600.0))
            return self._json(200, {"ok": True, "series": self.monitor.series(run, window)})
        if action == "chunk":
            index = safe_int((query.get("i") or ["0"])[0], 0)
            if index is None or index < 0:
                return self._error(400, "bad chunk index")
            chunk = self.monitor.chunk(run, index)
            if not chunk:
                return self._error(404, "chunk not found")
            return self._json(200, {"ok": True, "chunk": chunk})
        if action == "frame":
            cam = (query.get("cam") or ["head"])[0]
            obs = (query.get("obs") or [""])[0]
            number = safe_int((query.get("n") or ["0"])[0], 0) or 0
            if not obs:
                obs = "observation_%03d.npz" % number
            # 只允许读 run 目录内的观测文件，防目录穿越
            obs_path = resolve_obs_path(run, obs)
            if obs_path is None:
                return self._error(400, "bad observation name")
            jpeg = None
            state = self.monitor.current()
            if state and state.id == run.id and run.live:
                jpeg = self.monitor.frames.latest(cam)
            if not jpeg:
                jpeg = self.monitor.frames.frame_for_path(obs_path, cam)
            if not jpeg:
                return self._error(404, "frame not available")
            return self._send(200, jpeg, "image/jpeg")
        if action == "stream":
            return self._stream(run)
        if action == "marks":
            return self._json(200, {"ok": True, "marks": self.monitor.marks.list(run_id)})
        return self._error(404, "unknown run action")

    def _export(self, run_id: str):
        run = self.monitor.index.get(run_id) if run_id else None
        if run is None:
            state = self.monitor.current()
            if state is None:
                return self._error(404, "no run")
            run_id = state.id
            run = state.run
        meta = run_meta(run)
        series = self.monitor.series(run, 3600)
        secs = [i["sec"] for i in series.get("inferences", []) if i.get("sec") is not None]
        # 计数必须读全量 commands.jsonl：series 是降采样过的，不能拿来数次数
        errs = []
        joint_clamps = 0
        gripper_clamps = 0
        try:
            for obj in _iter_jsonl(run.path / "commands.jsonl"):
                if obj.get("limit_clamped_joints"):
                    joint_clamps += 1
                if obj.get("gripper_clamped"):
                    gripper_clamps += 1
                err = safe_float(obj.get("tracking_error_rad"))
                if err is not None:
                    errs.append(err)
        except Exception:
            pass
        rels = [i["rel"] for i in series.get("inferences", []) if i.get("rel") is not None]
        intervals = [b - a for a, b in zip(rels, rels[1:]) if b > a]
        cycle = median(intervals)
        lim = meta.get("limits") or {}
        f = safe_float(lim.get("frequency"), 45.0) or 45.0
        rp = safe_int(lim.get("replan_steps"), 8) or 8
        exec_vals = [i.get("exec_steps") for i in series.get("inferences", []) if i.get("exec_steps")]
        exec_steps = max(1, int(round(median(exec_vals)))) if exec_vals else rp
        exec_plan = exec_steps / f
        is_rtc = bool((meta.get("capabilities") or {}).get("rtc"))
        p50 = percentile(secs, 50)

        def ms(v):
            return ("%.0f" % (v * 1000)) if v is not None else "—"

        if is_rtc:
            loop_line = "RTC 窗口: %s ms（%d 步 = H−D）；推理 %s ms 后台并行，余量 %s ms" % (
                ms(exec_plan), exec_steps, ms(p50),
                ms((exec_plan - p50) if p50 is not None else None))
        else:
            loop_line = "循环    : %s ms（推理 %s + 执行 %s；计划执行 %s，慢 %s）" % (
                ms(cycle), ms(p50),
                ms((cycle - p50) if (cycle and p50 is not None) else None),
                ms(exec_plan),
                ms((cycle - p50 - exec_plan) if (cycle and p50 is not None) else None))

        lines = [
            "TA2 推理摘录",
            "run     : %s" % run.id,
            "prompt  : %s" % (meta.get("prompt") or "—"),
            "port    : %s" % (meta.get("port") or "—"),
            "status  : %s" % (meta.get("status") or "运行中"),
            "推理    : %d 次，p50 %.0f ms，p95 %.0f ms" % (
                len(secs), (p50 or 0) * 1000, (percentile(secs, 95) or 0) * 1000),
            loop_line,
            "跟踪误差: max %.3f rad" % (max(errs) if errs else 0),
            "关节限位/夹爪夹紧: %d / %d 次" % (joint_clamps, gripper_clamps),
            "actuations: %s" % (meta.get("actuations") or "—"),
        ]
        text = "\n".join(lines) + "\n"
        return self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")

    def _stream(self, run: RunInfo):
        state = self.monitor.current()
        if state is None or state.id != run.id:
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            body = json_bytes({"ok": False, "error": "run is not live; use /series"})
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        q = self.monitor.broadcaster.subscribe(run.id)
        self.monitor.frames.add_viewer(+1)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        def send(event: str, data):
            payload = "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        try:
            # hello：带最近 120 秒数据，断线重连不用空窗
            series = self.monitor.series(run, float(self.monitor.config["window_s"]))
            send("hello", {
                "run": run.id,
                "summary": run_meta(run),
                "series": series,
                "status": {
                    "current_run": run.id,
                    "frames": self.monitor.frames.status(),
                },
            })
            last_ping = now()
            while True:
                try:
                    event, data = q.get(timeout=2.0)
                except queue.Empty:
                    if now() - last_ping > 15:
                        send("ping", {"t": now()})
                        last_ping = now()
                    continue
                send(event, data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.monitor.broadcaster.unsubscribe(q)
            self.monitor.frames.add_viewer(-1)

    def _route_post(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = safe_int(self.headers.get("Content-Length"), 0) or 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return self._error(400, "bad json")
        if path == "/api/mark":
            run_id = str(body.get("run") or "")
            if not run_id:
                return self._error(400, "missing run")
            rec = self.monitor.marks.add(run_id, safe_float(body.get("rel"), 0.0),
                                         str(body.get("note") or ""), source="user")
            self.monitor.broadcaster.publish(run_id, "mark", rec)
            return self._json(200, {"ok": True, "mark": rec})
        return self._error(404, "not found")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # Windows 上 SO_REUSEADDR 会允许第二个进程绑同一端口，出现「旧进程接请求」的假象；
    # Linux 保留复用以便快速重启。
    allow_reuse_address = (os.name != "nt")

    def __init__(self, addr, handler, monitor: Monitor):
        self.monitor = monitor
        super().__init__(addr, handler)


# --------------------------------------------------------------------------
# selftest / simulate
# --------------------------------------------------------------------------


def write_npy(path: Path, array_shape, raw: bytes):
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': %r, }" % (tuple(array_shape),)
    header_bytes = header.encode("latin1")
    pad = 64 - ((10 + len(header_bytes) + 1) % 64)
    header_bytes = header_bytes + b" " * pad + b"\n"
    payload = NPY_MAGIC + bytes([1, 0]) + struct.pack("<H", len(header_bytes)) + header_bytes + raw
    with path.open("wb") as fh:
        fh.write(payload)


def write_npz(path: Path, arrays: dict):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, (shape, raw) in arrays.items():
            tmp = io.BytesIO()
            header = "{'descr': '|u1', 'fortran_order': False, 'shape': %r, }" % (tuple(shape),)
            header_bytes = header.encode("latin1")
            pad = 64 - ((10 + len(header_bytes) + 1) % 64)
            header_bytes = header_bytes + b" " * pad + b"\n"
            tmp.write(NPY_MAGIC + bytes([1, 0]) + struct.pack("<H", len(header_bytes)) + header_bytes + raw)
            zf.writestr(name + ".npy", tmp.getvalue())


def make_synthetic_run(root: Path, name: str = "selftest_run", infer_count: int = 40,
                       rtc: bool = False):
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    probe = root / (name + "_probe")
    probe.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc) - timedelta(seconds=infer_count / 5.625 + 2)
    freq, replan = 45.0, 8
    inf_path = run / "inferences.jsonl"
    cmd_path = run / "commands.jsonl"
    with inf_path.open("w", encoding="utf-8") as fh:
        for i in range(infer_count):
            at = start + timedelta(seconds=i / 5.625)
            chunk = []
            for s in range(30):
                row = []
                for j in range(19):
                    base = 0.4 * math.sin((i * replan + s) * 0.05 + j * 0.3)
                    row.append(0.02 * j + base)
                chunk.append(row)
            if i == 30:  # 接缝跳变：前 16 维整体偏移
                chunk = [[v + 0.25 for v in row[:16]] + list(row[16:]) for row in chunk]
            sec = 0.177
            if rtc:
                sec = 0.18
            else:
                if i == 20:
                    sec = 0.45  # 延迟尖峰
                if i == 32:
                    sec = 0.95  # 预算超限
            line = {
                "inference": i,
                "at": at.isoformat().replace("+00:00", "Z"),
                "seconds": sec,
                "observation": "observation_%03d.npz" % i,
                "rtc_chunk_id": 300 + i,
                "actions19": chunk,
            }
            if rtc:
                line.update({"delay_steps": 9, "horizon_steps": 18, "rtc_prefix_start": 9})
            fh.write(json.dumps(line) + "\n")
    with cmd_path.open("w", encoding="utf-8") as fh:
        total = int(infer_count / 5.625 * freq)
        for k in range(total):
            state = [0.02 * j + 0.3 * math.sin(k * 0.03 + j) for j in range(16)]
            state += [0.0] * 16
            state += [0.1] * 16
            cmd = [state[j] + 0.01 * math.sin(k * 0.05 + j) for j in range(16)]
            err = 0.05
            if k == 200:
                err = 0.35  # 跟踪误差尖峰
            clamped = [3] if k == 250 else []
            fh.write(json.dumps({
                "actuation": k,
                "elapsed_s": k / freq,
                "state48": state,
                "command16": cmd,
                "raw_target_gap_rad": 0.04 if k < 300 else 0.7,
                "command_step_rad": 0.00556,
                "tracking_error_rad": err,
                "gripper_clamped": False,
                "limit_clamped_joints": clamped,
            }) + "\n")
    for i in range(infer_count):
        imgs = {
            "observation/images/head_camera": ((8, 8, 3), bytes([20 + i] * (8 * 8 * 3))),
            "observation/images/left_color": ((6, 4, 3), bytes([30] * (6 * 4 * 3))),
            "observation/images/right_color": ((6, 4, 3), bytes([40] * (6 * 4 * 3))),
        }
        write_npz(run / ("observation_%03d.npz" % i), imgs)
    report = {
        "started_at_utc": start.isoformat(),
        "finished_at_utc": (start + timedelta(seconds=infer_count / 5.625)).isoformat(),
        "status": "success",
        "exit_code": 0,
        "actuations": int(infer_count / 5.625 * freq),
        "live_inferences": infer_count,
        "limits": {
            "frequency": freq, "replan_steps": replan, "port": 8002,
            "prompt": "抓取红色沙包", "max_tracking": 0.2, "max_initial_gap": 0.6,
            "max_speed": 0.25, "max_seconds": 60, "rtc": bool(rtc),
        },
        "preflight": {"passed": True, "raw_target_gap_rad": 0.04, "tracking_error_rad": 0.0},
        "rtc": "selftest",
    }
    (run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness = {
        "started_at_utc": start.isoformat(),
        "observation_ready": True,
        "prompt": "抓取红色沙包",
        "mode": {"meta_mode": 1, "current_mode": 1, "enable_chassis": True},
        "mode_fresh": True,
        "api_mode_observed": True,
        "joint_mode_observed": True,
        "motion_mode_compatible": True,
        "camera_mapping": {"head_camera": {"shape": [960, 960, 3]}},
    }
    (probe / "readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    return run


def make_temp_dir(prefix: str) -> Path:
    """在当前目录建临时目录。

    不用 tempfile.mkdtemp：某些 Windows 沙箱下它建出的目录不继承 ACL，
    后续 mkdir/写入会被拒绝；改用普通 os.makedirs 继承父目录权限。
    """
    import random
    import tempfile

    bases = (["/tmp", os.getcwd()] if os.name != "nt" else [os.getcwd(), "/tmp"])
    for base in bases:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for _ in range(20):
            candidate = base_path / (prefix + format(random.getrandbits(40), "010x"))
            try:
                os.makedirs(candidate, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
            except OSError:
                break
    return Path(tempfile.mkdtemp(prefix=prefix))


def selftest(no_frames=True) -> int:
    import urllib.request

    tmp = make_temp_dir("ta2mon_selftest_")
    try:
        run = make_synthetic_run(tmp)
        rtc_run = make_synthetic_run(tmp, "selftest_rtc_run", infer_count=10, rtc=True)
        config = Config({
            "port": 0,
            "host": "127.0.0.1",
            "run_roots": [str(tmp)],
            "cache_dir": str(tmp / "cache"),
            "window_s": 120,
            "frame": {"enabled": not no_frames},
        })
        monitor = Monitor(config, force_run=run.name)
        monitor.start()
        server = Server(("127.0.0.1", 0), Handler, monitor)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(1.2)
        base = "http://127.0.0.1:%d" % port

        def get_json(path):
            with urllib.request.urlopen(base + path, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))

        failures = []
        status = get_json("/api/status")
        if not status.get("ok"):
            failures.append("status ok=false")
        if status.get("current_run") != run.name:
            failures.append("current_run mismatch: %r" % status.get("current_run"))
        current = status.get("current") or {}
        counts = current.get("counts") or {}
        if counts.get("inferences") != 40:
            failures.append("inferences=%r" % counts.get("inferences"))
        if counts.get("actuations") != 320:
            failures.append("actuations=%r" % counts.get("actuations"))
        window = current.get("window") or {}
        if not window.get("sec_max") or window["sec_max"] < 0.9:
            failures.append("max latency 未捕获尖峰：%r" % window.get("sec_max"))
        if not window.get("clamps"):
            failures.append("clamp 未被计数")
        if not window.get("tracking_max") or window["tracking_max"] < 0.2:
            failures.append("tracking_max 未捕获尖峰：%r" % window.get("tracking_max"))
        summary = get_json("/api/run/%s/summary" % run.name)["summary"]
        if not summary.get("preflight", {}).get("passed"):
            failures.append("preflight 未读到")
        series = get_json("/api/run/%s/series" % run.name)["series"]
        if len(series.get("inferences", [])) < 30:
            failures.append("series inferences=%d" % len(series.get("inferences", [])))
        if len(series.get("commands", [])) < 100:
            failures.append("series commands=%d" % len(series.get("commands", [])))
        # 单块接口：回放热力图按时间轴取块
        chunk0 = get_json("/api/run/%s/chunk?i=0" % run.name)["chunk"]
        if not chunk0.get("chunk16") or len(chunk0["chunk16"]) != 30:
            failures.append("chunk i=0 形状异常")
        chunk30 = get_json("/api/run/%s/chunk?i=30" % run.name)["chunk"]
        if chunk30.get("seam") is None:
            failures.append("chunk i=30 缺少接缝")
        # mark
        req = urllib.request.Request(
            base + "/api/mark",
            data=json.dumps({"run": run.name, "rel": 1.0, "note": "selftest"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            marked = json.loads(resp.read().decode())
        if not marked.get("ok"):
            failures.append("mark POST failed")
        marks = get_json("/api/run/%s/marks" % run.name)["marks"]
        if not any(m.get("note") == "selftest" for m in marks):
            failures.append("mark 未落盘")
        # alerts: 检查事件/告警存在
        events = series.get("events") or []
        kinds = {e.get("kind") for e in events}
        for needed in ("latency_spike", "tracking_over", "joint_clamp", "seam_jump"):
            if needed not in kinds:
                failures.append("缺少告警 %s（已有 %s）" % (needed, sorted(kinds)))
        # export
        with urllib.request.urlopen(base + "/api/export?run=" + run.name, timeout=5) as resp:
            text = resp.read().decode("utf-8")
        if "TA2 推理摘录" not in text:
            failures.append("export 内容不对")

        # RTC 预算口径：窗口 = (horizon - delay) / frequency，不是 replan/frequency
        rtc_cfg = Config({
            "port": 0, "host": "127.0.0.1", "run_roots": [str(tmp)],
            "cache_dir": str(tmp / "cache_rtc"), "window_s": 120,
            "frame": {"enabled": False},
        })
        rtc_mon = Monitor(rtc_cfg, force_run=rtc_run.name)
        rtc_mon.start()
        rtc_server = Server(("127.0.0.1", 0), Handler, rtc_mon)
        rtc_base = "http://127.0.0.1:%d" % rtc_server.server_address[1]
        threading.Thread(target=rtc_server.serve_forever, daemon=True).start()
        time.sleep(0.8)

        def rtc_get(path):
            with urllib.request.urlopen(rtc_base + path, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))

        rtc_cur = rtc_get("/api/status").get("current") or {}
        rtc_budget = (rtc_cur.get("last") or {}).get("budget")
        rtc_win = (rtc_cur.get("window") or {}).get("budget_p50")
        rtc_loop = rtc_cur.get("loop") or {}
        # seconds=0.18, H-D=9, f=45 → 0.18/(9/45)=0.9
        if rtc_budget is None or abs(rtc_budget - 0.9) > 0.02:
            failures.append("RTC budget=%r（期望 0.9）" % rtc_budget)
        if rtc_win is None or abs(rtc_win - 0.9) > 0.02:
            failures.append("RTC budget_p50=%r（期望 0.9）" % rtc_win)
        if rtc_loop.get("mode") != "rtc":
            failures.append("RTC loop.mode=%r" % rtc_loop.get("mode"))
        if abs((rtc_loop.get("exec_plan_s") or 0) - 0.2) > 0.01:
            failures.append("RTC exec_plan_s=%r（期望 0.2）" % rtc_loop.get("exec_plan_s"))
        if abs((rtc_loop.get("target_inference_rate_hz") or 0) - 5.0) > 0.1:
            failures.append("RTC target_inference_rate=%r（期望 5.0）" % rtc_loop.get("target_inference_rate_hz"))
        rtc_chunk = rtc_get("/api/run/%s/chunk?i=0" % rtc_run.name)["chunk"]
        if rtc_chunk.get("delay_steps") != 9 or rtc_chunk.get("horizon_steps") != 18:
            failures.append("RTC chunk H/D=%r/%r" % (rtc_chunk.get("delay_steps"), rtc_chunk.get("horizon_steps")))
        if rtc_chunk.get("exec_steps") != 9:
            failures.append("RTC chunk exec_steps=%r（期望 9）" % rtc_chunk.get("exec_steps"))
        rtc_series = rtc_get("/api/run/%s/series" % rtc_run.name)["series"]
        rtc_inf = (rtc_series.get("inferences") or [{}])[0]
        if rtc_inf.get("budget") is None or abs(rtc_inf["budget"] - 0.9) > 0.02:
            failures.append("RTC series budget=%r" % rtc_inf.get("budget"))
        rtc_server.shutdown()
        rtc_mon.stop()
        result = {
            "ok": not failures,
            "failures": failures,
            "inferences": counts.get("inferences"),
            "actuations": counts.get("actuations"),
            "p95_ms": None if window.get("sec_p95") is None else round(window["sec_p95"] * 1000, 1),
            "clamps": window.get("clamps"),
            "events": sorted(kinds),
            "rtc_budget": round(rtc_budget, 3) if rtc_budget is not None else None,
            "rtc_window_ms": round((rtc_loop.get("exec_plan_s") or 0) * 1000, 1),
            "rtc_target_hz": rtc_loop.get("target_inference_rate_hz"),
            "root": str(tmp),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        server.shutdown()
        monitor.stop()
        return 0 if not failures else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def simulate(source: Path, speed: float, port: int, run_root: Path, no_frames: bool):
    """把历史 run 按真实节奏写进临时目录，模拟 live，用于无机器人验收。"""
    tmp = make_temp_dir("ta2mon_sim_")
    target = tmp / ("sim_" + source.name)
    target.mkdir(parents=True, exist_ok=True)
    inf_lines = [json.loads(l) for l in (source / "inferences.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cmd_lines = [json.loads(l) for l in (source / "commands.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    report = read_json(source / "report.json") or {}
    readiness = read_json(source.parent / (source.name + "_probe") / "readiness.json") or {}
    probe = tmp / (target.name + "_probe")
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "readiness.json").write_text(json.dumps(readiness, ensure_ascii=False), encoding="utf-8")
    # 帧：软链到源文件，保持零拷贝
    for line in inf_lines:
        obs = line.get("observation")
        if obs and (source / obs).exists():
            try:
                os.symlink(source / obs, target / obs)
            except OSError:
                pass
    inf_fh = (target / "inferences.jsonl").open("w", encoding="utf-8")
    cmd_fh = (target / "commands.jsonl").open("w", encoding="utf-8")
    stop = threading.Event()

    def feeder():
        start = time.time()
        idx = 0
        for line in inf_lines:
            if stop.is_set():
                break
            due = line.get("at")
            at = iso_to_epoch(due) if due else None
            # 以第一帧为基准按真实间隔排程
            if idx == 0:
                base = at or start
            delay = ((at - base) if at else idx / 5.625) / max(speed, 0.1)
            wait = (start + delay) - time.time()
            if wait > 0:
                stop.wait(wait)
            rec = dict(line)
            rec["at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            inf_fh.write(json.dumps(rec) + "\n")
            inf_fh.flush()
            idx += 1
        stop.wait(0.3)
        # 结束时补 report，时间按本次模拟重写，避免沿用源 run 的旧时间戳
        final = dict(report) if isinstance(report, dict) else {}
        final.setdefault("status", "success")
        final["started_at_utc"] = datetime.fromtimestamp(start, timezone.utc).isoformat()
        final["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (target / "report.json").write_text(json.dumps(final, ensure_ascii=False), encoding="utf-8")

    def cmd_feeder():
        base = time.time()
        freq = 45.0 * max(speed, 0.1)
        for line in cmd_lines:
            if stop.is_set():
                break
            rel = safe_float(line.get("elapsed_s"), 0.0) or 0.0
            wait = base + rel / max(speed, 0.1) - time.time()
            if wait > 0:
                stop.wait(wait)
            cmd_fh.write(json.dumps(line) + "\n")
            cmd_fh.flush()

    t1 = threading.Thread(target=feeder, daemon=True)
    t2 = threading.Thread(target=cmd_feeder, daemon=True)
    config = Config({
        "port": port,
        "run_roots": [str(tmp)],
        "cache_dir": str(tmp / "cache"),
        "frame": {"enabled": not no_frames},
    })
    monitor = Monitor(config, force_run=target.name)
    monitor.start()
    server = Server((config["host"], port), Handler, monitor)
    LOG.info("simulate: http://127.0.0.1:%d  （源 %s，%.1fx）", port, source, speed)
    print("simulate: http://127.0.0.1:%d  （源 %s，%.1fx，Ctrl+C 结束）" % (port, source, speed))
    t1.start()
    t2.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.shutdown()
        monitor.stop()
        inf_fh.close()
        cmd_fh.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TA2 π0.5 推理监控台（只读）")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--config", default=None, help="monitor.json 路径")
    parser.add_argument("--run", action="append", default=None, help="覆盖 run 根目录（可多次）")
    parser.add_argument("--run-id", default=None, help="固定观察某个 run（默认最新 live）")
    parser.add_argument("--ui", default=None, help="UI HTML 路径")
    parser.add_argument("--no-frames", action="store_true", help="关闭帧编码")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--simulate", default=None, help="历史 run 目录，按节奏重放")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--log", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.selftest:
        return selftest(no_frames=True)
    if args.simulate:
        cfg = load_config(args.config)
        port = args.port or int(cfg["port"])
        return simulate(Path(args.simulate), args.speed, port,
                        Path("/tmp"), no_frames=args.no_frames)
    config = load_config(args.config)
    if args.port:
        config.data["port"] = args.port
    if args.host:
        config.data["host"] = args.host
    if args.run:
        config.data["run_roots"] = args.run
    if args.ui:
        config.data["ui"] = args.ui
    if args.no_frames:
        config.data["frame"]["enabled"] = False
    if args.log:
        config.data["log"] = args.log

    # 日志文件（可选）
    log_path = Path(config.get("log") or "")
    if log_path and log_path.parent.exists():
        try:
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOG.addHandler(fh)
        except OSError:
            pass

    if not shutil.which("ffmpeg") and config.data["frame"].get("enabled"):
        LOG.warning("找不到 ffmpeg，帧将不可用（其余功能正常）")

    monitor = Monitor(config, force_run=args.run_id)
    monitor.start()
    host = config["host"]
    port = int(config["port"])
    try:
        server = Server((host, port), Handler, monitor)
    except OSError as exc:
        LOG.error("监听 %s:%s 失败：%s（换 --port 或检查占用）", host, port, exc)
        return 2
    LOG.info("监控台已启动：http://%s:%d  （只读）", host, port)
    print("TA2 推理监控台: http://%s:%d  （只读；Ctrl+C 退出）" % (host, port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        monitor.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
