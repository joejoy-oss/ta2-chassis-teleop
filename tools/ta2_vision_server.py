#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ta2_vision_server.py — TA2 机器人视觉查看器（B70 网页版）

把 ta2-vision/（Windows 上的 Tkinter 六路预览）搬到 B70，以网页形式给手机/电脑看。

架构
    机器人 192.168.31.218:8888 ──RTP/H.265 40Mbps──▶ B70:8890
        └─ ffmpeg（VAAPI 硬解，失败自动退软解）→ 缩放 → JPEG → stdout
              └─ 本服务（纯标准库，:8097）
                    ├─ /            网页（六宫格 / 全景 / 单路）
                    ├─ /mjpeg       multipart/x-mixed-replace，每客户端一条连接
                    ├─ /snapshot.jpg /api/status /api/regions /healthz
                    └─ POST /api/regions 裁剪坐标（归一化）持久化

为什么这么设计
    * 只解一次码、只编一次 JPEG，六路裁剪交给浏览器 canvas —— 一份带宽喂所有观看端。
    * 纯标准库：B70 系统 python 没有 Pillow / numpy，不需要装任何东西。
    * MJPEG <img> 是 iOS Safari / Chrome / Firefox 都不需要额外协议栈的实时通道。

用法
    python3 ta2_vision_server.py --port 8097
    python3 ta2_vision_server.py --selftest          # 不开服务，等 30 帧自检
    /data/ta2/tools/ta2_vision_ops.sh start|stop|status|logs|selftest
"""

import argparse
import hashlib
import base64
import uuid
import collections
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# ================================================================ 常量

DEFAULT_PORT = 8097
DEFAULT_RTP_PORT = 8890
DEFAULT_PAYLOAD = 96
DEFAULT_WIDTH = 640
DEFAULT_FPS = 12
DEFAULT_QUALITY = 6
DEFAULT_REGIONS = "/data/ta2/tools/regions.json"
DEFAULT_LOG_DIR = "/data/ta2/vision"

# 官方拼接图：1280 宽 × 2720 高
SRC_W, SRC_H = 1280, 2720

# 六路在界面上的排布（与 ta2-vision/six_view.py 一致）
REGION_LAYOUT = [
    ("head_left_eye", "head_right_eye"),
    ("left_wrist_left_eye", "left_wrist_right_eye"),
    ("right_wrist_left_eye", "right_wrist_right_eye"),
]
REGION_NAMES = [n for row in REGION_LAYOUT for n in row]

MAX_RESTARTS = 30
NO_FRAME_WAITING_S = 5.0     # 超过这么久没新帧 → waiting
LIVE_FRAME_AGE_S = 2.0       # 有 2 秒内的帧 → live
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

log = logging.getLogger("ta2vision")


# ================================================================ 裁剪表

def _empty_regions_doc():
    """内置一份和 ta2-vision/regions.json 一致的默认值（部署缺文件时兜底）。"""
    return {
        "_comment": "TA2 拼接大图裁剪表，归一化 [x1,y1,x2,y2]",
        "regions": {
            "head_right_eye": [0.1250, 0.0000000000, 0.8750, 0.3529411765],
            "head_left_eye": [0.1250, 0.3529411765, 0.8750, 0.7058823529],
            "right_wrist_left_eye": [0.0000, 0.7058823529, 0.5000, 0.8529411765],
            "right_wrist_right_eye": [0.5000, 0.7058823529, 1.0000, 0.8529411765],
            "left_wrist_left_eye": [0.0000, 0.8529411765, 0.5000, 1.0000000000],
            "left_wrist_right_eye": [0.5000, 0.8529411765, 1.0000, 1.0000000000],
        },
        "policy_mapping": {
            "head_camera": "head_left_eye",
            "left_color": "left_wrist_right_eye",
            "right_color": "right_wrist_left_eye",
        },
    }


def validate_regions_doc(doc):
    """校验并规整裁剪表；不合法直接抛 ValueError。"""
    if not isinstance(doc, dict):
        raise ValueError("裁剪表必须是对象")
    regs = doc.get("regions")
    if not isinstance(regs, dict) or not regs:
        raise ValueError("缺少 regions")

    out = {}
    for name, box in regs.items():
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("%s 必须是 4 个数" % name)
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            raise ValueError("%s 含非数字" % name)
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            raise ValueError("%s 超出 0~1" % name)
        if x1 >= x2 or y1 >= y2:
            raise ValueError("%s 需要 x1<x2 且 y1<y2" % name)
        out[str(name)] = [x1, y1, x2, y2]

    pm = doc.get("policy_mapping") or {}
    if not isinstance(pm, dict):
        raise ValueError("policy_mapping 必须是对象")
    pm = {str(k): str(v) for k, v in pm.items() if str(v) in out}

    # 补上缺失的标准六路（用默认值），保证页面固定六格
    defaults = _empty_regions_doc()["regions"]
    for name in REGION_NAMES:
        if name not in out and name in defaults:
            out[name] = list(defaults[name])

    return {"regions": out, "policy_mapping": pm}


class RegionsStore:
    """裁剪表的内存副本 + 原子落盘。"""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._doc = validate_regions_doc(_empty_regions_doc())
        self.source = "default"

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._doc = validate_regions_doc(raw)
            self.source = self.path
            log.info("裁剪表已载入 %s（%d 路）", self.path, len(self._doc["regions"]))
        except FileNotFoundError:
            log.warning("裁剪表不存在 %s，用内置默认值", self.path)
        except Exception as exc:
            log.warning("裁剪表不合法（%s），用内置默认值: %s", self.path, exc)

    def get(self):
        with self._lock:
            return json.loads(json.dumps(self._doc))

    def snapshot(self):
        with self._lock:
            return self._doc["regions"], self._doc["policy_mapping"]

    def save(self, doc):
        """校验 → 备份 → 原子写 → 更新内存。校验失败抛 ValueError。"""
        if isinstance(doc, dict) and "policy_mapping" not in doc:
            # 页面只回传 regions 时保留原有策略映射，别把它抹掉
            with self._lock:
                doc = dict(doc)
                doc["policy_mapping"] = dict(self._doc["policy_mapping"])
        clean = validate_regions_doc(doc)
        target_dir = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(target_dir, exist_ok=True)
        payload = dict(clean)
        payload["_comment"] = "TA2 拼接大图裁剪表，归一化 [x1,y1,x2,y2]，原点左上角"
        with self._lock:
            if os.path.exists(self.path):
                try:
                    shutil.copyfile(self.path, self.path + ".bak")
                except OSError:
                    pass
            fd, tmp = tempfile.mkstemp(prefix=".regions-", dir=target_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                os.replace(tmp, self.path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            self._doc = {"regions": clean["regions"], "policy_mapping": clean["policy_mapping"]}
            self.source = self.path
        log.info("裁剪表已保存 %s", self.path)
        return self.get()


# ================================================================ 帧中枢

class FrameHub:
    """只保留最新一帧；多个观看端共享。"""

    def __init__(self):
        self.session_id = uuid.uuid4().hex
        self._cv = threading.Condition()
        self._seq = 0
        self._jpeg = None
        self._ts = 0.0
        self._total = 0
        self._ticks = collections.deque(maxlen=240)
        self.width = 0
        self.height = 0

    def publish(self, jpeg, width, height):
        now = time.time()
        with self._cv:
            self._seq += 1
            self._jpeg = jpeg
            self._ts = now
            self._total += 1
            self.width = width
            self.height = height
            self._ticks.append(now)
            self._cv.notify_all()

    def latest(self):
        with self._cv:
            return self._seq, self._jpeg, self._ts, self._total

    def packet(self, regions):
        """One immutable frame and its geometry; receive time is NOT exposure time."""
        with self._cv:
            return {
                "session_id": self.session_id, "seq": self._seq,
                "received_at": self._ts, "width": self.width, "height": self.height,
                "timestamp_kind": "host_decode_receive", "capture_time": None,
                "regions": regions["regions"],
                "regions_version": hashlib.sha256(json.dumps(regions["regions"], sort_keys=True).encode()).hexdigest()[:16],
                "jpeg_b64": base64.b64encode(self._jpeg).decode() if self._jpeg else None,
            }

    def wait_new(self, since_seq, timeout):
        with self._cv:
            if self._seq == since_seq:
                self._cv.wait(timeout)
            return self._seq, self._jpeg

    def fps(self, window=3.0):
        now = time.time()
        with self._cv:
            ticks = [t for t in self._ticks if now - t <= window]
        if len(ticks) < 2:
            return 0.0
        span = max(1e-6, ticks[-1] - ticks[0])
        return (len(ticks) - 1) / span


# ================================================================ 解码器

def frame_size(width, ref_w=SRC_W, ref_h=SRC_H):
    """按拼接图比例算输出高宽（高取偶数，和 scale=h=-2 一致）。"""
    h = int(round(width * ref_h / ref_w))
    if h % 2:
        h += 1
    return width, h


MAX_WIDTH = SRC_W          # 原生 1280，标定用「最高」就是它
MIN_WIDTH = 160


def clamp_int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def write_env_file(args, width=None, fps=None, quality=None):
    """把当前运行参数写进 systemd 的 EnvironmentFile，服务重启后仍然生效。"""
    path = os.path.join(args.log_dir, "env")
    w = width or args.width
    f = fps or args.fps
    q = quality or args.quality
    try:
        os.makedirs(args.log_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("TA2V_PORT=%d\n" % args.port)
            fh.write("TA2V_RTP_PORT=%d\n" % args.rtp_port)
            fh.write("TA2V_WIDTH=%d\n" % w)
            fh.write("TA2V_FPS=%d\n" % f)
            fh.write("TA2V_QUALITY=%d\n" % q)
            fh.write("TA2V_REGIONS=%s\n" % args.regions)
            fh.write("TA2V_LOG_DIR=%s\n" % args.log_dir)
        os.replace(tmp, path)
        log.info("参数已写入 %s（width=%d fps=%d q=%d）", path, w, f, q)
        return True
    except OSError as exc:
        log.warning("写 %s 失败: %s", path, exc)
        return False


def find_vaapi_device(preferred="/dev/dri/renderD128"):
    cands = [preferred] + ["/dev/dri/renderD129", "/dev/dri/renderD128"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def build_sdp(rtp_port, payload=96):
    return (
        "v=0\n"
        "o=- 0 0 IN IP4 127.0.0.1\n"
        "s=TA2\n"
        "c=IN IP4 0.0.0.0\n"
        "t=0 0\n"
        "m=video %d RTP/AVP %d\n"
        "a=rtpmap:%d H265/90000\n" % (rtp_port, payload, payload)
    )


class Decoder(threading.Thread):
    """跑 ffmpeg，把 JPEG 帧灌进 FrameHub；负责重启与后端降级。"""

    HW_ERROR_HINTS = (
        "vaapi", "hwaccel", "Device creation failed", "Cannot load lib",
        "Failed to initialise", "No such device", "Function not implemented",
        "Impossible to convert", "hardware accelerator failed",
    )
    BENIGN_HINTS = (
        "Error parsing NAL unit", "PPS id out of range", "max delay reached",
        "Non-existing PPS", "Could not find ref", "PPS id out of range",
        "SPS id out of range", "Ignoring NAL unit", "Data doesn't look like",
    )

    def __init__(self, args, hub):
        super().__init__(daemon=True, name="decoder")
        self.args = args
        self.hub = hub
        self.out_w, self.out_h = frame_size(args.width)
        self.backend = args.backend if args.backend in ("vaapi", "sw") else "vaapi"
        self.vaapi_dev = find_vaapi_device() if args.backend != "sw" else None
        if self.backend == "vaapi" and not self.vaapi_dev:
            self.backend = "sw"

        self.sdp_path = os.path.join(
            tempfile.gettempdir(), "ta2_vision_%d.sdp" % args.rtp_port)
        self._stop = threading.Event()
        self._proc = None
        self._proc_lock = threading.Lock()
        self._run_had_frame = False

        self.state = "starting"
        self.restarts = 0
        self.started_at = time.time()
        self.proc_started = 0.0
        self.err_tail = collections.deque(maxlen=40)
        self.err_seen = set()
        self.err_repeat = 0
        self.missed = 0
        self.jitter_full = 0
        self.decode_errors = 0
        self.parse_warnings = 0
        self.last_cmd = ""
        self.note = ""

    # ---------------------------------------------------------- 生命周期

    def stop(self):
        self._stop.set()
        self._kill_proc()
        try:
            os.unlink(self.sdp_path)
        except OSError:
            pass

    def reload(self, width=None, fps=None, quality=None):
        """运行时改分辨率/帧率/画质：杀掉 ffmpeg，让守护循环用新参数重拉。

        页面的「高清」按钮走这里，服务本身不停、HTTP 连接不断，画面只黑 ~1 秒。
        """
        changed = False
        with self._proc_lock:
            if width and int(width) != self.args.width:
                self.args.width = int(width)
                self.out_w, self.out_h = frame_size(self.args.width)
                changed = True
            if fps and int(fps) != self.args.fps:
                self.args.fps = int(fps)
                changed = True
            if quality and int(quality) != self.args.quality:
                self.args.quality = int(quality)
                changed = True
        if not changed:
            return False
        self.restarts = 0          # 手动切换不算故障重启
        self.note = ""
        log.info("运行时切换：%dx%d @%dfps q%d", self.args.width, self.out_h,
                 self.args.fps, self.args.quality)
        self._kill_proc()
        # 等内核把 8890 放掉再让守护循环重拉，否则新旧 socket 抢包，新的会一帧收不到
        time.sleep(0.8)
        return True

    def _kill_proc(self):
        with self._proc_lock:
            proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def alive(self):
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    # ---------------------------------------------------------- 命令

    def _vf(self):
        fps = self.args.fps
        w = self.args.width
        if self.backend == "vaapi":
            parts = []
            if w < SRC_W:
                # 只有真要缩小时才挂 scale_vaapi；1280 原生分辨率下这步是空转，
                # 还会多占一层 hw 表面，容易在跟推理抢 GPU 时 sync 失败
                parts.append("scale_vaapi=w=%d:h=-2" % w)
            parts += ["hwdownload", "format=nv12", "format=yuvj420p", "fps=%d" % fps]
            return ",".join(parts)
        return "fps=%d,scale=%d:-2:flags=fast_bilinear" % (fps, w)

    def _cmd(self):
        a = self.args
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning", "-nostats"]
        if self.backend == "vaapi":
            cmd += ["-init_hw_device", "vaapi=va:%s" % self.vaapi_dev,
                    "-filter_hw_device", "va",
                    # 多给解码器几块表面：高分辨率下默认池容易在跟推理抢 GPU 时耗尽
                    "-extra_hw_frames", "16",
                    "-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]
        else:
            cmd += ["-threads", str(max(2, min(16, os.cpu_count() or 4)))]
        cmd += ["-protocol_whitelist", "file,udp,rtp",
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-i", self.sdp_path,
                "-vf", self._vf(),
                "-c:v", "mjpeg", "-q:v", str(a.quality),
                "-f", "image2pipe", "pipe:1"]
        return cmd

    # ---------------------------------------------------------- 主循环

    def run(self):
        try:
            with open(self.sdp_path, "w", encoding="utf-8") as f:
                f.write(build_sdp(self.args.rtp_port, self.args.payload))
        except OSError as exc:
            self.state = "error"
            self.note = "写 SDP 失败: %s" % exc
            log.error(self.note)
            return

        backoff = 1.0
        while not self._stop.is_set():
            self._run_once()
            if self._stop.is_set():
                break
            self.restarts += 1
            if self.restarts >= MAX_RESTARTS:
                self.state = "error"
                self.note = "连续失败 %d 次，暂停 30s 后重试" % self.restarts
                log.error(self.note)
                self._stop.wait(30.0)
                self.restarts = 0
                continue
            self.state = "restarting"
            recent = " ".join(list(self.err_tail)[-4:])
            if "Address already in use" in recent:
                # 端口还攥在别人（或上一个 ffmpeg）手里：慢一点，别连着重启刷屏
                self.note = "UDP %d 被占用，2 秒后重试" % self.args.rtp_port
                self._stop.wait(2.0)
            elif (not self._run_had_frame) and (time.time() - self.proc_started) < 5.0:
                # 切档后端口交接没完成导致的空跑：0.6 秒就重试，别走指数退避
                self._stop.wait(0.6)
                backoff = 1.0
            else:
                self._stop.wait(backoff)
                backoff = min(5.0, backoff * 1.6)

    def _run_once(self):
        cmd = self._cmd()
        self.last_cmd = " ".join(cmd)
        self._run_had_frame = False
        self.err_seen = set()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        except FileNotFoundError:
            self.state = "error"
            self.note = "找不到 ffmpeg，请先安装（apt-get install -y ffmpeg）"
            log.error(self.note)
            self._stop.wait(10.0)
            return
        except Exception as exc:
            self.state = "error"
            self.note = "启动 ffmpeg 失败: %r" % exc
            log.error(self.note)
            self._stop.wait(5.0)
            return

        with self._proc_lock:
            self._proc = proc
        self.proc_started = time.time()
        self.state = "waiting"
        log.info("ffmpeg 启动 backend=%s pid=%d %dx%d@%dfps",
                 self.backend, proc.pid, self.args.width, self.out_h, self.args.fps)

        t_out = threading.Thread(target=self._read_frames, args=(proc,), daemon=True)
        t_err = threading.Thread(target=self._read_stderr, args=(proc,), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait()
        finally:
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            with self._proc_lock:
                if self._proc is proc:
                    self._proc = None

        if self._stop.is_set():
            return

        # 一次跑完都没出帧：VAAPI 起不来就降级
        if not self._run_had_frame and self.backend == "vaapi":
            blob = " ".join(self.err_tail).lower()
            if any(h.lower() in blob for h in self.HW_ERROR_HINTS):
                self.backend = "sw"
                self.note = "VAAPI 不可用，已切换软解（会丢包、CPU 高）"
                log.warning(self.note)
            elif self.proc_started and time.time() - self.proc_started < 6:
                # 起来就死、又没有明显硬件报错 —— 先记下来
                self.note = "ffmpeg 早退且未出帧：%s" % (self.err_tail[-1] if self.err_tail else "-")
                log.warning(self.note)

    # ---------------------------------------------------------- 读帧 / 读日志

    def _read_frames(self, proc):
        buf = b""
        try:
            while not self._stop.is_set():
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    a = buf.find(JPEG_SOI)
                    if a < 0:
                        buf = buf[-1:]
                        break
                    b = buf.find(JPEG_EOI, a + 2)
                    if b < 0:
                        buf = buf[a:]
                        break
                    jpg = buf[a:b + 2]
                    buf = buf[b + 2:]
                    if len(jpg) < 512:
                        continue
                    self.hub.publish(jpg, self.args.width, self.out_h)
                    self._run_had_frame = True
                    if self.state == "waiting":
                        self.state = "live"
                if len(buf) > 16 * 1024 * 1024:
                    buf = b""
        except Exception as exc:  # 管道被 kill 时常见
            if not self._stop.is_set():
                self.err_tail.append("读帧异常: %r" % exc)

    def _read_stderr(self, proc):
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if "deprecated pixel format" in line or "Last message repeated" in line:
                    continue
                m = re.search(r"missed (\d+) packets", line)
                if m:
                    self.missed += int(m.group(1))
                    continue
                if "jitter buffer full" in line:
                    self.jitter_full += 1
                    continue
                if "Error while decoding" in line or "decode_slice_header" in line \
                        or "error while decoding MB" in line:
                    self.decode_errors += 1
                if any(b in line for b in self.BENIGN_HINTS):
                    # 入流对齐 / RTP 分片带来的解析告警：画面正常时属噪声，
                    # 单独计数，别把 err_repeat 和"故障"观感搞乱。
                    self.parse_warnings += 1
                    if line not in self.err_seen:
                        self.err_seen.add(line)
                        self.err_tail.append(line)
                    continue
                if self.err_tail and self.err_tail[-1] == line:
                    self.err_repeat += 1
                    continue
                if line in self.err_seen:
                    self.err_repeat += 1
                    continue
                self.err_seen.add(line)
                self.err_tail.append(line)
                log.debug("ffmpeg: %s", line)
        except Exception:
            pass

    # ---------------------------------------------------------- 状态

    def snapshot(self):
        seq, _, ts, total = self.hub.latest()
        age = (time.time() - ts) if ts else None
        proc_alive = self.alive()

        if self.state == "error":
            state = "error"
        elif age is not None and age <= LIVE_FRAME_AGE_S:
            state = "live"
        elif proc_alive:
            state = "waiting"
        else:
            state = self.state if self.state in ("starting", "restarting") else "waiting"

        with self._proc_lock:
            pid = self._proc.pid if self._proc else None

        return {
            "state": state,
            "backend": self.backend,
            "pid": pid,
            "restarts": self.restarts,
            "frames_total": total,
            "seq": seq,
            "frame_age_ms": int(age * 1000) if age is not None else None,
            "fps": round(self.hub.fps(), 1),
            "loss": {
                "missed": self.missed,
                "jitter_full": self.jitter_full,
                "decode_errors": self.decode_errors,
                "parse_warnings": self.parse_warnings,
            },
            "note": self.note,
            "err_tail": list(self.err_tail)[-20:],
            "err_repeat": self.err_repeat,
            "cmd": self.last_cmd,
            "uptime_s": int(time.time() - self.started_at),
        }


# ================================================================ 占位帧

def make_placeholder(w, h):
    """没画面时给 /mjpeg 持续发一张纯色小图，维持长连接不老化。"""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "lavfi", "-i", "color=c=0x14181f:s=16x34", "-frames:v", "1",
             "-f", "mjpeg", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        if out.returncode == 0 and out.stdout[:2] == JPEG_SOI:
            return out.stdout
    except Exception:
        pass
    return None


# ================================================================ 网页

PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0f1115">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>TA2 视觉查看器</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;
      --ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#4c8dff}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;-webkit-text-size-adjust:100%}
body{display:flex;flex-direction:column;overflow:hidden}
header{flex:0 0 auto;display:flex;align-items:center;gap:6px;padding:6px 8px;
  background:#12151b;border-bottom:1px solid var(--line);overflow-x:auto;white-space:nowrap}
h1{font-size:14px;margin:0 4px 0 0;font-weight:600}
.pill{background:var(--card);border:1px solid var(--line);border-radius:999px;
  padding:2px 9px;font-size:12px;color:var(--dim);flex:0 0 auto}
.pill b{color:var(--fg);font-variant-numeric:tabular-nums}
.pill.live{border-color:#1f6f34;color:var(--ok)}
.pill.bad{border-color:#6e2b2b;color:var(--bad)}
.pill.warn{border-color:#6b5320;color:var(--warn)}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:12px;flex:0 0 auto}
.seg button{background:transparent;color:var(--dim);border:0;padding:5px 10px;font:inherit;cursor:pointer}
.seg button.on{background:#22304a;color:#fff}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg);
  border-radius:8px;padding:5px 9px;font-size:12px;font-family:inherit;cursor:pointer;flex:0 0 auto}
button.ghost:active{background:#22304a}
button.ghost.busy{opacity:.5;pointer-events:none}
button.ghost .ic{font-style:normal;display:none}
.spacer{flex:1}
@media (max-width:600px){
  /* 窄屏：允许换行，宁可两行也不要横向截断 */
  header{flex-wrap:wrap;overflow-x:visible;gap:4px;padding:5px 6px}
  .spacer{display:none}
  h1,#pill-size,#pill-view,#p-loss,#b-fit{display:none}
  .seg button{padding:5px 8px}
  button.ghost{padding:5px 8px}
  button.ghost .tx{display:none}
  button.ghost .ic{display:inline}
}
main{position:relative;flex:1 1 auto;min-height:0;overflow:hidden}

#grid{position:absolute;inset:0;overflow-y:auto;overflow-x:hidden;-webkit-overflow-scrolling:touch;
  display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:8px;
  align-content:start;align-items:start;grid-auto-rows:max-content}
/* 铺满：按可用高度反推列宽，六格一屏全见（宽屏默认开） */
#grid.fit{grid-template-columns:repeat(2,minmax(0,min(calc(50% - 6px), calc((100dvh - 150px) / 3))));
  justify-content:center}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;
  display:flex;flex-direction:column;min-width:0}
.thead{display:flex;flex-wrap:wrap;align-items:center;gap:4px 5px;padding:4px 6px;
  font-size:11px;color:var(--dim);min-height:26px;min-width:0}
.thead input{accent-color:var(--accent);margin:0;width:14px;height:14px;flex:0 0 auto}
.thead .nm{color:var(--fg);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  min-width:0;flex:0 1 auto}
.thead .mt{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  min-width:0;flex:1 1 100%;font-variant-numeric:tabular-nums}
.thead .snap{margin-left:auto;background:transparent;border:0;color:var(--dim);font-size:15px;
  line-height:1;padding:2px 3px;cursor:pointer;flex:0 0 auto}
.tile canvas{display:block;width:100%;height:auto;background:#0b0d11;cursor:zoom-in}
.tile.off{border-color:#1c2029}
.tile.off .thead .nm{color:var(--dim)}
.tile.off canvas{opacity:.18}

#pano{position:absolute;inset:0;display:none}
#single{position:absolute;inset:0;display:none;padding:8px}
#single canvas{display:block;width:100%;height:100%;background:#0b0d11;border-radius:10px}
#single .bar{position:absolute;left:10px;top:10px;display:flex;gap:6px;z-index:3}
#single select{background:var(--card);border:1px solid var(--line);color:var(--fg);
  border-radius:8px;padding:5px 8px;font:inherit;font-size:12px}

#sink{position:fixed;left:-20000px;top:0;width:640px;height:1360px;pointer-events:none}
#sink img{width:100%;height:100%;display:block}
#sink.inview{position:fixed;left:0;right:0;width:auto;pointer-events:auto;z-index:1}
#sink.inview img{object-fit:contain;background:#000}

#overlay{position:fixed;left:0;right:0;pointer-events:none;z-index:2;display:none}
#overlay.on{display:block}
#overlay .hold{position:absolute;pointer-events:none}
#overlay .box{position:absolute;border:1px solid;pointer-events:auto;cursor:move;touch-action:none}
#overlay .box .lbl{position:absolute;left:0;top:0;font-size:10px;background:#000a;
  padding:0 4px;border-radius:0 0 3px 0;white-space:nowrap;color:#fff;font-variant-numeric:tabular-nums}
#overlay .h{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
  background:#0009;border:1px solid #fff8;touch-action:none}
#overlay .h.nw{left:0;top:0}.h.ne{left:100%;top:0}.h.sw{left:0;top:100%}.h.se{left:100%;top:100%}

#void{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;justify-content:center;
  gap:8px;color:var(--dim);font-size:15px;z-index:4;background:#0b0d1199;backdrop-filter:blur(2px)}
#void.show{display:flex}
#void .dot{width:10px;height:10px;border-radius:50%;background:var(--warn);animation:bl 1s infinite alternate}
@keyframes bl{from{opacity:.25}to{opacity:1}}

#diag{position:fixed;left:0;right:0;bottom:0;max-height:62%;overflow:auto;z-index:9;
  background:#11141a;border-top:1px solid var(--line);padding:8px 10px 16px;display:none;
  box-shadow:0 -8px 24px #0008}
#diag.on{display:block}
#diag .row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}
#diag pre{margin:0;font:11px/1.45 ui-monospace,Consolas,monospace;color:#b9c2d6;
  white-space:pre-wrap;word-break:break-all}
#diag h3{font-size:12px;margin:10px 0 4px;color:var(--dim);font-weight:600}
</style>
</head>
<body>
<header>
  <h1>TA2 视觉</h1>
  <span class="pill" id="p-state">连接中</span>
  <span class="pill">fps <b id="p-fps">-</b></span>
  <span class="pill" id="pill-size">尺寸 <b id="p-size">-</b></span>
  <span class="pill" id="pill-view">在线 <b id="p-view">-</b></span>
  <span class="pill" id="p-loss" title="ffmpeg 从本次启动起累计跳过的 RTP 包。启动入流时会一次性跳约 1 帧（20~130 包），之后应保持不涨；持续增长才是真丢包。">丢包 <b id="p-miss">-</b></span>
  <span class="spacer"></span>
  <span class="seg" id="seg">
    <button data-v="grid" class="on">六宫格</button>
    <button data-v="pano">全景</button>
    <button data-v="single">单路</button>
  </span>
  <button class="ghost" id="b-hd"><i class="ic">HD</i><i class="tx">高清</i></button>
  <button class="ghost" id="b-fit"><i class="ic">⤢</i><i class="tx">铺满</i></button>
  <button class="ghost" id="b-full"><i class="ic">⛶</i><i class="tx">全屏</i></button>
  <button class="ghost" id="b-diag"><i class="ic">☰</i><i class="tx">诊断</i></button>
</header>

<main id="main">
  <div id="grid"></div>
  <div id="pano"></div>
  <div id="single">
    <div class="bar"><select id="sel-region"></select></div>
    <canvas id="cv-single"></canvas>
  </div>
  <div id="void"><span class="dot"></span><span id="void-txt">无画面 · 等待视频流</span></div>
</main>

<div id="sink"><img id="feed" src="/mjpeg" alt="TA2 实时画面"></div>
<div id="overlay"><div class="hold" id="hold"></div></div>

<aside id="diag">
  <div class="row">
    <button class="ghost" id="d-reconnect">重连</button>
    <button class="ghost" id="d-copy">复制坐标</button>
    <button class="ghost" id="d-reset">恢复默认</button>
    <button class="ghost" id="d-close">收起</button>
  </div>
  <h3>状态</h3><pre id="d-status">-</pre>
  <h3>ffmpeg 尾部</h3><pre id="d-err">-</pre>
  <h3>命令</h3><pre id="d-cmd">-</pre>
</aside>

<script>
const BOOT = __BOOT__;
const $ = s => document.querySelector(s);
const feed = $('#feed'), main = $('#main'), sink = $('#sink');
const overlay = $('#overlay'), hold = $('#hold');
const grid = $('#grid'), pano = $('#pano'), single = $('#single');
const voidBox = $('#void'), voidTxt = $('#void-txt');

const COLORS = ['#4c8dff','#3fb950','#d29922','#a371f7','#f778ba','#39c5cf'];

const S = {
  view: (new URLSearchParams(location.search).get('view')) ||
        localStorage.getItem('ta2v.view') || 'grid',
  enabled: {},
  overlayOn: localStorage.getItem('ta2v.overlay') !== '0',
  targetFps: 15,
  regions: BOOT.regions,
  live: false,
  lastMissed: 0,
  missedHot: 0,
  missedInit: false,
  cfg: null,
  switching: false,
  fit: localStorage.getItem('ta2v.fit') === null
         ? window.innerWidth >= 900
         : localStorage.getItem('ta2v.fit') === '1'
};
BOOT.layout.forEach(n => { S.enabled[n] = localStorage.getItem('ta2v.on.' + n) !== '0'; });
if (!['grid', 'pano', 'single'].includes(S.view)) S.view = 'grid';

/* ---------------- 分辨率（标定用「高清」） ---------------- */
const PRESET_HD = {width: 1280, fps: 10, quality: 5};      // 原生 1280x2720
const PRESET_NORMAL = {width: 640, fps: 12, quality: 6};
const isHd = () => !!(S.cfg && S.cfg.width >= 1280);

function updateHdBtn(){
  const tx = $('#b-hd .tx');
  if (tx) tx.textContent = isHd() ? '标准' : '高清';
  const b = $('#b-hd');
  if (b) b.title = isHd() ? '切回 640x1360（省带宽）'
                          : '切到原生 1280x2720（视觉标定用，约 15~25 Mbps）';
}

async function setPreset(hd){
  if (S.switching) return;
  S.switching = true;
  const btn = $('#b-hd');
  if (btn) btn.classList.add('busy');
  try {
    const r = await fetch('/api/config', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(hd ? PRESET_HD : PRESET_NORMAL)
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    S.cfg = j.config;
    updateHdBtn();
    voidTxt.textContent = hd ? '正在切到 1280x2720…' : '正在切回 640x1360…';
    voidBox.classList.add('show');
    setTimeout(() => { if (S.live) voidBox.classList.remove('show'); }, 1800);
    setTimeout(fitHold, 1200);
  } catch (e) {
    voidTxt.textContent = '切换分辨率失败：' + e.message;
    voidBox.classList.add('show');
    setTimeout(() => { if (S.live) voidBox.classList.remove('show'); }, 3000);
  } finally {
    if (btn) btn.classList.remove('busy');
    S.switching = false;
  }
}

/* ---------------- 六宫格 ---------------- */
const tiles = [];
function buildGrid(){
  grid.innerHTML = '';
  tiles.length = 0;
  BOOT.rows.forEach(row => {
    row.forEach(name => {
      if (!S.regions[name]) return;
      const b = S.regions[name];
      const policy = Object.keys(BOOT.policy).find(k => BOOT.policy[k] === name) || '';
      const px_w = Math.round((b[2]-b[0]) * BOOT.feed.w);
      const px_h = Math.round((b[3]-b[1]) * BOOT.feed.h);
      const el = document.createElement('div');
      el.className = 'tile';
      el.innerHTML =
        '<div class="thead">' +
          '<input type="checkbox" ' + (S.enabled[name] ? 'checked' : '') + '>' +
          '<span class="nm">' + name + '</span>' +
          '<span class="mt">' + px_w + '×' + px_h + (policy ? ' → ' + policy : '') + '</span>' +
          '<button class="snap" title="保存快照">⤓</button>' +
        '</div><canvas></canvas>';
      grid.appendChild(el);
      const cb = el.querySelector('input');
      cb.addEventListener('change', () => {
        S.enabled[name] = cb.checked;
        localStorage.setItem('ta2v.on.' + name, cb.checked ? '1' : '0');
        el.classList.toggle('off', !cb.checked);
      });
      el.classList.toggle('off', !S.enabled[name]);
      const cv = el.querySelector('canvas');
      setAspect(cv, name);
      el.querySelector('.snap').addEventListener('click', () => saveSnapshot(cv, name));
      cv.title = '点击看单路大图';
      cv.addEventListener('click', () => {
        $('#sel-region').value = name;
        applyView('single');
        drawSingle();
      });
      tiles.push({ name, el, cv, ctx: cv.getContext('2d') });
    });
  });
}

/* ---------------- 全景标注 ---------------- */
function setAspect(cv, name){
  const b = S.regions[name];
  if (!b) return;
  const w = (b[2]-b[0]) * BOOT.feed.w;
  const h = (b[3]-b[1]) * BOOT.feed.h;
  if (w > 0 && h > 0) cv.style.aspectRatio = w + ' / ' + h;
}
function buildOverlay(){
  hold.innerHTML = '';
  BOOT.layout.forEach((name, i) => {
    if (!S.regions[name]) return;
    const b = S.regions[name];
    const el = document.createElement('div');
    el.className = 'box';
    el.dataset.name = name;
    el.style.borderColor = COLORS[i % COLORS.length];
    ['nw','ne','sw','se'].forEach(k => {
      const h = document.createElement('i');
      h.className = 'h ' + k; h.dataset.k = k;
      el.appendChild(h);
    });
    const lb = document.createElement('span');
    lb.className = 'lbl';
    el.appendChild(lb);
    hold.appendChild(el);
    paintBox(el, name);
  });
  overlay.classList.toggle('on', S.overlayOn);
}

function paintBox(el, name){
  const b = S.regions[name];
  el.style.left = (b[0]*100) + '%';
  el.style.top = (b[1]*100) + '%';
  el.style.width = ((b[2]-b[0])*100) + '%';
  el.style.height = ((b[3]-b[1])*100) + '%';
  const lb = el.querySelector('.lbl');
  if (lb) lb.textContent = name + ' ' + Math.round((b[2]-b[0])*BOOT.feed.w) + '×' + Math.round((b[3]-b[1])*BOOT.feed.h);
}

function fitHold(){
  const r = main.getBoundingClientRect();
  sink.style.top = r.top + 'px';
  sink.style.height = r.height + 'px';
  overlay.style.top = r.top + 'px';
  overlay.style.height = r.height + 'px';
  const W = feed.naturalWidth || BOOT.feed.w, H = feed.naturalHeight || BOOT.feed.h;
  const s = Math.min(r.width / W, r.height / H);
  const w = W * s, h = H * s;
  hold.style.left = ((r.width - w) / 2) + 'px';
  hold.style.top = ((r.height - h) / 2) + 'px';
  hold.style.width = w + 'px';
  hold.style.height = h + 'px';
}

/* ---------------- 绘制 ---------------- */
function drawGrid(){
  const W = feed.naturalWidth, H = feed.naturalHeight;
  if (!W || !H) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  for (const t of tiles) {
    if (!S.enabled[t.name]) continue;
    const cssW = t.cv.clientWidth;
    if (!cssW) continue;
    const b = S.regions[t.name];
    const ratio = ((b[3]-b[1]) * H) / ((b[2]-b[0]) * W);
    const cssH = Math.max(8, Math.round(cssW * ratio));
    // 关键：显式写死高度。canvas 是 width:100% 的替换元素，
    // 在 grid 里算不出内容高度，行会被均分压扁（桌面端六格只露出上半截）。
    const hs = cssH + 'px';
    if (t.cv.style.height !== hs) t.cv.style.height = hs;
    const wantW = Math.round(cssW * dpr);
    const wantH = Math.round(cssH * dpr);
    if (t.cv.width !== wantW || t.cv.height !== wantH) { t.cv.width = wantW; t.cv.height = wantH; }
    const sx = b[0]*W, sy = b[1]*H, sw = (b[2]-b[0])*W, sh = (b[3]-b[1])*H;
    try { t.ctx.drawImage(feed, sx, sy, sw, sh, 0, 0, t.cv.width, t.cv.height); } catch (e) {}
  }
}

function drawSingle(){
  const name = $('#sel-region').value;
  const b = S.regions[name]; if (!b) return;
  const cv = $('#cv-single'), ctx = cv.getContext('2d');
  const W = feed.naturalWidth, H = feed.naturalHeight;
  if (!W || !H) return;
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const wantW = Math.max(1, Math.round(r.width * dpr));
  const wantH = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== wantW || cv.height !== wantH) { cv.width = wantW; cv.height = wantH; }
  const sw = (b[2]-b[0])*W, sh = (b[3]-b[1])*H;
  const s = Math.min(cv.width / sw, cv.height / sh);
  const dw = sw * s, dh = sh * s;
  ctx.fillStyle = '#0b0d11'; ctx.fillRect(0,0,cv.width,cv.height);
  try { ctx.drawImage(feed, b[0]*W, b[1]*H, sw, sh,
        (cv.width-dw)/2, (cv.height-dh)/2, dw, dh); } catch (e) {}
}

let lastDraw = 0;
function raf(ts){
  requestAnimationFrame(raf);
  if (ts - lastDraw < 1000 / S.targetFps) return;
  lastDraw = ts;
  if (!feed.naturalWidth) return;
  if (S.view === 'grid') drawGrid();
  else if (S.view === 'single') drawSingle();
}

function applyFit(){
  grid.classList.toggle('fit', !!S.fit);
  const tx = $('#b-fit .tx');
  if (tx) tx.textContent = S.fit ? '放大' : '铺满';
  try { localStorage.setItem('ta2v.fit', S.fit ? '1' : '0'); } catch (e) {}
}

/* ---------------- 视图切换 ---------------- */
function applyView(v){
  S.view = v;
  localStorage.setItem('ta2v.view', v);
  document.querySelectorAll('#seg button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  grid.style.display = v === 'grid' ? 'grid' : 'none';
  pano.style.display = v === 'pano' ? 'block' : 'none';
  single.style.display = v === 'single' ? 'block' : 'none';
  sink.classList.toggle('inview', v === 'pano');
  overlay.classList.toggle('on', v === 'pano' && S.overlayOn);
  if (v === 'pano') fitHold();
}

/* ---------------- 裁剪拖动 ---------------- */
let drag = null;
overlay.addEventListener('pointerdown', e => {
  const box = e.target.closest('.box');
  if (!box) return;
  const k = e.target.dataset.k || 'move';
  drag = { name: box.dataset.name, k, x0: e.clientX, y0: e.clientY,
           b: [...S.regions[box.dataset.name]], r: hold.getBoundingClientRect() };
  box.setPointerCapture(e.pointerId);
  e.preventDefault();
});
overlay.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = (e.clientX - drag.x0) / drag.r.width;
  const dy = (e.clientY - drag.y0) / drag.r.height;
  let [x1, y1, x2, y2] = drag.b;
  const MIN = 0.02;
  if (drag.k === 'move') {
    const cdx = Math.max(-x1, Math.min(1 - x2, dx));
    const cdy = Math.max(-y1, Math.min(1 - y2, dy));
    x1 += cdx; x2 += cdx; y1 += cdy; y2 += cdy;
  } else {
    if (drag.k.includes('w')) x1 = Math.max(0, Math.min(x2 - MIN, x1 + dx));
    if (drag.k.includes('e')) x2 = Math.min(1, Math.max(x1 + MIN, x2 + dx));
    if (drag.k.includes('n')) y1 = Math.max(0, Math.min(y2 - MIN, y1 + dy));
    if (drag.k.includes('s')) y2 = Math.min(1, Math.max(y1 + MIN, y2 + dy));
  }
  S.regions[drag.name] = [x1, y1, x2, y2];
  const el = hold.querySelector('.box[data-name="' + drag.name + '"]');
  if (el) paintBox(el, drag.name);
  e.preventDefault();
});
overlay.addEventListener('pointerup', () => {
  if (!drag) return;
  drag = null;
  saveRegions();
});

let savePending = false, saveDirty = false;
async function saveRegions(){
  if (savePending) { saveDirty = true; return; }
  savePending = true;
  try {
    const res = await fetch('/api/regions', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ regions: S.regions })
    });
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    S.regions = j.regions;
    refreshMeta();
  } catch (e) {
    voidTxt.textContent = '坐标保存失败：' + e.message;
    voidBox.classList.add('show');
    setTimeout(() => voidBox.classList.remove('show'), 2500);
  } finally {
    savePending = false;
    if (saveDirty) { saveDirty = false; setTimeout(saveRegions, 60); }
  }
}

function refreshMeta(){
  // 用当前生效的分辨率算像素尺寸（切了「高清」之后要跟着变）
  const curW = (S.cfg && S.cfg.width) || BOOT.feed.w;
  const curH = (S.cfg && S.cfg.height) || BOOT.feed.h;
  for (const t of tiles) {
    const b = S.regions[t.name];
    const policy = Object.keys(BOOT.policy).find(k => BOOT.policy[k] === t.name) || '';
    t.el.querySelector('.mt').textContent =
      Math.round((b[2]-b[0]) * curW) + '×' + Math.round((b[3]-b[1]) * curH) +
      (policy ? ' → ' + policy : '');
    setAspect(t.cv, t.name);
  }
  hold.querySelectorAll('.box').forEach(el => paintBox(el, el.dataset.name));
}

/* ---------------- 快照 ---------------- */
function saveSnapshot(cv, name){
  if (!cv.width) return;
  cv.toBlob(b => {
    if (!b) return;
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.href = URL.createObjectURL(b);
    a.download = 'TA2_' + name + '_' + ts + '.png';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, 'image/png');
}

/* ---------------- 状态轮询 ---------------- */
const STATE_TEXT = { starting: '启动中', waiting: '等流中', live: '接收中', restarting: '重连中', error: '故障' };
async function poll(){
  try {
    const r = await fetch('/api/status', {cache: 'no-store'});
    const j = await r.json();
    const st = $('#p-state');
    st.textContent = STATE_TEXT[j.state] || j.state;
    st.className = 'pill ' + (j.state === 'live' ? 'live' : (j.state === 'error' ? 'bad' : 'warn'));
    $('#p-fps').textContent = (j.fps || 0).toFixed(1);
    $('#p-size').textContent = j.frame && j.frame.w ? (j.frame.w + '×' + j.frame.h) : '-';
    $('#p-view').textContent = j.viewers;
    const miss = j.loss ? j.loss.missed : 0;
    $('#p-miss').textContent = miss;
    if (!S.missedInit) {
      // 首帧只记录基线：启动入流的那一次性跳包不该标红
      S.missedInit = true;
      S.lastMissed = miss;
    } else {
      if (miss > S.lastMissed) S.missedHot = Date.now();
      S.lastMissed = miss;
    }
    const hot = Date.now() - (S.missedHot || 0) < 6000;
    $('#p-loss').className = 'pill' + (hot ? ' bad' : '');
    // 同步服务端配置：分辨率变了要刷新按钮文案和格子里的像素尺寸
    if (j.config) {
      const changed = !S.cfg || S.cfg.width !== j.config.width;
      S.cfg = j.config;
      if (changed) { updateHdBtn(); refreshMeta(); }
    }
    S.live = j.state === 'live';
    voidBox.classList.toggle('show', !S.live);
    if (!S.live) voidTxt.textContent = '无画面 · ' + (j.note || (j.state === 'waiting' ? '等待视频流' : STATE_TEXT[j.state] || j.state));
    if ($('#diag').classList.contains('on')) {
      $('#d-status').textContent = JSON.stringify(j, null, 2) + '\n\n本页 ' + location.href;
      $('#d-err').textContent = (j.err_tail || []).join('\n') || '（空）';
      $('#d-cmd').textContent = j.cmd || '-';
    }
  } catch (e) {
    $('#p-state').textContent = '服务离线'; $('#p-state').className = 'pill bad';
    voidBox.classList.add('show'); voidTxt.textContent = '服务离线 · 重试中';
  }
}

/* ---------------- 事件 ---------------- */
document.querySelectorAll('#seg button').forEach(b =>
  b.addEventListener('click', () => applyView(b.dataset.v)));

$('#b-hd').addEventListener('click', () => setPreset(!isHd()));
$('#b-fit').addEventListener('click', () => { S.fit = !S.fit; applyFit(); });
$('#b-full').addEventListener('click', () => {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();
});
$('#b-diag').addEventListener('click', () => $('#diag').classList.toggle('on'));
$('#d-close').addEventListener('click', () => $('#diag').classList.remove('on'));
$('#d-reconnect').addEventListener('click', () => {
  feed.src = '/mjpeg?_=' + Date.now();
});
$('#d-copy').addEventListener('click', async () => {
  const txt = JSON.stringify(S.regions, null, 2);
  try { await navigator.clipboard.writeText(txt); voidTxt.textContent = '坐标已复制'; }
  catch (e) { prompt('复制坐标', txt); }
  voidBox.classList.add('show'); setTimeout(() => { if (S.live) voidBox.classList.remove('show'); }, 1500);
});
$('#d-reset').addEventListener('click', () => {
  if (!confirm('恢复成服务端默认裁剪表？')) return;
  S.regions = JSON.parse(JSON.stringify(BOOT.defaults));
  hold.querySelectorAll('.box').forEach(el => paintBox(el, el.dataset.name));
  refreshMeta(); saveRegions();
});
$('#sel-region').addEventListener('change', drawSingle);

feed.addEventListener('error', () => {
  setTimeout(() => { feed.src = '/mjpeg?_=' + Date.now(); }, 2000);
});
feed.addEventListener('load', () => { if (S.view === 'pano') fitHold(); });

window.addEventListener('resize', () => { fitHold(); applyFit(); });
window.addEventListener('orientationchange', () => setTimeout(fitHold, 200));

/* ---------------- 启动 ---------------- */
$('#sel-region').innerHTML = BOOT.layout.map(n => '<option>' + n + '</option>').join('');
buildGrid(); buildOverlay(); applyFit(); applyView(S.view); updateHdBtn();
requestAnimationFrame(raf);
poll(); setInterval(poll, 1500);
if ('wakeLock' in navigator) { navigator.wakeLock.request('screen').catch(() => {}); }
</script>
</body>
</html>
"""


def render_page(boot):
    return PAGE_HTML.replace("__BOOT__", json.dumps(boot, ensure_ascii=False))


def build_boot(regions_store, args):
    regs, policy = regions_store.snapshot()
    w, h = frame_size(args.width)
    rows = []
    used = set()
    for row in REGION_LAYOUT:
        keep = [n for n in row if n in regs]
        if keep:
            rows.append(keep)
            used.update(keep)
    leftovers = [n for n in regs if n not in used]
    for i in range(0, len(leftovers), 2):
        rows.append(leftovers[i:i + 2])
    order = [n for row in rows for n in row]
    return {
        "regions": regs,
        "defaults": _empty_regions_doc()["regions"],
        "policy": policy,
        "layout": order,
        "rows": rows,
        "feed": {"w": w, "h": h},
        "port": args.port,
        "rtp": {"port": args.rtp_port, "payload": args.payload, "codec": "hevc"},
    }


# ================================================================ HTTP

class VisionServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler):
        super().__init__(addr, handler)
        self.hub = None
        self.decoder = None
        self.regions = None
        self.args = None
        self.placeholder = None
        self.viewers = 0
        self._vlock = threading.Lock()
        self.stop_event = threading.Event()
        self.http_log = collections.deque(maxlen=200)
        self.seen_ips = {}          # ip -> {"ua":..., "first":ts, "last":ts, "n":int}
        self.started_at = time.time()

    def viewer_add(self):
        with self._vlock:
            self.viewers += 1
            return self.viewers

    def viewer_del(self):
        with self._vlock:
            self.viewers = max(0, self.viewers - 1)

    def viewer_count(self):
        with self._vlock:
            return self.viewers


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ta2vision/1.0"

    # ---- 工具 ----
    def _touch_client(self, path):
        """记录客户端首次访问，方便现场判断"浏览器到底有没有打过来"。"""
        try:
            ip = self.client_address[0]
            ua = (self.headers.get("User-Agent") or "")[:120]
            srv = self.server
            now = time.time()
            with srv._vlock:
                rec = srv.seen_ips.get(ip)
                if rec is None:
                    srv.seen_ips[ip] = {"ua": ua, "first": now, "last": now, "n": 1,
                                        "path": path}
                    first = True
                else:
                    rec["last"] = now
                    rec["n"] += 1
                    first = False
            if first:
                log.info("首次访问 %s  %s  UA=%s", ip, path, ua)
        except Exception:
            pass

    def log_message(self, fmt, *args):
        try:
            self.server.http_log.append("%s %s" % (time.strftime("%H:%M:%S"), fmt % args))
        except Exception:
            pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _text(self, text, code=200, ctype="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _status(self):
        srv = self.server
        dec = srv.decoder.snapshot()
        hub = srv.hub
        seq, jpeg, ts, total = hub.latest()
        srv_h = hub.height or frame_size(srv.args.width)[1]
        body = {
            "ok": True,
            "state": dec["state"],
            "backend": dec["backend"],
            "rtp": {"port": srv.args.rtp_port, "payload": srv.args.payload, "codec": "hevc"},
            "frame": {
                "seq": seq,
                "w": hub.width or srv.args.width,
                "h": srv_h,
                "bytes": len(jpeg) if jpeg else 0,
                "age_ms": dec["frame_age_ms"],
            },
            "fps": dec["fps"],
            "frames_total": total,
            "loss": dec["loss"],
            "viewers": srv.viewer_count(),
            "restarts": dec["restarts"],
            "pid": dec["pid"],
            "uptime_s": int(time.time() - srv.started_at),
            "note": dec["note"],
            "err_tail": dec["err_tail"],
            "err_repeat": dec.get("err_repeat", 0),
            "cmd": dec["cmd"],
            "url_hint": "http://<本机IP>:%d" % srv.args.port,
            "regions_source": srv.regions.source,
            "config": {
                "width": srv.decoder.args.width,
                "height": frame_size(srv.decoder.args.width)[1],
                "fps": srv.decoder.args.fps,
                "quality": srv.decoder.args.quality,
                "max_width": MAX_WIDTH,
            },
        }
        try:
            now = time.time()
            with srv._vlock:
                clients = [{"ip": ip, "ua": r["ua"][:70], "n": r["n"],
                            "ago_s": round(now - r["last"]), "path": r.get("path", "")}
                           for ip, r in srv.seen_ips.items()]
            clients.sort(key=lambda c: c["ago_s"])
            body["clients"] = clients[-8:]
        except Exception:
            pass
        try:
            with open("/proc/loadavg") as f:
                body["load"] = float(f.read().split()[0])
        except Exception:
            pass
        return body

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        qs = parse_qs(u.query)
        self._touch_client(path)

        if path == "/api/frame":
            packet = self.server.hub.packet(self.server.regions.get())
            return self._json(packet, 200 if packet["jpeg_b64"] else 503)

        if path == "/perception":
            return self._text('<!doctype html><meta charset="utf-8"><script>location.port="8100";</script>正在打开识别定位页面', ctype="text/html; charset=utf-8")

        if path == "/api/perception":
            # Fixed loopback destination: never turn this into an arbitrary URL proxy.
            import urllib.request
            try:
                route = "/" if path == "/perception" else path
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open("http://127.0.0.1:8100" + route, timeout=1) as response:
                    body = response.read().decode("utf-8")
                return self._text(body, ctype="text/html; charset=utf-8" if route == "/" else "application/json")
            except Exception:
                return self._json({"state": "unavailable", "reason": "感知服务未启动", "grasp_candidates": []}, 503)

        if path in ("/", "/index.html"):
            boot = build_boot(self.server.regions, self.server.args)
            page = render_page(boot).replace("</body>", '<a href="/perception" style="position:fixed;right:18px;bottom:18px;z-index:50;background:#185c50;color:white;padding:10px 16px;border-radius:8px;text-decoration:none">沙包识别与标定 ↗</a></body>')
            return self._text(page, ctype="text/html; charset=utf-8")

        if path == "/healthz":
            return self._text("ok")

        if path == "/api/status":
            return self._json(self._status())

        if path == "/api/regions":
            regs, policy = self.server.regions.snapshot()
            w, h = frame_size(self.server.args.width)
            return self._json({"regions": regs, "policy_mapping": policy,
                               "frame": {"w": w, "h": h}})

        if path == "/snapshot.jpg":
            _, jpeg, _, _ = self.server.hub.latest()
            if not jpeg:
                return self._json({"ok": False, "error": "还没有画面"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Disposition",
                             'attachment; filename="ta2_snapshot_%s.jpg"'
                             % time.strftime("%Y%m%d_%H%M%S"))
            self.end_headers()
            try:
                self.wfile.write(jpeg)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if path == "/mjpeg":
            return self._mjpeg(qs)

        return self._json({"ok": False, "error": "not found", "path": path}, 404)

    def _mjpeg(self, qs):
        srv = self.server
        try:
            want_fps = float((qs.get("fps") or ["0"])[0])
        except ValueError:
            want_fps = 0.0
        min_interval = (1.0 / want_fps) if want_fps > 0.3 else 0.0

        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        srv.viewer_add()
        last_seq = srv.hub.latest()[0]
        last_sent = 0.0
        idle_sent = 0.0
        try:
            while not srv.stop_event.is_set():
                seq, jpeg = srv.hub.wait_new(last_seq, timeout=1.0)
                if seq == last_seq:
                    # 一秒内没新帧：最多每 2 秒发一张占位，保持长连接不老化
                    placeholder = srv.placeholder
                    if not placeholder or time.time() - idle_sent < 2.0:
                        continue
                    payload = placeholder
                    idle_sent = time.time()
                    seq_to_send = last_seq
                else:
                    if not jpeg:
                        continue
                    payload = jpeg
                    seq_to_send = seq
                if min_interval:
                    wait = min_interval - (time.time() - last_sent)
                    if wait > 0:
                        time.sleep(wait)
                hdr = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(payload)).encode() + b"\r\n\r\n")
                self.wfile.write(hdr + payload + b"\r\n")
                self.wfile.flush()
                last_sent = time.time()
                last_seq = seq_to_send
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass
        finally:
            srv.viewer_del()

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        self._touch_client(u.path)
        body = self._read_json_body()
        if body is None:
            return
        if u.path == "/api/regions":
            return self._post_regions(body)
        if u.path == "/api/config":
            return self._post_config(body)
        return self._json({"ok": False, "error": "not found"}, 404)

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 256 * 1024:
            self._json({"ok": False, "error": "body 长度不合法"}, 400)
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._json({"ok": False, "error": "JSON 解析失败: %s" % exc}, 400)
            return None

    def _post_regions(self, doc):
        if isinstance(doc, dict) and "regions" not in doc and "policy_mapping" not in doc:
            doc = {"regions": doc}
        try:
            clean = self.server.regions.save(doc)
        except ValueError as exc:
            return self._json({"ok": False, "error": str(exc)}, 400)
        except OSError as exc:
            return self._json({"ok": False, "error": "写文件失败: %s" % exc}, 500)
        return self._json({"ok": True, "regions": clean["regions"],
                           "policy_mapping": clean["policy_mapping"],
                           "source": self.server.regions.source})

    def _post_config(self, doc):
        """运行时改分辨率/帧率/画质（页面的「高清」按钮）。"""
        if not isinstance(doc, dict):
            return self._json({"ok": False, "error": "body 必须是对象"}, 400)
        dec = self.server.decoder
        args = self.server.args
        width = clamp_int(doc.get("width", dec.args.width), MIN_WIDTH, MAX_WIDTH, dec.args.width)
        fps = clamp_int(doc.get("fps", dec.args.fps), 1, 30, dec.args.fps)
        quality = clamp_int(doc.get("quality", dec.args.quality), 2, 31, dec.args.quality)

        changed = dec.reload(width=width, fps=fps, quality=quality)
        ok_env = write_env_file(args, width=width, fps=fps, quality=quality)
        log.info("配置变更 %dx%d @%dfps q%d（changed=%s, env=%s）",
                 width, frame_size(width)[1], fps, quality, changed, ok_env)
        return self._json({
            "ok": True,
            "changed": changed,
            "persisted": ok_env,
            "config": {"width": dec.args.width,
                       "height": frame_size(dec.args.width)[1],
                       "fps": dec.args.fps,
                       "quality": dec.args.quality,
                       "max_width": MAX_WIDTH},
        })


# ================================================================ 自检

def run_selftest(args, decoder, hub, need=30, timeout=40.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        seq, jpeg, ts, total = hub.latest()
        if total >= need:
            break
        if decoder.state == "error" and not decoder.alive():
            break
        time.sleep(0.2)
    seq, jpeg, ts, total = hub.latest()
    ok = total >= need
    print("[selftest] 帧数=%d 需要>=%d 用时=%.1fs" % (total, need, time.time() - t0))
    if jpeg:
        print("[selftest] 最新帧 %d 字节, 输出 %dx%d" %
              (len(jpeg), hub.width, hub.height))
        print("[selftest] SOI=%s EOI=%s" %
              (jpeg[:2] == JPEG_SOI, jpeg[-2:] == JPEG_EOI))
    snap = decoder.snapshot()
    print("[selftest] state=%s backend=%s fps=%.1f missed=%d jitter_full=%d decode_errors=%d" %
          (snap["state"], snap["backend"], snap["fps"], snap["loss"]["missed"],
           snap["loss"]["jitter_full"], snap["loss"]["decode_errors"]))
    if snap["note"]:
        print("[selftest] note:", snap["note"])
    for line in snap["err_tail"][-6:]:
        print("[selftest] ffmpeg:", line)
    if not ok:
        print("[selftest] 失败：没收到足够帧。检查机器人是否在推流、UDP 端口是否被占。")
        return 1
    if snap["loss"]["missed"] > 200:
        print("[selftest] 警告：丢包偏多，画面可能花屏")
    if jpeg and not (jpeg[:2] == JPEG_SOI and jpeg[-2:] == JPEG_EOI):
        print("[selftest] 失败：帧边界不对")
        return 2
    return 0


# ================================================================ main

def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_str(name, default):
    v = os.environ.get(name, "").strip()
    return v or default


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="TA2 机器人视觉查看器（网页版）")
    ap.add_argument("--host", default=_env_str("TA2V_HOST", "0.0.0.0"), help="监听地址")
    ap.add_argument("--port", type=int, default=_env_int("TA2V_PORT", DEFAULT_PORT),
                    help="HTTP 端口")
    ap.add_argument("--rtp-port", type=int, default=_env_int("TA2V_RTP_PORT", DEFAULT_RTP_PORT),
                    help="机器人推流 UDP 端口")
    ap.add_argument("--payload", type=int, default=_env_int("TA2V_PAYLOAD", DEFAULT_PAYLOAD),
                    help="RTP payload type")
    ap.add_argument("--width", type=int, default=_env_int("TA2V_WIDTH", DEFAULT_WIDTH),
                    help="解码后缩放宽（默认 640）")
    ap.add_argument("--fps", type=int, default=_env_int("TA2V_FPS", DEFAULT_FPS),
                    help="输出帧率（默认 12）")
    ap.add_argument("--quality", type=int, default=_env_int("TA2V_QUALITY", DEFAULT_QUALITY),
                    help="JPEG 质量，ffmpeg -q:v，2~31，越小越清晰（默认 6）")
    ap.add_argument("--backend", default=_env_str("TA2V_BACKEND", "auto"),
                    choices=("auto", "vaapi", "sw"),
                    help="解码后端：auto=先 VAAPI 失败退软解")
    ap.add_argument("--regions", default=_env_str("TA2V_REGIONS", DEFAULT_REGIONS),
                    help="裁剪表路径")
    ap.add_argument("--log-dir", default=_env_str("TA2V_LOG_DIR", DEFAULT_LOG_DIR),
                    help="日志/临时目录")
    ap.add_argument("--selftest", action="store_true", help="只跑自检，不提供服务")
    ap.add_argument("--selftest-frames", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        os.makedirs(args.log_dir, exist_ok=True)
    except OSError as exc:
        print("警告：建日志目录失败 %s: %s" % (args.log_dir, exc))

    if args.width < 160 or args.width > 1920:
        print("--width 建议 160~1920")
        args.width = max(160, min(1920, args.width))
    args.quality = max(2, min(31, args.quality))
    args.fps = max(1, min(30, args.fps))

    regions = RegionsStore(args.regions)
    regions.load()

    hub = FrameHub()
    decoder = Decoder(args, hub)

    if args.selftest:
        decoder.start()
        try:
            return run_selftest(args, decoder, hub, need=args.selftest_frames)
        finally:
            decoder.stop()
            time.sleep(0.3)

    try:
        srv = VisionServer((args.host, args.port), Handler)
    except OSError as exc:
        print("端口 %d 绑定失败：%s" % (args.port, exc), file=sys.stderr)
        print("换个端口：--port 8098；或先 ta2_vision_ops.sh stop", file=sys.stderr)
        return 2
    srv.hub = hub
    srv.decoder = decoder
    srv.regions = regions
    srv.args = args
    srv.placeholder = make_placeholder(args.width, frame_size(args.width)[1])

    def _on_signal(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    w, h = frame_size(args.width)
    print("视觉查看器: http://0.0.0.0:%d  （RTP :%d → %s → %dx%d@%dfps q%d）" %
          (args.port, args.rtp_port, decoder.backend, w, h, args.fps, args.quality),
          flush=True)
    print("裁剪表: %s  日志目录: %s" % (regions.path, args.log_dir), flush=True)

    decoder.start()
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n收到退出信号，正在收尾…", flush=True)
    finally:
        srv.stop_event.set()
        decoder.stop()
        try:
            srv.server_close()
        except Exception:
            pass
        print("已停止", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
