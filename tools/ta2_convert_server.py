#!/usr/bin/env python3
"""
ta2_convert_server.py — 转化控制台（mcap → LeRobot v2.1）

职责
    * 看：有哪些原始记录、哪些已转化、转出多少条/多少帧/多大、日志在哪、输出目录在哪
    * 做：勾选原始记录，填参数，起一次 docker 转化，实时看日志
    * 串：每个数据集给一个「在筛选台打开」的直链，回到下载台也有入口

端口 8095。纯标准库；转化本身用官方镜像 rosbag2lerobot:0.2.0-20261030。

用法
    python3 ta2_convert_server.py --port 8095
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from ta2_groups import GroupStore
except ImportError:  # 万一没跟着部署，降级为不可用
    GroupStore = None

DEFAULT_RECORDS_ROOT = "/data/ta2"
DEFAULT_CONVERT_ROOT = "/data/ta2/converted"
DEFAULT_REVIEW_ROOT = "/data/ta2/review"
DEFAULT_IMAGE = "rosbag2lerobot:0.2.0-20261030"
DEFAULT_PROJECT = "/data/ta2/software/convert/rosbag_to_dataset_TA2"
DEFAULT_PATCHED = "/data/ta2/converted/records_20261114_sandbag/scripts/convert_rosbag_to_lerobot.py"
DEFAULT_IFACE = "enp9s0"

CONVERT_JOBS = {}
JOBS_LOCK = threading.Lock()
CACHE = {"t": 0.0, "inputs": [], "datasets": []}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def human(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or u == "TiB":
            return f"{int(n)} B" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0


def dir_size(path, limit_mtime=None):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                st = os.stat(os.path.join(root, f))
                if limit_mtime and st.st_mtime < limit_mtime:
                    pass
                total += st.st_size
            except OSError:
                continue
    return total


# ================================================================ 扫盘

def list_inputs(records_root):
    """扫 <records_root>/records_*/bags/<rid>/<rid>/metadata.yaml"""
    out = []
    root = Path(records_root)
    for meta in sorted(root.glob("records_*/bags/*/*/metadata.yaml")):
        rec_dir = meta.parent
        rid = rec_dir.name
        batch = rec_dir.parent.parent.parent.name
        mcaps = list(rec_dir.glob("*.mcap")) + list(rec_dir.glob("*.db3"))
        size = sum(p.stat().st_size for p in mcaps if p.exists())
        out.append({
            "rid": rid,
            "batch": batch,
            "path": str(rec_dir),
            "bytes": size,
            "bags": len(mcaps),
        })
    return out


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_datasets(convert_root, review_root):
    out = []
    root = Path(convert_root)
    for info_path in sorted(root.rglob("meta/info.json")):
        ds = info_path.parent.parent
        rel = ds.relative_to(root)
        parts = rel.parts
        if any(p.lower() in {"reports", "backup", "old", "archive", "superseded", "tmp"} or p.startswith("_")
               for p in parts[:-1]):
            continue
        info = read_json(info_path) or {}
        marker = read_json(ds / "converted_from.json") or {}
        mtime = info_path.stat().st_mtime
        out.append({
            "name": ds.name,
            "path": str(ds),
            "key": str(rel).replace(os.sep, "__"),
            "episodes": int(info.get("total_episodes") or 0),
            "frames": int(info.get("total_frames") or 0),
            "fps": info.get("fps"),
            "cameras": len([k for k in (info.get("features") or {}) if "images" in k]),
            "created_at": datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds"),
            "marker": marker,
            "log": marker.get("log"),
            "task": marker.get("task") or _first_task(ds),
        })
    return out


def _first_task(ds):
    p = Path(ds) / "meta" / "tasks.jsonl"
    try:
        with open(p, "r", encoding="utf-8") as fh:
            line = fh.readline().strip()
        return json.loads(line).get("task") if line else None
    except (OSError, json.JSONDecodeError):
        return None


def scan(records_root, convert_root, review_root, force=False):
    now = time.time()
    if not force and now - CACHE["t"] < 6:
        return CACHE["inputs"], CACHE["datasets"]
    inputs = list_inputs(records_root)
    datasets = list_datasets(convert_root, review_root)

    # 已转化：marker 里记录的 rid
    converted = {}
    for d in datasets:
        for rid in (d["marker"].get("records") or []):
            converted.setdefault(rid, []).append(d["name"])
    for r in inputs:
        r["converted"] = converted.get(r["rid"], [])
        r["converted_any"] = bool(r["converted"])

    CACHE.update({"t": now, "inputs": inputs, "datasets": datasets})
    return inputs, datasets


# ================================================================ 任务

def job_dir(review_root, job_id):
    d = Path(review_root) / "convert_jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_input_dir(dest, rids, inputs):
    """只把选中的记录硬链接到一个临时输入目录，避免 --multibag 把整批都转了。

    用硬链接（同分区几乎零成本）而不是软链：容器里软链会指向不存在的宿主路径。
    跨分区时回退为复制。
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    by_rid = {r["rid"]: r for r in inputs}
    linked = 0
    for rid in rids:
        rec = by_rid.get(rid)
        if not rec:
            raise RuntimeError(f"记录不存在: {rid}")
        src = Path(rec["path"])
        dst = dest / rid / rid
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            target = dst / f.name
            if target.exists():
                continue
            try:
                os.link(f, target)
                linked += 1
            except OSError:
                shutil.copy2(f, target)
    return dest, linked


def build_command(cfg, rids, params, input_dir):
    output = params.get("dataset_name") or f"dataset_{datetime.now():%Y%m%d_%H%M%S}"
    cmd = [
        "docker", "run", "--rm",
        "--user", "0:0", "-e", "HOME=/tmp",
        "--device", "/dev/dri",
        "--group-add", str(os.stat("/dev/dri/renderD128").st_gid),
        "-e", "LIBVA_DRIVER_NAME=iHD",
        "-e", "R2L_VIDEO_BACKEND=vaapi",
        "--shm-size=8g",
        "-w", "/workspace/rosbag_to_dataset_TA2",
        "-v", f"{cfg['project']}:/workspace/rosbag_to_dataset_TA2",
        "-v", f"{cfg['patched']}:/workspace/rosbag_to_dataset_TA2/scripts/convert_rosbag_to_lerobot.py",
        "-v", f"{input_dir}:/data/rosbags:ro",
        "-v", f"{cfg['convert_root']}/{params.get('batch') or 'converted'}:/workspace/hf_lerobot",
        cfg["image"],
        "python", "scripts/convert_rosbag_to_lerobot.py",
        "--multibag",
        "--input_directory", "/data/rosbags",
        "--output", output,
        "--fps", str(params.get("fps") or 45),
        "--head_resolution", params.get("head_res") or "960x480",
        "--wrist_resolution", params.get("wrist_res") or "1280x400",
        "--task", params.get("task") or "抓取红色沙包",
        "--verbose",
    ]
    return cmd, output


def start_job(cfg, rids, params):
    with JOBS_LOCK:
        running = [j for j in CONVERT_JOBS.values() if j["status"] == "running"]
        if running:
            raise RuntimeError(f"已有转化任务在跑：{running[0]['id']}")

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        jd = job_dir(cfg["review_root"], job_id)

        input_dir, linked = prepare_input_dir(jd / "input", rids, cfg["inputs"]())
        cmd, output = build_command(cfg, rids, params, input_dir)

        out_ds = Path(cfg["convert_root"]) / (params.get("batch") or "converted") / output
        if out_ds.exists():
            raise RuntimeError(f"输出已存在（LeRobot 不允许覆盖）：{out_ds}")

        log_path = jd / "console.log"
        fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)

        job = {
            "id": job_id,
            "status": "running",
            "started_at": now_iso(),
            "records": list(rids),
            "group": params.get("group"),
            "params": params,
            "output": output,
            "output_dir": str(out_ds),
            "input_dir": str(input_dir),
            "linked_files": linked,
            "log": str(log_path),
            "cmd": " ".join(cmd),
            "proc": proc,
            "_fh": fh,
            "_saved": 0,
            "_current": None,
            "_store": cfg.get("groups"),
        }
        CONVERT_JOBS[job_id] = job
        safe = {k: v for k, v in job.items() if not k.startswith("_") and k != "proc"}
        (jd / "job.json").write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
        return job


def poll_job(job):
    if job["status"] != "running":
        return job
    proc = job["proc"]
    # 解析日志进度
    try:
        text = Path(job["log"]).read_text(encoding="utf-8", errors="replace")
        saved = len(re.findall(r"Saved episode \d+", text))
        cur = None
        m = re.findall(r"Processing (rec_\S+)", text)
        if m:
            cur = m[-1]
        job["_saved"] = saved
        job["_current"] = cur
    except OSError:
        pass

    if proc.poll() is not None:
        try:
            job["_fh"].close()
        except OSError:
            pass
        job["status"] = "done" if proc.returncode == 0 else "failed"
        job["returncode"] = proc.returncode
        job["finished_at"] = now_iso()
        # 写 converted_from.json，供筛选台/下次扫描识别来源
        if job["status"] == "done":
            ds = Path(job["output_dir"])
            if (ds / "meta" / "info.json").exists():
                marker = {
                    "records": job["records"],
                    "group": job.get("group"),
                    "params": job["params"],
                    "log": job["log"],
                    "task": job["params"].get("task"),
                    "created_at": now_iso(),
                    "cmd": job["cmd"],
                }
                try:
                    (ds / "converted_from.json").write_text(
                        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
                except OSError:
                    pass
                # 组上回填数据集名
                gid = job.get("group")
                store = job.get("_store")
                if gid and store is not None:
                    try:
                        store.link_dataset(gid, ds.name)
                    except (KeyError, OSError):
                        pass
        CACHE["t"] = 0.0
    return job


def job_public(job):
    return {k: v for k, v in job.items() if not k.startswith("_") and k not in ("proc", "log")} | {
        "log": job["log"],
        "saved_episodes": job.get("_saved"),
        "current_record": job.get("_current"),
    }


def log_tail(job, n=120):
    try:
        lines = Path(job["log"]).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


# ================================================================ HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "ta2convert/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        cfg = self.server.cfg

        for j in list(CONVERT_JOBS.values()):
            poll_job(j)

        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")

        if path == "/api/groups":
            store = cfg.get("groups")
            if store is None:
                return self._send(500, {"error": "训练组模块未部署"})
            groups = store.list()
            assigned = store.assigned_map()
            return self._send(200, {"groups": groups, "assigned": assigned})

        if path == "/api/overview":
            force = (q.get("force") or ["0"])[0] == "1"
            inputs, datasets = scan(cfg["records_root"], cfg["convert_root"],
                                    cfg["review_root"], force=force)
            total_bytes = sum(d["bytes"] for d in inputs)
            conv = sum(1 for d in inputs if d["converted_any"])
            return self._send(200, {
                "inputs": inputs,
                "datasets": datasets,
                "summary": {
                    "records_total": len(inputs),
                    "records_converted": conv,
                    "records_bytes": total_bytes,
                    "datasets_total": len(datasets),
                    "episodes_total": sum(d["episodes"] for d in datasets),
                    "frames_total": sum(d["frames"] for d in datasets),
                    "link_mbps": read_link_mbps(cfg["iface"]),
                    "disk_free": shutil.disk_usage(cfg["convert_root"]).free,
                    "jobs_running": len([j for j in CONVERT_JOBS.values() if j["status"] == "running"]),
                },
            })

        if path == "/api/jobs":
            return self._send(200, {"jobs": [job_public(j) for j in CONVERT_JOBS.values()]})

        if path.startswith("/api/jobs/"):
            jid = path.rsplit("/", 1)[-1]
            job = CONVERT_JOBS.get(jid)
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._send(200, {**job_public(job), "log_tail": log_tail(job)})

        if path.startswith("/api/log/"):
            jid = path.rsplit("/", 1)[-1]
            job = CONVERT_JOBS.get(jid)
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._send(200, {"log": log_tail(job, 400)})

        if path == "/api/dataset_log":
            ds_path = (q.get("path") or [""])[0]
            info = read_json(Path(ds_path) / "converted_from.json") or {}
            log_path = info.get("log")
            if not log_path or not Path(log_path).exists():
                return self._send(404, {"error": "该数据集没有登记转化日志"})
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as exc:
                return self._send(500, {"error": str(exc)})
            return self._send(200, {"log": "\n".join(text[-400:]), "path": log_path})

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        cfg = self.server.cfg
        body = self._body()

        if path == "/api/groups":
            store = cfg.get("groups")
            if store is None:
                return self._send(500, {"error": "训练组模块未部署"})
            action = body.get("action") or "create"
            try:
                if action == "create":
                    g = store.create(body.get("name") or "未命名组",
                                     scene=body.get("scene"), light=body.get("light"),
                                     objects=body.get("objects"),
                                     operator=body.get("operator"),
                                     task=body.get("task"), note=body.get("note"))
                elif action == "update":
                    g = store.update(body["id"], **{k: body.get(k) for k in
                                                    ("name", "scene", "light", "objects",
                                                     "operator", "task", "note")})
                elif action == "assign":
                    g = store.assign(body["id"], body.get("records") or [],
                                     force=bool(body.get("force")))
                elif action == "unassign":
                    store.unassign(body.get("records") or [])
                    g = {"unassigned": body.get("records")}
                elif action == "delete":
                    store.delete(body["id"])
                    g = {"deleted": body["id"]}
                else:
                    return self._send(400, {"error": f"未知 action: {action}"})
            except (KeyError, ValueError) as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(200, {"ok": True, "group": g})

        if path == "/api/convert":
            rids = [str(x) for x in (body.get("records") or [])]
            if not rids:
                return self._send(400, {"error": "没有选择原始记录"})
            params = body.get("params") or {}
            try:
                job = start_job(cfg, rids, params)
            except (RuntimeError, OSError) as exc:
                return self._send(409, {"error": str(exc)})
            return self._send(200, job_public(job))

        return self._send(404, {"error": "not found"})


def read_link_mbps(iface):
    try:
        with open(f"/sys/class/net/{iface}/speed", "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


# ================================================================ 页面

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>转化控制台 · TA2</title>
<style>
 :root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;
       --ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#4c8dff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
 header{position:sticky;top:0;z-index:20;background:#12151b;border-bottom:1px solid var(--line);padding:10px 18px;
        display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600}
 .pill{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;color:var(--dim)}
 .pill b{color:var(--fg)}
 a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
 button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:6px 12px;font-size:13px;cursor:pointer}
 button:disabled{opacity:.45;cursor:not-allowed}
 button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
 button.tiny{padding:3px 9px;font-size:12px}
 input,select{background:var(--card);border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:5px 8px;font-size:13px;font-family:inherit}
 main{padding:14px 18px 80px}
 .tabs{display:flex;gap:6px;margin-bottom:14px}
 .tabs button{background:transparent;border:1px solid var(--line);color:var(--dim);border-radius:6px}
 .tabs button.on{background:#22304a;color:#fff;border-color:#3b5bdb}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
 th{color:var(--dim);font-weight:500;position:sticky;top:0;background:var(--bg)}
 tr:hover td{background:#1b1f28}
 .ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
 .muted{color:var(--dim);font-size:12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
 .kv{display:flex;gap:14px;flex-wrap:wrap;color:var(--dim);font-size:12px;margin-top:6px}
 .kv b{color:var(--fg)}
 pre{background:#0b0d11;border:1px solid var(--line);border-radius:6px;padding:10px;font:12px/1.5 ui-monospace,Consolas,monospace;
     color:#b9c0cf;max-height:420px;overflow:auto;white-space:pre-wrap;margin:8px 0 0}
 .form{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;align-items:end}
 .form label{display:block;font-size:11px;color:var(--dim);margin-bottom:3px}
 .form input,.form select{width:100%}
 .bar{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
 .nav{display:flex;gap:8px;flex-wrap:wrap}
 .nav a{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:12px;color:var(--fg)}
 .nav a:hover{border-color:var(--accent)}
 #toast{position:fixed;right:18px;bottom:18px;background:#1f2937;border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:13px;display:none;z-index:60}
</style>
</head>
<body>
<header>
  <h1>转化控制台</h1>
  <span class="pill">原始记录 <b id="s-records">-</b></span>
  <span class="pill">已转化 <b id="s-conv">-</b></span>
  <span class="pill">数据集 <b id="s-ds">-</b></span>
  <span class="pill">Episode <b id="s-ep">-</b></span>
  <span class="pill">链路 <b id="s-link">-</b></span>
  <span class="pill">磁盘 <b id="s-disk">-</b></span>
  <span style="flex:1"></span>
  <span class="nav">
    <a href="http://192.168.31.228:8090" target="_blank">下载台</a>
    <a href="http://192.168.31.228:8093" target="_blank">筛选台</a>
    <a href="#" id="refresh">刷新</a>
  </span>
</header>

<main>
  <div class="tabs">
    <button data-t="inputs" class="on">待转化</button>
    <button data-t="done">已转化</button>
    <button data-t="groups">训练组</button>
    <button data-t="jobs">任务 / 日志</button>
  </div>

  <!-- 待转化 -->
  <section id="tab-inputs">
    <div class="bar">
      <span class="muted">勾选要转化的原始记录（只会转你勾的那些）</span>
      <span style="flex:1"></span>
      <label class="muted"><input type="checkbox" id="only-new" checked> 只看未转化</label>
      <label class="muted"><input type="checkbox" id="only-nogroup"> 只看未分组</label>
    </div>
    <table>
      <thead><tr>
        <th style="width:30px"><input type="checkbox" id="chk-all"></th>
        <th>记录</th><th>批次</th><th>大小</th><th>训练组</th><th>状态</th>
      </tr></thead>
      <tbody id="rows"><tr><td colspan="6" class="muted">加载中…</td></tr></tbody>
    </table>

    <div class="card" style="margin-top:14px">
      <div class="bar" style="margin-bottom:6px">
        <b>归入训练组</b>
        <select id="assign-group" style="min-width:220px"></select>
        <button class="ghost tiny" id="btn-assign">把选中记录归入该组</button>
        <span class="muted" id="assign-info"></span>
      </div>
      <div class="muted">一条记录只能属于一个组；已在别的组时会被移动（组里会提示）。</div>
    </div>

    <div class="card" style="margin-top:14px">
      <div style="font-weight:600;margin-bottom:10px">转化参数 <span class="muted">（默认值按官方文档推荐）</span></div>
      <div class="form">
        <div><label>输出批次目录</label><input id="p-batch" value="records_20261115"></div>
        <div><label>数据集名</label><input id="p-name" placeholder="grasp_red_sandbag"></div>
        <div><label>fps</label><input id="p-fps" value="45"></div>
        <div><label>头部分辨率</label><input id="p-head" value="960x480"></div>
        <div><label>腕部分辨率</label><input id="p-wrist" value="1280x400"></div>
        <div><label>任务 prompt</label><input id="p-task" value="抓取红色沙包"></div>
      </div>
      <div class="bar" style="margin-top:12px;margin-bottom:0">
        <button id="btn-convert">开始转化</button>
        <span class="muted" id="sel-info">未选择记录</span>
      </div>
    </div>
  </section>

  <!-- 已转化 -->
  <section id="tab-done" style="display:none">
    <div class="grid" id="ds-grid"></div>
  </section>

  <!-- 训练组 -->
  <section id="tab-groups" style="display:none">
    <div class="card">
      <div style="font-weight:600;margin-bottom:10px">新建训练组</div>
      <div class="form">
        <div><label>组名</label><input id="g-name" placeholder="桌面A-白光-红沙包"></div>
        <div><label>场景</label><input id="g-scene" placeholder="桌面A"></div>
        <div><label>光照</label><input id="g-light" placeholder="白光"></div>
        <div><label>物体</label><input id="g-objects" placeholder="红沙包,蓝垫"></div>
        <div><label>采集人</label><input id="g-operator" placeholder="侯兴家"></div>
        <div><label>任务 prompt</label><input id="g-task" placeholder="抓取红色沙包"></div>
      </div>
      <div class="bar" style="margin-top:12px;margin-bottom:0">
        <button id="btn-create-group">创建</button>
      </div>
    </div>
    <div class="grid" id="group-grid"></div>
  </section>

  <!-- 任务 -->
  <section id="tab-jobs" style="display:none">
    <div id="jobs-list"></div>
  </section>
</main>

<div id="toast"></div>
<script>
const $ = s => document.querySelector(s);
let OV = null, CURTAB = 'inputs', JOBS = [], POLL = null, GROUPS = [], ASSIGNED = {};

function toast(m, ms=2600){ const t=$('#toast'); t.textContent=m; t.style.display='block';
  clearTimeout(t._h); t._h=setTimeout(()=>t.style.display='none', ms); }
function human(n){ if(n==null) return '-'; const u=['B','KiB','MiB','GiB','TiB']; let i=0; n=+n;
  while(n>=1024 && i<u.length-1){n/=1024;i++;} return (i?n.toFixed(1):n)+' '+u[i]; }
async function api(p, body){
  const o = body? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{};
  const r = await fetch(p,o); const d = await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(d.error||r.status); return d;
}

async function loadOverview(force){
  OV = await api('/api/overview' + (force?'?force=1':''));
  const s = OV.summary;
  $('#s-records').textContent = s.records_total;
  $('#s-conv').textContent = s.records_converted;
  $('#s-ds').textContent = s.datasets_total;
  $('#s-ep').textContent = s.episodes_total;
  $('#s-link').textContent = s.link_mbps? s.link_mbps+' Mbps':'未知';
  $('#s-disk').textContent = human(s.disk_free);
  await loadGroups();
  renderInputs(); renderDone(); renderGroups();
}

async function loadGroups(){
  try{
    const d = await api('/api/groups');
    GROUPS = d.groups || []; ASSIGNED = d.assigned || {};
  }catch(e){ GROUPS = []; ASSIGNED = {}; }
  const sel = $('#assign-group');
  if(sel){
    sel.innerHTML = '<option value="">— 选择组 —</option>' +
      GROUPS.map(g => `<option value="${g.id}">${g.name} (${(g.records||[]).length} 条)</option>`).join('');
  }
}

function renderInputs(){
  const onlyNew = $('#only-new').checked;
  const onlyNoGroup = $('#only-nogroup').checked;
  const rows = OV.inputs.filter(r => (!onlyNew || !r.converted_any) &&
                                     (!onlyNoGroup || !ASSIGNED[r.rid]));
  const tb = $('#rows');
  if(!rows.length){ tb.innerHTML = '<tr><td colspan="6" class="muted">没有符合条件的记录</td></tr>'; return; }
  const gname = gid => (GROUPS.find(g => g.id===gid) || {}).name || gid;
  tb.innerHTML = rows.map(r => {
    const gid = ASSIGNED[r.rid];
    return `
    <tr>
      <td><input type="checkbox" class="pick" value="${r.rid}"></td>
      <td>${r.rid}</td>
      <td class="muted">${r.batch}</td>
      <td>${human(r.bytes)}</td>
      <td>${gid? '<span class="ok">'+gname(gid)+'</span>' : '<span class="muted">未分组</span>'}</td>
      <td>${r.converted_any? '<span class="ok">已转化</span> <span class="muted">'+r.converted.join(', ')+'</span>' : '<span class="warn">未转化</span>'}</td>
    </tr>`;
  }).join('');
  tb.querySelectorAll('.pick').forEach(c => c.onchange = syncSel);
  syncSel();
}

function renderGroups(){
  const g = $('#group-grid');
  if(!GROUPS.length){ g.innerHTML = '<div class="muted">还没有训练组</div>'; return; }
  g.innerHTML = GROUPS.map(x => `
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
        <b>${x.name}</b>
        <span class="muted">${(x.records||[]).length} 条记录</span>
      </div>
      <div class="kv">
        ${x.scene? '<span>场景 <b>'+x.scene+'</b></span>':''}
        ${x.light? '<span>光照 <b>'+x.light+'</b></span>':''}
        ${x.operator? '<span>采集人 <b>'+x.operator+'</b></span>':''}
        ${x.task? '<span>任务 <b>'+x.task+'</b></span>':''}
      </div>
      ${(x.objects||[]).length? '<div class="kv"><span>物体 <b>'+x.objects.join('、')+'</b></span></div>':''}
      <div class="kv"><span>数据集 <b>${(x.datasets||[]).join(', ')||'—'}</b></span></div>
      <div class="kv"><span class="muted">${x.id}</span></div>
    </div>`).join('');
}

$('#btn-create-group').onclick = async () => {
  const name = $('#g-name').value.trim();
  if(!name){ toast('组名必填'); return; }
  try{
    await api('/api/groups', {action:'create', name,
      scene:$('#g-scene').value.trim(), light:$('#g-light').value.trim(),
      objects:$('#g-objects').value.split(',').map(s=>s.trim()).filter(Boolean),
      operator:$('#g-operator').value.trim(), task:$('#g-task').value.trim()});
    toast('组已创建'); $('#g-name').value='';
    await loadGroups(); renderGroups(); renderInputs();
  }catch(e){ toast('创建失败：'+e.message, 4000); }
};

$('#btn-assign').onclick = async () => {
  const gid = $('#assign-group').value;
  const recs = [...document.querySelectorAll('.pick:checked')].map(c=>c.value);
  if(!gid){ toast('先选一个组'); return; }
  if(!recs.length){ toast('先勾选记录'); return; }
  try{
    await api('/api/groups', {action:'assign', id:gid, records:recs, force:true});
    toast(`已把 ${recs.length} 条归入组`);
    await loadGroups(); renderInputs(); renderGroups();
  }catch(e){ toast('归组失败：'+e.message, 4000); }
};

function syncSel(){
  const n = document.querySelectorAll('.pick:checked').length;
  $('#sel-info').textContent = n? `已选 ${n} 条` : '未选择记录';
  $('#btn-convert').disabled = n===0;
}

function renderDone(){
  const g = $('#ds-grid');
  if(!OV.datasets.length){ g.innerHTML = '<div class="muted">还没有转化产物</div>'; return; }
  g.innerHTML = OV.datasets.map(d => {
    const recs = (d.marker.records||[]);
    return `<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
        <b>${d.name}</b>
        <span class="muted">${d.created_at.slice(0,19).replace('T',' ')}</span>
      </div>
      <div class="kv">
        <span>Episode <b>${d.episodes}</b></span>
        <span>帧 <b>${d.frames}</b></span>
        <span>fps <b>${d.fps}</b></span>
        <span>相机 <b>${d.cameras}</b></span>
      </div>
      <div class="kv"><span>task <b>${d.task||'—'}</b></span></div>
      <div class="kv" style="word-break:break-all"><span class="muted">${d.path}</span></div>
      <div class="kv"><span>来源记录 <b>${recs.length? recs.length+' 条':'未知（非本台转化）'}</b></span></div>
      <div class="bar" style="margin-top:10px;margin-bottom:0">
        <a class="ghost" href="http://192.168.31.228:8093/?ds=${encodeURIComponent(d.key)}" target="_blank"
           style="border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:12px">在筛选台打开</a>
        <button class="ghost tiny" data-log="${d.path}">查看日志</button>
      </div>
      <pre id="log-${btoa(d.name).replace(/=/g,'')}" style="display:none"></pre>
    </div>`;
  }).join('');
  g.querySelectorAll('[data-log]').forEach(b => b.onclick = async () => {
    const pre = b.closest('.card').querySelector('pre');
    if(pre.style.display==='block'){ pre.style.display='none'; return; }
    try{ const d = await api('/api/dataset_log?path='+encodeURIComponent(b.dataset.log));
      pre.textContent = d.log || '(空)'; pre.style.display='block';
    }catch(e){ pre.textContent = '读取失败：'+e.message; pre.style.display='block'; }
  });
}

async function loadJobs(){
  const d = await api('/api/jobs');
  JOBS = d.jobs || [];
  const box = $('#jobs-list');
  if(!JOBS.length){ box.innerHTML = '<div class="muted">还没有任务</div>'; }
  else {
    box.innerHTML = JOBS.slice().reverse().map(j => `
      <div class="card" data-job="${j.id}">
        <div style="display:flex;justify-content:space-between;gap:10px;align-items:center">
          <b>${j.id}</b>
          <span class="${j.status==='done'?'ok':(j.status==='failed'?'bad':'warn')}">${j.status}</span>
        </div>
        <div class="kv">
          <span>数据集 <b>${j.output}</b></span>
          <span>已存 <b>${j.saved_episodes||0}</b> 条</span>
          <span>当前 <b>${j.current_record||'—'}</b></span>
        </div>
        <div class="kv"><span class="muted">${j.output_dir}</span></div>
        <pre></pre>
      </div>`).join('');
  }
  // 有运行中的任务就拉日志
  const running = JOBS.filter(j => j.status==='running');
  await Promise.all(JOBS.map(async j => {
    if(j.status!=='running') return;
    try{
      const d = await api('/api/jobs/'+j.id);
      const el = document.querySelector(`[data-job="${j.id}"] pre`);
      if(el) el.textContent = d.log_tail || '';
    }catch(e){}
  }));
  if(running.length){ clearTimeout(POLL); POLL = setTimeout(()=>{ loadJobs().catch(()=>{}); }, 2000); }
}

$('#refresh').onclick = async e => { e.preventDefault(); await loadOverview(true); toast('已刷新'); };
$('#only-new').onchange = renderInputs;
$('#only-nogroup').onchange = renderInputs;
$('#chk-all').onchange = e => { document.querySelectorAll('.pick').forEach(c=>c.checked=e.target.checked); syncSel(); };

$('#btn-convert').onclick = async () => {
  const recs = [...document.querySelectorAll('.pick:checked')].map(c=>c.value);
  if(!recs.length) return;
  const params = {
    batch: $('#p-batch').value.trim(),
    dataset_name: $('#p-name').value.trim() || ('dataset_'+Date.now().toString().slice(-6)),
    fps: parseFloat($('#p-fps').value)||45,
    head_res: $('#p-head').value.trim(),
    wrist_res: $('#p-wrist').value.trim(),
    task: $('#p-task').value.trim(),
  };
  // 如果选中的记录都在同一个组，自动带上组 id
  const gids = [...new Set(recs.map(r => ASSIGNED[r]).filter(Boolean))];
  if(gids.length === 1) params.group = gids[0];
  if(!confirm(`用 ${recs.length} 条记录转化？\\n输出：${params.batch}/${params.dataset_name}\\nfps=${params.fps} head=${params.head_res} wrist=${params.wrist_res}\\ntask=${params.task}`)) return;
  try{
    const job = await api('/api/convert', {records:recs, params});
    toast('任务 '+job.id+' 已启动');
    switchTab('jobs'); loadJobs();
  }catch(e){ toast('启动失败：'+e.message, 5000); }
};

document.querySelector('.tabs').onclick = e => {
  const b = e.target.closest('button'); if(!b) return; switchTab(b.dataset.t);
};
function switchTab(t){
  CURTAB = t;
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on', x.dataset.t===t));
  $('#tab-inputs').style.display = t==='inputs'?'block':'none';
  $('#tab-done').style.display   = t==='done'  ?'block':'none';
  $('#tab-groups').style.display = t==='groups'?'block':'none';
  $('#tab-jobs').style.display   = t==='jobs'  ?'block':'none';
  if(t==='jobs') loadJobs().catch(()=>{});
  if(t==='done') renderDone();
  if(t==='groups') renderGroups();
}

loadOverview(true).then(()=>{ $('#p-name').value = 'grasp_' + new Date().toISOString().slice(0,10).replace(/-/g,''); }).catch(e => {
  $('#rows').innerHTML = '<tr><td colspan="5" class="muted">加载失败：'+e.message+'</td></tr>';
});
setInterval(()=>loadOverview(false).catch(()=>{}), 15000);
</script>
</body>
</html>
"""


# ================================================================ 启动

def main(argv=None):
    ap = argparse.ArgumentParser(description="TA2 转化控制台")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--records-root", default=DEFAULT_RECORDS_ROOT)
    ap.add_argument("--convert-root", default=DEFAULT_CONVERT_ROOT)
    ap.add_argument("--review-root", default=DEFAULT_REVIEW_ROOT)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--patched", default=DEFAULT_PATCHED)
    ap.add_argument("--iface", default=DEFAULT_IFACE)
    args = ap.parse_args(argv)

    def inputs():
        return scan(args.records_root, args.convert_root, args.review_root)[0]

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.cfg = {
        "records_root": args.records_root,
        "convert_root": args.convert_root,
        "review_root": args.review_root,
        "image": args.image,
        "project": args.project,
        "patched": args.patched,
        "iface": args.iface,
        "inputs": inputs,
        "groups": GroupStore(args.review_root) if GroupStore else None,
    }
    print(f"转化控制台: http://{args.host}:{args.port}  records={args.records_root} "
          f"convert={args.convert_root}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n退出", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
