#!/usr/bin/env python3
"""
ta2_download_web.py — TA2 数采记录下载控制台（纯标准库 http.server）

为什么要标准库：B70 上没有 venv 也能直接跑，不引入 Flask 依赖。

路由
  GET  /                  单页控制台（内联 HTML/CSS/JS，无 CDN 依赖）
  GET  /api/records       代理记录 API，附加"本地是否已下载"
  GET  /api/system        磁盘余量 / 链路速率 / 输出目录
  GET  /api/jobs          当前与历史任务
  POST /api/jobs          {"ids":[...]} 起一个下载任务
  GET  /api/jobs/<id>     任务当前 progress.json
  GET  /api/events/<id>   SSE 实时进度
  POST /api/speedtest     跑一次测速，返回报告

设计约束
  * 同一时间只允许一个下载任务（100M 链路上并发只会互相拖慢）
  * 所有下载逻辑复用 tools/ta2_download.py，本文件只做编排与展示
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

HERE = Path(__file__).resolve().parent
CLI = HERE / "ta2_download.py"

DEFAULT_API = "http://192.168.31.218:5173/api/v1/records"
DEFAULT_ROOT = "/data/ta2"
DEFAULT_IFACE = "enp9s0"

JOBS = {}
JOBS_LOCK = threading.Lock()

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ------------------------------------------------------------------ 工具

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def human_bytes(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def api_json(url, timeout=20):
    with _OPENER.open(url, timeout=timeout) as fh:
        payload = json.load(fh)
    if payload.get("code") != 20000:
        raise RuntimeError(f"接口异常 {payload.get('code')} {payload.get('message')}")
    return payload["data"]


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            v = json.loads(value)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def read_link_mbps(iface):
    try:
        with open(f"/sys/class/net/{iface}/speed", "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def find_cli():
    if CLI.is_file():
        return CLI
    raise FileNotFoundError(f"找不到 ta2_download.py（应在 {CLI}）")


# ------------------------------------------------------------------ 任务

def new_job(ids, root, api, iface, keep_archive=False):
    refresh_jobs()
    with JOBS_LOCK:
        running = [j for j in JOBS.values() if j["status"] == "running"]
        if running:
            raise RuntimeError(f"已有任务在跑: {running[0]['id']}")
        started = datetime.now()
        job_id = f"{started:%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        job_dir = Path(root) / "web_jobs" / job_id
        out_dir = Path(root) / f"records_{started:%Y%m%d}"
        run_dir = job_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-u", str(find_cli()),
               "--root", str(root), "--api", api,
               "download", *ids,
               "--out", str(out_dir), "--run-dir", str(run_dir),
               "--iface", iface]
        if keep_archive:
            cmd.append("--keep-archive")

        with open(job_dir / "console.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        job = {
            "id": job_id,
            "ids": ids,
            "status": "running",
            "started_at": now_iso(),
            "cmd": " ".join(cmd),
            "dir": str(job_dir),
            "out_dir": str(out_dir),
            "run_dir": str(run_dir),
            "proc": proc,
        }
        JOBS[job_id] = job
        return job


def job_public(job):
    return {k: v for k, v in job.items() if k != "proc"}


def list_jobs():
    with JOBS_LOCK:
        return [job_public(j) for j in JOBS.values()]


def refresh_jobs():
    with JOBS_LOCK:
        for job in JOBS.values():
            proc = job.get("proc")
            if proc is not None and job["status"] == "running" and proc.poll() is not None:
                job["status"] = "done" if proc.returncode == 0 else "failed"
                job["returncode"] = proc.returncode
                job["finished_at"] = now_iso()


def read_progress(job):
    p = Path(job["run_dir"]) / "progress.json"
    try:
        with open(p, "r", encoding="utf-8") as fh:
            progress = json.load(fh)
    except (OSError, json.JSONDecodeError):
        progress = {"phase": "starting", "percent": 0.0, "message": "正在准备下载"}
    if progress.get("idx") and progress.get("n"):
        progress["percent"] = round(
            ((progress["idx"] - 1) * 100 + progress["percent"]) / progress["n"], 2
        )
    if job["status"] == "done":
        progress.update(phase="done", percent=100.0, message="下载并解压完成")
    elif job["status"] == "failed":
        progress.update(phase="failed", message="任务失败，原因见下方日志")
    return progress


def job_snapshot(job):
    with JOBS_LOCK:
        snapshot = job_public(job)
    log_tail = ""
    try:
        with (Path(snapshot["dir"]) / "console.log").open("rb") as fh:
            fh.seek(max(0, os.fstat(fh.fileno()).st_size - 4000))
            log_tail = fh.read(4000).decode("utf-8", errors="replace")
    except OSError:
        pass
    return {**snapshot, "progress": read_progress(snapshot), "log_tail": log_tail}


def downloaded_ids(root):
    local_ids = {p.name for p in root.glob("records_*/bags/*") if p.is_dir()}
    for summary in root.glob("web_jobs/*/out/transfer_results.json"):
        try:
            results = json.loads(summary.read_text(encoding="utf-8"))["results"]
        except (OSError, ValueError, KeyError):
            continue
        for result in results:
            rid = result.get("record_id")
            if result.get("ok") and rid and (summary.parent / "bags" / rid).is_dir():
                local_ids.add(rid)
    return local_ids


# ------------------------------------------------------------------ 页面

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TA2 数采下载控制台</title>
<style>
  :root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#4c8dff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:center;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;font-weight:600}
  .pill{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:3px 10px;color:var(--dim);font-size:12px}
  .pill b{color:var(--fg);font-weight:600}
  .nav a{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:3px 10px;font-size:12px;color:var(--fg);text-decoration:none;white-space:nowrap}
  .nav a:hover{border-color:var(--accent)}
  main{padding:16px 20px 60px}
  .bar{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
  button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:7px 14px;font-size:13px;cursor:pointer}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
  input,select{background:var(--card);border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:6px 8px;font-size:13px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
  th{color:var(--dim);font-weight:500;position:sticky;top:0;background:var(--bg)}
  tr:hover td{background:#1b1f28}
  .tag{font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid var(--line);color:var(--dim)}
  .q-pass{color:var(--ok)}.q-failed{color:var(--bad)}.q-unknown{color:var(--warn)}
  .done{color:var(--ok)}
  #progress{margin-bottom:16px;padding:14px;background:var(--card);border:1px solid var(--line);border-radius:8px;display:none}
  .track{height:12px;background:#0b0d11;border-radius:999px;overflow:hidden;margin:8px 0}
  .fill{height:100%;width:0;background:linear-gradient(90deg,#2f6fe0,#4c8dff);transition:width .3s}
  .kv{display:flex;gap:18px;flex-wrap:wrap;color:var(--dim);font-size:12px}
  .kv b{color:var(--fg)}
  #log{background:#0b0d11;border:1px solid var(--line);border-radius:6px;padding:10px;margin-top:10px;font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--dim);max-height:150px;overflow:auto;white-space:pre-wrap;display:none}
  .muted{color:var(--dim);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>TA2 数采下载控制台</h1>
  <span class="nav"><a href="http://192.168.31.228:8095" target="_blank">转化台</a>
    <a href="http://192.168.31.228:8093" target="_blank">筛选台</a></span>
  <span class="pill">磁盘可用 <b id="sys-disk">-</b></span>
  <span class="pill">链路 <b id="sys-link">-</b></span>
  <span class="pill">B70 下载目录 <b id="sys-out">-</b></span>
</header>
<main>
  <div class="bar">
    <select id="f-quality">
      <option value="">全部质量</option>
      <option value="pass">pass</option>
      <option value="failed">failed</option>
      <option value="unknown">unknown</option>
    </select>
    <input id="f-since" placeholder="起始日期 2026-11-15" size="16">
    <button class="ghost" id="btn-refresh">刷新列表</button>
    <button id="btn-download" disabled>下载选中 (0)</button>
    <button class="ghost" id="btn-speed">测速</button>
    <span class="muted" id="hint"></span>
  </div>

  <div id="progress">
    <div class="kv">
      <span>阶段 <b id="p-phase">-</b></span>
      <span>总进度 <b id="p-pct">0%</b></span>
      <span>当前条 <b id="p-rid">-</b></span>
      <span>速度 <b id="p-speed">-</b></span>
      <span>已传 <b id="p-bytes">-</b></span>
      <span>ETA <b id="p-eta">-</b></span>
    </div>
    <div class="track"><div class="fill" id="p-fill"></div></div>
    <div class="muted" id="p-msg"></div>
    <div class="muted" id="p-out"></div>
    <div id="log"></div>
  </div>

  <table>
    <thead><tr>
      <th style="width:28px"><input type="checkbox" id="chk-all"></th>
      <th>record_id</th><th>时间</th><th>时长</th><th>大小</th><th>质量</th><th>标签</th><th>本地</th>
    </tr></thead>
    <tbody id="rows"><tr><td colspan="8" class="muted">加载中…</td></tr></tbody>
  </table>
</main>
<script>
const $ = s => document.querySelector(s);
let currentJob = null, submitting = false;
const phaseLabels = {starting:'准备', idle:'准备', probe:'查询归档', packing:'打包',
  transfer:'传输', verify:'校验', extract:'解压', done:'完成', failed:'失败'};

async function getJSON(url, options){
  const response = await fetch(url, {...options, signal: AbortSignal.timeout(25000)});
  const data = await response.json();
  if(!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function escapeHTML(value){
  const entities = {'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'};
  return String(value).replace(/[&<>"']/g, c=>entities[c]);
}

function fmt(n){ if(n==null) return '-'; const u=['B','KiB','MiB','GiB','TiB']; let i=0; n=+n;
  while(n>=1024 && i<u.length-1){n/=1024;i++;} return (i?n.toFixed(1):n)+' '+u[i]; }
function fmtSec(s){ if(s==null) return '-'; s=Math.round(s);
  return s<60? s+'s' : (s<3600? Math.floor(s/60)+'m'+(s%60)+'s' : Math.floor(s/3600)+'h'); }

async function loadSystem(){
  try{
    const d = await (await fetch('/api/system')).json();
    $('#sys-disk').textContent = fmt(d.disk_free);
    $('#sys-link').textContent = d.link_mbps? d.link_mbps+' Mbps' : '未知';
    $('#sys-out').textContent = d.out_dir;
  }catch(e){}
}

async function loadRecords(){
  const q = new URLSearchParams();
  if($('#f-quality').value) q.set('quality', $('#f-quality').value);
  if($('#f-since').value) q.set('since', $('#f-since').value);
  const tbody = $('#rows');
  tbody.innerHTML = '<tr><td colspan="8" class="muted">加载中…</td></tr>';
  $('#chk-all').checked = false;
  syncCount();
  try{
    const d = await getJSON('/api/records?'+q.toString());
    const rows = d.records || [];
    if(!rows.length){ tbody.innerHTML = '<tr><td colspan="8" class="muted">没有匹配记录</td></tr>'; return; }
    tbody.innerHTML = rows.map(r=>{
      const q = r.quality_status || '';
      const cls = q==='pass'?'q-pass':(q==='failed'?'q-failed':'q-unknown');
      const id = escapeHTML(r.record_id);
      const tags = r.metadata?.tags;
      return `<tr>
        <td><input type="checkbox" class="pick" value="${id}"></td>
        <td>${id}</td>
        <td>${escapeHTML(String(r.created_at||'').slice(0,19).replace('T',' '))}</td>
        <td>${Number(r.duration||0).toFixed(1)}s</td>
        <td>${fmt(r.rosbag_size)}</td>
        <td class="${cls}">${escapeHTML(q)}</td>
        <td>${escapeHTML(Array.isArray(tags) ? tags.join(',') : (tags || ''))}</td>
        <td>${r.local? '<span class="done">已下载</span>' : '-'}</td>
      </tr>`;
    }).join('');
    syncCount();
  }catch(e){ tbody.innerHTML = `<tr><td colspan="8" class="muted">加载失败：${escapeHTML(e.message)}</td></tr>`; }
}

function syncCount(){
  const n = document.querySelectorAll('.pick:checked').length;
  $('#btn-download').disabled = n===0 || Boolean(currentJob) || submitting;
  $('#btn-download').textContent = `下载选中 (${n})`;
}

function showProgress(p){
  if(!p) return;
  $('#progress').style.display='block';
  $('#p-phase').textContent = phaseLabels[p.phase] || p.phase || '-';
  $('#p-pct').textContent = (p.percent||0).toFixed(1)+'%';
  $('#p-rid').textContent = p.rid ? `${p.idx}/${p.n} ${p.rid}` : '-';
  $('#p-speed').textContent = p.phase==='transfer' && p.inst_mbps!=null
    ? `${p.inst_mbps.toFixed(1)} MiB/s（平均 ${p.avg_mbps.toFixed(1)}）` : '-';
  $('#p-bytes').textContent = fmt(p.bytes_done)+' / '+fmt(p.bytes_total||0);
  $('#p-eta').textContent = p.phase==='transfer' ? fmtSec(p.eta_s) : '-';
  $('#p-fill').style.width = Math.max(0,Math.min(100,p.percent||0))+'%';
  $('#p-msg').textContent = p.message || '';
}

function follow(job){
  currentJob = job.id;
  showProgress({phase:'starting', percent:0, message:'正在读取任务进度…'});
  $('#p-out').textContent = 'B70 数据目录：'+job.out_dir+'/bags';
  $('#log').style.display='block';
  $('#log').textContent = '';
  syncCount();
  async function poll(){
    try{
      const d = await getJSON('/api/jobs/'+encodeURIComponent(job.id));
      showProgress(d.progress);
      $('#log').textContent = d.log_tail || '';
      if(d.status!=='running'){
        currentJob = null;
        $('#hint').textContent = d.status==='done' ? '下载并解压完成' : '任务失败，见下方日志';
        loadRecords(); loadSystem();
        return;
      }
      $('#hint').textContent = '任务 '+job.id+' 运行中';
    }catch(e){
      if(e.message === 'no such job'){
        currentJob = null;
        $('#hint').textContent = '服务器已重启，无法继续查看此任务';
        loadRecords();
        return;
      }
      $('#hint').textContent = '进度暂时无法读取，正在重试…';
    }
    setTimeout(poll, 1000);
  }
  poll();
}

async function restoreJob(){
  try{
    const {jobs} = await getJSON('/api/jobs');
    const job = jobs.find(j=>j.status==='running') || jobs.at(-1);
    if(job && !currentJob && !submitting) follow(job);
  }catch(e){
    $('#hint').textContent = '无法读取任务状态：'+e.message;
  }
}

document.addEventListener('change', e=>{
  if(e.target.classList.contains('pick')) syncCount();
  if(e.target.id==='chk-all'){
    document.querySelectorAll('.pick').forEach(c=>c.checked=e.target.checked); syncCount();
  }
});

$('#btn-refresh').onclick = ()=>{ loadRecords(); loadSystem(); };

$('#btn-download').onclick = async ()=>{
  const ids = [...document.querySelectorAll('.pick:checked')].map(c=>c.value);
  if(!ids.length || currentJob || submitting) return;
  submitting = true;
  syncCount();
  $('#hint').textContent = '正在启动…';
  try{
    const job = await getJSON('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ids})});
    follow(job);
  }catch(e){
    $('#hint').textContent='启动失败：'+e.message;
  }finally{
    submitting = false;
    syncCount();
  }
};

$('#btn-speed').onclick = async ()=>{
  $('#hint').textContent = '测速中（约 10–30 秒）…';
  try{
    const r = await fetch('/api/speedtest',{method:'POST'});
    const d = await r.json();
    $('#hint').textContent = d.summary || (d.error||'测速结束');
  }catch(e){ $('#hint').textContent='测速失败: '+e; }
};

loadSystem(); loadRecords(); restoreJob();
setInterval(loadSystem, 15000);
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "ta2dl/1.0"

    def log_message(self, fmt, *args):  # 安静一点
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers
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

    def _json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    # -- routes
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        cfg = self.server.cfg
        refresh_jobs()

        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")

        if path == "/api/system":
            try:
                free = shutil.disk_usage(str(cfg["root"])).free
            except OSError:
                free = None
            return self._send(200, {
                "root": cfg["root"],
                "out_dir": cfg["root"],
                "disk_free": free,
                "link_mbps": read_link_mbps(cfg["iface"]),
                "jobs": list_jobs(),
            })

        if path == "/api/records":
            try:
                rows = api_json(cfg["api"]).get("records") or []
            except (RuntimeError, urllib.error.URLError, OSError) as exc:
                return self._send(502, {"error": f"取记录失败: {exc}"})
            quality = (query.get("quality") or [""])[0]
            since = (query.get("since") or [""])[0]
            out = []
            local_ids = downloaded_ids(Path(cfg["root"]))
            for r in rows:
                if quality and r.get("quality_status") != quality:
                    continue
                if since and (r.get("created_at") or "")[:10] < since:
                    continue
                item = dict(r)
                meta = as_dict(r.get("metadata"))
                item["metadata"] = {"tags": meta.get("tags") or []}
                item["local"] = r["record_id"] in local_ids
                out.append(item)
            return self._send(200, {"records": out, "total": len(out)})

        if path == "/api/jobs":
            return self._send(200, {"jobs": list_jobs()})

        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._send(200, job_snapshot(job))

        if path.startswith("/api/events/"):
            job_id = path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._stream_events(job)

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        cfg = self.server.cfg

        if path == "/api/jobs":
            body = self._json_body()
            ids = body.get("ids") if isinstance(body, dict) else None
            if not isinstance(ids, list) or not ids or any(
                not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", rid)
                for rid in ids
            ):
                return self._send(400, {"error": "请选择有效的下载记录"})
            try:
                job = new_job(list(dict.fromkeys(ids)), cfg["root"], cfg["api"], cfg["iface"],
                              keep_archive=bool(body.get("keep_archive")))
            except RuntimeError as exc:
                return self._send(409, {"error": str(exc)})
            except OSError as exc:
                return self._send(500, {"error": f"无法启动下载: {exc}"})
            return self._send(200, job_public(job))

        if path == "/api/speedtest":
            try:
                out = Path(cfg["root"]) / "web_jobs" / f"speed_{datetime.now():%Y%m%d_%H%M%S}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, str(find_cli()),
                                "--root", cfg["root"], "--api", cfg["api"],
                                "speedtest", "--iface", cfg["iface"], "--out", str(out)],
                               capture_output=True, timeout=180)
                report = {}
                if out.exists():
                    report = json.loads(out.read_text(encoding="utf-8"))
                summary = (f"链路 {report.get('link_mbps')} Mbps，"
                           f"实测 {report.get('MiB_per_second', '?')} MiB/s"
                           f"（{report.get('Mbps', '?')} Mbps，"
                           f"{report.get('achieved_percent', '?')}% 达成）")
                return self._send(200, {"report": report, "summary": summary})
            except subprocess.TimeoutExpired:
                return self._send(504, {"error": "测速超时"})
            except Exception as exc:  # noqa: BLE001
                return self._send(500, {"error": f"测速失败: {exc}"})

        return self._send(404, {"error": "not found"})

    # -- SSE
    def _stream_events(self, job):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        last_sig = None
        deadline = time.time() + 3600
        try:
            while time.time() < deadline:
                refresh_jobs()
                snapshot = job_snapshot(job)
                payload = {key: snapshot[key] for key in ("status", "progress", "log_tail")}
                sig = json.dumps(payload, ensure_ascii=False)
                if sig != last_sig:
                    self.wfile.write(b"data: " + sig.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                    last_sig = sig
                if snapshot["status"] != "running":
                    break
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            pass


def pick_port(host, start):
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"{start}-{start + 19} 端口都被占用")


def main(argv=None):
    ap = argparse.ArgumentParser(description="TA2 下载控制台")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"下载根目录（默认 {DEFAULT_ROOT}）")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--iface", default=DEFAULT_IFACE)
    args = ap.parse_args(argv)

    port = pick_port(args.host, args.port)
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    httpd.cfg = {"root": str(Path(args.root)), "api": args.api, "iface": args.iface}
    print(f"控制台: http://{args.host}:{port}  root={args.root}  api={args.api}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n退出", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
