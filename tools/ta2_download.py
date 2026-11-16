#!/usr/bin/env python3
"""
ta2_download.py — 灵御 TA2 数采记录：列出 / 下载 / 测速 / 校验

设计约束
  * 纯标准库，唯一的系统依赖是 curl（B70 无 pip 也能跑）
  * 只读 API，不做任何服务端写操作（除了显式触发打包 POST /downloads）
  * 进度双通道：人看单行刷新，机器读 progress.json（原子写）+ events.jsonl

子命令
  list       列出远端记录（可选：拉详情做 topic 预筛，输出 csv）
  download   下载 + 校验 + 解压
  speedtest  链路 / RTT / 首字节 / 持续吞吐
  verify     离线校验一个已有归档（不联网）

典型用法
  python3 ta2_download.py list --prescreen prescreen.csv
  python3 ta2_download.py download rec_20261115_094024_a16bb1e8 rec_20261115_093329_99d3a3d2
  python3 ta2_download.py download --all --quality pass --since 2026-11-15
  python3 ta2_download.py download --all --since 2026-11-15 --before 2026-11-15T11:00
      ↑ 采集还在继续时用 --before 锁一个批次截止点，之后新采的不会被打进来
        时间统一用 T 分隔（2026-11-15T11:00），带空格的写法在某些 shell 里会被拆开
  python3 ta2_download.py speedtest
  python3 ta2_download.py verify /data/ta2/records_20261114/archives/xxx.archive
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

DEFAULT_API = "http://192.168.31.218:5173/api/v1/records"
DEFAULT_ROOT = "/data/ta2"

# 阶段 -> (总进度下界, 上界)
PHASE_SPAN = {
    "probe": (0.0, 2.0),
    "packing": (2.0, 40.0),
    "transfer": (40.0, 85.0),
    "verify": (85.0, 92.0),
    "extract": (92.0, 100.0),
}

# 预筛期望的 topic 特征（依据 11/14 实测）
TOPIC_RULES = [
    ("left_arm", "/left_arm/", 50.0),
    ("right_arm", "/right_arm/", 50.0),
    ("video_head", "/xr_video_topic", None),
    ("video_left", "/left/color/image_raw", None),
    ("video_right", "/right/color/image_raw", None),
]


# ---------------------------------------------------------------- 基础工具

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def human_bytes(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def human_secs(s):
    if s is None:
        return "?"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def make_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_OPENER = make_opener()


class ApiError(RuntimeError):
    pass


def as_dict(value):
    """API 里同一字段有时是 dict、有时是 JSON 字符串，统一成 dict。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def record_tags(rec):
    return as_dict(rec.get("metadata")).get("tags") or []


def api_call(url, post=False, timeout=30):
    body = b"" if post else None
    req = urllib.request.Request(url, data=body)
    try:
        with _OPENER.open(req, timeout=timeout) as fh:
            payload = json.load(fh)
    except urllib.error.HTTPError as exc:
        raise ApiError(f"HTTP {exc.code} {url}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"连不上 {url}: {exc.reason}") from exc
    if payload.get("code") != 20000:
        raise ApiError(f"接口返回异常: {payload.get('code')} {payload.get('message')}")
    return payload["data"]


# ---------------------------------------------------------------- 进度

class Progress:
    """把进度原子写进 progress.json，并追加 events.jsonl。"""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "progress.json"
        self.events = self.dir / "events.jsonl"
        self.state = {
            "updated_at": now_iso(),
            "rid": None,
            "idx": 0,
            "n": 0,
            "phase": "idle",
            "phase_percent": 0.0,
            "percent": 0.0,
            "bytes_done": 0,
            "bytes_total": 0,
            "inst_mbps": 0.0,
            "avg_mbps": 0.0,
            "eta_s": None,
            "link_mbps": None,
            "message": "",
            "ok": None,
        }
        self._lock = threading.Lock()

    def update(self, **kw):
        with self._lock:
            self.state.update(kw)
            phase = self.state.get("phase") or "idle"
            lo, hi = PHASE_SPAN.get(phase, (0.0, 100.0))
            pct = float(self.state.get("phase_percent") or 0.0)
            pct = max(0.0, min(100.0, pct))
            self.state["percent"] = round(lo + (hi - lo) * pct / 100.0, 2)
            self.state["updated_at"] = now_iso()
            tmp = self.path.with_suffix(".tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self.state, fh, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
            except OSError:
                pass

    def event(self, name, **kw):
        rec = {"ts": now_iso(), "event": name}
        rec.update(kw)
        try:
            with open(self.events, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass


def read_link_mbps(ifname="enp9s0"):
    try:
        with open(f"/sys/class/net/{ifname}/speed", "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- 下载核心

class SizeSampler(threading.Thread):
    """每 0.5s 采样文件大小，算瞬时/平均速度与 ETA，写进 Progress。"""

    def __init__(self, path, total, progress, interval=0.5):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.total = int(total or 0)
        self.progress = progress
        self.interval = interval
        self._stop_evt = threading.Event()
        self.samples = []

    def run(self):
        t0 = time.time()
        base = 0
        while not self._stop_evt.is_set():
            try:
                size = self.path.stat().st_size
            except OSError:
                size = base
            now = time.time()
            self.samples.append((now, size))
            if len(self.samples) > 24:
                self.samples.pop(0)
            inst = 0.0
            if len(self.samples) >= 2:
                t_a, s_a = self.samples[-2]
                t_b, s_b = self.samples[-1]
                if t_b > t_a:
                    inst = (s_b - s_a) / (t_b - t_a) / 1e6
            elapsed = max(now - t0, 1e-6)
            avg = (size - base) / elapsed / 1e6
            pct = (size / self.total * 100.0) if self.total else 0.0
            eta = None
            if self.total and avg > 0.01:
                eta = (self.total - size) / (avg * 1e6)
            self.progress.update(
                phase="transfer",
                phase_percent=pct,
                bytes_done=size,
                bytes_total=self.total,
                inst_mbps=round(inst, 2),
                avg_mbps=round(avg, 2),
                eta_s=int(eta) if eta is not None else None,
            )
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()


def run_curl(url, dest, log_path):
    cmd = [
        "curl", "--noproxy", "*", "-fL",
        "--retry", "3", "--retry-delay", "2",
        "--connect-timeout", "15",
        "--speed-limit", "524288", "--speed-time", "120",
        "-sS", "-o", str(dest), url,
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def verify_archive(path, expected_size=None):
    """返回 (ok, info)。ok=False 时 info['error'] 说明原因。"""
    path = Path(path)
    info = {"file": str(path)}
    if not path.exists():
        return False, {"error": "文件不存在"}
    size = path.stat().st_size
    info["bytes"] = size
    if expected_size and size != int(expected_size):
        return False, {"error": f"大小不符: 实际 {size} != 期望 {int(expected_size)}"}

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    info["sha256"] = h.hexdigest()

    info["format"] = "zip" if zipfile.is_zipfile(path) else ("tar" if tarfile.is_tarfile(path) else "unknown")
    if info["format"] == "zip":
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                if bad:
                    return False, {"error": f"zip CRC 失败: {bad}", **info}
                info["members"] = len(zf.namelist())
        except zipfile.BadZipFile as exc:
            return False, {"error": f"坏 zip: {exc}", **info}
    elif info["format"] == "unknown":
        return False, {"error": "既不是 zip 也不是 tar", **info}
    return True, info


def safe_extract(archive, dest):
    """解压到 dest，拒绝路径穿越；返回解压后文件数。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    count = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                target = (dest / name).resolve()
                if not str(target).startswith(str(root)):
                    raise RuntimeError(f"不安全的路径: {name}")
            zf.extractall(dest)
            count = len(zf.namelist())
    else:
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                tf.extractall(dest)
            count = len(tf.getnames())
    return count


def check_payload(dest):
    """解压后必须同时存在 metadata.yaml 与 mcap/db3。"""
    dest = Path(dest)
    metas = list(dest.rglob("metadata.yaml"))
    bags = list(dest.rglob("*.mcap")) + list(dest.rglob("*.db3"))
    return metas, bags


def download_one(rid, api, out_dir, progress, keep_archive=False, dry_run=False,
                 work_dir=None):
    """单条记录：打包 -> 传输 -> 校验 -> 解压。返回结果 dict。"""
    out_dir = Path(out_dir)
    work_dir = Path(work_dir) if work_dir else out_dir
    archives = work_dir / "archives"
    bags = work_dir / "bags"
    logs = work_dir / "logs"
    for d in (archives, bags, logs):
        d.mkdir(parents=True, exist_ok=True)

    base = api.rstrip("/") + "/"
    result = {"record_id": rid, "ok": False}
    t_start = time.time()
    final_dest = out_dir / "bags" / rid

    try:
        if work_dir != out_dir and not dry_run and final_dest.exists():
            metas, bagfiles = check_payload(final_dest)
            if not metas or not bagfiles:
                raise RuntimeError(f"已有目录缺少数采文件，请检查 {final_dest}")
            result.update(ok=True, skipped=True,
                          metadata_files=[str(p) for p in metas],
                          bag_files=[str(p) for p in bagfiles])
            progress.update(rid=rid, phase="extract", phase_percent=100, ok=True,
                            bytes_done=0, bytes_total=0, message="已存在，跳过下载")
            progress.event("done", rid=rid, ok=True, skipped=True)
            return result

        progress.update(
            rid=rid, phase="probe", phase_percent=5, message="查询归档状态",
            bytes_done=0, bytes_total=0, inst_mbps=0.0, avg_mbps=0.0, eta_s=None, ok=None,
        )
        status = api_call(base + rid + "/downloads/latest")
        if not status.get("ready_to_download") and status.get("archive_status") != "packing":
            progress.event("packing_request", rid=rid)
            api_call(base + rid + "/downloads", post=True)

        t_pack = time.time()
        waited = 0
        while True:
            status = api_call(base + rid + "/downloads/latest")
            if status.get("ready_to_download"):
                break
            if status.get("archive_status") == "failed":
                raise RuntimeError(f"服务端打包失败: {status}")
            if waited > 600:
                raise RuntimeError("打包超时 (>600s)")
            prog = float(status.get("archive_progress") or 0.0)
            progress.update(
                phase="packing",
                phase_percent=prog * 100.0,
                bytes_total=status.get("archive_size") or 0,
                message=f"服务端打包中 {prog * 100:.0f}%",
            )
            time.sleep(1.5)
            waited += 1.5
        result["prepare_seconds"] = round(time.time() - t_pack, 2)

        expected = int(status.get("archive_size") or 0)
        result["archive_bytes"] = expected
        if dry_run:
            result["ok"] = True
            result["note"] = "dry-run（未实际下载）"
            progress.update(phase="transfer", phase_percent=100, message="dry-run")
            return result

        target = archives / f"{rid}.archive"
        part = target.with_suffix(".archive.part")
        if part.exists():
            part.unlink()

        progress.update(
            phase="transfer", phase_percent=0,
            bytes_done=0, bytes_total=expected,
            message="开始传输",
        )
        sampler = SizeSampler(part, expected, progress)
        sampler.start()
        url = base + rid + "/download"
        rc = run_curl(url, part, logs / f"{rid}.curl.log")
        sampler.stop()
        sampler.join(timeout=2)
        if rc != 0:
            raise RuntimeError(f"curl 失败 rc={rc}，见 {logs / (rid + '.curl.log')}")

        actual = part.stat().st_size
        result["bytes"] = actual
        result["transfer_seconds"] = round(time.time() - t_start - result["prepare_seconds"], 2)
        if result["transfer_seconds"] > 0:
            mbps = actual * 8 / result["transfer_seconds"] / 1e6
            result["Mbps"] = round(mbps, 2)
            result["MiB_per_second"] = round(actual / result["transfer_seconds"] / 1048576, 2)

        progress.update(
            phase="verify", phase_percent=10, bytes_done=actual, bytes_total=actual,
            message="校验大小与哈希",
        )
        ok, info = verify_archive(part, expected)
        result.update({k: v for k, v in info.items() if k != "file"})
        if not ok:
            raise RuntimeError(f"校验失败: {info.get('error')}")
        os.replace(part, target)

        progress.update(phase="extract", phase_percent=10, message="解压")
        dest = bags / rid
        if dest.exists():
            shutil.rmtree(dest)
        n = safe_extract(target, dest)
        metas, bagfiles = check_payload(dest)
        if not metas or not bagfiles:
            raise RuntimeError(
                f"解压后缺内容: metadata={len(metas)} mcap/db3={len(bagfiles)}"
            )
        if dest != final_dest:
            final_dest.parent.mkdir(parents=True, exist_ok=True)
            if final_dest.exists():
                raise FileExistsError(f"另一个任务已创建 {final_dest}")
            dest.rename(final_dest)
            metas = [final_dest / p.relative_to(dest) for p in metas]
            bagfiles = [final_dest / p.relative_to(dest) for p in bagfiles]
        result["extracted_files"] = n
        result["metadata_files"] = [str(p) for p in metas]
        result["bag_files"] = [str(p) for p in bagfiles]

        if not keep_archive:
            target.unlink(missing_ok=True)
            result["archive_kept"] = False
        else:
            result["archive_kept"] = True

        result["ok"] = True
        result["total_seconds"] = round(time.time() - t_start, 2)
        progress.update(
            phase="extract", phase_percent=100, ok=True,
            message=f"完成 {human_bytes(result['bytes'])}",
        )
        progress.event("done", rid=rid, ok=True, bytes=result.get("bytes"))
    except Exception as exc:  # noqa: BLE001 - 单条失败不中断批次
        result["error"] = f"{type(exc).__name__}: {exc}"
        progress.update(ok=False, message=result["error"])
        progress.event("error", rid=rid, error=result["error"])
    return result


# ---------------------------------------------------------------- 子命令

def fetch_records(api):
    """GET /records 的 data 是 {total, records:[...]}，这里统一取出记录列表。"""
    data = api_call(api)
    if isinstance(data, dict):
        rows = data.get("records")
        if isinstance(rows, list):
            return rows
        return []
    if isinstance(data, list):
        return data
    return []


def norm_ts(value):
    """把 2026-11-15T10:20:46.1 或 2026-11-15 10:20 统一成可比较的 'YYYY-MM-DD HH:MM:SS'。"""
    if not value:
        return ""
    s = str(value).strip().replace("T", " ")
    if len(s) == 10:            # 只有日期
        s += " 00:00:00"
    elif len(s) == 16:          # 到分钟
        s += ":00"
    return s[:19]


def filter_records(rows, quality=None, since=None, before=None, limit=None):
    """since/before 都按 created_at 过滤；before 用于"只下这个时间点之前的批次"。"""
    out = []
    lo = norm_ts(since) if since else ""
    hi = norm_ts(before) if before else ""
    for r in rows:
        if quality and r.get("quality_status") != quality:
            continue
        created = norm_ts(r.get("created_at"))
        if lo and created < lo:
            continue
        if hi and created >= hi:
            continue
        out.append(r)
    if limit:
        out = out[:limit]
    return out


def prescreen(api, rows, csv_path):
    """拉每条详情里的 quality_result.bag_info.topics，做 topic 预筛。"""
    writer = None
    fh = None
    if csv_path:
        fh = open(csv_path, "w", encoding="utf-8-sig", newline="")
        writer = csv.writer(fh)
        writer.writerow(["record_id", "created_at", "duration_s", "rosbag_size",
                         "quality_status", "verdict", "missing", "low_hz", "note"])
    verdicts = {}
    try:
        for i, r in enumerate(rows, 1):
            rid = r["record_id"]
            missing, low = [], []
            note = ""
            try:
                detail = api_call(api.rstrip("/") + "/" + rid)
                qr = as_dict(detail.get("quality_result"))
                topics = as_dict(as_dict(qr.get("bag_info")).get("topics"))
                if not topics:
                    verdict = "unknown"
                    note = "无 bag_info.topics（未质检或质检未出）"
                else:
                    for label, needle, min_hz in TOPIC_RULES:
                        hit = None
                        for name, meta in topics.items():
                            if needle in name:
                                hit = (name, as_dict(meta))
                                break
                        if hit is None:
                            missing.append(label)
                            continue
                        if min_hz is not None:
                            hz = float(hit[1].get("frequency") or 0.0)
                            if hz < min_hz:
                                low.append(f"{label}@{hz:.0f}Hz")
                    verdict = "ok" if not missing and not low else ("drop" if missing else "warn")
            except ApiError as exc:
                verdict = "unknown"
                note = str(exc)

            if r.get("quality_status") == "failed":
                verdict = "drop"
                note = (note + " quality=failed").strip()
            verdicts[rid] = verdict
            if writer:
                writer.writerow([rid, r.get("created_at"), round(float(r.get("duration") or 0), 1),
                                 r.get("rosbag_size"), r.get("quality_status"), verdict,
                                 ";".join(missing), ";".join(low), note])
            print(f"  [{i:>2}/{len(rows)}] {rid} {verdict:<8} "
                  f"missing={missing or '-'} low={low or '-'}", flush=True)
    finally:
        if fh:
            fh.close()
    return verdicts


def cmd_list(args):
    rows = fetch_records(args.api)
    rows = filter_records(rows, args.quality, args.since, args.before, args.limit)
    print(f"{'record_id':<30} {'created':<19} {'dur':>8} {'size':>10} {'quality':<8} tags")
    print("-" * 100)
    for r in rows:
        tags = ",".join(record_tags(r))
        print(f"{r['record_id']:<30} {r.get('created_at', ''):<19} "
              f"{float(r.get('duration') or 0):>7.1f}s {human_bytes(r.get('rosbag_size')):>10} "
              f"{r.get('quality_status', ''):<8} {tags}")
    print(f"\n合计 {len(rows)} 条，"
          f"{human_bytes(sum(int(r.get('rosbag_size') or 0) for r in rows))}")
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.prescreen:
        print(f"\n== topic 预筛 -> {args.prescreen} ==")
        prescreen(args.api, rows, args.prescreen)
    return 0


def resolve_ids(args):
    if args.ids:
        return list(args.ids)
    rows = fetch_records(args.api)
    rows = filter_records(rows, args.quality, args.since, args.before, args.limit)
    ids = [r["record_id"] for r in rows]
    if not ids:
        raise SystemExit("按筛选条件没有匹配到任何记录")
    return ids


def cmd_download(args):
    ids = resolve_ids(args)
    out_dir = Path(args.out) if args.out else \
        Path(args.root) / f"records_{datetime.now().strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    rows = {r["record_id"]: r for r in fetch_records(args.api)}
    for rid in ids:
        total_bytes += int((rows.get(rid) or {}).get("rosbag_size") or 0)

    free = shutil.disk_usage(str(out_dir)).free
    need = total_bytes * args.disk_factor
    print(f"批次: {len(ids)} 条, 预计 {human_bytes(total_bytes)}, "
          f"需要 {human_bytes(need)} (x{args.disk_factor}), 可用 {human_bytes(free)}")
    if need > free:
        raise SystemExit(f"磁盘不足: 需要 {human_bytes(need)}，只有 {human_bytes(free)}（--disk-factor 可调）")

    run_dir = Path(args.run_dir) if args.run_dir else out_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = (run_dir if args.run_dir else out_dir) / "transfer_results.json"
    progress = Progress(run_dir)
    progress.update(n=len(ids), link_mbps=read_link_mbps(args.iface))
    progress.event("start", n=len(ids), ids=ids)

    results = []
    t_all = time.time()
    for i, rid in enumerate(ids, 1):
        print(f"\n[{i}/{len(ids)}] {rid}", flush=True)
        progress.update(idx=i, n=len(ids))
        res = download_one(rid, args.api, out_dir, progress,
                           keep_archive=args.keep_archive, dry_run=args.dry_run,
                           work_dir=run_dir if args.run_dir else None)
        results.append(res)
        if res.get("skipped"):
            print(f"  SKIP 已存在: {out_dir / 'bags' / rid}")
        elif res.get("ok"):
            print(f"  OK  {human_bytes(res.get('bytes'))}  "
                  f"打包 {res.get('prepare_seconds', 0):.1f}s  "
                  f"传输 {res.get('transfer_seconds', 0):.1f}s  "
                  f"{res.get('MiB_per_second', 0):.2f} MiB/s "
                  f"({res.get('Mbps', 0):.1f} Mbps)")
        else:
            print(f"  FAIL {res.get('error')}")
        with open(summary, "w", encoding="utf-8") as fh:
            json.dump({"updated_at": now_iso(), "results": results}, fh,
                      ensure_ascii=False, indent=2)

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"\n完成 {ok_n}/{len(results)}，总耗时 {human_secs(time.time() - t_all)}")
    print(f"结果: {summary}")
    print(f"进度: {run_dir / 'progress.json'}")
    return 0 if ok_n == len(results) else 1


def cmd_speedtest(args):
    link = read_link_mbps(args.iface)
    report = {"ts": now_iso(), "iface": args.iface, "link_mbps": link}

    print(f"== 物理链路 ==\n  {args.iface}: {link} Mbps" if link else "== 物理链路 ==\n  链路速率不可读")
    if link:
        print(f"  理论上限: {link / 8:.1f} MB/s")

    print("\n== RTT (10 次 GET /api/v1/records) ==")
    rtts = []
    for _ in range(10):
        t0 = time.time()
        try:
            api_call(args.api, timeout=10)
        except ApiError as exc:
            print(f"  失败: {exc}")
            break
        rtts.append((time.time() - t0) * 1000)
    if rtts:
        rtts.sort()
        print(f"  min {rtts[0]:.1f} ms / median {rtts[len(rtts) // 2]:.1f} ms / max {rtts[-1]:.1f} ms")
        report["rtt_ms"] = {"min": round(rtts[0], 2),
                            "median": round(rtts[len(rtts) // 2], 2),
                            "max": round(rtts[-1], 2)}

    if args.rid:
        print(f"\n== 持续吞吐（{args.rid} 前 {args.sample_mib} MiB）==")
        base = args.api.rstrip("/") + "/" + args.rid
        try:
            status = api_call(base + "/downloads/latest")
            if not status.get("ready_to_download"):
                print("  归档未就绪，跳过吞吐测试（先跑一次 download 触发打包）")
            else:
                req = urllib.request.Request(
                    base + "/download",
                    headers={"Range": f"bytes=0-{args.sample_mib * 1048576 - 1}"},
                )
                t0 = time.time()
                got = 0
                first = None
                with _OPENER.open(req, timeout=30) as resp:
                    while got < args.sample_mib * 1048576:
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        if first is None:
                            first = time.time() - t0
                        got += len(chunk)
                dt = time.time() - t0
                mbps = got * 8 / dt / 1e6
                print(f"  {human_bytes(got)} / {dt:.2f}s = {got / dt / 1048576:.2f} MiB/s "
                      f"({mbps:.1f} Mbps)")
                if first is not None:
                    print(f"  首字节 {first * 1000:.1f} ms")
                report.update({"sample_bytes": got, "sample_seconds": round(dt, 3),
                               "MiB_per_second": round(got / dt / 1048576, 2),
                               "Mbps": round(mbps, 1),
                               "first_byte_ms": round((first or 0) * 1000, 1)})
                if link:
                    report["achieved_percent"] = round(mbps / link * 100, 1)
                    print(f"  达成率 {report['achieved_percent']}% of {link} Mbps")
                    if report["achieved_percent"] > 85:
                        print("  结论：链路已饱和，软件层无优化空间；要提速只能换线/换口。")
                    else:
                        print("  结论：未打满链路，可能是服务端打包/磁盘/机器人侧限制。")
        except (ApiError, OSError) as exc:
            print(f"  吞吐测试失败: {exc}")

    out = Path(args.out or "speed_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n报告: {out}")
    return 0


def cmd_verify(args):
    ok_all = True
    for p in args.files:
        ok, info = verify_archive(p, None)
        if ok:
            print(f"OK   {human_bytes(info['bytes'])} {info.get('format')} "
                  f"sha256={info.get('sha256', '')[:16]}...  {p}")
        else:
            ok_all = False
            print(f"FAIL {info.get('error')}  {p}")
    return 0 if ok_all else 1


def build_parser():
    ap = argparse.ArgumentParser(
        description="TA2 数采记录下载/测速/校验工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--api", default=DEFAULT_API, help=f"记录 API 基址（默认 {DEFAULT_API}）")
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"数据根目录（默认 {DEFAULT_ROOT}）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出远端记录")
    p_list.add_argument("--quality", help="按 quality_status 过滤，如 pass / failed / unknown")
    p_list.add_argument("--since", help="created_at 不早于此时间，如 2026-11-15 或 2026-11-15 10:00")
    p_list.add_argument("--before", help="created_at 早于此时间（批次截止）")
    p_list.add_argument("--limit", type=int)
    p_list.add_argument("--json", action="store_true", help="额外打印原始 JSON")
    p_list.add_argument("--prescreen", metavar="CSV", help="拉详情做 topic 预筛并写 CSV")
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="下载记录")
    p_dl.add_argument("ids", nargs="*", help="record_id 列表；留空配合 --all 使用")
    p_dl.add_argument("--all", action="store_true", help="按筛选条件下载全部")
    p_dl.add_argument("--quality", help="按 quality_status 过滤，如 pass")
    p_dl.add_argument("--since", help="created_at 不早于此时间")
    p_dl.add_argument("--before", help="created_at 早于此时间（批次截止，采集与下载并行时用）")
    p_dl.add_argument("--limit", type=int)
    p_dl.add_argument("--out", help="输出批次目录（默认 <root>/records_<今天>）")
    p_dl.add_argument("--run-dir", help="显式指定本次运行的进度目录（网页端用）")
    p_dl.add_argument("--keep-archive", action="store_true", help="解压后保留归档 zip")
    p_dl.add_argument("--dry-run", action="store_true", help="只触发打包与查询，不落盘")
    p_dl.add_argument("--disk-factor", type=float, default=2.5,
                      help="磁盘余量倍数门槛（默认 2.5）")
    p_dl.add_argument("--iface", default="enp9s0")
    p_dl.set_defaults(func=cmd_download)

    p_st = sub.add_parser("speedtest", help="链路/延迟/吞吐测试")
    p_st.add_argument("--rid", help="用该记录的归档做吞吐采样")
    p_st.add_argument("--sample-mib", type=int, default=10)
    p_st.add_argument("--iface", default="enp9s0")
    p_st.add_argument("--out", help="报告输出路径（默认 speed_report.json）")
    p_st.set_defaults(func=cmd_speedtest)

    p_vf = sub.add_parser("verify", help="离线校验已有归档")
    p_vf.add_argument("files", nargs="+")
    p_vf.set_defaults(func=cmd_verify)

    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ApiError as exc:
        print(f"接口错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())



