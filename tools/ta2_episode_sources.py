#!/usr/bin/env python3
"""
ta2_episode_sources.py — 生成「LeRobot episode → 原始记录 / 采集时间」映射

背景
    转换日志里每条 bag 打印一次 `=== Processing rec_xxx ===`，随后每存一条打印
    `✅ Saved episode N with M frames`。所以从日志就能还原 episode 属于哪个 bag。

时间口径
    record_id 形如 rec_20261115_094024_a16bb1e8，内嵌录制开始时间（UTC+8）。
    同一 bag 切出的多条 episode 共享该时间 —— 这是 **bag 级**精度，不是 episode 级。

用法
    python3 ta2_episode_sources.py --dataset /path/to/dataset --log /tmp/full_convert.log
    python3 ta2_episode_sources.py --dataset ... --log ... --api http://192.168.31.218:5173/api/v1/records
    python3 ta2_episode_sources.py --dataset ... --log ... --dry-run
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_REVIEW_ROOT = "/data/ta2/review"
DEFAULT_OFFICIAL_SCRIPTS = "/data/ta2/software/convert/rosbag_to_dataset_TA2/scripts"

RE_PROCESSING = re.compile(r"Processing\s+(rec_\S+)")
RE_SAVED = re.compile(r"Saved episode\s+(\d+)\s+with\s+(\d+)\s+frames")
RE_RID_TIME = re.compile(r"rec_(\d{8})_(\d{6})")


def dataset_key(dataset_path):
    """数据集目录名做 key；冲突时由调用方加后缀。这里只取 basename。"""
    return Path(dataset_path).resolve().name


def parse_rid_time(rid):
    """rec_20261115_094024_xxx -> '2026-11-15 09:40:24'；解析不了返回 None。"""
    m = RE_RID_TIME.search(rid or "")
    if not m:
        return None
    day, clock = m.group(1), m.group(2)
    try:
        dt = datetime.strptime(day + clock, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_log(log_path):
    """返回 {episode_index: {'record_id':.., 'frames':..}}，保持日志里的出现顺序。"""
    out = {}
    current_rid = None
    if not os.path.exists(log_path):
        return out, "log 不存在"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = RE_PROCESSING.search(line)
                if m:
                    current_rid = m.group(1)
                    continue
                m = RE_SAVED.search(line)
                if m and current_rid:
                    idx = int(m.group(1))
                    out[idx] = {"record_id": current_rid, "frames": int(m.group(2))}
    except OSError as exc:
        return out, f"读日志失败: {exc}"
    return out, None


def load_dataset_lengths(dataset_path):
    """从 meta/episodes.jsonl 取 episode 列表（用于发现日志里漏掉的条目）。"""
    p = Path(dataset_path) / "meta" / "episodes.jsonl"
    rows = []
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def fetch_robot_records(api):
    """可选：从机器人接口取 duration / quality，失败则返回空表，不影响主流程。"""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(api, timeout=10) as fh:
            payload = json.load(fh)
        data = payload.get("data") or {}
        rows = data.get("records") if isinstance(data, dict) else data
        return {r["record_id"]: r for r in (rows or []) if r.get("record_id")}
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return {}


def build(dataset, log_path, api=None):
    parsed, err = parse_log(log_path)
    episodes = load_dataset_lengths(dataset)
    robot = fetch_robot_records(api) if api else {}

    n_meta = len(episodes)
    rows = []
    unresolved = 0
    for row in episodes:
        idx = row.get("episode_index")
        info = parsed.get(idx)
        rec = {
            "episode_index": idx,
            "length": row.get("length"),
            "record_id": None,
            "bag_started_at": None,
            "bag_started_at_source": None,
            "quality_status": None,
            "note": "",
        }
        if info:
            rec["record_id"] = info["record_id"]
            ts = parse_rid_time(info["record_id"])
            if ts:
                rec["bag_started_at"] = ts
                rec["bag_started_at_source"] = "record_id"
                rec["note"] = "bag 级时间：同一条记录切出的多条共享该录制开始时间"
            meta = robot.get(info["record_id"]) or {}
            if meta:
                rec["quality_status"] = meta.get("quality_status")
                rec["record_duration_s"] = round(float(meta.get("duration") or 0), 2) or None
                rec["record_created_at"] = meta.get("created_at")
        else:
            unresolved += 1
            rec["note"] = "日志中未找到对应 bag 记录"
        rows.append(rec)
    return rows, {"log_error": err, "log_episodes": len(parsed), "meta_episodes": n_meta,
                  "unresolved": unresolved}


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 episode -> 原始记录/采集时间 映射")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--log", required=True, help="转换日志路径")
    ap.add_argument("--review-root", default=DEFAULT_REVIEW_ROOT)
    ap.add_argument("--out", default=None, help="输出 jsonl；默认 <review-root>/episode_sources/<key>.jsonl")
    ap.add_argument("--api", default=None, help="可选：机器人记录接口，补 duration/quality")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rows, summary = build(args.dataset, args.log, args.api)
    key = dataset_key(args.dataset)
    out = Path(args.out) if args.out else Path(args.review_root) / "episode_sources" / f"{key}.jsonl"

    print(f"数据集: {args.dataset}")
    print(f"  meta episodes = {summary['meta_episodes']}, 日志 episodes = {summary['log_episodes']}, "
          f"未解析 = {summary['unresolved']}")
    if summary["log_error"]:
        print(f"  [!] {summary['log_error']}")
    if summary["log_episodes"] and summary["log_episodes"] != summary["meta_episodes"]:
        print("  [!] 日志条数与数据集条数不一致，可能有 episode 被丢弃或日志被截断")

    for r in rows[:5]:
        print(f"  ep{r['episode_index']:>3}  {r['record_id'] or '—':<32} {r['bag_started_at'] or '—'}")

    if args.dry_run:
        print(f"\n[dry-run] 未写文件；目标为 {out}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {"dataset": str(Path(args.dataset).resolve()),
                                       "log": str(Path(args.log).resolve()),
                                       "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                       **summary}}, ensure_ascii=False) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
