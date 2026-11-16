#!/usr/bin/env python3
"""
ta2_export_subset.py — 按标注导出训练子集（只保留「有效」条目）

做法
    1. 硬链接复制整个数据集（同分区不占额外空间，几秒完成）
    2. 调官方 delete_wrong_episodes.py 删除「无效」与「未标记」的 episode 并重排索引
    3. 写入 prompt（统一 prompt + 单条覆盖）
    4. 校验产出并写 export_manifest.json

约定
    * 只保留 mark == "valid"；invalid 与 unmarked 一律排除
    * 绝不修改源数据集
    * 需要用 venv 的 python 运行（要 pyarrow）

用法
    /data/ta2/tools/venv/bin/python ta2_export_subset.py \
        --dataset /data/ta2/converted/.../grasp_red_sandbag \
        --name grasp_red_sandbag_valid
    python3 ta2_export_subset.py ... --dry-run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_REVIEW_ROOT = "/data/ta2/review"
DEFAULT_OFFICIAL_SCRIPTS = "/data/ta2/software/convert/rosbag_to_dataset_TA2/scripts"
DEFAULT_VENV_PYTHON = "/data/ta2/tools/venv/bin/python"


# ---------------------------------------------------------------- 基础

def load_annotations(path):
    if not os.path.exists(path):
        raise SystemExit(f"标注文件不存在: {path}（先在网页里标注）")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_episodes(dataset):
    p = Path(dataset) / "meta" / "episodes.jsonl"
    rows = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(dataset):
    p = Path(dataset) / "meta" / "tasks.jsonl"
    rows = []
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def split_marks(annotations, n_episodes):
    """返回 (valid_sorted, invalid_sorted, unmarked_sorted)。"""
    eps = annotations.get("episodes") or {}
    valid, invalid, unmarked = [], [], []
    for idx in range(n_episodes):
        mark = (eps.get(str(idx)) or {}).get("mark") or "unmarked"
        if mark == "valid":
            valid.append(idx)
        elif mark == "invalid":
            invalid.append(idx)
        else:
            unmarked.append(idx)
    return valid, invalid, unmarked


# ---------------------------------------------------------------- 复制

def hardlink_copy(src, dst):
    """混合复制：meta/data 真实复制（会被官方脚本改写），videos 硬链接（只读，省 3GB）。

    绝对不能用 cp -al 全量硬链接：delete_wrong_episodes.py 会重写 meta/*.jsonl 和
    parquet 的索引列，硬链接下改的是同一个 inode，会把源数据集一起改掉。
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    videos_linked = False

    for item in sorted(src.iterdir()):
        target = dst / item.name
        if item.is_dir() and item.name == "videos":
            target.mkdir()
            r = subprocess.run(["cp", "-al", f"{item}/.", str(target)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                videos_linked = True
                continue
            shutil.rmtree(target, ignore_errors=True)
        r = subprocess.run(["cp", "-a", str(item), str(dst)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"复制失败 {item}: {r.stderr.strip()[:300]}")

    if videos_linked:
        return "hybrid", "meta/data 真实复制（会被改写），videos 硬链接（省约 3GB）"
    return "copy", "硬链接不可用，已整份复制"


# ---------------------------------------------------------------- prompt

def apply_single_prompt(dataset, prompt):
    """整份数据集统一 prompt：重写 tasks.jsonl + episodes.jsonl 的 tasks 字段。"""
    ds = Path(dataset)
    (ds / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": prompt}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    episodes = load_episodes(ds)
    for e in episodes:
        e["tasks"] = [prompt]
    with open(ds / "meta" / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    # task_index 全部为 0，parquet 无需改写
    update_total_tasks(ds, 1)
    return {"mode": "single", "tasks": [prompt]}


def apply_prompt_map(dataset, prompt_of_new_index):
    """按新索引给每条 episode 指定 prompt：重写 tasks.jsonl + parquet 的 task_index。"""
    import pyarrow.parquet as pq
    import pyarrow as pa

    ds = Path(dataset)
    uniq = []
    for p in prompt_of_new_index.values():
        if p not in uniq:
            uniq.append(p)
    task_index = {p: i for i, p in enumerate(uniq)}

    with open(ds / "meta" / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for i, p in enumerate(uniq):
            fh.write(json.dumps({"task_index": i, "task": p}, ensure_ascii=False) + "\n")

    episodes = load_episodes(ds)
    for e in episodes:
        idx = e["episode_index"]
        p = prompt_of_new_index.get(idx)
        if p is None:
            continue
        e["tasks"] = [p]
        ti = task_index[p]
        e["task_index"] = ti
        # 改 parquet 的 task_index 列
        pq_path = ds / "data" / f"chunk-{idx // 1000:03d}" / f"episode_{idx:06d}.parquet"
        if pq_path.exists():
            table = pq.read_table(pq_path)
            col = pa.array([ti] * table.num_rows, type=table.schema.field("task_index").type)
            table = table.set_column(table.schema.get_field_index("task_index"), "task_index", col)
            pq.write_table(table, pq_path)

    with open(ds / "meta" / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    update_total_tasks(ds, len(uniq))
    return {"mode": "per_episode", "tasks": uniq}


def update_total_tasks(dataset, n):
    p = Path(dataset) / "meta" / "info.json"
    if not p.exists():
        return
    info = json.loads(p.read_text(encoding="utf-8"))
    info["total_tasks"] = n
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 校验

def validate(dataset):
    ds = Path(dataset)
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    n = int(info.get("total_episodes") or 0)
    parquet = list((ds / "data").rglob("*.parquet"))
    mp4 = list((ds / "videos").rglob("*.mp4"))
    cams = len([k for k in (info.get("features") or {}) if "images" in k])
    tasks = load_tasks(ds)
    problems = []
    if len(parquet) != n:
        problems.append(f"parquet 数 {len(parquet)} != total_episodes {n}")
    if cams and len(mp4) != n * cams:
        problems.append(f"mp4 数 {len(mp4)} != {n}×{cams}")
    if not tasks:
        problems.append("tasks.jsonl 为空")
    return {
        "total_episodes": n,
        "total_frames": info.get("total_frames"),
        "parquet_files": len(parquet),
        "mp4_files": len(mp4),
        "cameras": cams,
        "tasks": tasks,
        "problems": problems,
    }


# ---------------------------------------------------------------- 主流程

def main(argv=None):
    ap = argparse.ArgumentParser(description="按标注导出只含有效条目的训练子集")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--annotations", default=None,
                    help="标注 json；默认 <review-root>/annotations/<key>.json")
    ap.add_argument("--review-root", default=DEFAULT_REVIEW_ROOT)
    ap.add_argument("--name", default=None, help="输出子集目录名")
    ap.add_argument("--out-root", default=None, help="输出根目录；默认 <review-root>/exports")
    ap.add_argument("--official-scripts", default=DEFAULT_OFFICIAL_SCRIPTS)
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="把结果以 JSON 打到 stdout 最后一行")
    args = ap.parse_args(argv)

    ds = Path(args.dataset).resolve()
    key = ds.name
    ann_path = Path(args.annotations) if args.annotations else \
        Path(args.review_root) / "annotations" / f"{key}.json"
    annotations = load_annotations(ann_path)

    episodes = load_episodes(ds)
    n = len(episodes)
    valid, invalid, unmarked = split_marks(annotations, n)
    excluded = sorted(invalid + unmarked)

    result = {
        "dataset": str(ds),
        "annotations": str(ann_path),
        "episodes_total": n,
        "valid": len(valid),
        "invalid": len(invalid),
        "unmarked": len(unmarked),
        "will_exclude": len(excluded),
    }
    print(f"数据集 {ds}")
    print(f"  共 {n} 条：有效 {len(valid)} / 无效 {len(invalid)} / 未标记 {len(unmarked)}")
    print(f"  将排除 {len(excluded)} 条（只保留有效）")

    if not valid:
        result["error"] = "没有标记为「有效」的条目，拒绝导出空数据集"
        print(f"  [!] {result['error']}")
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        return 2

    if args.dry_run:
        print("\n[dry-run] 未做任何改动")
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{key}_valid_{ts}"
    out_root = Path(args.out_root) if args.out_root else Path(args.review_root) / "exports"
    target = out_root / name
    if target.exists():
        raise SystemExit(f"目标已存在: {target}")

    # 源数据集指纹，导出后复核，防止误改源数据
    src_before = len(load_episodes(ds))
    src_meta_before = {p.name: p.stat().st_size for p in (ds / "meta").iterdir()}

    # 1. 复制
    mode, msg = hardlink_copy(ds, target)
    print(f"  [1/4] {msg}")
    result["copy_mode"] = mode

    # 2. 删除被排除条目
    if excluded:
        script = Path(args.official_scripts) / "delete_wrong_episodes.py"
        if not script.exists():
            raise SystemExit(f"找不到官方脚本: {script}")
        cmd = [args.venv_python, str(script), "--dataset", str(target),
               "--episode-indices", *[str(i) for i in excluded]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"删除条目失败:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        print(f"  [2/4] 已删除 {len(excluded)} 条并重排索引")
    else:
        print("  [2/4] 无需删除（全部有效）")

    # 旧索引 -> 新索引：保留项按原顺序重排
    new_of_old = {old: new for new, old in enumerate(valid)}

    # 3. prompt
    default_prompt = annotations.get("default_prompt") or ""
    eps_ann = annotations.get("episodes") or {}
    overrides = {}
    for old in valid:
        p = (eps_ann.get(str(old)) or {}).get("prompt")
        if p:
            overrides[old] = p
    if not default_prompt:
        existing = load_tasks(target)
        default_prompt = existing[0]["task"] if existing else "task description"
    if overrides:
        pmap = {new_of_old[old]: p for old, p in overrides.items()}
        for new in new_of_old.values():
            pmap.setdefault(new, default_prompt)
        info = apply_prompt_map(target, pmap)
        print(f"  [3/4] prompt：统一「{default_prompt}」+ {len(overrides)} 条覆盖，共 {len(info['tasks'])} 个任务")
    else:
        info = apply_single_prompt(target, default_prompt)
        print(f"  [3/4] prompt：统一「{default_prompt}」")
    result["prompt"] = info
    result["default_prompt"] = default_prompt

    # 4. 校验
    v = validate(target)
    result.update(v)
    result["output"] = str(target)

    # 复核源数据集未被改动
    src_after = len(load_episodes(ds))
    src_meta_after = {p.name: p.stat().st_size for p in (ds / "meta").iterdir()}
    if src_after != src_before or src_meta_after != src_meta_before:
        print(f"  [!] 源数据集被改动！条目 {src_before} -> {src_after}，meta 大小变化 "
              f"{src_meta_before != src_meta_after}")
        result["source_modified"] = True
        v["problems"].append("源数据集被改动（硬链接污染）")
    else:
        result["source_modified"] = False

    print(f"  [4/4] 校验：{v['total_episodes']} 条 / {v['total_frames']} 帧 / "
          f"parquet {v['parquet_files']} / mp4 {v['mp4_files']}（{v['cameras']} 相机）")
    if v["problems"]:
        print("  [!] 问题：" + "; ".join(v["problems"]))
        result["ok"] = False
    else:
        result["ok"] = True
        print("  ✅ 导出完成")

    manifest = Path(target) / "export_manifest.json"
    manifest.write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset": str(ds),
        "annotations_file": str(ann_path),
        "kept_original_indices": valid,
        "excluded_original_indices": excluded,
        **result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  清单：{manifest}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
