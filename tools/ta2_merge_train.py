#!/usr/bin/env python3
"""
ta2_merge_train.py — 把多个数据集「各自筛出的有效子集」合并成一个训练集

流程
    1. 对每个源数据集，按标注导出「只含有效条目」的子集（复用 ta2_export_subset）
    2. 调官方 merge_datasets.py 把子集合并
    3. 统一 prompt（tasks.jsonl / episodes.jsonl / parquet 的 task_index）
    4. 落到 <review-root>/train_sets/<name>/，写 train_manifest.json

约定
    * 每个源数据集只取 mark == valid 的条目
    * 必须在 venv python 下运行（要 pyarrow）
    * 源数据集只读，中间产物放在临时目录，成功后清理

用法
    /data/ta2/tools/venv/bin/python ta2_merge_train.py \
        --datasets /path/a /path/b --name train_all --prompt "抓取红色沙包"
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

DEFAULT_REVIEW_ROOT = "/data/ta2/review"
DEFAULT_CONVERT_ROOT = "/data/ta2/converted"
DEFAULT_VENV_PYTHON = "/data/ta2/tools/venv/bin/python"
DEFAULT_OFFICIAL = "/data/ta2/software/convert/rosbag_to_dataset_TA2/scripts"


def dataset_key(path, convert_root):
    try:
        rel = Path(path).resolve().relative_to(Path(convert_root).resolve())
        return str(rel).replace("/", "__")
    except ValueError:
        return Path(path).name


def find_annotations(review_root, ds_path, convert_root):
    base = Path(review_root) / "annotations"
    for cand in (base / f"{dataset_key(ds_path, convert_root)}.json",
                 base / f"{Path(ds_path).name}.json"):
        if cand.exists():
            return cand
    return None


def load_jsonl(p):
    rows = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unify_prompt(dataset, prompt):
    """把整个数据集的 task 统一成一条 prompt（含 parquet 的 task_index）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = Path(dataset)
    (ds / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": prompt}, ensure_ascii=False) + "\n",
        encoding="utf-8")

    eps = load_jsonl(ds / "meta" / "episodes.jsonl")
    changed = 0
    for e in eps:
        e["tasks"] = [prompt]
        e["task_index"] = 0
        idx = int(e["episode_index"])
        chunk = idx // 1000
        pq_path = ds / "data" / f"chunk-{chunk:03d}" / f"episode_{idx:06d}.parquet"
        if pq_path.exists():
            t = pq.read_table(pq_path)
            if "task_index" in t.column_names:
                field = t.schema.field("task_index")
                col = pa.array([0] * t.num_rows, type=field.type)
                t = t.set_column(t.schema.get_field_index("task_index"), "task_index", col)
                pq.write_table(t, pq_path)
                changed += 1
    with open(ds / "meta" / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    info_p = ds / "meta" / "info.json"
    if info_p.exists():
        info = json.loads(info_p.read_text(encoding="utf-8"))
        info["total_tasks"] = 1
        info_p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def validate(ds):
    ds = Path(ds)
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    n = int(info.get("total_episodes") or 0)
    parquet = list((ds / "data").rglob("*.parquet"))
    mp4 = list((ds / "videos").rglob("*.mp4"))
    cams = len([k for k in (info.get("features") or {}) if "images" in k])
    problems = []
    if len(parquet) != n:
        problems.append(f"parquet {len(parquet)} != episodes {n}")
    if cams and len(mp4) != n * cams:
        problems.append(f"mp4 {len(mp4)} != {n}x{cams}")
    return {"total_episodes": n, "total_frames": info.get("total_frames"),
            "parquet_files": len(parquet), "mp4_files": len(mp4),
            "cameras": cams, "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser(description="合并多个数据集的筛选结果为一个训练集")
    ap.add_argument("--datasets", nargs="+", required=True, help="源数据集路径")
    ap.add_argument("--name", required=True, help="训练集名字")
    ap.add_argument("--prompt", required=True, help="统一后的 prompt")
    ap.add_argument("--review-root", default=DEFAULT_REVIEW_ROOT)
    ap.add_argument("--convert-root", default=DEFAULT_CONVERT_ROOT)
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON)
    ap.add_argument("--official-scripts", default=DEFAULT_OFFICIAL)
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    review_root = Path(args.review_root)
    export_script = Path(__file__).resolve().parent / "ta2_export_subset.py"
    if not export_script.exists():
        export_script = Path("/data/ta2/tools/ta2_export_subset.py")
    merge_script = Path(args.official_scripts) / "merge_datasets.py"

    sources = []
    for ds in args.datasets:
        ds = str(Path(ds).resolve())
        ann = find_annotations(review_root, ds, args.convert_root)
        sources.append({"path": ds, "annotations": str(ann) if ann else None})

    print(f"源数据集 {len(sources)} 个，统一 prompt = 「{args.prompt}」")
    for s in sources:
        print(f"  {s['path']}  标注={s['annotations'] or '缺失'}")
    if any(not s["annotations"] for s in sources):
        msg = "有数据集缺少标注文件，无法导出"
        print("  [!] " + msg)
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
        return 2
    if args.dry_run:
        if args.json:
            print(json.dumps({"ok": True, "dry_run": True, "sources": sources}, ensure_ascii=False))
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="ta2_merge_", dir=str(review_root)))
    result = {"sources": sources, "name": args.name, "prompt": args.prompt,
              "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        # 1. 逐个导出有效子集
        subsets = []
        for s in sources:
            sub_name = Path(s["path"]).name
            print(f"\n[1/3] 导出 {sub_name} 的有效子集…")
            cmd = [args.venv_python, str(export_script),
                   "--dataset", s["path"], "--annotations", s["annotations"],
                   "--review-root", str(review_root), "--out-root", str(tmp),
                   "--name", sub_name, "--json"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-1500:]); print(r.stderr[-1500:])
                raise SystemExit(f"导出失败: {sub_name}")
            out = str(tmp / sub_name)
            if not (Path(out) / "meta" / "info.json").exists():
                raise SystemExit(f"导出产物缺失: {out}")
            # 清理 export_manifest，避免被 merge 当作数据集文件
            mf = Path(out) / "export_manifest.json"
            if mf.exists():
                mf.unlink()
            subsets.append(out)
            print(f"      -> {out}")

        # 2. 合并
        merged = tmp / "_merged"
        if len(subsets) == 1:
            print("\n[2/3] 只有一个数据集，跳过合并")
            merged = Path(subsets[0])
        else:
            print(f"\n[2/3] 合并 {len(subsets)} 个子集…")
            if not merge_script.exists():
                raise SystemExit(f"找不到 merge 脚本: {merge_script}")
            r = subprocess.run([args.venv_python, str(merge_script),
                                "--root", str(tmp), "--output", str(merged)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:]); print(r.stderr[-2000:])
                raise SystemExit("合并失败")
            print(f"      -> {merged}")

        # 3. 统一 prompt
        print(f"\n[3/3] 统一 prompt…")
        changed = unify_prompt(merged, args.prompt)
        print(f"      改写 {changed} 个 parquet 的 task_index")

        # 4. 落地
        train_root = review_root / "train_sets"
        train_root.mkdir(parents=True, exist_ok=True)
        final = train_root / args.name
        if final.exists():
            raise SystemExit(f"目标已存在: {final}")
        shutil.move(str(merged), str(final))

        v = validate(final)
        result.update(v)
        result["output"] = str(final)
        result["ok"] = not v["problems"]

        manifest = {
            "created_at": result["created_at"],
            "name": args.name,
            "prompt": args.prompt,
            "sources": [{"dataset": s["path"], "annotations": s["annotations"]} for s in sources],
            "per_source_episodes": {},
            **v,
        }
        for s in sources:
            sub = Path(s["path"]).name
            info = tmp / sub / "meta" / "info.json"
            if info.exists():
                manifest["per_source_episodes"][sub] = json.loads(
                    info.read_text(encoding="utf-8")).get("total_episodes")
        (final / "train_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n✅ 训练集: {final}")
        print(f"   {v['total_episodes']} 条 / {v['total_frames']} 帧 / "
              f"parquet {v['parquet_files']} / mp4 {v['mp4_files']}（{v['cameras']} 相机）")
        if v["problems"]:
            print("   [!] " + "; ".join(v["problems"]))
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 1
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"临时目录保留: {tmp}")


if __name__ == "__main__":
    sys.exit(main())
