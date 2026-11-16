#!/usr/bin/env python3
"""
lerobot_episodes.py — LeRobot v2.x 数据集体检 / 筛数据 / 抽帧速览

用法（在 B70 的 LeRobot 环境里，python3 直接跑）：

  1) 体检 + 生成待审清单
     python3 lerobot_episodes.py report  <dataset_root>  [--gripper-dims 14,15] [--csv review.csv]

  2) 生成每条的抽帧拼图（9 帧 3x3），人眼一次过几十条
     python3 lerobot_episodes.py sheets  <dataset_root>  --out sheets/  [--camera observation.images.head]

  3) 生成一个 review.html：缩略图 + 指标 + 点开播原视频
     python3 lerobot_episodes.py html    <dataset_root>  --out review.html

  4) 自测（不依赖数据集，验证指标算法）
     python3 lerobot_episodes.py selftest

设计原则：
  * 只读，不改数据集。
  * 自动指标用来「缩小人工review范围」，不能替代人眼。最终判据仍是视频。
  * 阈值都相对数据集自身分布算（中位数倍数），避免不同任务量纲不一致。
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

# Windows 控制台常是 GBK，输出中文/勾号会炸；模块加载即统一切到 UTF-8，
# 编不出的字符用 ? 代替。放在模块级，保证任何入口方式都不会炸。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# 纯函数：从 (state, action) 矩阵算单条 episode 的指标
# ----------------------------------------------------------------------------

def compute_metrics(state, action, fps, gripper_dims):
    """state/action: (T, D) float64。返回 dict。"""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    T, D = action.shape
    m = {
        "n_frames": int(T),
        "duration_s": round(T / fps, 3) if fps else np.nan,
        "action_dim": int(D),
        "state_dim": int(state.shape[1]) if state.ndim == 2 else 0,
        "nan_count": int(np.isnan(action).sum() + np.isnan(state).sum()),
        "inf_count": int(np.isinf(action).sum() + np.isinf(state).sum()),
    }
    if T < 3 or m["nan_count"] or m["inf_count"]:
        # 数据不可用时给中性值，避免除零/传播
        m.update({"speed_med": 0.0, "speed_p95": 0.0, "speed_max": 0.0,
                  "idle_ratio": 1.0, "jerk_max": 0.0, "jerk_med": 0.0,
                  "jerk_ratio": 0.0, "state_jump_max": 0.0})
        m["gripper_travel"] = [0.0] * len(gripper_dims)
        return m

    # 动作逐步变化量（指令层面的「有没有在动」）
    step = np.linalg.norm(np.diff(action, axis=0), axis=1)      # (T-1,)
    speed_med = float(np.median(step))
    speed_p95 = float(np.percentile(step, 95))
    speed_max = float(step.max())

    # 「呆住」占比：变化量低于典型动作的 2%
    eps = max(speed_p95 * 0.02, 1e-9)
    idle_ratio = float((step < eps).mean())

    # 抖动 / 撞桌：二阶差分
    jerk = np.linalg.norm(np.diff(action, n=2, axis=0), axis=1)
    jerk_med = float(np.median(jerk))
    jerk_max = float(jerk.max())
    jerk_ratio = jerk_max / jerk_med if jerk_med > 1e-12 else float("inf")

    # 状态瞬跳：丢帧 / 通信中断
    sjump = np.linalg.norm(np.diff(state, axis=0), axis=1) if state.shape[0] == T else np.zeros(1)
    state_jump_max = float(np.abs(sjump).max()) if sjump.size else 0.0

    # 夹爪行程：抓没抓
    gt = []
    for d in gripper_dims:
        if 0 <= d < D:
            col = action[:, d]
            gt.append(float(col.max() - col.min()))
        else:
            gt.append(0.0)

    m.update({
        "speed_med": round(speed_med, 6),
        "speed_p95": round(speed_p95, 6),
        "speed_max": round(speed_max, 6),
        "idle_ratio": round(idle_ratio, 4),
        "jerk_max": round(jerk_max, 6),
        "jerk_med": round(jerk_med, 6),
        "jerk_ratio": round(jerk_ratio, 3) if np.isfinite(jerk_ratio) else 999.0,
        "state_jump_max": round(state_jump_max, 6),
        "gripper_travel": [round(x, 5) for x in gt],
    })
    return m


def flag_episodes(records, grasp_min=0.05, short_frac=0.4, idle_max=0.5,
                  jerk_mult=20.0, jump_mult=10.0):
    """records: [{'episode':.., 'metrics':..}]. 就地加 flags/score，返回同列表。"""
    if not records:
        return records
    lens = np.array([r["metrics"]["n_frames"] for r in records], dtype=float)
    med_len = float(np.median(lens)) if lens.size else 0.0
    jr = np.array([r["metrics"]["jerk_ratio"] for r in records], dtype=float)
    med_jr = float(np.median(jr)) if jr.size else 0.0
    sj = np.array([r["metrics"]["state_jump_max"] for r in records], dtype=float)
    med_sj = float(np.median(sj)) if sj.size else 0.0

    for r in records:
        mm = r["metrics"]
        f = []
        if mm["nan_count"] or mm["inf_count"]:
            f.append("NAN")
        if med_len and mm["n_frames"] < short_frac * med_len:
            f.append("TOO_SHORT")
        if mm["idle_ratio"] > idle_max:
            f.append("IDLE")
        if len(mm["gripper_travel"]) and max(mm["gripper_travel"]) < grasp_min:
            f.append("NO_GRASP")
        if med_jr > 1e-9 and mm["jerk_ratio"] > jerk_mult * max(med_jr, 1.0):
            f.append("SPIKE")
        if med_sj > 1e-9 and mm["state_jump_max"] > jump_mult * med_sj:
            f.append("STATE_TELEPORT")
        r["flags"] = f
        # 可疑度：硬伤权重高，统计异常权重低
        w = {"NAN": 100, "NO_GRASP": 40, "IDLE": 30, "TOO_SHORT": 25,
             "SPIKE": 20, "STATE_TELEPORT": 20}
        r["score"] = sum(w.get(x, 10) for x in f)
    return records


# ----------------------------------------------------------------------------
# 读取数据集
# ----------------------------------------------------------------------------

def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_meta(root):
    info_path = os.path.join(root, "meta", "info.json")
    info = _read_json(info_path) if os.path.exists(info_path) else {}
    eps = []
    for name in ("episodes.jsonl", "episodes.json"):
        p = os.path.join(root, "meta", name)
        if os.path.exists(p):
            eps = _read_jsonl(p) if name.endswith("jsonl") else _read_json(p)
            break
    tasks = []
    for name in ("tasks.jsonl", "tasks.json"):
        p = os.path.join(root, "meta", name)
        if os.path.exists(p):
            tasks = _read_jsonl(p) if name.endswith("jsonl") else _read_json(p)
            break
    stats = []
    p = os.path.join(root, "meta", "episodes_stats.jsonl")
    if os.path.exists(p):
        stats = _read_jsonl(p)
    return info, eps, tasks, stats


def _col_to_2d(df, col):
    import numpy as np
    v = df[col].values
    if isinstance(v, np.ndarray) and v.ndim == 2:
        return v.astype(np.float64)
    return np.stack([np.asarray(x, dtype=np.float64) for x in v])


def load_episode_arrays(parquet_path):
    """返回 (state, action) 两个 (T,D) 数组。优先 pandas，退回 pyarrow。"""
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
    except ImportError:
        import pyarrow.parquet as pq
        df = pq.read_table(parquet_path).to_pandas()

    cols = list(df.columns)
    a_col = "action" if "action" in cols else next((c for c in cols if "action" in c), None)
    s_col = ("observation.state" if "observation.state" in cols
             else next((c for c in cols if "state" in c), None))
    if a_col is None or s_col is None:
        raise KeyError(f"找不到 action/state 列，实际列: {cols}")
    return _col_to_2d(df, s_col), _col_to_2d(df, a_col)


def discover_episode_files(root):
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    return files


def episode_index_from_name(path):
    """从 episode_000123.mp4 / .parquet 里取 123。

    注意：必须先去掉扩展名，否则 '.mp4' 里的 '4' 会被当成集号。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


# ----------------------------------------------------------------------------
# 命令：report
# ----------------------------------------------------------------------------

def cmd_report(args):
    root = args.root
    info, eps_meta, tasks, stats = load_meta(root)
    files = discover_episode_files(root)
    if not files:
        print(f"[!] 在 {root}/data 下没找到 parquet。确认路径是不是 LeRobot 数据集根目录。")
        return 1

    fps = args.fps or info.get("fps") or 45
    gripper_dims = [int(x) for x in args.gripper_dims.split(",")] if args.gripper_dims else []

    task_of = {}
    for e in eps_meta:
        idx = e.get("episode_index", e.get("episode_id"))
        t = e.get("tasks") or e.get("task")
        if isinstance(t, list):
            t = ",".join(map(str, t))
        task_of[idx] = t

    print("=" * 78)
    print(f"数据集: {os.path.abspath(root)}")
    print(f"  fps={fps}  episodes(parquet)={len(files)}  meta_episodes={len(eps_meta)}")
    if info:
        print(f"  total_frames={info.get('total_frames')}  total_episodes={info.get('total_episodes')}")
        feats = info.get("features", {})
        for k, v in feats.items():
            shape = v.get("shape")
            print(f"  feature {k:32s} dtype={v.get('dtype')} shape={shape}")
    print("=" * 78)

    records = []
    for p in files:
        ep = episode_index_from_name(p)
        try:
            state, action = load_episode_arrays(p)
        except Exception as ex:
            print(f"[!] ep{ep:03d} 读取失败: {ex}")
            records.append({"episode": ep, "file": p,
                            "metrics": {"n_frames": 0, "nan_count": 1, "inf_count": 0,
                                        "idle_ratio": 1.0, "jerk_ratio": 0.0,
                                        "state_jump_max": 0.0, "gripper_travel": [],
                                        "duration_s": 0.0, "action_dim": 0, "state_dim": 0,
                                        "speed_med": 0, "speed_p95": 0, "speed_max": 0,
                                        "jerk_max": 0, "jerk_med": 0}})
            continue
        dims = gripper_dims or ([action.shape[1] - 2, action.shape[1] - 1]
                                if action.shape[1] >= 2 else [])
        m = compute_metrics(state, action, fps, dims)
        records.append({"episode": ep, "file": p, "task": task_of.get(ep), "metrics": m})

    # 自动识别夹爪维（若用户没指定，用最后一维做参考再报一次）
    if not gripper_dims:
        print(f"[i] 未指定 --gripper-dims，默认用 action 最后两维 "
              f"{records[0]['metrics']['action_dim']-2},{records[0]['metrics']['action_dim']-1} 当夹爪。")

    records = flag_episodes(records, grasp_min=args.grasp_min,
                            idle_max=args.idle_max, jerk_mult=args.jerk_mult)

    records.sort(key=lambda r: (-r["score"], r["episode"]))
    print(f"\n{'ep':>5} {'frames':>7} {'sec':>7} {'idle%':>6} {'spike':>6} "
          f"{'grip':>7} {'score':>5}  flags / task")
    print("-" * 78)
    for r in records:
        m = r["metrics"]
        grip = max(m["gripper_travel"]) if m["gripper_travel"] else float("nan")
        print(f"{r['episode']:>5} {m['n_frames']:>7} {m['duration_s']:>7.2f} "
              f"{m['idle_ratio']*100:>5.1f}% {m['jerk_ratio']:>6.1f} "
              f"{grip:>7.3f} {r['score']:>5}  {','.join(r['flags']) or '-':<28} {r.get('task') or ''}")

    bad = [r for r in records if r["score"] > 0]
    print("-" * 78)
    print(f"共 {len(records)} 条，疑似有问题 {len(bad)} 条；其余 {len(records)-len(bad)} 条进人工复核。")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["episode", "flags", "score", "n_frames", "duration_s",
                        "idle_ratio", "jerk_ratio", "state_jump_max",
                        "gripper_travel", "speed_med", "speed_p95", "speed_max",
                        "task", "file"])
            for r in records:
                m = r["metrics"]
                w.writerow([r["episode"], ";".join(r["flags"]), r["score"],
                            m["n_frames"], m["duration_s"], m["idle_ratio"],
                            m["jerk_ratio"], m["state_jump_max"],
                            ";".join(map(str, m["gripper_travel"])),
                            m["speed_med"], m["speed_p95"], m["speed_max"],
                            r.get("task") or "", os.path.relpath(r["file"], root)])
        print(f"[✓] 待审清单已写: {args.csv}")
    return 0


# ----------------------------------------------------------------------------
# 命令：sheets / html（ffmpeg 抽帧拼图）
# ----------------------------------------------------------------------------

def pick_video_dirs(root):
    vids = sorted(glob.glob(os.path.join(root, "videos", "**", "*.mp4"), recursive=True))
    keys = {}
    for p in vids:
        key = os.path.basename(os.path.dirname(p))
        keys.setdefault(key, []).append(p)
    return keys


def ffprobe_duration(path):
    """用临时文件接收输出，避免依赖管道（更便于在受限环境里跑）。"""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w") as fh:
            subprocess.run(
                [exe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", path],
                stdout=fh, stderr=subprocess.DEVNULL, timeout=60)
        with open(tmp, "r") as fh:
            out = fh.read().strip()
        return float(out) if out else None
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def make_sheet(video, out_png, cols=3, rows=3, width=320):
    """抽 cols*rows 帧拼一张图。返回 True/False。"""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    n = cols * rows
    dur = ffprobe_duration(video) or 0.0
    if dur <= 0:
        return False
    rate = n / dur                       # 每秒抽帧数，使整条正好 n 帧
    vf = f"fps={rate:.6f},scale={width}:-2,tile={cols}x{rows}"
    cmd = [exe, "-y", "-v", "error", "-i", video, "-vf", vf,
           "-frames:v", "1", out_png]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=180)
        return r.returncode == 0 and os.path.exists(out_png)
    except Exception:
        return False


def cmd_sheets(args):
    root = args.root
    keys = pick_video_dirs(root)
    if not keys:
        print(f"[!] {root}/videos 下没找到 mp4")
        return 1
    print("视频流:")
    for k, v in keys.items():
        print(f"  {k:40s} {len(v)} 个 mp4")
    cam = args.camera or list(keys.keys())[0]
    if cam not in keys:
        print(f"[!] 没有 '{cam}'，可选: {list(keys.keys())}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    ok = fail = 0
    for v in sorted(keys[cam]):
        ep = episode_index_from_name(v)
        dst = os.path.join(args.out, f"ep{ep:06d}.png")
        if make_sheet(v, dst, width=args.width):
            ok += 1
        else:
            fail += 1
            print(f"[!] ep{ep:03d} 抽帧失败: {v}")
    print(f"[✓] 拼图 {ok} 张 -> {args.out}  (失败 {fail})")
    return 0


def cmd_html(args):
    root = args.root
    keys = pick_video_dirs(root)
    cam = args.camera or (list(keys.keys())[0] if keys else None)
    vids = {episode_index_from_name(p): p for p in (keys.get(cam, []) if cam else [])}
    _, eps_meta, _, _ = load_meta(root)
    task_of = {}
    for e in eps_meta:
        idx = e.get("episode_index", e.get("episode_id"))
        task_of[idx] = e.get("tasks") or e.get("task")

    rows = []
    for ep in sorted(vids):
        rel = os.path.relpath(vids[ep], os.path.dirname(os.path.abspath(args.out)))
        rel = rel.replace(os.sep, "/")          # HTML 里必须用正斜杠
        sheet = f"sheets/ep{ep:06d}.png"
        t = task_of.get(ep)
        if isinstance(t, list):
            t = ",".join(map(str, t))
        rows.append(f"""<tr><td>{ep}</td><td>{t or ''}</td>
        <td><img src="{sheet}" style="width:300px" onerror="this.style.opacity=.2"></td>
        <td><video src="{rel}" controls preload="none" style="width:240px"></video></td></tr>""")

    html = f"""<!doctype html><meta charset="utf-8"><title>LeRobot episode review</title>
<style>body{{font:14px/1.5 system-ui;margin:24px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:6px;vertical-align:top}}th{{background:#f5f5f5;position:sticky;top:0}}
img{{display:block}}</style>
<h2>LeRobot episode review — {os.path.abspath(root)}</h2>
<p>相机: <b>{cam}</b>　共 {len(rows)} 条。缩略图 = 9 帧均分；点视频可播。<br>
判定标准见 <code>筛数据作业单.md</code>。</p>
<table><tr><th>ep</th><th>task / prompt</th><th>9 帧速览</th><th>原始视频</th></tr>
{''.join(rows)}
</table>"""
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[✓] {args.out}  ({len(rows)} 条)")
    return 0


# ----------------------------------------------------------------------------
# 命令：selftest
# ----------------------------------------------------------------------------

def cmd_selftest(_args):
    rng = np.random.default_rng(0)
    fps, T, D = 45, 100, 16
    grip = [14, 15]

    def mk(kind):
        s = np.cumsum(rng.normal(0, 0.01, (T, D)), axis=0)
        a = s + rng.normal(0, 0.005, (T, D))
        a[:, 14] = np.linspace(0.0, 1.0, T)          # 夹爪有开合
        a[:, 15] = np.linspace(1.0, 0.2, T)
        if kind == "idle":
            a[:] = a[0]
        elif kind == "nograsp":
            a[:, 14] = 0.5
            a[:, 15] = 0.5
        elif kind == "nan":
            a[10, 3] = np.nan
        elif kind == "spike":
            a[50] += 5.0
        elif kind == "short":
            return s[:10], a[:10]
        return s, a

    recs = []
    for i, k in enumerate(["normal", "idle", "nograsp", "nan", "spike", "short"]):
        s, a = mk(k)
        recs.append({"episode": i, "metrics": compute_metrics(s, a, fps, grip)})
    recs = flag_episodes(recs, grasp_min=0.05, idle_max=0.5, jerk_mult=20.0)

    got = {r["episode"]: set(r["flags"]) for r in recs}
    exp = {0: set(), 1: {"IDLE"}, 2: {"NO_GRASP"}, 3: {"NAN"},
           4: {"SPIKE"}, 5: {"TOO_SHORT"}}
    ok = True
    for ep in exp:
        miss = exp[ep] - got[ep]
        print(f"ep{ep}: flags={sorted(got[ep])}  期望包含={sorted(exp[ep])}  "
              f"{'OK' if not miss else 'MISSING ' + str(sorted(miss))}")
        if miss:
            ok = False
    # 正常条不应被误标
    if got[0]:
        print(f"[!] 正常样本被误标: {sorted(got[0])}")
        ok = False
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LeRobot 数据集体检与筛数据")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("report", help="体检 + 待审清单")
    r.add_argument("root")
    r.add_argument("--fps", type=float, default=None)
    r.add_argument("--gripper-dims", default=None, help="如 14,15；默认最后两维")
    r.add_argument("--grasp-min", type=float, default=0.05, help="夹爪行程下限，低于判 NO_GRASP")
    r.add_argument("--idle-max", type=float, default=0.5, help="呆滞占比上限")
    r.add_argument("--jerk-mult", type=float, default=20.0, help="抖动倍数阈值")
    r.add_argument("--csv", default="episodes_review.csv")

    s = sub.add_parser("sheets", help="抽帧拼图")
    s.add_argument("root")
    s.add_argument("--out", default="sheets")
    s.add_argument("--camera", default=None)
    s.add_argument("--width", type=int, default=320)

    h = sub.add_parser("html", help="生成 review.html")
    h.add_argument("root")
    h.add_argument("--out", default="review.html")
    h.add_argument("--camera", default=None)

    sub.add_parser("selftest", help="指标算法自测")

    args = ap.parse_args()
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "sheets":
        return cmd_sheets(args)
    if args.cmd == "html":
        return cmd_html(args)
    if args.cmd == "selftest":
        return cmd_selftest(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
