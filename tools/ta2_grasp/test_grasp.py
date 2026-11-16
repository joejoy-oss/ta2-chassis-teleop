"""抓取判定自测：合成序列（无需机器人/网络）+ 可选实拍夹具 + 可选实时冒烟。

  python3 test_grasp.py --selftest
  python3 test_grasp.py --fixtures /data/ta2/grasp/fixtures
  python3 test_grasp.py --live http://127.0.0.1:8097 --frames 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from grasp_core import (DEFAULTS, GraspJudge, contact_geometry, draw_overlay, jaw_geometry,
                        merge, red_mask, track_points)
import frames

SIZE = (400, 640)  # h, w


# --------------------------------------------------------------------------- 合成场景

def world_base(seed=0, size=(1200, 1600)):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (9, 9), 0)
    # 合成一块蓝色带纹理的「桌面」，让 table_mask 有可信参考（真实现场是蓝桌布）
    x0, y0, x1, y1 = 30, 230, 900, 1000
    base[y0:y1, x0:x1] = (150, 80, 40)
    for i in range(14):
        cv2.line(base, (x0 + 10, y0 + 20 + i * 50), (x1 - 10, y0 + 30 + i * 50), (120, 60, 28), 3)
    for i in range(10):
        cv2.circle(base, (x0 + 60 + i * 80, y0 + 120 + (i % 3) * 180), 18, (135, 70, 35), -1)
    return base


def make_frame(base, world_dx, bag_xy, bag_visible=True, size=SIZE):
    """world_dx 模拟相机相对世界移动；bag_xy 是沙包在「当前画面」里的左上角。"""
    h, w = size
    x0 = 40 + int(round(world_dx))
    y0 = 30
    frame = base[y0:y0 + h, x0:x0 + w].copy()
    if bag_visible:
        x, y = int(bag_xy[0]), int(bag_xy[1])
        cv2.rectangle(frame, (x, y), (x + 190, y + 230), (30, 30, 160), -1)
        # 织物纹理：几条深色斜线，给 LK 提供可跟踪角点
        for i in range(6):
            cv2.line(frame, (x + 12, y + 20 + i * 36), (x + 180, y + 8 + i * 36), (25, 25, 110), 2)
        cv2.rectangle(frame, (x, y), (x + 190, y + 230), (10, 10, 90), 2)
    return frame


def run_sequence(judge, frames_list, dt=0.1):
    out = []
    ts = 1000.0
    for frame in frames_list:
        out.append(judge.update(frame, ts=ts, seq=len(out)))
        ts += dt
    return out


def test_red_mask():
    gray = np.full((400, 640, 3), 45, np.uint8)
    bag = gray.copy()
    cv2.rectangle(bag, (380, 40), (580, 280), (30, 30, 170), -1)
    ok = red_mask(bag)
    assert ok["ok"], ok
    assert 0.10 < ok["area_ratio"] < 0.30, ok["area_ratio"]

    dot = gray.copy()
    cv2.circle(dot, (320, 200), 10, (30, 30, 200), -1)
    assert not red_mask(dot)["ok"], "小红点（瓶盖）不应被当成沙包"

    line = gray.copy()
    cv2.rectangle(line, (40, 320), (600, 324), (30, 30, 200), -1)
    assert not red_mask(line)["ok"], "细长红胶带不应被当成沙包"
    print("[ok] red_mask")


def test_jaw_and_contact():
    geo = jaw_geometry([[250, 300], [250, 340], [390, 300], [390, 340]], (640, 400))
    assert abs(geo["closed_gap_px"] - 140.0) < 1.0, geo
    # 只夹角部：沙包大部分在通道上方，只探进来 20px
    mask = np.zeros((400, 640), np.uint8)
    cv2.rectangle(mask, (300, 120), (380, 320), 255, -1)
    contact = contact_geometry(mask, geo)
    assert contact is not None and contact["touching"], contact
    assert contact["partial"], contact

    # 完全夹在通道里
    mask2 = np.zeros((400, 640), np.uint8)
    cv2.rectangle(mask2, (255, 290), (385, 350), 255, -1)
    contact2 = contact_geometry(mask2, geo)
    assert contact2 is not None and not contact2["partial"], contact2
    print("[ok] jaw/contact")


def test_tracking():
    base = world_base(seed=3)
    a = base[0:400, 0:640]
    b = base[0:400, 5:645]
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    mask = np.full((400, 640), 255, np.uint8)
    pts = cv2.goodFeaturesToTrack(gray_a, 80, 0.01, 8, mask=mask).reshape(-1, 2).astype(np.float32)
    disp, good, _ = track_points(gray_a, gray_b, pts)
    assert np.count_nonzero(good) >= 20, np.count_nonzero(good)
    median = float(np.median(disp[good][:, 0]))
    assert -7.0 < median < -2.5, median
    print("[ok] tracking")


def test_judge_sequences():
    base = world_base(seed=7)
    cfg = {"judge": {"window_s": 0.8, "excite_px": 1.5, "still_px": 1.2, "move_px": 2.5}}
    h, w = SIZE

    # 1) 抓住：世界动、沙包在画面里不动
    judge = GraspJudge(cfg)
    seq = [make_frame(base, 4 * i, (410, 40)) for i in range(14)]
    results = run_sequence(judge, seq)
    assert results[-1]["state"] in ("held", "held_partial"), results[-1]["state"]
    assert results[-1]["success"] is True
    assert results[-1]["confidence"] > 0.5, results[-1]["confidence"]
    print("[ok] judge/held  state=%s conf=%.2f world=%.1f bag=%.1f" % (
        results[-1]["state"], results[-1]["confidence"],
        results[-1]["features"]["world_flow_px"] or -1,
        results[-1]["features"]["bag_flow_px"] or -1))

    # 2) 没抓住：沙包跟着世界一起动
    judge = GraspJudge(cfg)
    seq = [make_frame(base, 4 * i, (410 + 4 * i, 40)) for i in range(14)]
    results = run_sequence(judge, seq)
    assert results[-1]["state"] == "not_held", results[-1]["state"]
    assert results[-1]["success"] is False
    print("[ok] judge/not_held state=%s conf=%.2f" % (results[-1]["state"], results[-1]["confidence"]))

    # 3) 静止：没有运动激励，不猜
    judge = GraspJudge(cfg)
    seq = [make_frame(base, 0, (410, 40)) for _ in range(15)]
    results = run_sequence(judge, seq)
    assert results[-1]["state"] == "target_in_view", results[-1]["state"]
    assert results[-1]["success"] is None
    print("[ok] judge/target_in_view")

    # 3b) 相机没动、沙包自己在动（有人用手拿着沙包晃）-> 不是夹爪带着它
    judge = GraspJudge(cfg)
    seq = [make_frame(base, 0, (410 + 4 * i, 40)) for i in range(14)]
    results = run_sequence(judge, seq)
    assert results[-1]["state"] == "not_held", results[-1]["state"]
    print("[ok] judge/not_held_static_camera")

    # 3c) 软沙包只夹住一角：世界在动、沙包在腕图里也摆动，但头部视角被抬离桌面
    judge = GraspJudge(cfg)
    seq = [make_frame(base, 4 * i, (410 + 3 * i, 40)) for i in range(14)]
    ts, results = 1000.0, []
    for frame in seq:
        results.append(judge.update(frame, ts=ts, seq=len(results), head_lift=12.0))
        ts += 0.1
    assert results[-1]["state"] in ("held", "held_partial"), results[-1]["state"]
    print("[ok] judge/held_by_head_lift")

    # 4) 抓住后滑落：先 held，再世界静止、沙包向下滑
    judge = GraspJudge(cfg)
    held = [make_frame(base, 4 * i, (410, 40)) for i in range(12)]
    slip = [make_frame(base, 4 * 11, (410, 40 + 4 * i)) for i in range(1, 8)]
    results = run_sequence(judge, held + slip)
    states = [r["state"] for r in results]
    assert "held" in states, states
    assert results[-1]["state"] == "slipping", states
    print("[ok] judge/slipping states=%s" % ",".join(states[-6:]))

    # 5) 没有目标
    judge = GraspJudge(cfg)
    seq = [make_frame(base, 4 * i, (410, 40), bag_visible=False) for i in range(8)]
    results = run_sequence(judge, seq)
    assert results[-1]["state"] == "no_target", results[-1]["state"]
    print("[ok] judge/no_target")

    # 6) JSON 可序列化 + 必填字段
    payload = json.dumps(results[-1], ensure_ascii=False, allow_nan=False)
    assert json.loads(payload)["state"] == "no_target"
    print("[ok] json")


def test_fixtures(directory):
    import pathlib
    root = pathlib.Path(directory)
    images = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg")])
    if not images:
        print("[skip] 没有夹具：%s" % directory)
        return
    held, empty, wrong = 0, 0, []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        bag = red_mask(image)
        name = path.name.lower()
        if "held" in name:
            if bag["ok"]:
                held += 1
            else:
                wrong.append(str(path))
        elif "empty" in name or "floor" in name:
            if not bag["ok"]:
                empty += 1
            else:
                wrong.append(str(path))
    print("[fixtures] held 命中 %d，empty 正确 %d，错判 %d" % (held, empty, len(wrong)))
    for item in wrong[:5]:
        print("  ! %s" % item)
    assert not wrong, "夹具错判：%s" % wrong[:3]


def test_live(source, count):
    packet, image = frames.fetch(source)
    print("[live] %s  %dx%d seq=%s  regions=%s" %
          (source, packet["width"], packet["height"], packet.get("seq"),
           ",".join(sorted((packet.get("regions") or {}).keys()))))
    judge = GraspJudge({"side": "right"})
    last_seq = None
    seen = 0
    while seen < count:
        packet, image = frames.fetch(source)
        if packet.get("seq") == last_seq:
            time.sleep(0.05)
            continue
        last_seq = packet.get("seq")
        wrist = frames.extract_wrist(packet, image, "right")
        result = judge.update(wrist, ts=float(packet["received_at"]), seq=last_seq)
        seen += 1
        print("  seq=%-6s state=%-15s world=%s bag=%s area=%s" % (
            last_seq, result["state"], result["features"]["world_flow_px"],
            result["features"]["bag_flow_px"], result["features"]["area_ratio"]))
        if seen < count:
            time.sleep(0.12)


def replay_captures(root, config_path=None, jaw_path=None):
    """把采集包按顺序喂回判定器，打印每条的判决时间线（离线复核用）。"""
    import pathlib
    root = pathlib.Path(root)
    config = {}
    if config_path and pathlib.Path(config_path).exists():
        config = json.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    jaw = None
    if jaw_path and pathlib.Path(jaw_path).exists():
        jaw = json.loads(pathlib.Path(jaw_path).read_text(encoding="utf-8"))
    rows = []
    for directory in sorted([p for p in root.iterdir() if p.is_dir()]):
        images = sorted((directory / "wrist").glob("*.jpg"))
        if len(images) < 5:
            continue
        judge = GraspJudge(config, jaw=jaw)
        states = []
        for index, path in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                continue
            result = judge.update(image, ts=1000.0 + index * 0.1, seq=index)
            states.append(result["state"])
        trans = [(i, s) for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
        meta = {}
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        rows.append((directory.name, meta.get("scenario"), meta.get("human_label"), trans,
                     states[-1] if states else None))
    for name, scenario, label, trans, final in rows:
        print("%-32s scenario=%s label=%-9s final=%-15s transitions=%s" %
              (name, scenario, label, final, trans))
    return rows


def replay_dataset(root, episodes, fps_target=10.0, crop="left", config_path=None,
                   save_strip=None, quiet=False):
    """把 LeRobot 数据集里的右腕视频（1280x400 双目并排）切内眼、降采样后回放。

    用来离线验证「真机抓住」分支：视频里是红色沙包真实被抓起的片段。
    episodes=None 时跑 meta/episodes.jsonl 里的全部 episode。
    """
    import pathlib
    root = pathlib.Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 45.0))
    step = max(1, int(round(fps / float(fps_target))))
    if not episodes:
        rows = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        episodes = [int(row["episode_index"]) for row in rows]
    config = {}
    if config_path and pathlib.Path(config_path).exists():
        config = json.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    video_dir = root / "videos" / "chunk-000" / "observation.images.right_color"
    head_dir = root / "videos" / "chunk-000" / "observation.images.head_camera"
    head_red = merge(DEFAULTS["red"], DEFAULTS["red_head"])
    summary = []
    for episode in episodes:
        path = video_dir / ("episode_%06d.mp4" % int(episode))
        head_path = head_dir / ("episode_%06d.mp4" % int(episode))
        if not path.exists():
            print("ep %s 缺少视频 %s" % (episode, path))
            continue
        cap = cv2.VideoCapture(str(path))
        head_cap = cv2.VideoCapture(str(head_path)) if head_path.exists() else None
        judge = GraspJudge(config)
        states, kept, index = [], 0, 0
        tiles = []
        head_hist = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hok, hframe = head_cap.read() if head_cap is not None else (False, None)
            if index % step == 0:
                height, width = frame.shape[:2]
                eye = frame[:, :width // 2] if crop == "left" else frame[:, width // 2:]
                ts = 1000.0 + kept * step / fps
                head_lift = None
                if hok and hframe is not None:
                    hb = red_mask(hframe, head_red)
                    head_hist.append((ts, hb["centroid"][1] if hb["ok"] else None))
                    head_hist = [row for row in head_hist if ts - row[0] <= 2.5]
                    ys = [row[1] for row in head_hist if row[1] is not None]
                    if len(ys) >= 4:
                        head_lift = float(np.median(ys[:3]) - np.median(ys[-3:]))
                result = judge.update(eye, ts=ts, seq=kept, head_lift=head_lift)
                states.append((index, result["state"], result["features"].get("world_flow_px"),
                               result["features"].get("bag_flow_px"),
                               result["features"].get("area_ratio"),
                               result["features"].get("head_lift_px")))
                if save_strip and len(tiles) < 6 and kept % 12 == 0:
                    overlay = draw_overlay(eye, {**result, "_bag": getattr(judge, "last_bag", None),
                                                  "_jaw": getattr(judge, "last_jaw", None)})
                    tiles.append(overlay)
                kept += 1
            index += 1
        cap.release()
        if head_cap is not None:
            head_cap.release()
        names = [s[1] for s in states]
        transitions = [(i, s) for i, s in enumerate(names) if i == 0 or s != names[i - 1]]
        held = names.count("held") + names.count("held_partial")
        first_held = next((i for i, name in enumerate(names) if name in ("held", "held_partial")), None)
        flows_w = [s[2] for s in states if s[2] is not None]
        flows_b = [s[3] for s in states if s[3] is not None]
        lifts = [s[5] for s in states if s[5] is not None]
        summary.append({"episode": int(episode), "states": names, "transitions": transitions,
                        "held_frames": held, "first_held": first_held,
                        "final": names[-1] if names else None,
                        "world_max": max(flows_w) if flows_w else None,
                        "head_lift_max": max(lifts) if lifts else None})
        if not quiet:
            print("ep %-4d 原帧=%-4d 采样=%-3d transitions=%s" % (episode, index, kept, transitions))
            print("      held 帧=%d/%d (%.1fs) 首帧=%s  world_max=%s bag_max=%s head_lift_max=%s" % (
                held, kept, held * step / fps,
                None if first_held is None else round(first_held * step / fps, 1),
                round(max(flows_w), 2) if flows_w else None,
                round(max(flows_b), 2) if flows_b else None,
                round(max(lifts), 1) if lifts else None))
        if save_strip and tiles and episode == episodes[0]:
            strip = np.vstack([cv2.resize(t, (640, 400)) for t in tiles])
            cv2.imwrite(str(save_strip), strip)
            print("      已保存叠加条 %s" % save_strip)
    held_eps = [row for row in summary if row["first_held"] is not None]
    times = [row["first_held"] * step / fps for row in held_eps]
    print("[dataset] episodes=%d 出现 held=%d (%.0f%%) 首帧中位=%.1fs 范围=%s..%s" % (
        len(summary), len(held_eps), 100.0 * len(held_eps) / max(1, len(summary)),
        float(np.median(times)) if times else -1,
        round(min(times), 1) if times else None, round(max(times), 1) if times else None))
    if quiet:
        for row in summary:
            if row["first_held"] is None:
                print("  [no-held] ep=%-4d final=%-14s world_max=%s head_lift_max=%s" % (
                    row["episode"], row["final"], row["world_max"], row["head_lift_max"]))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--fixtures", default=None)
    parser.add_argument("--live", default=None)
    parser.add_argument("--captures", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--episodes", default="54,65,43,84,108,9,79,69,101,116")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps-target", type=float, default=10.0)
    parser.add_argument("--crop", default="left", choices=["left", "right", "full"])
    parser.add_argument("--strip", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--jaw", default=None)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if not (args.selftest or args.fixtures or args.live or args.captures or args.dataset):
        args.selftest = True
    if args.selftest:
        test_red_mask()
        test_jaw_and_contact()
        test_tracking()
        test_judge_sequences()
    if args.fixtures:
        test_fixtures(args.fixtures)
    if args.captures:
        replay_captures(args.captures, args.config, args.jaw)
    if args.dataset:
        episodes = None if args.all else [int(x) for x in args.episodes.split(",") if x.strip()]
        replay_dataset(args.dataset, episodes, args.fps_target, args.crop,
                       args.config, args.strip, quiet=args.quiet)
    if args.live:
        test_live(args.live, args.frames)
    print("全部通过")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("FAIL:", exc, file=sys.stderr)
        sys.exit(1)
