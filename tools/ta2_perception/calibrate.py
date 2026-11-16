"""ChArUco board, paired capture, hold-out stereo validation and hand-eye tooling."""
import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from core import Board, PAIRS, crop, digest, rigid, transform


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fetch(source):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(source.rstrip("/") + "/api/frame", timeout=3) as response:
        packet = json.load(response)
    if not 0 <= time.time() - packet["received_at"] <= .5:
        raise ValueError("视频帧过期")
    image = cv2.imdecode(np.frombuffer(base64.b64decode(packet.pop("jpeg_b64")), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (packet["height"], packet["width"]):
        raise ValueError("图像尺寸与元数据不符")
    return packet, image


def board_output(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    b = Board(config["board"])
    sx, sy = config["board"]["squares_x"], config["board"]["squares_y"]
    image = b.board.generateImage((sx * 200, sy * 200), marginSize=0)
    cv2.imwrite(str(directory / "charuco.png"), image)
    png = base64.b64encode(cv2.imencode(".png", image)[1]).decode()
    w, h = sx * config["board"]["square_m"] * 1000, sy * config["board"]["square_m"] * 1000
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w+40}mm" height="{h+40}mm" viewBox="0 0 {w+40} {h+40}">'
           f'<rect width="100%" height="100%" fill="white"/><image x="20" y="15" width="{w}" height="{h}" href="data:image/png;base64,{png}"/>'
           f'<path d="M20 {h+23}h100m-100 -2v4m100 -4v4" stroke="black" stroke-width=".3"/>'
           f'<text x="20" y="{h+32}" font-size="3">100 mm verification line. Print 100%, landscape. No fit-to-page.</text></svg>')
    (directory / "charuco-print.svg").write_text(svg, encoding="utf-8")
    atomic_json(directory / "board.json", config["board"])


MIN_COMMON_CORNERS = 12
MIN_BOARD_COVER = 0.25


def check_shot(left, right, board, directory):
    """Reject a stereo shot that cannot become a trustworthy calibration sample.

    Small or centre-clustered boards leave focal length and distortion unidentifiable,
    which shows up later as large epipolar residuals rather than as a failed solve.
    """
    found = {}
    for eye, image in (("left", left), ("right", right)):
        corners = board.corners(image)
        if corners is None:
            raise ValueError(f"{eye} 眼没识别到足够的标记角点；请调整板姿态后停稳")
        found[eye] = corners
    common = np.intersect1d(found["left"][1], found["right"][1])
    if len(common) < MIN_COMMON_CORNERS:
        raise ValueError(f"左右眼共同角点只有 {len(common)} 个（至少 {MIN_COMMON_CORNERS} 个）；"
                         "请让整块板同时落在两眼画面内")
    pixels = found["left"][0]
    height, width = left.shape[:2]
    cover = float(max(np.ptp(pixels[:, 0]) / width, np.ptp(pixels[:, 1]) / height))
    if cover < MIN_BOARD_COVER:
        raise ValueError(f"标定板只占画面 {cover * 100:.0f}%（至少 {MIN_BOARD_COVER * 100:.0f}%）；"
                         "请靠近到约 0.4–0.5 m 再拍")
    previous = sorted(Path(directory).glob("*_left.png"))
    if previous:
        reference = cv2.imread(str(previous[0]))
        if reference is not None and reference.shape != left.shape:
            raise ValueError(f"分辨率与已拍帧不一致（{reference.shape[1]}x{reference.shape[0]} → "
                             f"{width}x{height}）；请清空该目录或换一组重拍")
    return len(common), cover


def capture(config, pair, split, scene):
    packet, image = fetch(config["source"])
    directory = Path(config["data_dir"]) / "calibration_captures" / pair / split
    directory.mkdir(parents=True, exist_ok=True)
    identity = f'{time.time_ns()}_{packet["seq"]}'
    board = Board(config["board"])
    names = PAIRS[pair]
    shots = {"left": crop(image, packet["regions"][names[0]]),
             "right": crop(image, packet["regions"][names[1]])}
    check_shot(shots["left"], shots["right"], board, directory)
    # Only save complete paired captures that passed the shot check.
    for eye in ("left", "right"):
        cv2.imwrite(str(directory / f"{identity}_{eye}.png"), shots[eye])
    atomic_json(directory / f"{identity}.json", {**packet, "pair": pair, "scene": scene, "board": config["board"]})
    return identity


def correspondences(directory, board):
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        left = cv2.imread(str(path.with_name(path.stem + "_left.png")))
        right = cv2.imread(str(path.with_name(path.stem + "_right.png")))
        if left is None or right is None or left.shape != right.shape:
            continue
        a, b = board.corners(left), board.corners(right)
        if a is None or b is None:
            continue
        common, ia, ib = np.intersect1d(a[1], b[1], return_indices=True)
        if len(common) < 8:
            continue
        rows.append((board.object_points(common).astype(np.float32), a[0][ia].astype(np.float32),
                     b[0][ib].astype(np.float32), meta, left.shape[1::-1]))
    return rows


def solve(config, pair, measured_square_mm):
    if abs(measured_square_mm / 1000 - config["board"]["square_m"]) > .0003:
        raise ValueError("实测方格尺寸与配置差异超过 0.3 mm；先更正 board.square_m 和 marker_m")
    board = Board(config["board"])
    root = Path(config["data_dir"]) / "calibration_captures" / pair
    training, validation = [correspondences(root / s, board) for s in ("train", "validation")]
    if len(training) < 25 or len(validation) < 5:
        raise ValueError(f"需要至少 25 组训练＋5 组独立验证，目前 {len(training)}＋{len(validation)}")
    rows = training + validation
    versions = {r[3]["regions_version"] for r in rows}
    sizes = {r[4] for r in rows}
    if len(versions) != 1 or len(sizes) != 1 or any(r[3]["board"] != config["board"] for r in rows):
        raise ValueError("采集过程中裁剪、尺寸或标定板规格发生变化")
    if {r[3]["session_id"] + str(r[3]["seq"]) for r in training} & {r[3]["session_id"] + str(r[3]["seq"]) for r in validation}:
        raise ValueError("验证集包含训练帧")
    objects, left, right = [[r[k] for r in training] for k in range(3)]
    size = training[0][4]
    err1, K1, D1, rv1, tv1 = cv2.calibrateCamera(objects, left, size, None, None)
    err2, K2, D2, _, _ = cv2.calibrateCamera(objects, right, size, None, None)
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rv1])
    diversity = float(np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    if diversity < 20:
        raise ValueError("板姿态变化不足 20 度；补拍倾斜视角以避免退化标定")
    # Per-eye calibration seeds the intrinsics; the stereo step then refines both eyes
    # jointly. Fixing the per-eye intrinsics leaves a systematic vertical residual
    # (epipolar p95 1.9 px on this head pair) that joint refinement removes (0.4 px).
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(objects, left, right, K1, D1, K2, D2, size,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    epi, metric = [], []
    for obj, a, b, _, _ in validation:
        aa = cv2.undistortPoints(a.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
        bb = cv2.undistortPoints(b.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
        epi.extend(np.abs(aa[:, 1]-bb[:, 1]).tolist())
        xyz = cv2.triangulatePoints(P1, P2, aa.T, bb.T)
        xyz = (xyz[:3] / xyz[3]).T
        d1 = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
        d2 = np.linalg.norm(obj[:, None] - obj[None, :], axis=2)
        metric.extend(np.abs(d1-d2)[np.triu_indices(len(obj), 1)].tolist())
    ep95, metric95 = float(np.percentile(epi, 95)), float(np.percentile(metric, 95))
    passed = bool(max(err1, err2, rms) <= 1 and ep95 <= 1 and metric95 <= .005
                  and .001 < np.linalg.norm(T) < .5 and P2[0, 3] < 0 and abs(P2[1, 3]) < abs(P2[0, 3]))
    data = {"pair": pair, "size": size, "board": config["board"], "lens_model": "pinhole_brown",
            "regions_version": next(iter(versions)), "measured_square_m": measured_square_mm/1000,
            "validation": {"passed": passed, "train_pairs": len(training), "validation_pairs": len(validation),
               "rms_px": float(rms), "epipolar_p95_px": ep95, "metric_error_m": metric95,
               "normal_range_deg": diversity, "note": "实测板尺度下的内部几何验证；仍需独立桌面测点验收"}}
    for k, v in zip(("K1", "D1", "K2", "D2", "R", "T"), (K1, D1, K2, D2, R, T)):
        data[k] = v.tolist()
    data["version"] = digest(data)
    atomic_json(Path(config["data_dir"]) / "calibration" / f"{pair}.json", data)
    return data["validation"]


def handeye(path, output):
    """Offline eye-in-hand solver; input explicitly uses base<-gripper and camera<-table."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    train = [s for s in data["samples"] if s["split"] == "train"]
    test = [s for s in data["samples"] if s["split"] == "validation"]
    if len(train) < 12 or len(test) < 5 or not data.get("robot_frame"):
        raise ValueError("需要至少 12 个训练姿态、5 个验证姿态及明确 robot_frame")
    for s in train + test:
        if not rigid(s["T_base_gripper"]) or not rigid(s["T_camera_table"]):
            raise ValueError("位姿必须是以米为单位的刚体变换")
    poses = [np.array(s["T_base_gripper"]) for s in train]
    rotations = [cv2.Rodrigues(poses[0][:3, :3].T @ p[:3, :3])[0].ravel() for p in poses[1:]]
    singular = np.linalg.svd(rotations, compute_uv=False)
    if singular[1] < .15:
        raise ValueError("手眼运动退化：需要围绕至少两个不同轴的明显旋转")
    targets = [np.array(s["T_camera_table"]) for s in train]
    R, t = cv2.calibrateHandEye([p[:3, :3] for p in poses], [p[:3, 3] for p in poses],
                               [p[:3, :3] for p in targets], [p[:3, 3] for p in targets], method=cv2.CALIB_HAND_EYE_PARK)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.ravel()
    if not rigid(X):
        raise ValueError("手眼求解失败")
    estimates = [np.array(s["T_base_gripper"]) @ X @ np.array(s["T_camera_table"]) for s in train]
    # Closest rotation matrix to training mean.
    U, _, V = np.linalg.svd(np.mean([p[:3, :3] for p in estimates], axis=0))
    T = np.eye(4)
    T[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ V)]) @ V
    T[:3, 3] = np.mean([p[:3, 3] for p in estimates], axis=0)
    errors = [np.array(s["T_base_gripper"]) @ X @ np.array(s["T_camera_table"]) for s in test]
    trans95 = float(np.percentile([np.linalg.norm(p[:3, 3]-T[:3, 3]) for p in errors], 95))
    rot95 = float(np.percentile([np.linalg.norm(cv2.Rodrigues(T[:3, :3].T @ p[:3, :3])[0]) for p in errors], 95))
    record = {"T_robot_table": T.tolist(), "T_gripper_camera": X.tolist(), "robot_frame": data["robot_frame"],
              "validation": {"passed": trans95 <= .01 and rot95 <= np.deg2rad(2), "translation_p95_m": trans95,
                             "rotation_p95_rad": rot95}, "setup_id": data.get("setup_id"),
              "note": "需另行实测轴向和控制参考点；固定基座/升降/桌面工况有效"}
    atomic_json(output, record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("board"); q.add_argument("--out", required=True)
    q = sub.add_parser("capture"); q.add_argument("--pair", choices=PAIRS, required=True)
    q.add_argument("--split", choices=["train", "validation"], required=True); q.add_argument("--scene", required=True)
    q = sub.add_parser("solve"); q.add_argument("--pair", choices=PAIRS, required=True); q.add_argument("--measured-square-mm", type=float, required=True)
    q = sub.add_parser("handeye"); q.add_argument("--input", required=True); q.add_argument("--out", required=True)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.command == "board": board_output(config, args.out)
    elif args.command == "capture": print(capture(config, args.pair, args.split, args.scene))
    elif args.command == "solve": print(json.dumps(solve(config, args.pair, args.measured_square_mm), ensure_ascii=False))
    else: print(json.dumps(handeye(args.input, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
