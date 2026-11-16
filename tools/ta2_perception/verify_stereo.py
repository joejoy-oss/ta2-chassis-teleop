"""Independent stereo acceptance check.

Triangulates the printed ChArUco board with a saved calibration and reports the
metric scale factor (1.000 = correct), the plane residual and the board centre
distance. Run it with the board held still at a tape-measured position, move the
board by a known distance, run again: the difference of the reported distances
must match the tape.
"""
import argparse
import base64
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from core import Board, PAIRS, crop


def fetch(source):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(source.rstrip("/") + "/api/frame", timeout=3) as response:
        packet = json.load(response)
    image = cv2.imdecode(np.frombuffer(base64.b64decode(packet.pop("jpeg_b64")), np.uint8), cv2.IMREAD_COLOR)
    return packet, image


def similarity(source, target):
    """Least-squares similarity (scale, rotation, translation) mapping source onto target."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    cov = xt.T @ xs / len(source)
    u, sing, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, sign]) @ vt
    scale = float((sing * np.diag([1, 1, sign])).sum() / (xs ** 2).sum() * len(source))
    translation = mu_t - scale * rotation @ mu_s
    residual = float(np.sqrt(np.mean(np.sum((target - (scale * rotation @ source.T).T - translation) ** 2, axis=1))))
    return scale, residual


def measure(config, calib, pair):
    board = Board(config["board"])
    packet, image = fetch(config["source"])
    names = PAIRS[pair]
    left = crop(image, packet["regions"][names[0]])
    right = crop(image, packet["regions"][names[1]])
    a, b = board.corners(left), board.corners(right)
    if a is None or b is None:
        raise SystemExit("至少一只眼看不到足够的角点；请把整块板同时放进两眼画面")
    common, ia, ib = np.intersect1d(a[1], b[1], return_indices=True)
    if len(common) < 12:
        raise SystemExit(f"两眼共同角点只有 {len(common)} 个；请让整块板同时落在两眼画面内")
    K1, D1 = np.array(calib["K1"]), np.array(calib["D1"])
    K2, D2 = np.array(calib["K2"]), np.array(calib["D2"])
    R, T = np.array(calib["R"]), np.array(calib["T"])
    l = cv2.undistortPoints(a[0][ia].reshape(-1, 1, 2), K1, D1).reshape(-1, 2)
    r = cv2.undistortPoints(b[0][ib].reshape(-1, 1, 2), K2, D2).reshape(-1, 2)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])
    xyz = cv2.triangulatePoints(P1, P2, l.T, r.T)
    xyz = (xyz[:3] / xyz[3]).T
    obj = board.object_points(common)
    scale, residual = similarity(obj, xyz)
    centre = xyz.mean(0)
    # Plane normal from the triangulated points, for a sanity look at the pose.
    _, _, vt = np.linalg.svd(xyz - centre)
    normal = vt[2]
    tilt = float(np.degrees(np.arccos(abs(normal[2]))))
    return dict(corners=int(len(common)), scale=scale, residual_mm=residual * 1000, distance_mm=float(np.linalg.norm(centre)),
                size_mm=(int(left.shape[1]), int(left.shape[0])), tilt_deg=tilt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    p.add_argument("--pair", choices=PAIRS, default="head")
    p.add_argument("--runs", type=int, default=1, help="repeat N times for stability")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    path = Path(config["data_dir"]) / "calibration" / f"{args.pair}.json"
    calib = json.loads(path.read_text(encoding="utf-8"))
    print(f"pair {args.pair}  calibration {calib['version']}  board {calib['board']}")
    for i in range(args.runs):
        m = measure(config, calib, args.pair)
        print(f"run {i + 1}: corners {m['corners']:2d}  scale {m['scale']:.4f} ({(m['scale'] - 1) * 100:+.2f}%)  "
              f"plane rms {m['residual_mm']:.2f} mm  distance {m['distance_mm']:.1f} mm  tilt {m['tilt_deg']:.1f}deg")


if __name__ == "__main__":
    main()
