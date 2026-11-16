"""Read-only perception geometry. All distances are metres; fail closed on missing evidence."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

PAIRS = {"head": ("head_left_eye", "head_right_eye"),
         "left_wrist": ("left_wrist_left_eye", "left_wrist_right_eye"),
         "right_wrist": ("right_wrist_left_eye", "right_wrist_right_eye")}
CLASSES = ["red_sandbag", "medicine_bottle", "reagent_bottle", "graduated_bottle"]
LABELS = ["红沙包", "药品瓶", "方试剂瓶", "刻度瓶"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(np.array(box) * [w, h, w, h]).astype(int)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("invalid crop")
    return frame[y1:y2, x1:x2].copy()


def transform(points, T):
    T = np.asarray(T, float)
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def rigid(T):
    a = np.asarray(T, float)
    return (a.shape == (4, 4) and np.isfinite(a).all()
            and np.allclose(a[3], [0, 0, 0, 1])
            and np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-4)
            and abs(np.linalg.det(a[:3, :3]) - 1) < 1e-4)


class Board:
    def __init__(self, spec):
        self.spec = spec
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]),
                                           spec["square_m"], spec["marker_m"], dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def corners(self, image):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) < 8:
            return None
        return corners.reshape(-1, 2), ids.reshape(-1)

    def object_points(self, ids):
        # OpenCV board y points down; flip y so table z faces camera above table.
        p = self.board.getChessboardCorners()[ids].copy()
        p[:, 1] *= -1
        return p

    def pose(self, image, K, D):
        found = self.corners(image)
        if found is None:
            return None
        pixels, ids = found
        obj = self.object_points(ids)
        ok, rvec, tvec = cv2.solvePnP(obj, pixels, K, D)
        if not ok:
            return None
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
        R = cv2.Rodrigues(rvec)[0]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, tvec.ravel()
        if error > 1.0 or tvec[2, 0] <= 0 or np.linalg.inv(T)[2, 3] <= 0:
            return None
        return T, error


class Stereo:
    def __init__(self, data, regions_version, pair, shapes):
        self.data = data
        if data.get("regions_version") != regions_version or data.get("pair") != pair:
            raise ValueError("裁剪版本或双目组与标定不符")
        if data.get("validation", {}).get("passed") is not True:
            raise ValueError("双目标定尚未通过独立验证")
        size = tuple(data["size"])
        if any(tuple(s) != size for s in shapes):
            raise ValueError("图像尺寸与标定不符")
        self.size = size
        self.K1, self.D1, self.K2, self.D2, self.R, self.T = [np.array(data[k], float) for k in ("K1", "D1", "K2", "D2", "R", "T")]
        if data.get("lens_model", "pinhole_brown") != "pinhole_brown":
            raise ValueError("当前仅接受已验证的针孔畸变模型；鱼眼参数不能混用")
        if not all(np.isfinite(a).all() for a in (self.K1, self.D1, self.K2, self.D2, self.R, self.T)):
            raise ValueError("标定含非有限数值")
        if self.K1.shape != (3, 3) or self.K2.shape != (3, 3) or min(self.K1[0, 0], self.K1[1, 1], self.K2[0, 0], self.K2[1, 1]) <= 0:
            raise ValueError("相机内参无效")
        if not 0.001 < np.linalg.norm(self.T) < 0.5:
            raise ValueError("双目基线无效")
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(self.K1, self.D1, self.K2, self.D2, size, self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if abs(P2[1, 3]) > abs(P2[0, 3]) or P2[0, 3] >= 0:
            raise ValueError("需要水平正视差双目；检查旋转与左右眼顺序")
        self.R1, self.R2, self.P1, self.P2 = R1, R2, P1, P2
        self.maps = [cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
                     for K, D, R, P in [(self.K1, self.D1, R1, P1), (self.K2, self.D2, R2, P2)]]
        n = min(192, max(16, (size[0] // 4 // 16) * 16))
        args = dict(numDisparities=n, blockSize=5, P1=8 * 25, P2=32 * 25,
                    uniquenessRatio=12, speckleWindowSize=80, speckleRange=2,
                    disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left = cv2.StereoSGBM_create(minDisparity=0, **args)
        self.right = cv2.StereoSGBM_create(minDisparity=-n, **args)
        self.n = n

    def rectify(self, left, right):
        return [cv2.remap(im, *m, cv2.INTER_LINEAR) for im, m in zip((left, right), self.maps)]

    def depth(self, left, right):
        l, r = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (left, right)]
        dl = self.left.compute(l, r).astype(np.float32) / 16
        dr = self.right.compute(r, l).astype(np.float32) / 16
        yy, xx = np.indices(dl.shape)
        xr = np.rint(xx - dl).astype(int)
        in_bounds = (xr >= 0) & (xr < dl.shape[1])
        dback = dr[yy, np.clip(xr, 0, dl.shape[1] - 1)]
        valid = in_bounds & (dl > 0.5) & (dback > -self.n - 0.5) & (np.abs(dl + dback) <= 1)
        points = cv2.reprojectImageTo3D(dl, self.Q)
        valid &= np.isfinite(points).all(axis=2) & (points[:, :, 2] > .08) & (points[:, :, 2] < 2.5)
        return points, valid, dl


def sandbag_geometry(mask, points, valid, T_table_camera, stereo):
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    keep = inner & valid
    if keep.sum() < 30 or keep.sum() / max(1, inner.sum()) < .25:
        return None
    # Q emits rectified-left coordinates, not original optical coordinates.
    table_points = transform(points[keep] @ stereo.R1, T_table_camera)
    table_points = table_points[(table_points[:, 2] > .003) & (table_points[:, 2] < .15)]
    if len(table_points) < 30:
        return None
    med = np.median(table_points, axis=0)
    extent = np.percentile(table_points[:, :2], 95, axis=0)-np.percentile(table_points[:, :2], 5, axis=0)
    if max(extent) > .14 or max(extent) < .01:
        return None
    radius = np.linalg.norm(table_points - med, axis=1)
    table_points = table_points[radius < max(.015, float(np.percentile(radius, 85)))]
    if len(table_points) < 20:
        return None
    med = np.median(table_points, axis=0)
    spread = 1.4826 * np.median(np.abs(table_points - med), axis=0)
    # Include a floor and calibration contribution; do not pretend pixels are independent.
    sigma = np.maximum(.004, spread / 3 + float(stereo.data["validation"].get("metric_error_m", .005)))
    eig, vectors = np.linalg.eigh(np.cov(table_points[:, :2].T))
    direction = vectors[:, -1]
    yaw = float(np.arctan2(direction[1], direction[0])) if eig[-1] > eig[0] * 1.3 else None
    # Prefer a measured interior point nearest the robust centre, not a hallucinated centre.
    candidate = table_points[np.argmin(np.linalg.norm(table_points[:, :2] - med[:2], axis=1))]
    return dict(position=med.tolist(), sigma=sigma.tolist(), surface_candidate=candidate.tolist(),
                yaw_rad=yaw, support_points=len(table_points), valid_fraction=float(keep.sum()/max(1, inner.sum())))


def ray_plane(pixels, K, T_table_camera):
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K).T
    rays = rays @ T_table_camera[:3, :3].T
    origin = T_table_camera[:3, 3]
    good = np.abs(rays[:, 2]) > 1e-5
    distance = np.full(len(pixels), np.nan)
    distance[good] = -origin[2] / rays[good, 2]
    good &= (distance > 0) & (distance < 4)
    return origin + rays * distance[:, None], good


def bottle_geometry(mask, K, T_table_camera, dimensions, contacts_verified=False):
    """Only approved upright/contact evidence enables a metric envelope.

    A lower image contour alone is not proof of table contact. Default is unknown.
    """
    if not dimensions or not contacts_verified:
        return None
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return None
    bottom = ys >= np.percentile(ys, 97)
    p, valid = ray_plane(np.column_stack([xs[bottom], ys[bottom]]), K, T_table_camera)
    p = p[valid]
    if len(p) < 5:
        return None
    center = np.median(p[:, :2], axis=0)
    radius = max(dimensions["width_m"], dimensions["depth_m"]) / 2 + .02
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    footprint = center + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    return {"footprint": footprint.tolist(), "height_m": dimensions["height_m"],
            "method": "verified_upright_contact_plane", "padding_m": .02}


def bottle_possible_region(polygon, K, T_table_camera, dimensions):
    """Conservative uncertain region, NOT a bottle centre or verified free-space boundary.

    Enclose image rays between the table and measured maximum object height, then
    expand by the object's maximum horizontal dimension for partial occlusion.
    """
    if not dimensions or not all(0 < dimensions.get(k, 0) < 1 for k in ("width_m", "depth_m", "height_m")):
        return None
    pixels = np.asarray(polygon, float)
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K).T @ T_table_camera[:3, :3].T
    origin = T_table_camera[:3, 3]
    if origin[2] <= dimensions["height_m"] + .02 or np.any(rays[:, 2] >= -1e-4):
        return None
    points = []
    for z in (0, dimensions["height_m"] + .02):
        distance = (z-origin[2]) / rays[:, 2]
        if np.any(distance > 4): return None
        points.extend((origin + distance[:, None] * rays)[:, :2])
    points = np.asarray(points)
    radius = max(dimensions["width_m"], dimensions["depth_m"]) + .02
    theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
    expanded = points[:, None, :] + radius*np.column_stack([np.cos(theta), np.sin(theta)])[None]
    hull = cv2.convexHull(expanded.reshape(-1, 2).astype(np.float32)).reshape(-1, 2)
    return {"footprint": hull.tolist(), "height_m": dimensions["height_m"],
            "method": "uncertain_silhouette_height_envelope", "contact_verified": False,
            "position_m": None, "padding_m": radius}


def fuse(measurements, max_separation=.03):
    """One target only. Ambiguity is handled by caller, never merge distinct bags."""
    if not measurements:
        return None, "no_depth"
    p = np.array([m["position"] for m in measurements])
    if np.max(np.linalg.norm(p[:, None] - p[None, :], axis=2)) > max_separation:
        return None, "view_conflict"
    sig = np.array([m["sigma"] for m in measurements])
    weights = 1 / np.maximum(sig, .004) ** 2
    mean = (p * weights).sum(axis=0) / weights.sum(axis=0)
    # Shared calibration errors are correlated: do not shrink below the best input.
    sigma = np.maximum(np.sqrt(1 / weights.sum(axis=0)), sig.min(axis=0))
    best = measurements[int(np.argmin(np.sum(sig, axis=1)))]
    return {"position": mean.tolist(), "sigma": sigma.tolist(), "sources": [m["source"] for m in measurements],
            "point": best["surface_candidate"], "yaw_rad": best["yaw_rad"]}, None


def expire(result, now=None, max_age=.5):
    result = json.loads(json.dumps(result))
    now = time.time() if now is None else now
    result["age_ms"] = max(0, (now - result.get("received_at", 0)) * 1000)
    if not 0 <= now - result.get("received_at", 0) <= max_age:
        result["state"] = "stale"
        result["grasp_candidates"] = []
        result["target"] = None
        for item in result.get("objects", []):
            item["position_table_m"] = item["position_robot_m"] = None
            item["obstacle"] = None
            item["reason"] = "stale"
    return result


class Detector:
    def __init__(self, model_path, device="cpu"):
        self.error, self.model, self.device = "未安装现场训练模型", None, device
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path, task="segment")
                if list(self.model.names.values()) != CLASSES:
                    raise ValueError("模型类别必须与现场四类标注一致；通用 COCO 模型不能直接当作沙包模型")
                self.error = None
            except Exception as e:
                self.model = None
                self.error = str(e)

    def predict(self, images):
        if self.model is None:
            return [[] for _ in images]
        # Exported OpenVINO model has static batch=1. Do not submit six as one batch.
        results = []
        for image in images:
            try:
                results.extend(self.model.predict(image, device=self.device, conf=.35, imgsz=640, verbose=False))
            except Exception:
                if self.device == "cpu": raise
                self.device = "cpu"
                results.extend(self.model.predict(image, device="cpu", conf=.35, imgsz=640, verbose=False))
        out = []
        for image, prediction in zip(images, results):
            detections = []
            if prediction.masks is not None:
                for polygon, box in zip(prediction.masks.xy, prediction.boxes):
                    cls, score = int(box.cls.item()), float(box.conf.item())
                    mask = np.zeros(image.shape[:2], np.uint8)
                    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
                    detections.append({"class_id": cls, "confidence": score, "mask": mask,
                                       "polygon": polygon.tolist()})
            out.append(detections)
        return out
