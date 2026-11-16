"""腕部相机抓取判定（纯视觉、无机器人 IO、无网络）。

判据的物理前提：腕部相机与夹爪刚性固连。因此

* 沙包真的被夹住  -> 相机怎么动，沙包在腕图里都基本不动；
* 沙包留在桌上/掉落 -> 相机相对世界一动，沙包在腕图里就跟着动。

所以主判据是「画面里有没有运动激励 + 沙包自身是否静止」：
只有「世界在动、沙包不动」才判成功；「世界没动」时一律不猜（target_in_view）。
这和「画面上出现红色」是两件事 —— 沙包就在视野里但夹爪空合，会被判为没抓住。

本模块只依赖 numpy / opencv，所有函数可直接单测。
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

# 默认参数集中放这里，server 的 config.json 按字段覆盖。
DEFAULTS = {
    "side": "right",
    "red": {
        "h_lo": 0.0, "h_hi": 10.0,      # H 的双区间：H<=10
        "h2_lo": 170.0, "h2_hi": 180.0, # 或 H>=170
        "s_min": 90, "v_min": 50,
        "min_area_ratio": 0.005,        # 640x400 -> 1280 px
        "open_ksize": 3, "close_ksize": 7,
        "min_solidity": 0.45, "max_aspect": 3.5,
    },
    "red_head": {                       # 头部视角里沙包 ~40px，面积占比小得多
        "min_area_ratio": 0.0006,
        "min_solidity": 0.35,
        "max_aspect": 4.0,
    },
    "flow": {
        "max_bag_points": 60,
        "max_world_points": 80,
        "min_points": 6,
        "win": 21,
        "levels": 3,
        "fb_err": 1.0,
        "world_band": 0.78,   # y > band*H 视为夹爪/相机自身，不当作世界
        "edge_margin": 8,
    },
    # 桌面参考：蓝色桌布上的光流才是「相机相对世界在动」。腕图里机器人自己的
    # 夹爪/前臂占很大面积，若用通用背景做参考，光流会被刚体结构稀释成 ~0。
    "table": {
        "enabled": True,
        "h_lo": 95, "h_hi": 135, "s_min": 70, "v_min": 30,
        "min_area_ratio": 0.02, "max_points": 80,
    },
    "judge": {
        "window_s": 0.8,
        "excite_px": 0.8,     # 桌面光流中值 >= 这个值才算「相机在动」
        "still_px": 1.2,      # 沙包光流中值 <= 这个值算「不动」
        "move_px": 2.5,       # 沙包光流中值 >= 这个值算「在动」
        "head_lift_px": 8.0,  # 头部视角沙包抬升超过这个值：软沙包在爪里摆动也算夹住
        "head_window_s": 2.5,
        "area_still": [0.90, 1.10],
        "area_move": [0.78, 1.28],
        "center_still_px": 6.0,
        "center_move_px": 12.0,
        "hold_frames": 5,
        "lost_frames": 3,
        "quiet_frames": 12,   # 连续多少帧没有激励 -> target_in_view
        "slip_px": 1.2,       # HELD 期间向下的相对滑动
        "lost_grace": 3,
        "max_age_s": 0.6,
        "iou_still": 0.88,    # 沙包 mask 与上一帧 IoU 高 -> 在画面里没动
        "iou_move": 0.75,
    },
    "jaw": None,
}

STATE_SUCCESS = {
    "held": True,
    "held_partial": True,
    "not_held": False,
    "slipping": False,
    "no_target": False,
    "target_in_view": None,
    "stale": None,
    "error": None,
    "starting": None,
}

TERMINAL = ("held", "held_partial", "not_held", "slipping")


def merge(base, override):
    """按 dict 递归合并（override 优先），不改动入参。"""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def jpeg(image, quality=85):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return buf.tobytes()


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


# --------------------------------------------------------------------------- 红色沙包分割

def red_mask(image, red=None, *, allow_multi=False):
    """HSV 红色双区间 + 面积/实心度/长宽比门控。

    返回 dict：ok / mask / area / area_ratio / bbox / centroid / solidity /
    aspect / clipped / candidates / reason。
    红瓶盖（小）、红标签（小且方）、地面红胶带（细长）会被面积与长宽比挡掉。
    """
    red = merge(DEFAULTS["red"], red or {})
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    m1 = (hue >= red["h_lo"]) & (hue <= red["h_hi"])
    m2 = (hue >= red["h2_lo"]) & (hue <= red["h2_hi"])
    raw = ((m1 | m2) & (sat >= red["s_min"]) & (val >= red["v_min"])).astype(np.uint8) * 255
    k_open = max(3, int(red["open_ksize"]) | 1)
    k_close = max(1, int(red["close_ksize"]) | 1)
    if k_open > 1:
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    if k_close > 1:
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    min_area = red["min_area_ratio"] * w * h
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > red["max_aspect"]:
            continue
        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < red["min_solidity"]:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        clipped = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        candidates.append({
            "mask": component, "area": area, "area_ratio": area / float(w * h),
            "bbox": [x, y, bw, bh], "centroid": [float(centroids[i][0]), float(centroids[i][1])],
            "solidity": float(solidity), "aspect": float(aspect), "clipped": bool(clipped),
            "contour": contour,
        })
    if not candidates:
        return {"ok": False, "mask": None, "reason": "视野内没有红色沙包", "candidates": []}
    candidates.sort(key=lambda c: c["area"], reverse=True)
    best = candidates[0]
    if not allow_multi:
        candidates = candidates[:1]
    best = dict(best)
    best["ok"] = True
    best["reason"] = None
    best["candidates"] = [
        {k: v for k, v in c.items() if k not in ("mask", "contour")} for c in candidates[1:]
    ]
    return best


# --------------------------------------------------------------------------- 桌面参考

def table_mask(image, table=None):
    """蓝色桌布区域（HSV）。返回 (mask, 面积占比)；关掉或没有桌布时 mask=None。"""
    table = merge(DEFAULTS["table"], table or {})
    if not table.get("enabled", True):
        return None, 0.0
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] >= table["h_lo"]) & (hsv[..., 0] <= table["h_hi"]) &
            (hsv[..., 1] >= table["s_min"]) & (hsv[..., 2] >= table["v_min"])).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask, float(np.count_nonzero(mask)) / float(w * h)


# --------------------------------------------------------------------------- 光流

def feature_points(gray, mask, max_n, quality=0.01, min_distance=5):
    mask = (mask > 0).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return np.empty((0, 2), np.float32)
    pts = cv2.goodFeaturesToTrack(gray, int(max_n), quality, float(min_distance),
                                  mask=mask, blockSize=7)
    if pts is None:
        return np.empty((0, 2), np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def track_points(prev_gray, gray, points, win=21, levels=3, fb_err=1.0):
    """Lucas-Kanade + 前后向校验。返回 (位移 Nx2, 有效掩码, 当前点 Nx2)。"""
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), bool), np.empty((0, 2), np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None,
                                           winSize=(win, win), maxLevel=levels, criteria=criteria)
    if nxt is None:
        return np.empty((0, 2), np.float32), np.empty((0,), bool), np.empty((0, 2), np.float32)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None,
                                            winSize=(win, win), maxLevel=levels, criteria=criteria)
    if back is None:
        return np.empty((0, 2), np.float32), np.empty((0,), bool), np.empty((0, 2), np.float32)
    fb = np.linalg.norm(back.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_err)
    return (nxt.reshape(-1, 2) - points.reshape(-1, 2)), good, nxt.reshape(-1, 2)


def flow_stats(disp, good):
    if disp is None or len(disp) == 0 or good is None or not np.any(good):
        return None, None, 0
    d = disp[good]
    mag = np.linalg.norm(d, axis=1)
    return float(np.median(mag)), float(np.median(d[:, 1])), int(len(d))


# --------------------------------------------------------------------------- 夹爪通道

def jaw_geometry(closed_points, image_size):
    """由「空合」时两片夹爪内侧各 2 点（共 4 点）构造通道。

    约定点击顺序：先点一片指垫内侧两点，再点另一片两点；内部按平均 x 排序，
    所以点选顺序在左右两片之间可以对调。
    """
    pts = np.asarray(closed_points, np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("需要 4 个点（两片指垫内侧各两点）")
    a, b = pts[:2], pts[2:]
    if a[:, 0].mean() > b[:, 0].mean():
        a, b = b, a
    corridor = np.array([a[0], a[1], b[1], b[0]], np.float32)
    pad_a_mid = a.mean(axis=0)
    pad_b_mid = b.mean(axis=0)
    gap = float(np.linalg.norm(pad_a_mid - pad_b_mid))
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "pads": [a.tolist(), b.tolist()],
        "corridor": corridor.tolist(),
        "grasp_line": [pad_a_mid.tolist(), pad_b_mid.tolist()],
        "closed_gap_px": gap,
    }


def contact_geometry(mask, jaw, *, pad_margin=6.0):
    """沙包与夹爪通道的几何关系；jaw 缺失或尺寸不符时返回 None。"""
    if not jaw or mask is None:
        return None
    corridor = np.asarray(jaw.get("corridor", []), np.float32)
    if corridor.shape != (4, 2):
        return None
    area = float(np.count_nonzero(mask))
    if area <= 0:
        return None
    inside = np.zeros_like(mask)
    cv2.fillPoly(inside, [np.rint(corridor).astype(np.int32)], 255)
    overlap = float(np.count_nonzero(mask & inside)) / area
    line = np.asarray(jaw.get("grasp_line", []), np.float32)
    crosses = False
    pad_min = None
    if line.shape == (2, 2):
        # 抓取线两侧都要有沙包像素才算「横跨闭合线」
        x0, y0 = line[0]
        x1, y1 = line[1]
        dx, dy = x1 - x0, y1 - y0
        yy, xx = np.nonzero(mask)
        if len(xx):
            side = dx * (yy - y0) - dy * (xx - x0)
            crosses = bool(np.any(side > 0) and np.any(side < 0))
        # 轮廓到两片指垫的距离
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            dists = []
            for pad in jaw.get("pads", []):
                seg = np.asarray(pad, np.float32)
                for point in seg:
                    dists.append(float(np.min(np.linalg.norm(cnt - point, axis=1))))
            pad_min = min(dists) if dists else None
    touching = (overlap > 0.01) or (crosses and (pad_min is None or pad_min <= max(pad_margin, 2.0)))
    return {
        "overlap": overlap,
        "crosses_grasp_line": bool(crosses),
        "pad_distance_px": pad_min,
        "touching": bool(touching),
        "partial": bool(touching and overlap < 0.15),
    }


# --------------------------------------------------------------------------- 判定器

class GraspJudge:
    """逐帧喂入腕图，输出带迟滞的状态机判决。"""

    def __init__(self, config=None, jaw=None):
        cfg = merge(DEFAULTS, config or {})
        self.cfg = cfg
        self.red = cfg["red"]
        self.flow = cfg["flow"]
        self.judge = cfg["judge"]
        self.side = cfg.get("side", "right")
        self.jaw = jaw
        self._reset()

    def _reset(self):
        self.prev_gray = None
        self.prev_bag_pts = np.empty((0, 2), np.float32)
        self.prev_world_pts = np.empty((0, 2), np.float32)
        self.prev_mask = None
        self.hist = []
        self.state = "starting"
        self.still_run = 0
        self.move_run = 0
        self.quiet_run = 0
        self.gone_run = 0
        self.release_run = 0
        self.bag_motion_run = 0
        self.last_bag = None
        self.last_contact = None
        self.last_jaw = None
        self.reference = "world"
        self.sequence = 0

    def reset(self):
        self._reset()

    def update(self, image, ts=None, seq=0, head_lift=None):
        """喂一帧 BGR 腕图，返回判决 dict（不包含图片）。

        head_lift：头部视角里红沙包的竖直抬升量（px，向上为正）。软沙包只夹住
        一个角时会在爪里摆动，腕图里「相对静止」会失效，这时用「沙包确实被抬离
        桌面」作为附着证据。
        """
        started = time.perf_counter()
        ts = time.time() if ts is None else float(ts)
        self.sequence = int(seq)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        bag = red_mask(image, self.red)
        t_mask, t_ratio = table_mask(image, self.cfg.get("table"))
        table_ok = t_mask is not None and t_ratio >= self.cfg["table"]["min_area_ratio"]
        if bag["ok"]:
            self.last_bag = bag
            mask = bag["mask"]
            iou_prev = None
            if self.prev_mask is not None and self.prev_mask.shape == mask.shape:
                inter = np.count_nonzero(mask & self.prev_mask)
                union = np.count_nonzero(mask | self.prev_mask)
                iou_prev = float(inter) / float(union) if union else None
            self.prev_mask = mask.copy()
            kernel = np.ones((5, 5), np.uint8)
            bag_eroded = cv2.erode(mask, kernel)
            bag_pts = feature_points(gray, bag_eroded, self.flow["max_bag_points"])
            world_mask = np.full((h, w), 255, np.uint8)
            world_mask = cv2.bitwise_and(world_mask, cv2.bitwise_not(cv2.dilate(mask, kernel)))
            band_y = int(self.flow["world_band"] * h)
            world_mask[band_y:, :] = 0
            if self.jaw:
                corridor = np.asarray(self.jaw.get("corridor", []), np.float32)
                if corridor.shape == (4, 2):
                    cv2.fillPoly(world_mask, [np.rint(corridor).astype(np.int32)], 0)
            margin = int(self.flow["edge_margin"])
            world_mask[:margin, :] = 0
            world_mask[-margin:, :] = 0
            world_mask[:, :margin] = 0
            world_mask[:, -margin:] = 0
            if table_ok:
                world_pts = feature_points(gray, t_mask, self.flow["max_world_points"])
                self.reference = "table"
            else:
                world_pts = feature_points(gray, world_mask, self.flow["max_world_points"])
                self.reference = "world"
            bag_flow, bag_dy, bag_n = (None, None, 0)
            world_flow, world_dy, world_n = (None, None, 0)
            if self.prev_gray is not None:
                disp, good, cur = track_points(self.prev_gray, gray, self.prev_bag_pts,
                                               self.flow["win"], self.flow["levels"], self.flow["fb_err"])
                bag_flow, bag_dy, bag_n = flow_stats(disp, good)
                self.tracks_bag = (disp, good, self.prev_bag_pts) if len(disp) else (None, None, None)
                disp, good, cur = track_points(self.prev_gray, gray, self.prev_world_pts,
                                               self.flow["win"], self.flow["levels"], self.flow["fb_err"])
                world_flow, world_dy, world_n = flow_stats(disp, good)
                self.tracks_world = (disp, good, self.prev_world_pts) if len(disp) else (None, None, None)
            self.prev_bag_pts = bag_pts
            self.prev_world_pts = world_pts
        else:
            self.last_bag = None
            iou_prev = None
            self.prev_mask = None
            bag_pts = np.empty((0, 2), np.float32)
            if table_ok:
                world_pts = feature_points(gray, t_mask, self.flow["max_world_points"])
                self.reference = "table"
            else:
                world_pts = feature_points(gray, np.full((h, w), 255, np.uint8),
                                           self.flow["max_world_points"])
                self.reference = "world"
            bag_flow = bag_dy = world_flow = world_dy = None
            bag_n = world_n = 0
            self.tracks_bag = self.tracks_world = (None, None, None)
        self.prev_gray = gray

        self.hist.append({
            "ts": ts, "bag_flow": bag_flow, "bag_dy": bag_dy, "bag_n": bag_n,
            "world_flow": world_flow, "world_dy": world_dy, "world_n": world_n,
            "area": float(bag["area"]) if bag["ok"] else 0.0,
            "centroid": bag["centroid"] if bag["ok"] else None,
            "iou_prev": iou_prev,
        })
        window_s = float(self.judge["window_s"])
        while self.hist and ts - self.hist[0]["ts"] > window_s + 0.4:
            self.hist.pop(0)
        window = [e for e in self.hist if ts - e["ts"] <= window_s]
        if len(window) < 2:
            window = self.hist[-2:] if len(self.hist) >= 2 else self.hist

        wx = _median([e["world_flow"] for e in window])
        bf = _median([e["bag_flow"] for e in window])
        start = window[0] if window else None
        area_ratio = None
        center_shift = None
        if bag["ok"] and start and start["area"] > 0 and start["centroid"]:
            area_ratio = float(bag["area"]) / start["area"]
            center_shift = float(math.hypot(bag["centroid"][0] - start["centroid"][0],
                                            bag["centroid"][1] - start["centroid"][1]))
        contact = contact_geometry(bag["mask"], self.jaw) if bag["ok"] else None
        if contact is not None:
            contact["jaw_detected"] = True

        excited = wx is not None and wx >= self.judge["excite_px"]
        iou = _median([e.get("iou_prev") for e in window[-3:]])
        flow_still = bf is not None and bf <= self.judge["still_px"]
        # 软性沙包被夹住时可能在爪里轻微转动、被手指遮挡或出画，面积/质心会变；
        # 主判据只用「沙包在画面里的光流」，光流缺特征时才退回 mask IoU。
        mask_still = iou is not None and iou >= self.judge["iou_still"]
        bag_still = bool(bag["ok"] and (flow_still or (bf is None and mask_still)))
        moving_evidence = bool(
            (bf is not None and bf >= self.judge["move_px"]) or
            (area_ratio is not None and not (0.35 <= area_ratio <= 2.0)) or
            (center_shift is not None and center_shift >= 60.0)
        )
        bag_moving = bool(bag["ok"] and moving_evidence)
        attached_by_head = bool(
            bag["ok"] and head_lift is not None and float(head_lift) >= self.judge["head_lift_px"]
        )

        if not bag["ok"]:
            self.gone_run += 1
        else:
            self.gone_run = 0

        # ---- 状态机（只在这里改 self.state / 计数器） ----
        if self.state in ("starting", "stale", "error", "no_target"):
            self.state = "no_target" if not bag["ok"] else "target_in_view"
        if not bag["ok"] and self.state not in ("stale", "error"):
            if self.gone_run >= self.judge["lost_grace"]:
                self.state = "no_target"
            self.still_run = self.move_run = self.quiet_run = self.release_run = 0
        elif bag["ok"]:
            if excited:
                self.quiet_run = 0
                self.bag_motion_run = 0
                if bag_still or attached_by_head:
                    self.still_run += 1
                    self.move_run = 0
                elif bag_moving:
                    self.move_run += 1
                    self.still_run = 0
                else:
                    self.still_run = max(0, self.still_run - 1)
                    self.move_run = max(0, self.move_run - 1)
                if self.state in ("held", "held_partial"):
                    pass
                elif self.still_run >= self.judge["hold_frames"]:
                    self.state = "held"
                elif self.move_run >= self.judge["lost_frames"]:
                    self.state = "not_held"
            else:
                self.quiet_run += 1
                if bag_moving:
                    self.bag_motion_run += 1
                else:
                    self.bag_motion_run = 0
                if self.state in ("held", "held_partial"):
                    # 手臂静止时，沙包相对相机持续往下滑 -> 滑脱
                    if not attached_by_head and bf is not None and bag_dy is not None and \
                            bf >= self.judge["slip_px"] and bag_dy > 0.5 * bf:
                        self.release_run += 1
                    else:
                        self.release_run = 0
                    if self.release_run >= self.judge["lost_frames"]:
                        self.state = "slipping"
                        self.release_run = 0
                elif self.bag_motion_run >= self.judge["lost_frames"] and self.state != "slipping" \
                        and not attached_by_head:
                    # 相机没动、沙包自己在动：不可能是「刚体固连」的夹持。
                    # 但只有在「桌面参考可信」时才敢下这个结论；否则可能是相机在动、
                    # 只是视野里没有桌面可测（例如沙包把画面占满），只能说不确定。
                    if self.reference == "table" and world_n >= self.flow["min_points"]:
                        self.state = "not_held"
                    elif self.quiet_run >= 5 and self.state == "not_held":
                        self.state = "target_in_view"
                elif self.quiet_run >= self.judge["quiet_frames"] and \
                        self.state not in ("held", "held_partial", "not_held", "slipping"):
                    self.state = "target_in_view"
                elif bag_moving and self.state == "not_held":
                    pass
            if self.state in ("held", "held_partial"):
                if not bag["ok"]:
                    self.state = "not_held"
                elif bag_moving and not attached_by_head and bag_dy is not None and \
                        bag_dy > 0.5 * max(bf or 0, 0.01):
                    self.release_run += 1
                    if self.release_run >= self.judge["lost_frames"]:
                        self.state = "slipping"
                        self.release_run = 0
                else:
                    self.release_run = 0
            if self.state == "held":
                if contact is not None and contact.get("partial"):
                    self.state = "held_partial"
            if self.state == "held_partial" and contact is not None and not contact.get("partial"):
                self.state = "held"
            if self.state == "not_held" and self.still_run >= self.judge["hold_frames"]:
                self.state = "held"

        self.last_contact = contact
        self.last_jaw = self.jaw

        # 记录判决用的证据快照，供 attempt 报告与页面使用
        self.last_evidence = {
            "window_s": window_s,
            "world_flow_px": wx,
            "bag_flow_px": bf,
            "bag_dy_px": bag_dy,
            "bag_points": int(bag_n),
            "world_points": int(world_n),
            "area_ratio": area_ratio,
            "center_shift_px": center_shift,
            "mask_iou": iou,
            "head_lift_px": None if head_lift is None else round(float(head_lift), 1),
            "attached_by_head": bool(attached_by_head),
            "excited": bool(excited),
            "bag_still": bool(bag_still),
            "bag_moving": bool(bag_moving),
            "reference": getattr(self, "reference", "world"),
            "trusted_reference": bool(self.reference == "table" and world_n >= self.flow["min_points"]),
            "table_ratio": round(float(t_ratio), 3),
        }
        result = self.result(image, bag, contact, started=started, ts=ts)
        return result

    # ---- 结果 ----------------------------------------------------------
    def result(self, image, bag, contact, *, started=None, ts=None, stale_ms=None):
        cfg = self.judge
        state = self.state
        ev = getattr(self, "last_evidence", {})
        wx, bf = ev.get("world_flow_px"), ev.get("bag_flow_px")
        excite = cfg["excite_px"]
        overlap = None if contact is None else contact.get("overlap")
        if contact is None:
            contact_score = None
        else:
            contact_score = _clip01(1.0 - abs(0.35 - float(overlap)) / 0.65) if contact.get("touching") else 0.0
        attachment = None
        if bf is not None:
            attachment = _clip01(1.0 - float(bf) / max(1e-6, cfg["still_px"] * 1.5))
        excitation = None if wx is None else _clip01(float(wx) / max(1e-6, 2.0 * excite))
        no_slip = 1.0
        if state == "slipping":
            no_slip = 0.0
        elif state in ("held", "held_partial") and ev.get("bag_dy_px") is not None:
            no_slip = _clip01(1.0 - max(0.0, float(ev["bag_dy_px"])) / max(1e-6, cfg["slip_px"] * 2.0))
        confidence = 0.35
        if state in ("held", "held_partial"):
            confidence = 0.55 + 0.2 * (excitation or 0) + 0.12 * (contact_score or 0) + 0.08 * no_slip
            confidence = min(0.98, confidence)
            if exciter_missing(ev):
                confidence = min(confidence, 0.7)
            if ev.get("attached_by_head"):
                confidence = min(0.98, confidence + 0.08)
        elif state == "not_held":
            confidence = min(0.95, 0.6 + 0.2 * (excitation or 0) + 0.15 * _clip01((bf or 0) / 4.0))
        elif state == "slipping":
            confidence = 0.8
        elif state == "target_in_view":
            confidence = 0.35
        elif state == "no_target":
            confidence = 0.9
        reasons = self._reasons(state, bag, ev, contact)

        scores = {
            "target": 1.0 if bag["ok"] else 0.0,
            "contact": None if contact_score is None else round(contact_score, 3),
            "attachment": None if attachment is None else round(attachment, 3),
            "no_slip": round(no_slip, 3),
            "excitation": None if excitation is None else round(excitation, 3),
        }
        features = {
            "bag_area_ratio": round(float(bag["area_ratio"]), 4) if bag["ok"] else 0.0,
            "bag_bbox": bag["bbox"] if bag["ok"] else None,
            "bag_clipped": bag["clipped"] if bag["ok"] else None,
            "corridor_overlap": None if overlap is None else round(float(overlap), 3),
            "pad_distance_px": None if contact is None else (
                None if contact.get("pad_distance_px") is None else round(float(contact["pad_distance_px"]), 1)),
            "world_flow_px": None if wx is None else round(float(wx), 2),
            "bag_flow_px": None if bf is None else round(float(bf), 2),
            "bag_points": ev.get("bag_points"),
            "world_points": ev.get("world_points"),
            "area_ratio": None if ev.get("area_ratio") is None else round(float(ev["area_ratio"]), 3),
            "center_shift_px": None if ev.get("center_shift_px") is None else round(float(ev["center_shift_px"]), 1),
            "mask_iou": None if ev.get("mask_iou") is None else round(float(ev["mask_iou"]), 3),
            "head_lift_px": ev.get("head_lift_px"),
            "attached_by_head": ev.get("attached_by_head"),
            "jaw_calibrated": bool(self.jaw),
            "reference": ev.get("reference"),
            "trusted_reference": ev.get("trusted_reference"),
            "table_ratio": ev.get("table_ratio"),
        }
        result = {
            "state": state,
            "success": STATE_SUCCESS.get(state),
            "partial": True if state == "held_partial" else (False if state in ("held", "not_held") else None),
            "confidence": round(float(confidence), 3),
            "side": self.side,
            "scores": scores,
            "features": features,
            "reasons": reasons,
            "frame": {
                "seq": int(self.sequence),
                "age_ms": None if stale_ms is None else round(float(stale_ms), 1),
                "regions_version": None,
            },
            "latency_ms": None if started is None else round((time.perf_counter() - started) * 1000.0, 1),
            "updated_at": round(float(ts if ts is not None else time.time()), 3),
        }
        return result

    def _reasons(self, state, bag, ev, contact):
        r = []
        if bag["ok"]:
            r.append("红色沙包在视野内（占画面 %.1f%%%s）" %
                     (bag["area_ratio"] * 100.0, "，部分出画" if bag["clipped"] else ""))
        else:
            r.append("视野内没有红色沙包")
        if ev.get("excited"):
            r.append("画面有运动激励（%s光流中值 %.1f px/帧）" %
                     ("桌面" if ev.get("reference") == "table" else "背景",
                      ev.get("world_flow_px") or 0.0))
        elif bag["ok"] and state not in ("held", "held_partial"):
            r.append("画面没有明显运动：缺少运动激励，不猜是否夹住（请抬升 3–5cm 或转动腕部）")
        elif bag["ok"]:
            r.append("画面已静止：沿用刚才的运动结论，置信度不升")

        if state in ("held", "held_partial"):
            if ev.get("attached_by_head") and not ev.get("bag_still"):
                r.append("头部视角：沙包被抬离桌面 %.0f px（软沙包在爪里摆动也算夹住）" %
                         (ev.get("head_lift_px") or 0.0))
            elif ev.get("excited"):
                r.append("世界在动、沙包在腕图里基本不动 → 沙包跟着夹爪走（已夹住）")
            else:
                r.append("沙包与夹爪在腕图里都静止，且沙包仍留在夹爪处（沿用已夹住）")
            if contact is None:
                r.append("夹爪通道未标定，接触证据缺失（不影响成功判定）")
            elif contact.get("touching"):
                r.append("沙包进入夹爪通道（重叠 %.0f%%）" % (contact["overlap"] * 100.0))
            if state == "held_partial":
                r.append("局部夹持：只有一角/一条边进入夹爪通道")
        elif state == "not_held":
            if ev.get("excited"):
                r.append("世界在动、沙包也跟着动 → 沙包没有跟着夹爪（没夹住）")
            else:
                r.append("相机没动而沙包自己在动 → 不是夹爪带着它（没夹住/正在滑脱）")
            if contact is not None and not contact.get("touching"):
                r.append("沙包没有进入夹爪通道")
        elif state == "slipping":
            r.append("沙包相对夹爪向下滑移 → 正在滑脱")
        elif state == "target_in_view":
            r.append("结论：目标在视野内，但证据不足，判为不确定")
        elif state == "no_target":
            r.append("结论：视野内没有目标")
        return r


def exciter_missing(ev):
    return not ev.get("excited")


# --------------------------------------------------------------------------- 画图

def draw_overlay(image, analysis, cfg=None, tracks=None):
    """把 mask / 通道 / 光流画到腕图上。图上只用 ASCII + 数字（OpenCV 不带中文字体）。"""
    out = image.copy()
    h, w = out.shape[:2]
    bag = analysis.get("_bag")
    if bag and bag.get("ok") and bag.get("mask") is not None:
        contours, _ = cv2.findContours(bag["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (80, 255, 80), 2)
        x, y, bw, bh = bag["bbox"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (80, 255, 80), 1)
        cx, cy = int(bag["centroid"][0]), int(bag["centroid"][1])
        cv2.circle(out, (cx, cy), 4, (80, 255, 80), -1)
    jaw = analysis.get("_jaw")
    if jaw and len(jaw.get("corridor", [])) == 4:
        corridor = np.rint(np.asarray(jaw["corridor"], np.float32)).astype(np.int32)
        cv2.polylines(out, [corridor], True, (0, 220, 255), 2)
        for pad in jaw.get("pads", []):
            for point in pad:
                cv2.circle(out, (int(point[0]), int(point[1])), 3, (255, 220, 0), -1)
    state = analysis.get("state", "?")
    conf = analysis.get("confidence", 0.0)
    ev = analysis.get("features", {})
    color = (80, 255, 80) if state in ("held", "held_partial") else (
        (80, 80, 255) if state in ("not_held", "slipping", "no_target") else (200, 200, 200))
    cv2.rectangle(out, (0, 0), (w, 30), (28, 28, 28), -1)
    cv2.putText(out, "%s  %.2f" % (state.upper(), conf), (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(out, "world %.1f  bag %s  area %s" % (
        ev.get("world_flow_px") or -1,
        "-" if ev.get("bag_flow_px") is None else "%.1f" % ev["bag_flow_px"],
        "-" if ev.get("area_ratio") is None else "%.2f" % ev["area_ratio"]),
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    # 光流箭头：世界（蓝）与沙包（红），按 px/帧 方向画短箭头（×4 放大便于肉眼观察）
    if tracks:
        for key, color in (("world", (255, 160, 0)), ("bag", (60, 210, 255))):
            disp, good, pts = tracks.get(key) or (None, None, None)
            if disp is None or good is None or pts is None or not np.any(good):
                continue
            points = pts[good]
            displacement = disp[good]
            step = max(1, len(points) // 10)
            for index in range(0, len(points), step):
                x, y = int(points[index][0]), int(points[index][1])
                dx, dy = float(displacement[index][0]) * 4.0, float(displacement[index][1]) * 4.0
                if abs(dx) + abs(dy) < 1.0:
                    continue
                cv2.arrowedLine(out, (x, y), (int(x + dx), int(y + dy)), color, 1,
                                cv2.LINE_AA, tipLength=0.4)
    return out
