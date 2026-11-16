"""取帧：默认从 B70 视觉查看器（:8097）的 /api/frame 旁路取拼接图并裁剪。

只读、无机器人 IO。帧包字段（session_id / seq / received_at / regions）与
tools/ta2_perception/calibrate.py 的 fetch 一致，这样裁剪坐标始终跟着视觉查看器走。
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request

import cv2
import numpy as np

PAIRS = {
    "head": ("head_left_eye", "head_right_eye"),
    "left_wrist": ("left_wrist_left_eye", "left_wrist_right_eye"),
    "right_wrist": ("right_wrist_left_eye", "right_wrist_right_eye"),
}
# 腕部取「内眼」：左腕取右眼、右腕取左眼（与 regions.json 的 policy_mapping 一致）。
SIDE_PAIRS = {"right": "right_wrist", "left": "left_wrist"}
INNER_EYE = {"right": 0, "left": 1}   # 0=left_eye, 1=right_eye


def fetch(source, timeout=3.0):
    """返回 (packet, BGR image)。packet 里去掉 jpeg_b64，保留 regions/seq/received_at。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(source.rstrip("/") + "/api/frame", timeout=timeout) as response:
        packet = json.load(response)
    data = packet.pop("jpeg_b64", None)
    if not data:
        raise ValueError("上游没有画面（视觉查看器还没出帧）")
    image = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (packet.get("height"), packet.get("width")):
        raise ValueError("图像尺寸与元数据不符")
    return packet, image


def crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(np.asarray(box, float) * [w, h, w, h]).astype(int)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("非法裁剪框：%s" % (box,))
    return frame[y1:y2, x1:x2].copy()


def views(packet, side):
    """按 side 返回 {"wrist": 内眼裁剪, "wrist_outer": 外眼, "head": 左眼} 的名字与框。"""
    pair = SIDE_PAIRS.get(side)
    if pair is None:
        raise ValueError("side 只能是 right/left")
    names = PAIRS[pair]
    inner = names[INNER_EYE[side]]
    outer = names[1 - INNER_EYE[side]]
    regions = packet.get("regions") or {}
    out = {"wrist": (inner, regions.get(inner)), "wrist_outer": (outer, regions.get(outer))}
    head_name = PAIRS["head"][0]
    out["head"] = (head_name, regions.get(head_name))
    return out


def extract_wrist(packet, frame, side):
    spec = views(packet, side)
    if spec["wrist"][1] is None:
        raise ValueError("帧包里没有腕部视角 %s" % spec["wrist"][0])
    return crop(frame, spec["wrist"][1])


def extract_head(packet, frame):
    name, box = views(packet, "right")["head"]
    if box is None:
        return None
    return crop(frame, box)


def age_s(packet):
    return max(0.0, time.time() - float(packet.get("received_at") or 0.0))
