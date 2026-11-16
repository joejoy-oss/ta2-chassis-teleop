"""YOLO26 head-camera inference and offline evaluation."""

import argparse
import base64
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset import CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ta2_grasp.frames import extract_head, fetch

MAX_AGE = 1.0


@dataclass(frozen=True)
class Frame:
    id: str
    received: float
    image: np.ndarray
    jpeg: bytes

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def fresh(self):
        return 0 <= time.time() - self.received <= MAX_AGE

    def payload(self):
        return {"frame_id": self.id, "width": self.width, "height": self.height,
                "jpeg": base64.b64encode(self.jpeg).decode("ascii")}


def read_frame(source, previous=None):
    packet, composite = fetch(source, timeout=0.8)
    identity = f'{packet["session_id"]}:{packet["seq"]}'
    received = float(packet["received_at"])
    if not 0 <= time.time() - received <= MAX_AGE:
        raise ValueError("上游相机画面已过期")
    if identity == previous:
        return None
    image = extract_head(packet, composite)
    if image is None:
        raise ValueError("帧源没有头顶左眼画面")
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 87])
    if not ok:
        raise ValueError("无法编码头顶图像")
    return Frame(identity, received, image, jpeg.tobytes())


class Detector:
    def __init__(self, path):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        self.model = YOLO(str(path), task="detect")
        names = self.model.names
        if tuple(names[i] for i in range(len(names))) != CLASSES:
            raise ValueError("权重的类别或顺序与四类现场道具不一致")
        self.predict(np.zeros((640, 640, 3), dtype=np.uint8), 0.5)

    def predict(self, image, confidence):
        result = self.model.predict(image, device="cpu", imgsz=640, conf=confidence, verbose=False)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        rects = boxes.xyxyn.cpu().tolist()
        ids = boxes.cls.int().cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        return [{"class_id": label, "xyxy": [min(1.0, max(0.0, v)) for v in rect], "confidence": score}
                for label, rect, score in zip(ids, rects, scores)]


class Camera:
    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self.frames = deque(maxlen=12)
        self.camera_error = "正在等待头顶相机"
        self.inference_error = None
        self.model, self.model_id, self.result = None, None, None
        self.confidence = 0.5
        self.generation = 0
        self.stop = threading.Event()
        for target in (self.read_loop, self.predict_loop):
            threading.Thread(target=target, daemon=True).start()

    def read_loop(self):
        previous = None
        while not self.stop.is_set():
            start = time.monotonic()
            try:
                frame = read_frame(self.source, previous)
                if frame is not None:
                    previous = frame.id
                    with self.lock:
                        self.frames.append(frame)
                        self.camera_error = None
            except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
                with self.lock:
                    self.camera_error = str(exc)
            self.stop.wait(max(0.01, 0.18 - (time.monotonic() - start)))

    def predict_loop(self):
        previous = None
        while not self.stop.is_set():
            with self.lock:
                frame = self.frames[-1] if self.frames else None
                model, confidence, generation = self.model, self.confidence, self.generation
            if frame is None or not frame.fresh or model is None or (generation, frame.id) == previous:
                self.stop.wait(0.035)
                continue
            previous = (generation, frame.id)
            try:
                boxes = model.predict(frame.image, confidence)
                with self.lock:
                    if generation == self.generation:
                        self.result = (frame, boxes)
                        self.inference_error = None
            except Exception as exc:
                with self.lock:
                    if generation == self.generation:
                        self.result = None
                        self.inference_error = str(exc)
                self.stop.wait(0.25)

    def set_model(self, model, model_id, confidence):
        with self.lock:
            self.model, self.model_id, self.confidence = model, model_id, confidence
            self.generation += 1
            self.result = None
            self.inference_error = None

    def frame(self, identity):
        with self.lock:
            return next((frame for frame in reversed(self.frames) if frame.id == identity and frame.fresh), None)

    def snapshot(self, since):
        with self.lock:
            latest = self.frames[-1] if self.frames else None
            state, note, frame, boxes = "camera_offline", self.camera_error, None, []
            if latest is not None and latest.fresh:
                frame = latest
                if not self.model:
                    state, note = "model_missing", None
                elif self.inference_error:
                    state, note = "inference_error", self.inference_error
                elif self.result and self.result[0].fresh:
                    frame, boxes = self.result
                    state, note = "ready", None
                else:
                    state, note = "loading", None
            if frame is None:
                return {"state": state, "note": note, "boxes": [], "frame_id": None}
            revision = f"{frame.id}:{self.generation}:{state}"
            return {"state": state, "note": note, "boxes": boxes, "revision": revision, "frame_id": frame.id,
                    "age_ms": round(1000 * (time.time() - frame.received)),
                    **(frame.payload() if revision != since else {})}


def iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / ((first[2] - first[0]) * (first[3] - first[1]) +
                           (second[2] - second[0]) * (second[3] - second[1]) - intersection)


def evaluate(model, path, confidence):
    images = sorted((path / "images" / "test").glob("*.jpg"))
    if not images:
        raise ValueError("这个训练快照没有已标注的测试图片")
    totals, hits = [0] * len(CLASSES), [0] * len(CLASSES)
    negatives = negative_false_positives = 0
    for path_image in images:
        lines = (path / "labels" / "test" / f"{path_image.stem}.txt").read_text().splitlines()
        truth = []
        for line in lines:
            label, x, y, w, h = map(float, line.split())
            label = int(label)
            truth.append((label, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
            totals[label] += 1
        boxes = model.predict(cv2.imread(str(path_image)), confidence)
        if not truth:
            negatives += 1
            negative_false_positives += len(boxes)
        for box in sorted(boxes, key=lambda item: -item["confidence"]):
            candidates = [(iou(box["xyxy"], rect), index) for index, (label, rect) in enumerate(truth)
                          if label == box["class_id"]]
            if candidates:
                overlap, index = max(candidates)
                if overlap >= 0.5:
                    hits[box["class_id"]] += 1
                    truth.pop(index)
    return {"test_images": len(images), "negative_images": negatives, "negative_false_positives": negative_false_positives,
            "classes": {name: {"hits": hits[i], "total": totals[i], "recall": hits[i] / totals[i] if totals[i] else None}
                        for i, name in enumerate(CLASSES)}}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path)
    source.add_argument("--test-dataset", type=Path)
    parser.add_argument("--source", default="http://127.0.0.1:8097")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = Detector(args.model)
    if args.test_dataset:
        result = evaluate(model, args.test_dataset, args.confidence)
    elif args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            parser.error("无法读取图片")
        result = model.predict(image, args.confidence)
    else:
        frame = read_frame(args.source)
        result = {"frame_id": frame.id, "boxes": model.predict(frame.image, args.confidence)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
