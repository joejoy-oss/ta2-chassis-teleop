"""Local capture store and repeatable YOLO Detect dataset exports."""

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path


CLASSES = ("red_sandbag", "medicine_bottle", "reagent_bottle", "graduated_bottle")
LABELS = ("红沙包", "药品瓶", "方试剂瓶", "刻度瓶")
SPLITS = ("train", "val", "test")


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def validated_boxes(boxes):
    if not isinstance(boxes, list) or len(boxes) > 100:
        raise ValueError("标注格式不正确")
    result = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("矩形框格式不正确")
        cls, rect = box.get("class_id"), box.get("xyxy")
        if type(cls) is not int or cls not in range(len(CLASSES)):
            raise ValueError("请选择一个有效类别")
        if not isinstance(rect, list) or len(rect) != 4 or any(
            type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in rect
        ):
            raise ValueError("矩形框坐标超出图片范围")
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            raise ValueError("矩形框需要有宽度和高度")
        result.append({"class_id": cls, "xyxy": rect})
    return result


class Dataset:
    def __init__(self, root):
        self.db_path = Path(root) / "captures.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS groups (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    split TEXT NOT NULL, negative INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY, group_id TEXT NOT NULL REFERENCES groups(id),
                    frame_id TEXT NOT NULL UNIQUE, jpeg BLOB NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL, boxes TEXT
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def new_group(self, name, split, negative):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 60:
            raise ValueError("组名需要 1–60 个字符")
        if split not in SPLITS or type(negative) is not bool:
            raise ValueError("请选择分组用途和目标类型")
        identity = uuid.uuid4().hex
        try:
            with self.connect() as db:
                db.execute("INSERT INTO groups VALUES (?, ?, ?, ?)", (identity, name.strip(), split, negative))
        except sqlite3.IntegrityError as error:
            raise ValueError("组名已存在") from error
        return identity

    def prepare_groups(self):
        with self.connect() as db:
            for split, prefix, count in (("train", "训练", 6), ("val", "验证", 2), ("test", "测试", 2)):
                for index in range(1, count + 1):
                    negative = index == count
                    label = "无目标" if negative else "含目标"
                    db.execute("INSERT OR IGNORE INTO groups VALUES (?, ?, ?, ?)", (
                        uuid.uuid4().hex, f"{prefix} {index:02d} · {label}", split, negative,
                    ))

    def capture(self, group, frame):
        if not isinstance(group, str):
            raise ValueError("请先选一个采集组")
        identity = uuid.uuid4().hex
        try:
            with self.connect() as db:
                if db.execute("SELECT 1 FROM groups WHERE id=?", (group,)).fetchone() is None:
                    raise ValueError("请先选一个采集组")
                db.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, NULL)", (
                    identity, group, frame.id, frame.jpeg, frame.width, frame.height,
                ))
        except sqlite3.IntegrityError as error:
            raise ValueError("这帧已经保存，请移动道具并等待画面更新") from error
        return identity

    def label(self, identity, boxes, negative):
        if not isinstance(identity, str) or type(negative) is not bool:
            raise ValueError("标注格式不正确")
        boxes = validated_boxes(boxes)
        if negative != (not boxes):
            raise ValueError("请添加目标框，或明确保存为无目标")
        with self.connect() as db:
            row = db.execute("SELECT g.negative FROM groups g JOIN images i ON g.id=i.group_id WHERE i.id=?",
                             (identity,)).fetchone()
            if row is None:
                raise ValueError("找不到照片")
            if bool(row["negative"]) != negative:
                raise ValueError("照片标注须与本组的有目标／无目标设置一致")
            db.execute("UPDATE images SET boxes=? WHERE id=?", (json.dumps(boxes), identity))

    def image(self, identity):
        with self.connect() as db:
            row = db.execute("SELECT jpeg FROM images WHERE id=?", (identity,)).fetchone()
            return row["jpeg"] if row else None

    def groups(self):
        with self.connect() as db:
            groups = {row["id"]: dict(row) | {"images": []}
                      for row in db.execute("SELECT * FROM groups ORDER BY rowid")}
            for row in db.execute("SELECT id, group_id, width, height, boxes FROM images ORDER BY rowid"):
                sample = dict(row)
                sample["boxes"] = None if row["boxes"] is None else json.loads(row["boxes"])
                groups[row["group_id"]]["images"].append(sample)
        return list(groups.values())

    def export(self, folder):
        with self.connect() as db:
            rows = db.execute("""
                SELECT i.*, g.split, g.negative FROM images i JOIN groups g ON g.id=i.group_id
                WHERE i.boxes IS NOT NULL ORDER BY i.rowid
            """).fetchall()
        counts = {split: 0 for split in SPLITS}
        classes = {split: set() for split in SPLITS}
        for row in rows:
            counts[row["split"]] += 1
            classes[row["split"]].update(box["class_id"] for box in json.loads(row["boxes"]))
        for split, label in (("train", "训练"), ("val", "验证")):
            missing = [name for index, name in enumerate(LABELS) if index not in classes[split]]
            if missing:
                raise ValueError(f"{label}组还缺已标注的类别：{'、'.join(missing)}")

        folder.mkdir(parents=True)
        for split in SPLITS:
            for kind in ("images", "labels"):
                (folder / kind / split).mkdir(parents=True)
        for row in rows:
            split, identity = row["split"], row["id"]
            (folder / "images" / split / f"{identity}.jpg").write_bytes(row["jpeg"])
            lines = []
            for box in json.loads(row["boxes"]):
                x1, y1, x2, y2 = box["xyxy"]
                xywh = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                lines.append(str(box["class_id"]) + " " + " ".join(f"{v:.9f}" for v in xywh))
            (folder / "labels" / split / f"{identity}.txt").write_text("\n".join(lines), encoding="utf-8")
        data = {"path": str(folder.resolve()), "train": "images/train", "val": "images/val", "names": list(CLASSES)}
        if counts["test"]:
            data["test"] = "images/test"
        # JSON is valid YAML and handles paths with spaces.
        write_json(folder / "data.yaml", data)
        write_json(folder / "manifest.json", {"counts": counts, "samples": [
            {key: row[key] for key in ("id", "group_id", "frame_id", "split")} for row in rows
        ]})
        return counts
