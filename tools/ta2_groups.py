#!/usr/bin/env python3
"""
ta2_groups.py — 训练组清单（一条 mcap 只属于一个组）

组是逻辑概念，不移动任何数据文件。清单落在 <review-root>/groups.json。

结构
    {"version": 1,
     "groups": {
        "<gid>": {"id","name","scene","light","objects":[],"operator","task","note",
                  "records":[...], "datasets":[...], "created_at","updated_at"}}}

用法
    python3 ta2_groups.py list
    python3 ta2_groups.py create --name "桌面A-白光" --scene 桌面A --light 白光 \
        --objects 红沙包,蓝垫 --operator 侯兴家 --task 抓取红色沙包
    python3 ta2_groups.py assign --group g_xxx --records rec_a rec_b
    python3 ta2_groups.py assign --group g_xxx --records rec_c --force   # 从别的组抢过来
    python3 ta2_groups.py show --group g_xxx
    python3 ta2_groups.py unassigned
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_REVIEW_ROOT = "/data/ta2/review"


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(name):
    s = re.sub(r"[^\w\u4e00-\u9fa5\-]+", "-", name.strip())
    return s.strip("-") or "group"


class GroupStore:
    def __init__(self, review_root=DEFAULT_REVIEW_ROOT):
        self.path = Path(review_root) / "groups.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        if not self.path.exists():
            return {"version": 1, "groups": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        data.setdefault("version", 1)
        data.setdefault("groups", {})
        return data

    def save(self, data):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---------- 查询 ----------
    def list(self):
        data = self.load()
        return list(data["groups"].values())

    def get(self, gid):
        return self.load()["groups"].get(gid)

    def group_of(self, rid):
        for g in self.load()["groups"].values():
            if rid in (g.get("records") or []):
                return g["id"]
        return None

    def assigned_map(self):
        """rid -> gid"""
        out = {}
        for g in self.load()["groups"].values():
            for rid in (g.get("records") or []):
                out[rid] = g["id"]
        return out

    # ---------- 写入 ----------
    def create(self, name, **meta):
        data = self.load()
        gid = meta.pop("id", None) or f"g_{datetime.now():%Y%m%d_%H%M%S}_{slugify(name)[:16]}"
        if gid in data["groups"]:
            raise ValueError(f"组已存在: {gid}")
        g = {
            "id": gid,
            "name": name,
            "scene": meta.get("scene") or "",
            "light": meta.get("light") or "",
            "objects": list(meta.get("objects") or []),
            "operator": meta.get("operator") or "",
            "task": meta.get("task") or "",
            "note": meta.get("note") or "",
            "records": [],
            "datasets": [],
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        data["groups"][gid] = g
        self.save(data)
        return g

    def update(self, gid, **fields):
        data = self.load()
        g = data["groups"].get(gid)
        if not g:
            raise KeyError(gid)
        for k in ("name", "scene", "light", "operator", "task", "note"):
            if k in fields and fields[k] is not None:
                g[k] = fields[k]
        if "objects" in fields and fields["objects"] is not None:
            g["objects"] = list(fields["objects"])
        g["updated_at"] = now_iso()
        self.save(data)
        return g

    def assign(self, gid, rids, force=False):
        """把记录归入组。一条记录只能属于一个组；不 force 时遇到冲突报错。"""
        data = self.load()
        g = data["groups"].get(gid)
        if not g:
            raise KeyError(gid)
        conflicts = []
        for rid in rids:
            for other in data["groups"].values():
                if other["id"] != gid and rid in (other.get("records") or []):
                    conflicts.append((rid, other["id"]))
        if conflicts and not force:
            raise ValueError("以下记录已属于其他组（加 --force 可移动）: " +
                             ", ".join(f"{r}->{o}" for r, o in conflicts))
        if conflicts:
            for rid, other_id in conflicts:
                other = data["groups"][other_id]
                other["records"] = [x for x in other["records"] if x != rid]
                other["updated_at"] = now_iso()
        recs = list(g.get("records") or [])
        for rid in rids:
            if rid not in recs:
                recs.append(rid)
        g["records"] = recs
        g["updated_at"] = now_iso()
        self.save(data)
        return g

    def unassign(self, rids):
        data = self.load()
        removed = []
        for g in data["groups"].values():
            before = len(g.get("records") or [])
            g["records"] = [x for x in (g.get("records") or []) if x not in rids]
            if len(g["records"]) != before:
                removed.append(g["id"])
                g["updated_at"] = now_iso()
        self.save(data)
        return removed

    def link_dataset(self, gid, dataset_name):
        data = self.load()
        g = data["groups"].get(gid)
        if not g:
            raise KeyError(gid)
        ds = list(g.get("datasets") or [])
        if dataset_name not in ds:
            ds.append(dataset_name)
        g["datasets"] = ds
        g["updated_at"] = now_iso()
        self.save(data)
        return g

    def delete(self, gid):
        data = self.load()
        if gid not in data["groups"]:
            raise KeyError(gid)
        del data["groups"][gid]
        self.save(data)


def main(argv=None):
    ap = argparse.ArgumentParser(description="训练组清单管理")
    ap.add_argument("--review-root", default=DEFAULT_REVIEW_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--scene", default="")
    c.add_argument("--light", default="")
    c.add_argument("--objects", default="")
    c.add_argument("--operator", default="")
    c.add_argument("--task", default="")
    c.add_argument("--note", default="")

    a = sub.add_parser("assign")
    a.add_argument("--group", required=True)
    a.add_argument("--records", nargs="+", required=True)
    a.add_argument("--force", action="store_true")

    u = sub.add_parser("unassign")
    u.add_argument("--records", nargs="+", required=True)

    s = sub.add_parser("show")
    s.add_argument("--group", required=True)

    sub.add_parser("unassigned")

    args = ap.parse_args(argv)
    store = GroupStore(args.review_root)

    if args.cmd == "list":
        groups = store.list()
        if not groups:
            print("(暂无训练组)")
            return 0
        for g in groups:
            print(f"{g['id']}  {g['name']}  记录 {len(g.get('records') or [])}  数据集 {len(g.get('datasets') or [])}")
            bits = [f"{k}={g[k]}" for k in ("scene", "light", "operator", "task") if g.get(k)]
            if g.get("objects"):
                bits.append("objects=" + ",".join(g["objects"]))
            if bits:
                print("    " + "  ".join(bits))
        return 0

    if args.cmd == "create":
        objs = [x.strip() for x in args.objects.split(",") if x.strip()]
        g = store.create(args.name, scene=args.scene, light=args.light, objects=objs,
                         operator=args.operator, task=args.task, note=args.note)
        print("已创建:", g["id"])
        return 0

    if args.cmd == "assign":
        g = store.assign(args.group, args.records, force=args.force)
        print(f"已归入 {g['id']}，当前 {len(g['records'])} 条记录")
        return 0

    if args.cmd == "unassign":
        changed = store.unassign(args.records)
        print("已从这些组移除:", ", ".join(changed) or "(无变化)")
        return 0

    if args.cmd == "show":
        g = store.get(args.group)
        if not g:
            print("组不存在", file=sys.stderr)
            return 1
        print(json.dumps(g, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "unassigned":
        assigned = store.assigned_map()
        print(f"已归组 {len(assigned)} 条；未归组的记录请到转化台查看")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
