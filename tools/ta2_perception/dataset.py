"""Scene-separated dataset, training, export, detector evaluation and replay."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from calibrate import atomic_json
from core import CLASSES


def import_video(video, root, scene, view, interval, limit):
    from core import PAIRS
    import re
    if not re.fullmatch(r'[\w-]{1,64}',scene) or view not in {v for pair in PAIRS.values() for v in pair}:
        raise ValueError('需要有效场次和六路之一的视角名称')
    root=Path(root)
    for kind in ('images','labels','metadata'):(root/kind).mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps<=0:raise ValueError('不能打开视频或帧率无效')
    step=max(1,round(fps*interval));count=0;frame_no=0;prefix=time.time_ns()
    try:
        while count<limit:
            ok,frame=cap.read()
            if not ok:break
            if frame_no%step==0:
                identity=f'{scene}_{prefix}_{view}_{frame_no}'
                cv2.imwrite(str(root/'images'/(identity+'.png')),frame)
                atomic_json(root/'metadata'/(identity+'.json'),{'scene':scene,'view':view,'labeled':False,
                    'source_video':str(video),'frame_no':frame_no,'width':frame.shape[1],'height':frame.shape[0],
                    'note':'用于二维分割标注；单目录像不可用作双目标定'})
                count+=1
            frame_no+=1
    finally:cap.release()
    return {'imported':count,'scene':scene,'view':view}


def split_dataset(root, output, seed=42):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("输出目录已存在；请使用新版本目录，避免残留样本混入")
    scenes = {}
    for path in sorted((root / "metadata").glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("labeled") and (root/"labels"/(path.stem+".txt")).exists():
            scenes.setdefault(meta["scene"], []).append(path.stem)
    if len(scenes) < 7:
        raise ValueError(f"只有 {len(scenes)} 个已标注场次；至少需要 7 个独立摆放场次，以划分训练/验证/测试")
    names = sorted(scenes); random.Random(seed).shuffle(names)
    val = max(1, round(len(names)*.15)); test = max(1, round(len(names)*.15))
    partition = {"train": names[:len(names)-val-test], "val": names[len(names)-val-test:len(names)-test], "test": names[len(names)-test:]}
    manifest = {"seed": seed, "scenes": partition, "counts": {}, "classes": CLASSES}
    distribution = {}
    for split, scene_names in partition.items():
        for kind in ("images", "labels"): (output/kind/split).mkdir(parents=True, exist_ok=True)
        counts = [0]*len(CLASSES); total = negatives = 0
        for scene in scene_names:
            for identity in scenes[scene]:
                image = root/"images"/(identity+".png")
                label = root/"labels"/(identity+".txt")
                text = label.read_text().strip()
                if not text: negatives += 1
                for row in text.splitlines():
                    values = row.split(); cls = int(values[0]); pts = np.array([float(v) for v in values[1:]])
                    if cls not in range(4) or len(pts)<6 or len(pts)%2 or not np.isfinite(pts).all() or np.any((pts<0)|(pts>1)):
                        raise ValueError(f"无效分割标签 {label}")
                    counts[cls] += 1
                shutil.copy2(image, output/"images"/split/image.name)
                shutil.copy2(label, output/"labels"/split/label.name)
                total += 1
        manifest["counts"][split] = {"images": total, "objects": counts, "negative_images": negatives}
    # JSON string values are legal YAML scalars; no YAML dependency is required.
    yaml = f'path: {json.dumps(str(output.resolve()), ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'
    yaml += ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASSES))
    (output/"data.yaml").write_text(yaml, encoding="utf-8")
    atomic_json(output/"manifest.json", manifest)
    return manifest


def train(dataset, output, epochs, device):
    manifest = json.loads((Path(dataset)/"manifest.json").read_text())
    if sum(c["images"] for c in manifest["counts"].values()) < 300:
        raise ValueError("正式现场模型至少准备 300 张已复核图片；不能把少量重复帧训练结果当作已完成识别")
    for split, counts in manifest["counts"].items():
        if min(counts["objects"]) == 0 or counts["negative_images"] == 0:
            raise ValueError(f"{split} 缺少某类道具或负样本，请补充")
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")
    model.train(data=str(Path(dataset)/"data.yaml"), epochs=epochs, imgsz=640, batch=8,
                device=device, workers=2, seed=42, deterministic=True, project=output,
                name="sandbag", exist_ok=False, patience=20, hsv_h=.01, hsv_s=.3, hsv_v=.25)


def evaluate_detector(model_path, dataset, device, output):
    from ultralytics import YOLO
    model = YOLO(model_path, task="segment")
    if list(model.names.values()) != CLASSES: raise ValueError("模型类别错误")
    metrics = model.val(data=str(Path(dataset)/"data.yaml"), split="test", device=device, imgsz=640, conf=.7)
    indices = list(metrics.box.ap_class_index)
    idx = indices.index(0) if 0 in indices else None
    precision = float(metrics.box.p[idx]) if idx is not None else 0.
    recall = float(metrics.box.r[idx]) if idx is not None else 0.
    report = {"precision_sandbag": precision, "recall_sandbag": recall,
              "segmentation_map50": float(metrics.seg.map50), "test_only": True,
              "threshold": .7, "passed": bool(precision >= .98 and recall >= .95),
              "note": "还需红瓶盖专项、三维定位和端到端延迟验收；检测指标不等于抓取成功率"}
    atomic_json(output, report)
    return report


def evaluate_positions(path, output):
    """JSONL: group, point_id, expected_m, measured_m, latency_ms; no dropped-error hiding."""
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    report = {}
    for group in ("head", "left_wrist", "right_wrist", "fused"):
        selected = [r for r in rows if r["group"] == group]
        points = {}
        for r in selected: points.setdefault(r["point_id"], []).append(r)
        errors, stds, latencies = [], [], []
        complete = len(points) >= 10 and all(len(rs) >= 20 for rs in points.values())
        invalid = 0
        for point_id, rs in points.items():
            valid = [r for r in rs if r.get("measured_m") is not None]
            invalid += len(rs)-len(valid)
            if len(valid)<20: complete=False
            if valid:
                measured=np.array([r["measured_m"] for r in valid],float)
                truth=np.array([r["expected_m"] for r in valid],float)
                if measured.shape[1:] != (3,) or not np.isfinite(measured).all() or not np.isfinite(truth).all(): raise ValueError("nonfinite position")
                errors.extend(np.linalg.norm(measured-truth,axis=1).tolist())
                stds.append(float(np.max(np.std(measured,axis=0))))
                latencies.extend(r["latency_ms"] for r in valid)
        e95=float(np.percentile(errors,95)) if errors else None
        smax=max(stds) if stds else None
        l95=float(np.percentile(latencies,95)) if latencies else None
        report[group]={"positions":len(points),"samples":len(selected),"invalid":invalid,"error_p95_m":e95,"repeat_std_max_m":smax,"latency_p95_ms":l95,
                       "passed":bool(complete and invalid==0 and e95<=.015 and smax<=.005 and l95<=300)}
    atomic_json(output,report)
    return report


def replay(config_path, directory, output):
    from server import Worker
    w = Worker(config_path)
    lines=[]
    for p in sorted(Path(directory).glob('*.json')):
        data=json.loads(p.read_text());frame=cv2.imread(str(p.with_suffix('.jpg')))
        if frame is None: continue
        r=w.process(data['packet'],frame)
        r.pop('views',None);r['offline_replay']=True
        # Historical coordinates are for offline analysis only, never fresh robot targets.
        r['grasp_candidates']=[]
        lines.append(json.dumps(r,ensure_ascii=False,allow_nan=False))
    Path(output).write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('split');s.add_argument('--dataset',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('train');s.add_argument('--dataset',required=True);s.add_argument('--out',required=True);s.add_argument('--epochs',type=int,default=100);s.add_argument('--device',default='cpu')
    s=sub.add_parser('export');s.add_argument('--model',required=True)
    s=sub.add_parser('evaluate');s.add_argument('--model',required=True);s.add_argument('--dataset',required=True);s.add_argument('--device',default='cpu');s.add_argument('--out',required=True)
    s=sub.add_parser('positions');s.add_argument('--input',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('replay');s.add_argument('--config',required=True);s.add_argument('--input',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('import-video');s.add_argument('--input',required=True);s.add_argument('--dataset',required=True);s.add_argument('--scene',required=True);s.add_argument('--view',required=True);s.add_argument('--interval',type=float,default=2);s.add_argument('--limit',type=int,default=60)
    a=p.parse_args()
    if a.command=='split': print(json.dumps(split_dataset(a.dataset,a.out),ensure_ascii=False))
    elif a.command=='train':train(a.dataset,a.out,a.epochs,a.device)
    elif a.command=='export':
        from ultralytics import YOLO
        print(YOLO(a.model).export(format='openvino',half=False,dynamic=False,imgsz=640))
    elif a.command=='evaluate':print(json.dumps(evaluate_detector(a.model,a.dataset,a.device,a.out)))
    elif a.command=='positions':print(json.dumps(evaluate_positions(a.input,a.out)))
    elif a.command=='replay':replay(a.config,a.input,a.out)
    else:print(json.dumps(import_video(a.input,a.dataset,a.scene,a.view,a.interval,a.limit)))


if __name__=='__main__':main()
