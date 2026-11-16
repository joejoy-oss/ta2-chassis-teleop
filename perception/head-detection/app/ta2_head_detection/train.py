"""Train the four-class YOLO26 detector from a local dataset snapshot."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to data.yaml")
    parser.add_argument("--weights", type=Path, required=True, help="Local pretrained yolo26n.pt")
    parser.add_argument("--run", type=Path, required=True, help="Output directory for this training run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.dataset.is_file():
        parser.error("本地基础权重或数据快照不存在")
    if not 1 <= args.epochs <= 80:
        parser.error("训练轮数必须在 1–80 之间")

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(8)
    run = args.run.resolve()
    print(f"Training on {args.device}: {args.dataset}", flush=True)
    YOLO(str(args.weights.resolve()), task="detect").train(
        data=str(args.dataset.resolve()), project=str(run.parent), name=run.name, exist_ok=True,
        device=args.device, imgsz=640, epochs=args.epochs, batch=8, workers=2, patience=15,
        seed=42, amp=False, plots=False,
    )
    if not (run / "weights" / "best.pt").is_file():
        raise RuntimeError("训练未生成 best.pt")


if __name__ == "__main__":
    main()
