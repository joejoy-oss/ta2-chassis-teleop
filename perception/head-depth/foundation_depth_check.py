"""Run NVIDIA FoundationStereo ONNX on the saved, rectified TA2 head pairs."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def load_rgb(path, size):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read {path}")
    rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    mean = np.float32([0.485, 0.456, 0.406])
    std = np.float32([0.229, 0.224, 0.225])
    return np.ascontiguousarray(((rgb.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None])


def session(model, device, size, precision):
    if device.startswith("ov-"):
        import openvino as ov

        core = ov.Core()
        print("OpenVINO devices:", core.available_devices, flush=True)
        graph = core.read_model(str(model))
        print("OpenVINO ONNX loaded", flush=True)
        graph.reshape({port.any_name: [1, 3, size, size] for port in graph.inputs})
        print(f"OpenVINO shape fixed to {size}x{size}; compiling {device[3:]}", flush=True)
        compiled = core.compile_model(graph, device[3:], {
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": precision})
        print("Execution devices:", compiled.get_property("EXECUTION_DEVICES"), flush=True)
        names = [port.any_name for port in compiled.inputs]
        return names, lambda inputs: list(compiled(inputs).values())

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.inter_op_num_threads = 1
    instance = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
    return [value.name for value in instance.get_inputs()], lambda inputs: instance.run(None, inputs)


def metric_depth(disparity, K, baseline, rect_valid):
    height, width = disparity.shape
    yy, xx = np.indices(disparity.shape)
    valid = rect_valid & np.isfinite(disparity) & (disparity > 0.5) & ((xx - disparity) >= 0)
    depth = np.zeros(disparity.shape, np.float32)
    depth[valid] = K[0, 0] * baseline / disparity[valid]
    valid &= (depth >= 0.08) & (depth < 2.5)
    depth[~valid] = 0
    points = np.stack(((xx - K[0, 2]) * depth / K[0, 0],
                       (yy - K[1, 2]) * depth / K[1, 1], depth), axis=-1)
    return depth, points, valid


def sandbag_mask(image):
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    red = ((hue < 12) | (hue > 169)) & (saturation > 115) & (value > 45)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    if n <= 1:
        return np.zeros(red.shape, bool)
    index = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return cv2.erode((labels == index).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--device", choices=["cpu", "ov-GPU", "ov-CPU"], default="cpu")
    p.add_argument("--size", default=480, type=int)
    p.add_argument("--precision", choices=["f16", "f32"], default="f32")
    p.add_argument("--output", type=Path, help="Separate destination for benchmark outputs")
    args = p.parse_args()
    if args.size < 224 or args.size % 32:
        p.error("--size must be a multiple of 32 and at least 224")

    calibration = json.loads((args.pairs / "head-calibration.json").read_text())
    matrices = [np.array(calibration[key]) for key in ("K1", "D1", "K2", "D2")]
    R, T = [np.array(calibration[key]) for key in ("R", "T")]
    original_size = tuple(calibration["size"])
    R1, _, P1, P2, _, _, _ = cv2.stereoRectify(*matrices, original_size, R, T,
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    K = P1[:3, :3]
    baseline = abs(float(P2[0, 3] / P2[0, 0]))
    normal = np.array([0.041103020454557625, -0.8875971641429703, -0.4587829725642282])
    nominal_offset = 0.8010129836867057
    cv2.setNumThreads(2)
    output = args.output or args.pairs
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    names, run = session(args.model, args.device, args.size, args.precision)
    setup_seconds = time.monotonic() - start
    print("Model input names:", names, flush=True)
    left = [name for name in names if "left" in name.lower()]
    right = [name for name in names if "right" in name.lower()]
    if len(names) != 2 or len(left) != 1 or len(right) != 1:
        raise ValueError(f"Unexpected input names: {names}")

    records = []
    for folder in sorted(args.pairs.glob("pair-*")):
        t0 = time.monotonic()
        frame = json.loads((folder / "frame.json").read_text())
        if frame["calibration_version"] != calibration["version"] or frame["regions_version"] != calibration["regions_version"]:
            raise ValueError(f"Calibration mismatch in {folder}")
        size = (args.size, args.size)
        feed = {left[0]: load_rgb(folder / "left.png", size), right[0]: load_rgb(folder / "right.png", size)}
        inference_started = time.monotonic()
        disp = np.asarray(run(feed)[0], dtype=np.float32).squeeze()
        inference_seconds = time.monotonic() - inference_started
        if disp.ndim != 2 or disp.shape != size[::-1]:
            raise ValueError(f"Unexpected disparity shape: {disp.shape}")
        image = cv2.imread(str(folder / "left.png"))
        disp = cv2.resize(disp, original_size, interpolation=cv2.INTER_LINEAR) * original_size[0] / args.size
        depth, points, valid = metric_depth(disp, K, baseline, np.any(image > 0, axis=2))
        mask = sandbag_mask(image)
        good = valid & mask
        heights = (points[good] @ R1) @ normal + nominal_offset
        record = {**frame, "seconds": time.monotonic() - t0, "inference_seconds": inference_seconds,
                  "valid_fraction": float(valid.mean()),
                  "sandbag_coverage": float(good.sum() / max(1, mask.sum())),
                  "sandbag_median_gap_mm": float(np.median(heights) * 1000) if len(heights) else None,
                  "sandbag_p05_gap_mm": float(np.percentile(heights, 5) * 1000) if len(heights) else None}
        destination = output / folder.name
        destination.mkdir(exist_ok=True)
        np.save(destination / "foundation-disparity.npy", disp)
        cv2.imwrite(str(destination / "foundation-depth-mm.png"), np.rint(depth * 1000).astype(np.uint16))
        colors = cv2.applyColorMap(np.clip((depth - 0.25) / 1.75 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colors[~valid] = 0
        cv2.imwrite(str(destination / "foundation-preview.png"), colors)
        records.append(record)
        print(json.dumps(record), flush=True)
    steps = []
    for a, b in zip(records, records[1:]):
        if a["sandbag_median_gap_mm"] is not None and b["sandbag_median_gap_mm"] is not None:
            steps.append(a["sandbag_median_gap_mm"] - b["sandbag_median_gap_mm"])
        else:
            steps.append(None)
    report = {"device": args.device, "model_input_size": args.size, "precision_hint": args.precision,
              "setup_seconds": setup_seconds,
              "sandbag_visible_surface_median_drop_mm": steps,
              "operator_reported_steps_mm": [50, 50, 50], "frames": records,
              "lowest_point_verified": False,
              "note": "coverage is geometric plausibility only; a nominal earlier table plane and different visible object points are used across frames"}
    (output / "foundation-comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
