#!/usr/bin/env python3
"""Offline open-loop evaluation of a served policy against recorded LeRobot episodes (no robot).

For every chosen episode, frames at 0 / 25 / 50 / 75 % and near the end are sent to the policy server;
the predicted 30-step chunk is compared with the recorded actions.
Metrics per frame: first-step max joint error (deg), 30-step mean joint error (deg), gripper sign agreement
(pred effort vs recorded effort, both grippers, steps where |recorded| > 0.1).

usage: eval_ckpt.py --ds DIR [--episodes valid|all|0,3,5|every:N] [--annotations FILE] --prompt P
                    --port 8001 --tag NAME --out FILE.json
Runs inside a policy container (openpi_client, PyAV, pyarrow).
"""
import argparse, json, sys, time
import numpy as np, pyarrow.parquet as pq, av
sys.path.insert(0, "/app/packages/openpi-client/src")
from openpi_client import websocket_client_policy

J = [*range(7), *range(8, 15)]
CAMS = ["head_camera", "left_color", "right_color"]
GRIP_PRED, GRIP_REC = (7, 15), (39, 47)        # 19-dim model output vs 76-dim recorded action


def frames(ds, ep, cam, want):
    c = av.open(f"{ds}/videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4")
    out = {}
    for i, f in enumerate(c.decode(video=0)):
        if i in want:
            out[i] = f.to_ndarray(format="rgb24")
        if i >= max(want):
            break
    c.close()
    return out


def pick_episodes(a, n_total):
    if a.episodes == "all":
        return list(range(n_total))
    if a.episodes == "valid":
        ann = json.load(open(a.annotations))["episodes"]
        return sorted(int(k) for k, v in ann.items() if v.get("mark") == "valid")
    if a.episodes.startswith("every:"):
        k = int(a.episodes[6:]); return list(range(0, n_total, k))
    return [int(x) for x in a.episodes.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--episodes", default="all")
    ap.add_argument("--annotations"); ap.add_argument("--prompt", required=True)
    ap.add_argument("--port", type=int, default=8001); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = json.load(open(f"{a.ds}/meta/info.json"))
    eps = pick_episodes(a, info["total_episodes"])
    pol = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=a.port)
    rows, t0 = [], time.time()
    for ep in eps:
        print(f"PROGRESS episode {ep}, completed_frames={len(rows)}", flush=True)
        t = pq.read_table(f"{a.ds}/data/chunk-000/episode_{ep:06d}.parquet", columns=["observation.state", "action"])
        S = np.array(t["observation.state"].to_pylist(), dtype=np.float64)
        A = np.array(t["action"].to_pylist(), dtype=np.float64)
        n = len(S)
        want = sorted({0, n // 4, n // 2, (3 * n) // 4, max(0, n - 15)})
        imgs = {c: frames(a.ds, ep, c, want) for c in CAMS}
        for f in want:
            obs = {"observation/state": S[f].astype(np.float32), "prompt": a.prompt}
            for c in CAMS:
                obs[f"observation/images/{c}"] = imgs[c][f]
            pred = np.asarray(pol.infer(obs)["actions"])
            h = min(30, n - f)
            err = np.degrees(np.abs(pred[:h, J] - A[f:f + h, J]))
            gp = pred[:h, list(GRIP_PRED)]; gr = A[f:f + h, list(GRIP_REC)]
            mask = np.abs(gr) > 0.1
            agree = float((np.sign(gp) == np.sign(gr))[mask].mean()) if mask.any() else None
            rows.append({"ep": ep, "frame": f, "phase": round(f / max(n - 1, 1), 2),
                         "first_deg": round(float(err[0].max()), 2), "mae30_deg": round(float(err.mean()), 3),
                         "grip_agree": None if agree is None else round(agree, 3)})
    fd = np.array([r["first_deg"] for r in rows]); m = np.array([r["mae30_deg"] for r in rows])
    g = np.array([r["grip_agree"] for r in rows if r["grip_agree"] is not None])
    summary = {"tag": a.tag, "dataset": a.ds, "prompt": a.prompt, "episodes": len(eps), "frames": len(rows),
               "first_deg_median": round(float(np.median(fd)), 2), "first_deg_p90": round(float(np.percentile(fd, 90)), 2),
               "first_deg_lt3_pct": round(float((fd < 3).mean() * 100), 1),
               "mae30_deg_median": round(float(np.median(m)), 3), "mae30_deg_p90": round(float(np.percentile(m, 90)), 3),
               "grip_agree_mean": round(float(g.mean()), 3) if len(g) else None, "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
