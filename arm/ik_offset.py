#!/usr/bin/env python3
"""Offline IK with the official MuJoCo model (run inside ta2-sim-pipeline): joint target that moves an arm's
end effector by (dx, dy, dz) metres in the robot's shoulder frame (z = vertical up), orientation unchanged.

  fit once:  ik_offset.py --fit-samples fk_samples.npz --save-fit right_fk_fit.json
  solve:     ik_offset.py --fit right_fk_fit.json --now right_now.json --dz -0.05 --out target.json
Refuses (exit 2, reason on the last line) if the IK residual, path or height checks fail."""
import argparse, json, sys
import numpy as np, mujoco

MODEL = "/workspace/assets/official_mujoco_20260804/mujoco/robot.xml"
Z_MIN = -0.40          # m, robot frame: lowest EE height allowed (training data reached -0.397)
ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("left", "right"), default="right"); ap.add_argument("--fit"); ap.add_argument("--now"); ap.add_argument("--out")
ap.add_argument("--dx", type=float, default=0.0); ap.add_argument("--dy", type=float, default=0.0); ap.add_argument("--dz", type=float, default=0.0)
ap.add_argument("--fit-samples"); ap.add_argument("--save-fit")
a = ap.parse_args()
if a.fit_samples and not a.save_fit:
    ap.error("--fit-samples requires --save-fit")
if not a.fit_samples and not all((a.fit, a.now, a.out)):
    ap.error("solve requires --fit, --now and --out")
m = mujoco.MjModel.from_xml_path(MODEL); d = mujoco.MjData(m)
jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{a.arm[0]}_joint{i}") for i in range(1, 8)]
if any(j < 0 for j in jid):
    ap.error(f"model lacks {a.arm} arm joints")
qadr = [m.jnt_qposadr[j] for j in jid]; dadr = [m.jnt_dofadr[j] for j in jid]
lo = np.array([m.jnt_range[j][0] for j in jid]); hi = np.array([m.jnt_range[j][1] for j in jid])
B7 = m.jnt_bodyid[jid[-1]]


def fk(q):
    d.qpos[:] = 0; d.qpos[qadr] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    return d.xpos[B7].copy(), d.xmat[B7].reshape(3, 3).copy()


def rotvec_R(v):
    th = np.linalg.norm(v)
    if th < 1e-12: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9: return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


if a.fit_samples:                                  # p_robot = Ra (p7 + R7 t) + c
    S = np.load(a.fit_samples); Q, E = S["q"], S["ee"]
    P7, R7 = map(np.array, zip(*[fk(q) for q in Q])); Prec = E[:, :3]
    def resid(x):
        return ((P7 + np.einsum("nij,j->ni", R7, x[6:9])) @ rotvec_R(x[:3]).T + x[3:6] - Prec).ravel()
    x = np.zeros(9); x[3:6] = Prec.mean(0) - P7.mean(0)
    for _ in range(60):
        r = resid(x); J = np.stack([(resid(x + np.eye(9)[k] * 1e-6) - r) / 1e-6 for k in range(9)], 1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]; x += step
        if np.linalg.norm(step) < 1e-10: break
    err = np.linalg.norm(resid(x).reshape(-1, 3), axis=1) * 1000
    json.dump({"arm": a.arm, "rotvec": x[:3].tolist(), "offset": x[3:6].tolist(), "tool": x[6:9].tolist(),
               "samples": len(E), "err_mm_median": float(np.median(err)), "err_mm_p95": float(np.percentile(err, 95))},
              open(a.save_fit, "w"), indent=1)
    print(f"fit saved: median {np.median(err):.2f} mm, p95 {np.percentile(err, 95):.2f} mm"); sys.exit(0)

fit = json.load(open(a.fit))
if fit.get("arm") != a.arm:
    ap.error("fit arm does not match --arm")
Ra = rotvec_R(np.array(fit["rotvec"])); c = np.array(fit["offset"]); t = np.array(fit["tool"])
now = json.load(open(a.now)); q0 = np.array(now["q"][1]); ee0 = np.array(now["ee"][:3])
p0, R0 = fk(q0); tool0 = p0 + R0 @ t
if np.linalg.norm(Ra @ tool0 + c - ee0) > 0.01:
    print(f"model and robot disagree by {np.linalg.norm(Ra @ tool0 + c - ee0) * 1000:.1f} mm: refusing"); sys.exit(2)
delta_robot = np.array([a.dx, a.dy, a.dz])
# Z_MIN limits only the downward part of a move. A horizontal/upward move may start below it (arms hanging
# after a reboot) but its path may not go lower than where it started.
floor = Z_MIN if a.dz < 0 else min(Z_MIN, ee0[2])
if a.dz < 0 and (ee0 + delta_robot)[2] < Z_MIN:    # go only as far as the height limit
    if ee0[2] - Z_MIN < 0.002:
        print(f"already at the height limit (EE z {ee0[2]:.3f} m, limit {Z_MIN} m): nothing to do"); sys.exit(2)
    print(f"height limit: asked for EE z {(ee0 + delta_robot)[2]:.3f} m, going only to {Z_MIN} m")
    delta_robot[2] = Z_MIN - ee0[2]
target = tool0 + Ra.T @ delta_robot
q = q0.copy(); jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
for _ in range(300):
    p, R = fk(q); tool = p + R @ t
    e = np.concatenate([target - tool, R_log(R0 @ R.T)])
    if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < np.radians(0.1): break
    mujoco.mj_jac(m, d, jp, jr, tool, B7)
    Jm = np.vstack([jp[:, dadr], jr[:, dadr]])
    q = np.clip(q + Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e), lo + 0.02, hi - 0.02)
p, R = fk(q); tool = p + R @ t
pos_err = np.linalg.norm(target - tool) * 1000; rot_err = np.degrees(np.linalg.norm(R_log(R0 @ R.T)))
if pos_err > 1.0 or rot_err > 0.5:
    print(f"IK did not converge (pos {pos_err:.1f} mm, rot {rot_err:.2f} deg): refusing"); sys.exit(2)
dev, prog, heights = [], [], []
step_dir = (target - tool0) / max(np.linalg.norm(target - tool0), 1e-9)
for s in np.linspace(0, 1, 41):
    pp, RR = fk(q0 + s * (q - q0)); tt = pp + RR @ t; v = tt - tool0
    heights.append(float((Ra @ tt + c)[2]))
    prog.append(float(v @ step_dir)); dev.append(float(np.linalg.norm(v - (v @ step_dir) * step_dir)) * 1000)
max_dev = max(5.0, float(np.linalg.norm(delta_robot)) * 80.0)
if min(heights) < floor - 1e-3 or max(dev) > max_dev or not all(np.diff(prog) >= -5e-5):
    print(f"path check failed (min_z {min(heights):.4f} m, sideways {max(dev):.1f}/{max_dev:.1f} mm, monotonic {all(np.diff(prog) >= -5e-5)}): refusing"); sys.exit(2)
exp = (Ra @ tool + c).tolist()
json.dump({a.arm: q.tolist(), f"{a.arm}_now": q0.tolist(), "delta_m": delta_robot.tolist(), "expected_ee": exp},
          open(a.out, "w"))
print(f"IK ok: joints change {np.round(np.degrees(q - q0), 1).tolist()} deg, EE z {ee0[2]:.4f} -> {exp[2]:.4f} m, "
      f"sideways {max(dev):.1f} mm")
