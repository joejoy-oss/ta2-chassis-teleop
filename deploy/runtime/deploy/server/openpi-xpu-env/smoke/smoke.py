"""阶段 1 冒烟测试：torch XPU 在 B70 上的基本可用性（PLAN §4 阶段 1 (b)）。

用法：/opt/venv/bin/python smoke.py [--label ppa|host] [--quick]
每项检查独立 try/except，最后输出一行 `SMOKE_SUMMARY {json}`。
验收（PLAN）：bf16 4096² matmul 相对误差 < 1e-2；compile 与 eager 最大差 < 1e-3（fp32）。
"""

import argparse
import json
import os
import time
import traceback

import torch

results: dict[str, dict] = {}


def check(name):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                info = fn() or {}
                ok = info.pop("ok", True)
                results[name] = {"ok": bool(ok), "sec": round(time.perf_counter() - t0, 3), **info}
            except Exception as e:  # noqa: BLE001
                results[name] = {
                    "ok": False,
                    "sec": round(time.perf_counter() - t0, 3),
                    "error": f"{type(e).__name__}: {e}"[:2000],
                    "trace": traceback.format_exc()[-3000:],
                }
            print(f"[{'PASS' if results[name]['ok'] else 'FAIL'}] {name}: {json.dumps(results[name], ensure_ascii=False)[:600]}", flush=True)

        run.__name__ = name
        return run

    return deco


def rel_err(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


DEV = "xpu"


@check("env")
def c_env():
    props = []
    for i in range(torch.xpu.device_count()):
        p = torch.xpu.get_device_properties(i)
        props.append(
            {
                k: (str(getattr(p, k)) if not isinstance(getattr(p, k), (int, float, bool)) else getattr(p, k))
                for k in ("name", "platform_name", "driver_version", "total_memory", "max_compute_units", "gpu_eu_count", "has_fp64")
                if hasattr(p, k)
            }
        )
    return {
        "torch": torch.__version__,
        "xpu_available": torch.xpu.is_available(),
        "device_count": torch.xpu.device_count(),
        "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
        "props": props,
        "ok": torch.xpu.is_available() and torch.xpu.device_count() >= 1,
    }


@check("first_op_latency")
def c_first_op():
    x = torch.ones(1 << 20, device=DEV)
    t0 = time.perf_counter()
    y = x * 2 + 1
    torch.xpu.synchronize()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    y = x * 3 + 2
    torch.xpu.synchronize()
    second = time.perf_counter() - t0
    ok = float(y[0].item()) == 5.0
    return {"first_s": round(first, 4), "second_s": round(second, 6), "ok": ok}


@check("matmul_4096")
def c_matmul():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4096, 4096, generator=g)
    b = torch.randn(4096, 4096, generator=g)
    ref = a @ b
    out_bf16 = (a.to(DEV, torch.bfloat16) @ b.to(DEV, torch.bfloat16)).float().cpu()
    try:  # 新版 torch 用此开关控制 fp32 matmul 是否允许降精度（TF32 类），旧版没有该属性
        torch.backends.mkldnn.matmul.fp32_precision = "ieee"
    except AttributeError:
        pass
    out_fp32 = (a.to(DEV) @ b.to(DEV)).cpu()
    e_bf16, e_fp32 = rel_err(out_bf16, ref), rel_err(out_fp32, ref)
    # 计时（bf16）
    ad, bd = a.to(DEV, torch.bfloat16), b.to(DEV, torch.bfloat16)
    for _ in range(3):
        ad @ bd
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        ad @ bd
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / n
    tflops = 2 * 4096**3 / dt / 1e12
    return {"rel_err_bf16": e_bf16, "rel_err_fp32": e_fp32, "bf16_tflops": round(tflops, 1), "ok": e_bf16 < 1e-2 and e_fp32 < 1e-2}


@check("softmax_sdpa")
def c_sdpa():
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 8, 512, 64, generator=g) for _ in range(3))
    mask = torch.rand(2, 1, 512, 512, generator=g) > 0.1
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    sm_ref = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
    out = {}
    for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        qd, kd, vd, md = (t.to(DEV) for t in (q, k, v, mask))
        qd, kd, vd = (t.to(dt) for t in (qd, kd, vd))
        o = torch.nn.functional.scaled_dot_product_attention(qd, kd, vd, attn_mask=md).float().cpu()
        s = torch.softmax(qd @ kd.transpose(-1, -2), dim=-1).float().cpu()
        out[f"sdpa_maxabs_{dt_name}"] = float((o - ref).abs().max())
        out[f"softmax_maxabs_{dt_name}"] = float((s - sm_ref).abs().max())
    out["ok"] = out["sdpa_maxabs_fp32"] < 1e-3 and out["sdpa_maxabs_bf16"] < 5e-2
    return out


def _mlp(dtype):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(4096, 1024))
    return m.to(DEV, dtype).eval()


def _compile_case(mode, dtype, tol):
    m = _mlp(dtype)
    x = torch.randn(64, 1024, device=DEV, dtype=dtype)
    with torch.no_grad():
        ref = m(x)
        torch._dynamo.reset()
        cm = torch.compile(m, mode=mode)
        t0 = time.perf_counter()
        y = cm(x)
        torch.xpu.synchronize()
        compile_s = time.perf_counter() - t0
        for _ in range(5):
            cm(x)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            cm(x)
        torch.xpu.synchronize()
        steady_ms = (time.perf_counter() - t0) / 50 * 1e3
    diff = float((y.float() - ref.float()).abs().max())
    return {"compile_s": round(compile_s, 2), "steady_ms": round(steady_ms, 4), "maxdiff_vs_eager": diff, "ok": diff < tol}


@check("compile_default_fp32")
def c_comp_default():
    return _compile_case("default", torch.float32, 1e-3)


@check("compile_maxautotune_nocg_fp32")
def c_comp_mat():
    return _compile_case("max-autotune-no-cudagraphs", torch.float32, 1e-3)


@check("compile_maxautotune_bf16")
def c_comp_mat_bf16():
    # openpi 默认模式 max-autotune；XPU 上 cudagraph trees 未实现，看是否报错/降级
    return _compile_case("max-autotune", torch.bfloat16, 5e-2)


@check("autograd_grad_through_compiled")
def c_grad():
    m = _mlp(torch.float32)
    for p in m.parameters():
        p.requires_grad_(False)
    torch._dynamo.reset()
    cm = torch.compile(m, mode="default")
    x0 = torch.randn(64, 1024, device=DEV)
    u = torch.randn(64, 1024, device=DEV)
    out = {}
    for name, fn in (("eager", m), ("compiled", cm)):
        x = x0.clone().requires_grad_(True)
        with torch.enable_grad():
            y = fn(x)
            (g,) = torch.autograd.grad(y, x, grad_outputs=u)
        out[name] = g.detach()
    # torch.func.vjp 版本
    from torch.func import vjp

    _, vjp_fn = vjp(cm, x0)
    (g_func,) = vjp_fn(u)
    d1 = float((out["compiled"] - out["eager"]).abs().max())
    d2 = float((g_func - out["eager"]).abs().max())
    scale = float(out["eager"].abs().max())
    return {"maxdiff_autograd": d1, "maxdiff_func_vjp": d2, "grad_scale": scale, "ok": d1 < 1e-3 * max(1.0, scale) and d2 < 1e-3 * max(1.0, scale)}


@check("loop_1000_small_kernels")
def c_loop():
    x = torch.zeros(4096, device=DEV)
    t0 = time.perf_counter()
    for _ in range(1000):
        x = x * 1.0001 + 0.001
    torch.xpu.synchronize()
    dt = time.perf_counter() - t0
    v = float(x[0].item())
    expected = sum(0.001 * 1.0001**i for i in range(1000))
    return {"sec_1000": round(dt, 4), "value": v, "expected": expected, "ok": abs(v - expected) / expected < 1e-3}


@check("memory")
def c_mem():
    return {
        "max_allocated_MB": round(torch.xpu.max_memory_allocated() / 2**20, 1),
        "max_reserved_MB": round(torch.xpu.max_memory_reserved() / 2**20, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--quick", action="store_true", help="跳过 max-autotune 编译用例")
    args = ap.parse_args()
    torch.manual_seed(0)
    t_all = time.perf_counter()
    c_env()
    if results["env"]["ok"]:
        c_first_op()
        c_matmul()
        c_sdpa()
        c_comp_default()
        c_grad()
        if not args.quick:
            c_comp_mat()
            c_comp_mat_bf16()
        c_loop()
        c_mem()
    summary = {
        "label": args.label,
        "all_ok": all(r["ok"] for r in results.values()),
        "total_sec": round(time.perf_counter() - t_all, 1),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "trace"} for k, v in results.items()},
    }
    print("SMOKE_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
