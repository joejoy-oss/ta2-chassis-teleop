"""PyTorch port of the Real-Time Chunking math in `openpi.models.rtc`.

The modes, schedules and `RTCConfig` are shared with the JAX implementation; only the two
numeric helpers are re-implemented on tensors. `start`/`end`/`time` may be 0-d tensors so a
changing measured delay never forces a recompile. See docs/real_time_chunking.md.
"""

import math

import torch

from openpi.models.rtc import PrefixAttentionSchedule
from openpi.models.rtc import RTCConfig
from openpi.models.rtc import RTCMode

__all__ = ["PrefixAttentionSchedule", "RTCConfig", "RTCMode", "get_prefix_weights", "guidance_weight"]


def get_prefix_weights(
    start: torch.Tensor | int,
    end: torch.Tensor | int,
    total: int,
    schedule: PrefixAttentionSchedule,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Soft prefix-attention weights of shape [total]: 1 below `start` (already executed),
    decaying to 0 at `end` (free to react to the new observation). Mirrors
    `openpi.models.rtc.get_prefix_weights` value for value (float32)."""
    if device is None:
        device = start.device if isinstance(start, torch.Tensor) else None
    start = torch.as_tensor(start, device=device).clamp(0, total)
    end = torch.as_tensor(end, device=device).clamp(0, total)
    start = torch.minimum(start, end)
    idx = torch.arange(total, device=start.device)

    if schedule is PrefixAttentionSchedule.ONES:
        return (idx < end).to(torch.float32)
    if schedule is PrefixAttentionSchedule.ZEROS:
        return (idx < start).to(torch.float32)

    # Matches torch.linspace(1, 0, n + 2)[1:-1] over the transition, n = end - start.
    ramp = 1.0 - (idx - start + 1).to(torch.float32) / (end - start + 1).to(torch.float32)
    weights = torch.where(idx < start, torch.ones_like(ramp), ramp).clamp(0.0, 1.0)
    weights = torch.where(idx >= end, torch.zeros_like(weights), weights)
    if schedule is PrefixAttentionSchedule.EXP:
        # Convex, and fixes both endpoints (0 -> 0, 1 -> 1).
        weights = weights * torch.expm1(weights) / (math.e - 1.0)
    return weights


def guidance_weight(time: torch.Tensor | float, max_guidance_weight: float) -> torch.Tensor:
    """Clipped guidance weight `(t^2 + (1-t)^2) / (t (1-t))`, see `openpi.models.rtc.guidance_weight`."""
    t = torch.as_tensor(time, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    return torch.clamp((t**2 + (1.0 - t) ** 2) / (t * (1.0 - t)), max=max_guidance_weight)
