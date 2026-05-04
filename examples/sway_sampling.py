"""Minimum-code reference for Sway Sampling (F5-TTS / Irodori-TTS-v2).

Convention: t=1 is noise, t=0 is data.

A linear schedule samples t uniformly. Sway Sampling reparametrizes the
uniform grid u in [0, 1] via

    u' = u + s * (cos(pi/2 * u) + u - 1)

and then uses t = (1 - u') * init_scale. Negative s densifies the noise
side (early steps), positive s densifies the data side. s in [-1, 1].

This is a training-free inference-time schedule for Rectified Flow /
Flow Matching models — no retraining required.
"""
from __future__ import annotations

import math

import torch


def sway_t_schedule(
    num_steps: int,
    *,
    sway_coeff: float = -1.0,
    init_scale: float = 0.999,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a 1-D tensor of length num_steps + 1, monotonically decreasing
    from init_scale to 0 (noise -> data)."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    u = u + float(sway_coeff) * (torch.cos(0.5 * math.pi * u) + u - 1.0)
    u = u.clamp(0.0, 1.0)
    return (1.0 - u) * float(init_scale)


@torch.no_grad()
def euler_sample(
    velocity_fn,
    x_noise: torch.Tensor,
    *,
    num_steps: int = 6,
    sway_coeff: float = -1.0,
) -> torch.Tensor:
    """Minimal Euler integrator on a Rectified Flow model with a Sway schedule.

    velocity_fn(x_t, t) -> v: predicts dx/dt of the conditional flow at time t.
    x_noise: initial sample from the prior (typically N(0, I)).
    """
    t = sway_t_schedule(num_steps, sway_coeff=sway_coeff, device=x_noise.device)
    x = x_noise
    for i in range(num_steps):
        t_cur, t_next = t[i], t[i + 1]
        v = velocity_fn(x, t_cur)
        x = x + (t_next - t_cur) * v
    return x


if __name__ == "__main__":
    for s in (-1.0, 0.0, 1.0):
        sched = sway_t_schedule(6, sway_coeff=s)
        print(f"sway_coeff={s:+.1f}  t = {[round(v, 4) for v in sched.tolist()]}")
