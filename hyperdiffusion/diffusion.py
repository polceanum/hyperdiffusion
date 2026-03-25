from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps: int = 20, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.num_steps = num_steps
        beta = torch.linspace(beta_start, beta_end, num_steps)
        alpha = 1.0 - beta
        abar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("abar", abar)

    def q_sample(self, z0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        abar_t = self.abar[t_idx].unsqueeze(-1)
        return abar_t.sqrt() * z0 + (1.0 - abar_t).sqrt() * noise

    def predict_z0(self, z_t: torch.Tensor, t_idx: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        abar_t = self.abar[t_idx].unsqueeze(-1)
        return (z_t - (1.0 - abar_t).sqrt() * pred_noise) / abar_t.sqrt().clamp_min(1e-6)


def ddim_sample(denoiser: nn.Module, schedule: DiffusionSchedule, context: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
    device = context.device
    b = context.shape[0]
    steps = num_steps or schedule.num_steps
    z = torch.randn(b, denoiser.out.out_features if hasattr(denoiser.out, 'out_features') else denoiser.out.weight.shape[0], device=device)
    time_indices = torch.linspace(schedule.num_steps - 1, 0, steps, device=device).long()
    for i, t_idx in enumerate(time_indices):
        t = torch.full((b,), float(t_idx.item()) / max(schedule.num_steps - 1, 1), device=device)
        pred_noise = denoiser(z, t, context)
        abar_t = schedule.abar[t_idx]
        z0 = (z - (1.0 - abar_t).sqrt() * pred_noise) / abar_t.sqrt().clamp_min(1e-6)
        if i == len(time_indices) - 1:
            z = z0
        else:
            t_prev = time_indices[i + 1]
            abar_prev = schedule.abar[t_prev]
            z = abar_prev.sqrt() * z0 + (1.0 - abar_prev).sqrt() * pred_noise
    return z
