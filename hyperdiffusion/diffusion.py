from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    num_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        beta = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar

    def to(self, device: torch.device) -> "DiffusionSchedule":
        new = DiffusionSchedule(self.num_steps, self.beta_start, self.beta_end)
        new.beta = self.beta.to(device)
        new.alpha = self.alpha.to(device)
        new.alpha_bar = self.alpha_bar.to(device)
        return new

    def q_sample(self, z0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bar[t_idx].unsqueeze(-1)
        return alpha_bar_t.sqrt() * z0 + (1.0 - alpha_bar_t).sqrt() * noise

    def predict_z0(self, z_t: torch.Tensor, t_idx: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bar[t_idx].unsqueeze(-1)
        return (z_t - (1.0 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt().clamp_min(1e-6)


def ddim_sample(denoiser, schedule: DiffusionSchedule, context: torch.Tensor, latent_dim: int) -> torch.Tensor:
    device = context.device
    batch_size = context.shape[0]
    z = torch.randn(batch_size, latent_dim, device=device)

    for step in reversed(range(schedule.num_steps)):
        t_idx = torch.full((batch_size,), step, device=device, dtype=torch.long)
        t = t_idx.float() / max(schedule.num_steps - 1, 1)
        pred_noise = denoiser(z, t, context)
        z0 = schedule.predict_z0(z, t_idx, pred_noise)

        if step == 0:
            z = z0
            continue

        alpha_bar_prev = schedule.alpha_bar[step - 1]
        z = alpha_bar_prev.sqrt() * z0 + (1.0 - alpha_bar_prev).sqrt() * pred_noise

    return z
