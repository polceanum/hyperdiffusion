from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import numpy as np


@dataclass(frozen=True)
class EpisodeBatch:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    family_name: List[str]
    family_instances: Optional[List] = None


class TaskFamily:
    name: str
    input_dim: int = 2
    task_type: str = "classification"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not implement reward")

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not implement dynamics")

    def rollout(
        self,
        policy_fn,
        initial_state: torch.Tensor,
        num_steps: int = 80,
        dt: float = 0.05,
    ) -> Dict[str, List[float]]:
        states, actions, rewards, cumulative_rewards = [], [], [], []
        state = initial_state.clone().to(torch.float32)
        total = 0.0
        for _ in range(num_steps):
            action = policy_fn(state)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu()
            action = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
            r = self.reward(state, action)
            state = self.dynamics(state, action, dt)
            states.append(state.detach().cpu().numpy())
            actions.append(float(action[0]))
            rewards.append(float(r))
            total += float(r)
            cumulative_rewards.append(total)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "cumulative_rewards": cumulative_rewards,
        }


def _rand_uniform(shape: tuple[int, ...], low: float, high: float, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.empty(*shape).uniform_(low, high, generator=generator)


def _randn(shape: tuple[int, ...], generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.randn(*shape, generator=generator)


def _rotation(angle: float) -> torch.Tensor:
    return torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=torch.float32)


def _sample_points(total: int, scale: float = 2.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return _rand_uniform((total, 2), -scale, scale, generator)


def _split_episode(x: torch.Tensor, y: torch.Tensor, support_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


# ----------------------------
# Classification families
# ----------------------------

class LinearFamily(TaskFamily):
    name = "linear"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        angle = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        normal = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
        bias = _rand_uniform((1,), -0.5, 0.5, generator).item()
        y = ((x @ normal) + bias > 0.0).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class XorFamily(TaskFamily):
    name = "xor"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 1.5, generator)
        angle = _rand_uniform((1,), 0.0, math.pi / 2.0, generator).item()
        x = x @ _rotation(angle).T
        y = ((x[:, 0] > 0.0) ^ (x[:, 1] > 0.0)).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class MoonsFamily(TaskFamily):
    name = "moons"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        n1 = total // 2
        n2 = total - n1
        theta1 = _rand_uniform((n1,), 0.0, math.pi, generator)
        theta2 = _rand_uniform((n2,), 0.0, math.pi, generator)
        moon1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=-1)
        moon2 = torch.stack([1.0 - torch.cos(theta2), -torch.sin(theta2) - 0.4], dim=-1)
        x = torch.cat([moon1, moon2], dim=0)
        x = x + 0.08 * _randn(x.shape, generator)
        angle = _rand_uniform((1,), -math.pi / 4, math.pi / 4, generator).item()
        x = x @ _rotation(angle).T
        y = torch.cat([torch.zeros(n1), torch.ones(n2)], dim=0).unsqueeze(-1)
        perm = torch.randperm(total, generator=generator)
        return _split_episode(x[perm], y[perm], support_size)


class CirclesFamily(TaskFamily):
    name = "circles"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        radius = _rand_uniform((1,), 0.7, 1.4, generator).item()
        y = (x.pow(2).sum(dim=-1) > radius**2).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class SineBoundaryFamily(TaskFamily):
    name = "sine"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        amp = _rand_uniform((1,), 0.5, 1.1, generator).item()
        freq = _rand_uniform((1,), 0.9, 1.8, generator).item()
        phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        shift = _rand_uniform((1,), -0.35, 0.35, generator).item()
        boundary = amp * torch.sin(freq * x[:, 0] + phase) + shift
        y = (x[:, 1] > boundary).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class DiamondFamily(TaskFamily):
    name = "diamond"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        radius = _rand_uniform((1,), 0.8, 1.5, generator).item()
        y = ((x[:, 0].abs() + x[:, 1].abs()) > radius).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class SpiralFamily(TaskFamily):
    name = "spiral"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        return _sample_spiral_family(support_size, query_size, generator, max_turns=2.7, pitch_low=0.16, pitch_high=0.28, noise_scale=0.08)


class GaussianMixtureFamily(TaskFamily):
    name = "gaussian"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        n0 = total // 2
        n1 = total - n0
        c0 = torch.tensor([-0.8, -0.6]) + 0.6 * _randn((2,), generator)
        c1 = torch.tensor([0.8, 0.6]) + 0.6 * _randn((2,), generator)
        x0 = c0 + 0.35 * _randn((n0, 2), generator)
        x1 = c1 + 0.35 * _randn((n1, 2), generator)
        x = torch.cat([x0, x1], dim=0)
        y = torch.cat([torch.zeros(n0), torch.ones(n1)], dim=0).unsqueeze(-1)
        perm = torch.randperm(total, generator=generator)
        return _split_episode(x[perm], y[perm], support_size)


class EllipseFamily(TaskFamily):
    name = "ellipse"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        a = _rand_uniform((1,), 0.7, 1.5, generator).item()
        b = _rand_uniform((1,), 0.4, 1.2, generator).item()
        angle = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        xr = x @ _rotation(angle).T
        value = (xr[:, 0] / a) ** 2 + (xr[:, 1] / b) ** 2
        threshold = _rand_uniform((1,), 0.8, 1.2, generator).item()
        y = (value > threshold).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class RotatedDiamondFamily(TaskFamily):
    name = "rotated_diamond"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        angle = _rand_uniform((1,), 0.2, 1.2, generator).item()
        xr = x @ _rotation(angle).T
        radius_x = _rand_uniform((1,), 0.8, 1.4, generator).item()
        radius_y = _rand_uniform((1,), 0.5, 1.1, generator).item()
        value = xr[:, 0].abs() / radius_x + xr[:, 1].abs() / radius_y
        y = (value > 1.0).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class ArcFamily(TaskFamily):
    name = "arc"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        center = _rand_uniform((2,), -0.4, 0.4, generator)
        xr = x - center
        radius = _rand_uniform((1,), 0.8, 1.5, generator).item()
        thickness = _rand_uniform((1,), 0.18, 0.35, generator).item()
        start = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        span = _rand_uniform((1,), math.pi / 2, math.pi * 1.25, generator).item()
        angle = torch.atan2(xr[:, 1], xr[:, 0])
        rel = torch.remainder(angle - start + 2 * math.pi, 2 * math.pi)
        angle_mask = rel < span
        radial = (xr.pow(2).sum(dim=-1).sqrt() - radius).abs() < thickness
        y = (angle_mask & radial).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class SineLowFreqFamily(TaskFamily):
    name = "sine_lowfreq"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        amp = _rand_uniform((1,), 0.35, 0.9, generator).item()
        freq = _rand_uniform((1,), 0.45, 0.9, generator).item()
        phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        slope = _rand_uniform((1,), -0.25, 0.25, generator).item()
        boundary = slope * x[:, 0] + amp * torch.sin(freq * x[:, 0] + phase)
        y = (x[:, 1] > boundary).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class RadialLobesFamily(TaskFamily):
    name = "radial_lobes"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        angle = torch.atan2(x[:, 1], x[:, 0])
        radius = x.pow(2).sum(dim=-1).sqrt()
        lobes = int(torch.randint(3, 6, (1,), generator=generator).item())
        threshold = 0.9 + 0.35 * torch.cos(lobes * angle + _rand_uniform((1,), -math.pi, math.pi, generator).item())
        y = (radius > threshold).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


def _sample_spiral_family(
    support_size: int,
    query_size: int,
    generator: Optional[torch.Generator],
    max_turns: float,
    pitch_low: float,
    pitch_high: float,
    noise_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = support_size + query_size
    t = _rand_uniform((total,), 0.2, max_turns * math.pi, generator)
    pitch = _rand_uniform((1,), pitch_low, pitch_high, generator).item()
    phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
    arm = torch.randint(0, 2, (total,), generator=generator).float()
    radius = pitch * t + noise_scale * _randn((total,), generator)
    angle = t + arm * math.pi + phase
    x = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
    x = x + (noise_scale * 0.75) * _randn(x.shape, generator)
    y = arm.unsqueeze(-1)
    perm = torch.randperm(total, generator=generator)
    return _split_episode(x[perm], y[perm], support_size)


class SpiralEasyFamily(TaskFamily):
    name = "spiral_easy"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        return _sample_spiral_family(support_size, query_size, generator, max_turns=1.3, pitch_low=0.32, pitch_high=0.42, noise_scale=0.04)


class SpiralMediumFamily(TaskFamily):
    name = "spiral_medium"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        return _sample_spiral_family(support_size, query_size, generator, max_turns=1.8, pitch_low=0.24, pitch_high=0.34, noise_scale=0.05)


class SpiralHardFamily(TaskFamily):
    name = "spiral_hard"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        return _sample_spiral_family(support_size, query_size, generator, max_turns=2.2, pitch_low=0.18, pitch_high=0.28, noise_scale=0.06)


class CheckerboardFamily(TaskFamily):
    name = "checkerboard"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        scale = _rand_uniform((1,), 0.55, 1.1, generator).item()
        angle = _rand_uniform((1,), 0.0, math.pi / 4, generator).item()
        xr = x @ _rotation(angle).T
        ix = torch.floor(xr[:, 0] / scale).long()
        iy = torch.floor(xr[:, 1] / scale).long()
        y = ((ix + iy) % 2 == 0).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class ConcentricRingsFamily(TaskFamily):
    name = "concentric_rings"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        r = x.pow(2).sum(dim=-1).sqrt()
        ring_width = _rand_uniform((1,), 0.45, 0.85, generator).item()
        phase = _rand_uniform((1,), 0.0, ring_width, generator).item()
        y = (torch.floor((r + phase) / ring_width).long() % 2 == 0).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class WedgesFamily(TaskFamily):
    name = "wedges"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        angle = torch.atan2(x[:, 1], x[:, 0])
        phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        n_wedges = int(torch.randint(3, 7, (1,), generator=generator).item())
        wedge_idx = torch.floor((angle - phase + math.pi) / (2 * math.pi / n_wedges)).long()
        y = (wedge_idx % 2 == 0).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


class ParallelBandsFamily(TaskFamily):
    name = "parallel_bands"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.2, generator)
        angle = _rand_uniform((1,), 0.0, math.pi, generator).item()
        width = _rand_uniform((1,), 0.5, 1.0, generator).item()
        phase = _rand_uniform((1,), 0.0, width, generator).item()
        proj = x[:, 0] * math.cos(angle) + x[:, 1] * math.sin(angle)
        y = (torch.floor((proj + phase) / width).long() % 2 == 0).float().unsqueeze(-1)
        return _split_episode(x, y, support_size)


CLASSIFICATION_FAMILIES: Dict[str, TaskFamily] = {
    family.name: family()
    for family in [
        LinearFamily,
        XorFamily,
        MoonsFamily,
        CirclesFamily,
        SineBoundaryFamily,
        DiamondFamily,
        SpiralFamily,
        GaussianMixtureFamily,
        EllipseFamily,
        RotatedDiamondFamily,
        ArcFamily,
        SineLowFreqFamily,
        RadialLobesFamily,
        SpiralEasyFamily,
        SpiralMediumFamily,
        SpiralHardFamily,
        CheckerboardFamily,
        ConcentricRingsFamily,
        WedgesFamily,
        ParallelBandsFamily,
    ]
}

CLASSIFICATION_FAMILIES_CLASSES: Dict[str, type] = {
    family().name: family
    for family in [
        LinearFamily,
        XorFamily,
        MoonsFamily,
        CirclesFamily,
        SineBoundaryFamily,
        DiamondFamily,
        SpiralFamily,
        GaussianMixtureFamily,
        EllipseFamily,
        RotatedDiamondFamily,
        ArcFamily,
        SineLowFreqFamily,
        RadialLobesFamily,
        SpiralEasyFamily,
        SpiralMediumFamily,
        SpiralHardFamily,
        CheckerboardFamily,
        ConcentricRingsFamily,
        WedgesFamily,
        ParallelBandsFamily,
    ]
}

DEFAULT_TRAIN_FAMILIES = ["linear", "xor", "moons", "circles", "sine", "diamond", "checkerboard", "concentric_rings"]
BRIDGE_FAMILIES = [
    "ellipse",
    "rotated_diamond",
    "arc",
    "sine_lowfreq",
    "radial_lobes",
    "spiral_easy",
    "spiral_medium",
    "spiral_hard",
    "wedges",
    "parallel_bands",
]
EXPANDED_TRAIN_FAMILIES = DEFAULT_TRAIN_FAMILIES + BRIDGE_FAMILIES
ORIGINAL_TRAIN_GROUPS: Dict[str, List[str]] = {
    "core": DEFAULT_TRAIN_FAMILIES,
    "bridges": BRIDGE_FAMILIES,
}

# ----------------------------
# Regression families
# ----------------------------


def _sample_1d_inputs(total: int, low: float = -3.0, high: float = 3.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return _rand_uniform((total, 1), low, high, generator)


class RegressionFamily(TaskFamily):
    input_dim: int = 1
    task_type: str = "regression"


class SineRegressionFamily(RegressionFamily):
    name = "sine_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -3.0, 3.0, generator)
        amp = _rand_uniform((1,), 0.6, 1.4, generator).item()
        freq = _rand_uniform((1,), 0.7, 1.7, generator).item()
        phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        slope = _rand_uniform((1,), -0.25, 0.25, generator).item()
        y = amp * torch.sin(freq * x + phase) + slope * x
        y = y + 0.03 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class PiecewiseRegressionFamily(RegressionFamily):
    name = "piecewise_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -3.0, 3.0, generator)
        # More distinct breakpoints and stronger slopes
        b1 = _rand_uniform((1,), -2.0, -1.0, generator).item()  # left breakpoint
        b2 = _rand_uniform((1,), 0.5, 1.5, generator).item()    # right breakpoint
        s1 = _rand_uniform((1,), -2.5, -1.2, generator).item()  # steeper negative
        s2 = _rand_uniform((1,), -0.3, 0.3, generator).item()   # flat middle
        s3 = _rand_uniform((1,), 1.2, 2.5, generator).item()    # steeper positive
        c = _rand_uniform((1,), -0.3, 0.3, generator).item()
        y = torch.where(x < b1, s1 * x + c, torch.where(x < b2, s2 * x + c, s3 * x + c))
        y = y + 0.008 * _randn(y.shape, generator)  # very low noise
        return _split_episode(x, y, support_size)


class QuadraticRegressionFamily(RegressionFamily):
    name = "quadratic_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -2.5, 2.5, generator)
        a = _rand_uniform((1,), -0.3, 0.3, generator).item()
        b = _rand_uniform((1,), -1.2, 1.2, generator).item()
        c = _rand_uniform((1,), -0.6, 0.6, generator).item()
        y = a * x.pow(2) + b * x + c
        y = y + 0.03 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class SawtoothRegressionFamily(RegressionFamily):
    name = "sawtooth_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -3.0, 3.0, generator)
        freq = _rand_uniform((1,), 0.3, 0.7, generator).item()  # even wider freq range
        phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        amp = _rand_uniform((1,), 3.0, 5.0, generator).item()  # much stronger amplitude
        raw = ((freq * x + phase) / (2.0 * math.pi))
        frac = raw - torch.floor(raw + 0.5)
        y = amp * 2.0 * frac  # stronger amplitude
        y = y + 0.005 * _randn(y.shape, generator)  # minimal noise
        return _split_episode(x, y, support_size)


class CubicRegressionFamily(RegressionFamily):
    name = "cubic_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -2.2, 2.2, generator)
        a = _rand_uniform((1,), -0.15, 0.15, generator).item()
        b = _rand_uniform((1,), -0.4, 0.4, generator).item()
        c = _rand_uniform((1,), -1.0, 1.0, generator).item()
        d = _rand_uniform((1,), -0.4, 0.4, generator).item()
        y = a * x.pow(3) + b * x.pow(2) + c * x + d
        y = y + 0.03 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class AbsRegressionFamily(RegressionFamily):
    name = "abs_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -3.0, 3.0, generator)
        shift = _rand_uniform((1,), -0.8, 0.8, generator).item()
        scale = _rand_uniform((1,), 0.6, 1.4, generator).item()
        tilt = _rand_uniform((1,), -0.3, 0.3, generator).item()
        y = scale * (x - shift).abs() + tilt * x
        y = y + 0.03 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class DampedSineFamily(RegressionFamily):
    name = "damped_sine"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -3.0, 3.0, generator)
        amp = _rand_uniform((1,), 0.8, 1.5, generator).item()
        freq = _rand_uniform((1,), 0.8, 2.0, generator).item()
        phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        decay = _rand_uniform((1,), 0.3, 0.8, generator).item()
        y = amp * torch.exp(-decay * x.abs()) * torch.sin(freq * x + phase)
        y = y + 0.03 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class GaussianBumpFamily(RegressionFamily):
    name = "gaussian_bump"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -3.0, 3.0, generator)
        n_bumps = int(torch.randint(2, 5, (1,), generator=generator).item())
        y = torch.zeros(total, 1)
        for _ in range(n_bumps):
            center = _rand_uniform((1,), -2.5, 2.5, generator).item()
            width = _rand_uniform((1,), 0.3, 1.0, generator).item()
            height = _rand_uniform((1,), -1.5, 1.5, generator).item()
            y = y + height * torch.exp(-0.5 * ((x - center) / width) ** 2)
        y = y + 0.02 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class StepRegressionFamily(RegressionFamily):
    name = "step_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -3.0, 3.0, generator)
        n_steps = int(torch.randint(3, 6, (1,), generator=generator).item())
        bps = torch.sort(_rand_uniform((n_steps - 1,), -2.5, 2.5, generator)).values
        heights = _rand_uniform((n_steps,), -1.5, 1.5, generator)
        y = torch.zeros(total, 1)
        for i in range(total):
            bucket = int((bps < x[i, 0]).sum().item())
            y[i, 0] = heights[bucket]
        y = y + 0.02 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class ExpRegressionFamily(RegressionFamily):
    name = "exp_regression"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_1d_inputs(total, -2.5, 2.5, generator)
        rate = _rand_uniform((1,), -0.7, 0.7, generator).item()
        amp = _rand_uniform((1,), 0.4, 1.2, generator).item()
        offset = _rand_uniform((1,), -0.6, 0.6, generator).item()
        y = amp * torch.exp(rate * x) + offset
        y = torch.clamp(y, -4.0, 4.0)
        y = y + 0.03 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class BanditLinearFamily(RegressionFamily):
    name = "bandit_linear"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        # Strong linear in both dimensions including a cross term for some 2D curvature effect
        self.coeff1 = _rand_uniform((1,), 2.5, 4.0, generator).item()
        self.coeff2 = _rand_uniform((1,), -2.5, 2.5, generator).item()
        self.coeff_cross = _rand_uniform((1,), -1.5, 1.5, generator).item()
        self.bias = _rand_uniform((1,), -0.2, 0.2, generator).item()
        y = (self.coeff1 * x[:, 0:1] + self.coeff2 * x[:, 1:2] + self.coeff_cross * x[:, 0:1] * x[:, 1:2] + self.bias)
        y = y + 0.005 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (n,2)
        return (self.coeff1 * x[:, 0:1] + self.coeff2 * x[:, 1:2] + self.coeff_cross * x[:, 0:1] * x[:, 1:2] + self.bias).unsqueeze(-1)


class BanditQuadraticFamily(RegressionFamily):
    name = "bandit_quadratic"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        # Strong 2D quadratic surface
        self.coeff_lin_x = _rand_uniform((1,), -0.8, 0.8, generator).item()
        self.coeff_lin_y = _rand_uniform((1,), -0.8, 0.8, generator).item()
        self.coeff_quad1 = _rand_uniform((1,), -5.0, 5.0, generator).item()
        self.coeff_quad2 = _rand_uniform((1,), -5.0, 5.0, generator).item()
        self.coeff_xy = _rand_uniform((1,), -2.0, 2.0, generator).item()
        self.bias = _rand_uniform((1,), -0.1, 0.1, generator).item()
        y = (
            self.coeff_lin_x * x[:, 0:1]
            + self.coeff_lin_y * x[:, 1:2]
            + self.coeff_quad1 * (x[:, 0:1] ** 2)
            + self.coeff_quad2 * (x[:, 1:2] ** 2)
            + self.coeff_xy * x[:, 0:1] * x[:, 1:2]
            + self.bias
        )
        y = y + 0.001 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.coeff_lin_x * x[:, 0:1]
            + self.coeff_lin_y * x[:, 1:2]
            + self.coeff_quad1 * (x[:, 0:1] ** 2)
            + self.coeff_quad2 * (x[:, 1:2] ** 2)
            + self.coeff_xy * x[:, 0:1] * x[:, 1:2]
            + self.bias
        ).unsqueeze(-1)


class BanditShiftedFamily(RegressionFamily):
    name = "bandit_shifted"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        self.shift_x = 0.5
        self.shift_y = -0.5

        self.coeff1 = _rand_uniform((1,), 2.0, 4.0, generator).item()
        self.coeff2 = _rand_uniform((1,), -2.0, 2.0, generator).item()
        self.coeff_cross = _rand_uniform((1,), -1.0, 1.0, generator).item()
        self.coeff_quad = _rand_uniform((1,), 0.8, 1.8, generator).item()
        self.bias = _rand_uniform((1,), -0.3, 0.3, generator).item()

        y = (
            self.coeff1 * (x[:, 0:1] + self.shift_x)
            + self.coeff2 * (x[:, 1:2] + self.shift_y)
            + self.coeff_cross * (x[:, 0:1] * x[:, 1:2])
            + self.coeff_quad * ((x[:, 0:1] + self.shift_x) ** 2 + (x[:, 1:2] + self.shift_y) ** 2)
            + self.bias
        )
        y = y + 0.005 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.coeff1 * (x[:, 0:1] + self.shift_x)
            + self.coeff2 * (x[:, 1:2] + self.shift_y)
            + self.coeff_cross * (x[:, 0:1] * x[:, 1:2])
            + self.coeff_quad * ((x[:, 0:1] + self.shift_x) ** 2 + (x[:, 1:2] + self.shift_y) ** 2)
            + self.bias
        ).unsqueeze(-1)


class BanditSineRadialFamily(RegressionFamily):
    name = "bandit_sine_radial"
    input_dim = 2

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        self.freq = _rand_uniform((1,), 1.0, 2.5, generator).item()
        self.amp = _rand_uniform((1,), 1.0, 2.5, generator).item()
        self.phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        r = x.pow(2).sum(dim=-1, keepdim=True).sqrt()
        y = self.amp * torch.sin(self.freq * r + self.phase)
        y = y + 0.01 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        r = x.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return (self.amp * torch.sin(self.freq * r + self.phase)).unsqueeze(-1)


class BanditGaussianFamily(RegressionFamily):
    name = "bandit_gaussian"
    input_dim = 2

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        self.center = _rand_uniform((2,), -1.0, 1.0, generator)
        self.sigma = _rand_uniform((1,), 0.3, 1.0, generator).item()
        self.amp = _rand_uniform((1,), 1.5, 3.5, generator).item()
        diff = x - self.center.unsqueeze(0)
        r2 = diff.pow(2).sum(dim=-1, keepdim=True)
        y = self.amp * torch.exp(-r2 / (2 * self.sigma ** 2))
        y = y + 0.01 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.center.unsqueeze(0)
        r2 = diff.pow(2).sum(dim=-1, keepdim=True)
        return (self.amp * torch.exp(-r2 / (2 * self.sigma ** 2))).unsqueeze(-1)


class BanditCubicFamily(RegressionFamily):
    name = "bandit_cubic"
    input_dim = 2

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        self.a = _rand_uniform((1,), -1.0, 1.0, generator).item()
        self.b = _rand_uniform((1,), -1.5, 1.5, generator).item()
        self.c = _rand_uniform((1,), -1.0, 1.0, generator).item()
        self.cross = _rand_uniform((1,), -1.0, 1.0, generator).item()
        y = (self.a * x[:, 0:1] ** 3 + self.b * x[:, 1:2] ** 2
             + self.c * x[:, 0:1] * x[:, 1:2] + self.cross * x[:, 1:2] ** 3)
        y = y + 0.01 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a * x[:, 0:1] ** 3 + self.b * x[:, 1:2] ** 2
                + self.c * x[:, 0:1] * x[:, 1:2] + self.cross * x[:, 1:2] ** 3).unsqueeze(-1)


class LinearControlFamily(RegressionFamily):
    name = "linear_control"
    input_dim = 2  # state: position, velocity
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.0, generator)
        self.k1 = _rand_uniform((1,), 1.0, 3.0, generator).item()
        self.k2 = _rand_uniform((1,), 0.5, 2.0, generator).item()
        actions = -self.k1 * states[:, 0:1] - self.k2 * states[:, 1:2]
        actions = actions + 0.01 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return (-self.k1 * x[:, 0:1] - self.k2 * x[:, 1:2]).unsqueeze(-1)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.1 * v ** 2 + 0.01 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + (-x - 0.1 * v + action[0]) * dt
        return torch.stack([x_next, v_next])


class LinearControlShiftedFamily(RegressionFamily):
    name = "linear_control_shifted"
    input_dim = 2
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.0, generator)
        self.k1 = _rand_uniform((1,), 1.2, 3.2, generator).item()
        self.k2 = _rand_uniform((1,), 0.7, 2.2, generator).item()
        actions = -self.k1 * states[:, 0:1] - self.k2 * states[:, 1:2]
        actions = actions + 0.01 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return (-self.k1 * x[:, 0:1] - self.k2 * x[:, 1:2]).unsqueeze(-1)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.1 * v ** 2 + 0.01 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + (-x - 0.1 * v + action[0]) * dt
        return torch.stack([x_next, v_next])


class PendulumControlFamily(RegressionFamily):
    name = "pendulum_control"
    input_dim = 3  # state: cos(theta), sin(theta), omega
    task_type = "control"

    def __init__(self):
        super().__init__()
        # Hardcoded LQR gain for linearized pendulum (approximate)
        self.K = torch.tensor([[15.0, 5.0]], dtype=torch.float32)

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        # Sample states: theta, omega
        theta = _rand_uniform((total,), -np.pi, np.pi, generator)
        omega = _rand_uniform((total,), -8, 8, generator)
        # Convert to cos, sin, omega
        states = torch.stack([torch.cos(theta), torch.sin(theta), omega], dim=1)
        # Optimal control for linearized: u = -K @ [theta, omega]
        # But since we have cos, sin, for small theta, cos~1, sin~theta
        # Approximate theta ≈ sin(theta) / cos(theta) but for small, theta ≈ sin
        # For simplicity, assume small angles, use sin as theta
        theta_approx = torch.asin(torch.clamp(states[:, 1], -1, 1))
        state_lin = torch.stack([theta_approx, states[:, 2]], dim=1)
        actions = - (self.K @ state_lin.T).T
        actions = actions + 0.01 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # reward for stabilization around upright (theta=0)
        cos_t, sin_t, omega = state[0], state[1], state[2]
        theta = torch.atan2(sin_t, cos_t)
        return -(theta ** 2 + 0.1 * omega ** 2 + 0.01 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        g = 10.0
        l = 1.0
        cos_t, sin_t, omega = state[0], state[1], state[2]
        theta = torch.atan2(sin_t, cos_t)
        theta_next = theta + omega * dt
        omega_next = omega + (- (g / l) * torch.sin(theta) - 0.1 * omega + action[0]) * dt
        return torch.stack([torch.cos(theta_next), torch.sin(theta_next), omega_next])


class NonlinearControlFamily(RegressionFamily):
    name = "nonlinear_control"
    input_dim = 2  # state: position, velocity
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        # Sample states: position and velocity
        states = _sample_points(total, 2.0, generator)  # [-2, 2] for both
        # For nonlinear system: \dot{x} = v, \dot{v} = -x - 0.1*v + u + 0.1*x^2
        # Approximate optimal control using LQR on linearized system
        # Linearized: A = [[0,1], [-1, -0.1]], B = [[0], [1]]
        # But since nonlinear, use the same gains as linear for simplicity
        k1 = _rand_uniform((1,), 1.0, 3.0, generator).item()
        k2 = _rand_uniform((1,), 0.5, 2.0, generator).item()
        actions = -k1 * states[:, 0:1] - k2 * states[:, 1:2]
        # Add some noise and perhaps adjust for nonlinearity
        actions = actions + 0.01 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.1 * v ** 2 + 0.01 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + (-x - 0.1 * v + action[0] + 0.1 * x ** 2) * dt
        return torch.stack([x_next, v_next])


class LinearControlStiffFamily(RegressionFamily):
    name = "linear_control_stiff"
    input_dim = 2
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.4, generator)
        self.k1 = _rand_uniform((1,), 2.2, 4.2, generator).item()
        self.k2 = _rand_uniform((1,), 1.0, 2.8, generator).item()
        actions = -self.k1 * states[:, 0:1] - self.k2 * states[:, 1:2]
        actions = actions + 0.02 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(1.3 * x ** 2 + 0.2 * v ** 2 + 0.03 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + (-1.6 * x - 0.25 * v + action[0]) * dt
        return torch.stack([x_next, v_next])


class LinearControlDriftFamily(RegressionFamily):
    name = "linear_control_drift"
    input_dim = 2
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.2, generator)
        self.k1 = _rand_uniform((1,), 1.4, 3.4, generator).item()
        self.k2 = _rand_uniform((1,), 0.8, 2.4, generator).item()
        actions = -self.k1 * states[:, 0:1] - self.k2 * states[:, 1:2]
        actions = actions + 0.02 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.15 * v ** 2 + 0.02 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        drift = 0.18 * torch.sin(1.3 * x) + 0.08 * v * x
        x_next = x + v * dt
        v_next = v + (-x - 0.12 * v + action[0] + drift) * dt
        return torch.stack([x_next, v_next])


class NonlinearControlHardFamily(RegressionFamily):
    name = "nonlinear_control_hard"
    input_dim = 2
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.5, generator)
        k1 = _rand_uniform((1,), 1.8, 3.6, generator).item()
        k2 = _rand_uniform((1,), 1.0, 2.8, generator).item()
        actions = -k1 * states[:, 0:1] - k2 * states[:, 1:2]
        actions = actions + 0.03 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(1.15 * x ** 2 + 0.2 * v ** 2 + 0.03 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        nonlin = 0.18 * x ** 2 - 0.04 * x ** 3 + 0.08 * torch.sin(2.0 * x)
        v_next = v + (-1.1 * x - 0.18 * v + action[0] + nonlin) * dt
        return torch.stack([x_next, v_next])


class LinearControlNoisyShiftedFamily(RegressionFamily):
    name = "linear_control_noisy_shifted"
    input_dim = 2
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.6, generator)
        self.k1 = _rand_uniform((1,), 1.6, 3.8, generator).item()
        self.k2 = _rand_uniform((1,), 0.9, 2.7, generator).item()
        actions = -self.k1 * states[:, 0:1] - self.k2 * states[:, 1:2]
        actions = actions + 0.04 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(1.1 * x ** 2 + 0.18 * v ** 2 + 0.03 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        process_noise = 0.03 * torch.tanh(2.0 * x) + 0.015 * torch.sin(3.0 * v)
        x_next = x + v * dt
        v_next = v + (-1.15 * x - 0.15 * v + action[0] + process_noise) * dt
        return torch.stack([x_next, v_next])


class VanDerPolControlFamily(RegressionFamily):
    name = "van_der_pol"
    input_dim = 2
    task_type = "control"
    mu = 0.7  # Fixed parameter (average of 0.2-1.2 range)

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.5, generator)
        k1 = _rand_uniform((1,), 1.2, 2.8, generator).item()
        k2 = _rand_uniform((1,), 0.6, 2.0, generator).item()
        actions = -k1 * states[:, 0:1] - k2 * states[:, 1:2]
        actions = actions + 0.02 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.1 * v ** 2 + 0.01 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + (self.mu * (1.0 - x ** 2) * v - x + action[0]) * dt
        return torch.stack([x_next, v_next])


class DuffingControlFamily(RegressionFamily):
    name = "duffing_control"
    input_dim = 2
    task_type = "control"
    alpha = 0.0  # Fixed parameter (average of -0.5 to 0.5 range)
    beta = 1.0   # Fixed parameter (average of 0.5 to 1.5 range)

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.0, generator)
        k1 = _rand_uniform((1,), 1.5, 3.5, generator).item()
        k2 = _rand_uniform((1,), 0.8, 2.2, generator).item()
        actions = -k1 * states[:, 0:1] - k2 * states[:, 1:2]
        actions = actions + 0.02 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.1 * v ** 2 + 0.01 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + (-self.alpha * x - self.beta * x ** 3 - 0.1 * v + action[0]) * dt
        return torch.stack([x_next, v_next])


class DoubleIntegratorFamily(RegressionFamily):
    name = "double_integrator"
    input_dim = 2
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.0, generator)
        k1 = _rand_uniform((1,), 1.0, 3.0, generator).item()
        k2 = _rand_uniform((1,), 1.5, 3.5, generator).item()
        actions = -k1 * states[:, 0:1] - k2 * states[:, 1:2]
        actions = actions + 0.01 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.05 * v ** 2 + 0.005 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + action[0] * dt
        return torch.stack([x_next, v_next])


class OverdampedControlFamily(RegressionFamily):
    name = "overdamped_control"
    input_dim = 2
    task_type = "control"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.2, generator)
        # Use fixed average parameters for consistency with dynamics()
        k1 = 1.4  # Fixed (average of 0.8-2.0 range)
        k2 = 3.75  # Fixed (average of 2.5-5.0 range)
        actions = -k1 * states[:, 0:1] - k2 * states[:, 1:2]
        actions = actions + 0.01 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.3 * v ** 2 + 0.02 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        x_next = x + v * dt
        v_next = v + (-x - 0.8 * v + action[0]) * dt
        return torch.stack([x_next, v_next])


class AsymmetricControlFamily(RegressionFamily):
    name = "asymmetric_control"
    input_dim = 2
    task_type = "control"
    kp = 2.0  # Fixed (average of 1.0-3.0 range)
    kn = 0.9  # Fixed (average of 0.3-1.5 range)
    kv = 1.15  # Fixed (average of 0.5-1.8 range)

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        states = _sample_points(total, 2.0, generator)
        pos_mask = (states[:, 0:1] > 0).float()
        actions = -(self.kp * pos_mask + self.kn * (1.0 - pos_mask)) * states[:, 0:1] - self.kv * states[:, 1:2]
        actions = actions + 0.02 * _randn(actions.shape, generator)
        return _split_episode(states, actions, support_size)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[0], state[1]
        return -(x ** 2 + 0.1 * v ** 2 + 0.01 * action[0] ** 2)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        x, v = state[0], state[1]
        restoring = torch.where(x > 0, -1.5 * x, -0.7 * x)
        x_next = x + v * dt
        v_next = v + (restoring - 0.1 * v + action[0]) * dt
        return torch.stack([x_next, v_next])


REGRESSION_FAMILIES: Dict[str, TaskFamily] = {
    family.name: family()
    for family in [
        SineRegressionFamily,
        PiecewiseRegressionFamily,
        QuadraticRegressionFamily,
        SawtoothRegressionFamily,
        CubicRegressionFamily,
        AbsRegressionFamily,
        DampedSineFamily,
        GaussianBumpFamily,
        StepRegressionFamily,
        ExpRegressionFamily,
    ]
}

REGRESSION_FAMILIES_CLASSES: Dict[str, type] = {
    family().name: family
    for family in [
        SineRegressionFamily,
        PiecewiseRegressionFamily,
        QuadraticRegressionFamily,
        SawtoothRegressionFamily,
        CubicRegressionFamily,
        AbsRegressionFamily,
        DampedSineFamily,
        GaussianBumpFamily,
        StepRegressionFamily,
        ExpRegressionFamily,
    ]
}

CONTROL_FAMILIES: Dict[str, TaskFamily] = {
    family.name: family()
    for family in [
        LinearControlFamily,
        LinearControlShiftedFamily,
        NonlinearControlFamily,
        LinearControlStiffFamily,
        LinearControlDriftFamily,
        NonlinearControlHardFamily,
        LinearControlNoisyShiftedFamily,
        PendulumControlFamily,
        VanDerPolControlFamily,
        DuffingControlFamily,
        DoubleIntegratorFamily,
        OverdampedControlFamily,
        AsymmetricControlFamily,
    ]
}

CONTROL_FAMILIES_CLASSES: Dict[str, type] = {
    family().name: family
    for family in [
        LinearControlFamily,
        LinearControlShiftedFamily,
        NonlinearControlFamily,
        LinearControlStiffFamily,
        LinearControlDriftFamily,
        NonlinearControlHardFamily,
        LinearControlNoisyShiftedFamily,
        PendulumControlFamily,
        VanDerPolControlFamily,
        DuffingControlFamily,
        DoubleIntegratorFamily,
        OverdampedControlFamily,
        AsymmetricControlFamily,
    ]
}

BANDIT_FAMILIES: Dict[str, TaskFamily] = {
    family.name: family()
    for family in [
        BanditLinearFamily,
        BanditQuadraticFamily,
        BanditShiftedFamily,
        BanditSineRadialFamily,
        BanditGaussianFamily,
        BanditCubicFamily,
    ]
}

BANDIT_FAMILIES_CLASSES: Dict[str, type] = {
    family().name: family
    for family in [
        BanditLinearFamily,
        BanditQuadraticFamily,
        BanditShiftedFamily,
        BanditSineRadialFamily,
        BanditGaussianFamily,
        BanditCubicFamily,
    ]
}

DEFAULT_REGRESSION_TRAIN_FAMILIES = [
    "sine_regression",
    "piecewise_regression",
    "quadratic_regression",
    "sawtooth_regression",
    "damped_sine",
    "gaussian_bump",
]
DEFAULT_REGRESSION_EVAL_FAMILIES = ["cubic_regression", "abs_regression", "step_regression", "exp_regression"]
REGRESSION_TRAIN_GROUPS: Dict[str, List[str]] = {
    "core": DEFAULT_REGRESSION_TRAIN_FAMILIES,
    "held_out": DEFAULT_REGRESSION_EVAL_FAMILIES,
}

DEFAULT_CONTROL_TRAIN_FAMILIES = [
    "linear_control",
    "nonlinear_control",
    "linear_control_stiff",
    "linear_control_drift",
    "van_der_pol",
    "duffing_control",
    "double_integrator",
    "overdamped_control",
]
DEFAULT_CONTROL_EVAL_FAMILIES = [
    "linear_control_shifted",
    "nonlinear_control_hard",
    "linear_control_noisy_shifted",
    "asymmetric_control",
]
CONTROL_TRAIN_GROUPS: Dict[str, List[str]] = {
    "core": DEFAULT_CONTROL_TRAIN_FAMILIES,
    "held_out": DEFAULT_CONTROL_EVAL_FAMILIES,
}

TASK_TEXT_DESCRIPTIONS: Dict[str, str] = {
    "linear_control": "Linear second-order control task with state [position, velocity]. Stabilize near zero using smooth negative feedback control.",
    "linear_control_shifted": "Shifted linear second-order control task with changed gain ranges. Stabilize position and velocity near zero under mild dynamics shift.",
    "nonlinear_control": "Nonlinear control task with quadratic state coupling in acceleration dynamics. Learn corrective control from support trajectories.",
    "linear_control_stiff": "Stiff linear oscillator control with stronger restoring force and damping. Requires precise actions to avoid overshoot.",
    "linear_control_drift": "Linear control with structured state-dependent drift disturbance. Policy must compensate for sinusoidal and multiplicative drift.",
    "nonlinear_control_hard": "Hard nonlinear control regime with stronger polynomial and sinusoidal dynamics. Robust stabilization under sharper nonlinearities.",
    "linear_control_noisy_shifted": "Shifted linear control with process perturbations and noisier actions. Generalize to noisy, shifted dynamics.",
    "pendulum_control": "Pendulum swing stabilization task using [cos(theta), sin(theta), angular velocity] state representation.",
    "van_der_pol": "Van der Pol oscillator control with nonlinear damping. Mu parameter varies the limit cycle strength. Stabilize near origin.",
    "duffing_control": "Duffing oscillator control with cubic nonlinearity. Alpha and beta parameters vary the double-well potential shape.",
    "double_integrator": "Pure double integrator control. No natural restoring force; requires policy to provide all stabilization.",
    "overdamped_control": "Overdamped linear control with heavy velocity damping. Critically or over-damped dynamics require low-gain stabilization.",
    "asymmetric_control": "Asymmetric control task with different restoring forces for positive and negative positions. Policy must account for left-right asymmetry.",
}

DEFAULT_BANDIT_TRAIN_FAMILIES = ["bandit_linear", "bandit_quadratic", "bandit_sine_radial", "bandit_gaussian"]
DEFAULT_BANDIT_EVAL_FAMILIES = ["bandit_shifted", "bandit_cubic"]
BANDIT_TRAIN_GROUPS: Dict[str, List[str]] = {
    "core": DEFAULT_BANDIT_TRAIN_FAMILIES,
    "held_out": DEFAULT_BANDIT_EVAL_FAMILIES,
}


def get_registry(task_type: str) -> Dict[str, TaskFamily]:
    if task_type == "classification":
        return CLASSIFICATION_FAMILIES
    if task_type == "regression":
        return REGRESSION_FAMILIES
    if task_type == "bandit_regression":
        return BANDIT_FAMILIES
    if task_type == "control":
        return CONTROL_FAMILIES
    raise ValueError(f"Unknown task_type: {task_type}")


def make_episode_batch(
    batch_size: int,
    support_size: int,
    query_size: int,
    family_names: Iterable[str],
    generator: Optional[torch.Generator] = None,
    task_type: str = "classification",
) -> EpisodeBatch:
    # Get the class registry for the task type
    if task_type == "classification":
        class_registry = CLASSIFICATION_FAMILIES_CLASSES
    elif task_type == "regression":
        class_registry = REGRESSION_FAMILIES_CLASSES
    elif task_type == "control":
        class_registry = CONTROL_FAMILIES_CLASSES
    elif task_type == "bandit_regression":
        class_registry = BANDIT_FAMILIES_CLASSES
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    selected = list(family_names)
    if not selected:
        raise ValueError("family_names must not be empty")

    support_x, support_y, query_x, query_y, names, families_used = [], [], [], [], [], []
    for _ in range(batch_size):
        idx = torch.randint(0, len(selected), (1,), generator=generator).item()
        family_name = selected[idx]
        
        # Create a fresh instance for this batch element
        if family_name not in class_registry:
            raise ValueError(f"Unknown family: {family_name}")
        family = class_registry[family_name]()
        
        sx, sy, qx, qy = family.sample_episode(support_size=support_size, query_size=query_size, generator=generator)
        support_x.append(sx)
        support_y.append(sy)
        query_x.append(qx)
        query_y.append(qy)
        names.append(family.name)
        families_used.append(family)

    return EpisodeBatch(
        support_x=torch.stack(support_x, dim=0),
        support_y=torch.stack(support_y, dim=0),
        query_x=torch.stack(query_x, dim=0),
        query_y=torch.stack(query_y, dim=0),
        family_name=names,
        family_instances=families_used,
    )
