from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class EpisodeBatch:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    family_name: List[str]


class TaskFamily:
    name: str

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


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
        total = support_size + query_size
        t = _rand_uniform((total,), 0.3, 2.7 * math.pi, generator)
        pitch = _rand_uniform((1,), 0.16, 0.28, generator).item()
        arm_phase = _rand_uniform((1,), -math.pi, math.pi, generator).item()
        arm = torch.randint(0, 2, (total,), generator=generator).float()
        r = pitch * t + 0.08 * _randn((total,), generator)
        angle = t + arm * math.pi + arm_phase
        x = torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=-1)
        x = x + 0.05 * _randn(x.shape, generator)
        y = arm.unsqueeze(-1)
        perm = torch.randperm(total, generator=generator)
        return _split_episode(x[perm], y[perm], support_size)


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


FAMILIES: Dict[str, TaskFamily] = {
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
    ]
}


DEFAULT_TRAIN_FAMILIES = ["linear", "xor", "moons", "circles"]
BRIDGE_FAMILIES = [
    "ellipse",
    "rotated_diamond",
    "arc",
    "sine_lowfreq",
    "radial_lobes",
    "spiral_easy",
    "spiral_medium",
    "spiral_hard",
]
EXPANDED_TRAIN_FAMILIES = DEFAULT_TRAIN_FAMILIES + BRIDGE_FAMILIES
ALL_FAMILIES = list(FAMILIES.keys())
OOD_FAMILIES = [name for name in ALL_FAMILIES if name not in DEFAULT_TRAIN_FAMILIES]


ORIGINAL_TRAIN_GROUPS: Dict[str, List[str]] = {
    "core": DEFAULT_TRAIN_FAMILIES,
    "bridges": BRIDGE_FAMILIES,
}


def make_episode_batch(
    batch_size: int,
    support_size: int,
    query_size: int,
    family_names: Iterable[str],
    generator: Optional[torch.Generator] = None,
) -> EpisodeBatch:
    selected = list(family_names)
    if not selected:
        raise ValueError("family_names must not be empty")

    support_x, support_y, query_x, query_y, names = [], [], [], [], []
    for _ in range(batch_size):
        idx = torch.randint(0, len(selected), (1,), generator=generator).item()
        family = FAMILIES[selected[idx]]
        sx, sy, qx, qy = family.sample_episode(support_size=support_size, query_size=query_size, generator=generator)
        support_x.append(sx)
        support_y.append(sy)
        query_x.append(qx)
        query_y.append(qy)
        names.append(family.name)

    return EpisodeBatch(
        support_x=torch.stack(support_x, dim=0),
        support_y=torch.stack(support_y, dim=0),
        query_x=torch.stack(query_x, dim=0),
        query_y=torch.stack(query_y, dim=0),
        family_name=names,
    )
