from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class EpisodeBatch:
    support_x: torch.Tensor  # [B, K, 2]
    support_y: torch.Tensor  # [B, K, 1]
    query_x: torch.Tensor    # [B, Q, 2]
    query_y: torch.Tensor    # [B, Q, 1]
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
    return torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float32,
    )


class LinearFamily(TaskFamily):
    name = "linear"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        x = _rand_uniform((total, 2), -2.0, 2.0, generator)
        angle = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        normal = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
        bias = _rand_uniform((1,), -0.5, 0.5, generator).item()
        margin = x @ normal + bias
        y = (margin > 0.0).float().unsqueeze(-1)
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


class XorFamily(TaskFamily):
    name = "xor"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        x = _rand_uniform((total, 2), -1.5, 1.5, generator)
        angle = _rand_uniform((1,), 0.0, math.pi / 2.0, generator).item()
        x = x @ _rotation(angle).T
        y = ((x[:, 0] > 0.0) ^ (x[:, 1] > 0.0)).float().unsqueeze(-1)
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


class MoonsFamily(TaskFamily):
    name = "moons"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        n1 = total // 2
        n2 = total - n1
        theta1 = _rand_uniform((n1,), 0.0, math.pi, generator)
        theta2 = _rand_uniform((n2,), 0.0, math.pi, generator)

        moon1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=-1)
        moon2 = torch.stack([1.0 - torch.cos(theta2), -torch.sin(theta2) - 0.4], dim=-1)
        x = torch.cat([moon1, moon2], dim=0)
        y = torch.cat([torch.zeros(n1, 1), torch.ones(n2, 1)], dim=0)

        x = x + 0.08 * _randn(x.shape, generator)
        scale = _rand_uniform((1,), 0.8, 1.2, generator).item()
        angle = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        x = (x * scale) @ _rotation(angle).T
        order = torch.randperm(total, generator=generator)
        x = x[order]
        y = y[order]
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


class CirclesFamily(TaskFamily):
    name = "circles"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        n1 = total // 2
        n2 = total - n1
        theta1 = _rand_uniform((n1,), 0.0, 2.0 * math.pi, generator)
        theta2 = _rand_uniform((n2,), 0.0, 2.0 * math.pi, generator)
        r_inner = torch.empty(n1).normal_(mean=0.7, std=0.04, generator=generator)
        r_outer = torch.empty(n2).normal_(mean=1.4, std=0.06, generator=generator)

        inner = torch.stack([r_inner * torch.cos(theta1), r_inner * torch.sin(theta1)], dim=-1)
        outer = torch.stack([r_outer * torch.cos(theta2), r_outer * torch.sin(theta2)], dim=-1)
        x = torch.cat([inner, outer], dim=0)
        y = torch.cat([torch.zeros(n1, 1), torch.ones(n2, 1)], dim=0)

        x = x + 0.05 * _randn(x.shape, generator)
        order = torch.randperm(total, generator=generator)
        x = x[order]
        y = y[order]
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


class SineBoundaryFamily(TaskFamily):
    name = "sine"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        x = _rand_uniform((total, 2), -2.0, 2.0, generator)
        amp = _rand_uniform((1,), 0.4, 1.0, generator).item()
        freq = _rand_uniform((1,), 1.0, 2.5, generator).item()
        phase = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        offset = _rand_uniform((1,), -0.4, 0.4, generator).item()
        angle = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        xr = x @ _rotation(angle).T
        boundary = amp * torch.sin(freq * xr[:, 0] + phase) + offset
        y = (xr[:, 1] > boundary).float().unsqueeze(-1)
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


class DiamondFamily(TaskFamily):
    name = "diamond"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        x = _rand_uniform((total, 2), -2.0, 2.0, generator)
        angle = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        xr = x @ _rotation(angle).T
        radius = _rand_uniform((1,), 1.0, 1.8, generator).item()
        score = xr.abs().sum(dim=-1)
        invert = bool(torch.randint(0, 2, (1,), generator=generator).item())
        y = (score > radius).float()
        if invert:
            y = 1.0 - y
        y = y.unsqueeze(-1)
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


class SpiralFamily(TaskFamily):
    name = "spiral"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        n1 = total // 2
        n2 = total - n1
        theta1 = _rand_uniform((n1,), 0.4, 3.8 * math.pi, generator)
        theta2 = theta1[:n2] + math.pi
        if n2 > n1:
            extra = _rand_uniform((n2 - n1,), 0.4, 3.8 * math.pi, generator) + math.pi
            theta2 = torch.cat([theta2, extra], dim=0)
        theta2 = theta2[:n2]

        r1 = 0.18 * theta1
        r2 = 0.18 * theta2
        arm1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=-1)
        arm2 = torch.stack([r2 * torch.cos(theta2), r2 * torch.sin(theta2)], dim=-1)
        x = torch.cat([arm1, arm2], dim=0)
        y = torch.cat([torch.zeros(n1, 1), torch.ones(n2, 1)], dim=0)

        x = x + 0.08 * _randn(x.shape, generator)
        angle = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        scale = _rand_uniform((1,), 0.8, 1.15, generator).item()
        x = (x * scale) @ _rotation(angle).T
        order = torch.randperm(total, generator=generator)
        x = x[order]
        y = y[order]
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


class GaussianMixtureFamily(TaskFamily):
    name = "gaussian"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        n1 = total // 2
        n2 = total - n1
        center1 = _rand_uniform((2,), -1.6, 1.6, generator)
        center2 = _rand_uniform((2,), -1.6, 1.6, generator)
        while torch.norm(center1 - center2).item() < 1.2:
            center2 = _rand_uniform((2,), -1.6, 1.6, generator)
        angle1 = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        angle2 = _rand_uniform((1,), 0.0, 2.0 * math.pi, generator).item()
        scales1 = _rand_uniform((2,), 0.12, 0.45, generator)
        scales2 = _rand_uniform((2,), 0.12, 0.45, generator)
        cov1 = _rotation(angle1) @ torch.diag(scales1)
        cov2 = _rotation(angle2) @ torch.diag(scales2)
        x1 = _randn((n1, 2), generator) @ cov1.T + center1
        x2 = _randn((n2, 2), generator) @ cov2.T + center2
        x = torch.cat([x1, x2], dim=0)
        y = torch.cat([torch.zeros(n1, 1), torch.ones(n2, 1)], dim=0)
        order = torch.randperm(total, generator=generator)
        x = x[order]
        y = y[order]
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


FAMILIES: Dict[str, TaskFamily] = {
    cls.name: cls()
    for cls in [
        LinearFamily,
        XorFamily,
        MoonsFamily,
        CirclesFamily,
        SineBoundaryFamily,
        DiamondFamily,
        SpiralFamily,
        GaussianMixtureFamily,
    ]
}


DEFAULT_TRAIN_FAMILIES = ["linear", "xor", "moons", "circles"]
ALL_FAMILIES = list(FAMILIES.keys())
OOD_FAMILIES = [name for name in ALL_FAMILIES if name not in DEFAULT_TRAIN_FAMILIES]


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
        sx, sy, qx, qy = family.sample_episode(
            support_size=support_size,
            query_size=query_size,
            generator=generator,
        )
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
