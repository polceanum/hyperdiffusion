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


class LinearFamily(TaskFamily):
    name = "linear"

    def sample_episode(
        self,
        support_size: int,
        query_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = support_size + query_size
        x = torch.empty(total, 2).uniform_(-2.0, 2.0, generator=generator)
        angle = torch.empty(1).uniform_(0.0, 2.0 * math.pi, generator=generator).item()
        normal = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
        bias = torch.empty(1).uniform_(-0.5, 0.5, generator=generator).item()
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
        x = torch.empty(total, 2).uniform_(-1.5, 1.5, generator=generator)
        angle = torch.empty(1).uniform_(0.0, math.pi / 2.0, generator=generator).item()
        rot = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=torch.float32,
        )
        x = x @ rot.T
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
        theta1 = torch.empty(n1).uniform_(0.0, math.pi, generator=generator)
        theta2 = torch.empty(n2).uniform_(0.0, math.pi, generator=generator)

        moon1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=-1)
        moon2 = torch.stack([1.0 - torch.cos(theta2), -torch.sin(theta2) - 0.4], dim=-1)
        x = torch.cat([moon1, moon2], dim=0)
        y = torch.cat([torch.zeros(n1, 1), torch.ones(n2, 1)], dim=0)

        noise = 0.08 * torch.randn(x.shape, generator=generator)
        x = x + noise
        scale = torch.empty(1).uniform_(0.8, 1.2, generator=generator).item()
        angle = torch.empty(1).uniform_(0.0, 2.0 * math.pi, generator=generator).item()
        rot = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=torch.float32,
        )
        x = (x * scale) @ rot.T
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
        theta1 = torch.empty(n1).uniform_(0.0, 2.0 * math.pi, generator=generator)
        theta2 = torch.empty(n2).uniform_(0.0, 2.0 * math.pi, generator=generator)
        r_inner = torch.empty(n1).normal_(mean=0.7, std=0.04, generator=generator)
        r_outer = torch.empty(n2).normal_(mean=1.4, std=0.06, generator=generator)

        inner = torch.stack([r_inner * torch.cos(theta1), r_inner * torch.sin(theta1)], dim=-1)
        outer = torch.stack([r_outer * torch.cos(theta2), r_outer * torch.sin(theta2)], dim=-1)
        x = torch.cat([inner, outer], dim=0)
        y = torch.cat([torch.zeros(n1, 1), torch.ones(n2, 1)], dim=0)

        x = x + 0.05 * torch.randn(x.shape, generator=generator)
        order = torch.randperm(total, generator=generator)
        x = x[order]
        y = y[order]
        return x[:support_size], y[:support_size], x[support_size:], y[support_size:]


FAMILIES: Dict[str, TaskFamily] = {
    cls.name: cls() for cls in [LinearFamily, XorFamily, MoonsFamily, CirclesFamily]
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
