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
    input_dim: int = 2
    task_type: str = "classification"

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


class BanditLinearFamily(RegressionFamily):
    name = "bandit_linear"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        # Strong linear in both dimensions including a cross term for some 2D curvature effect
        coeff1 = _rand_uniform((1,), 2.5, 4.0, generator).item()
        coeff2 = _rand_uniform((1,), -2.5, 2.5, generator).item()
        coeff_cross = _rand_uniform((1,), -1.5, 1.5, generator).item()
        bias = _rand_uniform((1,), -0.2, 0.2, generator).item()
        y = (coeff1 * x[:, 0:1] + coeff2 * x[:, 1:2] + coeff_cross * x[:, 0:1] * x[:, 1:2] + bias)
        y = y + 0.005 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class BanditQuadraticFamily(RegressionFamily):
    name = "bandit_quadratic"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        # Strong 2D quadratic surface
        coeff_lin_x = _rand_uniform((1,), -0.8, 0.8, generator).item()
        coeff_lin_y = _rand_uniform((1,), -0.8, 0.8, generator).item()
        coeff_quad1 = _rand_uniform((1,), -5.0, 5.0, generator).item()
        coeff_quad2 = _rand_uniform((1,), -5.0, 5.0, generator).item()
        coeff_xy = _rand_uniform((1,), -2.0, 2.0, generator).item()
        bias = _rand_uniform((1,), -0.1, 0.1, generator).item()
        y = (
            coeff_lin_x * x[:, 0:1]
            + coeff_lin_y * x[:, 1:2]
            + coeff_quad1 * (x[:, 0:1] ** 2)
            + coeff_quad2 * (x[:, 1:2] ** 2)
            + coeff_xy * x[:, 0:1] * x[:, 1:2]
            + bias
        )
        y = y + 0.001 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


class BanditShiftedFamily(RegressionFamily):
    name = "bandit_shifted"

    def sample_episode(self, support_size: int, query_size: int, generator: Optional[torch.Generator] = None):
        total = support_size + query_size
        x = _sample_points(total, 2.0, generator)
        # Same as bandit_linear but with subtle shift
        coeff1 = _rand_uniform((1,), 0.8, 1.5, generator).item()
        coeff2 = _rand_uniform((1,), -0.3, 0.3, generator).item()
        bias = _rand_uniform((1,), -0.2, 0.2, generator).item()
        shift = _rand_uniform((2,), -0.2, 0.2, generator)  # small shift
        x_shifted = x + shift
        y = (coeff1 * x_shifted[:, 0:1] + coeff2 * x_shifted[:, 1:2] + bias)
        y = y + 0.02 * _randn(y.shape, generator)
        return _split_episode(x, y, support_size)


REGRESSION_FAMILIES: Dict[str, TaskFamily] = {
    family.name: family()
    for family in [
        SineRegressionFamily,
        PiecewiseRegressionFamily,
        QuadraticRegressionFamily,
        SawtoothRegressionFamily,
        CubicRegressionFamily,
        AbsRegressionFamily,
    ]
}

BANDIT_FAMILIES: Dict[str, TaskFamily] = {
    family.name: family()
    for family in [
        BanditLinearFamily,
        BanditQuadraticFamily,
        BanditShiftedFamily,
    ]
}

DEFAULT_REGRESSION_TRAIN_FAMILIES = [
    "sine_regression",
    "piecewise_regression",
    "quadratic_regression",
    "sawtooth_regression",
]
DEFAULT_REGRESSION_EVAL_FAMILIES = ["cubic_regression", "abs_regression"]
REGRESSION_TRAIN_GROUPS: Dict[str, List[str]] = {
    "core": DEFAULT_REGRESSION_TRAIN_FAMILIES,
    "held_out": DEFAULT_REGRESSION_EVAL_FAMILIES,
}

DEFAULT_BANDIT_TRAIN_FAMILIES = ["bandit_linear", "bandit_quadratic"]
DEFAULT_BANDIT_EVAL_FAMILIES = ["bandit_shifted"]
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
    raise ValueError(f"Unknown task_type: {task_type}")


def make_episode_batch(
    batch_size: int,
    support_size: int,
    query_size: int,
    family_names: Iterable[str],
    generator: Optional[torch.Generator] = None,
    task_type: str = "classification",
) -> EpisodeBatch:
    registry = get_registry(task_type)
    selected = list(family_names)
    if not selected:
        raise ValueError("family_names must not be empty")

    support_x, support_y, query_x, query_y, names = [], [], [], [], []
    for _ in range(batch_size):
        idx = torch.randint(0, len(selected), (1,), generator=generator).item()
        family = registry[selected[idx]]
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
