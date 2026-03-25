from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .diffusion import DiffusionSchedule, ddim_sample
from .models import DiffusionDenoiser, HyperNetworkSystem, TargetArchitecture, functional_target_network
from .tasks import (
    DEFAULT_REGRESSION_EVAL_FAMILIES,
    DEFAULT_REGRESSION_TRAIN_FAMILIES,
    DEFAULT_TRAIN_FAMILIES,
    EXPANDED_TRAIN_FAMILIES,
    ORIGINAL_TRAIN_GROUPS,
    REGRESSION_TRAIN_GROUPS,
    EpisodeBatch,
    make_episode_batch,
)


@dataclass
class ExperimentConfig:
    task_type: str = "classification"
    families: List[str] | None = None
    eval_families: List[str] | None = None
    train_steps_stage1: int = 1200
    train_steps_stage2: int = 1200
    eval_batches: int = 40
    batch_size: int = 32
    support_size: int = 16
    query_size: int = 64
    latent_dim: int = 32
    cond_dim: int = 64
    encoder_hidden: int = 128
    decoder_hidden: int = 256
    denoiser_hidden: int = 128
    denoiser_depth: int = 4
    target_hidden_dim: int = 64
    target_depth: int = 4
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 2e-4
    grad_clip: float = 1.0
    latent_l2_weight: float = 1e-4
    num_diffusion_steps: int = 20
    diagnostic_samples: int = 8
    mismatch_batches: int = 16
    support_size_sweep: List[int] | None = None
    visualization_count: int = 4
    visualization_grid_size: int = 120
    encoder_type: str = "attention"
    attention_heads: int = 4
    attention_layers: int = 3
    support_sweep_batches: int = 16
    device: str = "cpu"
    seed: int = 0


class MetricsTracker:
    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.history.setdefault(key, []).append(float(value))

    def mean_last(self, key: str, window: int = 50) -> float:
        values = self.history.get(key, [])[-window:]
        return float(sum(values) / max(len(values), 1))

    def save_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


class Experiment:
    def __init__(self, config: ExperimentConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        self.device = torch.device(config.device)
        input_dim = 2 if config.task_type == "classification" else 1
        self.arch = TargetArchitecture(in_dim=input_dim, hidden_dim=config.target_hidden_dim, depth=config.target_depth)
        self.system = HyperNetworkSystem(
            arch=self.arch,
            cond_dim=config.cond_dim,
            latent_dim=config.latent_dim,
            encoder_hidden=config.encoder_hidden,
            decoder_hidden=config.decoder_hidden,
            encoder_type=config.encoder_type,
            attention_heads=config.attention_heads,
            attention_layers=config.attention_layers,
            x_dim=input_dim,
            y_dim=1,
        ).to(self.device)
        self.denoiser = DiffusionDenoiser(
            latent_dim=config.latent_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.denoiser_hidden,
            depth=config.denoiser_depth,
        ).to(self.device)
        self.schedule = DiffusionSchedule(num_steps=config.num_diffusion_steps).to(self.device)

        self.opt_stage1 = torch.optim.AdamW(self.system.parameters(), lr=config.learning_rate_stage1)
        self.opt_stage2 = torch.optim.AdamW(self.denoiser.parameters(), lr=config.learning_rate_stage2)
        self.generator = torch.Generator().manual_seed(config.seed)

    def sample_batch(self, support_size: Optional[int] = None, family_names: Optional[List[str]] = None, batch_size: Optional[int] = None) -> EpisodeBatch:
        return make_episode_batch(
            batch_size=batch_size or self.config.batch_size,
            support_size=support_size or self.config.support_size,
            query_size=self.config.query_size,
            family_names=family_names or self.config.families,
            generator=self.generator,
            task_type=self.config.task_type,
        )

    def to_device(self, batch: EpisodeBatch) -> EpisodeBatch:
        return EpisodeBatch(
            support_x=batch.support_x.to(self.device),
            support_y=batch.support_y.to(self.device),
            query_x=batch.query_x.to(self.device),
            query_y=batch.query_y.to(self.device),
            family_name=batch.family_name,
        )

    def _task_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.task_type == "classification":
            return F.binary_cross_entropy_with_logits(pred, target)
        return F.mse_loss(pred, target)

    def _item_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.task_type == "classification":
            return F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean(dim=(1, 2))
        return F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))

    def _item_metric(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.task_type == "classification":
            return ((pred > 0.0) == (target > 0.5)).float().mean(dim=(1, 2))
        sse = ((pred - target) ** 2).sum(dim=(1, 2))
        mean_target = target.mean(dim=(1, 2), keepdim=True)
        sst = ((target - mean_target) ** 2).sum(dim=(1, 2)).clamp_min(1e-6)
        return 1.0 - sse / sst

    def _metric_name(self) -> str:
        return "acc" if self.config.task_type == "classification" else "r2"

    def stage1_step(self, batch: EpisodeBatch) -> Dict[str, float]:
        self.system.train()
        batch = self.to_device(batch)
        context, latent = self.system.encode(batch.support_x, batch.support_y)
        params = self.system.decode(latent, context)
        pred = functional_target_network(batch.query_x, params)
        task_loss = self._task_loss(pred, batch.query_y)
        latent_reg = latent.pow(2).mean()
        loss = task_loss + self.config.latent_l2_weight * latent_reg

        self.opt_stage1.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.config.grad_clip)
        self.opt_stage1.step()

        metric = self._item_metric(pred, batch.query_y).mean().item()
        return {"stage1_loss": loss.item(), "stage1_task_loss": task_loss.item(), f"stage1_{self._metric_name()}": metric, "latent_norm": latent.norm(dim=-1).mean().item()}

    def stage2_step(self, batch: EpisodeBatch) -> Dict[str, float]:
        self.denoiser.train()
        self.system.eval()
        batch = self.to_device(batch)
        with torch.no_grad():
            context, latent = self.system.encode(batch.support_x, batch.support_y)

        b = latent.shape[0]
        t_idx = torch.randint(0, self.schedule.num_steps, (b,), device=self.device)
        t = t_idx.float() / max(self.schedule.num_steps - 1, 1)
        noise = torch.randn_like(latent)
        z_t = self.schedule.q_sample(latent, t_idx, noise)
        pred_noise = self.denoiser(z_t, t, context)
        loss = F.mse_loss(pred_noise, noise)

        self.opt_stage2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.config.grad_clip)
        self.opt_stage2.step()

        z0_hat = self.schedule.predict_z0(z_t, t_idx, pred_noise)
        rmse = (z0_hat - latent).pow(2).mean().sqrt().item()
        return {"stage2_loss": loss.item(), "stage2_rmse": rmse}

    @staticmethod
    def _mean(values: List[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    @staticmethod
    def _pairwise_mean_l2(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 2:
            return torch.zeros(x.shape[0], device=x.device)
        d = torch.cdist(x, x, p=2)
        k = x.shape[1]
        mask = torch.ones(k, k, device=x.device, dtype=torch.bool).triu(diagonal=1)
        return d[:, mask].mean(dim=-1)

    def _prediction_disagreement(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] < 2:
            return torch.zeros(pred.shape[0], device=pred.device)
        if self.config.task_type == "classification":
            discrete = (pred > 0.0).float().squeeze(-1)
        else:
            discrete = pred.squeeze(-1)
            std = discrete.std(dim=1, keepdim=True).clamp_min(1e-6)
            discrete = discrete / std
        disagree = (discrete[:, :, None, :] - discrete[:, None, :, :]).abs().mean(dim=-1)
        k = pred.shape[1]
        mask = torch.ones(k, k, device=pred.device, dtype=torch.bool).triu(diagonal=1)
        return disagree[:, mask].mean(dim=-1)

    def _decode_and_score(self, query_x: torch.Tensor, query_y: torch.Tensor, z: torch.Tensor, context: torch.Tensor):
        params = self.system.decode(z, context)
        pred = functional_target_network(query_x, params)
        flat_params = torch.cat([v.reshape(v.shape[0], -1) for v in params.values()], dim=-1)
        return pred, self._item_metric(pred, query_y), flat_params

    @torch.no_grad()
    def evaluate(self, num_batches: Optional[int] = None, family_names: Optional[List[str]] = None) -> Dict[str, object]:
        self.system.eval()
        self.denoiser.eval()
        num_batches = num_batches or self.config.eval_batches
        families = family_names or self.config.families

        per_family: Dict[str, Dict[str, List[float]]] = {}
        overall: Dict[str, List[float]] = {
            "encoder_metric": [], "diffusion_metric": [], "encoder_loss": [], "diffusion_loss": [],
            "diffusion_metric_mean_k": [], "diffusion_metric_best_k": [], "prediction_disagreement": [],
            "weight_pairwise_l2": [], "latent_pairwise_l2": []
        }

        for _ in range(num_batches):
            batch = self.to_device(self.sample_batch(family_names=families))
            context, latent = self.system.encode(batch.support_x, batch.support_y)
            pred_enc, metric_enc, _ = self._decode_and_score(batch.query_x, batch.query_y, latent, context)
            loss_enc = self._item_loss(pred_enc, batch.query_y)

            z_samples, pred_samples, metric_samples, flat_params = [], [], [], []
            for _k in range(self.config.diagnostic_samples):
                z = ddim_sample(self.denoiser, self.schedule, context, num_steps=self.config.num_diffusion_steps)
                pred, metric, flat = self._decode_and_score(batch.query_x, batch.query_y, z, context)
                z_samples.append(z)
                pred_samples.append(pred)
                metric_samples.append(metric)
                flat_params.append(flat)

            z_stack = torch.stack(z_samples, dim=1)
            pred_stack = torch.stack(pred_samples, dim=1)
            metric_stack = torch.stack(metric_samples, dim=1)
            flat_stack = torch.stack(flat_params, dim=1)
            pred_diff = pred_stack[:, 0]
            metric_diff = metric_stack[:, 0]
            loss_diff = self._item_loss(pred_diff, batch.query_y)

            metric_mean = metric_stack.mean(dim=1)
            metric_best = metric_stack.max(dim=1).values
            disagreement = self._prediction_disagreement(pred_stack)
            weight_l2 = self._pairwise_mean_l2(flat_stack)
            latent_l2 = self._pairwise_mean_l2(z_stack)

            for idx, family in enumerate(batch.family_name):
                fam = per_family.setdefault(family, {k: [] for k in overall})
                vals = {
                    "encoder_metric": metric_enc[idx].item(),
                    "diffusion_metric": metric_diff[idx].item(),
                    "encoder_loss": loss_enc[idx].item(),
                    "diffusion_loss": loss_diff[idx].item(),
                    "diffusion_metric_mean_k": metric_mean[idx].item(),
                    "diffusion_metric_best_k": metric_best[idx].item(),
                    "prediction_disagreement": disagreement[idx].item(),
                    "weight_pairwise_l2": weight_l2[idx].item(),
                    "latent_pairwise_l2": latent_l2[idx].item(),
                }
                for k, v in vals.items():
                    overall[k].append(v)
                    fam[k].append(v)

        metric_name = self._metric_name()
        def pack(values: Dict[str, List[float]]) -> Dict[str, float]:
            return {
                f"encoder_{metric_name}": self._mean(values["encoder_metric"]),
                f"diffusion_{metric_name}": self._mean(values["diffusion_metric"]),
                "encoder_loss": self._mean(values["encoder_loss"]),
                "diffusion_loss": self._mean(values["diffusion_loss"]),
                f"diffusion_{metric_name}_mean_k": self._mean(values["diffusion_metric_mean_k"]),
                f"diffusion_{metric_name}_best_k": self._mean(values["diffusion_metric_best_k"]),
                "prediction_disagreement": self._mean(values["prediction_disagreement"]),
                "weight_pairwise_l2": self._mean(values["weight_pairwise_l2"]),
                "latent_pairwise_l2": self._mean(values["latent_pairwise_l2"]),
            }

        summary = {
            "overall": pack(overall),
            "macro_average": pack({k: [pack(v)[kk] for kk in []] for k, v in overall.items()}) if False else pack({
                key: [self._mean(fam[key]) for fam in per_family.values()] for key in overall
            }),
            "by_family": {family: pack(vals) for family, vals in per_family.items()},
            "diagnostics": self._diagnostics(families),
            "baseline_comparison": {
                "deterministic_encoder": {metric_name: pack(overall)[f"encoder_{metric_name}"], "loss": pack(overall)["encoder_loss"]},
                "diffusion_sampler": {metric_name: pack(overall)[f"diffusion_{metric_name}"], "loss": pack(overall)["diffusion_loss"]},
            },
        }
        return summary

    @torch.no_grad()
    def _diagnostics(self, families: List[str]) -> Dict[str, object]:
        metric_name = self._metric_name()
        mismatch_encoder, mismatch_diff = [], []
        if len(families) > 1:
            for _ in range(self.config.mismatch_batches):
                support_batch = self.to_device(self.sample_batch(family_names=families))
                query_batch = self.to_device(self.sample_batch(family_names=families))
                mismatch_mask = [a != b for a, b in zip(support_batch.family_name, query_batch.family_name)]
                if not any(mismatch_mask):
                    continue
                idx = torch.tensor(mismatch_mask, device=self.device)
                context, latent = self.system.encode(support_batch.support_x[idx], support_batch.support_y[idx])
                pred_enc, metric_enc, _ = self._decode_and_score(query_batch.query_x[idx], query_batch.query_y[idx], latent, context)
                z = ddim_sample(self.denoiser, self.schedule, context, num_steps=self.config.num_diffusion_steps)
                pred_diff, metric_diff, _ = self._decode_and_score(query_batch.query_x[idx], query_batch.query_y[idx], z, context)
                mismatch_encoder.extend(metric_enc.tolist())
                mismatch_diff.extend(metric_diff.tolist())

        sweep = {}
        for size in self.config.support_size_sweep or []:
            enc_vals, diff_vals = [], []
            for _ in range(self.config.support_sweep_batches):
                batch = self.to_device(self.sample_batch(support_size=size, family_names=families))
                context, latent = self.system.encode(batch.support_x, batch.support_y)
                pred_enc, metric_enc, _ = self._decode_and_score(batch.query_x, batch.query_y, latent, context)
                enc_vals.extend(metric_enc.tolist())
                sample_metrics = []
                for _k in range(self.config.diagnostic_samples):
                    z = ddim_sample(self.denoiser, self.schedule, context, num_steps=self.config.num_diffusion_steps)
                    _, metric, _ = self._decode_and_score(batch.query_x, batch.query_y, z, context)
                    sample_metrics.append(metric)
                sample_metrics = torch.stack(sample_metrics, dim=1).mean(dim=1)
                diff_vals.extend(sample_metrics.tolist())
            sweep[str(size)] = {f"encoder_{metric_name}": self._mean(enc_vals), f"diffusion_{metric_name}_mean": self._mean(diff_vals)}

        return {
            "sample_count": self.config.diagnostic_samples,
            "mismatch": {
                f"encoder_{metric_name}": self._mean(mismatch_encoder),
                f"diffusion_{metric_name}": self._mean(mismatch_diff),
                "num_mismatch_episodes": float(len(mismatch_encoder)),
            },
            "support_size_sweep": sweep,
        }

    def _plot_episode_set(self, family_names: List[str], prefix: str) -> List[str]:
        if self.config.visualization_count <= 0:
            return []
        batch = self.to_device(self.sample_batch(batch_size=self.config.visualization_count, family_names=family_names))
        return self._plot_classification_batch(batch, prefix) if self.config.task_type == "classification" else self._plot_regression_batch(batch, prefix)

    @torch.no_grad()
    def _plot_classification_batch(self, batch: EpisodeBatch, prefix: str) -> List[str]:
        paths: List[str] = []
        context, latent = self.system.encode(batch.support_x, batch.support_y)
        b = batch.support_x.shape[0]
        grid_size = self.config.visualization_grid_size
        grid_lin = torch.linspace(-2.4, 2.4, grid_size, device=self.device)
        gx, gy = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0).expand(b, -1, -1)
        enc_grid = functional_target_network(grid, self.system.decode(latent, context))
        acc_enc = self._item_metric(functional_target_network(batch.query_x, self.system.decode(latent, context)), batch.query_y)

        pred_list, grid_list, acc_list = [], [], []
        for _ in range(self.config.diagnostic_samples):
            z = ddim_sample(self.denoiser, self.schedule, context, num_steps=self.config.num_diffusion_steps)
            params = self.system.decode(z, context)
            pred_list.append(functional_target_network(batch.query_x, params))
            grid_list.append(torch.sigmoid(functional_target_network(grid, params)))
            acc_list.append(self._item_metric(pred_list[-1], batch.query_y))
        grid_stack = torch.stack(grid_list, dim=1)
        acc_stack = torch.stack(acc_list, dim=1)
        best_idx = acc_stack.argmax(dim=1)
        best_grid = grid_stack[torch.arange(b, device=self.device), best_idx]
        mean_grid = grid_stack.mean(dim=1)
        disagreement = (grid_stack > 0.5).float().std(dim=1)

        for i in range(b):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
            panels = [
                (torch.sigmoid(enc_grid[i].squeeze(-1)).reshape(grid_size, grid_size).cpu(), f"encoder {acc_enc[i].item():.3f}"),
                (best_grid[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), f"best {acc_stack[i, best_idx[i]].item():.3f}"),
                (mean_grid[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), "mean prob"),
                (disagreement[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), "sample std"),
            ]
            for ax, (image, title) in zip(axes, panels):
                ax.imshow(image.T, origin="lower", extent=(-2.4, 2.4, -2.4, 2.4), cmap="coolwarm", alpha=0.85)
                sx = batch.support_x[i, :, 0].cpu().numpy(); sy = batch.support_x[i, :, 1].cpu().numpy(); sl = batch.support_y[i, :, 0].cpu().numpy()
                ax.scatter(sx[sl < 0.5], sy[sl < 0.5], s=30, edgecolor="black", facecolor="tab:blue", linewidth=0.6)
                ax.scatter(sx[sl > 0.5], sy[sl > 0.5], s=30, edgecolor="black", facecolor="tab:orange", linewidth=0.6)
                ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{prefix} | family={batch.family_name[i]}")
            out = self.output_dir / "plots" / f"{prefix}_{i:02d}_{batch.family_name[i]}.png"
            fig.savefig(out, dpi=140); plt.close(fig)
            paths.append(str(out.relative_to(self.output_dir)))
        return paths

    @torch.no_grad()
    def _plot_regression_batch(self, batch: EpisodeBatch, prefix: str) -> List[str]:
        paths: List[str] = []
        context, latent = self.system.encode(batch.support_x, batch.support_y)
        b = batch.support_x.shape[0]
        grid_x = torch.linspace(-3.2, 3.2, 256, device=self.device).view(1, -1, 1).expand(b, -1, -1)
        enc_curve = functional_target_network(grid_x, self.system.decode(latent, context))
        pred_enc = functional_target_network(batch.query_x, self.system.decode(latent, context))
        metric_enc = self._item_metric(pred_enc, batch.query_y)

        curve_list, metric_list = [], []
        for _ in range(self.config.diagnostic_samples):
            z = ddim_sample(self.denoiser, self.schedule, context, num_steps=self.config.num_diffusion_steps)
            params = self.system.decode(z, context)
            curve_list.append(functional_target_network(grid_x, params))
            metric_list.append(self._item_metric(functional_target_network(batch.query_x, params), batch.query_y))
        curve_stack = torch.stack(curve_list, dim=1)
        metric_stack = torch.stack(metric_list, dim=1)
        best_idx = metric_stack.argmax(dim=1)
        best_curve = curve_stack[torch.arange(b, device=self.device), best_idx]
        mean_curve = curve_stack.mean(dim=1)
        std_curve = curve_stack.std(dim=1)

        for i in range(b):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
            xg = grid_x[i, :, 0].cpu().numpy()
            support_x = batch.support_x[i, :, 0].cpu().numpy()
            support_y = batch.support_y[i, :, 0].cpu().numpy()
            query_x = batch.query_x[i, :, 0].cpu().numpy()
            query_y = batch.query_y[i, :, 0].cpu().numpy()
            panels = [
                (enc_curve[i, :, 0].cpu().numpy(), f"encoder r2={metric_enc[i].item():.3f}"),
                (best_curve[i, :, 0].cpu().numpy(), f"best r2={metric_stack[i, best_idx[i]].item():.3f}"),
                (mean_curve[i, :, 0].cpu().numpy(), "diffusion mean"),
                (std_curve[i, :, 0].cpu().numpy(), "sample std"),
            ]
            for ax, (curve, title) in zip(axes, panels):
                ax.scatter(query_x, query_y, s=10, alpha=0.25, color="tab:gray")
                ax.scatter(support_x, support_y, s=28, color="tab:orange", edgecolors="black", linewidths=0.5)
                ax.plot(xg, curve)
                ax.set_title(title)
            fig.suptitle(f"{prefix} | family={batch.family_name[i]}")
            out = self.output_dir / "plots" / f"{prefix}_{i:02d}_{batch.family_name[i]}.png"
            fig.savefig(out, dpi=140); plt.close(fig)
            paths.append(str(out.relative_to(self.output_dir)))
        return paths

    def _group_evaluations(self) -> Dict[str, object]:
        if self.config.task_type == "classification":
            groups = ORIGINAL_TRAIN_GROUPS
        else:
            groups = REGRESSION_TRAIN_GROUPS
        configured = set(self.config.families)
        out: Dict[str, object] = {}
        for name, fams in groups.items():
            active = [f for f in fams if f in configured]
            if active:
                out[name] = {"families": active, "summary": self.evaluate(num_batches=self.config.eval_batches, family_names=active)}
        return out

    def save_checkpoint(self) -> None:
        payload = {"config": asdict(self.config), "system": self.system.state_dict(), "denoiser": self.denoiser.state_dict()}
        torch.save(payload, self.output_dir / "checkpoint.pt")

    def run(self) -> Dict[str, object]:
        stage1 = MetricsTracker(); stage2 = MetricsTracker(); t0 = time.time()
        metric_name = self._metric_name()
        for step in range(1, self.config.train_steps_stage1 + 1):
            metrics = self.stage1_step(self.sample_batch())
            stage1.update(**metrics)
            if step % 100 == 0 or step == 1:
                print(f"[stage1] step={step:4d} loss={stage1.mean_last('stage1_loss',20):.4f} {metric_name}={stage1.mean_last(f'stage1_{metric_name}',20):.3f} latent_norm={stage1.mean_last('latent_norm',20):.3f}")
        for step in range(1, self.config.train_steps_stage2 + 1):
            metrics = self.stage2_step(self.sample_batch())
            stage2.update(**metrics)
            if step % 100 == 0 or step == 1:
                print(f"[stage2] step={step:4d} loss={stage2.mean_last('stage2_loss',20):.4f} z0_rmse={stage2.mean_last('stage2_rmse',20):.4f}")

        summary = self.evaluate()
        summary["train_group_evals"] = self._group_evaluations()
        summary["artifacts"] = {"plots_train": self._plot_episode_set(self.config.families, "train")}
        if self.config.eval_families:
            summary["generalization"] = {"eval_families": self.config.eval_families, "eval_summary": self.evaluate(num_batches=self.config.eval_batches, family_names=self.config.eval_families)}
            summary["artifacts"]["plots_eval"] = self._plot_episode_set(self.config.eval_families, "eval")
        summary["runtime_seconds"] = time.time() - t0
        summary["config"] = asdict(self.config)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        stage1.save_json(self.output_dir / "stage1_metrics.json"); stage2.save_json(self.output_dir / "stage2_metrics.json")
        self.save_checkpoint()
        return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperdiffusion toy experiment")
    p.add_argument("--output-dir", type=str, default="runs/default")
    p.add_argument("--task-type", type=str, choices=["classification", "regression"], default="classification")
    p.add_argument("--families", type=str, nargs="+", default=None)
    p.add_argument("--expanded-train-families", action="store_true")
    p.add_argument("--eval-families", type=str, nargs="*", default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps-stage1", type=int, default=1200)
    p.add_argument("--train-steps-stage2", type=int, default=1200)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--support-size", type=int, default=16)
    p.add_argument("--query-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--cond-dim", type=int, default=64)
    p.add_argument("--encoder-hidden", type=int, default=128)
    p.add_argument("--decoder-hidden", type=int, default=256)
    p.add_argument("--denoiser-hidden", type=int, default=128)
    p.add_argument("--denoiser-depth", type=int, default=4)
    p.add_argument("--target-hidden-dim", type=int, default=64)
    p.add_argument("--target-depth", type=int, default=4)
    p.add_argument("--learning-rate-stage1", type=float, default=1e-3)
    p.add_argument("--learning-rate-stage2", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--latent-l2-weight", type=float, default=1e-4)
    p.add_argument("--num-diffusion-steps", type=int, default=20)
    p.add_argument("--diagnostic-samples", type=int, default=8)
    p.add_argument("--mismatch-batches", type=int, default=16)
    p.add_argument("--support-size-sweep", type=int, nargs="*", default=[2,4,8,16])
    p.add_argument("--visualization-count", type=int, default=4)
    p.add_argument("--visualization-grid-size", type=int, default=120)
    p.add_argument("--encoder-type", type=str, choices=["attention", "deepset"], default="attention")
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--attention-layers", type=int, default=3)
    p.add_argument("--support-sweep-batches", type=int, default=16)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.task_type == "classification":
        families = EXPANDED_TRAIN_FAMILIES if args.expanded_train_families else (args.families or DEFAULT_TRAIN_FAMILIES)
        eval_families = args.eval_families
    else:
        families = args.families or DEFAULT_REGRESSION_TRAIN_FAMILIES
        eval_families = args.eval_families if args.eval_families is not None else DEFAULT_REGRESSION_EVAL_FAMILIES

    config = ExperimentConfig(
        task_type=args.task_type,
        families=families,
        eval_families=eval_families,
        train_steps_stage1=args.train_steps_stage1,
        train_steps_stage2=args.train_steps_stage2,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        support_size=args.support_size,
        query_size=args.query_size,
        latent_dim=args.latent_dim,
        cond_dim=args.cond_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        denoiser_hidden=args.denoiser_hidden,
        denoiser_depth=args.denoiser_depth,
        target_hidden_dim=args.target_hidden_dim,
        target_depth=args.target_depth,
        learning_rate_stage1=args.learning_rate_stage1,
        learning_rate_stage2=args.learning_rate_stage2,
        grad_clip=args.grad_clip,
        latent_l2_weight=args.latent_l2_weight,
        num_diffusion_steps=args.num_diffusion_steps,
        diagnostic_samples=args.diagnostic_samples,
        mismatch_batches=args.mismatch_batches,
        support_size_sweep=args.support_size_sweep,
        visualization_count=args.visualization_count,
        visualization_grid_size=args.visualization_grid_size,
        encoder_type=args.encoder_type,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        support_sweep_batches=args.support_sweep_batches,
        device=args.device,
        seed=args.seed,
    )
    exp = Experiment(config=config, output_dir=Path(args.output_dir))
    summary = exp.run()
    print("\nFinal summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
