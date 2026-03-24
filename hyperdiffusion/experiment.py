from __future__ import annotations

import argparse
import json
import math
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
from .tasks import DEFAULT_TRAIN_FAMILIES, EpisodeBatch, make_episode_batch


@dataclass
class ExperimentConfig:
    families: List[str]
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
    device: str = "cpu"
    seed: int = 0


class MetricsTracker:
    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.history.setdefault(key, []).append(float(value))

    def mean_last(self, key: str, window: int = 50) -> float:
        values = self.history.get(key, [])
        if not values:
            return float("nan")
        values = values[-window:]
        return sum(values) / len(values)

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
        self.arch = TargetArchitecture(hidden_dim=config.target_hidden_dim, depth=config.target_depth)
        self.system = HyperNetworkSystem(
            arch=self.arch,
            cond_dim=config.cond_dim,
            latent_dim=config.latent_dim,
            encoder_hidden=config.encoder_hidden,
            decoder_hidden=config.decoder_hidden,
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

    def sample_batch(
        self,
        support_size: Optional[int] = None,
        family_names: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> EpisodeBatch:
        return make_episode_batch(
            batch_size=batch_size or self.config.batch_size,
            support_size=support_size or self.config.support_size,
            query_size=self.config.query_size,
            family_names=family_names or self.config.families,
            generator=self.generator,
        )

    def to_device(self, batch: EpisodeBatch) -> EpisodeBatch:
        return EpisodeBatch(
            support_x=batch.support_x.to(self.device),
            support_y=batch.support_y.to(self.device),
            query_x=batch.query_x.to(self.device),
            query_y=batch.query_y.to(self.device),
            family_name=batch.family_name,
        )

    def stage1_step(self, batch: EpisodeBatch) -> Dict[str, float]:
        self.system.train()
        batch = self.to_device(batch)
        context, latent = self.system.encode(batch.support_x, batch.support_y)
        params = self.system.decode(latent, context)
        logits = functional_target_network(batch.query_x, params)
        task_loss = F.binary_cross_entropy_with_logits(logits, batch.query_y)
        latent_reg = latent.pow(2).mean()
        loss = task_loss + self.config.latent_l2_weight * latent_reg

        self.opt_stage1.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.config.grad_clip)
        self.opt_stage1.step()

        accuracy = ((logits > 0.0) == (batch.query_y > 0.5)).float().mean().item()
        return {
            "stage1_loss": loss.item(),
            "stage1_task_loss": task_loss.item(),
            "stage1_acc": accuracy,
            "latent_norm": latent.norm(dim=-1).mean().item(),
        }

    def stage2_step(self, batch: EpisodeBatch) -> Dict[str, float]:
        self.denoiser.train()
        self.system.eval()
        batch = self.to_device(batch)
        with torch.no_grad():
            context, latent = self.system.encode(batch.support_x, batch.support_y)

        batch_size = latent.shape[0]
        t_idx = torch.randint(0, self.schedule.num_steps, (batch_size,), device=self.device)
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
        z0_error = (z0_hat - latent).pow(2).mean().sqrt().item()
        return {"stage2_loss": loss.item(), "stage2_rmse": z0_error}

    @staticmethod
    def _bce_per_item(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=(1, 2))

    @staticmethod
    def _acc_per_item(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ((logits > 0.0) == (targets > 0.5)).float().mean(dim=(1, 2))

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

    @staticmethod
    def _prediction_disagreement(logits: torch.Tensor) -> torch.Tensor:
        pred = (logits > 0.0).float().squeeze(-1)
        if pred.shape[1] < 2:
            return torch.zeros(pred.shape[0], device=pred.device)
        disagree = (pred[:, :, None, :] != pred[:, None, :, :]).float().mean(dim=-1)
        k = pred.shape[1]
        mask = torch.ones(k, k, device=pred.device, dtype=torch.bool).triu(diagonal=1)
        return disagree[:, mask].mean(dim=-1)

    def _decode_and_score(
        self,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.system.decode(z, context)
        logits = functional_target_network(query_x, params)
        flat_params = torch.cat([v.reshape(v.shape[0], -1) for v in params.values()], dim=-1)
        return logits, self._acc_per_item(logits, query_y), flat_params

    @torch.no_grad()
    def _sample_many(self, context: torch.Tensor, num_samples: int) -> torch.Tensor:
        return torch.stack(
            [ddim_sample(self.denoiser, self.schedule, context, self.config.latent_dim) for _ in range(num_samples)],
            dim=1,
        )

    @torch.no_grad()
    def _support_sweep_summary(self, family_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        sweep_sizes = self.config.support_size_sweep or []
        active_families = family_names or self.config.families
        result: Dict[str, Dict[str, float]] = {}
        for support_size in sweep_sizes:
            batch = self.to_device(self.sample_batch(support_size=support_size, family_names=active_families))
            context, latent = self.system.encode(batch.support_x, batch.support_y)
            _, acc, _ = self._decode_and_score(batch.query_x, batch.query_y, latent, context)
            z_samples = self._sample_many(context, self.config.diagnostic_samples)
            diff_accs = []
            for sample_idx in range(z_samples.shape[1]):
                _, acc_k, _ = self._decode_and_score(batch.query_x, batch.query_y, z_samples[:, sample_idx], context)
                diff_accs.append(acc_k)
            diff_acc = torch.stack(diff_accs, dim=1).mean(dim=1)
            result[str(support_size)] = {
                "encoder_acc": float(acc.mean().item()),
                "diffusion_acc_mean": float(diff_acc.mean().item()),
            }
        return result

    @torch.no_grad()
    def _mismatch_summary(self, family_names: Optional[List[str]] = None) -> Dict[str, float]:
        active_families = family_names or self.config.families
        if len(active_families) < 2:
            return {"encoder_acc": float("nan"), "diffusion_acc": float("nan"), "num_mismatch_episodes": 0.0}

        encoder_accs: List[float] = []
        diffusion_accs: List[float] = []
        attempts = 0
        collected = 0
        while collected < self.config.mismatch_batches and attempts < self.config.mismatch_batches * 20:
            attempts += 1
            batch_a = self.to_device(self.sample_batch(family_names=active_families))
            batch_b = self.to_device(self.sample_batch(family_names=active_families))
            keep = [i for i, (fa, fb) in enumerate(zip(batch_a.family_name, batch_b.family_name)) if fa != fb]
            if not keep:
                continue
            idx = torch.tensor(keep, device=self.device, dtype=torch.long)
            mixed = EpisodeBatch(
                support_x=batch_a.support_x.index_select(0, idx),
                support_y=batch_a.support_y.index_select(0, idx),
                query_x=batch_b.query_x.index_select(0, idx),
                query_y=batch_b.query_y.index_select(0, idx),
                family_name=[f"support:{batch_a.family_name[i]}->query:{batch_b.family_name[i]}" for i in keep],
            )
            context, latent = self.system.encode(mixed.support_x, mixed.support_y)
            _, acc_enc, _ = self._decode_and_score(mixed.query_x, mixed.query_y, latent, context)
            z_sample = ddim_sample(self.denoiser, self.schedule, context, self.config.latent_dim)
            _, acc_diff, _ = self._decode_and_score(mixed.query_x, mixed.query_y, z_sample, context)
            encoder_accs.extend(acc_enc.cpu().tolist())
            diffusion_accs.extend(acc_diff.cpu().tolist())
            collected += 1
        return {
            "encoder_acc": self._mean(encoder_accs),
            "diffusion_acc": self._mean(diffusion_accs),
            "num_mismatch_episodes": float(len(encoder_accs)),
        }

    @torch.no_grad()
    def evaluate(self, num_batches: Optional[int] = None, family_names: Optional[List[str]] = None) -> Dict[str, object]:
        self.system.eval()
        self.denoiser.eval()
        num_batches = num_batches or self.config.eval_batches
        num_samples = self.config.diagnostic_samples
        active_families = family_names or self.config.families

        family_scores: Dict[str, Dict[str, List[float]]] = {}
        overall: Dict[str, List[float]] = {
            "encoder_acc": [],
            "diffusion_acc": [],
            "encoder_loss": [],
            "diffusion_loss": [],
            "diffusion_acc_mean_k": [],
            "diffusion_acc_best_k": [],
            "prediction_disagreement": [],
            "weight_pairwise_l2": [],
            "latent_pairwise_l2": [],
        }

        for _ in range(num_batches):
            batch = self.to_device(self.sample_batch(family_names=active_families))
            context, latent = self.system.encode(batch.support_x, batch.support_y)
            logits_enc, acc_enc, _ = self._decode_and_score(batch.query_x, batch.query_y, latent, context)
            loss_enc = self._bce_per_item(logits_enc, batch.query_y)

            z_samples = self._sample_many(context, num_samples)
            logits_all = []
            acc_all = []
            flat_all = []
            for sample_idx in range(num_samples):
                logits_k, acc_k, flat_k = self._decode_and_score(batch.query_x, batch.query_y, z_samples[:, sample_idx], context)
                logits_all.append(logits_k)
                acc_all.append(acc_k)
                flat_all.append(flat_k)

            logits_all_t = torch.stack(logits_all, dim=1)
            acc_all_t = torch.stack(acc_all, dim=1)
            flat_all_t = torch.stack(flat_all, dim=1)

            logits_diff = logits_all_t[:, 0]
            acc_diff = acc_all_t[:, 0]
            loss_diff = self._bce_per_item(logits_diff, batch.query_y)
            acc_mean_k = acc_all_t.mean(dim=1)
            acc_best_k = acc_all_t.max(dim=1).values
            disagreement = self._prediction_disagreement(logits_all_t)
            weight_l2 = self._pairwise_mean_l2(flat_all_t)
            latent_l2 = self._pairwise_mean_l2(z_samples)

            overall["encoder_acc"].extend(acc_enc.cpu().tolist())
            overall["diffusion_acc"].extend(acc_diff.cpu().tolist())
            overall["encoder_loss"].extend(loss_enc.cpu().tolist())
            overall["diffusion_loss"].extend(loss_diff.cpu().tolist())
            overall["diffusion_acc_mean_k"].extend(acc_mean_k.cpu().tolist())
            overall["diffusion_acc_best_k"].extend(acc_best_k.cpu().tolist())
            overall["prediction_disagreement"].extend(disagreement.cpu().tolist())
            overall["weight_pairwise_l2"].extend(weight_l2.cpu().tolist())
            overall["latent_pairwise_l2"].extend(latent_l2.cpu().tolist())

            for idx, family in enumerate(batch.family_name):
                family_scores.setdefault(
                    family,
                    {
                        "encoder_acc": [],
                        "diffusion_acc": [],
                        "encoder_loss": [],
                        "diffusion_loss": [],
                        "diffusion_acc_mean_k": [],
                        "diffusion_acc_best_k": [],
                        "prediction_disagreement": [],
                        "weight_pairwise_l2": [],
                        "latent_pairwise_l2": [],
                    },
                )
                family_scores[family]["encoder_acc"].append(float(acc_enc[idx].item()))
                family_scores[family]["diffusion_acc"].append(float(acc_diff[idx].item()))
                family_scores[family]["encoder_loss"].append(float(loss_enc[idx].item()))
                family_scores[family]["diffusion_loss"].append(float(loss_diff[idx].item()))
                family_scores[family]["diffusion_acc_mean_k"].append(float(acc_mean_k[idx].item()))
                family_scores[family]["diffusion_acc_best_k"].append(float(acc_best_k[idx].item()))
                family_scores[family]["prediction_disagreement"].append(float(disagreement[idx].item()))
                family_scores[family]["weight_pairwise_l2"].append(float(weight_l2[idx].item()))
                family_scores[family]["latent_pairwise_l2"].append(float(latent_l2[idx].item()))

        def summarize(d: Dict[str, List[float]]) -> Dict[str, float]:
            return {k: self._mean(v) for k, v in d.items()}

        return {
            "overall": summarize(overall),
            "by_family": {family: summarize(values) for family, values in family_scores.items()},
            "diagnostics": {
                "sample_count": num_samples,
                "mismatch": self._mismatch_summary(active_families),
                "support_size_sweep": self._support_sweep_summary(active_families),
            },
        }

    @torch.no_grad()
    def _build_grid(self, grid_size: int) -> torch.Tensor:
        axis = torch.linspace(-2.4, 2.4, grid_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    @torch.no_grad()
    def _plot_episode_set(self, family_names: List[str], prefix: str) -> List[str]:
        if self.config.visualization_count <= 0:
            return []
        paths: List[str] = []
        grid = self._build_grid(self.config.visualization_grid_size)
        batch = self.to_device(self.sample_batch(family_names=family_names, batch_size=self.config.visualization_count))
        context, latent = self.system.encode(batch.support_x, batch.support_y)
        z_samples = self._sample_many(context, self.config.diagnostic_samples)
        grid_batch = grid.unsqueeze(0).expand(batch.support_x.shape[0], -1, -1)

        logits_enc, acc_enc, _ = self._decode_and_score(batch.query_x, batch.query_y, latent, context)
        grid_logits_enc, _, _ = self._decode_and_score(grid_batch, torch.zeros(batch.support_x.shape[0], grid_batch.shape[1], 1, device=self.device), latent, context)

        sample_query_accs = []
        sample_grid_probs = []
        for sample_idx in range(self.config.diagnostic_samples):
            logits_q, acc_k, _ = self._decode_and_score(batch.query_x, batch.query_y, z_samples[:, sample_idx], context)
            sample_query_accs.append(acc_k)
            grid_logits_k, _, _ = self._decode_and_score(grid_batch, torch.zeros(batch.support_x.shape[0], grid_batch.shape[1], 1, device=self.device), z_samples[:, sample_idx], context)
            sample_grid_probs.append(torch.sigmoid(grid_logits_k.squeeze(-1)))

        acc_stack = torch.stack(sample_query_accs, dim=1)
        best_idx = acc_stack.argmax(dim=1)
        grid_probs_stack = torch.stack(sample_grid_probs, dim=1)
        best_grid_probs = grid_probs_stack[torch.arange(grid_probs_stack.shape[0], device=self.device), best_idx]
        mean_grid_probs = grid_probs_stack.mean(dim=1)
        disagreement_grid = ((grid_probs_stack > 0.5).float().std(dim=1) > 0).float().mean(dim=-1)

        grid_size = self.config.visualization_grid_size
        for row_idx in range(batch.support_x.shape[0]):
            family = batch.family_name[row_idx]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
            panels = [
                (torch.sigmoid(grid_logits_enc[row_idx].squeeze(-1)).reshape(grid_size, grid_size).cpu(), f"encoder acc={acc_enc[row_idx].item():.3f}"),
                (best_grid_probs[row_idx].reshape(grid_size, grid_size).cpu(), f"best sample acc={acc_stack[row_idx, best_idx[row_idx]].item():.3f}"),
                (mean_grid_probs[row_idx].reshape(grid_size, grid_size).cpu(), "diffusion mean prob"),
                ((grid_probs_stack[row_idx] > 0.5).float().std(dim=0).reshape(grid_size, grid_size).cpu(), f"sample disagreement={disagreement_grid[row_idx].item():.3f}"),
            ]
            extent = (-2.4, 2.4, -2.4, 2.4)
            for ax, (image, title) in zip(axes, panels):
                ax.imshow(image.T, origin="lower", extent=extent, vmin=0.0, vmax=1.0, cmap="coolwarm", alpha=0.85)
                sx = batch.support_x[row_idx, :, 0].cpu().numpy()
                sy = batch.support_x[row_idx, :, 1].cpu().numpy()
                sl = batch.support_y[row_idx, :, 0].cpu().numpy()
                qx = batch.query_x[row_idx, :, 0].cpu().numpy()
                qy = batch.query_x[row_idx, :, 1].cpu().numpy()
                ql = batch.query_y[row_idx, :, 0].cpu().numpy()
                ax.scatter(qx[ql < 0.5], qy[ql < 0.5], s=8, marker=".", color="black", alpha=0.15)
                ax.scatter(qx[ql > 0.5], qy[ql > 0.5], s=8, marker=".", color="white", alpha=0.2)
                ax.scatter(sx[sl < 0.5], sy[sl < 0.5], s=30, edgecolor="black", facecolor="tab:blue", linewidth=0.6)
                ax.scatter(sx[sl > 0.5], sy[sl > 0.5], s=30, edgecolor="black", facecolor="tab:orange", linewidth=0.6)
                ax.set_title(title)
                ax.set_xlim(-2.4, 2.4)
                ax.set_ylim(-2.4, 2.4)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{prefix} | family={family}")
            out_path = self.output_dir / "plots" / f"{prefix}_{row_idx:02d}_{family}.png"
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            paths.append(str(out_path.relative_to(self.output_dir)))
        return paths

    def save_checkpoint(self) -> None:
        payload = {
            "config": asdict(self.config),
            "system": self.system.state_dict(),
            "denoiser": self.denoiser.state_dict(),
        }
        torch.save(payload, self.output_dir / "checkpoint.pt")

    def run(self) -> Dict[str, object]:
        stage1 = MetricsTracker()
        stage2 = MetricsTracker()

        t0 = time.time()
        for step in range(1, self.config.train_steps_stage1 + 1):
            metrics = self.stage1_step(self.sample_batch())
            stage1.update(**metrics)
            if step % 100 == 0 or step == 1:
                print(
                    f"[stage1] step={step:4d} loss={stage1.mean_last('stage1_loss', 20):.4f} "
                    f"acc={stage1.mean_last('stage1_acc', 20):.3f} latent_norm={stage1.mean_last('latent_norm', 20):.3f}"
                )

        for step in range(1, self.config.train_steps_stage2 + 1):
            metrics = self.stage2_step(self.sample_batch())
            stage2.update(**metrics)
            if step % 100 == 0 or step == 1:
                print(
                    f"[stage2] step={step:4d} loss={stage2.mean_last('stage2_loss', 20):.4f} "
                    f"z0_rmse={stage2.mean_last('stage2_rmse', 20):.4f}"
                )

        summary = self.evaluate()
        summary["artifacts"] = {"plots_train": self._plot_episode_set(self.config.families, "train")}
        if self.config.eval_families:
            summary["generalization"] = {
                "eval_families": self.config.eval_families,
                "eval_summary": self.evaluate(
                    num_batches=max(24, self.config.eval_batches // 2),
                    family_names=self.config.eval_families,
                ),
            }
            summary["artifacts"]["plots_eval"] = self._plot_episode_set(self.config.eval_families, "eval")

        elapsed = time.time() - t0
        summary["runtime_seconds"] = elapsed
        summary["config"] = asdict(self.config)

        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        stage1.save_json(self.output_dir / "stage1_metrics.json")
        stage2.save_json(self.output_dir / "stage2_metrics.json")
        self.save_checkpoint()
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy hyperdiffusion experiment")
    parser.add_argument("--output-dir", type=str, default="runs/default")
    parser.add_argument("--families", type=str, nargs="+", default=DEFAULT_TRAIN_FAMILIES)
    parser.add_argument("--eval-families", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps-stage1", type=int, default=1200)
    parser.add_argument("--train-steps-stage2", type=int, default=1200)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-size", type=int, default=16)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--encoder-hidden", type=int, default=128)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--denoiser-hidden", type=int, default=128)
    parser.add_argument("--denoiser-depth", type=int, default=4)
    parser.add_argument("--target-hidden-dim", type=int, default=64)
    parser.add_argument("--target-depth", type=int, default=4)
    parser.add_argument("--learning-rate-stage1", type=float, default=1e-3)
    parser.add_argument("--learning-rate-stage2", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--num-diffusion-steps", type=int, default=20)
    parser.add_argument("--diagnostic-samples", type=int, default=8)
    parser.add_argument("--mismatch-batches", type=int, default=16)
    parser.add_argument("--support-size-sweep", type=int, nargs="*", default=[2, 4, 8, 16])
    parser.add_argument("--visualization-count", type=int, default=4)
    parser.add_argument("--visualization-grid-size", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        families=args.families,
        eval_families=args.eval_families,
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
        device=args.device,
        seed=args.seed,
    )
    experiment = Experiment(config=config, output_dir=Path(args.output_dir))
    summary = experiment.run()
    print("\nFinal summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
