from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .diffusion import DiffusionSchedule, ddim_sample
from .models import CandidateSelector, DiffusionDenoiser, HyperNetworkSystem, TargetArchitecture, functional_target_network
from .protocol import PROTOCOL_SUITES, resolve_protocol_split
from .tasks import (
    BANDIT_FAMILIES,
    BANDIT_TRAIN_GROUPS,
    CONTROL_FAMILIES,
    CONTROL_TRAIN_GROUPS,
    DEFAULT_BANDIT_EVAL_FAMILIES,
    DEFAULT_BANDIT_TRAIN_FAMILIES,
    DEFAULT_CONTROL_EVAL_FAMILIES,
    DEFAULT_CONTROL_TRAIN_FAMILIES,
    DEFAULT_REGRESSION_EVAL_FAMILIES,
    DEFAULT_REGRESSION_TRAIN_FAMILIES,
    DEFAULT_TRAIN_FAMILIES,
    EXPANDED_TRAIN_FAMILIES,
    ORIGINAL_TRAIN_GROUPS,
    REGRESSION_TRAIN_GROUPS,
    TASK_TEXT_DESCRIPTIONS,
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
    selector_enabled: bool = False
    selector_hidden: int = 128
    selector_lr: float = 1e-3
    selector_num_samples: int = 8
    encoding_mode: str = "support"
    text_embedding_dim: int = 768
    text_mix_alpha: float = 0.5
    reward_audit_batches: int = 8
    reward_audit_batch_size: int = 16
    device: str = "cpu"
    seed: int = 0
    protocol_suite: str = "held_out"
    strict_ood: bool = True


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
        protocol = resolve_protocol_split(
            task_type=config.task_type,
            suite=config.protocol_suite,
            train_families=config.families,
            eval_families=config.eval_families,
            strict_ood=config.strict_ood,
        )
        config.families = list(protocol.train_families)
        config.eval_families = list(protocol.eval_families)

        self.config = config
        self.protocol = protocol
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        self.device = torch.device(config.device)
        if config.task_type == "classification":
            input_dim = 2
        elif config.task_type in ("regression", "bandit_regression"):
            input_dim = 2 if config.task_type == "bandit_regression" else 1
        elif config.task_type == "control":
            input_dim = 2  # state dimension for control tasks
        else:
            raise ValueError(f"Unknown task_type: {config.task_type}")
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

        self.encoding_mode = config.encoding_mode
        self.text_embedding_dim = config.text_embedding_dim
        self.text_mix_alpha = config.text_mix_alpha
        _distilbert_dim = 768
        self.text_projector = torch.nn.Sequential(
            torch.nn.Linear(_distilbert_dim, config.encoder_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(config.encoder_hidden, config.cond_dim + config.latent_dim),
        ).to(self.device)
        family_set = set(config.families or []) | set(config.eval_families or []) | set(TASK_TEXT_DESCRIPTIONS.keys())
        self.family_to_index = {name: idx for idx, name in enumerate(sorted(family_set))}
        self._text_emb_cache: Dict[str, torch.Tensor] = {}
        if config.encoding_mode in ("text", "hybrid"):
            self._build_text_embedding_cache()

        # Ablation baseline with no hypernetwork / fast weights
        from .models import BaselineNetwork
        self.baseline = BaselineNetwork(input_dim=input_dim, output_dim=1, hidden_dim=config.encoder_hidden, depth=4).to(self.device)
        self.opt_baseline = torch.optim.AdamW(self.baseline.parameters(), lr=config.learning_rate_stage1)
        self.denoiser = DiffusionDenoiser(
            latent_dim=config.latent_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.denoiser_hidden,
            depth=config.denoiser_depth,
        ).to(self.device)
        self.schedule = DiffusionSchedule(num_steps=config.num_diffusion_steps).to(self.device)

        self.selector = None
        if config.selector_enabled:
            self.selector = CandidateSelector(cond_dim=config.cond_dim, latent_dim=config.latent_dim, hidden_dim=config.selector_hidden).to(self.device)
            self.opt_selector = torch.optim.AdamW(self.selector.parameters(), lr=config.selector_lr)

        stage1_params = list(self.system.parameters()) + list(self.text_projector.parameters())
        self.opt_stage1 = torch.optim.AdamW(stage1_params, lr=config.learning_rate_stage1)
        self.opt_stage2 = torch.optim.AdamW(self.denoiser.parameters(), lr=config.learning_rate_stage2)
        self.generator = torch.Generator().manual_seed(config.seed)

    def _family_description(self, family_name: str) -> str:
        return TASK_TEXT_DESCRIPTIONS.get(family_name, f"Task family {family_name.replace('_', ' ')}")

    def _build_text_embedding_cache(self) -> None:
        """Pre-compute distilbert CLS embeddings for all known task family descriptions."""
        from transformers import AutoTokenizer, AutoModel  # lazy import to avoid startup cost
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert = AutoModel.from_pretrained(model_name)
        bert.eval()
        all_names = sorted(set(self.family_to_index.keys()) | set(TASK_TEXT_DESCRIPTIONS.keys()))
        texts = [self._family_description(n) for n in all_names]
        with torch.no_grad():
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=64, padding=True)
            out = bert(**inputs)
            cls_embs = out.last_hidden_state[:, 0, :]  # (N, 768)
        for name, emb in zip(all_names, cls_embs):
            self._text_emb_cache[name] = emb.to(self.device)

    def _text_embedding(self, family_names: List[str]) -> torch.Tensor:
        fallback = torch.zeros(768, dtype=torch.float32, device=self.device)
        return torch.stack([self._text_emb_cache.get(n, fallback) for n in family_names], dim=0)

    def _encode_batch(self, support_x: torch.Tensor, support_y: torch.Tensor, family_names: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        context_support, latent_support = self.system.encode(support_x, support_y)
        mode = self.encoding_mode
        if mode == "support":
            return context_support, latent_support

        if mode == "oracle":
            indices = torch.tensor([self.family_to_index.get(name, 0) for name in family_names], device=self.device, dtype=torch.long)
            one_hot = F.one_hot(indices, num_classes=max(len(self.family_to_index), 1)).float()
            target_dim = self.config.cond_dim + self.config.latent_dim
            if one_hot.shape[-1] < target_dim:
                one_hot = F.pad(one_hot, (0, target_dim - one_hot.shape[-1]))
            oracle = one_hot[:, :target_dim]
            context_oracle, latent_oracle = torch.split(oracle, [self.config.cond_dim, self.config.latent_dim], dim=-1)
            return context_oracle, latent_oracle

        text_embed = self._text_embedding(family_names)
        text_proj = self.text_projector(text_embed)
        context_text, latent_text = torch.split(text_proj, [self.config.cond_dim, self.config.latent_dim], dim=-1)

        if mode == "text":
            return context_text, latent_text
        if mode == "hybrid":
            alpha = float(self.text_mix_alpha)
            context = (1.0 - alpha) * context_support + alpha * context_text
            latent = (1.0 - alpha) * latent_support + alpha * latent_text
            return context, latent
        raise ValueError(f"Unknown encoding_mode: {mode}")

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
            family_instances=batch.family_instances,
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
        context, latent = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
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
        # Baseline update: train static baseline on support points (no per-episode fast weights)
        self.baseline.train()
        baseline_x = batch.support_x.view(-1, batch.support_x.shape[-1])
        baseline_y = batch.support_y.view(-1, batch.support_y.shape[-1])
        baseline_pred = self.baseline(baseline_x)
        baseline_loss = self._task_loss(baseline_pred, baseline_y)
        self.opt_baseline.zero_grad(set_to_none=True)
        baseline_loss.backward()
        self.opt_baseline.step()
        with torch.no_grad():
            query_baseline_pred = self.baseline(batch.query_x.view(-1, batch.query_x.shape[-1])).view(batch.query_x.shape[0], batch.query_x.shape[1], -1)
            baseline_metric = self._item_metric(query_baseline_pred, batch.query_y).mean().item()

        return {
            "stage1_loss": loss.item(),
            "stage1_task_loss": task_loss.item(),
            f"stage1_{self._metric_name()}": metric,
            "latent_norm": latent.norm(dim=-1).mean().item(),
            "baseline_loss": baseline_loss.item(),
            "baseline_metric": baseline_metric,
        }

    def stage2_step(self, batch: EpisodeBatch) -> Dict[str, float]:
        self.denoiser.train()
        self.system.eval()
        batch = self.to_device(batch)
        with torch.no_grad():
            context, latent = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)

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
        result = {"stage2_loss": loss.item(), "stage2_rmse": rmse}

        if self.selector is not None:
            k = self.config.selector_num_samples
            z_cands, pred_cands, metric_cands = [], [], []
            with torch.no_grad():
                for _ in range(k):
                    z_temp = ddim_sample(self.denoiser, self.schedule, context, num_steps=self.config.num_diffusion_steps)
                    preds_temp = functional_target_network(batch.query_x, self.system.decode(z_temp, context))
                    metric_temp = self._item_metric(preds_temp, batch.query_y)
                    z_cands.append(z_temp)
                    metric_cands.append(metric_temp)

            z_cands = torch.stack(z_cands, dim=1)
            metric_cands = torch.stack(metric_cands, dim=1)
            best_idx = metric_cands.argmax(dim=1)

            scores = self.selector(context, z_cands)
            selector_loss = F.cross_entropy(scores, best_idx)

            self.opt_selector.zero_grad(set_to_none=True)
            selector_loss.backward()
            self.opt_selector.step()

            result["selector_loss"] = selector_loss.item()
        else:
            result["selector_loss"] = 0.0

        return result

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
            "encoder_metric": [], "diffusion_metric": [], "selector_metric": [], "baseline_metric": [], "encoder_loss": [], "diffusion_loss": [], "selector_loss": [], "baseline_loss": [],
            "diffusion_metric_mean_k": [], "diffusion_metric_best_k": [], "prediction_disagreement": [],
            "weight_pairwise_l2": [], "latent_pairwise_l2": [],
            "uncertainty_error_correlation": [], "uncertainty_mean": [], "uncertainty_on_high_error_points": [], "uncertainty_on_low_error_points": []
        }

        for _ in range(num_batches):
            batch = self.to_device(self.sample_batch(family_names=families))
            context, latent = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
            pred_enc, metric_enc, _ = self._decode_and_score(batch.query_x, batch.query_y, latent, context)
            loss_enc = self._item_loss(pred_enc, batch.query_y)

            # Baseline evaluation on query set
            b, q, _ = batch.query_x.shape
            baseline_query = self.baseline(batch.query_x.view(b * q, -1)).view(b, q, -1)
            baseline_metric = self._item_metric(baseline_query, batch.query_y)
            baseline_loss = self._item_loss(baseline_query, batch.query_y)

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
            b = metric_stack.shape[0]

            if self.selector is not None:
                scores = self.selector(context, z_stack)
                selected_idx = scores.argmax(dim=1)
                selected_metric = metric_stack[torch.arange(b), selected_idx]
                selected_pred = pred_stack[torch.arange(b), selected_idx]
            else:
                selected_idx = torch.zeros(b, dtype=torch.long, device=self.device)
                selected_metric = metric_diff
                selected_pred = pred_stack[:, 0]

            if self.config.task_type in ("regression", "bandit_regression", "control"):
                pred_mean = pred_stack.mean(dim=1)
                pred_var = pred_stack.var(dim=1, unbiased=False)
                se_mean = (pred_mean - batch.query_y).pow(2)
                mean_uncertainty = pred_var.mean(dim=(1, 2))
                mean_error = se_mean.mean(dim=(1, 2))
                cov = ((mean_uncertainty - mean_uncertainty.mean()) * (mean_error - mean_error.mean())).mean()
                denom = mean_uncertainty.std(unbiased=False) * mean_error.std(unbiased=False) + 1e-8
                uncertainty_error_correlation = float(cov / denom)
                flat_err = se_mean.view(-1)
                flat_unc = pred_var.view(-1)
                if flat_err.numel() > 0:
                    high_cut = flat_err.quantile(0.75)
                    low_cut = flat_err.quantile(0.25)
                    high_mask = flat_err >= high_cut
                    low_mask = flat_err <= low_cut
                    uncertainty_on_high_error_points = float(flat_unc[high_mask].mean()) if high_mask.any() else 0.0
                    uncertainty_on_low_error_points = float(flat_unc[low_mask].mean()) if low_mask.any() else 0.0
                else:
                    uncertainty_on_high_error_points = 0.0
                    uncertainty_on_low_error_points = 0.0
                uncertainty_mean = float(mean_uncertainty.mean())
            else:
                uncertainty_error_correlation = 0.0
                uncertainty_mean = 0.0
                uncertainty_on_high_error_points = 0.0
                uncertainty_on_low_error_points = 0.0

            for idx, family in enumerate(batch.family_name):
                fam = per_family.setdefault(family, {k: [] for k in overall})
                vals = {
                    "encoder_metric": metric_enc[idx].item(),
                    "diffusion_metric": metric_diff[idx].item(),
                    "selector_metric": selected_metric[idx].item(),
                    "encoder_loss": loss_enc[idx].item(),
                    "diffusion_loss": loss_diff[idx].item(),
                    "selector_loss": 0.0,
                    "baseline_metric": baseline_metric[idx].item(),
                    "baseline_loss": baseline_loss[idx].item(),
                    "diffusion_metric_mean_k": metric_mean[idx].item(),
                    "diffusion_metric_best_k": metric_best[idx].item(),
                    "prediction_disagreement": disagreement[idx].item(),
                    "weight_pairwise_l2": weight_l2[idx].item(),
                    "latent_pairwise_l2": latent_l2[idx].item(),
                    "uncertainty_error_correlation": uncertainty_error_correlation,
                    "uncertainty_mean": uncertainty_mean,
                    "uncertainty_on_high_error_points": uncertainty_on_high_error_points,
                    "uncertainty_on_low_error_points": uncertainty_on_low_error_points,
                }
                for k, v in vals.items():
                    overall[k].append(v)
                    fam[k].append(v)

        metric_name = self._metric_name()
        def pack(values: Dict[str, List[float]]) -> Dict[str, float]:
            packed = {
                f"encoder_{metric_name}": self._mean(values["encoder_metric"]),
                f"diffusion_{metric_name}": self._mean(values["diffusion_metric"]),
                f"selector_{metric_name}": self._mean(values.get("selector_metric", [])) if values.get("selector_metric") else 0.0,
                    f"baseline_{metric_name}": self._mean(values.get("baseline_metric", [])) if values.get("baseline_metric") else 0.0,
                "encoder_loss": self._mean(values["encoder_loss"]),
                "diffusion_loss": self._mean(values["diffusion_loss"]),
                "selector_loss": self._mean(values.get("selector_loss", [])) if values.get("selector_loss") else 0.0,
                    "baseline_loss": self._mean(values.get("baseline_loss", [])) if values.get("baseline_loss") else 0.0,
                f"diffusion_{metric_name}_mean_k": self._mean(values["diffusion_metric_mean_k"]),
                f"diffusion_{metric_name}_best_k": self._mean(values["diffusion_metric_best_k"]),
                "prediction_disagreement": self._mean(values["prediction_disagreement"]),
                "weight_pairwise_l2": self._mean(values["weight_pairwise_l2"]),
                "latent_pairwise_l2": self._mean(values["latent_pairwise_l2"]),
                "uncertainty_error_correlation": self._mean(values.get("uncertainty_error_correlation", [])) if values.get("uncertainty_error_correlation") else 0.0,
                "uncertainty_mean": self._mean(values.get("uncertainty_mean", [])) if values.get("uncertainty_mean") else 0.0,
                "uncertainty_on_high_error_points": self._mean(values.get("uncertainty_on_high_error_points", [])) if values.get("uncertainty_on_high_error_points") else 0.0,
                "uncertainty_on_low_error_points": self._mean(values.get("uncertainty_on_low_error_points", [])) if values.get("uncertainty_on_low_error_points") else 0.0,
            }
            return packed

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
                "selector": {metric_name: pack(overall)[f"selector_{metric_name}"], "loss": pack(overall)["selector_loss"]},
                "static_baseline": {metric_name: pack(overall)[f"baseline_{metric_name}"], "loss": pack(overall)["baseline_loss"]},
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
                support_names = [name for keep, name in zip(mismatch_mask, support_batch.family_name) if keep]
                context, latent = self._encode_batch(support_batch.support_x[idx], support_batch.support_y[idx], support_names)
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
                context, latent = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
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
        paths: List[str] = []
        plot_fn = self._plot_classification_batch if self.config.task_type == "classification" else self._plot_regression_batch
        for family_name in family_names:
            batch = self.to_device(self.sample_batch(batch_size=self.config.visualization_count, family_names=[family_name]))
            paths.extend(plot_fn(batch, f"{prefix}_{family_name}"))
        return paths

    @torch.no_grad()
    def _plot_classification_batch(self, batch: EpisodeBatch, prefix: str) -> List[str]:
        paths: List[str] = []
        context, latent = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
        b = batch.support_x.shape[0]
        grid_size = self.config.visualization_grid_size
        grid_lin = torch.linspace(-2.4, 2.4, grid_size, device=self.device)
        gx, gy = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0).expand(b, -1, -1)
        enc_grid = functional_target_network(grid, self.system.decode(latent, context))
        acc_enc = self._item_metric(functional_target_network(batch.query_x, self.system.decode(latent, context)), batch.query_y)
        baseline_grid = torch.sigmoid(self.baseline(grid.reshape(b * grid.shape[1], -1)).reshape(b, grid.shape[1], -1))
        baseline_pred = self.baseline(batch.query_x.reshape(batch.query_x.shape[0] * batch.query_x.shape[1], -1)).reshape(batch.query_x.shape[0], batch.query_x.shape[1], -1)
        baseline_acc = self._item_metric(baseline_pred, batch.query_y)

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
        baseline_gap = (best_grid - baseline_grid).abs()

        for i in range(b):
            fig, axes = plt.subplots(1, 6, figsize=(15, 2.5), constrained_layout=True)
            best_acc_i = acc_stack[i, best_idx[i]].item()
            base_acc_i = baseline_acc[i].item()
            panels = [
                (torch.sigmoid(enc_grid[i].squeeze(-1)).reshape(grid_size, grid_size).cpu(), f"HN-det {acc_enc[i].item():.3f}"),
                (best_grid[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), f"best {best_acc_i:.3f}"),
                (baseline_grid[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), f"MLP-fixed {baseline_acc[i].item():.3f}"),
                (mean_grid[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), "mean prob"),
                (baseline_gap[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), "best−MLP-fixed |Δp|"),
                (disagreement[i].squeeze(-1).reshape(grid_size, grid_size).cpu(), "sample std"),
            ]
            for ax, (image, title) in zip(axes, panels):
                ax.imshow(image.T, origin="lower", extent=(-2.4, 2.4, -2.4, 2.4), cmap="coolwarm", alpha=0.85)
                sx = batch.support_x[i, :, 0].cpu().numpy(); sy = batch.support_x[i, :, 1].cpu().numpy(); sl = batch.support_y[i, :, 0].cpu().numpy()
                ax.scatter(sx[sl < 0.5], sy[sl < 0.5], s=12, edgecolor="black", facecolor="tab:blue", linewidth=0.4)
                ax.scatter(sx[sl > 0.5], sy[sl > 0.5], s=12, edgecolor="black", facecolor="tab:orange", linewidth=0.4)
                ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{prefix} | family={batch.family_name[i]} | Δ(best−MLP-fixed)={best_acc_i - base_acc_i:.3f}", fontsize=8)
            out = self.output_dir / "plots" / f"{prefix}_{i:02d}.png"
            fig.savefig(out, dpi=90, bbox_inches='tight'); plt.close(fig)
            paths.append(str(out.relative_to(self.output_dir)))
        return paths

    @torch.no_grad()
    def _plot_regression_batch(self, batch: EpisodeBatch, prefix: str) -> List[str]:
        paths: List[str] = []
        context, latent = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
        b = batch.support_x.shape[0]
        input_dim = batch.support_x.shape[-1]
        
        if input_dim == 1:
            # 1D regression - use line plots
            grid_x = torch.linspace(-3.2, 3.2, 256, device=self.device).view(1, -1, 1).expand(b, -1, -1)

            baseline_curve = self.baseline(grid_x.reshape(b * grid_x.shape[1], -1)).reshape(b, grid_x.shape[1], -1)
            baseline_query = self.baseline(batch.query_x.reshape(b * batch.query_x.shape[1], -1)).reshape(b, batch.query_x.shape[1], -1)
            baseline_metric = self._item_metric(baseline_query, batch.query_y)
            
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
            baseline_gap_curve = (best_curve - baseline_curve).abs()

            for i in range(b):
                fig, axes = plt.subplots(1, 6, figsize=(15, 2.5), constrained_layout=True)
                xg = grid_x[i, :, 0].cpu().numpy()
                support_x = batch.support_x[i, :, 0].cpu().numpy()
                support_y = batch.support_y[i, :, 0].cpu().numpy()
                query_x = batch.query_x[i, :, 0].cpu().numpy()
                query_y = batch.query_y[i, :, 0].cpu().numpy()
                best_r2_i = metric_stack[i, best_idx[i]].item()
                base_r2_i = baseline_metric[i].item()
                panels = [
                    (enc_curve[i, :, 0].cpu().numpy(), f"HN-det r2={metric_enc[i].item():.3f}"),
                    (best_curve[i, :, 0].cpu().numpy(), f"best r2={best_r2_i:.3f}"),
                    (baseline_curve[i, :, 0].cpu().numpy(), f"MLP-fixed r2={baseline_metric[i].item():.3f}"),
                    (mean_curve[i, :, 0].cpu().numpy(), "HN-diff mean"),
                    (baseline_gap_curve[i, :, 0].cpu().numpy(), "best−MLP-fixed |Δy|"),
                    (std_curve[i, :, 0].cpu().numpy(), "sample std"),
                ]
                for ax, (curve, title) in zip(axes, panels):
                    ax.scatter(query_x, query_y, s=5, alpha=0.25, color="tab:gray")
                    ax.scatter(support_x, support_y, s=12, color="tab:orange", edgecolors="black", linewidths=0.4)
                    ax.plot(xg, curve, lw=0.9)
                    ax.set_title(title, fontsize=7)
                    ax.tick_params(labelsize=6)
                mode = "Evaluation (generalization)" if prefix.startswith("eval") else "Training (in-distribution)"
                fig.suptitle(f"{prefix} | {mode} | family={batch.family_name[i]} | Δ(best−MLP-fixed)={best_r2_i - base_r2_i:.3f}", fontsize=8)
                out = self.output_dir / "plots" / f"{prefix}_{i:02d}.png"
                fig.savefig(out, dpi=90, bbox_inches='tight'); plt.close(fig)
                paths.append(str(out.relative_to(self.output_dir)))
        else:
            # 2D regression (bandit tasks) - use contour plots
            grid_size = self.config.visualization_grid_size
            grid_lin = torch.linspace(-2.5, 2.5, grid_size, device=self.device)
            gx, gy = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
            grid_2d = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0).expand(b, -1, -1)

            baseline_surface = self.baseline(grid_2d.reshape(b * grid_2d.shape[1], -1)).reshape(b, grid_size, grid_size)
            baseline_pred = self.baseline(batch.query_x.reshape(batch.query_x.shape[0] * batch.query_x.shape[1], -1)).reshape(batch.query_x.shape[0], batch.query_x.shape[1], -1)
            baseline_metric = self._item_metric(baseline_pred, batch.query_y)

            enc_surface = functional_target_network(grid_2d, self.system.decode(latent, context)).reshape(b, grid_size, grid_size)
            pred_enc = functional_target_network(batch.query_x, self.system.decode(latent, context))
            metric_enc = self._item_metric(pred_enc, batch.query_y)

            surface_list, metric_list, pred_list = [], [], []
            for _ in range(self.config.diagnostic_samples):
                z = ddim_sample(self.denoiser, self.schedule, context, num_steps=self.config.num_diffusion_steps)
                params = self.system.decode(z, context)
                surface_list.append(functional_target_network(grid_2d, params).reshape(b, grid_size, grid_size))
                pred = functional_target_network(batch.query_x, params)
                pred_list.append(pred)
                metric_list.append(self._item_metric(pred, batch.query_y))

            surface_stack = torch.stack(surface_list, dim=1)
            pred_stack = torch.stack(pred_list, dim=1)
            metric_stack = torch.stack(metric_list, dim=1)
            best_idx = metric_stack.argmax(dim=1)
            best_surface = surface_stack[torch.arange(b, device=self.device), best_idx]
            mean_surface = surface_stack.mean(dim=1)
            std_surface = surface_stack.std(dim=1)
            best_baseline_gap_surface = (best_surface - baseline_surface).abs()

            # Compute ground truth surface if available
            gt_surfaces = []
            for i in range(b):
                family = batch.family_instances[i] if batch.family_instances else None
                if family and hasattr(family, 'f'):
                    gt_surf = family.f(grid_2d[i]).squeeze().reshape(grid_size, grid_size).cpu().numpy()
                else:
                    gt_surf = None
                gt_surfaces.append(gt_surf)

            global_vmin = torch.cat([enc_surface.flatten(), best_surface.flatten(), mean_surface.flatten(), baseline_surface.flatten()]).min().item()
            global_vmax = torch.cat([enc_surface.flatten(), best_surface.flatten(), mean_surface.flatten(), baseline_surface.flatten()]).max().item()
            std_vmax = std_surface.max().item() if std_surface.numel() > 0 else 1.0

            for i in range(b):
                support_x = batch.support_x[i, :, 0].cpu().numpy()
                support_y = batch.support_x[i, :, 1].cpu().numpy()
                support_z = batch.support_y[i, :, 0].cpu().numpy()
                query_x = batch.query_x[i, :, 0].cpu().numpy()
                query_y = batch.query_x[i, :, 1].cpu().numpy()
                query_z = batch.query_y[i, :, 0].cpu().numpy()

                # Compute errors for each surface
                enc_pred = pred_enc[i].squeeze(-1).cpu().numpy()
                best_pred = pred_stack[i, best_idx[i]].squeeze(-1).cpu().numpy()
                mean_pred = pred_stack[i].mean(dim=0).squeeze(-1).cpu().numpy()
                baseline_query_pred = baseline_pred[i].squeeze(-1).cpu().numpy()
                enc_error = np.abs(enc_pred - query_z)
                best_error = np.abs(best_pred - query_z)
                mean_error = np.abs(mean_pred - query_z)
                baseline_error = np.abs(baseline_query_pred - query_z)
                error_vmax = max(enc_error.max(), best_error.max(), mean_error.max(), baseline_error.max())

                surfaces = [
                    (enc_surface[i].cpu().numpy(), f"HN-det r2={metric_enc[i].item():.3f}", enc_error),
                    (best_surface[i].cpu().numpy(), f"best r2={metric_stack[i, best_idx[i]].item():.3f}", best_error),
                    (baseline_surface[i].cpu().numpy(), f"MLP-fixed r2={baseline_metric[i].item():.3f}", baseline_error),
                    (mean_surface[i].cpu().numpy(), "HN-diff mean", mean_error),
                    (best_baseline_gap_surface[i].cpu().numpy(), "best−MLP-fixed |Δy|", None),
                    (std_surface[i].cpu().numpy(), "sample std", None),
                ]
                if gt_surfaces[i] is not None:
                    surfaces.insert(3, (gt_surfaces[i], "ground truth", None))

                num_panels = len(surfaces)
                if num_panels == 6:
                    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
                else:
                    fig, axes = plt.subplots(2, 3 if num_panels > 4 else 2, figsize=(12 if num_panels > 4 else 9, 7), constrained_layout=True)

                for ax, (surface, title, error) in zip(axes.flat, surfaces):
                    if surface is None:
                        ax.text(0.5, 0.5, "No ground truth", ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(title)
                        continue
                    if title == "sample std":
                        img = ax.imshow(surface, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5), cmap="viridis", vmin=0.0, vmax=std_vmax, aspect="equal")
                    elif title == "best-static |Δy|":
                        gap_vmax = max(best_baseline_gap_surface[i].max().item(), 1e-6)
                        img = ax.imshow(surface, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5), cmap="magma", vmin=0.0, vmax=gap_vmax, aspect="equal")
                    else:
                        img = ax.imshow(surface, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5), cmap="viridis", vmin=global_vmin, vmax=global_vmax, aspect="equal")
                    ax.contour(surface, levels=12, colors="k", linewidths=0.6, extent=(-2.5, 2.5, -2.5, 2.5), alpha=0.6)
                    ax.scatter(support_x, support_y, c=support_z, s=30, edgecolors="red", linewidth=0.8, cmap="RdYlBu", vmin=global_vmin, vmax=global_vmax)
                    if error is not None:
                        ax.scatter(query_x, query_y, c=error, s=20, alpha=0.8, marker="x", cmap="hot", vmin=0, vmax=error_vmax, edgecolors="k")
                    else:
                        ax.scatter(query_x, query_y, c=query_z, s=15, alpha=0.6, marker="x", cmap="RdYlBu", vmin=global_vmin, vmax=global_vmax)
                    ax.set_title(title, fontsize=7)
                    ax.set_xlabel("x₁", fontsize=6)
                    ax.set_ylabel("x₂", fontsize=6)
                    ax.tick_params(labelsize=5)
                    fig.colorbar(img, ax=ax, shrink=0.65)

                for ax in axes.flat[len(surfaces):]:
                    ax.axis("off")

                mode = "Evaluation (generalization)" if prefix.startswith("eval") else "Training (in-distribution)"
                fig.suptitle(f"{prefix} | {mode} | family={batch.family_name[i]} | Δ(best−MLP-fixed)={metric_stack[i, best_idx[i]].item() - baseline_metric[i].item():.3f}", fontsize=8)
                out = self.output_dir / "plots" / f"{prefix}_{i:02d}.png"
                fig.savefig(out, dpi=90, bbox_inches='tight'); plt.close(fig)
                paths.append(str(out.relative_to(self.output_dir)))

                # Control-specific reward-over-time plot
                if self.config.task_type == "control":
                    family = CONTROL_FAMILIES.get(batch.family_name[i])
                    if family is not None:
                        params = self.system.decode(latent[i : i + 1], context[i : i + 1])

                        def policy_fn(state: torch.Tensor) -> torch.Tensor:
                            with torch.no_grad():
                                x_reshaped = state.view(1, 1, -1).to(self.device)
                                u = functional_target_network(x_reshaped, params).squeeze().item()  # scalar action
                                return torch.tensor([u], dtype=torch.float32)

                        def zero_action_policy(state: torch.Tensor) -> torch.Tensor:
                            return torch.zeros(1, dtype=torch.float32)

                        def baseline_policy(state: torch.Tensor) -> torch.Tensor:
                            with torch.no_grad():
                                u = self.baseline(state.view(1, -1).to(self.device)).squeeze().item()
                                return torch.tensor([u], dtype=torch.float32)

                        initial_state = batch.support_x[i, 0]
                        rollout_data = family.rollout(policy_fn, initial_state, num_steps=80, dt=0.05)
                        baseline_rollout = family.rollout(zero_action_policy, initial_state, num_steps=80, dt=0.05)
                        static_rollout = family.rollout(baseline_policy, initial_state, num_steps=80, dt=0.05)

                        fig_r, ax_r = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)
                        ax_r.plot(rollout_data["rewards"], label="HN-det step", color="tab:blue", lw=0.9)
                        ax_r.plot(static_rollout["rewards"], label="MLP-fixed step", color="tab:red", linestyle="--", lw=0.8)
                        ax_r.plot(baseline_rollout["rewards"], label="zero step", color="tab:green", linestyle="--", lw=0.8)
                        ax_r.plot(rollout_data["cumulative_rewards"], label="HN-det cumul.", color="tab:orange", lw=0.9)
                        ax_r.plot(static_rollout["cumulative_rewards"], label="MLP-fixed cumul.", color="tab:red", linestyle=":", lw=0.8)
                        ax_r.plot(baseline_rollout["cumulative_rewards"], label="zero cumul.", color="tab:green", linestyle=":", lw=0.8)
                        ax_r.set_xlabel("time step", fontsize=7)
                        ax_r.set_ylabel("higher is better", fontsize=7)
                        ax_r.set_title(f"{prefix} reward | family={batch.family_name[i]}", fontsize=7)
                        ax_r.legend(fontsize=6, ncol=2)
                        ax_r.tick_params(labelsize=6)
                        out_r = self.output_dir / "plots" / f"{prefix}_{i:02d}_reward.png"
                        fig_r.savefig(out_r, dpi=90, bbox_inches='tight')
                        plt.close(fig_r)
                        paths.append(str(out_r.relative_to(self.output_dir)))

        return paths

    def _group_evaluations(self) -> Dict[str, object]:
        if self.config.task_type == "classification":
            groups = ORIGINAL_TRAIN_GROUPS
        elif self.config.task_type == "regression":
            groups = REGRESSION_TRAIN_GROUPS
        elif self.config.task_type == "bandit_regression":
            groups = BANDIT_TRAIN_GROUPS
        elif self.config.task_type == "control":
            groups = CONTROL_TRAIN_GROUPS
        else:
            groups = {}
        configured = set(self.config.families)
        out: Dict[str, object] = {}
        for name, fams in groups.items():
            active = [f for f in fams if f in configured]
            if active:
                out[name] = {"families": active, "summary": self.evaluate(num_batches=self.config.eval_batches, family_names=active)}
        return out

    @staticmethod
    def _summarize_reward_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
        if not rows:
            return {
                "episodes": 0,
                "mean_delta_ls": 0.0,
                "median_delta_ls": 0.0,
                "win_rate_vs_static": 0.0,
                "p10_delta_ls": 0.0,
                "p90_delta_ls": 0.0,
                "mean_delta_lz": 0.0,
                "win_rate_vs_zero": 0.0,
            }
        dls = np.asarray([r["delta_ls"] for r in rows], dtype=float)
        dlz = np.asarray([r["delta_lz"] for r in rows], dtype=float)
        return {
            "episodes": int(len(rows)),
            "mean_delta_ls": float(dls.mean()),
            "median_delta_ls": float(np.median(dls)),
            "win_rate_vs_static": float((dls > 0).mean()),
            "p10_delta_ls": float(np.quantile(dls, 0.10)),
            "p90_delta_ls": float(np.quantile(dls, 0.90)),
            "mean_delta_lz": float(dlz.mean()),
            "win_rate_vs_zero": float((dlz > 0).mean()),
        }

    @torch.no_grad()
    def _control_reward_audit(self, family_names: List[str], num_batches: int = 20, batch_size: int = 32) -> Dict[str, object]:
        rows: List[Dict[str, float]] = []
        for _ in range(num_batches):
            batch = self.to_device(self.sample_batch(batch_size=batch_size, family_names=family_names))
            context, latent = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
            for i in range(batch.support_x.shape[0]):
                family = CONTROL_FAMILIES.get(batch.family_name[i])
                if family is None:
                    continue
                params = self.system.decode(latent[i : i + 1], context[i : i + 1])

                def learned_policy(state: torch.Tensor) -> torch.Tensor:
                    x_reshaped = state.view(1, 1, -1).to(self.device)
                    action = functional_target_network(x_reshaped, params).squeeze().item()
                    return torch.tensor([action], dtype=torch.float32)

                def static_policy(state: torch.Tensor) -> torch.Tensor:
                    action = self.baseline(state.view(1, -1).to(self.device)).squeeze().item()
                    return torch.tensor([action], dtype=torch.float32)

                def zero_policy(state: torch.Tensor) -> torch.Tensor:
                    return torch.zeros(1, dtype=torch.float32)

                initial_state = batch.support_x[i, 0]
                learned = family.rollout(learned_policy, initial_state, num_steps=80, dt=0.05)
                static = family.rollout(static_policy, initial_state, num_steps=80, dt=0.05)
                zero = family.rollout(zero_policy, initial_state, num_steps=80, dt=0.05)
                l_final = float(learned["cumulative_rewards"][-1])
                s_final = float(static["cumulative_rewards"][-1])
                z_final = float(zero["cumulative_rewards"][-1])
                rows.append({
                    "family": batch.family_name[i],
                    "delta_ls": l_final - s_final,
                    "delta_lz": l_final - z_final,
                })

        by_family: Dict[str, object] = {}
        for family in sorted(set(r["family"] for r in rows)):
            family_rows = [r for r in rows if r["family"] == family]
            by_family[family] = self._summarize_reward_rows(family_rows)
        return {"overall": self._summarize_reward_rows(rows), "by_family": by_family}

    def save_checkpoint(self) -> None:
        payload = {
            "config": asdict(self.config),
            "system": self.system.state_dict(),
            "denoiser": self.denoiser.state_dict(),
            "baseline": self.baseline.state_dict(),
            "text_projector": self.text_projector.state_dict(),
        }
        torch.save(payload, self.output_dir / "checkpoint.pt")

    def run(self) -> Dict[str, object]:
        stage1 = MetricsTracker(); stage2 = MetricsTracker(); t0 = time.time()
        metric_name = self._metric_name()
        log_freq = max(1, self.config.train_steps_stage1 // 5)
        for step in range(1, self.config.train_steps_stage1 + 1):
            metrics = self.stage1_step(self.sample_batch())
            stage1.update(**metrics)
            if step % log_freq == 0 or step == 1:
                print(f"[stage1] step={step:4d} loss={stage1.mean_last('stage1_loss',20):.4f} {metric_name}={stage1.mean_last(f'stage1_{metric_name}',20):.3f} latent_norm={stage1.mean_last('latent_norm',20):.3f}")
        log_freq = max(1, self.config.train_steps_stage2 // 5)
        for step in range(1, self.config.train_steps_stage2 + 1):
            metrics = self.stage2_step(self.sample_batch())
            stage2.update(**metrics)
            if step % log_freq == 0 or step == 1:
                print(f"[stage2] step={step:4d} loss={stage2.mean_last('stage2_loss',20):.4f} z0_rmse={stage2.mean_last('stage2_rmse',20):.4f}")

        summary = self.evaluate()
        summary["train_group_evals"] = self._group_evaluations()
        summary["artifacts"] = {"plots_train": self._plot_episode_set(self.config.families, "train")}
        if self.config.task_type == "control":
            summary["reward_audit"] = {
                "train": self._control_reward_audit(
                    self.config.families,
                    num_batches=self.config.reward_audit_batches,
                    batch_size=self.config.reward_audit_batch_size,
                ),
            }
        if self.config.eval_families:
            summary["generalization"] = {"eval_families": self.config.eval_families, "eval_summary": self.evaluate(num_batches=self.config.eval_batches, family_names=self.config.eval_families)}
            summary["artifacts"]["plots_eval"] = self._plot_episode_set(self.config.eval_families, "eval")
            if self.config.task_type == "control":
                summary["reward_audit"]["eval"] = self._control_reward_audit(
                    self.config.eval_families,
                    num_batches=self.config.reward_audit_batches,
                    batch_size=self.config.reward_audit_batch_size,
                )
        summary["runtime_seconds"] = time.time() - t0
        summary["config"] = asdict(self.config)
        summary["protocol"] = self.protocol.to_dict()
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        export_latest_paper_results(summary)
        plots_dir = self.output_dir / "plots"
        paper_plots_dir = Path(__file__).resolve().parents[1] / "paper" / "figures" / "plots"
        paper_plots_dir.mkdir(parents=True, exist_ok=True)
        for png in plots_dir.glob("*.png"):
            shutil.copy2(png, paper_plots_dir / png.name)
        stage1.save_json(self.output_dir / "stage1_metrics.json"); stage2.save_json(self.output_dir / "stage2_metrics.json")
        self.save_checkpoint()
        return summary


def export_latest_paper_results(summary):
    root = Path(__file__).resolve().parents[1]
    path = root / "paper" / "results" / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print("[paper] exported results")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperdiffusion toy experiment")
    p.add_argument("--output-dir", type=str, default="runs/default")
    p.add_argument("--task-type", type=str, choices=["classification", "regression", "bandit_regression", "control"], default="classification")
    p.add_argument("--families", type=str, nargs="+", default=None)
    p.add_argument("--expanded-train-families", action="store_true")
    p.add_argument("--eval-families", type=str, nargs="*", default=None)
    p.add_argument("--protocol-suite", type=str, choices=list(PROTOCOL_SUITES), default="held_out")
    p.add_argument(
        "--allow-eval-train-overlap",
        action="store_true",
        help="Disable strict train/eval disjointness checks (not recommended)",
    )
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
    p.add_argument("--encoding-mode", type=str, choices=["support", "text", "hybrid", "oracle"], default="support")
    p.add_argument("--text-embedding-dim", type=int, default=128)
    p.add_argument("--text-mix-alpha", type=float, default=0.5)
    p.add_argument("--reward-audit-batches", type=int, default=8)
    p.add_argument("--reward-audit-batch-size", type=int, default=16)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--attention-layers", type=int, default=3)
    p.add_argument("--support-sweep-batches", type=int, default=16)
    p.add_argument("--selector-enabled", action="store_true", help="Enable learned candidate selector during training/eval")
    p.add_argument("--selector-hidden", type=int, default=128)
    p.add_argument("--selector-lr", type=float, default=1e-3)
    p.add_argument("--selector-num-samples", type=int, default=8)
    return p


def main() -> None:
    args = build_parser().parse_args()
    families = args.families
    eval_families = args.eval_families

    if args.task_type == "classification" and args.expanded_train_families and args.families is None:
        families = list(EXPANDED_TRAIN_FAMILIES)

    strict_ood = not bool(args.allow_eval_train_overlap)

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
        selector_enabled=args.selector_enabled,
        selector_hidden=args.selector_hidden,
        selector_lr=args.selector_lr,
        selector_num_samples=args.selector_num_samples,
        encoding_mode=args.encoding_mode,
        text_embedding_dim=args.text_embedding_dim,
        text_mix_alpha=args.text_mix_alpha,
        reward_audit_batches=args.reward_audit_batches,
        reward_audit_batch_size=args.reward_audit_batch_size,
        device=args.device,
        seed=args.seed,
        protocol_suite=args.protocol_suite,
        strict_ood=strict_ood,
    )
    exp = Experiment(config=config, output_dir=Path(args.output_dir))
    summary = exp.run()
    print("\nFinal summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
