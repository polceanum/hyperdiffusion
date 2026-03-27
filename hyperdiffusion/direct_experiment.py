"""Direct baseline experiment: end-to-end prediction without hypernetworks."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .direct_baseline import DirectSystem
from .experiment import ExperimentConfig, MetricsTracker
from .tasks import (
    DEFAULT_BANDIT_EVAL_FAMILIES,
    DEFAULT_BANDIT_TRAIN_FAMILIES,
    DEFAULT_CONTROL_EVAL_FAMILIES,
    DEFAULT_CONTROL_TRAIN_FAMILIES,
    DEFAULT_REGRESSION_EVAL_FAMILIES,
    DEFAULT_REGRESSION_TRAIN_FAMILIES,
    DEFAULT_TRAIN_FAMILIES,
    EpisodeBatch,
    make_episode_batch,
)


class DirectExperiment:
    """Direct baseline experiment for control task."""

    def __init__(self, config: ExperimentConfig, output_dir: Path, device: str = "cpu"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Set seed
        torch.manual_seed(config.seed)

        if config.task_type == "classification":
            input_dim = 2
        elif config.task_type == "regression":
            input_dim = 1
        elif config.task_type in ("bandit_regression", "control"):
            input_dim = 2
        else:
            raise ValueError(f"Unsupported task_type: {config.task_type}")

        self.input_dim = input_dim

        # Initialize model
        self.system = DirectSystem(
            x_dim=input_dim,
            y_dim=1,
            encoder_hidden=config.encoder_hidden,
            cond_dim=config.cond_dim,
            attention_heads=config.attention_heads,
            attention_layers=config.attention_layers,
            text_dim=config.text_embedding_dim,
            text_mix_alpha=config.text_mix_alpha,
            num_families=len(config.families or []),
        ).to(device)

        # Static baseline model (for comparison)
        self.baseline = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config.encoder_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(config.encoder_hidden, config.encoder_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(config.encoder_hidden, 1),
        ).to(device)

        # Optimizers
        self.opt_main = torch.optim.AdamW(self.system.parameters(), lr=config.learning_rate_stage1)
        self.opt_baseline = torch.optim.AdamW(self.baseline.parameters(), lr=config.learning_rate_stage1)

        # Metrics
        self.metrics = MetricsTracker()

        # Family to index mapping for oracle
        families = config.families or self._default_train_families()
        self.family_to_index = {name: idx for idx, name in enumerate(families)}

    def _default_train_families(self) -> List[str]:
        if self.config.task_type == "classification":
            return DEFAULT_TRAIN_FAMILIES
        if self.config.task_type == "regression":
            return DEFAULT_REGRESSION_TRAIN_FAMILIES
        if self.config.task_type == "bandit_regression":
            return DEFAULT_BANDIT_TRAIN_FAMILIES
        if self.config.task_type == "control":
            return DEFAULT_CONTROL_TRAIN_FAMILIES
        raise ValueError(f"Unsupported task_type: {self.config.task_type}")

    def _default_eval_families(self) -> List[str]:
        if self.config.task_type == "classification":
            return self._default_train_families()
        if self.config.task_type == "regression":
            return DEFAULT_REGRESSION_EVAL_FAMILIES
        if self.config.task_type == "bandit_regression":
            return DEFAULT_BANDIT_EVAL_FAMILIES
        if self.config.task_type == "control":
            return DEFAULT_CONTROL_EVAL_FAMILIES
        raise ValueError(f"Unsupported task_type: {self.config.task_type}")

    def _metric_name(self) -> str:
        return "acc" if self.config.task_type == "classification" else "r2"

    def _text_embedding(self, family_names: List[str]) -> torch.Tensor:
        """Get text embeddings for families (using DistilBERT-like mock)."""
        # Simplified: return random embeddings (would be DistilBERT in full implementation)
        batch_size = len(family_names)
        return torch.randn(batch_size, self.config.text_embedding_dim, device=self.device)

    def _encode_batch(
        self, support_x: torch.Tensor, support_y: torch.Tensor, family_names: List[str]
    ) -> torch.Tensor:
        """Encode support batch to context based on encoding mode."""
        mode = self.config.encoding_mode

        if mode == "support":
            return self.system.encode_support(support_x, support_y)

        if mode == "oracle":
            indices = torch.tensor(
                [self.family_to_index.get(name, 0) for name in family_names],
                device=self.device,
                dtype=torch.long,
            )
            one_hot = F.one_hot(indices, num_classes=max(len(self.family_to_index), 1)).float()
            target_dim = self.config.cond_dim
            if one_hot.shape[-1] < target_dim:
                one_hot = F.pad(one_hot, (0, target_dim - one_hot.shape[-1]))
            return one_hot[:, :target_dim]

        text_embed = self._text_embedding(family_names)
        context_text = self.system.encode_text(text_embed)

        if mode == "text":
            return context_text

        if mode == "hybrid":
            context_support = self.system.encode_support(support_x, support_y)
            alpha = float(self.config.text_mix_alpha)
            return (1.0 - alpha) * context_support + alpha * context_text

        raise ValueError(f"Unknown encoding_mode: {mode}")

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
        """Task loss (MSE for control)."""
        if self.config.task_type == "classification":
            return F.binary_cross_entropy_with_logits(pred, target)
        return F.mse_loss(pred, target)

    def _metric(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """R² metric for regression/control."""
        if self.config.task_type == "classification":
            return ((pred > 0.0) == (target > 0.5)).float().mean(dim=(1, 2))
        sse = ((pred - target) ** 2).sum(dim=(1, 2))
        mean_target = target.mean(dim=(1, 2), keepdim=True)
        sst = ((target - mean_target) ** 2).sum(dim=(1, 2)).clamp_min(1e-6)
        return 1.0 - sse / sst

    def train_step(self, batch: EpisodeBatch) -> Dict[str, float]:
        """Single training step."""
        self.system.train()
        batch = self.to_device(batch)

        # Direct prediction
        context = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
        pred = self.system(batch.support_x, batch.support_y, batch.query_x, context=context)
        loss = self._task_loss(pred, batch.query_y)

        self.opt_main.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.config.grad_clip)
        self.opt_main.step()

        metric = self._metric(pred, batch.query_y).mean().item()

        # Static baseline
        self.baseline.train()
        baseline_x = batch.support_x.view(-1, batch.support_x.shape[-1])
        baseline_y = batch.support_y.view(-1, batch.support_y.shape[-1])
        baseline_pred = self.baseline(baseline_x)
        baseline_loss = self._task_loss(baseline_pred, baseline_y)
        self.opt_baseline.zero_grad(set_to_none=True)
        baseline_loss.backward()
        self.opt_baseline.step()

        with torch.no_grad():
            q_baseline_pred = self.baseline(batch.query_x.view(-1, batch.query_x.shape[-1])).view(
                batch.query_x.shape[0], batch.query_x.shape[1], -1
            )
            baseline_metric = self._metric(q_baseline_pred, batch.query_y).mean().item()

        return {
            "loss": loss.item(),
            "metric": metric,
            "baseline_loss": baseline_loss.item(),
            "baseline_metric": baseline_metric,
        }

    @torch.no_grad()
    def evaluate(self, families: Optional[List[str]] = None) -> Dict[str, any]:
        """Evaluate on held-out families."""
        self.system.eval()
        self.baseline.eval()

        families = families or (self.config.eval_families or self._default_eval_families())

        direct_metrics, baseline_metrics = [], []

        for _ in range(self.config.eval_batches):
            batch = self.to_device(make_episode_batch(
                batch_size=self.config.batch_size,
                support_size=self.config.support_size,
                query_size=self.config.query_size,
                family_names=families,
                task_type=self.config.task_type,
            ))

            context = self._encode_batch(batch.support_x, batch.support_y, batch.family_name)
            pred = self.system(batch.support_x, batch.support_y, batch.query_x, context=context)
            direct_metrics.extend(self._metric(pred, batch.query_y).tolist())

            b, q, _ = batch.query_x.shape
            baseline_pred = self.baseline(batch.query_x.view(b * q, -1)).view(b, q, -1)
            baseline_metrics.extend(self._metric(baseline_pred, batch.query_y).tolist())

        metric_name = self._metric_name()
        return {
            f"direct_{metric_name}": sum(direct_metrics) / len(direct_metrics) if direct_metrics else 0.0,
            f"baseline_{metric_name}": sum(baseline_metrics) / len(baseline_metrics) if baseline_metrics else 0.0,
            "num_eval_episodes": len(direct_metrics),
        }

    def run(self) -> None:
        """Run full training."""
        print(f"[DirectExperiment] Starting training")
        print(f"  Task: {self.config.task_type}, Mode: {self.config.encoding_mode}")
        print(f"  Seed: {self.config.seed}, Output: {self.output_dir}")

        families = self.config.families or self._default_train_families()

        # Training loop
        for step in range(self.config.train_steps_stage1):
            batch = self.to_device(make_episode_batch(
                batch_size=self.config.batch_size,
                support_size=self.config.support_size,
                query_size=self.config.query_size,
                family_names=families,
                task_type=self.config.task_type,
            ))
            stats = self.train_step(batch)
            self.metrics.update(**stats)

            if (step + 1) % 100 == 0:
                print(f"  Step {step + 1}: loss={self.metrics.mean_last('loss'):.4f}, "
                      f"metric={self.metrics.mean_last('metric'):.4f}")

        print(f"[DirectExperiment] Training complete, evaluating...")

        # Evaluation
        eval_result = self.evaluate()

        # Save results
        metric_name = self._metric_name()
        summary = {
            "config": asdict(self.config),
            "overall": {
                f"direct_{metric_name}": eval_result[f"direct_{metric_name}"],
                f"baseline_{metric_name}": eval_result[f"baseline_{metric_name}"],
            },
            "eval_batches": self.config.eval_batches,
            "eval_families": self.config.eval_families or self._default_eval_families(),
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[DirectExperiment] Results saved to {summary_path}")
        if metric_name == "acc":
            print(f"  Direct Acc: {eval_result['direct_acc']:.4f}")
            print(f"  Baseline Acc: {eval_result['baseline_acc']:.4f}")
        else:
            print(f"  Direct R²: {eval_result['direct_r2']:.4f}")
            print(f"  Baseline R²: {eval_result['baseline_r2']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Direct baseline experiment")
    parser.add_argument("--task-type", type=str, default="control", choices=["classification", "regression", "bandit_regression", "control"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--encoding-mode", type=str, default="support", choices=["support", "text", "hybrid", "oracle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps-stage1", type=int, default=1000)
    parser.add_argument("--train-steps-stage2", type=int, default=1000)  # Not used for direct
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-sweep-batches", type=int, default=8)  # accepted for wrapper compatibility
    parser.add_argument("--visualization-count", type=int, default=0)  # accepted for wrapper compatibility
    parser.add_argument("--reward-audit-batches", type=int, default=6)  # accepted for wrapper compatibility
    parser.add_argument("--reward-audit-batch-size", type=int, default=16)  # accepted for wrapper compatibility
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.task_type == "classification":
        families = DEFAULT_TRAIN_FAMILIES
        eval_families = None
    elif args.task_type == "regression":
        families = DEFAULT_REGRESSION_TRAIN_FAMILIES
        eval_families = DEFAULT_REGRESSION_EVAL_FAMILIES
    elif args.task_type == "bandit_regression":
        families = DEFAULT_BANDIT_TRAIN_FAMILIES
        eval_families = DEFAULT_BANDIT_EVAL_FAMILIES
    else:
        families = DEFAULT_CONTROL_TRAIN_FAMILIES
        eval_families = DEFAULT_CONTROL_EVAL_FAMILIES

    config = ExperimentConfig(
        task_type=args.task_type,
        families=families,
        eval_families=eval_families,
        encoding_mode=args.encoding_mode,
        seed=args.seed,
        train_steps_stage1=args.train_steps_stage1,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        device=args.device,
    )

    experiment = DirectExperiment(config, Path(args.output_dir), device=args.device)
    experiment.run()


if __name__ == "__main__":
    main()
