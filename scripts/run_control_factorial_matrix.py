#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperdiffusion.experiment import Experiment, ExperimentConfig
from hyperdiffusion.tasks import DEFAULT_CONTROL_EVAL_FAMILIES, DEFAULT_CONTROL_TRAIN_FAMILIES


def run_one(mode: str, root: Path, args: argparse.Namespace) -> dict:
    out_dir = root / mode
    cfg = ExperimentConfig(
        task_type="control",
        families=args.families or DEFAULT_CONTROL_TRAIN_FAMILIES,
        eval_families=args.eval_families or DEFAULT_CONTROL_EVAL_FAMILIES,
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
        encoding_mode=mode,
        text_embedding_dim=args.text_embedding_dim,
        text_mix_alpha=args.text_mix_alpha,
        reward_audit_batches=args.reward_audit_batches,
        reward_audit_batch_size=args.reward_audit_batch_size,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        support_sweep_batches=args.support_sweep_batches,
        selector_enabled=False,
        selector_hidden=args.selector_hidden,
        selector_lr=args.selector_lr,
        selector_num_samples=args.selector_num_samples,
        device=args.device,
        seed=args.seed,
    )
    exp = Experiment(cfg, out_dir)
    summary = exp.run()
    gen = summary["generalization"]["eval_summary"]["overall"]
    reward_eval = summary.get("reward_audit", {}).get("eval", {}).get("overall", {})
    reward_train = summary.get("reward_audit", {}).get("train", {}).get("overall", {})
    row = {
        "encoding_mode": mode,
        "eval_encoder_r2": gen.get("encoder_r2"),
        "eval_diffusion_r2": gen.get("diffusion_r2"),
        "eval_static_baseline_r2": gen.get("baseline_r2"),
        "eval_gap_encoder_minus_static": (gen.get("encoder_r2") - gen.get("baseline_r2")) if gen.get("encoder_r2") is not None and gen.get("baseline_r2") is not None else None,
        "eval_reward_mean_delta_ls": reward_eval.get("mean_delta_ls"),
        "eval_reward_winrate_vs_static": reward_eval.get("win_rate_vs_static"),
        "train_reward_mean_delta_ls": reward_train.get("mean_delta_ls"),
        "train_reward_winrate_vs_static": reward_train.get("win_rate_vs_static"),
        "output_dir": str(out_dir),
    }
    return {"config": asdict(cfg), "summary": summary, "row": row}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run control encoding/decoder bottleneck matrix")
    p.add_argument("--output-dir", type=str, default="runs/control_matrix")
    p.add_argument("--modes", type=str, nargs="+", default=["support", "text", "hybrid", "oracle"])
    p.add_argument("--families", type=str, nargs="*", default=None)
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
    p.add_argument("--support-size-sweep", type=int, nargs="*", default=[2, 4, 8, 16])
    p.add_argument("--visualization-count", type=int, default=0)
    p.add_argument("--visualization-grid-size", type=int, default=120)
    p.add_argument("--encoder-type", type=str, choices=["attention", "deepset"], default="attention")
    p.add_argument("--text-embedding-dim", type=int, default=768)
    p.add_argument("--text-mix-alpha", type=float, default=0.5)
    p.add_argument("--reward-audit-batches", type=int, default=8)
    p.add_argument("--reward-audit-batch-size", type=int, default=16)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--attention-layers", type=int, default=3)
    p.add_argument("--support-sweep-batches", type=int, default=16)
    p.add_argument("--selector-hidden", type=int, default=128)
    p.add_argument("--selector-lr", type=float, default=1e-3)
    p.add_argument("--selector-num-samples", type=int, default=8)
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    modes = args.modes
    results = {}
    table = []
    for mode in modes:
        if mode not in {"support", "text", "hybrid", "oracle"}:
            raise ValueError(f"Unsupported mode: {mode}")
        print(f"[matrix] running mode={mode}")
        mode_result = run_one(mode, root, args)
        results[mode] = mode_result
        table.append(mode_result["row"])

    payload = {
        "modes": modes,
        "table": table,
        "results": results,
    }
    out = root / "matrix_summary.json"
    out.write_text(json.dumps(payload, indent=2))
    print("\n[matrix] summary table")
    for row in table:
        print(
            f"- {row['encoding_mode']}: "
            f"eval_encoder_r2={row['eval_encoder_r2']:.4f}, "
            f"eval_static_r2={row['eval_static_baseline_r2']:.4f}, "
            f"eval_reward_delta_ls={row['eval_reward_mean_delta_ls']:.4f}, "
            f"eval_winrate_vs_static={row['eval_reward_winrate_vs_static']:.3f}"
        )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
