#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .validate import validate_experiment_outputs, validate_paper_artifacts, write_provenance_manifest
from .workflows import (
    build_paper,
    clean_paper_artifacts,
    full_refresh,
    refresh_reports_and_plots,
    run_v2_benchmark,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified modular pipeline for HyperDiffusion")
    parser.add_argument(
        "command",
        choices=["full-refresh", "benchmark", "refresh-artifacts", "build-paper", "clean-paper", "validate"],
    )
    parser.add_argument("--python", dest="python_exec", default=None, help="Python executable to use")
    parser.add_argument("--train-steps-stage1", type=int, default=1000)
    parser.add_argument("--train-steps-stage2", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-sweep-batches", type=int, default=8)
    parser.add_argument("--diagnostic-samples", type=int, default=4)
    parser.add_argument("--visualization-count", type=int, default=4)
    parser.add_argument("--reward-audit-batches", type=int, default=6)
    parser.add_argument("--reward-audit-batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    base = PipelineConfig.from_root(_project_root(), python_exec=args.python_exec)
    return PipelineConfig(
        root=base.root,
        python_exec=base.python_exec,
        train_steps_stage1=args.train_steps_stage1,
        train_steps_stage2=args.train_steps_stage2,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        support_sweep_batches=args.support_sweep_batches,
        diagnostic_samples=args.diagnostic_samples,
        visualization_count=args.visualization_count,
        reward_audit_batches=args.reward_audit_batches,
        reward_audit_batch_size=args.reward_audit_batch_size,
        seeds=tuple(args.seeds),
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)

    if args.command == "full-refresh":
        full_refresh(cfg)
    elif args.command == "benchmark":
        run_v2_benchmark(cfg)
    elif args.command == "refresh-artifacts":
        refresh_reports_and_plots(cfg)
    elif args.command == "build-paper":
        build_paper(cfg)
    elif args.command == "clean-paper":
        clean_paper_artifacts(cfg)
    elif args.command == "validate":
        exp_validation = validate_experiment_outputs(cfg)
        write_provenance_manifest(cfg, stage="validate-experiments", validation=exp_validation)
        paper_validation = validate_paper_artifacts(cfg)
        write_provenance_manifest(cfg, stage="validate-paper-artifacts", validation=paper_validation)


if __name__ == "__main__":
    main()
