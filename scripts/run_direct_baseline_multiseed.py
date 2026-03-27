#!/usr/bin/env python3
"""Run direct baseline (non-fast-weights) ablation across task types, modes, and seeds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_direct_baseline(args: argparse.Namespace) -> None:
    """Run direct baseline experiment."""
    
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    modes = args.modes or ["support"]
    seeds = args.seeds or [0, 1, 2]
    task_types = args.task_types or ["classification", "regression", "bandit_regression", "control"]
    
    print("Running direct baseline experiments")
    print(f"  Task types: {task_types}")
    print(f"  Modes: {modes}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {output_root}")
    print()
    
    # For each task/mode/seed combination, run the experiment
    for task_type in task_types:
        task_root = output_root / task_type
        task_root.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            for seed in seeds:
                output_dir = task_root / f"direct_{mode}_seed{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    args.python,
                    "-m", "hyperdiffusion.direct_experiment",
                    "--task-type", task_type,
                    "--output-dir", str(output_dir),
                    "--encoding-mode", mode,
                    "--seed", str(seed),
                    "--train-steps-stage1", str(args.train_steps_stage1),
                    "--train-steps-stage2", str(args.train_steps_stage2),
                    "--eval-batches", str(args.eval_batches),
                    "--batch-size", str(args.batch_size),
                    "--support-sweep-batches", str(args.support_sweep_batches),
                    "--visualization-count", str(args.visualization_count),
                    "--reward-audit-batches", str(args.reward_audit_batches),
                    "--reward-audit-batch-size", str(args.reward_audit_batch_size),
                ]

                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"ERROR: Experiment failed for task={task_type}, mode={mode}, seed={seed}")
                    sys.exit(1)
    
    print("\n[DONE] All direct baseline experiments completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run direct baseline ablation experiments")
    parser.add_argument("--output-root", type=str, default="runs/direct_baseline_v2_multiseed", help="Root output directory")
    parser.add_argument("--task-types", nargs="+", default=None, choices=["classification", "regression", "bandit_regression", "control"], help="Task types to test")
    parser.add_argument("--modes", nargs="+", default=None, help="Encoding modes to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds")
    parser.add_argument("--python", type=str, default="python", help="Python executable")
    parser.add_argument("--train-steps-stage1", type=int, default=1000)
    parser.add_argument("--train-steps-stage2", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-sweep-batches", type=int, default=8)
    parser.add_argument("--visualization-count", type=int, default=0)
    parser.add_argument("--reward-audit-batches", type=int, default=6)
    parser.add_argument("--reward-audit-batch-size", type=int, default=16)
    
    args = parser.parse_args()
    run_direct_baseline(args)
