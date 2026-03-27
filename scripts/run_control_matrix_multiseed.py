#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run control encoding-mode matrix for multiple seeds")
    p.add_argument("--output-root", type=str, default="runs/control_matrix_v2_multiseed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--train-steps-stage1", type=int, default=1000)
    p.add_argument("--train-steps-stage2", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--support-sweep-batches", type=int, default=8)
    p.add_argument("--diagnostic-samples", type=int, default=4)
    p.add_argument("--visualization-count", type=int, default=0)
    p.add_argument("--reward-audit-batches", type=int, default=6)
    p.add_argument("--reward-audit-batch-size", type=int, default=16)
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    run_summaries = {}
    for seed in args.seeds:
        out_dir = root / f"seed_{seed}"
        cmd = [
            args.python,
            "scripts/run_control_factorial_matrix.py",
            "--output-dir", str(out_dir),
            "--modes", "support", "text", "hybrid", "oracle",
            "--train-steps-stage1", str(args.train_steps_stage1),
            "--train-steps-stage2", str(args.train_steps_stage2),
            "--eval-batches", str(args.eval_batches),
            "--batch-size", str(args.batch_size),
            "--support-sweep-batches", str(args.support_sweep_batches),
            "--diagnostic-samples", str(args.diagnostic_samples),
            "--visualization-count", str(args.visualization_count),
            "--reward-audit-batches", str(args.reward_audit_batches),
            "--reward-audit-batch-size", str(args.reward_audit_batch_size),
            "--text-embedding-dim", "768",
            "--seed", str(seed),
        ]
        print(f"[multiseed] running seed={seed}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        summary_path = out_dir / "matrix_summary.json"
        run_summaries[str(seed)] = str(summary_path)

    (root / "runs_index.json").write_text(json.dumps({"seeds": args.seeds, "runs": run_summaries}, indent=2))
    print(f"[multiseed] completed seeds={args.seeds} at {root}")


if __name__ == "__main__":
    main()
