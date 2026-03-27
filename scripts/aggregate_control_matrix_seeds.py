#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = [
    "eval_encoder_r2",
    "eval_diffusion_r2",
    "eval_static_baseline_r2",
    "eval_reward_mean_delta_ls",
    "eval_reward_winrate_vs_static",
    "train_reward_mean_delta_ls",
    "train_reward_winrate_vs_static",
]


def mean_std(vals):
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals), "n": len(vals)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate multiseed control matrix runs")
    p.add_argument("--input-root", type=str, default="runs/control_matrix_v2_multiseed")
    p.add_argument("--output", type=str, default="runs/control_matrix_v2_multiseed/aggregate.json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.input_root)
    seed_dirs = sorted([d for d in root.glob("seed_*") if d.is_dir()])
    if not seed_dirs:
        raise SystemExit(f"No seed directories found under {root}")

    per_seed = {}
    by_mode = {}
    modes = ["support", "text", "hybrid", "oracle"]
    for m in modes:
        by_mode[m] = {k: [] for k in METRICS}

    for sd in seed_dirs:
        seed = sd.name.replace("seed_", "")
        p = sd / "matrix_summary.json"
        data = json.loads(p.read_text())
        rows = data.get("table", [])
        row_map = {r["encoding_mode"]: r for r in rows}
        per_seed[seed] = row_map
        for mode in modes:
            r = row_map.get(mode, {})
            for k in METRICS:
                v = r.get(k)
                if v is not None:
                    by_mode[mode][k].append(v)

    summary_table = []
    for mode in modes:
        row = {"encoding_mode": mode}
        for k in METRICS:
            stats = mean_std(by_mode[mode][k])
            row[k + "_mean"] = stats["mean"]
            row[k + "_std"] = stats["std"]
            row[k + "_n"] = stats["n"]
        summary_table.append(row)

    payload = {
        "input_root": str(root),
        "num_seeds": len(seed_dirs),
        "seeds": [d.name.replace("seed_", "") for d in seed_dirs],
        "metrics": METRICS,
        "summary_table": summary_table,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
