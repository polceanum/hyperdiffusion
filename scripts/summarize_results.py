"""
Consolidated results summary for HyperDiffusion experiments.

Reads:
  - runs/{task}_v2/summary.json      : per-task benchmarks (classification, regression, bandit, control)
  - runs/control_matrix_main/        : encoding-mode ablation, hashed BOW text (300 steps)
  - runs/control_matrix_v2/          : encoding-mode ablation, semantic distilbert text (1000 steps)

Usage:
  python scripts/summarize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path


# ──────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────

def fmt(v, pct=False, digits=3):
    if v is None:
        return "  n/a  "
    if pct:
        return f"{v * 100:6.1f}%"
    return f"{v:{6}.{digits}f}"


def read_v2_results():
    """Return list of (task, metric_name, enc, diff, base) from *_v2 summary.json files."""
    rows = []
    for task in ["classification", "regression", "bandit", "control"]:
        p = Path(f"runs/{task}_v2/summary.json")
        if not p.exists():
            rows.append((task, "?", None, None, None))
            continue
        data = json.loads(p.read_text())
        overall = data.get("overall", {})
        if not overall:
            # try nested path used in older runs
            overall = data.get("generalization", {}).get("eval_summary", {}).get("overall", {})
        metric = "acc" if "encoder_acc" in overall else "r2"
        enc = overall.get(f"encoder_{metric}")
        diff = overall.get(f"diffusion_{metric}", overall.get(f"diffusion_{metric}_mean"))
        base = overall.get(f"baseline_{metric}", overall.get(f"static_baseline_{metric}"))
        rows.append((task, metric, enc, diff, base))
    return rows


def read_matrix_results(matrix_dir: str):
    """Return list of row dicts from matrix_summary.json, or None if not found."""
    p = Path(matrix_dir) / "matrix_summary.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("table", [])


def read_multiseed_aggregate(path: str = "runs/control_matrix_v2_multiseed/aggregate.json"):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ──────────────────────────────────────────────────────────
# section 1 – per-task benchmark
# ──────────────────────────────────────────────────────────

def print_task_benchmark():
    rows = read_v2_results()
    print("=" * 68)
    print("  SECTION 1 — Per-Task Benchmark  (runs/*_v2, encoding_mode=support)")
    print("=" * 68)
    print(f"  {'Task':<18} {'Metric':<6}  {'FW(det.)':>9}  {'FW(diff.)':>9}  {'Baseline':>9}  {'FW-Base':>9}")
    print("  " + "-" * 64)
    for task, metric, enc, diff, base in rows:
        gap = (enc - base) if (enc is not None and base is not None) else None
        print(
            f"  {task:<18} {metric:<6}  "
            f"{fmt(enc):>9}  {fmt(diff):>9}  {fmt(base):>9}  {fmt(gap):>9}"
        )
    print()
    print("  FW (det.) = fast weights via deterministic attention over support set (no sampling).")
    print("  FW (diff.) = fast weights via DDIM sampling of latent space (main model).")
    print("  Baseline = static MLP, no support set (task-agnostic lower bound).")
    print()


# ──────────────────────────────────────────────────────────
# section 2 – encoding-mode ablation
# ──────────────────────────────────────────────────────────

MODE_LEGEND = {
    "support": "Attention encoder over (x,y) pairs",
    "text":    "Family description → distilbert → projector",
    "hybrid":  "Weighted mix of support + text (α=0.5)",
    "oracle":  "One-hot family index (upper bound)",
}


def print_matrix_section(title, matrix_dir, note=None):
    rows = read_matrix_results(matrix_dir)
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)
    if rows is None:
        print(f"  [not found: {matrix_dir}/matrix_summary.json]")
        print()
        return

    header = f"  {'Mode':<10}  {'Enc R²':>8}  {'Diff R²':>8}  {'Base R²':>8}  {'Rew WinRate':>12}  {'ΔReward':>9}"
    print(header)
    print("  " + "-" * 64)
    for row in rows:
        mode = row.get("encoding_mode", "?")
        enc_r2  = row.get("eval_encoder_r2")
        diff_r2 = row.get("eval_diffusion_r2")
        base_r2 = row.get("eval_static_baseline_r2")
        winrate = row.get("eval_reward_winrate_vs_static")
        delta   = row.get("eval_reward_mean_delta_ls")
        print(
            f"  {mode:<10}  {fmt(enc_r2):>8}  {fmt(diff_r2):>8}  {fmt(base_r2):>8}  "
            f"{fmt(winrate, pct=True):>12}  {fmt(delta):>9}"
        )

    if note:
        print()
        print(f"  Note: {note}")

    print()
    print("  Modes:")
    for m, desc in MODE_LEGEND.items():
        print(f"    {m:<8} — {desc}")
    print()


def print_multiseed_section():
    data = read_multiseed_aggregate()
    print("=" * 68)
    print("  SECTION 2c — Encoding-Mode Ablation  [DistilBERT, 3 seeds]")
    print("=" * 68)
    if data is None:
        print("  [not found: runs/control_matrix_v2_multiseed/aggregate.json]")
        print()
        return

    rows = data.get("summary_table", [])
    n = data.get("num_seeds", "?")
    print(f"  Aggregated over {n} seeds (mean ± std)")
    print(f"  {'Mode':<10}  {'Enc R²':>18}  {'Diff R²':>18}  {'Base R²':>18}  {'Rew WinRate':>18}  {'ΔReward':>16}")
    print("  " + "-" * 130)
    for row in rows:
        mode = row.get("encoding_mode", "?")
        enc_m, enc_s = row.get("eval_encoder_r2_mean"), row.get("eval_encoder_r2_std")
        diff_m, diff_s = row.get("eval_diffusion_r2_mean"), row.get("eval_diffusion_r2_std")
        base_m, base_s = row.get("eval_static_baseline_r2_mean"), row.get("eval_static_baseline_r2_std")
        win_m, win_s = row.get("eval_reward_winrate_vs_static_mean"), row.get("eval_reward_winrate_vs_static_std")
        d_m, d_s = row.get("eval_reward_mean_delta_ls_mean"), row.get("eval_reward_mean_delta_ls_std")
        print(
            f"  {mode:<10}  "
            f"{f'{enc_m:.3f}±{enc_s:.3f}' if enc_m is not None else 'n/a':>18}  "
            f"{f'{diff_m:.3f}±{diff_s:.3f}' if diff_m is not None else 'n/a':>18}  "
            f"{f'{base_m:.3f}±{base_s:.3f}' if base_m is not None else 'n/a':>18}  "
            f"{f'{100*win_m:.1f}%±{100*win_s:.1f}%' if win_m is not None else 'n/a':>18}  "
            f"{f'{d_m:.2f}±{d_s:.2f}' if d_m is not None else 'n/a':>16}"
        )
    print()


# ──────────────────────────────────────────────────────────
# section 3 – key conclusions
# ──────────────────────────────────────────────────────────

def print_conclusions():
    print("=" * 68)
    print("  SECTION 3 — Key Conclusions")
    print("=" * 68)

    conclusions = [
        ("C1", "Meta-learning beats task-agnostic baseline across all tasks",
         "FW (det.) and FW (diff.) consistently outperform the static MLP on "
         "classification (acc), regression, bandit, and control (R²)."),

        ("C2", "FW (diff.) ≈ FW (det.) in R²",
         "FW (diff.) matches FW (det.) quality on most tasks, showing DDIM "
         "sampling of the latent space captures the full task-posterior rather "
         "than just the mean."),

        ("C3", "Text mode improves semantics, not precision",
         "Text encoding lags support/hybrid on reconstruction R² but remains "
         "competitive on reward transfer with high seed-to-seed variance. "
         "Its behavioural impact is real but less stable."),

        ("C4", "Support mode wins on parameter reconstruction",
         "Attention over (x,y) pairs produces better R² than text or hybrid, "
         "because it has direct access to the task's actual data rather than "
         "a fixed natural-language description."),

        ("C5", "Oracle (one-hot family index) underperforms",
         "Oracle mode is consistently weak on reward transfer, indicating that "
         "memorising a family label alone is insufficient — the model needs "
         "either actual demonstrations or semantic task priors to generalise."),

        ("C6", "Hybrid often improves reward robustness",
         "Across multiseed runs, hybrid has the strongest average reward win-rate, "
         "suggesting that blending semantic priors with support evidence can "
         "improve robustness even when support-only wins on R²."),
    ]

    for code, title, detail in conclusions:
        print(f"\n  [{code}] {title}")
        # wrap detail at ~60 chars
        words = detail.split()
        line, out = "       ", []
        for w in words:
            if len(line) + len(w) + 1 > 68:
                out.append(line)
                line = "       " + w
            else:
                line += (" " if line.strip() else "") + w
        out.append(line)
        print("\n".join(out))

    print()


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────

def main():
    print()
    print_task_benchmark()

    print_matrix_section(
        "SECTION 2a — Encoding-Mode Ablation  [hashed BOW text, 300 steps]",
        "runs/control_matrix_main",
        note="Text embedding was MD5 hash bag-of-words (not semantic) in this run.",
    )

    print_matrix_section(
        "SECTION 2b — Encoding-Mode Ablation  [distilbert text, 1000 steps]",
        "runs/control_matrix_v2",
        note="Text embedding uses distilbert-base-uncased CLS token (768-dim). "
             "If this section is missing the run is still in progress.",
    )

    print_multiseed_section()

    print_conclusions()


if __name__ == "__main__":
    main()
