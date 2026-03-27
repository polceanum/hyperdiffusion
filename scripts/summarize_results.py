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


# ──────────────────────────────────────────────────────────
# section 1 – per-task benchmark
# ──────────────────────────────────────────────────────────

def print_task_benchmark():
    rows = read_v2_results()
    print("=" * 68)
    print("  SECTION 1 — Per-Task Benchmark  (runs/*_v2, encoding_mode=support)")
    print("=" * 68)
    print(f"  {'Task':<18} {'Metric':<6}  {'Encoder':>9}  {'Diffusion':>9}  {'Baseline':>9}  {'Enc-Base':>9}")
    print("  " + "-" * 64)
    for task, metric, enc, diff, base in rows:
        gap = (enc - base) if (enc is not None and base is not None) else None
        print(
            f"  {task:<18} {metric:<6}  "
            f"{fmt(enc):>9}  {fmt(diff):>9}  {fmt(base):>9}  {fmt(gap):>9}"
        )
    print()
    print("  Encoder = attention meta-learner over support set.")
    print("  Diffusion = DDIM sampler conditioned on support context (main model).")
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


# ──────────────────────────────────────────────────────────
# section 3 – key conclusions
# ──────────────────────────────────────────────────────────

def print_conclusions():
    print("=" * 68)
    print("  SECTION 3 — Key Conclusions")
    print("=" * 68)

    conclusions = [
        ("C1", "Meta-learning beats task-agnostic baseline across all tasks",
         "Encoder and diffusion consistently outperform the static MLP on "
         "classification (acc), regression, bandit, and control (R²)."),

        ("C2", "Diffusion ≈ Encoder in R²",
         "Diffusion model matches encoder quality on most tasks, showing DDIM "
         "sampling of the latent space captures the full task-posterior rather "
         "than just the mean."),

        ("C3", "Text mode improves semantics, not precision",
         "Text encoding tends to lag support mode on parameter-reconstruction R², "
         "but can still provide useful behavioural priors for control transfer. "
         "Its reward impact is protocol-sensitive and should be validated per run."),

        ("C4", "Support mode wins on parameter reconstruction",
         "Attention over (x,y) pairs produces better R² than text or hybrid, "
         "because it has direct access to the task's actual data rather than "
         "a fixed natural-language description."),

        ("C5", "Oracle (one-hot family index) underperforms",
         "Oracle mode is consistently weak on reward transfer, indicating that "
         "memorising a family label alone is insufficient — the model needs "
         "either actual demonstrations or semantic task priors to generalise."),

        ("C6", "Hybrid mode does not improve over support alone",
         "Mixing text and support with α=0.5 hurts reward performance compared to "
         "pure support mode, suggesting the text signal introduces noise into an "
         "otherwise well-tuned support representation."),
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

    print_conclusions()


if __name__ == "__main__":
    main()
