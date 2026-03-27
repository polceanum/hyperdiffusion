import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = root.parent / "runs"
plots_dir = root / "figures" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
data = json.loads((root / "results/latest.json").read_text())


# ──────────────────────────────────────────────────────────
# NEW: Per-task benchmark bar chart
# ──────────────────────────────────────────────────────────

TASKS = ["classification", "regression", "bandit", "control"]
TASK_LABELS = ["Classification\n(Acc.)", "Regression\n(R²)", "Bandit\n(R²)", "Control\n(R²)"]

enc_vals, diff_vals, base_vals = [], [], []
for task in TASKS:
    p = runs / f"{task}_v2" / "summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        ov = d.get("overall", {})
        is_acc = "encoder_acc" in ov
        enc_vals.append(ov.get("encoder_acc" if is_acc else "encoder_r2"))
        diff_vals.append(ov.get("diffusion_acc" if is_acc else "diffusion_r2"))
        base_vals.append(ov.get("baseline_acc" if is_acc else "baseline_r2"))
    else:
        enc_vals.append(None); diff_vals.append(None); base_vals.append(None)

if any(v is not None for v in enc_vals):
    x = np.arange(len(TASKS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars_enc  = ax.bar(x - w,   [v or 0 for v in enc_vals],  w, label="Encoder",   color="#4C72B0")
    bars_diff = ax.bar(x,       [v or 0 for v in diff_vals], w, label="Diffusion",  color="#DD8452")
    bars_base = ax.bar(x + w,   [v or 0 for v in base_vals], w, label="Baseline",   color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=8)
    ax.set_ylabel("Performance", fontsize=8)
    ax.set_title("Per-Task Benchmark (OOD Eval Families)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    ax.set_ylim(bottom=min(0, min(v for v in base_vals if v is not None) - 0.1))
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(plots_dir / "task_benchmark.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[gen_plots] wrote task_benchmark.png")


# ──────────────────────────────────────────────────────────
# NEW: Encoding-mode ablation chart (R² + reward win-rate)
# ──────────────────────────────────────────────────────────

def load_matrix(dirname):
    p = runs / dirname / "matrix_summary.json"
    if p.exists():
        return json.loads(p.read_text()).get("table", [])
    return None

matrix_rows = load_matrix("control_matrix_v2") or load_matrix("control_matrix_main")

if matrix_rows:
    modes        = [r["encoding_mode"] for r in matrix_rows]
    mode_labels  = [m.capitalize() for m in modes]
    enc_r2s      = [r.get("eval_encoder_r2",  0) for r in matrix_rows]
    diff_r2s     = [r.get("eval_diffusion_r2", 0) for r in matrix_rows]
    base_r2s     = [r.get("eval_static_baseline_r2", 0) for r in matrix_rows]
    winrates     = [r.get("eval_reward_winrate_vs_static", 0) for r in matrix_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Left: R² grouped bars
    x = np.arange(len(modes))
    w = 0.26
    ax1.bar(x - w,  enc_r2s,  w, label="Encoder",   color="#4C72B0")
    ax1.bar(x,      diff_r2s, w, label="Diffusion",  color="#DD8452")
    ax1.bar(x + w,  base_r2s, w, label="Baseline",   color="#55A868")
    ax1.set_xticks(x); ax1.set_xticklabels(mode_labels, fontsize=8)
    ax1.set_ylabel("R² (parameter reconstruction)", fontsize=8)
    ax1.set_title("R² by Encoding Mode", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)
    ax1.set_ylim(-0.3, 1.05)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Right: reward win-rate bars
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax2.bar(mode_labels, winrates, color=colors[:len(modes)])
    ax2.set_ylabel("Reward Win-Rate vs. Baseline", fontsize=8)
    ax2.set_title("OOD Reward Win-Rate by Mode", fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    for i, v in enumerate(winrates):
        ax2.text(i, v + 0.03, f"{v*100:.0f}%", ha="center", fontsize=8, fontweight="bold")
    ax2.tick_params(labelsize=7)

    fig.suptitle("Encoding-Mode Ablation (Control Tasks, OOD Eval)", fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(plots_dir / "encoding_mode_ablation.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[gen_plots] wrote encoding_mode_ablation.png")

sweep = data.get("diagnostics", {}).get("support_size_sweep", {})

plt.figure(figsize=(3.5, 2.5))
if sweep:
    xs = sorted(int(k) for k in sweep.keys())
    enc = [sweep[str(k)].get("encoder_r2", sweep[str(k)].get("encoder_acc")) for k in xs]
    diff = [sweep[str(k)].get("diffusion_r2_mean", sweep[str(k)].get("diffusion_acc_mean")) for k in xs]

    plt.plot(xs, enc, label="encoder")
    plt.plot(xs, diff, label="diffusion")
    plt.legend(fontsize=7)
    plt.xlabel("Support Size", fontsize=8)
    plt.ylabel("Metric", fontsize=8)
    plt.tick_params(labelsize=7)
else:
    plt.text(0.5, 0.5, "No data available", ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

out = root / "figures/plots/support_sweep.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=90, bbox_inches='tight')
plt.close()

# Uncertainty summary plot (regression / bandit)
unc = data.get("overall", {})
if any(key in unc for key in ["uncertainty_mean", "uncertainty_error_correlation"]):
    plt.figure(figsize=(3.5, 2.5))
    metrics = [
        ("uncertainty_mean", unc.get("uncertainty_mean")),
        ("uncertainty_error_correlation", unc.get("uncertainty_error_correlation")),
        ("uncertainty_on_high_error_points", unc.get("uncertainty_on_high_error_points")),
        ("uncertainty_on_low_error_points", unc.get("uncertainty_on_low_error_points")),
    ]
    names = [name for name, val in metrics if val is not None]
    vals = [val for name, val in metrics if val is not None]
    if vals:
        plt.bar(names, vals)
        plt.xticks(rotation=30, ha='right', fontsize=7)
        plt.title('Uncertainty Diagnostics', fontsize=8)
        plt.tight_layout()
        out = root / "figures/plots/uncertainty_summary.png"
        plt.savefig(out, dpi=90, bbox_inches='tight')
        plt.close()

# Adaptation curve plot via support size sweep (if available)
if sweep:
    plt.figure(figsize=(3.5, 2.5))
    xs = sorted(int(k) for k in sweep.keys())
    variant = "diffusion_r2_mean" if any(s.get("diffusion_r2_mean") is not None for s in sweep.values()) else "diffusion_acc_mean"
    selected = [sweep[str(k)].get(variant) for k in xs if sweep[str(k)].get(variant) is not None]
    if selected:
        plt.plot(xs[:len(selected)], selected, marker='o', markersize=4, label='diffusion mean')
        plt.title('Adaptation Curve (support size sweep)', fontsize=8)
        plt.xlabel('Support size', fontsize=8)
        plt.ylabel('Metric', fontsize=8)
        plt.legend(fontsize=7)
        plt.tick_params(labelsize=7)
        out = root / "figures/plots/adaptation_curve.png"
        plt.savefig(out, dpi=90, bbox_inches='tight')
        plt.close()
# Baseline comparison summary (diffusion vs encoder vs selector vs static baseline)
baseline = data.get('baseline_comparison', {})
if baseline:
    plt.figure(figsize=(3.5, 2.5))
    names = []
    values = []
    for key in ['deterministic_encoder', 'diffusion_sampler', 'selector', 'static_baseline']:
        item = baseline.get(key)
        if item is not None:
            names.append(key.replace('_', ' ').title())
            values.append(item.get('r2', item.get('loss', None)))
    if values:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][:len(values)]
        plt.bar(names, values, color=colors)
        plt.title('Baseline Comparison (R2 or Loss)', fontsize=8)
        plt.ylabel('Value', fontsize=8)
        plt.xticks(fontsize=7)
        plt.tight_layout()
        out = root / 'figures/plots/baseline_comparison.png'
        plt.savefig(out, dpi=90, bbox_inches='tight')
        plt.close()
# Generate annex.tex with better captions
annex_path = root / "sections" / "annex.tex"
pngs = sorted((root / "figures" / "plots").glob("*.png"))

# Map filenames to descriptive captions
caption_map = {
    "support_sweep.png": "Support Size Sweep: Performance vs. number of support examples",
    "adaptation_curve.png": "Adaptation Curve: Model performance across different support set sizes",
    "baseline_comparison.png": "Baseline Comparison: encoder/diffusion/selector/static baseline comparison",
    "uncertainty_summary.png": "Uncertainty Diagnostics: Summary of uncertainty metrics for current experiment",
}

# Default caption generator for plots
def get_caption(filename):
    if filename in caption_map:
        return caption_map[filename]

    # Parse train/eval plots — new format: {train|eval}_{family_name}_{idx}.png
    # or reward variant: {train|eval}_{family_name}_{idx}_reward.png
    parts = filename.replace('.png', '').split('_')
    if len(parts) >= 3 and parts[0] in ('train', 'eval'):
        plot_type = parts[0]
        is_reward = parts[-1] == 'reward'
        # Strip trailing 'reward' if present
        core = parts[1:-1] if is_reward else parts[1:]
        # Last element of core should be the numeric index
        if core and core[-1].isdigit():
            idx = int(core[-1])
            family = '_'.join(core[:-1])
        else:
            # Fallback: second element is index (old format)
            idx = int(core[0]) if core[0].isdigit() else 0
            family = '_'.join(core[1:])
        family_name = family.replace('_', ' ').title()
        suffix = " (reward trajectory)" if is_reward else ""
        if plot_type == 'train':
            return f"Training Example {idx+1}: {family_name} task{suffix}."
        elif plot_type == 'eval':
            return f"Evaluation Example {idx+1}: {family_name} task generalization{suffix}."

    # Fallback
    name = filename.replace('_', ' ').replace('.png', '').title()
    return f"{name}"

with open(annex_path, "w") as f:
    f.write("\\section{Annex: Diagnostic Plots}\n\n")
    for png in pngs:
        caption = get_caption(png.name)
        f.write(f"\\begin{{center}}\n\\includegraphics[width=0.95\\linewidth]{{figures/plots/{png.name}}}\n\\end{{center}}\n\\noindent\\textbf{{Caption:}} {caption}\n\n")
