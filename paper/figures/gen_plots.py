"""
Paper figure generation - refactored for clarity and consistency.

This script generates all paper figures from experimental runs.
Each figure has a single, clear execution path with consistent colors and styling.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = root.parent / "runs"
plots_dir = root / "figures" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────
# COLORS AND STYLING  - Define once, use everywhere
# ──────────────────────────────────────────────────────────

# Main model colors (used consistently across all plots)
COLORS = {
    "encoder": "#4C72B0",        # Blue
    "diffusion": "#DD8452",      # Orange
    "static_baseline": "#55A868", # Green
    "direct_baseline": "#8C6D31", # Brown/tan (changed from red to avoid conflict)
    
    # Encoding modes - used only in encoding_mode plots
    "support": "#4C72B0",        # Blue (matches encoder)
    "text": "#DD8452",           # Orange (matches diffusion)
    "hybrid": "#55A868",         # Green (matches static baseline)
    "oracle": "#C44E52",         # Red (kept for oracle since it's distinctive)
}


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def safe_mean(values):
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def safe_std(values):
    vals = [float(v) for v in values if v is not None]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None)


def load_direct_baseline_multiseed(path: Path):
    """Load multiseed direct baseline results from a directory."""
    if not path.exists():
        return {}

    grouped = {}
    for summary_path in sorted(path.glob("direct_*_seed*/summary.json")):
        parent_name = summary_path.parent.name
        if not parent_name.startswith("direct_"):
            continue

        mode_seed = parent_name[len("direct_"):]
        mode = mode_seed.rsplit("_seed", 1)[0]
        data = load_json(summary_path)
        if not data:
            continue

        overall = data.get("overall", {})
        direct_metric = overall.get("direct_r2", overall.get("direct_acc"))
        static_metric = overall.get("baseline_r2", overall.get("baseline_acc"))
        reward_audit = (data.get("reward_audit") or {}).get("eval", {}).get("overall", {})
        reward_winrate = reward_audit.get("win_rate_vs_static")
        grouped.setdefault(mode, {"direct": [], "static": [], "reward_winrate": []})
        grouped[mode]["direct"].append(direct_metric)
        grouped[mode]["static"].append(static_metric)
        grouped[mode]["reward_winrate"].append(reward_winrate)

    out = {}
    for mode, vals in grouped.items():
        direct_list = vals.get("direct", [])
        out[mode] = {
            "direct_mean": safe_mean(direct_list),
            "direct_std": safe_std(direct_list),
            "reward_winrate_mean": safe_mean(vals.get("reward_winrate", [])),
            "reward_winrate_std": safe_std(vals.get("reward_winrate", [])),
            "num_seeds": len([v for v in direct_list if v is not None]),
        }
    return out


def load_task_direct_baseline(task: str):
    """Load direct baseline results for a specific task."""
    task_dir = "bandit_regression" if task == "bandit" else task
    structured = runs / "direct_baseline_v2_multiseed" / task_dir
    out = load_direct_baseline_multiseed(structured) if structured.exists() else {}
    
    # Backward compatibility: for control, also check flat layout
    if task == "control" and not out:
        legacy = runs / "direct_baseline_v2_multiseed"
        if legacy.exists():
            out = load_direct_baseline_multiseed(legacy)
    
    return out


def load_matrix_aggregate():
    """Load control matrix multiseed aggregate (if available)."""
    p = runs / "control_matrix_v2_multiseed" / "aggregate.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_matrix_single():
    """Load single-seed control matrix (fallback)."""
    p = runs / "control_matrix_v2" / "matrix_summary.json"
    if p.exists():
        return json.loads(p.read_text()).get("table", [])
    p = runs / "control_matrix_main" / "matrix_summary.json"
    if p.exists():
        return json.loads(p.read_text()).get("table", [])
    return None


# ──────────────────────────────────────────────────────────
# PRE-LOAD ALL DATA
# ──────────────────────────────────────────────────────────

TASKS = ["classification", "regression", "bandit", "control"]
TASK_LABELS = ["Classification\n(Acc.)", "Regression\n(R²)", "Bandit\n(R²)", "Control\n(R²)"]

# Load main task summaries
task_data = {}
for task, task_dir in zip(TASKS, [f"{t}_v2" for t in TASKS]):
    task_data[task] = load_json(runs / task_dir / "summary.json") or {}

# Load direct baseline multiseed
direct_baseline_multiseed = {task: load_task_direct_baseline(task) for task in TASKS}

# Load control matrix for encoding mode ablation
control_matrix_agg = load_matrix_aggregate()
control_matrix_single = load_matrix_single()

print("[gen_plots] Data loaded successfully")


# ──────────────────────────────────────────────────────────
# PLOT 1: PER-TASK BENCHMARK
# ──────────────────────────────────────────────────────────

def plot_task_benchmark():
    """Generate per-task performance benchmark (all 4 main methods across all tasks)."""
    enc_vals, base_vals, direct_vals = [], [], []
    
    for task in TASKS:
        ov = task_data.get(task, {}).get("overall", {})
        is_acc = "encoder_acc" in ov
        suffix = "_acc" if is_acc else "_r2"
        
        enc_vals.append(ov.get(f"encoder{suffix}"))
        base_vals.append(ov.get(f"baseline{suffix}"))
        
        # Get direct baseline mean if available
        db = direct_baseline_multiseed.get(task, {}).get("support", {})
        direct_vals.append(db.get("direct_mean"))
    
    if not any(v is not None for v in enc_vals):
        print("[gen_plots] SKIPPED task_benchmark: no encoder data")
        return
    
    x = np.arange(len(TASKS))
    has_direct = any(v is not None for v in direct_vals)
    w = 0.26 if has_direct else 0.35
    
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    
    if has_direct:
        ax.bar(x - w, [v or 0 for v in enc_vals], w, label="HN-det",
               color=COLORS["encoder"])
        ax.bar(x, [v or 0 for v in base_vals], w, label="MLP-fixed",
               color=COLORS["static_baseline"])
        ax.bar(x + w, [v or 0 for v in direct_vals], w, label="MLP-adapt",
               color=COLORS["direct_baseline"])
        
        # Add value labels for direct baseline
        for idx, v in enumerate(direct_vals):
            if v is not None:
                ax.text(x[idx] + w, v + 0.03, f"{v:.3f}", ha="center",
                       va="bottom", fontsize=6, color=COLORS["direct_baseline"])
    else:
        ax.bar(x - 0.5 * w, [v or 0 for v in enc_vals], w, label="HN-det",
               color=COLORS["encoder"])
        ax.bar(x + 0.5 * w, [v or 0 for v in base_vals], w, label="MLP-fixed",
               color=COLORS["static_baseline"])
    
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=8)
    ax.set_ylabel("Performance", fontsize=8)
    title_suffix = " (w/ Direct Baseline)" if has_direct else ""
    ax.set_title(f"Per-Task Benchmark{title_suffix}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    ax.set_ylim(bottom=min(0, min(v for v in base_vals if v is not None) - 0.1))
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(plots_dir / "task_benchmark.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[gen_plots] wrote task_benchmark.png")


# ──────────────────────────────────────────────────────────
# PLOT 2: ENCODING MODE ABLATION (control task only)
# ──────────────────────────────────────────────────────────

def plot_encoding_mode_ablation():
    """Generate encoding mode ablation plot (R² by mode + reward win-rate)."""
    
    # Try multiseed aggregate first
    if control_matrix_agg and control_matrix_agg.get("summary_table"):
        rows = control_matrix_agg.get("summary_table", [])
        modes = [r["encoding_mode"] for r in rows]
        mode_labels = [m.capitalize() for m in modes]
        enc_r2s = [r.get("eval_encoder_r2_mean", 0) for r in rows]
        diff_r2s = [r.get("eval_diffusion_r2_mean", 0) for r in rows]
        base_r2s = [r.get("eval_static_baseline_r2_mean", 0) for r in rows]
        control_direct = direct_baseline_multiseed.get("control", {})
        direct_r2s = [(control_direct.get(m) or {}).get("direct_mean") for m in modes]
        winrates = [r.get("eval_reward_winrate_vs_static_mean", 0) for r in rows]
        enc_std = [r.get("eval_encoder_r2_std", 0) for r in rows]
        diff_std = [r.get("eval_diffusion_r2_std", 0) for r in rows]
        base_std = [r.get("eval_static_baseline_r2_std", 0) for r in rows]
        direct_std = [(control_direct.get(m) or {}).get("direct_std") for m in modes]
        win_std = [r.get("eval_reward_winrate_vs_static_std", 0) for r in rows]
        
        source = "multiseed"
    elif control_matrix_single:
        modes        = [r["encoding_mode"] for r in control_matrix_single]
        mode_labels  = [m.capitalize() for m in modes]
        enc_r2s      = [r.get("eval_encoder_r2",  0) for r in control_matrix_single]
        diff_r2s     = [r.get("eval_diffusion_r2", 0) for r in control_matrix_single]
        base_r2s     = [r.get("eval_static_baseline_r2", 0) for r in control_matrix_single]
        control_direct = direct_baseline_multiseed.get("control", {})
        direct_r2s   = [(control_direct.get(m) or {}).get("direct_mean") for m in modes]
        winrates     = [r.get("eval_reward_winrate_vs_static", 0) for r in control_matrix_single]
        enc_std = [0] * len(modes)
        diff_std = [0] * len(modes)
        base_std = [0] * len(modes)
        direct_std = [0] * len(modes)
        win_std = [0] * len(modes)
        
        source = "single-seed"
    else:
        print("[gen_plots] SKIPPED encoding_mode_ablation: no control matrix data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(modes))
    has_direct = any(v is not None for v in direct_r2s)
    w = 0.26 if has_direct else 0.35
    
    # Left panel: R² by mode
    if has_direct:
        ax1.bar(x - w, enc_r2s, w, yerr=enc_std, capsize=3, label="HN-det",
               color=COLORS["encoder"])
        ax1.bar(x, base_r2s, w, yerr=base_std, capsize=3, label="MLP-fixed",
               color=COLORS["static_baseline"])
        ax1.bar(x + w, [v if v is not None else np.nan for v in direct_r2s], w,
                yerr=[v if v is not None else 0.0 for v in direct_std], capsize=3,
                label="MLP-adapt", color=COLORS["direct_baseline"])
        
        for i, v in enumerate(direct_r2s):
            if v is not None:
                ax1.text(x[i] + w, v + 0.02, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=6, color=COLORS["direct_baseline"])
    else:
        ax1.bar(x - 0.5 * w, enc_r2s, w, yerr=enc_std, capsize=3, label="HN-det",
               color=COLORS["encoder"])
        ax1.bar(x + 0.5 * w, base_r2s, w, yerr=base_std, capsize=3, label="MLP-fixed",
               color=COLORS["static_baseline"])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels, fontsize=8)
    ax1.set_ylabel("R² (mean ± std)", fontsize=8)
    ax1.set_title("R² by Encoding Mode", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)
    ax1.set_ylim(-0.3, 1.05)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    
    # Right panel: Reward win-rate (mode bars + direct support marker/bar when available)
    mode_colors = [COLORS.get(m, "#cccccc") for m in modes]
    ax2.bar(mode_labels, winrates, yerr=win_std, capsize=3, color=mode_colors, label="HN-det modes")

    direct_support_reward = (direct_baseline_multiseed.get("control", {}).get("support") or {}).get("reward_winrate_mean")
    direct_support_reward_std = (direct_baseline_multiseed.get("control", {}).get("support") or {}).get("reward_winrate_std")
    if direct_support_reward is not None:
        x_direct = len(mode_labels)
        ax2.bar(
            [x_direct],
            [direct_support_reward],
            yerr=[direct_support_reward_std if direct_support_reward_std is not None else 0.0],
            capsize=3,
            color=COLORS["direct_baseline"],
            label="MLP-adapt",
        )
        labels = mode_labels + ["Direct\n(support)"]
        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.text(
            x_direct,
            direct_support_reward + 0.03,
            f"{direct_support_reward*100:.1f}%",
            ha="center",
            fontsize=7,
            fontweight="bold",
            color=COLORS["direct_baseline"],
        )
    else:
        ax2.set_xticks(np.arange(len(mode_labels)))
        ax2.set_xticklabels(mode_labels, fontsize=8)

    ax2.set_ylabel("Reward Win-Rate vs. Baseline (mean ± std)", fontsize=8)
    ax2.set_title("OOD Reward Win-Rate by Mode", fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    
    for i, v in enumerate(winrates):
        ax2.text(i, v + 0.03, f"{v*100:.1f}%", ha="center", fontsize=7, fontweight="bold")
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, loc="upper right")
    
    n_seeds = control_matrix_agg.get("num_seeds", "?") if source == "multiseed" else "1"
    fig.suptitle(f"Encoding-Mode Ablation (Control OOD, {n_seeds} seed{'s' if n_seeds != '1' else ''})", 
                fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "encoding_mode_ablation.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen_plots] wrote encoding_mode_ablation.png ({source})")


# ──────────────────────────────────────────────────────────
# PLOT 3: SUPPORT SWEEP
# ──────────────────────────────────────────────────────────

def plot_support_sweep():
    """Generate support size sweep plot."""
    support_plot_data = task_data.get("control", {}) or task_data.get("regression", {}) or {}
    sweep = support_plot_data.get("diagnostics", {}).get("support_size_sweep", {})

    if not sweep:
        print("[gen_plots] SKIPPED support_sweep: no sweep data")
        return

    fig = plt.figure(figsize=(3.5, 2.5))
    xs = sorted(int(k) for k in sweep.keys())
    enc = [sweep[str(k)].get("encoder_r2", sweep[str(k)].get("encoder_acc")) for k in xs]

    plt.plot(xs, enc, label="HN-det", marker='o', markersize=4)
    plt.legend(fontsize=7)
    plt.xlabel("Support Size", fontsize=8)
    plt.ylabel("Metric", fontsize=8)
    plt.title("Support Size Sweep", fontsize=8)
    plt.tick_params(labelsize=7)
    plt.tight_layout()

    fig.savefig(plots_dir / "support_sweep.png", dpi=90, bbox_inches='tight')
    plt.close(fig)
    print("[gen_plots] wrote support_sweep.png")


# ──────────────────────────────────────────────────────────
# PLOT 4: UNCERTAINTY SUMMARY
# ──────────────────────────────────────────────────────────

def plot_uncertainty_summary():
    """Generate uncertainty diagnostics plot (regression/bandit/control)."""
    uncertainty_plot_data = (task_data.get("regression", {}) or task_data.get("bandit", {})
                            or task_data.get("control", {}) or {})
    unc = uncertainty_plot_data.get("overall", {})

    if not any(key in unc for key in ["uncertainty_mean", "uncertainty_error_correlation"]):
        print("[gen_plots] SKIPPED uncertainty_summary: no uncertainty data")
        return

    metrics = [
        ("uncertainty_mean", unc.get("uncertainty_mean")),
        ("uncertainty_error_correlation", unc.get("uncertainty_error_correlation")),
        ("uncertainty_on_high_error_points", unc.get("uncertainty_on_high_error_points")),
        ("uncertainty_on_low_error_points", unc.get("uncertainty_on_low_error_points")),
    ]
    names = [name for name, val in metrics if val is not None]
    vals = [val for name, val in metrics if val is not None]

    if vals:
        fig = plt.figure(figsize=(3.5, 2.5))
        plt.bar(names, vals, color=COLORS["diffusion"])
        plt.xticks(rotation=30, ha='right', fontsize=7)
        plt.title('Uncertainty Diagnostics', fontsize=8)
        plt.tight_layout()
        fig.savefig(plots_dir / "uncertainty_summary.png", dpi=90, bbox_inches='tight')
        plt.close(fig)
        print("[gen_plots] wrote uncertainty_summary.png")
    else:
        print("[gen_plots] SKIPPED uncertainty_summary: no valid metrics")


# ──────────────────────────────────────────────────────────
# PLOT 6: BASELINE COMPARISON
# ──────────────────────────────────────────────────────────

def plot_baseline_comparison():
    """Generate baseline comparison (encoder vs selector vs static vs direct)."""
    baseline_data = task_data.get("control", {})
    baseline = baseline_data.get('baseline_comparison', {})

    if not baseline:
        print("[gen_plots] SKIPPED baseline_comparison: no baseline_comparison data")
        return

    names = []
    values = []

    _name_map = {
        'deterministic_encoder': 'HN-det',
        'static_baseline': 'MLP-fixed',
        'selector': 'Selector',
    }
    for key in ['deterministic_encoder', 'selector', 'static_baseline']:
        item = baseline.get(key)
        if item is not None:
            method_name = _name_map.get(key, key.replace('_', ' ').title())
            metric_val = item.get('r2', item.get('loss', None))
            if metric_val is not None:
                names.append(method_name)
                values.append(metric_val)

    direct_support = (direct_baseline_multiseed.get("control", {}).get("support") or {}).get("direct_mean")
    if direct_support is not None:
        names.append("MLP-adapt")
        values.append(direct_support)

    if values:
        fig = plt.figure(figsize=(3.5, 2.5))
        colors = []
        for name in names:
            if "HN-det" in name:
                colors.append(COLORS["encoder"])
            elif "HN-diff" in name:
                colors.append(COLORS["diffusion"])
            elif "Selector" in name:
                colors.append("#999999")
            elif "MLP-fixed" in name:
                colors.append(COLORS["static_baseline"])
            elif "MLP-adapt" in name:
                colors.append(COLORS["direct_baseline"])
            else:
                colors.append("#cccccc")

        plt.bar(names, values, color=colors)
        plt.title('Baseline Comparison', fontsize=8)
        plt.ylabel('Value (R² or Loss)', fontsize=8)
        plt.xticks(fontsize=7, rotation=30, ha='right')
        plt.tight_layout()
        fig.savefig(plots_dir / "baseline_comparison.png", dpi=90, bbox_inches='tight')
        plt.close(fig)
        print("[gen_plots] wrote baseline_comparison.png")
    else:
        print("[gen_plots] SKIPPED baseline_comparison: no valid metrics")


# ──────────────────────────────────────────────────────────
# PLOT 7: ANNEX WITH ALL DIAGNOSTIC PLOTS
# ──────────────────────────────────────────────────────────

def generate_annex():
    """Generate annex.tex with captions for diagnostic plots."""
    annex_path = root / "sections" / "annex.tex"
    preferred_order = [
        "task_benchmark.png",
        "encoding_mode_ablation.png",
        "support_sweep.png",
        "uncertainty_summary.png",
        "baseline_comparison.png",
    ]
    all_pngs = sorted((root / "figures" / "plots").glob("*.png"))
    preferred = [root / "figures" / "plots" / name for name in preferred_order if (root / "figures" / "plots" / name).exists()]
    preferred_set = {p.name for p in preferred}
    extras = [p for p in all_pngs if p.name not in preferred_set]
    pngs = preferred + extras

    caption_map = {
        "support_sweep.png": "Support Size Sweep: Performance vs. number of support examples",
        "adaptation_curve.png": "Adaptation Curve: Model performance across different support set sizes",
        "baseline_comparison.png": "Baseline Comparison: HN-det / selector / MLP-fixed / MLP-adapt",
        "uncertainty_summary.png": "Uncertainty Diagnostics: Summary of uncertainty metrics",
        "task_benchmark.png": "Per-Task Benchmark: Performance across classification, regression, bandit, and control tasks",
        "encoding_mode_ablation.png": "Encoding-Mode Ablation: R² and reward win-rate by encoding mode (control task)",
    }

    def get_caption(filename):
        if filename in caption_map:
            return caption_map[filename]

        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 3 and parts[0] in ('train', 'eval'):
            plot_type = parts[0]
            is_reward = parts[-1] == 'reward'
            core = parts[1:-1] if is_reward else parts[1:]
            if core and core[-1].isdigit():
                idx = int(core[-1])
                family = '_'.join(core[:-1])
            else:
                idx = int(core[0]) if core[0].isdigit() else 0
                family = '_'.join(core[1:])
            family_name = family.replace('_', ' ').title()
            suffix = " (reward trajectory)" if is_reward else ""
            if plot_type == 'train':
                return f"Training Example {idx+1}: {family_name} task{suffix}."
            if plot_type == 'eval':
                return f"Evaluation Example {idx+1}: {family_name} task generalization{suffix}."

        name = filename.replace('_', ' ').replace('.png', '').title()
        return f"{name}"

    with open(annex_path, "w") as f:
        f.write("\\section{Annex: Diagnostic Plots}\n\n")
        for png in pngs:
            caption = get_caption(png.name)
            f.write("\\begin{center}\n")
            f.write(f"\\includegraphics[width=0.95\\linewidth]{{figures/plots/{png.name}}}\n")
            f.write("\\end{center}\n")
            f.write(f"\\noindent\\textbf{{Caption:}} {caption}\n\n")

    print(f"[gen_plots] wrote annex.tex ({len(pngs)} figures)")


if __name__ == "__main__":
    print("\n[gen_plots] Starting figure generation...")
    print(f"[gen_plots] Output dir: {plots_dir}")

    plot_task_benchmark()
    plot_encoding_mode_ablation()
    plot_support_sweep()
    plot_uncertainty_summary()
    plot_baseline_comparison()
    generate_annex()

    print("\n[gen_plots] Figure generation complete!")
