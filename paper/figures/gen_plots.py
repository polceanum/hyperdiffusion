import json
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "results/latest.json").read_text())

sweep = data.get("diagnostics", {}).get("support_size_sweep", {})

plt.figure(figsize=(6, 4))
if sweep:
    xs = sorted(int(k) for k in sweep.keys())
    enc = [sweep[str(k)].get("encoder_r2", sweep[str(k)].get("encoder_acc")) for k in xs]
    diff = [sweep[str(k)].get("diffusion_r2_mean", sweep[str(k)].get("diffusion_acc_mean")) for k in xs]

    plt.plot(xs, enc, label="encoder")
    plt.plot(xs, diff, label="diffusion")
    plt.legend()
    plt.xlabel("Support Size")
    plt.ylabel("Metric")
else:
    plt.text(0.5, 0.5, "No data available", ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

out = root / "figures/plots/support_sweep.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out)
plt.close()

# Uncertainty summary plot (regression / bandit)
unc = data.get("overall", {})
if any(key in unc for key in ["uncertainty_mean", "uncertainty_error_correlation"]):
    plt.figure(figsize=(6, 4))
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
        plt.xticks(rotation=30, ha='right')
        plt.title('Uncertainty Diagnostics')
        plt.tight_layout()
        out = root / "figures/plots/uncertainty_summary.png"
        plt.savefig(out)
        plt.close()

# Adaptation curve plot via support size sweep (if available)
if sweep:
    plt.figure(figsize=(6, 4))
    xs = sorted(int(k) for k in sweep.keys())
    variant = "diffusion_r2_mean" if any(s.get("diffusion_r2_mean") is not None for s in sweep.values()) else "diffusion_acc_mean"
    selected = [sweep[str(k)].get(variant) for k in xs if sweep[str(k)].get(variant) is not None]
    if selected:
        plt.plot(xs[:len(selected)], selected, marker='o', label='diffusion mean')
        plt.title('Adaptation Curve (support size sweep)')
        plt.xlabel('Support size')
        plt.ylabel('Metric')
        plt.legend()
        out = root / "figures/plots/adaptation_curve.png"
        plt.savefig(out)
        plt.close()
# Baseline comparison summary (diffusion vs encoder vs selector vs static baseline)
baseline = data.get('baseline_comparison', {})
if baseline:
    plt.figure(figsize=(6, 4))
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
        plt.title('Baseline Comparison (R2 or Loss)')
        plt.ylabel('Value')
        plt.tight_layout()
        out = root / 'figures/plots/baseline_comparison.png'
        plt.savefig(out)
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
        f.write(f"\\begin{{center}}\n\\includegraphics[width=0.8\\linewidth]{{figures/plots/{png.name}}}\n\\end{{center}}\n\\noindent\\textbf{{Caption:}} {caption}\n\n")
