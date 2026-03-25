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

# Generate annex.tex
annex_path = root / "sections" / "annex.tex"
pngs = sorted((root / "figures" / "plots").glob("*.png"))
with open(annex_path, "w") as f:
    f.write("\\section{Annex: Diagnostic Plots}\n\n")
    for png in pngs:
        name = png.stem.replace("_", " ").title()
        f.write(f"\\begin{{figure}}[h]\n\\centering\n\\includegraphics[width=0.8\\linewidth]{{figures/plots/{png.name}}}\n\\caption{{{name}}}\n\\end{{figure}}\n\n")
