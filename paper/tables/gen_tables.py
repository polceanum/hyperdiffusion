import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "results/latest.json").read_text())

def fmt(x):
    return "--" if x is None else f"{x:.3f}"

overall = data.get("overall", {})

encoder = overall.get("encoder_r2") or overall.get("encoder_acc")
diff = overall.get("diffusion_r2") or overall.get("diffusion_acc")
bestk = overall.get("diffusion_r2_best_k") or overall.get("diffusion_acc_best_k")

latex = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lccc}}
\\toprule
Method & Encoder & Diffusion & Best-of-k \\\
\\midrule
Value & {fmt(encoder)} & {fmt(diff)} & {fmt(bestk)} \\\
\\bottomrule
\\end{{tabular}}
\\caption{{Main results}}
\\end{{table}}
"""

(root / "tables/main_results.tex").write_text(latex)
