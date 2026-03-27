"""Generate LaTeX tables for the paper."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = root.parent / "runs"


def fmt(x, digits=3):
    return "--" if x is None else f"{x:.{digits}f}"


def fmt_pct(x):
    return "--" if x is None else f"{x * 100:.1f}\\%"


def fmt_pm(mean, std, digits=3):
    if mean is None:
        return "--"
    if std is None:
        return fmt(mean, digits)
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_pct_pm(mean, std):
    if mean is None:
        return "--"
    if std is None:
        return fmt_pct(mean)
    return f"{mean * 100:.1f}\\% $\\pm$ {std * 100:.1f}\\%"


TASK_DISPLAY = [
    ("classification", "Classification", "Acc."),
    ("regression", "Regression", "R\\textsuperscript{2}"),
    ("bandit", "Bandit", "R\\textsuperscript{2}"),
    ("control", "Control", "R\\textsuperscript{2}"),
]


def load_task_row(task):
    p = runs / f"{task}_v2" / "summary.json"
    if not p.exists():
        return None, None, None
    data = json.loads(p.read_text())
    overall = data.get("overall", {})
    is_acc = "encoder_acc" in overall
    enc = overall.get("encoder_acc" if is_acc else "encoder_r2")
    diff = overall.get("diffusion_acc" if is_acc else "diffusion_r2")
    base = overall.get("baseline_acc" if is_acc else "baseline_r2")
    return enc, diff, base


def gen_task_benchmark_table():
    body_lines = []
    for task, display, metric in TASK_DISPLAY:
        enc, diff, base = load_task_row(task)
        gap = None
        if enc is not None and base is not None and base != 0:
            gap = ((enc - base) / abs(base)) * 100
        gap_str = "--" if gap is None else f"{gap:+.1f}\\%"
        line = (
            f"  {display} & {metric} & {fmt(enc)} & {fmt(diff)} & "
            f"{fmt(base)} & {gap_str} \\\\" 
        )
        body_lines.append(line)
    body = "\n".join(body_lines)

    return "\n".join([
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Task & Metric & Encoder & Diffusion & Baseline & Enc Gain \\\\ ",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Per-task benchmark from run-level \\texttt{overall} summaries (v2 results, 1000 training steps per stage). "
        "\\emph{Encoder}: hypernetwork with attention-based support encoding. "
        "\\emph{Diffusion}: optional DDIM-sampled latent add-on. "
        "\\emph{Baseline}: static MLP with no per-task adaptation. Enc Gain shows relative improvement of Encoder over Baseline.}",
        "\\label{tab:task_benchmark}",
        "\\end{table}",
        "",
    ])


MODE_DISPLAY = {
    "support": "Support",
    "text": "Text (DistilBERT)",
    "hybrid": "Hybrid",
    "oracle": "Oracle",
}


def load_matrix_rows():
    for dirname in ["control_matrix_v2", "control_matrix_main"]:
        p = runs / dirname / "matrix_summary.json"
        if p.exists():
            data = json.loads(p.read_text())
            return data.get("table", []), dirname
    return [], None


def load_matrix_aggregate_rows():
    p = runs / "control_matrix_v2_multiseed" / "aggregate.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("summary_table", []), data.get("num_seeds", 0)


def gen_encoding_mode_table():
    agg = load_matrix_aggregate_rows()
    body_lines = []
    if agg is not None:
        rows, n_seeds = agg
        note = f"(DistilBERT embeddings, 1000 steps, {n_seeds} seeds; mean$\\pm$std)"
        for row in rows:
            mode = MODE_DISPLAY.get(row.get("encoding_mode", "?"), row.get("encoding_mode", "?"))
            line = (
                f"  {mode} & "
                f"{fmt_pm(row.get('eval_encoder_r2_mean'), row.get('eval_encoder_r2_std'))} & "
                f"{fmt_pm(row.get('eval_diffusion_r2_mean'), row.get('eval_diffusion_r2_std'))} & "
                f"{fmt_pm(row.get('eval_static_baseline_r2_mean'), row.get('eval_static_baseline_r2_std'))} & "
                f"{fmt_pct_pm(row.get('eval_reward_winrate_vs_static_mean'), row.get('eval_reward_winrate_vs_static_std'))} & "
                f"{fmt_pm(row.get('eval_reward_mean_delta_ls_mean'), row.get('eval_reward_mean_delta_ls_std'), 2)} \\\\"
            )
            body_lines.append(line)
    else:
        rows, src = load_matrix_rows()
        if src is None:
            note = "(results pending)"
        elif "v2" in src:
            note = "(DistilBERT embeddings, 1000 steps)"
        else:
            note = "(hashed bag-of-words, 300 steps)"
        for row in rows:
            mode = MODE_DISPLAY.get(row.get("encoding_mode", "?"), row.get("encoding_mode", "?"))
            enc_r2 = row.get("eval_encoder_r2")
            diff_r2 = row.get("eval_diffusion_r2")
            base_r2 = row.get("eval_static_baseline_r2")
            winrate = row.get("eval_reward_winrate_vs_static")
            delta = row.get("eval_reward_mean_delta_ls")
            line = (
                f"  {mode} & {fmt(enc_r2)} & {fmt(diff_r2)} & {fmt(base_r2)} & "
                f"{fmt_pct(winrate)} & {fmt(delta, 2)} \\\\"
            )
            body_lines.append(line)

    body = "\n".join(body_lines) if body_lines else "  \\multicolumn{6}{c}{(results pending)} \\\\"

    return "\n".join([
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Mode & Enc R\\textsuperscript{2} & Diff R\\textsuperscript{2} & Base R\\textsuperscript{2} & Reward Win & $\\Delta$Reward \\\\",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Encoding-mode ablation on control tasks " + note + ". "
        "\\emph{Win-rate}: fraction of OOD eval episodes where the meta-learned "
        "policy out-scores the static baseline. "
        "$\\Delta$Reward: mean reward improvement over static baseline.}",
        "\\label{tab:encoding_modes}",
        "\\end{table}",
        "",
    ])


main_tex = gen_task_benchmark_table() + "\n" + gen_encoding_mode_table()
(root / "tables" / "main_results.tex").write_text(main_tex)
print("[tables] wrote main_results.tex")
