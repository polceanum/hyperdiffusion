from __future__ import annotations

import json
import statistics
import subprocess
import shutil

from .config import PipelineConfig
from .runner import log, remove_paths, run_cmd, run_cmd_capture_stdout
from .validate import (
    validate_experiment_outputs,
    validate_paper_artifacts,
    write_provenance_manifest,
)


PRIMARY_PLOT_NAMES = {
    "task_benchmark.png",
    "encoding_mode_ablation.png",
    "support_sweep.png",
    "uncertainty_summary.png",
    "baseline_comparison.png",
}


def _sync_diagnostic_plots_from_runs(cfg: PipelineConfig) -> None:
    """Sync train/eval diagnostic PNGs from canonical run outputs into paper plots dir.

    This keeps annex inputs aligned with latest experiment artifacts even when users
    run refresh-artifacts without rerunning full training in the same process.
    """
    paper_plots_dir = cfg.paper_dir / "figures" / "plots"
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    # Remove previously-synced diagnostics while preserving curated summary plots.
    for png in paper_plots_dir.glob("*.png"):
        if png.name in PRIMARY_PLOT_NAMES:
            continue
        if png.name.startswith("train_") or png.name.startswith("eval_"):
            png.unlink(missing_ok=True)

    canonical_roots = [
        cfg.runs_dir / "classification_v2",
        cfg.runs_dir / "regression_v2",
        cfg.runs_dir / "bandit_v2",
        cfg.runs_dir / "control_v2",
        cfg.runs_dir / "classification_cross_family_v2",
        cfg.runs_dir / "regression_cross_family_v2",
        cfg.runs_dir / "bandit_cross_family_v2",
        cfg.runs_dir / "control_cross_family_v2",
        cfg.runs_dir / "control_matrix_v2_multiseed",
        cfg.runs_dir / "direct_baseline_v2_multiseed",
        cfg.runs_dir / "direct_baseline_cross_family_v2_multiseed",
    ]

    plot_dirs = []
    for root in canonical_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("plots")):
            if path.is_dir():
                plot_dirs.append(path)

    # Keep newest source per filename to avoid stale overwrite ordering.
    newest_by_name: dict[str, tuple[float, object]] = {}
    for plot_dir in plot_dirs:
        for png in plot_dir.glob("*.png"):
            name = png.name
            if not (name.startswith("train_") or name.startswith("eval_")):
                continue
            mtime = png.stat().st_mtime
            prev = newest_by_name.get(name)
            if prev is None or mtime >= prev[0]:
                newest_by_name[name] = (mtime, png)

    for _, src in newest_by_name.values():
        shutil.copy2(src, paper_plots_dir / src.name)

    log(
        "Synced diagnostic plots to paper figures "
        f"(files={len(newest_by_name)}, source_dirs={len(plot_dirs)})"
    )


def clean_refresh_outputs(cfg: PipelineConfig) -> None:
    log("Cleaning previous benchmark and matrix outputs")
    remove_paths(
        [
            cfg.runs_dir / "classification_v2",
            cfg.runs_dir / "regression_v2",
            cfg.runs_dir / "bandit_v2",
            cfg.runs_dir / "control_v2",
            cfg.runs_dir / "control_matrix_v2_multiseed",
            cfg.runs_dir / "direct_baseline_v2_multiseed",
            cfg.runs_dir / "classification_cross_family_v2",
            cfg.runs_dir / "regression_cross_family_v2",
            cfg.runs_dir / "bandit_cross_family_v2",
            cfg.runs_dir / "control_cross_family_v2",
            cfg.runs_dir / "direct_baseline_cross_family_v2_multiseed",
            cfg.runs_dir / "cls_v2.log",
            cfg.runs_dir / "reg_v2.log",
            cfg.runs_dir / "bandit_v2.log",
            cfg.runs_dir / "ctrl_v2.log",
            cfg.runs_dir / "cls_cross_family_v2.log",
            cfg.runs_dir / "reg_cross_family_v2.log",
            cfg.runs_dir / "bandit_cross_family_v2.log",
            cfg.runs_dir / "ctrl_cross_family_v2.log",
        ]
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)


def run_v2_benchmark(cfg: PipelineConfig) -> None:
    log("Running v2 benchmark suite (classification, regression, bandit, control)")
    tasks = [
        ("classification", "classification_v2", "cls_v2.log"),
        ("regression", "regression_v2", "reg_v2.log"),
        ("bandit_regression", "bandit_v2", "bandit_v2.log"),
        ("control", "control_v2", "ctrl_v2.log"),
    ]
    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for task_type, out_dir, log_name in tasks:
        cmd = [
            cfg.python_exec,
            "-m",
            "hyperweights.experiment",
            "--task-type",
            task_type,
            "--output-dir",
            str(cfg.runs_dir / out_dir),
                "--protocol-suite",
                "held_out",
            "--train-steps-stage1",
            str(cfg.train_steps_stage1),
            "--train-steps-stage2",
            str(cfg.train_steps_stage2),
            "--eval-batches",
            str(cfg.eval_batches),
            "--batch-size",
            str(cfg.batch_size),
            "--visualization-count",
            str(cfg.visualization_count),
        ]
        if task_type == "control":
            cmd += [
                "--reward-audit-batches",
                str(cfg.reward_audit_batches),
                "--reward-audit-batch-size",
                str(cfg.reward_audit_batch_size),
            ]
        log_path = cfg.runs_dir / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        out = log_path.open("w", encoding="utf-8")
        procs.append((task_type, subprocess.Popen(cmd, cwd=str(cfg.root), stdout=out, stderr=subprocess.STDOUT, text=True)))

    failed = []
    for task_type, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append(task_type)
            log(f"{task_type} FAILED")
        else:
            log(f"{task_type} DONE")
    if failed:
        raise RuntimeError(f"Benchmark tasks failed: {', '.join(failed)}")


def run_cross_family_benchmark(cfg: PipelineConfig) -> None:
    log("Running cross-family benchmark suite (disjoint train/eval partitions)")
    tasks = [
        ("classification", "classification_cross_family_v2", "cls_cross_family_v2.log"),
        ("regression", "regression_cross_family_v2", "reg_cross_family_v2.log"),
        ("bandit_regression", "bandit_cross_family_v2", "bandit_cross_family_v2.log"),
        ("control", "control_cross_family_v2", "ctrl_cross_family_v2.log"),
    ]
    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for task_type, out_dir, log_name in tasks:
        cmd = [
            cfg.python_exec,
            "-m",
            "hyperweights.experiment",
            "--task-type",
            task_type,
            "--protocol-suite",
            "cross_family",
            "--output-dir",
            str(cfg.runs_dir / out_dir),
            "--train-steps-stage1",
            str(cfg.train_steps_stage1),
            "--train-steps-stage2",
            str(cfg.train_steps_stage2),
            "--eval-batches",
            str(cfg.eval_batches),
            "--batch-size",
            str(cfg.batch_size),
            "--visualization-count",
            str(cfg.visualization_count),
        ]
        if task_type == "control":
            cmd += [
                "--reward-audit-batches",
                str(cfg.reward_audit_batches),
                "--reward-audit-batch-size",
                str(cfg.reward_audit_batch_size),
            ]
        log_path = cfg.runs_dir / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        out = log_path.open("w", encoding="utf-8")
        procs.append((task_type, subprocess.Popen(cmd, cwd=str(cfg.root), stdout=out, stderr=subprocess.STDOUT, text=True)))

    failed = []
    for task_type, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append(task_type)
            log(f"{task_type} cross_family FAILED")
        else:
            log(f"{task_type} cross_family DONE")
    if failed:
        raise RuntimeError(f"Cross-family benchmark tasks failed: {', '.join(failed)}")


def run_control_matrix_multiseed(cfg: PipelineConfig) -> None:
    log("Running control encoding matrix (multiseed)")
    output_root = cfg.runs_dir / "control_matrix_v2_multiseed"
    output_root.mkdir(parents=True, exist_ok=True)
    run_summaries: dict[str, str] = {}

    for seed in cfg.seeds:
        out_dir = output_root / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        modes = ["support", "text", "hybrid", "oracle"]
        table = []
        for mode in modes:
            mode_out = out_dir / mode
            cmd = [
                cfg.python_exec,
                "-m",
                "hyperweights.experiment",
                "--task-type",
                "control",
                "--encoding-mode",
                mode,
                "--protocol-suite",
                "held_out",
                "--output-dir",
                str(mode_out),
                "--seed",
                str(seed),
                "--train-steps-stage1",
                str(cfg.train_steps_stage1),
                "--train-steps-stage2",
                str(cfg.train_steps_stage2),
                "--eval-batches",
                str(cfg.eval_batches),
                "--batch-size",
                str(cfg.batch_size),
                "--support-sweep-batches",
                str(cfg.support_sweep_batches),
                "--diagnostic-samples",
                str(cfg.diagnostic_samples),
                "--visualization-count",
                str(cfg.visualization_count),
                "--reward-audit-batches",
                str(cfg.reward_audit_batches),
                "--reward-audit-batch-size",
                str(cfg.reward_audit_batch_size),
                "--text-embedding-dim",
                "768",
            ]
            run_cmd(cmd, cwd=cfg.root)
            summary = json.loads((mode_out / "summary.json").read_text())
            gen = summary.get("generalization", {}).get("eval_summary", {}).get("overall", {})
            reward_eval = summary.get("reward_audit", {}).get("eval", {}).get("overall", {})
            reward_train = summary.get("reward_audit", {}).get("train", {}).get("overall", {})
            table.append(
                {
                    "encoding_mode": mode,
                    "eval_encoder_r2": gen.get("encoder_r2"),
                    "eval_diffusion_r2": gen.get("diffusion_r2"),
                    "eval_static_baseline_r2": gen.get("baseline_r2"),
                    "eval_gap_encoder_minus_static": (
                        gen.get("encoder_r2") - gen.get("baseline_r2")
                        if gen.get("encoder_r2") is not None and gen.get("baseline_r2") is not None
                        else None
                    ),
                    "eval_reward_mean_delta_ls": reward_eval.get("mean_delta_ls"),
                    "eval_reward_winrate_vs_static": reward_eval.get("win_rate_vs_static"),
                    "train_reward_mean_delta_ls": reward_train.get("mean_delta_ls"),
                    "train_reward_winrate_vs_static": reward_train.get("win_rate_vs_static"),
                    "output_dir": str(mode_out),
                }
            )

        (out_dir / "matrix_summary.json").write_text(json.dumps({"modes": modes, "table": table}, indent=2))
        run_summaries[str(seed)] = str(out_dir / "matrix_summary.json")

    (output_root / "runs_index.json").write_text(
        json.dumps({"seeds": list(cfg.seeds), "runs": run_summaries}, indent=2)
    )


def aggregate_control_matrix(cfg: PipelineConfig) -> None:
    log("Aggregating control multiseed matrix")
    metrics = [
        "eval_encoder_r2",
        "eval_diffusion_r2",
        "eval_static_baseline_r2",
        "eval_reward_mean_delta_ls",
        "eval_reward_winrate_vs_static",
        "train_reward_mean_delta_ls",
        "train_reward_winrate_vs_static",
    ]
    modes = ["support", "text", "hybrid", "oracle"]
    input_root = cfg.runs_dir / "control_matrix_v2_multiseed"
    seed_dirs = sorted([d for d in input_root.glob("seed_*") if d.is_dir()])
    if not seed_dirs:
        raise RuntimeError(f"No seed directories found under {input_root}")

    by_mode: dict[str, dict[str, list[float]]] = {
        mode: {metric: [] for metric in metrics} for mode in modes
    }

    for seed_dir in seed_dirs:
        summary_path = seed_dir / "matrix_summary.json"
        data = json.loads(summary_path.read_text())
        rows = data.get("table", [])
        row_map = {row.get("encoding_mode"): row for row in rows if isinstance(row, dict)}
        for mode in modes:
            row = row_map.get(mode, {})
            for metric in metrics:
                value = row.get(metric)
                if value is not None:
                    by_mode[mode][metric].append(value)

    summary_table = []
    for mode in modes:
        row: dict[str, float | int | str | None] = {"encoding_mode": mode}
        for metric in metrics:
            values = by_mode[mode][metric]
            if not values:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_n"] = 0
            else:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                row[f"{metric}_n"] = len(values)
        summary_table.append(row)

    payload = {
        "input_root": str(input_root),
        "num_seeds": len(seed_dirs),
        "seeds": [d.name.replace("seed_", "") for d in seed_dirs],
        "metrics": metrics,
        "summary_table": summary_table,
    }

    out = input_root / "aggregate.json"
    out.write_text(json.dumps(payload, indent=2))


def run_direct_baseline_multiseed(
    cfg: PipelineConfig,
    *,
    protocol_suite: str = "held_out",
    output_root_name: str = "direct_baseline_v2_multiseed",
) -> None:
    log(f"Running direct baseline multiseed (all tasks, support mode, suite={protocol_suite})")
    output_root = cfg.runs_dir / output_root_name
    task_types = ["classification", "regression", "bandit_regression", "control"]
    for task_type in task_types:
        task_root = output_root / task_type
        task_root.mkdir(parents=True, exist_ok=True)
        for seed in cfg.seeds:
            output_dir = task_root / f"direct_support_seed{seed}"
            cmd = [
                cfg.python_exec,
                "-m",
                "hyperweights.direct_experiment",
                "--task-type",
                task_type,
                "--output-dir",
                str(output_dir),
                "--encoding-mode",
                "support",
                "--protocol-suite",
                protocol_suite,
                "--seed",
                str(seed),
                "--train-steps-stage1",
                str(cfg.train_steps_stage1),
                "--train-steps-stage2",
                str(cfg.train_steps_stage2),
                "--eval-batches",
                str(cfg.eval_batches),
                "--batch-size",
                str(cfg.batch_size),
                "--support-sweep-batches",
                str(cfg.support_sweep_batches),
                "--visualization-count",
                str(cfg.visualization_count),
                "--reward-audit-batches",
                str(cfg.reward_audit_batches),
                "--reward-audit-batch-size",
                str(cfg.reward_audit_batch_size),
            ]
            run_cmd(cmd, cwd=cfg.root)


def refresh_reports_and_plots(cfg: PipelineConfig) -> None:
    log("Refreshing reports, tables, and plots")
    exp_validation = validate_experiment_outputs(cfg)
    write_provenance_manifest(cfg, stage="pre-paper-artifacts", validation=exp_validation)
    _sync_diagnostic_plots_from_runs(cfg)

    run_cmd_capture_stdout(
        [cfg.python_exec, str(cfg.paper_dir / "audit_report.py")],
        cfg.paper_dir / "results" / "audit_report.txt",
        cwd=cfg.root,
    )
    run_cmd_capture_stdout(
        [cfg.python_exec, str(cfg.scripts_dir / "summarize_results.py")],
        cfg.paper_dir / "results" / "summary_report.txt",
        cwd=cfg.root,
    )
    run_cmd([cfg.python_exec, str(cfg.paper_dir / "tables" / "gen_tables.py")], cwd=cfg.root)
    run_cmd([cfg.python_exec, str(cfg.paper_dir / "figures" / "gen_plots.py")], cwd=cfg.root)

    artifact_validation = validate_paper_artifacts(cfg)
    write_provenance_manifest(cfg, stage="post-paper-artifacts", validation=artifact_validation)


def build_paper(cfg: PipelineConfig) -> None:
    log("Building paper.pdf")
    artifact_validation = validate_paper_artifacts(cfg)
    write_provenance_manifest(cfg, stage="pre-paper-build", validation=artifact_validation)

    paper = cfg.paper_dir
    run_cmd(["pdflatex", "-interaction=nonstopmode", "paper.tex"], cwd=paper)
    run_cmd(["bibtex", "paper"], cwd=paper)
    run_cmd(["pdflatex", "-interaction=nonstopmode", "paper.tex"], cwd=paper)
    run_cmd(["pdflatex", "-interaction=nonstopmode", "paper.tex"], cwd=paper)


def clean_paper_artifacts(cfg: PipelineConfig) -> None:
    log("Cleaning paper build artifacts")
    patterns = [
        "*.aux",
        "*.log",
        "*.out",
        "*.fdb_latexmk",
        "*.fls",
        "*.synctex.gz",
        "*.toc",
        "*.lof",
        "*.lot",
    ]
    for pattern in patterns:
        for path in cfg.paper_dir.glob(pattern):
            path.unlink(missing_ok=True)
    (cfg.paper_dir / "paper.pdf").unlink(missing_ok=True)
    for png in (cfg.paper_dir / "figures" / "plots").glob("*.png"):
        png.unlink(missing_ok=True)
    (cfg.paper_dir / "sections" / "annex.tex").unlink(missing_ok=True)


def full_refresh(cfg: PipelineConfig) -> None:
    clean_refresh_outputs(cfg)
    run_v2_benchmark(cfg)
    run_cross_family_benchmark(cfg)
    run_control_matrix_multiseed(cfg)
    aggregate_control_matrix(cfg)
    run_direct_baseline_multiseed(cfg)
    run_direct_baseline_multiseed(
        cfg,
        protocol_suite="cross_family",
        output_root_name="direct_baseline_cross_family_v2_multiseed",
    )
    exp_validation = validate_experiment_outputs(cfg)
    write_provenance_manifest(cfg, stage="post-experiments", validation=exp_validation)
    refresh_reports_and_plots(cfg)
    build_paper(cfg)
    log("Full refresh complete")
