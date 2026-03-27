from __future__ import annotations

import subprocess
from pathlib import Path

from .config import PipelineConfig
from .runner import log, remove_paths, run_cmd, run_cmd_capture_stdout


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
            cfg.runs_dir / "cls_v2.log",
            cfg.runs_dir / "reg_v2.log",
            cfg.runs_dir / "bandit_v2.log",
            cfg.runs_dir / "ctrl_v2.log",
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
            "hyperdiffusion.experiment",
            "--task-type",
            task_type,
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
            log(f"{task_type} FAILED")
        else:
            log(f"{task_type} DONE")
    if failed:
        raise RuntimeError(f"Benchmark tasks failed: {', '.join(failed)}")


def run_control_matrix_multiseed(cfg: PipelineConfig) -> None:
    log("Running control encoding matrix (multiseed)")
    cmd = [
        cfg.python_exec,
        str(cfg.scripts_dir / "run_control_matrix_multiseed.py"),
        "--output-root",
        str(cfg.runs_dir / "control_matrix_v2_multiseed"),
        "--seeds",
        *[str(s) for s in cfg.seeds],
        "--python",
        cfg.python_exec,
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
    ]
    run_cmd(cmd, cwd=cfg.root)


def aggregate_control_matrix(cfg: PipelineConfig) -> None:
    log("Aggregating control multiseed matrix")
    cmd = [
        cfg.python_exec,
        str(cfg.scripts_dir / "aggregate_control_matrix_seeds.py"),
        "--input-root",
        str(cfg.runs_dir / "control_matrix_v2_multiseed"),
        "--output",
        str(cfg.runs_dir / "control_matrix_v2_multiseed" / "aggregate.json"),
    ]
    run_cmd(cmd, cwd=cfg.root)


def run_direct_baseline_multiseed(cfg: PipelineConfig) -> None:
    log("Running direct baseline multiseed (all tasks, support mode)")
    cmd = [
        cfg.python_exec,
        str(cfg.scripts_dir / "run_direct_baseline_multiseed.py"),
        "--python",
        cfg.python_exec,
        "--output-root",
        str(cfg.runs_dir / "direct_baseline_v2_multiseed"),
        "--task-types",
        "classification",
        "regression",
        "bandit_regression",
        "control",
        "--modes",
        "support",
        "--seeds",
        *[str(s) for s in cfg.seeds],
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


def build_paper(cfg: PipelineConfig) -> None:
    log("Building paper.pdf")
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
    run_control_matrix_multiseed(cfg)
    aggregate_control_matrix(cfg)
    run_direct_baseline_multiseed(cfg)
    refresh_reports_and_plots(cfg)
    build_paper(cfg)
    log("Full refresh complete")
