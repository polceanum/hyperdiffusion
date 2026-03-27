from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .runner import log

TASK_SUMMARY_DIRS = {
    "classification": "classification_v2",
    "regression": "regression_v2",
    "bandit_regression": "bandit_v2",
    "control": "control_v2",
}

CONTROL_MODES = ("support", "text", "hybrid", "oracle")


class ValidationError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"Missing required artifact: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc


def _require_overall_metric(overall: dict[str, Any], keys: list[str], artifact: Path) -> None:
    for key in keys:
        value = overall.get(key)
        if value is None:
            raise ValidationError(f"Missing metric '{key}' in {artifact}")
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Non-numeric metric '{key}' in {artifact}: {value}")


def _discover_seed_dirs(root: Path, prefix: str = "seed_") -> list[int]:
    discovered: list[int] = []
    for path in sorted(root.glob(f"{prefix}*")):
        if not path.is_dir():
            continue
        seed_str = path.name.replace(prefix, "", 1)
        if seed_str.isdigit():
            discovered.append(int(seed_str))
    return discovered


def _validate_task_summaries(cfg: PipelineConfig) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for task_type, run_dir in TASK_SUMMARY_DIRS.items():
        summary_path = cfg.runs_dir / run_dir / "summary.json"
        data = _load_json(summary_path)
        overall = data.get("overall")
        if not isinstance(overall, dict):
            raise ValidationError(f"Missing 'overall' object in {summary_path}")

        if task_type == "classification":
            _require_overall_metric(overall, ["encoder_acc", "diffusion_acc", "baseline_acc"], summary_path)
        else:
            _require_overall_metric(overall, ["encoder_r2", "diffusion_r2", "baseline_r2"], summary_path)

        if task_type == "control":
            reward = data.get("reward_audit")
            if not isinstance(reward, dict):
                raise ValidationError(f"Missing reward_audit block in {summary_path}")
            eval_block = reward.get("eval", {}).get("overall", {})
            _require_overall_metric(eval_block, ["mean_delta_ls", "win_rate_vs_static"], summary_path)

        evidence[task_type] = str(summary_path)
    return evidence


def _validate_control_matrix(cfg: PipelineConfig) -> dict[str, Any]:
    root = cfg.runs_dir / "control_matrix_v2_multiseed"
    discovered_seeds = _discover_seed_dirs(root)
    if not discovered_seeds:
        raise ValidationError(f"No matrix seed directories found under {root}")

    for seed in discovered_seeds:
        seed_dir = root / f"seed_{seed}"
        _load_json(seed_dir / "matrix_summary.json")

    aggregate_path = root / "aggregate.json"
    aggregate = _load_json(aggregate_path)
    if int(aggregate.get("num_seeds", -1)) != len(discovered_seeds):
        raise ValidationError(
            f"control matrix aggregate num_seeds mismatch in {aggregate_path}: "
            f"expected={len(discovered_seeds)} found={aggregate.get('num_seeds')}"
        )

    aggregate_seeds = aggregate.get("seeds", [])
    if isinstance(aggregate_seeds, list):
        try:
            parsed_aggregate = sorted(int(str(seed)) for seed in aggregate_seeds)
        except ValueError as exc:
            raise ValidationError(f"Malformed seed value in {aggregate_path}: {aggregate_seeds}") from exc
        if parsed_aggregate != sorted(discovered_seeds):
            raise ValidationError(
                f"control matrix aggregate seed list mismatch in {aggregate_path}: "
                f"discovered={sorted(discovered_seeds)} aggregate={parsed_aggregate}"
            )

    rows = aggregate.get("summary_table")
    if not isinstance(rows, list):
        raise ValidationError(f"Missing summary_table in {aggregate_path}")
    mode_to_row = {row.get("encoding_mode"): row for row in rows if isinstance(row, dict)}
    for mode in CONTROL_MODES:
        row = mode_to_row.get(mode)
        if row is None:
            raise ValidationError(f"Mode '{mode}' missing from {aggregate_path}")
        for key in [
            "eval_encoder_r2_mean",
            "eval_diffusion_r2_mean",
            "eval_static_baseline_r2_mean",
            "eval_reward_mean_delta_ls_mean",
            "eval_reward_winrate_vs_static_mean",
        ]:
            if row.get(key) is None:
                raise ValidationError(f"Metric '{key}' missing for mode={mode} in {aggregate_path}")

    return {
        "aggregate": str(aggregate_path),
        "seeds": discovered_seeds,
    }


def _validate_direct_baseline(cfg: PipelineConfig) -> dict[str, Any]:
    root = cfg.runs_dir / "direct_baseline_v2_multiseed"
    task_types = ["classification", "regression", "bandit_regression", "control"]
    seed_sets: list[set[int]] = []

    for task in task_types:
        task_root = root / task
        if not task_root.is_dir():
            raise ValidationError(f"Missing direct baseline task dir: {task_root}")
        task_seeds: set[int] = set()
        for path in task_root.glob("direct_support_seed*"):
            if not path.is_dir():
                continue
            suffix = path.name.replace("direct_support_seed", "", 1)
            if suffix.isdigit():
                task_seeds.add(int(suffix))
        if not task_seeds:
            raise ValidationError(f"No direct support seed directories found under {task_root}")
        seed_sets.append(task_seeds)

    common_seeds = sorted(set.intersection(*seed_sets))
    if not common_seeds:
        raise ValidationError("No common direct baseline seeds found across all task types")

    artifact_count = 0
    for task in task_types:
        task_root = root / task
        for seed in common_seeds:
            summary_path = task_root / f"direct_support_seed{seed}" / "summary.json"
            data = _load_json(summary_path)
            overall = data.get("overall", {})
            if not isinstance(overall, dict):
                raise ValidationError(f"Missing direct overall block in {summary_path}")
            if task == "classification":
                if overall.get("direct_acc") is None:
                    raise ValidationError(f"Missing direct classification accuracy in {summary_path}")
            else:
                if overall.get("direct_r2") is None:
                    raise ValidationError(f"Missing direct R2 in {summary_path}")
            if task == "control":
                reward_eval = data.get("reward_audit", {}).get("eval", {}).get("overall", {})
                if reward_eval.get("win_rate_vs_static") is None:
                    raise ValidationError(f"Missing direct control reward audit in {summary_path}")
            artifact_count += 1

    return {
        "root": str(root),
        "seeds": common_seeds,
        "summary_files": artifact_count,
    }


def _validate_paper_artifacts(cfg: PipelineConfig) -> dict[str, str]:
    required = {
        "audit_report": cfg.paper_dir / "results" / "audit_report.txt",
        "summary_report": cfg.paper_dir / "results" / "summary_report.txt",
        "latest_json": cfg.paper_dir / "results" / "latest.json",
        "tables": cfg.paper_dir / "tables" / "main_results.tex",
        "annex": cfg.paper_dir / "sections" / "annex.tex",
    }
    for label, path in required.items():
        if not path.exists() or path.stat().st_size == 0:
            raise ValidationError(f"Missing/empty paper artifact '{label}': {path}")

    plot_dir = cfg.paper_dir / "figures" / "plots"
    pngs = sorted(plot_dir.glob("*.png"))
    if len(pngs) < 4:
        raise ValidationError(f"Expected curated plot outputs in {plot_dir}, found {len(pngs)}")

    return {label: str(path) for label, path in required.items()}


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def write_provenance_manifest(
    cfg: PipelineConfig,
    *,
    stage: str,
    validation: dict[str, Any],
) -> Path:
    manifest_dir = cfg.runs_dir / "pipeline"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "provenance_manifest.json"

    cfg_payload = asdict(cfg)
    cfg_payload["root"] = str(cfg.root)
    cfg_hash = sha256(json.dumps(cfg_payload, sort_keys=True).encode("utf-8")).hexdigest()

    payload = {
        "stage": stage,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(cfg.root),
        "config": cfg_payload,
        "config_sha256": cfg_hash,
        "validation": validation,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    log(f"Wrote provenance manifest: {manifest_path}")
    return manifest_path


def validate_experiment_outputs(cfg: PipelineConfig) -> dict[str, Any]:
    task_evidence = _validate_task_summaries(cfg)
    matrix_evidence = _validate_control_matrix(cfg)
    direct_evidence = _validate_direct_baseline(cfg)
    validation = {
        "tasks": task_evidence,
        "control_matrix": matrix_evidence,
        "direct_baseline": direct_evidence,
    }
    log("Experiment output validation passed")
    return validation


def validate_paper_artifacts(cfg: PipelineConfig) -> dict[str, Any]:
    artifacts = _validate_paper_artifacts(cfg)
    validation = {"paper_artifacts": artifacts}
    log("Paper artifact validation passed")
    return validation
