from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .tasks import (
    BRIDGE_FAMILIES,
    DEFAULT_BANDIT_EVAL_FAMILIES,
    DEFAULT_BANDIT_TRAIN_FAMILIES,
    DEFAULT_CONTROL_EVAL_FAMILIES,
    DEFAULT_CONTROL_TRAIN_FAMILIES,
    DEFAULT_REGRESSION_EVAL_FAMILIES,
    DEFAULT_REGRESSION_TRAIN_FAMILIES,
    DEFAULT_TRAIN_FAMILIES,
)


PROTOCOL_SUITES = ("held_out", "cross_family")


@dataclass(frozen=True)
class ProtocolSplit:
    suite: str
    task_type: str
    train_families: List[str]
    eval_families: List[str]
    strict_ood: bool

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "task_type": self.task_type,
            "strict_ood": self.strict_ood,
            "train_families": list(self.train_families),
            "eval_families": list(self.eval_families),
            "train_family_count": len(self.train_families),
            "eval_family_count": len(self.eval_families),
            "overlap": sorted(set(self.train_families) & set(self.eval_families)),
        }


def _dedupe(families: List[str] | None) -> List[str]:
    return list(dict.fromkeys(families or []))


def _all_families_for_task(task_type: str) -> List[str]:
    if task_type == "classification":
        return _dedupe(DEFAULT_TRAIN_FAMILIES + BRIDGE_FAMILIES)
    if task_type == "regression":
        return _dedupe(DEFAULT_REGRESSION_TRAIN_FAMILIES + DEFAULT_REGRESSION_EVAL_FAMILIES)
    if task_type == "bandit_regression":
        return _dedupe(DEFAULT_BANDIT_TRAIN_FAMILIES + DEFAULT_BANDIT_EVAL_FAMILIES)
    if task_type == "control":
        return _dedupe(DEFAULT_CONTROL_TRAIN_FAMILIES + DEFAULT_CONTROL_EVAL_FAMILIES)
    raise ValueError(f"Unknown task_type: {task_type}")


def _default_held_out(task_type: str) -> tuple[List[str], List[str]]:
    if task_type == "classification":
        return list(DEFAULT_TRAIN_FAMILIES), list(BRIDGE_FAMILIES)
    if task_type == "regression":
        return list(DEFAULT_REGRESSION_TRAIN_FAMILIES), list(DEFAULT_REGRESSION_EVAL_FAMILIES)
    if task_type == "bandit_regression":
        return list(DEFAULT_BANDIT_TRAIN_FAMILIES), list(DEFAULT_BANDIT_EVAL_FAMILIES)
    if task_type == "control":
        return list(DEFAULT_CONTROL_TRAIN_FAMILIES), list(DEFAULT_CONTROL_EVAL_FAMILIES)
    raise ValueError(f"Unknown task_type: {task_type}")


def _default_cross_family(task_type: str) -> tuple[List[str], List[str]]:
    all_families = sorted(_all_families_for_task(task_type))
    eval_families = [name for idx, name in enumerate(all_families) if idx % 3 == 0]
    train_families = [name for name in all_families if name not in set(eval_families)]
    if not train_families or not eval_families:
        raise ValueError(f"Invalid cross_family partition for task_type={task_type}")
    return train_families, eval_families


def resolve_protocol_split(
    *,
    task_type: str,
    suite: str = "held_out",
    train_families: List[str] | None = None,
    eval_families: List[str] | None = None,
    strict_ood: bool = True,
) -> ProtocolSplit:
    if suite not in PROTOCOL_SUITES:
        raise ValueError(f"Unknown protocol suite: {suite}. Expected one of {PROTOCOL_SUITES}")

    if suite == "held_out":
        default_train, default_eval = _default_held_out(task_type)
    else:
        default_train, default_eval = _default_cross_family(task_type)

    resolved_train = _dedupe(train_families) or default_train
    resolved_eval = _dedupe(eval_families) or default_eval

    if not resolved_train:
        raise ValueError("train_families resolved to empty")
    if not resolved_eval:
        raise ValueError("eval_families resolved to empty")

    overlap = set(resolved_train) & set(resolved_eval)
    if strict_ood and overlap:
        raise ValueError(
            "Strict OOD protocol violation: train/eval family overlap detected: "
            f"{sorted(overlap)}"
        )

    return ProtocolSplit(
        suite=suite,
        task_type=task_type,
        train_families=resolved_train,
        eval_families=resolved_eval,
        strict_ood=strict_ood,
    )
