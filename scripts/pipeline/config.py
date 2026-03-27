from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_python() -> str:
    env_python = os.environ.get("PY")
    if env_python:
        return env_python
    preferred = "/usr/local/Caskroom/miniforge/base/envs/aurel/bin/python"
    return preferred if Path(preferred).exists() else sys.executable


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    python_exec: str
    train_steps_stage1: int = 1000
    train_steps_stage2: int = 1000
    eval_batches: int = 16
    batch_size: int = 32
    support_sweep_batches: int = 8
    diagnostic_samples: int = 4
    visualization_count: int = 4
    reward_audit_batches: int = 6
    reward_audit_batch_size: int = 16
    seeds: tuple[int, ...] = (0, 1, 2)

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def paper_dir(self) -> Path:
        return self.root / "paper"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @classmethod
    def from_root(cls, root: Path, python_exec: str | None = None) -> "PipelineConfig":
        return cls(root=root, python_exec=python_exec or _default_python())
