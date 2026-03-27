from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_cmd_capture_stdout(cmd: list[str], out_path: Path, cwd: Path | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_file:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=out_file, check=True)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def remove_paths(paths: list[Path]) -> None:
    for path in paths:
        remove_path(path)
