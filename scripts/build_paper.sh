#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/aurel/bin/python}

cd "$ROOT"

if [[ "${1:-}" == "clean" ]]; then
  "$PY" -m scripts.pipeline.cli clean-paper --python "$PY"
  echo "[paper] cleaned"
  exit 0
fi

"$PY" -m scripts.pipeline.cli refresh-artifacts --python "$PY"
"$PY" -m scripts.pipeline.cli build-paper --python "$PY"

echo "[paper] built paper.pdf"
