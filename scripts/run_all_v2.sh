#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/aurel/bin/python}

cd "$ROOT"
"$PY" -m scripts.pipeline.cli benchmark \
  --python "$PY" \
  --train-steps-stage1 1000 \
  --train-steps-stage2 1000 \
  --eval-batches 16 \
  --batch-size 32 \
  --support-sweep-batches 8 \
  --diagnostic-samples 4 \
  --visualization-count 0 \
  --reward-audit-batches 6 \
  --reward-audit-batch-size 16
