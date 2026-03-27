#!/usr/bin/env bash
set -e
PY=/usr/local/Caskroom/miniforge/base/envs/aurel/bin/python
ROOT=/Users/mike/Work/hyperdiffusion
cd "$ROOT"
mkdir -p runs

echo "[$(date)] Starting classification..."
$PY -m hyperdiffusion.experiment \
  --task-type classification --output-dir runs/classification_v2 \
  --train-steps-stage1 1000 --train-steps-stage2 1000 \
  --eval-batches 16 --batch-size 32 --visualization-count 0 \
  > runs/cls_v2.log 2>&1 &
CLS_PID=$!

echo "[$(date)] Starting regression..."
$PY -m hyperdiffusion.experiment \
  --task-type regression --output-dir runs/regression_v2 \
  --train-steps-stage1 1000 --train-steps-stage2 1000 \
  --eval-batches 16 --batch-size 32 --visualization-count 0 \
  > runs/reg_v2.log 2>&1 &
REG_PID=$!

echo "[$(date)] Starting bandit..."
$PY -m hyperdiffusion.experiment \
  --task-type bandit_regression --output-dir runs/bandit_v2 \
  --train-steps-stage1 1000 --train-steps-stage2 1000 \
  --eval-batches 16 --batch-size 32 --visualization-count 0 \
  > runs/bandit_v2.log 2>&1 &
BANDIT_PID=$!

echo "[$(date)] Starting control..."
$PY -m hyperdiffusion.experiment \
  --task-type control --output-dir runs/control_v2 \
  --train-steps-stage1 1000 --train-steps-stage2 1000 \
  --eval-batches 16 --batch-size 32 --visualization-count 0 \
  --reward-audit-batches 8 --reward-audit-batch-size 16 \
  > runs/ctrl_v2.log 2>&1 &
CTRL_PID=$!

echo "PIDs: cls=$CLS_PID reg=$REG_PID bandit=$BANDIT_PID ctrl=$CTRL_PID"
echo "Logs: runs/cls_v2.log  runs/reg_v2.log  runs/bandit_v2.log  runs/ctrl_v2.log"

FAIL=0

if wait $CLS_PID; then
  echo "[$(date)] cls DONE"
else
  echo "[$(date)] cls FAILED"
  FAIL=1
fi

if wait $REG_PID; then
  echo "[$(date)] reg DONE"
else
  echo "[$(date)] reg FAILED"
  FAIL=1
fi

if wait $BANDIT_PID; then
  echo "[$(date)] bandit DONE"
else
  echo "[$(date)] bandit FAILED"
  FAIL=1
fi

if wait $CTRL_PID; then
  echo "[$(date)] ctrl DONE"
else
  echo "[$(date)] ctrl FAILED"
  FAIL=1
fi

echo "[$(date)] ALL DONE"

exit $FAIL
