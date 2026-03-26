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
  --eval-batches 16 --batch-size 32 --visualization-count 4 \
  > runs/cls_v2.log 2>&1 &
CLS_PID=$!

echo "[$(date)] Starting regression..."
$PY -m hyperdiffusion.experiment \
  --task-type regression --output-dir runs/regression_v2 \
  --train-steps-stage1 1000 --train-steps-stage2 1000 \
  --eval-batches 16 --batch-size 32 --visualization-count 4 \
  > runs/reg_v2.log 2>&1 &
REG_PID=$!

echo "[$(date)] Starting bandit..."
$PY -m hyperdiffusion.experiment \
  --task-type bandit_regression --output-dir runs/bandit_v2 \
  --train-steps-stage1 1000 --train-steps-stage2 1000 \
  --eval-batches 16 --batch-size 32 --visualization-count 4 \
  > runs/bandit_v2.log 2>&1 &
BANDIT_PID=$!

echo "[$(date)] Starting control..."
$PY -m hyperdiffusion.experiment \
  --task-type control --output-dir runs/control_v2 \
  --train-steps-stage1 1000 --train-steps-stage2 1000 \
  --eval-batches 16 --batch-size 32 --visualization-count 4 \
  --reward-audit-batches 8 --reward-audit-batch-size 16 \
  > runs/ctrl_v2.log 2>&1 &
CTRL_PID=$!

echo "PIDs: cls=$CLS_PID reg=$REG_PID bandit=$BANDIT_PID ctrl=$CTRL_PID"
echo "Logs: runs/cls_v2.log  runs/reg_v2.log  runs/bandit_v2.log  runs/ctrl_v2.log"

wait $CLS_PID && echo "[$(date)] cls DONE"  || echo "[$(date)] cls FAILED" &
wait $REG_PID && echo "[$(date)] reg DONE"  || echo "[$(date)] reg FAILED" &
wait $BANDIT_PID && echo "[$(date)] bandit DONE" || echo "[$(date)] bandit FAILED" &
wait $CTRL_PID && echo "[$(date)] ctrl DONE"  || echo "[$(date)] ctrl FAILED" &
wait
echo "[$(date)] ALL DONE"
