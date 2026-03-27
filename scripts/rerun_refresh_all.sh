#!/usr/bin/env bash
set -euo pipefail

PY=/usr/local/Caskroom/miniforge/base/envs/aurel/bin/python
ROOT=/Users/mike/Work/hyperdiffusion
cd "$ROOT"

mkdir -p runs

echo "[$(date)] Cleaning previous v2 benchmark outputs"
rm -rf runs/classification_v2 runs/regression_v2 runs/bandit_v2 runs/control_v2
rm -rf runs/control_matrix_v2_multiseed
rm -f runs/cls_v2.log runs/reg_v2.log runs/bandit_v2.log runs/ctrl_v2.log

echo "[$(date)] Running full v2 benchmark suite"
bash scripts/run_all_v2.sh

echo "[$(date)] Running control encoding multiseed matrix"
$PY scripts/run_control_matrix_multiseed.py \
  --output-root runs/control_matrix_v2_multiseed \
  --seeds 0 1 2 \
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

echo "[$(date)] Aggregating control multiseed matrix"
$PY scripts/aggregate_control_matrix_seeds.py \
  --input-root runs/control_matrix_v2_multiseed \
  --output runs/control_matrix_v2_multiseed/aggregate.json

echo "[$(date)] Collecting audit outputs"
$PY paper/audit_report.py > paper/results/audit_report.txt
$PY scripts/summarize_results.py > paper/results/summary_report.txt

echo "[$(date)] Refreshing tables and plots"
$PY paper/tables/gen_tables.py
$PY paper/figures/gen_plots.py

echo "[$(date)] Building paper"
cd paper
pdflatex -interaction=nonstopmode paper.tex >/tmp/paper_build1.log 2>&1
bibtex paper >/tmp/paper_bib.log 2>&1
pdflatex -interaction=nonstopmode paper.tex >/tmp/paper_build2.log 2>&1
pdflatex -interaction=nonstopmode paper.tex >/tmp/paper_build3.log 2>&1

echo "[$(date)] DONE"
