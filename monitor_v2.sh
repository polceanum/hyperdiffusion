#!/bin/bash
echo "Monitoring v2 experiments..."
for i in {1..180}; do
  echo "Check $i ($(date))"
  ps aux | grep "hyperdiffusion.experiment" | grep -v grep | wc -l
  if [ -f "runs/classification_v2/summary.json" ] && [ -f "runs/regression_v2/summary.json" ] && [ -f "runs/bandit_v2/summary.json" ] && [ -f "runs/control_v2/summary.json" ]; then
    echo "All experiments complete!"
    break
  fi
  sleep 300
done
