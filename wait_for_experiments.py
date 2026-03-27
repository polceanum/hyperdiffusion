#!/usr/bin/env python3
import time
import json
from pathlib import Path

experiments = ['classification_v2', 'regression_v2', 'bandit_v2', 'control_v2']

while True:
    completed = []
    for exp in experiments:
        summary_path = Path(f'runs/{exp}/summary.json')
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
                if 'overall' in data:
                    completed.append(exp)
            except:
                pass
    
    print(f"Progress: {len(completed)}/4 experiments complete")
    for exp in experiments:
        status = "✓ DONE" if exp in completed else "• Running"
        print(f"  {exp}: {status}")
    
    if len(completed) == 4:
        print("\nAll experiments completed!")
        break
    
    time.sleep(120)
    print()
