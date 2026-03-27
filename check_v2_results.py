import json
from pathlib import Path

for task_type in ['classification_v2', 'regression_v2', 'bandit_v2', 'control_v2']:
    summary_path = Path(f'runs/{task_type}/summary.json')
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        cfg = data.get('config', {})
        overall = data.get('overall', {})
        
        families = cfg.get('families') or []
        eval_fams = cfg.get('eval_families') or []
        
        # Find metric type
        if 'encoder_acc' in overall:
            metric = 'acc'
            enc = overall.get('encoder_acc', 0)
            diff = overall.get('diffusion_acc', 0)
            base = overall.get('baseline_acc', 0)
        else:
            metric = 'r2'
            enc = overall.get('encoder_r2', 0)
            diff = overall.get('diffusion_r2', 0)
            base = overall.get('baseline_r2', 0)
        
        name = task_type.replace('_v2', '').upper()
        print(f'{name:15} [{metric}]  train_fam={len(families):2}  eval_fam={len(eval_fams):2}')
        print(f'  encoder:  {enc:7.4f}  (gap: {enc-base:+.4f})')
        print(f'  diffusion:{diff:7.4f}  (gap: {diff-base:+.4f})')
        print(f'  baseline: {base:7.4f}')
        print()
