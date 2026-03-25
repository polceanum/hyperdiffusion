# hyperdiffusion

## New features

1. Learned selector over diffusion samples (`CandidateSelector`)
2. Regression uncertainty-quality diagnostics
3. Adaptive bandit regression mode (`bandit_regression`)
4. Paper pipeline scripts now include selector and uncertainty metrics

## CLI usage

Run classification (default):
```bash
python -m hyperdiffusion.experiment --task-type classification --output-dir runs/classification
```

Run regression with selector:
```bash
python -m hyperdiffusion.experiment --task-type regression --selector-enabled --selector-num-samples 8 --output-dir runs/regression
```

Run adaptive bandit regression:
```bash
python -m hyperdiffusion.experiment --task-type bandit_regression --output-dir runs/bandit
```

### New flags
- `--selector-enabled` (bool): enable learned candidate selector training and selection
- `--selector-num-samples` (int): number of diffusion candidates for selector/best-of-k computation
- `--selector-hidden` (int): hidden dim for selector network
- `--selector-lr` (float): selector learning rate

## Paper pipeline

1. `bash scripts/build_paper.sh clean` to clear artifacts
2. `bash scripts/build_paper.sh` to generate `paper.pdf`

New output stats exported to `paper/results/latest.json`:
- `selector_acc` / `selector_r2`
- `selector_loss`
- `uncertainty_*` regression diagnostics
- adaptation curves via `support_size_sweep`
