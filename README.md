# hyperweights

## New features

1. Learned selector over diffusion samples (`CandidateSelector`)
2. Regression uncertainty-quality diagnostics
3. Adaptive bandit regression mode (`bandit_regression`)
4. Paper pipeline scripts now include selector and uncertainty metrics

## CLI usage

Run classification (default):
```bash
python -m hyperweights.experiment --task-type classification --output-dir runs/classification
```

Run regression with selector:
```bash
python -m hyperweights.experiment --task-type regression --selector-enabled --selector-num-samples 8 --output-dir runs/regression
```

Run adaptive bandit regression:
```bash
python -m hyperweights.experiment --task-type bandit_regression --output-dir runs/bandit
```

### New flags
- `--selector-enabled` (bool): enable learned candidate selector training and selection
- `--selector-num-samples` (int): number of diffusion candidates for selector/best-of-k computation
- `--selector-hidden` (int): hidden dim for selector network
- `--selector-lr` (float): selector learning rate
- `--protocol-suite` (`held_out` or `cross_family`): unified split protocol used by both `hyperweights.experiment` and `hyperweights.direct_experiment`
- `--allow-eval-train-overlap`: disables strict disjoint split enforcement (not recommended)

### Protocol guarantees (scientific hygiene)

- By default (`strict_ood=True`), train and eval family sets must be disjoint; overlap raises an error.
- Both variants and baselines use the exact same train/eval family split resolver (`hyperweights.protocol`).
- `held_out` suite: train on core families, evaluate on unseen families.
- `cross_family` suite: deterministic disjoint partition across the full family set for each task.

## Paper pipeline

Preferred modular entrypoints:

1. `python -m scripts.pipeline.cli clean-paper` to clear paper artifacts
2. `python -m scripts.pipeline.cli validate` to enforce experiment/paper artifact integrity checks and write provenance
3. `python -m scripts.pipeline.cli refresh-artifacts` to regenerate tables/plots/reports (gated by experiment validation)
4. `python -m scripts.pipeline.cli build-paper` to compile `paper.pdf` (gated by paper artifact validation)
5. `python -m scripts.pipeline.cli full-refresh` to rerun full benchmark + matrix + direct baseline + paper
6. `python -m scripts.pipeline.cli benchmark` (alias to `full-refresh`) to ensure review-ready paper outputs
7. `python -m scripts.pipeline.cli benchmark-cross-family` (alias to `full-refresh`) to ensure review-ready paper outputs while including cross-family runs

Validation/provenance output:

- `runs/pipeline/provenance_manifest.json` stores config hash, git commit, timestamp, and validation evidence for each stage.

Legacy wrappers (still supported):

1. `bash scripts/build_paper.sh clean`
2. `bash scripts/build_paper.sh`
3. `bash scripts/rerun_refresh_all.sh`

New output stats exported to `paper/results/latest.json`:
- `selector_acc` / `selector_r2`
- `selector_loss`
- `uncertainty_*` regression diagnostics
- adaptation curves via `support_size_sweep`
