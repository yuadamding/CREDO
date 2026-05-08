# Archive and Storage Policy

The portable bundle should stay compact. Large artifacts belong outside it.

## Keep in Git and Zip

- Source code under `package/`, `runners/`, and `analysis/`
- Shell launchers under `scripts/`
- Tests under `tests/`
- Environment file under `env/`
- Documentation under `docs/`

## Keep Out of Git and Zip

- `runs/`
- `outputs/`
- `results/`
- `models/` unless a compact export is intentionally being distributed
- `*.h5ad`
- `*.pt`, `*.pth`, `*.ckpt`
- generated `__pycache__/`, `.pytest_cache/`, and `*.egg-info/`

## Recommended Archive Location

Completed runs should be moved to an external archive, for example:

```text
CAPE/run_archives/
```

The current large run archive is:

```text
CAPE/run_archives/hnscc_cape_transfer_bundle_20260328_runs_20260507_112921/
```

That archive contains trained checkpoints and biology outputs for the 4-GPU
with-guide/shared-guide reproduction run. It is intentionally separate from the
portable package.

## Pruning Guidance

For long-term storage, preserve:

- `checkpoint_best_ema.pt`
- `checkpoint_best.pt`
- VAE artifacts needed for split-safe latent reuse
- `split_assignments.csv`
- `run_report.md`
- `software_versions.json`
- `cv_summary.csv` and `cv_summary.md`
- biology outputs under `biology/`

Only prune periodic epoch checkpoints after confirming the best checkpoints and
reports are present. Do not prune the source bundle or split metadata needed to
reproduce the summary tables.
