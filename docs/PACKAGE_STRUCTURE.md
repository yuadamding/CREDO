# Package Structure

This bundle is meant to be small enough to move and inspect, while still being
complete enough to install, train, summarize, and extract HNSCC biology.

## Top-Level Directories

| Path | Contents | Keep in portable zip |
| --- | --- | --- |
| `package/` | Installable `credo` package, with historical implementation package `cape`. | Yes |
| `runners/` | Python training and CV-summary entry points. | Yes |
| `analysis/` | Biology extraction and signature/post-hoc analysis scripts. | Yes |
| `scripts/` | Shell launchers and shared shell utilities. | Yes |
| `tests/` | Smoke and regression tests. | Yes |
| `env/` | Minimal conda environment file. | Yes |
| `docs/` | Human-readable structure, runbook, biology workflow, and archive policy. | Yes |
| `runs/` | Training outputs, summaries, checkpoints, latent artifacts, monitor logs. | No |
| `outputs/` | Legacy or ad hoc model/data outputs. | No |
| `results/` | Ad hoc analysis outputs. | No |
| `models/` | Reusable exported model artifacts. | No |

## Python Package Layout

```text
package/src/
  cape/
    config/      Pydantic configuration schema
    data/        HNSCC AnnData loading, filtering, measure construction
    eval/        HNSCC endpoint/state evaluation helpers
    losses/      endpoint, weak-form, count, and regularization losses
    models/      CREDO dynamics, context, embeddings, simulator, VAE
    training/    trainer
  credo/         import alias for cape
```

`cape` remains the historical implementation namespace. `credo` is the public
package name and import alias, so both styles are valid:

```python
from cape.models.full_model import FullDynamicsModel
from credo.training.trainer import Trainer
```

## Script Boundaries

Shell launchers stay in `scripts/` rather than being moved into nested folders
because existing run logs and user commands reference these paths. Shared logic
lives in:

- `scripts/_conda_init.sh`
- `scripts/_run_hnscc_cv.sh`

Training Python entry points stay in `runners/`; post-training biological
analysis stays in `analysis/`.

## Output Boundaries

The portable bundle should not carry trained outputs. Use:

- `runs/` for CV/training runs
- `outputs/` only for legacy/ad hoc outputs that should be archived externally
- `runs/<run>/biology/` for post-hoc biology tables
- `CAPE/run_archives/` or another external archive for completed runs
- `models/` only for deliberately exported compact model artifacts

The `.gitignore` excludes these output paths plus Python caches.
