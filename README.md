# HNSCC CREDO Transfer Bundle

Compact, portable CREDO workflow for the HNSCC P4/P60 Perturb-seq transfer
analysis. The bundle contains source code, launchers, tests, and biology
post-processing utilities. It intentionally does not contain `.h5ad` data,
trained checkpoints, or run outputs.

## Bundle Layout

| Path | Purpose |
| --- | --- |
| `package/` | Installable Python package. Historical imports live under `cape`; `credo` is the compatibility alias. |
| `runners/` | Python entry points for training and CV summaries. |
| `analysis/` | Post-training HNSCC biology extraction, signature scoring, counterfactual merging, and bulk projection. |
| `scripts/` | Shell launchers for installation, H100 training, guide-vs-shared comparisons, and biology extraction. |
| `tests/` | Smoke and regression tests for package imports, weak-form math, expression loading, summaries, and biology tables. |
| `env/` | Minimal conda environment definition. |
| `docs/` | Detailed runbook, biology workflow, package structure, and archive policy. |

Generated outputs belong in `runs/`, `outputs/`, `results/`, or `models/`;
those paths are ignored by git and should be archived outside the portable
bundle.

## Required Data

Place the Renz HNSCC AnnData file beside this directory:

```bash
../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad
```

Override with `DATA_PATH=/path/to/file.h5ad` when needed.

## Install

```bash
CONDA_BIN=/rsrch8/home/bcb/$USER/miniforge3/bin/conda \
bash scripts/install_bundle.sh cape-hnscc
```

The installer defaults to the PyTorch CUDA 12.4 wheel index because the common
H100/Jupyter driver stack reports CUDA driver 12.4. On a newer-driver node, set
`TORCH_INDEX_URL`; to replace an incompatible existing Torch build, set
`INSTALL_TORCH=1`.

If the environment already exists and setup verification passes, refresh only
the editable package:

```bash
python -m pip install --no-cache-dir -e package
python scripts/verify_setup.py --data-path ../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad
```

## Main Commands

Run the current best with-guide versus shared-guide 4-fold comparison:

```bash
CONDA_BIN=/rsrch8/home/bcb/$USER/miniforge3/bin/conda \
GPU_LIST=0,1,2,3 \
bash scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh
```

Extract mechanistic biology from an existing trained with-guide/shared-guide run:

```bash
COMPARE_ROOT=/rsrch8/home/bcb/yding4/perturbseq/hnscc_cape_transfer_bundle_20260328/runs/hnscc_random_h100_heavy_f_best_ur01_repro_4gpu_20260506_224704 \
DATA_PATH=../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad \
RUN_COUNTERFACTUALS=1 CF_CONTEXT_CLAMPED=1 CF_PARTICLES=512 CF_STEPS=28 \
CF_DEVICE=auto \
bash scripts/run_hnscc_biological_findings.sh
```

Use `CF_DEVICE=cpu` if the active Python/Torch install is not compatible with
the local NVIDIA driver. Use `CF_DEVICE=cuda` only when this succeeds in the
`cape-hnscc` environment:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Documentation

- [Package Structure](docs/PACKAGE_STRUCTURE.md)
- [H100 Runbook](docs/H100_RUNBOOK.md)
- [Biology Extraction](docs/BIOLOGY_EXTRACTION.md)
- [Archive and Storage Policy](docs/ARCHIVE_AND_STORAGE.md)

## Implementation Notes

- New user-facing scripts and reports use the CREDO name.
- Legacy `cape` imports remain valid; `credo` aliases the same implementation.
- Every new training run writes `software_versions.json` with package version,
  command line, Python/PyTorch/CUDA versions, and data-file metadata. Set
  `CREDO_DATA_SHA256` to record a precomputed full H5AD hash.
- The endpoint metric named UOT in summaries is a normalized Sinkhorn geometry
  proxy plus log-mass penalty, not a full KL-relaxed finite-measure UOT objective.
- Ecological context is computed from rollout states, weights, and masses; guide
  identity affects context indirectly through dynamics.
- The built-in Lrp1 proxy signatures include the broad `lrp1_proxy_module` plus
  epithelial TSK/pEMT, CAF/ECM, and inflammatory/myeloid split modules for
  confounding checks. They are not decoded perturbation-specific CREDO residual
  modules.
