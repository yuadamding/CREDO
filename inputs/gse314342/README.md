# GSE314342 CREDO trajectory

This adapter models one continuous `Rest -> Stim8hr -> Stim48hr` trajectory.
Finite-measure views are donor-guide pairs; guides targeting the same curated
gene share one model embedding. NTC guides remain separate views and share the
`__NTC__` reference embedding. Dynamic counterfactuals start from the observed,
already perturbed Rest population, so they do not include pre-Rest effects.

## Data status

The complete release is mirrored at
`/home/yding1995/opscc_sc/data/GSE314342`. It includes 12 processed
donor-condition H5AD files (33,610,471 captured cells), derived release
products, the GEO raw archive and companion metadata, and the authors' analysis
code. See the [data inventory](../../../data/GSE314342/README.md) for locations,
sizes, checks, and cohort counts.

The generated CREDO input is
`/home/yding1995/opscc_sc/inputs/GSE314342/gse314342_credo_support.h5ad`.
Its [input README](../../../inputs/GSE314342/README.md) records QC, support,
mass, latent-model, and provenance details. The file contains only compact
32-dimensional supports; it does not duplicate the downloaded expression
matrix. The latent-only file has zero expression variables and records its
trajectory contract under `uns["credo_support_schema_version"]`.

The raw GEO lane title says `24 hr stim`, while the processed release calls the
late condition `Stim48hr`. The recorded decision is
[`late_time_resolution.json`](late_time_resolution.json); `--final` refuses an
unresolved provenance record.

## Build compact supports

```bash
cd /home/yding1995/opscc_sc
/home/yding1995/miniforge3/envs/cape-hnscc/bin/python \
  scripts/build_gse314342_credo_input.py \
  --resume --source-chunk-rows 50000 --vae-fit-chunk-rows 10000
```

The builder scans one source file at a time, selects genes from the downloaded
pseudobulk release, fits a donor-condition-balanced expression VAE, encodes the
cell supports out of core, and writes the final file under the workspace
`inputs/GSE314342` directory. It also writes full unsampled guide counts and
smoothed within-donor masses, donor-time count blocks, a measure manifest, and
build provenance. `--resume` reuses the completed scan, VAE, and any retained
valid shards; final builds remove shards unless `--keep-shards` is supplied.

QC excludes low-quality, unassigned, and multi-guide cells using the release's
authoritative guide-assignment field. It does not select cells or guides using
downstream differential expression.

Generate and run the deterministic D3 numerical pilot before a full fit:

```bash
cd /home/yding1995/opscc_sc/CREDO
python scripts/make_gse314342_pilot.py --overwrite

python ../scripts/run_credo_gse314342.py --preset pilot
```

The pilot contains 122 complete D3 donor-guide keys and 15,417 support atoms.
It is a smoke test only; selection uses stable hashes of annotations, not
expression outcomes, and preserves rather than renormalizes full-cohort masses.

Run the non-physical Rest baseline-priming companion analysis by selecting Rest
atoms from the combined support with unit mass:

```bash
python runners/run_credo_single_time.py \
  --data-path /home/yding1995/opscc_sc/inputs/GSE314342/gse314342_credo_support.h5ad \
  --output-dir runs/gse314342/rest_priming_seed0 \
  --latent-key X_credo \
  --perturbation-col guide_id --guide-col guide_id \
  --target-gene-col target_gene --control-col is_control \
  --sample-col sample_id --batch-col 10xrun_id \
  --snapshot-col time_label --snapshot-value Rest \
  --embedding-level target_gene --view-key-level sample_guide \
  --view-level view --mass-mode unit_mass --ecology-off
```

## Fit and continue

The `.yaml` presets are JSON-compatible YAML consumed by `--config`; later CLI
flags override preset values.

```bash
python runners/run_credo_gse314342.py \
  --config configs/gse314342/intrinsic.yaml \
  --data-path /home/yding1995/opscc_sc/inputs/GSE314342/gse314342_credo_support.h5ad \
  --output-dir runs/gse314342/intrinsic_seed0 \
  --seed 0

python runners/run_credo_gse314342.py \
  --data-path /home/yding1995/opscc_sc/inputs/GSE314342/gse314342_credo_support.h5ad \
  --count-table /home/yding1995/opscc_sc/inputs/GSE314342/guide_count_blocks.csv \
  --output-dir runs/gse314342/count_seed0 \
  --checkpoint runs/gse314342/intrinsic_seed0/checkpoint_best_ema.pt \
  --stage reaction --lambda-count 0.1 --sinkhorn-tau 0.1

python runners/run_credo_gse314342.py \
  --config configs/gse314342/context.yaml \
  --data-path /home/yding1995/opscc_sc/inputs/GSE314342/gse314342_credo_support.h5ad \
  --output-dir runs/gse314342/context_seed0 \
  --checkpoint runs/gse314342/count_seed0/checkpoint_best_ema.pt
```

Run the leakage-free intermediate-time check as a separate fit. `Stim8hr`
remains on the rollout grid and is reported in
`evaluation_only_metrics_by_key_time.csv`, but its geometry, mass, and count
targets are excluded from optimization and checkpoint selection:

```bash
python runners/run_credo_gse314342.py \
  --config configs/gse314342/intrinsic.yaml \
  --data-path /home/yding1995/opscc_sc/inputs/GSE314342/gse314342_credo_support.h5ad \
  --output-dir runs/gse314342/heldout_8h_seed0 \
  --evaluation-only-labels Stim8hr --seed 0
```

Leave-one-guide-out runs use `--validation-guide-ids` or
`--guide-cv-folds/--guide-cv-fold-index`; donor runs use
`--validation-sample-ids` or `--cv-folds/--cv-fold-index`. In grouped runs,
held-out Rest measures remain in the detached donor context catalog while
their stimulated endpoints remain validation-only.

On the selected final run, add `--export-counterfactuals` to write
`counterfactual_metrics_by_key_time.csv`. Controls are included for NTC-null
calibration; `--counterfactual-exclude-controls` disables them. Grouped models
use the checkpointed full-background context bank rather than making the focal
guide the entire ecosystem.

Consolidate the dynamic and Rest result families with:

```bash
python analysis/extract_gse314342_effects.py \
  --predicted runs/gse314342/final/predicted_metrics_by_key_time.csv \
  --counterfactual runs/gse314342/final/counterfactual_metrics_by_key_time.csv \
  --measure-manifest /home/yding1995/opscc_sc/inputs/GSE314342/measure_manifest.csv \
  --rest-effects runs/gse314342/rest_priming_seed0/single_time_effects.csv \
  --output-dir results/gse314342
```

Only retain grouped-context or ecological models when they improve held-out
guide, donor, or 8-hour prediction over the intrinsic model. Claim-grade output
also requires NTC and ineffective-guide nulls, leave-one-guide and
leave-one-donor evaluation, numerical convergence, and external bulk
validation. `analysis/extract_gse314342_effects.py` keeps `claim_ready=false`
until those gates are supplied.
