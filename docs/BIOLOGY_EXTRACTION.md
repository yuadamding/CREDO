# Biology Extraction

Use biology extraction after trained with-guide and shared-guide runs exist.
This step should rely on trained model checkpoints; it is not another broad
model-comparison run.

## Existing Trained Run

For the archived 4-GPU reproduction run:

```bash
COMPARE_ROOT=/rsrch8/home/bcb/yding4/perturbseq/hnscc_cape_transfer_bundle_20260328/runs/hnscc_random_h100_heavy_f_best_ur01_repro_4gpu_20260506_224704 \
DATA_PATH=../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad \
RUN_COUNTERFACTUALS=1 CF_CONTEXT_CLAMPED=1 CF_PARTICLES=512 CF_STEPS=28 \
CF_DEVICE=auto SEED=0 \
bash scripts/run_hnscc_biological_findings.sh
```

`CF_SEED` defaults to `SEED`, so rerunning the same trained checkpoints with
the same `SEED`, particles, steps, and device gives the same counterfactual
initial particles/noise stream.

If local Torch cannot use the available NVIDIA driver, use:

```bash
CF_DEVICE=cpu bash scripts/run_hnscc_biological_findings.sh
```

## Outputs

With `RUN_COUNTERFACTUALS=1`, the workflow writes:

```text
biology/signatures/signature_group_scores.csv
biology/counterfactual/counterfactual_biology_effects.csv
biology/biological_effects_per_perturbation.csv
biology/biological_effects_per_perturbation_v2.csv
```

The v2 table is the main mechanistic table. It merges endpoint CV, observed
P4-to-P60 signature deltas, shared-guide null gaps, and same-start
factual-versus-reference counterfactual effects when available.

## Built-In Signatures

The HNSCC signature scorer includes:

| Signature | Intended readout |
| --- | --- |
| `tnf_expansion` | AP-1/NF-kB/TNF-response clonal-expansion-like state. |
| `autocrine_tnf_tsk` | Autocrine-TNF/TSK/EMT-like invasive-state axis. |
| `pemt` | Partial EMT and invasion-front-like program. |
| `cis_like` | Squamous CIS-like/basal premalignant axis with mouse ortholog mapping. |
| `lrp1_proxy_module` | Broad CREDO-nominated Lrp1 TSK/pEMT proxy genes for localization tests. |
| `lrp1_epithelial_tsk_pemt_core` | Tight epithelial/invasive-state subset of the Lrp1 proxy. |
| `lrp1_caf_ecm_core` | CAF/ECM component used to detect stromal confounding. |
| `lrp1_inflammatory_myeloid_core` | Myeloid/inflammatory component used to detect niche confounding. |

The CIS-like signature resolves human-style `TP63`/`ATP1B3` concepts to mouse
symbols such as `Trp63` and `Atp1b3` when the HNSCC data use mouse symbols.
The `lrp1_proxy_module` and its split components are named proxy gene sets, not
decoded perturbation-specific residual modules. Use the split Lrp1 modules to
ask whether a raw Lrp1 proxy signal is epithelial TSK/pEMT-like, CAF/ECM-like,
or inflammatory/myeloid-like before interpreting human validation.

Custom signatures can be added to both HNSCC scoring and optional bulk
projection:

```bash
CUSTOM_SIGNATURES=custom_signatures.csv \
COMPARE_ROOT=runs/<comparison-root> \
bash scripts/run_hnscc_biological_findings.sh
```

The custom CSV needs `signature,gene` columns.

## Interpretation Rules

Use with-guide as the primary biological model and shared-guide as a negative
control. Shared-guide can have lower endpoint geometry/UOT because it smooths
terminal distributions; that does not make it the better perturbation biology
model.

The strongest CREDO contrast is same-start factual versus reference:

```text
same P4 initial measure
+ factual perturbation residual
versus
same P4 initial measure
+ learned soft-reference residual
```

Core v2 columns:

```text
delta_log_mass_fact_vs_ref
geom_shift_fact_vs_ref
growth_action_fact
drift_action_fact
diffusion_action_fact
context_dependence_mass
context_dependence_geom
shared_guide_null_gap
priority_class_v2
```

Treat `priority_class_v2` as triage, not final proof. Biological claims still
need fold stability, guide concordance when guide-level data are available,
artifact checks for large genes, Renz tumor-selection crosswalk, and human
validation.

## Human Bulk Projection

Optional signature-level projection:

```bash
BULK_EXPR=gse227919_expression.csv \
BULK_META=gse227919_metadata.csv \
COMPARE_ROOT=runs/<comparison-root> \
bash scripts/run_hnscc_biological_findings.sh
```

This is signature-level validation. It should not be described as
perturbation-specific human validation unless target-specific modules are
derived and projected separately.
