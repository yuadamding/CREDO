# H100 Runbook

This runbook keeps the detailed launch guidance out of the root README while
preserving the existing shell script paths.

## Current Recommended Comparison

Run the best `ur01` setting for four folds with distinct guide embeddings and
with one shared guide embedding:

```bash
CONDA_BIN=/rsrch8/home/bcb/$USER/miniforge3/bin/conda \
GPU_LIST=0,1,2,3 \
bash scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh
```

The wrapper assumes no comparison report exists yet. It writes combined
`cv_summary.md` and `cv_summary.csv` under `COMPARE_ROOT`.

Historical reproduction defaults are pinned in this wrapper:

```text
EXPRESSION_WORKERS=8
EXPRESSION_CHUNK_SIZE=2048
VAE_BATCH_SIZE=2048
VAE_ENCODE_BATCH_SIZE=8192
VAE_PRELOAD_DENSE_MAX_GB=4.0
```

Override those only for a deliberately non-comparable rerun.

## 4-GPU Use

The recommended strategy is one fold/setting job per H100:

```bash
GPU_LIST=0,1,2,3 MAX_PARALLEL_JOBS=4 \
bash scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh
```

On 4-GPU nodes the wrapper normally runs one comparison arm at a time with one
fold per GPU. On larger nodes, `PARALLEL_ARMS=auto` can split GPUs across
with-guide and shared-guide arms when there are enough useful jobs.

## Current Best Setting

The archived best setting is:

```text
h1344_d7_prog42_p352_s28_active40_lc2e3_lw20_gr5e4_nogint_e1800
```

Key parameters:

```text
hidden_dim=1344
depth=7
n_programs=42
n_particles=352
n_steps=28
eval_particles=1408
eval_steps=28
max_active_perturbations=40
lambda_control_ref=0.002
lambda_weak=0.20
lambda_reg_growth_bias=0.0005
growth_intercept=off
ecology=on
precision=bf16
```

## Search Launchers

| Script | Use |
| --- | --- |
| `scripts/run_hnscc_h100_heavy_f_optimal_search_4cv_2gpu_v2.sh` | Current heavy_f search launcher with one setting/fold job per GPU. |
| `scripts/run_hnscc_h100_heavy_f_no_guide_optimal_search_4cv_2gpu_v2.sh` | Same search, shared guide embedding ablation. |
| `scripts/run_hnscc_h100_heavy_f_random_4cv_2gpu.sh` | Heavy_f random 4-fold CV. |
| `scripts/run_hnscc_h100_heavy_f_parallel_4cv_2gpu.sh` | Heavy_f WTA CV. |
| `scripts/run_hnscc_h100_heavy_f_fast_300ep.sh` | Fast random smoke CV. |
| `scripts/run_hnscc_h100_heavy_c_default.sh` | Validated heavy_c default. |
| `scripts/run_hnscc_local_heavy_c_vae_9gb.sh` | Local single-GPU smoke setting. |

## Useful Controls

```bash
DRY_RUN=1                         # print planned grid
SEARCH_FOLDS=0                    # first-fold probe
SEED=0                            # train/test split, VAE, model, and eval seed
SEARCH_EPOCHS=300                 # smoke run
EPOCHS=2500                       # extended stress run; can overtrain/collapse some folds
USE_SETTING_EPOCHS=1              # keep historical per-row epoch budgets
STATE_KEY=None                    # skip state accuracy and rank by endpoint
GPU_MONITOR=1                     # write gpu_monitor.csv
THREADS_PER_GPU=56                # override CPU thread allocation
REQUIRE_FULL_GPU_QUEUE=1          # fail rather than silently underfill GPUs
```

`MULTI_GPU_PER_JOB=1` is only for deliberate single-fold true multi-GPU
experiments. Most high-active settings already fill one 80 GB H100 well, so
parallel one-GPU jobs are usually the better use of a node.

## GPU Utilization Notes

The H100 wrappers print a `CREDO resource plan` before each job. Confirm:

- `CUDA_VISIBLE_DEVICES` is set to the intended GPU.
- `gpu_resource` is non-empty.
- `max_active_perturbations` is high enough for a fat step.
- VAE expression loading is not stuck on CPU-only preprocessing longer than
  expected.

If the log shows `Building split-safe VAE latent...`, that phase can use CPU
heavily before CREDO training begins. GPU allocation should rise once VAE
training and CREDO training start.

## Epoch Budget Note

The current default is 1800 epochs because the four-fold `ur01` reproducibility
run produced stable with-guide mass and expansion metrics at this budget. A
2500-epoch rerun overfit or destabilized folds 0 and 3, with very large endpoint
UOT, mass error, and expansion-gap values, so treat 2500 as a diagnostic stress
run rather than the recommended biological model.
