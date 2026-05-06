# HNSCC CREDO Transfer Bundle

Portable runner for the HNSCC P4/P60 perturb-seq CREDO workflow.

Required beside this directory:

```bash
../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad
```

Install once:

```bash
CONDA_BIN=/rsrch8/home/bcb/$USER/miniforge3/bin/conda bash scripts/install_bundle.sh cape-hnscc
```

If the env already exists and `python scripts/verify_setup.py` works, skip `conda env update`; just refresh the editable package with `python -m pip install --no-cache-dir -e package`.

Recommended 2-H100 run:

```bash
GPU_LIST=0,1 SKIP_SUMMARY=0 bash scripts/run_hnscc_h100_heavy_f_random_4cv_2gpu.sh
```

Optimal-setting search with state test accuracy:

```bash
CONDA_BIN=/rsrch8/home/bcb/$USER/miniforge3/bin/conda \
GPU_LIST=0,1,2,3 MAX_PARALLEL_JOBS=4 \
bash scripts/run_hnscc_h100_heavy_f_optimal_search_4cv_2gpu_v2.sh
```

The default random and WTA heavy_f wrappers now use the fuller 2-H100 profile `h100_heavy_f_full`: `h1024_d6_prog32_p512_active16`, `N_STEPS=28`, `EVAL_STEPS=28`, `N_TEST_FUNCTIONS=12`, `MAX_TRAIN_TARGET_ATOMS=3072`, `EVAL_PARTICLES=2048`, and `EVAL_TARGET_PARTICLES=4096`. This keeps one fold per visible GPU in `RUN_MODE=parallel`, but gives each 80 GB H100 a much larger per-fold workload than the older `p256_active8` setting. Avoid carrying over the old explicit overrides (`N_PARTICLES=256`, `N_STEPS=24`, `MAX_ACTIVE_PERTURBATIONS=8`, `EVAL_PARTICLES=1024`, etc.) unless you intentionally want the lighter run.

The H100 wrappers default to `EXPRESSION_GENE_MASK_COL=hv_gene`, bf16, VAE latents, ecology-on, random stratification by `Time point,perturbation_id`, and one fold per visible GPU when `RUN_MODE=parallel`. CPU threads default to `nproc / visible_gpu_count`; override with `THREADS_PER_GPU`. Expression loading workers now default to `min(8, threads_per_job)`, with `EXPRESSION_CHUNK_SIZE=4096`, `VAE_BATCH_SIZE=4096`, and `VAE_ENCODE_BATCH_SIZE=16384` in the v2 H100 launcher to keep data staging from starving H100-side work. The legacy `h100_heavy_f` profile remains available as the lighter H100-fit setting (`h1024_d6_prog32_p256`, `MAX_ACTIVE_PERTURBATIONS=8`) because the earlier `h2048_d7_p608` setting OOMs at stage D on an 80 GB H100.

The optimal-search wrapper defaults `STATE_KEY="Cell type annotation"` and ranks by `test_acc`, the held-out dominant-state accuracy reported in `cv_summary.md` and `cv_summary.csv`. It schedules all setting/fold jobs through a GPU work queue, so the next fold starts as soon as any GPU is free instead of waiting at setting-level fold waves. Use the direct v2 launcher for current work: it defaults to `SETTINGS_PRESET=gpu_util_ladder`, a 20-setting first-fold sweep designed to improve H100 utilization by raising `max_active_perturbations` to 24-44 while reducing particles where needed to stay in roughly the 30 GB to 80 GB single-H100 range. This reduces perturbation chunk count and gives each training step fatter GPU work. After that screen, use `SETTINGS_PRESET=gpu_util_refine` to search around the best high-active branch (`h1344_d7_prog42_p352_s28_active40_lc2e3_lw20_gr5e4_nogint_e1800`) and probe active 40-60 with lower particles. Use `scripts/run_hnscc_h100_heavy_f_no_guide_optimal_search_4cv_2gpu_v2.sh` or set `SHARED_GUIDE_EMBEDDING=1` to keep the same guide-confident cell population and original perturbation groups while forcing every perturbation to use one shared guide embedding in the model. The older ladder includes smaller 24-step/1200-epoch models, mid-size 24/26-step 1400-1600-epoch models, and near-full-H100 28-step/1800-2000-epoch candidates, including the current 4-fold winner, `h1408_d7_prog44_p512_s28_active24_lc2e3_lw20_gr5e4_nogint`, and the close `h1280/lc1/lw15` accuracy competitor. Use `SETTINGS_PRESET=vram_epoch_ladder` for the older VRAM-fill ladder, `SETTINGS_PRESET=stable_capacity_s28` or `SETTINGS_PRESET=refine_winner_s28` for the focused 28-step high-VRAM grid, `SETTINGS_PRESET=winner_s28` to rerun only the current 4-fold winner, `SETTINGS_PRESET=finalists_s28` to rerun the three earlier finalist settings, `SETTINGS_PRESET=particle_probe_s28` for the endpoint/UOT-oriented particle-size probe, `SETTINGS_PRESET=broad_search` only for an explicit broad 28-step exploratory screen, `SETTINGS_PRESET=step32_vramfit` only for a high-risk 32-step experiment, `SETTINGS_PRESET=screen_s28` to rerun the full original six-setting screen, or `SETTINGS_PRESET=acc_only_s28` as an alias for the current winner. The recommended multi-GPU strategy is one fold/setting job per H100 (`GPU_LIST=0,1,2,3 MAX_PARALLEL_JOBS=4` on a 4-GPU node), because larger settings already use roughly 72-80 GB on one 80 GB GPU; the launcher pins each job to a disjoint CPU core range by default (`PIN_CPU=1`). Use `GUIDE_CONFIDENT_ONLY=0` only for a separate cell-population/noise sensitivity run that passes `--include-nonconfident`. Use `MAX_PARALLEL_JOBS` to use fewer GPUs, `THREADS_PER_GPU` to override CPU threads, `GPU_MONITOR=1` to write per-job `gpu_monitor.csv`, `EXPRESSION_WORKERS`/`EXPRESSION_CHUNK_SIZE`, `VAE_BATCH_SIZE`, `VAE_ENCODE_BATCH_SIZE`, and `VAE_PRELOAD_DENSE_MAX_GB` to tune VAE expression loading, and `MULTI_GPU_PER_JOB=1` only for a deliberate single-fold true multi-GPU experiment that consumes all visible GPUs and passes `--multi-gpu-devices` to the runner. The v2 launcher warns when the planned setting/fold queue has fewer jobs than GPU slots; set `REQUIRE_FULL_GPU_QUEUE=1` to fail early instead of silently leaving devices idle. Keep activation checkpointing on for high-active utilization presets; if `ACTIVATION_CHECKPOINTING=0` is requested and `max_active_perturbations > 36`, the launcher re-enables checkpointing for that job unless `ALLOW_UNSAFE_NO_CHECKPOINTING=1` is set. Custom `SETTINGS_FILE` rows use `tag|embedding|mediator|programs|hidden|depth|particles|eval_particles|eval_target|max_atoms|max_active|lambda_ctrl|lambda_weak|growth_reg` plus optional `steps|eval_steps|growth_intercept|program_basis|ecology|epochs`. The CV summary includes epochs, train/eval steps, particles, and train/eval peak GPU GB so stale or under-sized runs are visible in the table. Set `SEARCH_FOLDS=0` for a first-fold VRAM/performance probe before launching all folds. Set `STATE_KEY=None` to skip state accuracy and rank by endpoint UOT/mass instead. Set `SEARCH_EPOCHS=300` for a fast smoke run; it overrides per-row epoch values. Use `DRY_RUN=1` to print the planned grid.

Guide-vs-shared comparison from scratch:

```bash
CONDA_BIN=/rsrch8/home/bcb/$USER/miniforge3/bin/conda \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh
```

That wrapper assumes no comparison report exists yet. It runs the best `ur01` setting for 4 folds with distinct guide embeddings and with one shared guide embedding, then writes combined `cv_summary.md`/`cv_summary.csv` under `COMPARE_ROOT`. `PARALLEL_ARMS=auto` is the default: on nodes with more visible GPUs than jobs per arm, it splits GPUs across the two arms so all useful GPUs are fed at once; on 4-GPU nodes it keeps one arm at a time with one fold per GPU. Set `PARALLEL_ARMS=0` only if you deliberately want serial arms, and set `REQUIRE_FULL_GPU_QUEUE=0` if you accept an underfilled queue.

Entry points:

| Script | Use |
| --- | --- |
| `scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh` | best `ur01` 4-fold with-guide vs shared-guide comparison from scratch |
| `scripts/run_hnscc_h100_heavy_c_default.sh` | validated heavy_c default |
| `scripts/run_hnscc_h100_heavy_c_joint_4cv_2gpu.sh` | heavy_c, one fold at a time across two GPUs |
| `scripts/run_hnscc_h100_heavy_c_fast_300ep.sh` | heavy_c fast random 4-fold CV |
| `scripts/run_hnscc_h100_heavy_f_random_4cv_2gpu.sh` | heavy_f random 4-fold CV, one fold per GPU |
| `scripts/run_hnscc_h100_heavy_f_optimal_search_4cv_2gpu_v2.sh` | current heavy_f optimal search, one setting/fold job per GPU |
| `scripts/run_hnscc_h100_heavy_f_no_guide_optimal_search_4cv_2gpu_v2.sh` | same v2 search and same cell filter, but all perturbations share one guide embedding |
| `scripts/run_hnscc_h100_heavy_f_optimal_search_4cv_2gpu.sh` | legacy heavy_f setting search wrapper |
| `scripts/run_hnscc_h100_heavy_f_parallel_4cv_2gpu.sh` | heavy_f WTA CV, one fold per GPU |
| `scripts/run_hnscc_h100_heavy_f_fast_300ep.sh` | heavy_f fast random 4-fold CV |
| `scripts/run_hnscc_h100_heavy_f_vram60_75_search_300ep.sh` | heavy_f fit search |
| `scripts/run_hnscc_biological_findings.sh` | post-process CREDO CV outputs into perturbation biology, signature, and optional human-projection tables |
| `scripts/run_hnscc_local_heavy_c_vae_9gb.sh` | local single-GPU 9 GB VAE-first smoke setting |
| `scripts/prepare_hnscc_heavy_c_reuse.sh` | export a finished heavy_c CV run |

Biological interpretation:

```bash
COMPARE_ROOT=runs/hnscc_random_h100_heavy_f_best_ur01_guide_vs_shared_4cv_YYYYMMDD_HHMMSS \
bash scripts/run_hnscc_biological_findings.sh
```

This scores built-in TNF-expansion, autocrine-TNF/TSK, pEMT, and CIS-like signatures in the HNSCC AnnData, combines them with per-perturbation CV endpoint/state metrics, and writes `biological_effects_per_perturbation.csv`. Set `COUNTERFACTUAL_RUN_DIR=/path/to/fold/run` to add factual-vs-reference rollouts with mass, geometry, growth, drift, diffusion, and optional context-clamped readouts. Set `BULK_EXPR` and `BULK_META` to project the same signatures onto a human bulk cohort such as GSE227919.

Implementation notes:

- Reports, run metadata, and new entry points use `CREDO`.
- Legacy `cape` imports remain valid; `credo` is an alias package for new code.
- `scripts/_run_hnscc_cv.sh` owns shared launcher logic and CREDO profiles.
- Parallel expression loading is shard-based: each worker opens the backed h5ad read-only and processes multiple row chunks.
- Logs start with a `CREDO resource plan` showing GPU, CPU, expression-worker, and command details.
- Output goes to `runs/`; reusable model exports go to `models/`.
