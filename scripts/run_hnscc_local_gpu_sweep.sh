#!/usr/bin/env bash
set -euo pipefail

# Sequential local-GPU HNSCC training sweep.
#
# Defaults are intentionally heavier than smoke tests but smaller than a full
# 3-seed claim run: one seed, all 4 CV folds, 200 epochs per setting.
# Override SWEEP_EPOCHS/SWEEP_SEEDS/FOLD_INDICES to scale up or narrow the run.

SCRIPT_DIR_ABS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CREDO_DIR_ABS="$(cd -- "${SCRIPT_DIR_ABS}/.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT_ABS="$(cd -- "${CREDO_DIR_ABS}/.." >/dev/null 2>&1 && pwd)"
cd "${WORKSPACE_ROOT_ABS}"

ENV_NAME="${ENV_NAME:-cape-hnscc}"
CONDA_BIN="${CONDA_BIN:-conda}"
CREDO_DIR="${CREDO_DIR:-CREDO}"
DATA_PATH="${DATA_PATH:-inputs/hnscc/GSE235325_P4P60_allgenes_allcells_latest_states.h5ad}"
SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-HNSCC/credo_runs/local_gpu_settings_sweep_${SWEEP_STAMP}}"
SWEEP_SEEDS="${SWEEP_SEEDS:-1}"
CV_FOLDS="${CV_FOLDS:-4}"
FOLD_INDICES="${FOLD_INDICES:-}"
FOLD_GPU_DEVICES="${FOLD_GPU_DEVICES:-}"
FOLD_JOBS_PER_GPU="${FOLD_JOBS_PER_GPU:-1}"

SWEEP_EPOCHS="${SWEEP_EPOCHS:-200}"
SWEEP_STAGE_C_EPOCHS="${SWEEP_STAGE_C_EPOCHS:-40}"
SWEEP_STAGE_D_EPOCHS="${SWEEP_STAGE_D_EPOCHS:-40}"

mkdir -p "${SWEEP_ROOT}/logs" "${SWEEP_ROOT}/cv_summary"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

run_variant() {
  local name="$1"
  shift
  local output_root="${SWEEP_ROOT}/${name}"
  local log_path="${SWEEP_ROOT}/logs/${name}.log"

  log "Starting setting ${name}; output=${output_root}"
  if env \
      OUTPUT_ROOT="${output_root}" \
      DATA_PATH="${DATA_PATH}" \
      ENV_NAME="${ENV_NAME}" \
      CONDA_BIN="${CONDA_BIN}" \
      GPU_VRAM_PROFILE=rtx3080_9p5gb \
      FOLD_GPU_DEVICES="${FOLD_GPU_DEVICES}" \
      FOLD_JOBS_PER_GPU="${FOLD_JOBS_PER_GPU}" \
      SEEDS="${SWEEP_SEEDS}" \
      CV_FOLDS="${CV_FOLDS}" \
      FOLD_INDICES="${FOLD_INDICES}" \
      RUN_VALIDATE=0 \
      RUN_SIGNATURES=0 \
      RUN_TRAIN=1 \
      RUN_COUNTERFACTUAL=0 \
      RUN_MERGE_COUNTERFACTUAL=0 \
      RUN_SUMMARY=1 \
      RUN_SEARCH=0 \
      RUN_BIOLOGY=0 \
      GUIDE_EMBEDDING_MODE=distinct \
      EPOCHS="${SWEEP_EPOCHS}" \
      STAGE_C_EPOCHS="${SWEEP_STAGE_C_EPOCHS}" \
      STAGE_D_EPOCHS="${SWEEP_STAGE_D_EPOCHS}" \
      "$@" \
      bash "${CREDO_DIR}/scripts/run_hnscc_full_pipeline.sh" >"${log_path}" 2>&1; then
    log "Completed setting ${name}; log=${log_path}"
  else
    log "FAILED setting ${name}; log=${log_path}"
    tail -n 80 "${log_path}" >&2 || true
    return 1
  fi
}

run_variant genes768_p32 \
  EXPRESSION_TOP_GENES=768 \
  N_PARTICLES=32 \
  N_STEPS=8 \
  N_TEST_FUNCTIONS=4 \
  MAX_ACTIVE_PERTURBATIONS=2 \
  MAX_TRAIN_TARGET_ATOMS=384 \
  EVAL_PARTICLES=256 \
  EVAL_STEPS=16 \
  EVAL_TARGET_PARTICLES=512 \
  HIDDEN_DIM=256 \
  EMBEDDING_DIM=32 \
  MEDIATOR_DIM=32 \
  CAUSAL_TOKEN_DIM=48 \
  CAUSAL_N_MEDIATORS=8 \
  BUDGET_HEADROOM=0.60

run_variant balanced640_p36 \
  EXPRESSION_TOP_GENES=640 \
  N_PARTICLES=36 \
  N_STEPS=8 \
  N_TEST_FUNCTIONS=4 \
  MAX_ACTIVE_PERTURBATIONS=2 \
  MAX_TRAIN_TARGET_ATOMS=384 \
  EVAL_PARTICLES=256 \
  EVAL_STEPS=16 \
  EVAL_TARGET_PARTICLES=512 \
  HIDDEN_DIM=256 \
  EMBEDDING_DIM=32 \
  MEDIATOR_DIM=32 \
  CAUSAL_TOKEN_DIM=48 \
  CAUSAL_N_MEDIATORS=8 \
  BUDGET_HEADROOM=0.60

run_variant particles512_p48 \
  EXPRESSION_TOP_GENES=512 \
  N_PARTICLES=48 \
  N_STEPS=8 \
  N_TEST_FUNCTIONS=4 \
  MAX_ACTIVE_PERTURBATIONS=2 \
  MAX_TRAIN_TARGET_ATOMS=384 \
  EVAL_PARTICLES=256 \
  EVAL_STEPS=16 \
  EVAL_TARGET_PARTICLES=512 \
  HIDDEN_DIM=256 \
  EMBEDDING_DIM=32 \
  MEDIATOR_DIM=32 \
  CAUSAL_TOKEN_DIM=48 \
  CAUSAL_N_MEDIATORS=8 \
  BUDGET_HEADROOM=0.60

log "Summarizing sweep across settings"
"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
  env PYTHONPATH="${CREDO_DIR}/package/src" python "${CREDO_DIR}/runners/summarize_hnscc_cv.py" \
  --cv-root "${SWEEP_ROOT}" \
  --output-dir "${SWEEP_ROOT}/cv_summary" \
  --group-by setting

log "Sweep complete"
log "Sweep root: ${SWEEP_ROOT}"
log "Summary: ${SWEEP_ROOT}/cv_summary/cv_summary.md"
