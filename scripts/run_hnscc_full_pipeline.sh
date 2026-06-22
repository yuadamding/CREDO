#!/usr/bin/env bash
set -euo pipefail

# End-to-end HNSCC CREDO pipeline:
#   1. validate the HNSCC AnnData and software environment
#   2. score expression signatures once on the input data
#   3. train CREDO over random-k-fold HNSCC endpoint splits
#   4. run per-fold factual-vs-reference counterfactual biology
#   5. export completed folds into the CREDO search/RL selection framework
#   6. merge counterfactual outputs, summarize CV, and rank biology effects
#
# Runtime is intentionally controlled through environment variables so the same
# script can run a short dry/smoke pass or a full claim-grade run.

SCRIPT_DIR_ABS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CREDO_DIR_ABS="$(cd -- "${SCRIPT_DIR_ABS}/.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT_ABS="$(cd -- "${CREDO_DIR_ABS}/.." >/dev/null 2>&1 && pwd)"
cd "${WORKSPACE_ROOT_ABS}"

ENV_NAME="${ENV_NAME:-cape-hnscc}"
CONDA_BIN="${CONDA_BIN:-conda}"
CREDO_DIR="${CREDO_DIR:-CREDO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-.}"
PYTHONPATH_VALUE="${CREDO_DIR}/package/src"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

DATA_PATH="${DATA_PATH:-inputs/hnscc/GSE235325_P4P60_allgenes_allcells_latest_states.h5ad}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-HNSCC/credo_runs/full_pipeline_${RUN_STAMP}}"
SIGNATURE_DIR="${SIGNATURE_DIR:-${OUTPUT_ROOT}/signature_scores}"
BIOLOGY_DIR="${BIOLOGY_DIR:-${OUTPUT_ROOT}/biology}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUTPUT_ROOT}/cv_summary}"
SEARCH_DIR="${SEARCH_DIR:-${OUTPUT_ROOT}/search}"

SEEDS="${SEEDS:-1 2 3}"
CV_FOLDS="${CV_FOLDS:-4}"
FOLD_INDICES="${FOLD_INDICES:-}"

RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_SIGNATURES="${RUN_SIGNATURES:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_COUNTERFACTUAL="${RUN_COUNTERFACTUAL:-1}"
RUN_MERGE_COUNTERFACTUAL="${RUN_MERGE_COUNTERFACTUAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_SEARCH="${RUN_SEARCH:-1}"
RUN_BIOLOGY="${RUN_BIOLOGY:-1}"

PRECISION="${PRECISION:-bf16}"
LATENT_SOURCE="${LATENT_SOURCE:-expression}"
LATENT_KEY="${LATENT_KEY:-}"
VERIFY_LATENT_KEY="${VERIFY_LATENT_KEY:-}"
CONTEXT_KIND="${CONTEXT_KIND:-causal_attention}"
EPOCHS="${EPOCHS:-500}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-100}"
STAGE_D_EPOCHS="${STAGE_D_EPOCHS:-100}"
SEED_OFFSET="${SEED_OFFSET:-0}"

N_PARTICLES="${N_PARTICLES:-64}"
N_STEPS="${N_STEPS:-16}"
EVAL_PARTICLES="${EVAL_PARTICLES:-384}"
EVAL_STEPS="${EVAL_STEPS:-24}"
EVAL_TARGET_PARTICLES="${EVAL_TARGET_PARTICLES:-768}"
MAX_TRAIN_TARGET_ATOMS="${MAX_TRAIN_TARGET_ATOMS:-384}"
MAX_ACTIVE_PERTURBATIONS="${MAX_ACTIVE_PERTURBATIONS:-8}"
BUDGET_HEADROOM="${BUDGET_HEADROOM:-0.70}"

N_PROGRAMS="${N_PROGRAMS:-16}"
EMBEDDING_DIM="${EMBEDDING_DIM:-48}"
MEDIATOR_DIM="${MEDIATOR_DIM:-48}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
DEPTH="${DEPTH:-3}"
LAMBDA_WEAK="${LAMBDA_WEAK:-0.08}"
LAMBDA_REG_GROWTH_BIAS="${LAMBDA_REG_GROWTH_BIAS:-0.0001}"
LR_NET="${LR_NET:-3e-4}"
LR_EMBED="${LR_EMBED:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SINKHORN_EPSILON="${SINKHORN_EPSILON:-0.1}"
SINKHORN_TAU="${SINKHORN_TAU:-1.0}"
SINKHORN_MAX_ITER="${SINKHORN_MAX_ITER:-100}"

TRANSFORMER_TOKEN_DIM="${TRANSFORMER_TOKEN_DIM:-128}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_WITHIN_LAYERS="${TRANSFORMER_WITHIN_LAYERS:-1}"
TRANSFORMER_CROSS_LAYERS="${TRANSFORMER_CROSS_LAYERS:-1}"
TRANSFORMER_INDUCING="${TRANSFORMER_INDUCING:-16}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.05}"
MASS_ATTENTION_TEMPERATURE="${MASS_ATTENTION_TEMPERATURE:-0.5}"
LR_TRANSFORMER="${LR_TRANSFORMER:-5e-5}"
TRANSFORMER_WEIGHT_DECAY="${TRANSFORMER_WEIGHT_DECAY:-1e-4}"

CAUSAL_TOKEN_DIM="${CAUSAL_TOKEN_DIM:-64}"
CAUSAL_HEADS="${CAUSAL_HEADS:-4}"
CAUSAL_N_MEDIATORS="${CAUSAL_N_MEDIATORS:-12}"
CAUSAL_DROPOUT="${CAUSAL_DROPOUT:-0.05}"
CAUSAL_MASS_ATTENTION_TEMPERATURE="${CAUSAL_MASS_ATTENTION_TEMPERATURE:-0.5}"
CAUSAL_RESIDUAL_POLICY="${CAUSAL_RESIDUAL_POLICY:-edges_only}"
LR_CAUSAL_ATTENTION="${LR_CAUSAL_ATTENTION:-5e-5}"
CAUSAL_ATTENTION_WEIGHT_DECAY="${CAUSAL_ATTENTION_WEIGHT_DECAY:-1e-4}"
LAMBDA_CAUSAL_CTRL_EDGE="${LAMBDA_CAUSAL_CTRL_EDGE:-1e-3}"
LAMBDA_CAUSAL_GUIDE="${LAMBDA_CAUSAL_GUIDE:-0.0}"
LAMBDA_CAUSAL_SPARSE="${LAMBDA_CAUSAL_SPARSE:-1e-4}"
LAMBDA_CAUSAL_ORTH="${LAMBDA_CAUSAL_ORTH:-1e-4}"
LAMBDA_CAUSAL_CTX_SMOOTH="${LAMBDA_CAUSAL_CTX_SMOOTH:-1e-4}"
CAUSAL_LOSS_START_EPOCH="${CAUSAL_LOSS_START_EPOCH:-100}"
CAUSAL_LOSS_RAMP_EPOCHS="${CAUSAL_LOSS_RAMP_EPOCHS:-200}"

EXPRESSION_GENE_MASK_COL="${EXPRESSION_GENE_MASK_COL:-hv_gene}"
EXPRESSION_TOP_GENES="${EXPRESSION_TOP_GENES:-1024}"
EXPRESSION_WORKERS="${EXPRESSION_WORKERS:-0}"
EXPRESSION_CHUNK_SIZE="${EXPRESSION_CHUNK_SIZE:-1024}"
EXPRESSION_LAYER="${EXPRESSION_LAYER:-}"
EXPRESSION_USE_RAW="${EXPRESSION_USE_RAW:-0}"
EXPRESSION_ALLOW_FULL_GENE_SCAN="${EXPRESSION_ALLOW_FULL_GENE_SCAN:-0}"
EXPRESSION_ALLOW_EMPTY_GENE_MASK_FALLBACK="${EXPRESSION_ALLOW_EMPTY_GENE_MASK_FALLBACK:-0}"

CPU_THREADS="${CPU_THREADS:-8}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-2}"
MULTI_GPU_DEVICES="${MULTI_GPU_DEVICES:-}"
FOLD_GPU_DEVICES="${FOLD_GPU_DEVICES:-${GPU_DEVICES:-${MULTI_GPU_DEVICES}}}"
FOLD_JOBS_PER_GPU="${FOLD_JOBS_PER_GPU:-1}"
PER_FOLD_MULTI_GPU_DEVICES="${PER_FOLD_MULTI_GPU_DEVICES:-}"
FOLD_JOB_LOG_DIR="${FOLD_JOB_LOG_DIR:-${OUTPUT_ROOT}/job_logs}"

COUNTERFACTUAL_PARTICLES="${COUNTERFACTUAL_PARTICLES:-384}"
COUNTERFACTUAL_STEPS="${COUNTERFACTUAL_STEPS:-28}"
COUNTERFACTUAL_MAX_PERTURBATIONS="${COUNTERFACTUAL_MAX_PERTURBATIONS:-0}"
COUNTERFACTUAL_DEVICE="${COUNTERFACTUAL_DEVICE:-auto}"
COUNTERFACTUAL_CONTEXT_CLAMPED="${COUNTERFACTUAL_CONTEXT_CLAMPED:-1}"
COUNTERFACTUAL_INCLUDE_CONTROLS="${COUNTERFACTUAL_INCLUDE_CONTROLS:-1}"

BIOLOGY_TOP_N="${BIOLOGY_TOP_N:-60}"

SEARCH_PROFILE="${SEARCH_PROFILE:-pareto_refit}"
SEARCH_OBJECTIVES="${SEARCH_OBJECTIVES:-endpoint_geom_mass,mass_error,counterfactual_null_gap}"
SEARCH_SORT_BY="${SEARCH_SORT_BY:-}"
SEARCH_MIN_FOLDS="${SEARCH_MIN_FOLDS:-}"
SEARCH_MIN_SEEDS="${SEARCH_MIN_SEEDS:-}"
SEARCH_REQUIRED_FOLDS="${SEARCH_REQUIRED_FOLDS:-}"
SEARCH_REQUIRED_SEEDS="${SEARCH_REQUIRED_SEEDS:-}"
SEARCH_CLAIM_CONTROL_NULL_MAX="${SEARCH_CLAIM_CONTROL_NULL_MAX:-}"
SEARCH_CLAIM_LOG_MASS_ERROR_MAX="${SEARCH_CLAIM_LOG_MASS_ERROR_MAX:-}"
SEARCH_CLAIM_GUIDE_CONCORDANCE_MAX="${SEARCH_CLAIM_GUIDE_CONCORDANCE_MAX:-}"
SEARCH_CLAIM_REQUIRE_GUIDE_CONCORDANCE="${SEARCH_CLAIM_REQUIRE_GUIDE_CONCORDANCE:-0}"

run_py() {
  "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    env CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}" PYTHONPATH="${PYTHONPATH_VALUE}" python "$@"
}

run_py_on_gpu() {
  local gpu_device="${1:-}"
  shift
  if [[ -n "${gpu_device}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_device}" "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
      env CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}" CUDA_VISIBLE_DEVICES="${gpu_device}" \
      PYTHONPATH="${PYTHONPATH_VALUE}" python "$@"
  else
    run_py "$@"
  fi
}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

fold_label() {
  printf 'fold%02d' "$1"
}

fold_indices() {
  if [[ -n "${FOLD_INDICES}" ]]; then
    printf '%s\n' ${FOLD_INDICES}
  else
    seq 0 "$((CV_FOLDS - 1))"
  fi
}

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

fold_grid_csv() {
  local labels=()
  local fold
  for fold in $(fold_indices); do
    labels+=("$(fold_label "${fold}")")
  done
  join_by_comma "${labels[@]}"
}

fold_grid_count() {
  local count=0
  local fold
  for fold in $(fold_indices); do
    count=$((count + 1))
  done
  printf '%s' "${count}"
}

seed_grid_csv() {
  local values=()
  local seed
  for seed in ${SEEDS}; do
    values+=("$((seed + SEED_OFFSET))")
  done
  join_by_comma "${values[@]}"
}

seed_grid_count() {
  local count=0
  local seed
  for seed in ${SEEDS}; do
    count=$((count + 1))
  done
  printf '%s' "${count}"
}

parse_gpu_devices() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  for device in ${raw}; do
    device="${device#cuda:}"
    if [[ -n "${device}" && "${device}" != "-1" ]]; then
      printf '%s\n' "${device}"
    fi
  done
}

detect_gpu_devices() {
  if [[ -n "${FOLD_GPU_DEVICES}" ]]; then
    parse_gpu_devices "${FOLD_GPU_DEVICES}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    parse_gpu_devices "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' | xargs -r printf '%s\n'
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'Required file is missing: %s\n' "${path}" >&2
    exit 1
  fi
}

TRAIN_EXTRA_ARGS=(
  --expression-gene-mask-col "${EXPRESSION_GENE_MASK_COL}"
  --expression-top-genes "${EXPRESSION_TOP_GENES}"
  --expression-workers "${EXPRESSION_WORKERS}"
  --expression-chunk-size "${EXPRESSION_CHUNK_SIZE}"
)
if [[ -n "${EXPRESSION_LAYER}" ]]; then
  TRAIN_EXTRA_ARGS+=(--expression-layer "${EXPRESSION_LAYER}")
fi
if truthy "${EXPRESSION_USE_RAW}"; then
  TRAIN_EXTRA_ARGS+=(--expression-use-raw)
fi
if truthy "${EXPRESSION_ALLOW_FULL_GENE_SCAN}"; then
  TRAIN_EXTRA_ARGS+=(--expression-allow-full-gene-scan)
fi
if truthy "${EXPRESSION_ALLOW_EMPTY_GENE_MASK_FALLBACK}"; then
  TRAIN_EXTRA_ARGS+=(--expression-allow-empty-gene-mask-fallback)
fi

LATENT_ARGS=(--latent-source "${LATENT_SOURCE}")
if [[ "${LATENT_SOURCE}" == "obsm" ]]; then
  LATENT_ARGS+=(--latent-key "${LATENT_KEY:-X_pca}")
elif [[ -n "${LATENT_KEY}" ]]; then
  LATENT_ARGS+=(--latent-key "${LATENT_KEY}")
fi

require_file "${DATA_PATH}"
mkdir -p "${OUTPUT_ROOT}" "${SIGNATURE_DIR}" "${BIOLOGY_DIR}" "${SUMMARY_DIR}" "${SEARCH_DIR}" "${FOLD_JOB_LOG_DIR}"

mapfile -t GPU_DEVICE_LIST < <(detect_gpu_devices)
FOLD_JOBS_PER_GPU_INT="${FOLD_JOBS_PER_GPU}"
if ! [[ "${FOLD_JOBS_PER_GPU_INT}" =~ ^[0-9]+$ ]] || [[ "${FOLD_JOBS_PER_GPU_INT}" -lt 1 ]]; then
  FOLD_JOBS_PER_GPU_INT=1
fi
GPU_JOB_SLOTS=()
if [[ "${#GPU_DEVICE_LIST[@]}" -gt 0 ]]; then
  for gpu_device in "${GPU_DEVICE_LIST[@]}"; do
    for _ in $(seq 1 "${FOLD_JOBS_PER_GPU_INT}"); do
      GPU_JOB_SLOTS+=("${gpu_device}")
    done
  done
fi

log "CREDO dir: ${CREDO_DIR}"
log "Workspace: ${WORKSPACE_ROOT}"
log "Data: ${DATA_PATH}"
log "Output root: ${OUTPUT_ROOT}"
log "Conda env: ${ENV_NAME}"
if [[ "${#GPU_JOB_SLOTS[@]}" -gt 0 ]]; then
  log "Fold GPU devices: ${GPU_DEVICE_LIST[*]} (${FOLD_JOBS_PER_GPU_INT} fold job per GPU)"
  log "Per-fold runner GPU view: CUDA_VISIBLE_DEVICES=<assigned GPU>, --multi-gpu-devices ${PER_FOLD_MULTI_GPU_DEVICES:-0}"
else
  log "Fold GPU devices: none detected; fold jobs will run sequentially with runner device auto-detection"
fi

if truthy "${RUN_VALIDATE}"; then
  log "Validating environment and HNSCC AnnData contract"
  run_py "${CREDO_DIR}/scripts/verify_setup.py" \
    --check-data \
    --data-path "${DATA_PATH}" \
    --data-schema custom \
    --latent-key "${VERIFY_LATENT_KEY}" \
    --obs-column Library \
    --obs-column "Time point" \
    --obs-column "Cell type annotation" \
    --json \
    > "${OUTPUT_ROOT}/verify_setup_hnscc.json"
fi

if truthy "${RUN_SIGNATURES}"; then
  log "Scoring HNSCC expression signatures"
  run_py "${CREDO_DIR}/analysis/score_hnscc_signatures.py" \
    --data-path "${DATA_PATH}" \
    --output-dir "${SIGNATURE_DIR}" \
    --group-cols perturbation_id,time_label \
    --state-key "Cell type annotation" \
    --guide-confident-only \
    --log1p
fi

declare -a RUN_DIRS=()
declare -a COUNTERFACTUAL_FILES=()
declare -a FOLD_JOB_PIDS=()
declare -a FOLD_JOB_LABELS=()
declare -a FOLD_JOB_LOGS=()

run_fold_job() {
  local seed="$1"
  local fold="$2"
  local gpu_device="${3:-}"
  local fold_name
  local run_dir
  local biology_run_dir
  local cf_file
  local runner_devices

  fold_name="$(fold_label "${fold}")"
  run_dir="${OUTPUT_ROOT}/seed${seed}_${fold_name}"
  biology_run_dir="${run_dir}/biology"
  cf_file="${biology_run_dir}/counterfactual_biology_effects.csv"
  runner_devices="${PER_FOLD_MULTI_GPU_DEVICES}"
  if [[ -z "${runner_devices}" && -n "${gpu_device}" ]]; then
    runner_devices="0"
  fi

  if truthy "${RUN_TRAIN}"; then
    log "Training HNSCC CREDO seed=${seed} fold=${fold}/${CV_FOLDS} gpu=${gpu_device:-auto}"
    train_args=(
      "${CREDO_DIR}/runners/run_credo_hnscc_full.py"
      --data-path "${DATA_PATH}"
      --output-dir "${run_dir}"
      "${LATENT_ARGS[@]}"
      --seed "$((seed + SEED_OFFSET))"
      --precision "${PRECISION}"
      --split-strategy random_kfold
      --cv-folds "${CV_FOLDS}"
      --cv-fold-index "${fold}"
      --random-stratify-cols "Time point,perturbation_id"
      --state-key "Cell type annotation"
      --guide-confident-only
      --learned-programs
      --shared-guide-embedding
      --control-mode soft_ref
      --lambda-control-ref 0.0005
      --control-ref-warmup-epochs 100
      --training-schedule staged
      --stage-c-epochs "${STAGE_C_EPOCHS}"
      --stage-d-epochs "${STAGE_D_EPOCHS}"
      --ecology-on
      --growth-intercept-on
      --activation-checkpointing
      --context-kind "${CONTEXT_KIND}"
      --transformer-growth-only
      --transformer-token-dim "${TRANSFORMER_TOKEN_DIM}"
      --transformer-heads "${TRANSFORMER_HEADS}"
      --transformer-within-layers "${TRANSFORMER_WITHIN_LAYERS}"
      --transformer-cross-layers "${TRANSFORMER_CROSS_LAYERS}"
      --transformer-inducing "${TRANSFORMER_INDUCING}"
      --transformer-dropout "${TRANSFORMER_DROPOUT}"
      --mass-attention-temperature "${MASS_ATTENTION_TEMPERATURE}"
      --lr-transformer "${LR_TRANSFORMER}"
      --transformer-weight-decay "${TRANSFORMER_WEIGHT_DECAY}"
      --causal-growth-only
      --causal-sparse-edges
      --causal-token-dim "${CAUSAL_TOKEN_DIM}"
      --causal-heads "${CAUSAL_HEADS}"
      --causal-n-mediators "${CAUSAL_N_MEDIATORS}"
      --causal-dropout "${CAUSAL_DROPOUT}"
      --causal-mass-attention-temperature "${CAUSAL_MASS_ATTENTION_TEMPERATURE}"
      --causal-residual-policy "${CAUSAL_RESIDUAL_POLICY}"
      --lr-causal-attention "${LR_CAUSAL_ATTENTION}"
      --causal-attention-weight-decay "${CAUSAL_ATTENTION_WEIGHT_DECAY}"
      --lambda-causal-ctrl-edge "${LAMBDA_CAUSAL_CTRL_EDGE}"
      --lambda-causal-guide "${LAMBDA_CAUSAL_GUIDE}"
      --lambda-causal-sparse "${LAMBDA_CAUSAL_SPARSE}"
      --lambda-causal-orth "${LAMBDA_CAUSAL_ORTH}"
      --lambda-causal-ctx-smooth "${LAMBDA_CAUSAL_CTX_SMOOTH}"
      --causal-loss-start-epoch "${CAUSAL_LOSS_START_EPOCH}"
      --causal-loss-ramp-epochs "${CAUSAL_LOSS_RAMP_EPOCHS}"
      --n-programs "${N_PROGRAMS}"
      --embedding-dim "${EMBEDDING_DIM}"
      --mediator-dim "${MEDIATOR_DIM}"
      --hidden-dim "${HIDDEN_DIM}"
      --depth "${DEPTH}"
      --epochs "${EPOCHS}"
      --n-particles "${N_PARTICLES}"
      --n-steps "${N_STEPS}"
      --eval-particles "${EVAL_PARTICLES}"
      --eval-steps "${EVAL_STEPS}"
      --eval-target-particles "${EVAL_TARGET_PARTICLES}"
      --max-train-target-atoms "${MAX_TRAIN_TARGET_ATOMS}"
      --n-test-functions 12
      --lr-net "${LR_NET}"
      --lr-embed "${LR_EMBED}"
      --weight-decay "${WEIGHT_DECAY}"
      --grad-clip "${GRAD_CLIP}"
      --lambda-weak "${LAMBDA_WEAK}"
      --lambda-reg-growth-bias "${LAMBDA_REG_GROWTH_BIAS}"
      --sinkhorn-epsilon "${SINKHORN_EPSILON}"
      --sinkhorn-tau "${SINKHORN_TAU}"
      --sinkhorn-max-iter "${SINKHORN_MAX_ITER}"
      --max-active-perturbations "${MAX_ACTIVE_PERTURBATIONS}"
      --budget-headroom "${BUDGET_HEADROOM}"
      --auto-scale-budget
      --min-cells-p4 20
      --min-cells-p60 20
      --mass-scope subset_only
      --mass-mode count
      --cpu-threads "${CPU_THREADS}"
      --cpu-interop-threads "${CPU_INTEROP_THREADS}"
      "${TRAIN_EXTRA_ARGS[@]}"
    )
    if [[ -n "${runner_devices}" ]]; then
      train_args+=(--multi-gpu-devices "${runner_devices}")
    fi
    run_py_on_gpu "${gpu_device}" "${train_args[@]}"
  fi

  if truthy "${RUN_COUNTERFACTUAL}"; then
    require_file "${run_dir}/results_summary.json"
    log "Running counterfactual biology seed=${seed} fold=${fold}/${CV_FOLDS} gpu=${gpu_device:-auto}"
    cf_args=(
      "${CREDO_DIR}/analysis/run_counterfactual_biology.py"
      --run-dir "${run_dir}"
      --data-path "${DATA_PATH}"
      --output-dir "${biology_run_dir}"
      --source-split test
      --n-particles "${COUNTERFACTUAL_PARTICLES}"
      --n-steps "${COUNTERFACTUAL_STEPS}"
      --device "${COUNTERFACTUAL_DEVICE}"
      --seed "$((seed * 1000 + fold))"
      --max-perturbations "${COUNTERFACTUAL_MAX_PERTURBATIONS}"
      --fold-id "${fold_name}"
    )
    if truthy "${COUNTERFACTUAL_CONTEXT_CLAMPED}"; then
      cf_args+=(--context-clamped)
    fi
    if truthy "${COUNTERFACTUAL_INCLUDE_CONTROLS}"; then
      cf_args+=(--include-controls-for-null)
    fi
    run_py_on_gpu "${gpu_device}" "${cf_args[@]}"
  fi
}

launch_fold_job() {
  local seed="$1"
  local fold="$2"
  local gpu_device="${3:-}"
  local fold_name
  local label
  local log_file

  fold_name="$(fold_label "${fold}")"
  label="seed${seed}_${fold_name}"
  log_file="${FOLD_JOB_LOG_DIR}/${label}.log"
  (
    run_fold_job "${seed}" "${fold}" "${gpu_device}"
  ) >"${log_file}" 2>&1 &
  FOLD_JOB_PIDS+=("$!")
  FOLD_JOB_LABELS+=("${label}")
  FOLD_JOB_LOGS+=("${log_file}")
  log "Launched ${label} on gpu=${gpu_device:-auto}; log=${log_file}"
}

wait_for_fold_batch() {
  local failed=0
  local idx
  for idx in "${!FOLD_JOB_PIDS[@]}"; do
    if wait "${FOLD_JOB_PIDS[$idx]}"; then
      log "Completed ${FOLD_JOB_LABELS[$idx]}"
    else
      log "FAILED ${FOLD_JOB_LABELS[$idx]} (see ${FOLD_JOB_LOGS[$idx]})"
      failed=1
    fi
  done
  FOLD_JOB_PIDS=()
  FOLD_JOB_LABELS=()
  FOLD_JOB_LOGS=()
  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
}

job_index=0
slot_count="${#GPU_JOB_SLOTS[@]}"
for seed in ${SEEDS}; do
  for fold in $(fold_indices); do
    fold_name="$(fold_label "${fold}")"
    run_dir="${OUTPUT_ROOT}/seed${seed}_${fold_name}"
    RUN_DIRS+=("${run_dir}")
    biology_run_dir="${run_dir}/biology"
    cf_file="${biology_run_dir}/counterfactual_biology_effects.csv"
    COUNTERFACTUAL_FILES+=("${cf_file}")

    if truthy "${RUN_TRAIN}" || truthy "${RUN_COUNTERFACTUAL}"; then
      if [[ "${slot_count}" -gt 0 ]]; then
        launch_fold_job "${seed}" "${fold}" "${GPU_JOB_SLOTS[$((job_index % slot_count))]}"
        job_index=$((job_index + 1))
        if [[ "${#FOLD_JOB_PIDS[@]}" -ge "${slot_count}" ]]; then
          wait_for_fold_batch
        fi
      else
        run_fold_job "${seed}" "${fold}" ""
      fi
    fi
  done
done
if [[ "${#FOLD_JOB_PIDS[@]}" -gt 0 ]]; then
  wait_for_fold_batch
fi

if truthy "${RUN_MERGE_COUNTERFACTUAL}"; then
  log "Merging counterfactual biology tables"
  existing_cf=()
  for path in "${COUNTERFACTUAL_FILES[@]}"; do
    if [[ -f "${path}" ]]; then
      existing_cf+=("${path}")
    fi
  done
  if [[ "${#existing_cf[@]}" -eq 0 ]]; then
    printf 'No counterfactual_biology_effects.csv files found to merge.\n' >&2
    exit 1
  fi
  run_py "${CREDO_DIR}/analysis/merge_counterfactual_biology.py" \
    --inputs "${existing_cf[@]}" \
    --output "${BIOLOGY_DIR}/counterfactual_biology_effects_merged.csv"
fi

if truthy "${RUN_SUMMARY}"; then
  log "Summarizing HNSCC CV results"
  run_py "${CREDO_DIR}/runners/summarize_hnscc_cv.py" \
    --cv-root "${OUTPUT_ROOT}" \
    --output-dir "${SUMMARY_DIR}"
fi

if truthy "${RUN_SEARCH}"; then
  log "Exporting CREDO search records and selecting final candidates"
  search_required_folds="${SEARCH_REQUIRED_FOLDS:-$(fold_grid_csv)}"
  search_required_seeds="${SEARCH_REQUIRED_SEEDS:-$(seed_grid_csv)}"
  search_min_folds="${SEARCH_MIN_FOLDS:-$(fold_grid_count)}"
  search_min_seeds="${SEARCH_MIN_SEEDS:-$(seed_grid_count)}"
  search_args=(
    "${CREDO_DIR}/runners/export_hnscc_search_records.py"
    --cv-root "${OUTPUT_ROOT}"
    --output-dir "${SEARCH_DIR}"
    --profile "${SEARCH_PROFILE}"
    --required-folds "${search_required_folds}"
    --required-seeds "${search_required_seeds}"
    --min-folds "${search_min_folds}"
    --min-seeds "${search_min_seeds}"
    --overwrite
  )
  if [[ -n "${SEARCH_OBJECTIVES}" ]]; then
    search_args+=(--objectives "${SEARCH_OBJECTIVES}")
  fi
  if [[ -n "${SEARCH_SORT_BY}" ]]; then
    search_args+=(--sort-by "${SEARCH_SORT_BY}")
  fi
  if [[ -n "${SEARCH_CLAIM_CONTROL_NULL_MAX}" ]]; then
    search_args+=(--claim-control-null-max "${SEARCH_CLAIM_CONTROL_NULL_MAX}")
  fi
  if [[ -n "${SEARCH_CLAIM_LOG_MASS_ERROR_MAX}" ]]; then
    search_args+=(--claim-log-mass-error-max "${SEARCH_CLAIM_LOG_MASS_ERROR_MAX}")
  fi
  if [[ -n "${SEARCH_CLAIM_GUIDE_CONCORDANCE_MAX}" ]]; then
    search_args+=(--claim-guide-concordance-max "${SEARCH_CLAIM_GUIDE_CONCORDANCE_MAX}")
  fi
  if truthy "${SEARCH_CLAIM_REQUIRE_GUIDE_CONCORDANCE}"; then
    search_args+=(--claim-require-guide-concordance)
  fi
  run_py "${search_args[@]}"
fi

if truthy "${RUN_BIOLOGY}"; then
  log "Extracting claim-grade biological effect rankings"
  biology_args=(
    "${CREDO_DIR}/analysis/extract_biology_effects.py"
    --cv-root "${OUTPUT_ROOT}"
    --signature-scores "${SIGNATURE_DIR}/signature_group_scores.csv"
    --counterfactual-effects "${BIOLOGY_DIR}/counterfactual_biology_effects_merged.csv"
    --output-dir "${BIOLOGY_DIR}"
    --split test
    --top-n "${BIOLOGY_TOP_N}"
    --claim-grade
    --practical-null-floor-profile hnscc_claim_grade
  )
  run_py "${biology_args[@]}"
fi

log "Pipeline complete"
log "Output root: ${OUTPUT_ROOT}"
log "CV summary: ${SUMMARY_DIR}/cv_summary.md"
log "Search candidates: ${SEARCH_DIR}/final_candidates.csv"
log "Biology table: ${BIOLOGY_DIR}/biological_effects_per_perturbation.csv"
