#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(dirname "${BASH_SOURCE[0]}")/.."
cd "$BUNDLE_ROOT"

# shellcheck source=/dev/null
source scripts/_conda_init.sh
ensure_conda_available || {
  echo "conda is required but was not found on PATH." >&2
  exit 1
}
CONDA_BIN="$(resolve_conda_executable)" || {
  echo "A conda executable path could not be resolved." >&2
  exit 1
}

default_data_path() {
  local candidates=(
    "../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad"
  )
  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "${candidates[0]}"
}

default_split_items() {
  if [[ -n "${SPLIT_ITEMS:-}" ]]; then
    printf '%s\n' "$SPLIT_ITEMS"
    return 0
  fi
  case "${SPLIT_STRATEGY:-}" in
    random_kfold)
      local count="${CV_FOLDS:-4}"
      local joined=""
      local idx=0
      for ((idx=0; idx<count; idx++)); do
        if [[ "$idx" -gt 0 ]]; then
          joined+=";"
        fi
        joined+="$idx"
      done
      printf '%s\n' "$joined"
      ;;
    wta)
      printf '%s\n' "${DEFAULT_WTA_TEST_PAIRS:-wta8,wta11;wta8,wta12;wta10,wta11;wta10,wta12}"
      ;;
    random)
      printf '%s\n' "random"
      ;;
    *)
      echo "Unsupported SPLIT_STRATEGY: ${SPLIT_STRATEGY:-}" >&2
      exit 1
      ;;
  esac
}

build_train_wtas() {
  local test_csv="$1"
  local all_wtas_csv="${ALL_WTAS_CSV:-wta4,wta5,wta6,wta7,wta8,wta9,wta10,wta11,wta12,wta13,wta14,wta15,wta16,wta17,wta18}"
  local -A is_test=()
  local item=""
  IFS=',' read -r -a test_items <<< "$test_csv"
  for item in "${test_items[@]}"; do
    is_test["$item"]=1
  done
  local train=()
  IFS=',' read -r -a all_items <<< "$all_wtas_csv"
  for item in "${all_items[@]}"; do
    if [[ -z "${is_test[$item]:-}" ]]; then
      train+=("$item")
    fi
  done
  local joined=""
  local idx=0
  for item in "${train[@]}"; do
    if [[ "$idx" -gt 0 ]]; then
      joined+=","
    fi
    joined+="$item"
    idx=$((idx + 1))
  done
  printf '%s\n' "$joined"
}

build_split_args() {
  local split_item="$1"
  SPLIT_ARGS=()
  case "$SPLIT_STRATEGY" in
    random_kfold)
      SPLIT_ARGS+=(
        --split-strategy random_kfold
        --cv-folds "${CV_FOLDS:-4}"
        --cv-fold-index "$split_item"
      )
      if [[ -n "${RANDOM_STRATIFY_COLS:-}" ]]; then
        SPLIT_ARGS+=(--random-stratify-cols "$RANDOM_STRATIFY_COLS")
      fi
      ;;
    wta)
      SPLIT_ARGS+=(
        --split-strategy wta
        --wta-column "${WTA_COLUMN:-Library}"
        --train-wtas "$(build_train_wtas "$split_item")"
        --test-wtas "$split_item"
      )
      ;;
    random)
      SPLIT_ARGS+=(
        --split-strategy random
        --train-frac "${TRAIN_FRAC:-0.8}"
      )
      if [[ -n "${RANDOM_STRATIFY_COLS:-}" ]]; then
        SPLIT_ARGS+=(--random-stratify-cols "$RANDOM_STRATIFY_COLS")
      fi
      ;;
    *)
      echo "Unsupported SPLIT_STRATEGY: $SPLIT_STRATEGY" >&2
      exit 1
      ;;
  esac
}

build_common_cmd() {
  local out_dir="$1"
  local cpu_threads="$2"
  CMD=(
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python runners/run_credo_hnscc_full.py
    --data-path "$DATA_PATH"
    --output-dir "$out_dir"
    --latent-source "$LATENT_SOURCE"
    "${SPLIT_ARGS[@]}"
    --seed "$SEED"
    --precision "$PRECISION"
    --state-key "$STATE_KEY"
    --mass-scope "$MASS_SCOPE"
    --control-mode "$CONTROL_MODE"
    --training-schedule "$TRAINING_SCHEDULE"
    --stage-c-epochs "$STAGE_C_EPOCHS"
    --stage-d-epochs "$STAGE_D_EPOCHS"
    --lambda-control-ref "$LAMBDA_CONTROL_REF"
    --control-ref-warmup-epochs "$CONTROL_REF_WARMUP_EPOCHS"
    --n-programs "$N_PROGRAMS"
    --embedding-dim "$EMBEDDING_DIM"
    --mediator-dim "$MEDIATOR_DIM"
    --hidden-dim "$HIDDEN_DIM"
    --depth "$DEPTH"
    --epochs "$EPOCHS"
    --n-particles "$N_PARTICLES"
    --n-steps "$N_STEPS"
    --eval-particles "$EVAL_PARTICLES"
    --eval-steps "$EVAL_STEPS"
    --eval-target-particles "$EVAL_TARGET_PARTICLES"
    --max-train-target-atoms "$MAX_TRAIN_TARGET_ATOMS"
    --n-test-functions "$N_TEST_FUNCTIONS"
    --lambda-weak "$LAMBDA_WEAK"
    --lambda-reg-growth-bias "$LAMBDA_REG_GROWTH_BIAS"
    --max-active-perturbations "$MAX_ACTIVE_PERTURBATIONS"
    --min-cells-p4 "$MIN_CELLS_P4"
    --min-cells-p60 "$MIN_CELLS_P60"
    --cpu-threads "$cpu_threads"
    --cpu-interop-threads "$CPU_INTEROP_THREADS"
  )

  if [[ "$GUIDE_CONFIDENT_ONLY" == "1" ]]; then
    CMD+=(--guide-confident-only)
  else
    CMD+=(--include-nonconfident)
  fi

  if [[ "$ECOLOGICAL_GROWTH" == "1" ]]; then
    CMD+=(--ecology-on)
  else
    CMD+=(--ecology-off)
  fi

  if [[ "$USE_GROWTH_INTERCEPT" == "1" ]]; then
    CMD+=(--growth-intercept-on)
  else
    CMD+=(--growth-intercept-off)
  fi

  if [[ "$USE_STATE_CENTROIDS" == "1" ]]; then
    CMD+=(--use-state-centroids)
  else
    CMD+=(--learned-programs)
  fi

  if [[ "$AUTO_SCALE_BUDGET" == "1" ]]; then
    CMD+=(--auto-scale-budget)
  else
    CMD+=(--no-auto-scale-budget)
  fi

  if [[ "$LATENT_SOURCE" == "vae" ]]; then
    CMD+=(
      --expression-gene-mask-col "$EXPRESSION_GENE_MASK_COL"
      --expression-top-genes "$EXPRESSION_TOP_GENES"
      --vae-latent-dim "$VAE_LATENT_DIM"
      --vae-hidden-dim "$VAE_HIDDEN_DIM"
      --vae-depth "$VAE_DEPTH"
      --vae-dropout "$VAE_DROPOUT"
      --vae-epochs "$VAE_EPOCHS"
      --vae-batch-size "$VAE_BATCH_SIZE"
      --vae-lr "$VAE_LR"
      --vae-weight-decay "$VAE_WEIGHT_DECAY"
      --vae-kl-weight "$VAE_KL_WEIGHT"
      --vae-kl-warmup-epochs "$VAE_KL_WARMUP_EPOCHS"
      --vae-val-frac "$VAE_VAL_FRAC"
      --vae-early-stop-patience "$VAE_EARLY_STOP_PATIENCE"
      --vae-grad-clip "$VAE_GRAD_CLIP"
      --vae-target-sum "$VAE_TARGET_SUM"
      --vae-encode-batch-size "$VAE_ENCODE_BATCH_SIZE"
      --expression-workers "$EXPRESSION_WORKERS"
      --expression-chunk-size "$EXPRESSION_CHUNK_SIZE"
      --vae-hvg-batch-col "$VAE_HVG_BATCH_COL"
      --vae-hvg-time-col "$VAE_HVG_TIME_COL"
      --vae-hvg-min-cells-per-batch "$VAE_HVG_MIN_CELLS_PER_BATCH"
      --vae-preload-dense-max-gb "$VAE_PRELOAD_DENSE_MAX_GB"
      --vae-amp-dtype "$VAE_AMP_DTYPE"
    )
    if [[ "$VAE_USE_RAW" == "1" ]]; then
      CMD+=(--vae-use-raw)
    else
      CMD+=(--no-vae-use-raw)
    fi
    if [[ -n "${VAE_LAYER:-}" ]]; then
      CMD+=(--vae-layer "$VAE_LAYER")
    fi
    if [[ "$VAE_BATCH_AWARE_HVG" == "1" ]]; then
      CMD+=(--vae-batch-aware-hvg)
    else
      CMD+=(--no-vae-batch-aware-hvg)
    fi
    if [[ "$VAE_ALLOW_FULL_GENE_SCAN" == "1" ]]; then
      CMD+=(--vae-allow-full-gene-scan)
    else
      CMD+=(--no-vae-allow-full-gene-scan)
    fi
    if [[ "$VAE_REUSE_ARTIFACT" == "1" ]]; then
      CMD+=(--vae-reuse-artifact)
    else
      CMD+=(--no-vae-reuse-artifact)
    fi
    if [[ "$VAE_USE_AMP" == "1" ]]; then
      CMD+=(--vae-use-amp)
    else
      CMD+=(--no-vae-use-amp)
    fi
  else
    CMD+=(--latent-key "$LATENT_KEY")
  fi

  if [[ "$ACTIVATION_CHECKPOINTING" == "1" ]]; then
    CMD+=(--activation-checkpointing)
  else
    CMD+=(--no-activation-checkpointing)
  fi

  if [[ "${#EXTRA_RUN_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${EXTRA_RUN_ARGS[@]}")
  fi
}

detect_gpu_devices() {
  local raw="${GPU_LIST:-}"
  if [[ -z "$raw" ]]; then
    local pieces=()
    if [[ -n "${GPU_A:-}" ]]; then
      pieces+=("$GPU_A")
    fi
    if [[ -n "${GPU_B:-}" ]]; then
      pieces+=("$GPU_B")
    fi
    if [[ "${#pieces[@]}" -gt 0 ]]; then
      raw="$(IFS=,; printf '%s' "${pieces[*]}")"
    fi
  fi
  if [[ -z "$raw" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    raw="${CUDA_VISIBLE_DEVICES}"
  fi
  if [[ -n "$raw" ]]; then
    local item=""
    IFS=',' read -r -a parsed <<< "$raw"
    for item in "${parsed[@]}"; do
      item="${item// /}"
      [[ -n "$item" ]] && printf '%s\n' "$item"
    done
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}'
    return 0
  fi
  return 1
}

default_var() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    printf -v "$name" '%s' "$value"
    export "$name"
  fi
}

is_disabled_optional_name() {
  local value="${1:-}"
  value="${value,,}"
  [[ -z "$value" || "$value" == "none" || "$value" == "null" || "$value" == "na" ]]
}

split_output_name() {
  local fold_idx="$1"
  local split_item="$2"
  if [[ "${OUTPUT_SPLIT_LABEL_DIRS:-0}" == "1" ]]; then
    case "$SPLIT_STRATEGY" in
      random_kfold)
        printf 'fold_%s\n' "$split_item"
        return 0
        ;;
      random)
        printf 'random\n'
        return 0
        ;;
      wta)
        printf 'wta_%s\n' "${split_item//,/__}"
        return 0
        ;;
    esac
  fi
  printf 'fold_%s\n' "$fold_idx"
}

apply_h100_defaults() {
  default_var CPU_INTEROP_THREADS 2
  default_var PRECISION bf16
  default_var GUIDE_CONFIDENT_ONLY 1
  default_var STATE_KEY None
  default_var RANDOM_STRATIFY_COLS "Time point,perturbation_id"
  default_var MASS_SCOPE subset_only
  default_var CONTROL_MODE soft_ref
  default_var TRAINING_SCHEDULE staged
  default_var STAGE_C_EPOCHS 150
  default_var STAGE_D_EPOCHS 150
  default_var ECOLOGICAL_GROWTH 1
  default_var USE_GROWTH_INTERCEPT 1
  default_var USE_STATE_CENTROIDS 0
  default_var ACTIVATION_CHECKPOINTING 1
  default_var LATENT_SOURCE vae
  default_var LATENT_KEY X_pca
  default_var EXPRESSION_GENE_MASK_COL hv_gene
  default_var EXPRESSION_TOP_GENES 2000
  default_var VAE_LATENT_DIM 50
  default_var VAE_HIDDEN_DIM 512
  default_var VAE_DEPTH 2
  default_var VAE_DROPOUT 0.1
  default_var VAE_EPOCHS 50
  default_var VAE_BATCH_SIZE 1024
  default_var VAE_LR 1e-3
  default_var VAE_WEIGHT_DECAY 1e-6
  default_var VAE_KL_WEIGHT 1e-3
  default_var VAE_ENCODE_BATCH_SIZE 4096
  default_var EXPRESSION_WORKERS 8
  default_var EXPRESSION_CHUNK_SIZE 1024
  default_var LAMBDA_CONTROL_REF 5e-4
  default_var CONTROL_REF_WARMUP_EPOCHS 150
  default_var N_STEPS 32
  default_var EVAL_STEPS 32
  default_var N_TEST_FUNCTIONS 8
  default_var LAMBDA_WEAK 0.10
  default_var LAMBDA_REG_GROWTH_BIAS 1e-4
  default_var AUTO_SCALE_BUDGET 0
}

apply_h100_heavy_c_profile() {
  apply_h100_defaults
  default_var EMBEDDING_DIM 56
  default_var N_PROGRAMS 20
  default_var MEDIATOR_DIM 56
  default_var HIDDEN_DIM 896
  default_var DEPTH 5
  default_var EPOCHS 1800
  default_var N_PARTICLES 320
  default_var EVAL_PARTICLES 1408
  default_var EVAL_TARGET_PARTICLES 3584
  default_var MAX_TRAIN_TARGET_ATOMS 1920
}

apply_h100_heavy_f_profile() {
  default_var N_STEPS 24
  default_var EVAL_STEPS 24
  apply_h100_defaults
  default_var MAX_ACTIVE_PERTURBATIONS 8
  default_var EMBEDDING_DIM 96
  default_var N_PROGRAMS 32
  default_var MEDIATOR_DIM 96
  default_var HIDDEN_DIM 1024
  default_var DEPTH 6
  default_var EPOCHS 1800
  default_var N_PARTICLES 256
  default_var EVAL_PARTICLES 1024
  default_var EVAL_TARGET_PARTICLES 2560
  default_var MAX_TRAIN_TARGET_ATOMS 1536
}

apply_h100_heavy_f_full_profile() {
  default_var EXPRESSION_CHUNK_SIZE 2048
  default_var VAE_BATCH_SIZE 2048
  default_var VAE_ENCODE_BATCH_SIZE 8192
  default_var N_STEPS 28
  default_var EVAL_STEPS 28
  default_var N_TEST_FUNCTIONS 12
  apply_h100_defaults
  default_var MAX_ACTIVE_PERTURBATIONS 16
  default_var EMBEDDING_DIM 96
  default_var N_PROGRAMS 32
  default_var MEDIATOR_DIM 96
  default_var HIDDEN_DIM 1024
  default_var DEPTH 6
  default_var EPOCHS 1800
  default_var N_PARTICLES 512
  default_var EVAL_PARTICLES 2048
  default_var EVAL_TARGET_PARTICLES 4096
  default_var MAX_TRAIN_TARGET_ATOMS 3072
}

apply_local_heavy_c_vae_9gb_profile() {
  default_var THREADS_PER_GPU 16
  default_var CPU_INTEROP_THREADS 2
  default_var PIN_CPU 0
  default_var PRECISION bf16
  default_var GUIDE_CONFIDENT_ONLY 1
  default_var STATE_KEY None
  default_var RANDOM_STRATIFY_COLS "Time point,perturbation_id"
  default_var MASS_SCOPE subset_only
  default_var CONTROL_MODE soft_ref
  default_var TRAINING_SCHEDULE staged
  default_var STAGE_C_EPOCHS 15
  default_var STAGE_D_EPOCHS 45
  default_var ECOLOGICAL_GROWTH 1
  default_var USE_GROWTH_INTERCEPT 1
  default_var USE_STATE_CENTROIDS 0
  default_var ACTIVATION_CHECKPOINTING 0
  default_var LATENT_SOURCE vae
  default_var LATENT_KEY X_vae
  default_var EXPRESSION_GENE_MASK_COL hv_gene
  default_var EXPRESSION_TOP_GENES 1000
  default_var VAE_LATENT_DIM 32
  default_var VAE_HIDDEN_DIM 256
  default_var VAE_DEPTH 2
  default_var VAE_DROPOUT 0.1
  default_var VAE_EPOCHS 20
  default_var VAE_BATCH_SIZE 512
  default_var VAE_LR 1e-3
  default_var VAE_WEIGHT_DECAY 1e-6
  default_var VAE_KL_WEIGHT 1e-3
  default_var VAE_KL_WARMUP_EPOCHS 10
  default_var VAE_VAL_FRAC 0.05
  default_var VAE_EARLY_STOP_PATIENCE 5
  default_var VAE_GRAD_CLIP 1.0
  default_var VAE_TARGET_SUM 10000
  default_var VAE_ENCODE_BATCH_SIZE 4096
  default_var EXPRESSION_WORKERS 2
  default_var EXPRESSION_CHUNK_SIZE 512
  default_var VAE_BATCH_AWARE_HVG 1
  default_var VAE_HVG_BATCH_COL Library
  default_var VAE_HVG_TIME_COL "Time point"
  default_var VAE_HVG_MIN_CELLS_PER_BATCH 256
  default_var VAE_ALLOW_FULL_GENE_SCAN 0
  default_var VAE_PRELOAD_DENSE_MAX_GB 2.0
  default_var VAE_REUSE_ARTIFACT 1
  default_var VAE_USE_AMP 1
  default_var VAE_AMP_DTYPE bf16
  default_var LAMBDA_CONTROL_REF 5e-4
  default_var CONTROL_REF_WARMUP_EPOCHS 15
  default_var EMBEDDING_DIM 48
  default_var N_PROGRAMS 16
  default_var MEDIATOR_DIM 48
  default_var HIDDEN_DIM 512
  default_var DEPTH 4
  default_var EPOCHS 60
  default_var N_PARTICLES 64
  default_var N_STEPS 6
  default_var EVAL_PARTICLES 192
  default_var EVAL_STEPS 6
  default_var EVAL_TARGET_PARTICLES 768
  default_var MAX_TRAIN_TARGET_ATOMS 768
  default_var N_TEST_FUNCTIONS 4
  default_var LAMBDA_WEAK 0.05
  default_var LAMBDA_REG_GROWTH_BIAS 1e-4
  default_var MAX_ACTIVE_PERTURBATIONS 12
  default_var AUTO_SCALE_BUDGET 0
  default_var MIN_CELLS_P4 50
  default_var MIN_CELLS_P60 50
}

apply_credo_profile() {
  case "$1" in
    ""|none)
      ;;
    h100_heavy_c)
      apply_h100_heavy_c_profile
      ;;
    h100_heavy_f)
      apply_h100_heavy_f_profile
      ;;
    h100_heavy_f_full)
      apply_h100_heavy_f_full_profile
      ;;
    local_heavy_c_vae_9gb)
      apply_local_heavy_c_vae_9gb_profile
      ;;
    *)
      echo "Unsupported CREDO_PROFILE: $1" >&2
      exit 1
      ;;
  esac
}

run_joint_cv() {
  local gpu_a="${GPU_A:-0}"
  local gpu_b="${GPU_B:-1}"
  local cpu_threads="${CPU_THREADS:-$(nproc)}"
  if [[ "$gpu_a" == "$gpu_b" ]]; then
    echo "GPU_A and GPU_B must be different for RUN_MODE=joint." >&2
    exit 1
  fi
  local multi_gpu_devices="${MULTI_GPU_DEVICES:-$gpu_a,$gpu_b}"

  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$cpu_threads}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$cpu_threads}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$cpu_threads}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$cpu_threads}"

  local split_item=""
  local fold_idx=0
  for split_item in "${SPLIT_ITEMS_ARRAY[@]}"; do
    local out_dir="$SETTING_ROOT/$(split_output_name "$fold_idx" "$split_item")"
    if [[ -f "$out_dir/results_summary.json" ]]; then
      echo "Skipping completed $SETTING_TAG split $split_item" >&2
      fold_idx=$((fold_idx + 1))
      continue
    fi
    mkdir -p "$out_dir"
    build_split_args "$split_item"
    build_common_cmd "$out_dir" "$cpu_threads"
    CMD+=(--multi-gpu-devices "$multi_gpu_devices")
    {
      echo "CREDO resource plan: mode=joint split=$split_item fold=$fold_idx seed=$SEED devices=$multi_gpu_devices cpu_threads=$cpu_threads expression_workers=$EXPRESSION_WORKERS expression_chunk_size=$EXPRESSION_CHUNK_SIZE"
      printf 'CREDO command:'
      printf ' %q' "${CMD[@]}"
      printf '\n'
    } > "$out_dir/console.log"
    if ! "${CMD[@]}" >> "$out_dir/console.log" 2>&1; then
      echo "CREDO fold failed: split=$split_item fold=$fold_idx log=$out_dir/console.log" >&2
      tail -n 80 "$out_dir/console.log" >&2 || true
      exit 1
    fi
    fold_idx=$((fold_idx + 1))
  done
}

run_parallel_cv() {
  local pin_cpu="${PIN_CPU:-1}"
  local nproc_total="${NPROC_TOTAL:-$(nproc)}"
  mapfile -t GPU_DEVICES < <(detect_gpu_devices)
  if [[ "${#GPU_DEVICES[@]}" -eq 0 ]]; then
    echo "No GPU devices could be detected. Set GPU_LIST explicitly if needed." >&2
    exit 1
  fi

  local gpu_count="${#GPU_DEVICES[@]}"
  if [[ "$gpu_count" -gt "${#SPLIT_ITEMS_ARRAY[@]}" ]]; then
    gpu_count="${#SPLIT_ITEMS_ARRAY[@]}"
  fi
  local threads_per_gpu="${THREADS_PER_GPU:-$(( nproc_total / gpu_count ))}"
  if [[ "$threads_per_gpu" -lt 1 ]]; then
    threads_per_gpu=1
  fi

  run_parallel_fold() {
    local gpu="$1"
    local slot_idx="$2"
    local fold_idx="$3"
    local split_item="$4"
    local out_dir="$SETTING_ROOT/$(split_output_name "$fold_idx" "$split_item")"
    local core_range=""
    if [[ -f "$out_dir/results_summary.json" ]]; then
      echo "Skipping completed $SETTING_TAG split $split_item" >&2
      return 1
    fi
    if [[ "$pin_cpu" == "1" ]]; then
      local start_core=$(( slot_idx * threads_per_gpu ))
      local end_core=$(( start_core + threads_per_gpu - 1 ))
      if [[ "$start_core" -lt "$nproc_total" ]]; then
        if [[ "$end_core" -ge "$nproc_total" ]]; then
          end_core=$(( nproc_total - 1 ))
        fi
        core_range="${start_core}-${end_core}"
      fi
    fi
    mkdir -p "$out_dir"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
      export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
      export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
      export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$threads_per_gpu}"
      export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$threads_per_gpu}"
      export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$threads_per_gpu}"
      export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$threads_per_gpu}"
      build_split_args "$split_item"
      build_common_cmd "$out_dir" "$threads_per_gpu"
      echo "CREDO resource plan: mode=parallel gpu=$gpu slot=$slot_idx split=$split_item fold=$fold_idx seed=$SEED threads_per_gpu=$threads_per_gpu expression_workers=$EXPRESSION_WORKERS expression_chunk_size=$EXPRESSION_CHUNK_SIZE"
      echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
      printf 'CREDO command:'
      printf ' %q' "${CMD[@]}"
      printf '\n'
      if [[ "$pin_cpu" == "1" ]] && [[ -n "$core_range" ]] && command -v taskset >/dev/null 2>&1; then
        echo "CPU affinity: $core_range"
        taskset -c "$core_range" "${CMD[@]}"
      else
        echo "CPU affinity: unpinned"
        "${CMD[@]}"
      fi
    ) > "$out_dir/console.log" 2>&1 &
    RUN_FOLD_PID=$!
    RUN_FOLD_LOG="$out_dir/console.log"
    return 0
  }

  local fold_idx=0
  while [[ "$fold_idx" -lt "${#SPLIT_ITEMS_ARRAY[@]}" ]]; do
    local pids=()
    local logs=()
    local slot_idx=0
    for ((slot_idx=0; slot_idx<gpu_count && fold_idx<${#SPLIT_ITEMS_ARRAY[@]}; slot_idx++, fold_idx++)); do
      local gpu="${GPU_DEVICES[$slot_idx]}"
      local split_item="${SPLIT_ITEMS_ARRAY[$fold_idx]}"
      if run_parallel_fold "$gpu" "$slot_idx" "$fold_idx" "$split_item"; then
        pids+=("$RUN_FOLD_PID")
        logs+=("$RUN_FOLD_LOG")
      fi
    done
    local idx=0
    local failed=0
    for idx in "${!pids[@]}"; do
      if ! wait "${pids[$idx]}"; then
        failed=1
        echo "CREDO fold failed: log=${logs[$idx]}" >&2
        tail -n 80 "${logs[$idx]}" >&2 || true
      fi
    done
    if [[ "$failed" != "0" ]]; then
      exit 1
    fi
  done
}

ENV_NAME="${ENV_NAME:-cape-hnscc}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
CREDO_PROFILE="${CREDO_PROFILE:-}"
apply_credo_profile "$CREDO_PROFILE"
RUN_MODE="${RUN_MODE:-joint}"
SETTING_TAG="${SETTING_TAG:?SETTING_TAG must be set by the wrapper script}"
RUN_ROOT_PREFIX="${RUN_ROOT_PREFIX:?RUN_ROOT_PREFIX must be set by the wrapper script}"
DATA_PATH="${DATA_PATH:-$(default_data_path)}"
CV_ROOT="${CV_ROOT:-${RUN_ROOT_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
SETTING_ROOT="$CV_ROOT/$SETTING_TAG"
SEED="${SEED:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:?SPLIT_STRATEGY must be set by the wrapper script}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-2}"
PRECISION="${PRECISION:-bf16}"
GUIDE_CONFIDENT_ONLY="${GUIDE_CONFIDENT_ONLY:-1}"
STATE_KEY="${STATE_KEY:-None}"
RANDOM_STRATIFY_COLS="${RANDOM_STRATIFY_COLS:-Time point,perturbation_id}"
MASS_SCOPE="${MASS_SCOPE:-subset_only}"
CONTROL_MODE="${CONTROL_MODE:-soft_ref}"
TRAINING_SCHEDULE="${TRAINING_SCHEDULE:-staged}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-150}"
STAGE_D_EPOCHS="${STAGE_D_EPOCHS:-150}"
LAMBDA_CONTROL_REF="${LAMBDA_CONTROL_REF:-5e-4}"
CONTROL_REF_WARMUP_EPOCHS="${CONTROL_REF_WARMUP_EPOCHS:-150}"
ECOLOGICAL_GROWTH="${ECOLOGICAL_GROWTH:-1}"
USE_GROWTH_INTERCEPT="${USE_GROWTH_INTERCEPT:-1}"
USE_STATE_CENTROIDS="${USE_STATE_CENTROIDS:-0}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
LATENT_SOURCE="${LATENT_SOURCE:-vae}"
LATENT_KEY="${LATENT_KEY:-X_pca}"
EXPRESSION_GENE_MASK_COL="${EXPRESSION_GENE_MASK_COL:-hv_gene}"
EXPRESSION_TOP_GENES="${EXPRESSION_TOP_GENES:-2000}"
VAE_LATENT_DIM="${VAE_LATENT_DIM:-50}"
VAE_HIDDEN_DIM="${VAE_HIDDEN_DIM:-512}"
VAE_DEPTH="${VAE_DEPTH:-2}"
VAE_DROPOUT="${VAE_DROPOUT:-0.1}"
VAE_EPOCHS="${VAE_EPOCHS:-50}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-1024}"
VAE_LR="${VAE_LR:-1e-3}"
VAE_WEIGHT_DECAY="${VAE_WEIGHT_DECAY:-1e-6}"
VAE_KL_WEIGHT="${VAE_KL_WEIGHT:-1e-3}"
VAE_KL_WARMUP_EPOCHS="${VAE_KL_WARMUP_EPOCHS:-20}"
VAE_VAL_FRAC="${VAE_VAL_FRAC:-0.1}"
VAE_EARLY_STOP_PATIENCE="${VAE_EARLY_STOP_PATIENCE:-15}"
VAE_GRAD_CLIP="${VAE_GRAD_CLIP:-1.0}"
VAE_LAYER="${VAE_LAYER:-}"
VAE_USE_RAW="${VAE_USE_RAW:-0}"
VAE_TARGET_SUM="${VAE_TARGET_SUM:-10000}"
VAE_ENCODE_BATCH_SIZE="${VAE_ENCODE_BATCH_SIZE:-4096}"
EXPRESSION_WORKERS="${EXPRESSION_WORKERS:-0}"
EXPRESSION_CHUNK_SIZE="${EXPRESSION_CHUNK_SIZE:-1024}"
VAE_BATCH_AWARE_HVG="${VAE_BATCH_AWARE_HVG:-1}"
VAE_HVG_BATCH_COL="${VAE_HVG_BATCH_COL:-Library}"
VAE_HVG_TIME_COL="${VAE_HVG_TIME_COL:-Time point}"
VAE_HVG_MIN_CELLS_PER_BATCH="${VAE_HVG_MIN_CELLS_PER_BATCH:-256}"
VAE_ALLOW_FULL_GENE_SCAN="${VAE_ALLOW_FULL_GENE_SCAN:-0}"
VAE_PRELOAD_DENSE_MAX_GB="${VAE_PRELOAD_DENSE_MAX_GB:-4.0}"
VAE_REUSE_ARTIFACT="${VAE_REUSE_ARTIFACT:-1}"
VAE_USE_AMP="${VAE_USE_AMP:-1}"
VAE_AMP_DTYPE="${VAE_AMP_DTYPE:-bf16}"
EMBEDDING_DIM="${EMBEDDING_DIM:?EMBEDDING_DIM must be set by the wrapper script}"
N_PROGRAMS="${N_PROGRAMS:?N_PROGRAMS must be set by the wrapper script}"
MEDIATOR_DIM="${MEDIATOR_DIM:?MEDIATOR_DIM must be set by the wrapper script}"
HIDDEN_DIM="${HIDDEN_DIM:?HIDDEN_DIM must be set by the wrapper script}"
DEPTH="${DEPTH:?DEPTH must be set by the wrapper script}"
EPOCHS="${EPOCHS:-1800}"
N_PARTICLES="${N_PARTICLES:?N_PARTICLES must be set by the wrapper script}"
N_STEPS="${N_STEPS:-32}"
EVAL_PARTICLES="${EVAL_PARTICLES:?EVAL_PARTICLES must be set by the wrapper script}"
EVAL_STEPS="${EVAL_STEPS:-32}"
EVAL_TARGET_PARTICLES="${EVAL_TARGET_PARTICLES:?EVAL_TARGET_PARTICLES must be set by the wrapper script}"
MAX_TRAIN_TARGET_ATOMS="${MAX_TRAIN_TARGET_ATOMS:?MAX_TRAIN_TARGET_ATOMS must be set by the wrapper script}"
N_TEST_FUNCTIONS="${N_TEST_FUNCTIONS:-8}"
LAMBDA_WEAK="${LAMBDA_WEAK:-0.10}"
LAMBDA_REG_GROWTH_BIAS="${LAMBDA_REG_GROWTH_BIAS:-1e-4}"
MAX_ACTIVE_PERTURBATIONS="${MAX_ACTIVE_PERTURBATIONS:-0}"
AUTO_SCALE_BUDGET="${AUTO_SCALE_BUDGET:-0}"
MIN_CELLS_P4="${MIN_CELLS_P4:-20}"
MIN_CELLS_P60="${MIN_CELLS_P60:-20}"
EXTRA_RUN_ARGS=("$@")
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
if [[ -z "${SUMMARY_RANKING_MODE:-}" ]]; then
  if is_disabled_optional_name "$STATE_KEY"; then
    SUMMARY_RANKING_MODE="balanced"
  else
    SUMMARY_RANKING_MODE="test_acc"
  fi
fi

mkdir -p "$SETTING_ROOT"
IFS=';' read -r -a SPLIT_ITEMS_ARRAY <<< "$(default_split_items)"
echo "CREDO run root: $CV_ROOT"
echo "CREDO setting: $SETTING_TAG"
echo "CREDO mode: $RUN_MODE splits=${#SPLIT_ITEMS_ARRAY[@]} seed=$SEED py_hash_seed=$PYTHONHASHSEED skip_summary=$SKIP_SUMMARY summary_ranking=$SUMMARY_RANKING_MODE"

case "$RUN_MODE" in
  joint)
    run_joint_cv
    ;;
  parallel)
    run_parallel_cv
    ;;
  *)
    echo "Unsupported RUN_MODE: $RUN_MODE" >&2
    exit 1
    ;;
esac

if [[ "$SKIP_SUMMARY" != "1" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python runners/summarize_hnscc_cv.py \
    --cv-root "$CV_ROOT" \
    --output-dir "$CV_ROOT" \
    --group-by setting \
    --ranking-mode "$SUMMARY_RANKING_MODE"
fi

echo "$CV_ROOT"
