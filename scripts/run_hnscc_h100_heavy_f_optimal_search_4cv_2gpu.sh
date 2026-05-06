#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

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

detect_gpu_devices() {
  local raw="${GPU_LIST:-}"
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

is_disabled_optional_name() {
  local value="${1:-}"
  value="${value,,}"
  [[ -z "$value" || "$value" == "none" || "$value" == "null" || "$value" == "na" ]]
}

parse_search_folds() {
  SEARCH_FOLD_ITEMS=()
  local raw="${SEARCH_FOLDS:-0,1,2,3}"
  local item=""
  IFS=',' read -r -a parsed_folds <<< "$raw"
  for item in "${parsed_folds[@]}"; do
    item="${item// /}"
    [[ -z "$item" ]] && continue
    if ! [[ "$item" =~ ^[0-9]+$ ]]; then
      echo "Invalid SEARCH_FOLDS item: $item" >&2
      exit 1
    fi
    if [[ "$item" -ge "$CV_FOLDS" ]]; then
      echo "SEARCH_FOLDS item $item is outside CV_FOLDS=$CV_FOLDS." >&2
      exit 1
    fi
    SEARCH_FOLD_ITEMS+=("$item")
  done
  if [[ "${#SEARCH_FOLD_ITEMS[@]}" -eq 0 ]]; then
    echo "SEARCH_FOLDS resolved to no folds." >&2
    exit 1
  fi
}

load_settings() {
  if [[ -n "${SETTINGS_FILE:-}" ]]; then
    SETTINGS=()
    local line=""
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
      SETTINGS+=("$line")
    done < "$SETTINGS_FILE"
    return 0
  fi

  local preset="${SETTINGS_PRESET:-generalization_h1408}"
  # tag|embedding|mediator|programs|hidden|depth|particles|eval_particles|eval_target|max_train_atoms|max_active|lambda_ctrl|lambda_weak|stage_c|stage_d|epochs|use_state_centroids|lambda_reg_growth_bias|use_growth_intercept
  case "$preset" in
    generalization_h1408)
      # Anchored on observed h1408_d7_p576_active24 ~=72 GB allocated:
      # train state acc improved, but test acc/expansion lagged. Fold-0 probes
      # show fixed state centroids are poor, while disabling the growth
      # intercept gives the best state accuracy so far. Refine around learned
      # programs with growth_intercept=0, longer rollouts, and 2000 epochs,
      # then vary capacity and regularization to recover UOT/mass without
      # losing state accuracy. Tags include SEARCH_N_STEPS so explicit step
      # overrides do not collide with cached results from another rollout
      # length.
      SETTINGS=(
        "heavy_f_h1152_d7_prog36_p576_s${SEARCH_N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|104|104|36|1152|7|576|2304|4608|4096|24|1e-3|0.15|150|150|2000|0|3e-4|0"
        "heavy_f_h1280_d7_prog40_p576_s${SEARCH_N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|112|112|40|1280|7|576|2304|4608|4096|24|1e-3|0.15|150|150|2000|0|3e-4|0"
        "heavy_f_h1280_d7_prog40_p576_s${SEARCH_N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|112|112|40|1280|7|576|2304|4608|4096|24|2e-3|0.20|150|150|2000|0|5e-4|0"
        "heavy_f_h1280_d7_prog40_p576_s${SEARCH_N_STEPS}_active24_lc3e3_lw25_gr1e3_nogint|112|112|40|1280|7|576|2304|4608|4096|24|3e-3|0.25|150|150|2000|0|1e-3|0"
        "heavy_f_h1408_d7_prog44_p512_s${SEARCH_N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|1e-3|0.15|150|150|2000|0|3e-4|0"
        "heavy_f_h1408_d7_prog44_p512_s${SEARCH_N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|150|150|2000|0|5e-4|0"
      )
      ;;
    high_vram_78)
      # Anchored on observed h1024_d6_p512_active24 ~=54 GB allocated.
      # Active perturbations and max_atoms did not materially move peak VRAM,
      # so this ladder primarily increases width/depth/particles while staying
      # below the known-OOM h2048_d7_p608 class.
      SETTINGS=(
        "heavy_f_h1280_d7_prog40_p576_active24_lc5e4_lw10|112|112|40|1280|7|576|2304|4608|4096|24|5e-4|0.10|150|150|1500"
        "heavy_f_h1408_d7_prog44_p576_active24_lc5e4_lw10|120|120|44|1408|7|576|2304|4608|4096|24|5e-4|0.10|150|150|1500"
        "heavy_f_h1536_d7_prog48_p576_active20_lc5e4_lw10|128|128|48|1536|7|576|2304|4608|4096|20|5e-4|0.10|150|150|1500"
        "heavy_f_h1664_d7_prog52_p576_active20_lc5e4_lw10|136|136|52|1664|7|576|2304|4608|4096|20|5e-4|0.10|150|150|1500"
        "heavy_f_h1792_d7_prog56_p512_active20_lc5e4_lw10|144|144|56|1792|7|512|2048|4096|4096|20|5e-4|0.10|150|150|1500"
      )
      ;;
    conservative)
      SETTINGS=(
        "heavy_f_h896_d5_prog28_p512_active16_lc5e4_lw10|80|80|28|896|5|512|2048|4096|3072|16|5e-4|0.10|150|150|1500"
        "heavy_f_h1024_d6_prog32_p512_active16_lc5e4_lw10|96|96|32|1024|6|512|2048|4096|3072|16|5e-4|0.10|150|150|1500"
        "heavy_f_h1024_d6_prog32_p640_active16_lc5e4_lw10|96|96|32|1024|6|640|2560|5120|3584|16|5e-4|0.10|150|150|1500"
        "heavy_f_h1024_d6_prog40_p512_active16_lc5e4_lw10|96|96|40|1024|6|512|2048|4096|3072|16|5e-4|0.10|150|150|1500"
        "heavy_f_h1024_d6_prog32_p512_active16_lc1e3_lw15|96|96|32|1024|6|512|2048|4096|3072|16|1e-3|0.15|150|150|1500"
      )
      ;;
    *)
      echo "Unsupported SETTINGS_PRESET: $preset" >&2
      echo "Use generalization_h1408, high_vram_78, conservative, or provide SETTINGS_FILE." >&2
      exit 1
      ;;
  esac
}

ENV_NAME="${ENV_NAME:-cape-hnscc}"
CV_FOLDS="${CV_FOLDS:-4}"
SEARCH_N_STEPS="${SEARCH_N_STEPS:-28}"
SEARCH_EVAL_STEPS="${SEARCH_EVAL_STEPS:-$SEARCH_N_STEPS}"
SEARCH_ROOT="${CV_ROOT:-${RUN_ROOT_PREFIX:-runs/hnscc_random_h100_heavy_f_optimal_search_4cv_2gpu_$(date +%Y%m%d_%H%M%S)}}"
STATE_KEY="${STATE_KEY:-Cell type annotation}"
RANDOM_STRATIFY_COLS="${RANDOM_STRATIFY_COLS:-Time point,perturbation_id}"
if [[ -z "${SUMMARY_RANKING_MODE:-}" ]]; then
  if is_disabled_optional_name "$STATE_KEY"; then
    SUMMARY_RANKING_MODE="balanced"
  else
    SUMMARY_RANKING_MODE="test_acc"
  fi
fi

mapfile -t GPU_DEVICES < <(detect_gpu_devices)
if [[ "${#GPU_DEVICES[@]}" -eq 0 ]]; then
  echo "No GPU devices could be detected. Set GPU_LIST explicitly if needed." >&2
  exit 1
fi
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-70000}"
if [[ "${ALLOW_SMALL_GPU:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  for gpu in "${GPU_DEVICES[@]}"; do
    gpu_mem_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "$gpu_mem_mb" ]] && [[ "$gpu_mem_mb" =~ ^[0-9]+$ ]] && [[ "$gpu_mem_mb" -lt "$MIN_GPU_MEM_MB" ]]; then
      echo "GPU $gpu has ${gpu_mem_mb} MB, below MIN_GPU_MEM_MB=$MIN_GPU_MEM_MB for this H100 search." >&2
      echo "Use the local smoke script or set ALLOW_SMALL_GPU=1 only for deliberate dry runs." >&2
      exit 1
    fi
  done
fi
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${#GPU_DEVICES[@]}}"
if [[ "$MAX_PARALLEL_JOBS" -gt "${#GPU_DEVICES[@]}" ]]; then
  MAX_PARALLEL_JOBS="${#GPU_DEVICES[@]}"
fi
if [[ "$MAX_PARALLEL_JOBS" -lt 1 ]]; then
  MAX_PARALLEL_JOBS=1
fi
ACTIVE_GPU_DEVICES=("${GPU_DEVICES[@]:0:$MAX_PARALLEL_JOBS}")
THREADS_PER_SETTING="${THREADS_PER_GPU:-$(( $(nproc) / ${#ACTIVE_GPU_DEVICES[@]} ))}"
if [[ "$THREADS_PER_SETTING" -lt 1 ]]; then
  THREADS_PER_SETTING=1
fi

load_settings
parse_search_folds
mkdir -p "$SEARCH_ROOT"

echo "CREDO optimal search root: $SEARCH_ROOT"
echo "CREDO optimal search settings: ${#SETTINGS[@]}"
echo "CREDO optimal search settings preset: ${SETTINGS_PRESET:-generalization_h1408}"
echo "CREDO optimal search CV folds: $CV_FOLDS"
echo "CREDO optimal search queued folds: ${SEARCH_FOLD_ITEMS[*]}"
echo "CREDO optimal search train/eval steps: $SEARCH_N_STEPS/$SEARCH_EVAL_STEPS"
echo "CREDO optimal search state key: $STATE_KEY"
echo "CREDO optimal search summary ranking: $SUMMARY_RANKING_MODE"
echo "CREDO optimal search fold queue: jobs=$MAX_PARALLEL_JOBS gpus=${ACTIVE_GPU_DEVICES[*]} threads_per_job=$THREADS_PER_SETTING"
echo "CREDO optimal search setting table:"
printf '%s\n' "${SETTINGS[@]}" | awk -F'|' -v steps="$SEARCH_N_STEPS" -v eval_steps="$SEARCH_EVAL_STEPS" '{cent=($17=="" ? "0" : $17); gr=($18=="" ? "1e-4" : $18); gint=($19=="" ? "1" : $19); printf "  %s hidden=%s depth=%s programs=%s particles=%s steps=%s eval_steps=%s active=%s max_atoms=%s eval_particles=%s eval_target=%s lam_ctrl=%s lam_weak=%s growth_reg=%s state_centroids=%s growth_intercept=%s epochs=%s\n", $1, $5, $6, $4, $7, steps, eval_steps, $11, $10, $8, $9, $12, $13, gr, cent, gint, $16}'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "CREDO optimal search dry run complete."
  exit 0
fi

EXTRA_RUN_ARGS=("$@")
FREE_GPUS=("${ACTIVE_GPU_DEVICES[@]}")
RUNNING_PIDS=()
declare -A PID_LABEL=()
declare -A PID_LOG=()
declare -A PID_GPU=()
FAILED_JOBS=0

cleanup_running_jobs() {
  local status=$?
  if [[ "$status" -ne 0 ]] && [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; then
    echo "Stopping ${#RUNNING_PIDS[@]} running CREDO search job(s)." >&2
    kill "${RUNNING_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_running_jobs EXIT

remove_running_pid() {
  local remove_pid="$1"
  local kept=()
  local pid=""
  for pid in "${RUNNING_PIDS[@]}"; do
    if [[ "$pid" != "$remove_pid" ]]; then
      kept+=("$pid")
    fi
  done
  RUNNING_PIDS=("${kept[@]}")
}

wait_for_one_job() {
  local finished_pid=""
  local status=0
  set +e
  wait -n -p finished_pid "${RUNNING_PIDS[@]}"
  status=$?
  set -e
  if [[ -z "$finished_pid" ]]; then
    echo "CREDO optimal search scheduler could not resolve a finished job (wait status=$status)." >&2
    FAILED_JOBS=$((FAILED_JOBS + 1))
    return 0
  fi

  local label="${PID_LABEL[$finished_pid]:-unknown}"
  local log_path="${PID_LOG[$finished_pid]:-}"
  local gpu="${PID_GPU[$finished_pid]:-unknown}"
  remove_running_pid "$finished_pid"
  FREE_GPUS+=("$gpu")
  unset "PID_LABEL[$finished_pid]" "PID_LOG[$finished_pid]" "PID_GPU[$finished_pid]"

  if [[ "$status" -ne 0 ]]; then
    FAILED_JOBS=$((FAILED_JOBS + 1))
    echo "CREDO optimal search job failed: $label gpu=$gpu log=$log_path" >&2
    if [[ -n "$log_path" ]]; then
      tail -n 100 "$log_path" >&2 || true
    fi
  else
    echo "Completed job: $label gpu=$gpu"
  fi
}

wait_for_free_gpu() {
  while [[ "${#FREE_GPUS[@]}" -eq 0 ]]; do
    wait_for_one_job
  done
}

launch_fold_job() {
  local gpu="$1"
  local fold="$2"
  local setting="$3"
  local tag embedding mediator programs hidden depth particles eval_particles eval_target max_atoms max_active lambda_ctrl lambda_weak stage_c stage_d epochs use_state_centroids lambda_reg_growth_bias use_growth_intercept
  IFS='|' read -r tag embedding mediator programs hidden depth particles eval_particles eval_target max_atoms max_active lambda_ctrl lambda_weak stage_c stage_d epochs use_state_centroids lambda_reg_growth_bias use_growth_intercept <<< "$setting"
  local setting_epochs="${SEARCH_EPOCHS:-$epochs}"
  use_state_centroids="${use_state_centroids:-0}"
  lambda_reg_growth_bias="${lambda_reg_growth_bias:-1e-4}"
  use_growth_intercept="${use_growth_intercept:-1}"
  local log_path="$SEARCH_ROOT/${tag}.fold_${fold}.launcher.log"

  echo "Launching job: $tag fold_$fold gpu=$gpu"
  (
    export ENV_NAME="$ENV_NAME"
    export CREDO_PROFILE=h100_heavy_f_full
    export RUN_MODE=parallel
    export SKIP_SUMMARY=1
    export GPU_LIST="$gpu"
    export PIN_CPU=0
    export OUTPUT_SPLIT_LABEL_DIRS=1
    export CV_ROOT="$SEARCH_ROOT"
    export RUN_ROOT_PREFIX="$SEARCH_ROOT"
    export SETTING_TAG="$tag"
    export SPLIT_STRATEGY=random_kfold
    export SPLIT_ITEMS="$fold"
    export CV_FOLDS="$CV_FOLDS"
    export STATE_KEY="$STATE_KEY"
    export RANDOM_STRATIFY_COLS="$RANDOM_STRATIFY_COLS"
    export SUMMARY_RANKING_MODE="$SUMMARY_RANKING_MODE"
    export USE_STATE_CENTROIDS="$use_state_centroids"
    export USE_GROWTH_INTERCEPT="$use_growth_intercept"
    export THREADS_PER_GPU="$THREADS_PER_SETTING"
    export EMBEDDING_DIM="$embedding"
    export MEDIATOR_DIM="$mediator"
    export N_PROGRAMS="$programs"
    export HIDDEN_DIM="$hidden"
    export DEPTH="$depth"
    export EPOCHS="$setting_epochs"
    export N_PARTICLES="$particles"
    export N_STEPS="$SEARCH_N_STEPS"
    export EVAL_PARTICLES="$eval_particles"
    export EVAL_STEPS="$SEARCH_EVAL_STEPS"
    export EVAL_TARGET_PARTICLES="$eval_target"
    export MAX_TRAIN_TARGET_ATOMS="$max_atoms"
    export MAX_ACTIVE_PERTURBATIONS="$max_active"
    export LAMBDA_CONTROL_REF="$lambda_ctrl"
    export LAMBDA_WEAK="$lambda_weak"
    export LAMBDA_REG_GROWTH_BIAS="$lambda_reg_growth_bias"
    export STAGE_C_EPOCHS="$stage_c"
    export STAGE_D_EPOCHS="$stage_d"
    bash scripts/_run_hnscc_cv.sh "${EXTRA_RUN_ARGS[@]}"
  ) > "$log_path" 2>&1 &
  local pid=$!
  RUNNING_PIDS+=("$pid")
  PID_LABEL["$pid"]="$tag fold_$fold"
  PID_LOG["$pid"]="$log_path"
  PID_GPU["$pid"]="$gpu"
}

stop_scheduling=0
for setting in "${SETTINGS[@]}"; do
  tag="${setting%%|*}"
  for fold in "${SEARCH_FOLD_ITEMS[@]}"; do
    if [[ -f "$SEARCH_ROOT/$tag/fold_$fold/results_summary.json" ]]; then
      echo "Skipping completed job: $tag fold_$fold"
      continue
    fi
    wait_for_free_gpu
    if [[ "$FAILED_JOBS" -ne 0 ]]; then
      stop_scheduling=1
      break
    fi
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    launch_fold_job "$gpu" "$fold" "$setting"
  done
  if [[ "$stop_scheduling" -ne 0 ]]; then
    break
  fi
done
while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
  wait_for_one_job
done
if [[ "$FAILED_JOBS" -ne 0 ]]; then
  echo "CREDO optimal search failed jobs: $FAILED_JOBS" >&2
  exit 1
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python runners/summarize_hnscc_cv.py \
  --cv-root "$SEARCH_ROOT" \
  --output-dir "$SEARCH_ROOT" \
  --group-by setting \
  --ranking-mode "$SUMMARY_RANKING_MODE"

echo "CREDO optimal search summary:"
sed -n '1,100p' "$SEARCH_ROOT/cv_summary.md"
echo "$SEARCH_ROOT"
