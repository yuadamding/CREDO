#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Full 4-fold comparison for the best fold-0 local-log candidate.
#
# with_guide:
#   Normal guide-confident run with distinct perturbation guide embeddings.
#
# shared_guide:
#   Same guide-confident cell population and same perturbation_id evaluation
#   groups, but every perturbation receives the same trainable guide
#   embedding. This tests whether guide identity is helping beyond the cell
#   population and perturbation-specific target groups.

SETTINGS_FILE="${SETTINGS_FILE:-scripts/settings_hnscc_heavy_f_best_ur01.txt}"
SEARCH_FOLDS="${SEARCH_FOLDS:-0,1,2,3}"
SUMMARY_RANKING_MODE="${SUMMARY_RANKING_MODE:-test_acc}"
GUIDE_CONFIDENT_ONLY="${GUIDE_CONFIDENT_ONLY:-1}"
GPU_MONITOR="${GPU_MONITOR:-1}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-10}"
PARALLEL_ARMS="${PARALLEL_ARMS:-auto}"
REQUIRE_FULL_GPU_QUEUE="${REQUIRE_FULL_GPU_QUEUE:-1}"
NPROC_TOTAL="${NPROC_TOTAL:-$(nproc)}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
COMPARE_ROOT="${COMPARE_ROOT:-runs/hnscc_random_h100_heavy_f_best_ur01_guide_vs_shared_4cv_${RUN_STAMP}}"
WITH_GUIDE_ROOT="${WITH_GUIDE_ROOT:-${COMPARE_ROOT}/with_guide}"
SHARED_GUIDE_ROOT="${SHARED_GUIDE_ROOT:-${COMPARE_ROOT}/shared_guide}"

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

join_by_comma() {
  local joined=""
  local item=""
  for item in "$@"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$item"
  done
  printf '%s\n' "$joined"
}

count_csv_items() {
  local raw="$1"
  local count=0
  local item=""
  IFS=',' read -r -a parsed <<< "$raw"
  for item in "${parsed[@]}"; do
    item="${item// /}"
    [[ -n "$item" ]] && count=$((count + 1))
  done
  printf '%s\n' "$count"
}

count_settings_rows() {
  local file="$1"
  local count=0
  local line=""
  if [[ ! -f "$file" ]]; then
    echo "Settings file not found: $file" >&2
    exit 1
  fi
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    count=$((count + 1))
  done < "$file"
  printf '%s\n' "$count"
}

truthy_parallel_arms() {
  case "${PARALLEL_ARMS,,}" in
    1|true|on|yes)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_arm() {
  local label="$1"
  local root="$2"
  local shared="$3"
  local gpu_list="${4:-}"
  local max_jobs="${5:-}"

  echo "CREDO best ur01 comparison arm: $label"
  echo "  root=$root"
  echo "  settings_file=$SETTINGS_FILE"
  echo "  search_folds=$SEARCH_FOLDS"
  echo "  guide_confident_only=$GUIDE_CONFIDENT_ONLY"
  echo "  shared_guide_embedding=$shared"
  echo "  gpu_list=${gpu_list:-${GPU_LIST:-auto}}"
  echo "  max_parallel_jobs=${max_jobs:-${MAX_PARALLEL_JOBS:-auto}}"
  echo "  threads_per_gpu=${ARM_THREADS_PER_GPU:-${THREADS_PER_GPU:-auto}}"

  SETTINGS_FILE="$SETTINGS_FILE" \
  CV_ROOT="$root" \
  SEARCH_FOLDS="$SEARCH_FOLDS" \
  SUMMARY_RANKING_MODE="$SUMMARY_RANKING_MODE" \
  GUIDE_CONFIDENT_ONLY="$GUIDE_CONFIDENT_ONLY" \
  SHARED_GUIDE_EMBEDDING="$shared" \
  GPU_MONITOR="$GPU_MONITOR" \
  GPU_MONITOR_INTERVAL="$GPU_MONITOR_INTERVAL" \
  REQUIRE_FULL_GPU_QUEUE="$REQUIRE_FULL_GPU_QUEUE" \
  GPU_LIST="$gpu_list" \
  MAX_PARALLEL_JOBS="$max_jobs" \
  NPROC_TOTAL="$NPROC_TOTAL" \
  THREADS_PER_GPU="${ARM_THREADS_PER_GPU:-${THREADS_PER_GPU:-}}" \
  bash scripts/run_hnscc_h100_heavy_f_optimal_search_4cv_2gpu_v2.sh
}

mkdir -p "$COMPARE_ROOT"

mapfile -t GPU_DEVICES < <(detect_gpu_devices)
if [[ "${#GPU_DEVICES[@]}" -eq 0 ]]; then
  echo "No GPU devices could be detected. Set GPU_LIST explicitly." >&2
  exit 1
fi
ACTIVE_GPU_COUNT="${MAX_PARALLEL_JOBS:-${#GPU_DEVICES[@]}}"
if [[ "$ACTIVE_GPU_COUNT" -gt "${#GPU_DEVICES[@]}" ]]; then
  ACTIVE_GPU_COUNT="${#GPU_DEVICES[@]}"
fi
if [[ "$ACTIVE_GPU_COUNT" -lt 1 ]]; then
  ACTIVE_GPU_COUNT=1
fi
ACTIVE_GPU_DEVICES=("${GPU_DEVICES[@]:0:$ACTIVE_GPU_COUNT}")
SETTINGS_COUNT="$(count_settings_rows "$SETTINGS_FILE")"
FOLD_COUNT="$(count_csv_items "$SEARCH_FOLDS")"
JOBS_PER_ARM=$(( SETTINGS_COUNT * FOLD_COUNT ))
RUN_ARMS_CONCURRENT=0

if [[ "${MULTI_GPU_PER_JOB:-0}" == "1" ]]; then
  if truthy_parallel_arms; then
    echo "CREDO warning: PARALLEL_ARMS=$PARALLEL_ARMS ignored because MULTI_GPU_PER_JOB=1 consumes all visible GPUs per arm." >&2
  fi
elif truthy_parallel_arms; then
  RUN_ARMS_CONCURRENT=1
elif [[ "${PARALLEL_ARMS,,}" == "auto" ]] && [[ "$ACTIVE_GPU_COUNT" -gt "$JOBS_PER_ARM" ]] && [[ "$ACTIVE_GPU_COUNT" -ge 2 ]]; then
  RUN_ARMS_CONCURRENT=1
fi

echo "CREDO best ur01 comparison plan:"
echo "  compare_root=$COMPARE_ROOT"
echo "  settings=$SETTINGS_COUNT folds=$FOLD_COUNT jobs_per_arm=$JOBS_PER_ARM"
echo "  detected_gpus=${GPU_DEVICES[*]} active_gpus=${ACTIVE_GPU_DEVICES[*]}"
echo "  nproc_total=$NPROC_TOTAL"
echo "  parallel_arms=$PARALLEL_ARMS resolved_concurrent=$RUN_ARMS_CONCURRENT"

if [[ "$RUN_ARMS_CONCURRENT" == "1" ]]; then
  SPLIT_GPU_COUNT="$ACTIVE_GPU_COUNT"
  MAX_USEFUL_GPU_COUNT=$(( JOBS_PER_ARM * 2 ))
  if [[ "$SPLIT_GPU_COUNT" -gt "$MAX_USEFUL_GPU_COUNT" ]]; then
    SPLIT_GPU_COUNT="$MAX_USEFUL_GPU_COUNT"
  fi
  if [[ "$SPLIT_GPU_COUNT" -lt 2 ]]; then
    echo "CREDO warning: not enough useful setting/fold jobs to run both arms concurrently; falling back to sequential arms." >&2
    ALL_GPUS="$(join_by_comma "${ACTIVE_GPU_DEVICES[@]}")"
    ARM_THREADS_PER_GPU="${THREADS_PER_GPU:-$(( NPROC_TOTAL / ACTIVE_GPU_COUNT ))}"
    if [[ "$ARM_THREADS_PER_GPU" -lt 1 ]]; then
      ARM_THREADS_PER_GPU=1
    fi
    run_arm "with_guide" "$WITH_GUIDE_ROOT" 0 "$ALL_GPUS" "$ACTIVE_GPU_COUNT"
    run_arm "shared_guide" "$SHARED_GUIDE_ROOT" 1 "$ALL_GPUS" "$ACTIVE_GPU_COUNT"
  else
    WITH_GPU_COUNT=$(( (SPLIT_GPU_COUNT + 1) / 2 ))
    if [[ "$WITH_GPU_COUNT" -gt "$JOBS_PER_ARM" ]]; then
      WITH_GPU_COUNT="$JOBS_PER_ARM"
    fi
    SHARED_GPU_COUNT=$(( SPLIT_GPU_COUNT - WITH_GPU_COUNT ))
    if [[ "$SHARED_GPU_COUNT" -lt 1 ]]; then
      SHARED_GPU_COUNT=1
      WITH_GPU_COUNT=$(( SPLIT_GPU_COUNT - SHARED_GPU_COUNT ))
    fi
    WITH_GPU_DEVICES=("${ACTIVE_GPU_DEVICES[@]:0:$WITH_GPU_COUNT}")
    SHARED_GPU_DEVICES=("${ACTIVE_GPU_DEVICES[@]:$WITH_GPU_COUNT:$SHARED_GPU_COUNT}")
    WITH_GPU_LIST="$(join_by_comma "${WITH_GPU_DEVICES[@]}")"
    SHARED_GPU_LIST="$(join_by_comma "${SHARED_GPU_DEVICES[@]}")"
    ARM_THREADS_PER_GPU="${THREADS_PER_GPU:-$(( NPROC_TOTAL / SPLIT_GPU_COUNT ))}"
    if [[ "$ARM_THREADS_PER_GPU" -lt 1 ]]; then
      ARM_THREADS_PER_GPU=1
    fi
    echo "  concurrent split: with_guide=$WITH_GPU_LIST shared_guide=$SHARED_GPU_LIST"
    echo "  concurrent threads_per_gpu=$ARM_THREADS_PER_GPU"
    arm_failed=0
    run_arm "with_guide" "$WITH_GUIDE_ROOT" 0 "$WITH_GPU_LIST" "$WITH_GPU_COUNT" &
    with_pid="$!"
    run_arm "shared_guide" "$SHARED_GUIDE_ROOT" 1 "$SHARED_GPU_LIST" "$SHARED_GPU_COUNT" &
    shared_pid="$!"
    wait "$with_pid" || arm_failed=1
    wait "$shared_pid" || arm_failed=1
    if [[ "$arm_failed" -ne 0 ]]; then
      echo "CREDO best ur01 comparison failed in at least one arm." >&2
      exit 1
    fi
  fi
else
  ALL_GPUS="$(join_by_comma "${ACTIVE_GPU_DEVICES[@]}")"
  ARM_THREADS_PER_GPU="${THREADS_PER_GPU:-$(( NPROC_TOTAL / ACTIVE_GPU_COUNT ))}"
  if [[ "$ARM_THREADS_PER_GPU" -lt 1 ]]; then
    ARM_THREADS_PER_GPU=1
  fi
  echo "  sequential threads_per_gpu=$ARM_THREADS_PER_GPU"
  run_arm "with_guide" "$WITH_GUIDE_ROOT" 0 "$ALL_GPUS" "$ACTIVE_GPU_COUNT"
  run_arm "shared_guide" "$SHARED_GUIDE_ROOT" 1 "$ALL_GPUS" "$ACTIVE_GPU_COUNT"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "CREDO best ur01 comparison dry run complete."
  echo "$COMPARE_ROOT"
  exit 0
fi

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
ENV_NAME="${ENV_NAME:-cape-hnscc}"

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python runners/summarize_hnscc_cv.py \
  --cv-root "$COMPARE_ROOT" \
  --output-dir "$COMPARE_ROOT" \
  --group-by setting \
  --ranking-mode "$SUMMARY_RANKING_MODE"

echo "CREDO best ur01 guide-vs-shared combined summary:"
sed -n '1,140p' "$COMPARE_ROOT/cv_summary.md"
echo "$COMPARE_ROOT"
