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

SEARCH_ROOT="${RUN_ROOT_PREFIX:-runs/hnscc_random_h100_heavy_f_vram60_75_search_300ep_$(date +%Y%m%d_%H%M%S)}"
ENV_NAME="${ENV_NAME:-cape-hnscc}"
if [[ -z "${SUMMARY_RANKING_MODE:-}" ]]; then
  state_key_lc="${STATE_KEY:-None}"
  state_key_lc="${state_key_lc,,}"
  if [[ -z "$state_key_lc" || "$state_key_lc" == "none" || "$state_key_lc" == "null" || "$state_key_lc" == "na" ]]; then
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
THREADS_PER_SETTING="${THREADS_PER_GPU:-$(( $(nproc) / ${#GPU_DEVICES[@]} ))}"
if [[ "$THREADS_PER_SETTING" -lt 1 ]]; then
  THREADS_PER_SETTING=1
fi

# tag|embedding|mediator|programs|hidden|depth|particles|eval_particles|eval_target|max_train_atoms
SETTINGS=(
  "heavy_f_fit_h896_d5_prog28_p192_fast300|80|80|28|896|5|192|768|2048|1280"
  "heavy_f_fit_h1024_d6_prog32_p256_fast300|96|96|32|1024|6|256|1024|2560|1536"
  "heavy_f_fit_h1152_d6_prog32_p288_fast300|104|104|32|1152|6|288|1152|2816|1664"
  "heavy_f_fit_h1280_d6_prog36_p320_fast300|112|112|36|1280|6|320|1280|3200|1920"
)

launch_setting() {
  local gpu="$1"
  local setting="$2"
  local tag embedding mediator programs hidden depth particles eval_particles eval_target max_atoms
  IFS='|' read -r tag embedding mediator programs hidden depth particles eval_particles eval_target max_atoms <<< "$setting"
  (
    export ENV_NAME="$ENV_NAME"
    export CREDO_PROFILE=h100_heavy_f
    export GPU_LIST="$gpu"
    export PIN_CPU=0
    export SKIP_SUMMARY=1
    export RUN_MODE=parallel
    export RUN_ROOT_PREFIX="$SEARCH_ROOT"
    export CV_ROOT="$SEARCH_ROOT"
    export SETTING_TAG="$tag"
    export SPLIT_STRATEGY=random
    export SPLIT_ITEMS=random
    export SUMMARY_RANKING_MODE="$SUMMARY_RANKING_MODE"
    export THREADS_PER_GPU="$THREADS_PER_SETTING"
    export EMBEDDING_DIM="$embedding"
    export MEDIATOR_DIM="$mediator"
    export N_PROGRAMS="$programs"
    export HIDDEN_DIM="$hidden"
    export DEPTH="$depth"
    export EPOCHS="${EPOCHS:-300}"
    export N_PARTICLES="$particles"
    export N_STEPS="${N_STEPS:-24}"
    export EVAL_PARTICLES="$eval_particles"
    export EVAL_STEPS="${EVAL_STEPS:-24}"
    export EVAL_TARGET_PARTICLES="$eval_target"
    export MAX_TRAIN_TARGET_ATOMS="$max_atoms"
    bash scripts/_run_hnscc_cv.sh
  ) > "$SEARCH_ROOT/${tag}.launcher.log" 2>&1 &
  echo $!
}

mkdir -p "$SEARCH_ROOT"

setting_idx=0
while [[ "$setting_idx" -lt "${#SETTINGS[@]}" ]]; do
  pids=()
  gpu_slot=0
  for ((gpu_slot=0; gpu_slot<${#GPU_DEVICES[@]} && setting_idx<${#SETTINGS[@]}; gpu_slot++, setting_idx++)); do
    pids+=("$(launch_setting "${GPU_DEVICES[$gpu_slot]}" "${SETTINGS[$setting_idx]}")")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python runners/summarize_hnscc_cv.py \
  --cv-root "$SEARCH_ROOT" \
  --output-dir "$SEARCH_ROOT" \
  --group-by setting \
  --ranking-mode "$SUMMARY_RANKING_MODE"

echo "$SEARCH_ROOT"
