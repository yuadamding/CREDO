#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CREDO_PROFILE="${CREDO_PROFILE:-local_heavy_c_vae_9gb}"
export RUN_MODE="${RUN_MODE:-parallel}"
export RUN_ROOT_PREFIX="${RUN_ROOT_PREFIX:-runs/hnscc_local_heavy_c_vae_9gb}"
export SETTING_TAG="${SETTING_TAG:-heavy_c_local_vae_h512_d4_prog16_p64_b12}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-random}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
export SPLIT_ITEMS="${SPLIT_ITEMS:-random}"

exec bash scripts/_run_hnscc_cv.sh "$@"
