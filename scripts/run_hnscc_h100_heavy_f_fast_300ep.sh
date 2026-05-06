#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CREDO_PROFILE="${CREDO_PROFILE:-h100_heavy_f}"
export RUN_MODE="${RUN_MODE:-parallel}"
export RUN_ROOT_PREFIX="${RUN_ROOT_PREFIX:-runs/hnscc_random_h100_heavy_f_fast_300ep_4cv}"
export SETTING_TAG="${SETTING_TAG:-heavy_f_h1024_d6_prog32_p256_fast300}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-random_kfold}"
export CV_FOLDS="${CV_FOLDS:-4}"
export EPOCHS="${EPOCHS:-300}"

exec bash scripts/_run_hnscc_cv.sh "$@"
