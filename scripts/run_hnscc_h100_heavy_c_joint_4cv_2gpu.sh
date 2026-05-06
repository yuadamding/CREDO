#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CREDO_PROFILE="${CREDO_PROFILE:-h100_heavy_c}"
export RUN_MODE="${RUN_MODE:-joint}"
export RUN_ROOT_PREFIX="${RUN_ROOT_PREFIX:-runs/hnscc_random_h100_heavy_c_joint_4cv_2gpu}"
export SETTING_TAG="${SETTING_TAG:-heavy_c_finalist1_h896_d5_prog20_p320}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-random_kfold}"
export CV_FOLDS="${CV_FOLDS:-4}"

exec bash scripts/_run_hnscc_cv.sh "$@"
