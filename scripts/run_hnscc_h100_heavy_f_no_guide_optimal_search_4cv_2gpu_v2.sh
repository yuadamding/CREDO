#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Compare against the guide-confident-only baseline with the same cell
# population and same perturbation groups, but remove guide identity as a
# perturbation-specific model input. SHARED_GUIDE_EMBEDDING=1 makes every
# perturbation use one shared guide embedding while evaluation remains keyed by
# the original perturbation_id.
export GUIDE_CONFIDENT_ONLY="${GUIDE_CONFIDENT_ONLY:-1}"
export SHARED_GUIDE_EMBEDDING="${SHARED_GUIDE_EMBEDDING:-1}"
export SETTINGS_PRESET="${SETTINGS_PRESET:-gpu_util_refine}"
export SEARCH_FOLDS="${SEARCH_FOLDS:-0}"
export RUN_ROOT_PREFIX="${RUN_ROOT_PREFIX:-runs/hnscc_random_h100_heavy_f_no_guide_optimal_search_v2_4cv_2gpu_$(date +%Y%m%d_%H%M%S)}"

exec bash scripts/run_hnscc_h100_heavy_f_optimal_search_4cv_2gpu_v2.sh "$@"
