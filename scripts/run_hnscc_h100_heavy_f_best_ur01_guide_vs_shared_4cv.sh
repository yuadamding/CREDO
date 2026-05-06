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

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
COMPARE_ROOT="${COMPARE_ROOT:-runs/hnscc_random_h100_heavy_f_best_ur01_guide_vs_shared_4cv_${RUN_STAMP}}"
WITH_GUIDE_ROOT="${WITH_GUIDE_ROOT:-${COMPARE_ROOT}/with_guide}"
SHARED_GUIDE_ROOT="${SHARED_GUIDE_ROOT:-${COMPARE_ROOT}/shared_guide}"

run_arm() {
  local label="$1"
  local root="$2"
  local shared="$3"

  echo "CREDO best ur01 comparison arm: $label"
  echo "  root=$root"
  echo "  settings_file=$SETTINGS_FILE"
  echo "  search_folds=$SEARCH_FOLDS"
  echo "  guide_confident_only=$GUIDE_CONFIDENT_ONLY"
  echo "  shared_guide_embedding=$shared"

  SETTINGS_FILE="$SETTINGS_FILE" \
  CV_ROOT="$root" \
  SEARCH_FOLDS="$SEARCH_FOLDS" \
  SUMMARY_RANKING_MODE="$SUMMARY_RANKING_MODE" \
  GUIDE_CONFIDENT_ONLY="$GUIDE_CONFIDENT_ONLY" \
  SHARED_GUIDE_EMBEDDING="$shared" \
  GPU_MONITOR="$GPU_MONITOR" \
  bash scripts/run_hnscc_h100_heavy_f_optimal_search_4cv_2gpu_v2.sh
}

mkdir -p "$COMPARE_ROOT"

run_arm "with_guide" "$WITH_GUIDE_ROOT" 0
run_arm "shared_guide" "$SHARED_GUIDE_ROOT" 1

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
