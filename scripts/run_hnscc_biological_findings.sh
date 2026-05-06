#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Post-process trained CREDO HNSCC runs into biological interpretation tables.
#
# Typical usage after scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh:
#   COMPARE_ROOT=runs/.../hnscc_random_h100_heavy_f_best_ur01_guide_vs_shared_4cv_* \
#   bash scripts/run_hnscc_biological_findings.sh
#
# Optional human projection:
#   BULK_EXPR=gse227919_expression.csv BULK_META=gse227919_metadata.csv \
#   bash scripts/run_hnscc_biological_findings.sh

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
DATA_PATH="${DATA_PATH:-../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad}"
COMPARE_ROOT="${COMPARE_ROOT:-}"
WITH_GUIDE_ROOT="${WITH_GUIDE_ROOT:-${COMPARE_ROOT:+${COMPARE_ROOT}/with_guide}}"
SHARED_GUIDE_ROOT="${SHARED_GUIDE_ROOT:-${COMPARE_ROOT:+${COMPARE_ROOT}/shared_guide}}"
OUTPUT_DIR="${OUTPUT_DIR:-${COMPARE_ROOT:+${COMPARE_ROOT}/biology}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/biology}"
SCORE_SIGNATURES="${SCORE_SIGNATURES:-1}"
SPLIT="${SPLIT:-test}"

mkdir -p "$OUTPUT_DIR"

SIG_ARGS=()
if [[ "$SCORE_SIGNATURES" == "1" ]]; then
  SIG_OUT="${SIG_OUT:-${OUTPUT_DIR}/signatures}"
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python analysis/score_hnscc_signatures.py \
    --data-path "$DATA_PATH" \
    --output-dir "$SIG_OUT"
  SIG_ARGS=(--signature-scores "$SIG_OUT/signature_group_scores.csv")
fi

if [[ -z "$WITH_GUIDE_ROOT" ]]; then
  echo "Set WITH_GUIDE_ROOT or COMPARE_ROOT before running biological effect extraction." >&2
  exit 1
fi

SHARED_ARGS=()
if [[ -n "$SHARED_GUIDE_ROOT" && -d "$SHARED_GUIDE_ROOT" ]]; then
  SHARED_ARGS=(--shared-cv-root "$SHARED_GUIDE_ROOT")
fi

HUMAN_ARGS=()
if [[ -n "${BULK_EXPR:-}" || -n "${BULK_META:-}" ]]; then
  if [[ -z "${BULK_EXPR:-}" || -z "${BULK_META:-}" ]]; then
    echo "Both BULK_EXPR and BULK_META are required for human projection." >&2
    exit 1
  fi
  HUMAN_OUT="${HUMAN_OUT:-${OUTPUT_DIR}/human_projection}"
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python analysis/project_bulk_signatures.py \
    --expression "$BULK_EXPR" \
    --metadata "$BULK_META" \
    --output-dir "$HUMAN_OUT"
  HUMAN_ARGS=(--human-trends "$HUMAN_OUT/bulk_signature_stage_trends.csv")
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python analysis/extract_biology_effects.py \
  --cv-root "$WITH_GUIDE_ROOT" \
  "${SHARED_ARGS[@]}" \
  "${SIG_ARGS[@]}" \
  "${HUMAN_ARGS[@]}" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR"

if [[ -n "${COUNTERFACTUAL_RUN_DIR:-}" ]]; then
  CF_OUT="${CF_OUT:-${OUTPUT_DIR}/counterfactual}"
  CF_CONTEXT_CLAMPED="${CF_CONTEXT_CLAMPED:-1}"
  CF_ARGS=()
  if [[ "$CF_CONTEXT_CLAMPED" == "1" ]]; then
    CF_ARGS=(--context-clamped)
  fi
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python analysis/run_counterfactual_biology.py \
    --run-dir "$COUNTERFACTUAL_RUN_DIR" \
    --data-path "$DATA_PATH" \
    --output-dir "$CF_OUT" \
    --source-split "$SPLIT" \
    --n-particles "${CF_PARTICLES:-512}" \
    --n-steps "${CF_STEPS:-28}" \
    "${CF_ARGS[@]}"
fi

echo "CREDO biological findings outputs:"
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort
