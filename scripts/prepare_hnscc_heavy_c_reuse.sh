#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SETTING_TAG="heavy_c_finalist1_h896_d5_prog20_p320"

discover_cv_root() {
  find runs -maxdepth 1 -type d \( -name 'hnscc_random_h100_heavy_c_joint_4cv_2gpu_*' \) | sort | tail -n 1
}

RAW_CV_ROOT="${1:-${CV_ROOT:-}}"
if [[ -z "$RAW_CV_ROOT" ]]; then
  RAW_CV_ROOT="$(discover_cv_root)"
fi

if [[ -z "$RAW_CV_ROOT" ]]; then
  echo "Could not find a heavy_c CV run. Pass CV_ROOT explicitly." >&2
  exit 1
fi

if [[ -d "$RAW_CV_ROOT/$SETTING_TAG" ]]; then
  SOURCE_ROOT="$RAW_CV_ROOT/$SETTING_TAG"
  CV_ROOT="$RAW_CV_ROOT"
elif [[ -d "$RAW_CV_ROOT" ]]; then
  SOURCE_ROOT="$RAW_CV_ROOT"
  CV_ROOT="$(dirname "$RAW_CV_ROOT")"
else
  echo "Source root does not exist: $RAW_CV_ROOT" >&2
  exit 1
fi

DEST_ROOT="${2:-${DEST_ROOT:-models/heavy_c_joint_4cv_latest}}"
COPY_MODE="${COPY_MODE:-copy}"

mkdir -p "$DEST_ROOT"

link_or_copy() {
  local src="$1"
  local dst="$2"
  rm -f "$dst"
  if [[ "$COPY_MODE" == "copy" ]]; then
    cp -f "$src" "$dst"
  else
    local rel_src=""
    rel_src="$(realpath --relative-to="$(dirname "$dst")" "$src" 2>/dev/null || realpath "$src")"
    ln -s "$rel_src" "$dst"
  fi
}

copy_editable() {
  local src="$1"
  local dst="$2"
  rm -f "$dst"
  cp -f "$src" "$dst"
}

manifest_tsv="$DEST_ROOT/fold_manifest.tsv"
printf "fold\tpreferred_model\tconfig\tresults\tsummary\n" > "$manifest_tsv"

for fold_dir in "$SOURCE_ROOT"/fold_*; do
  [[ -d "$fold_dir" ]] || continue
  fold_name="$(basename "$fold_dir")"
  dest_fold="$DEST_ROOT/$fold_name"
  mkdir -p "$dest_fold"

  preferred=""
  if [[ -f "$fold_dir/checkpoint_best_ema.pt" ]]; then
    preferred="$fold_dir/checkpoint_best_ema.pt"
  elif [[ -f "$fold_dir/checkpoint_best.pt" ]]; then
    preferred="$fold_dir/checkpoint_best.pt"
  else
    echo "Missing reusable checkpoint in $fold_dir" >&2
    exit 1
  fi

  link_or_copy "$preferred" "$dest_fold/model.pt"

  for extra in \
    checkpoint_best_ema.pt \
    checkpoint_best.pt \
    config.json \
    summary.md \
    split_assignments.csv \
    state_reference.csv \
    supported_perturbations.txt \
    train_study_summary.csv \
    test_study_summary.csv \
    train_endpoint_metrics.csv \
    test_endpoint_metrics.csv \
    train_state_metrics.csv \
    test_state_metrics.csv \
    training_history.csv; do
    if [[ -f "$fold_dir/$extra" ]]; then
      link_or_copy "$fold_dir/$extra" "$dest_fold/$extra"
    fi
  done

  if [[ -f "$fold_dir/results_summary.json" ]]; then
    copy_editable "$fold_dir/results_summary.json" "$dest_fold/results_summary.json"
  fi

  if [[ -f "$dest_fold/results_summary.json" ]]; then
    DEST_FOLD="$dest_fold" python3 - <<'PY'
import json
import os
from pathlib import Path

dest_fold = Path(os.environ["DEST_FOLD"])
results_path = dest_fold / "results_summary.json"
data = json.loads(results_path.read_text())
source_best = data.get("best_checkpoint")
if source_best is not None:
    data["source_best_checkpoint"] = source_best
data["best_checkpoint"] = "model.pt"
source_output_dir = data.get("output_dir")
if source_output_dir is not None:
    data["source_output_dir"] = source_output_dir
data["output_dir"] = "."
results_path.write_text(json.dumps(data, indent=2) + "\n")
PY
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$fold_name" \
    "$fold_name/model.pt" \
    "$fold_name/config.json" \
    "$fold_name/results_summary.json" \
    "$fold_name/summary.md" >> "$manifest_tsv"
done

for top in cv_summary.md cv_summary.csv cv_results.csv; do
  if [[ -f "$CV_ROOT/$top" ]]; then
    link_or_copy "$CV_ROOT/$top" "$DEST_ROOT/$top"
  fi
done

cat > "$DEST_ROOT/README_REUSE.md" <<'EOF'
# CREDO heavy_c reuse bundle

This folder is a stable export of the heavy_c CV models for downstream work.

- Each fold lives under: `fold_*/`
- The preferred reusable checkpoint is always: `fold_*/model.pt`
- `model.pt` is copied from `checkpoint_best_ema.pt` when available, otherwise `checkpoint_best.pt`
- `fold_manifest.tsv` uses paths relative to this export folder so the bundle can be moved safely
- `results_summary.json` is normalized so `best_checkpoint` points to `model.pt`
- Use the matching `config.json` from the same fold to reconstruct the model
- `split_assignments.csv` preserves the exact held-out fold membership
- Top-level `cv_summary.md` and `cv_results.csv` preserve the validation summary

Recommended downstream patterns:

1. Single-fold inference:
   - choose one fold, then load `fold_k/model.pt` + `fold_k/config.json`
2. Fold ensemble inference:
   - load all `fold_*/model.pt` checkpoints and average predictions downstream
3. Reproducible analysis:
   - keep this exported folder immutable and version it by date or experiment name
EOF

echo "$DEST_ROOT"
