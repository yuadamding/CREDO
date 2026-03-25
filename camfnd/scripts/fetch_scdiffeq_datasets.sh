#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/home/yding1995/opscc_sc/scDiffeq/KleinLabData}"
DATA_DIR="${BASE_DIR}/scdiffeq_data"
INCLUDE_RAW="${INCLUDE_RAW:-0}"

download() {
  local file_id="$1"
  local out_path="$2"
  local label="$3"

  mkdir -p "$(dirname "$out_path")"

  if [[ -s "$out_path" ]]; then
    echo "Skip ${label}: ${out_path}"
    return 0
  fi

  rm -f "$out_path"
  echo "Download ${label}: ${out_path}"
  curl -L --fail --silent --show-error \
    "https://ndownloader.figshare.com/files/${file_id}" \
    -o "$out_path"

  if [[ ! -s "$out_path" ]]; then
    echo "Download failed for ${label}: ${out_path} is empty" >&2
    return 1
  fi
}

download "54151208" "${DATA_DIR}/pancreatic_endocrinogenesis/adata.pancreatic_endocrinogenesis.cytotrace.h5ad" "pancreas adata"
download "54151202" "${DATA_DIR}/pancreatic_endocrinogenesis/pancreatic_endocrinogenesis.scaler.pkl" "pancreas scaler"
download "54151205" "${DATA_DIR}/pancreatic_endocrinogenesis/pancreatic_endocrinogenesis.pca.pkl" "pancreas pca"
download "54151199" "${DATA_DIR}/pancreatic_endocrinogenesis/pancreatic_endocrinogenesis.umap.pkl" "pancreas umap"

download "54154232" "${DATA_DIR}/human_hematopoiesis/human_hematopoiesis.processed.h5ad" "hematopoiesis adata"
download "54154226" "${DATA_DIR}/human_hematopoiesis/human_hematopoiesis.scaler.pkl" "hematopoiesis scaler"
download "54154223" "${DATA_DIR}/human_hematopoiesis/human_hematopoiesis.pca.pkl" "hematopoiesis pca"
download "54154229" "${DATA_DIR}/human_hematopoiesis/human_hematopoiesis.umap.pkl" "hematopoiesis umap"

if [[ "${INCLUDE_RAW}" == "1" ]]; then
  download "54151331" "${DATA_DIR}/pancreatic_endocrinogenesis/_downloaded.pancreas.h5ad" "pancreas raw"
  download "54154235" "${DATA_DIR}/human_hematopoiesis/_hsc_all_combined_all_layers.h5ad" "hematopoiesis raw"
  download "54154238" "${DATA_DIR}/human_hematopoiesis/_dynamo_hematopoiesis_v1.h5ad" "hematopoiesis secondary"
fi

echo "Done. Files organized under ${DATA_DIR}"
