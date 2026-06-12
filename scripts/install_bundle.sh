#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(dirname "${BASH_SOURCE[0]}")/.."
cd "$BUNDLE_ROOT"
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
SOLVER_BIN="$(resolve_solver_executable "$CONDA_BIN")" || {
  echo "A conda or mamba solver executable could not be resolved." >&2
  exit 1
}
ENV_NAME="${1:-credo-hnscc}"
ENV_FILE="${ENV_FILE:-env/credo-hnscc-minimal.yml}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
SOLVER_NAME="$(basename "$SOLVER_BIN")"
default_data_path() {
  local candidates=(
    "../inputs/hnscc/GSE235325_P4P60_allgenes_allcells_latest_states.h5ad"
  )
  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "${candidates[0]}"
}
DATA_PATH="${DATA_PATH:-$(default_data_path)}"
VERIFY_CHECK_DATA="${VERIFY_CHECK_DATA:-auto}"
VERIFY_DATA_SCHEMA="${VERIFY_DATA_SCHEMA:-custom}"
VERIFY_LATENT_KEY="${VERIFY_LATENT_KEY:-X_pca}"
VERIFY_STRICT_DATA_SCHEMA="${VERIFY_STRICT_DATA_SCHEMA:-0}"
VERIFY_OBS_COLUMNS="${VERIFY_OBS_COLUMNS:-}"

create_env() {
  local condarc_tmp=""
  condarc_tmp="$(mktemp)"
  cat >"$condarc_tmp" <<'EOF'
channels:
  - conda-forge
default_channels: []
channel_priority: strict
EOF

  if [[ "$SOLVER_NAME" == "mamba" || "$SOLVER_NAME" == "micromamba" ]]; then
    CONDARC="$condarc_tmp" "$SOLVER_BIN" env create \
      --override-channels \
      -c conda-forge \
      -n "$ENV_NAME" \
      -f "$ENV_FILE"
  else
    CONDARC="$condarc_tmp" "$SOLVER_BIN" env create \
      --solver libmamba \
      --no-default-packages \
      -n "$ENV_NAME" \
      -f "$ENV_FILE"
  fi

  rm -f "$condarc_tmp"
}

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  create_env
fi

torch_ok=0
if "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
then
  torch_ok=1
fi

if [[ "$INSTALL_TORCH" == "1" || ( "$INSTALL_TORCH" == "auto" && "$torch_ok" != "1" ) ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install --no-cache-dir --force-reinstall --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install --no-cache-dir -e package
verify_args=(scripts/verify_setup.py)
if [[ "$VERIFY_CHECK_DATA" == "1" || "$VERIFY_CHECK_DATA" == "true" || ( "$VERIFY_CHECK_DATA" == "auto" && -f "$DATA_PATH" ) ]]; then
  verify_args+=(
    --check-data
    --data-path "$DATA_PATH"
    --data-schema "$VERIFY_DATA_SCHEMA"
    --latent-key "$VERIFY_LATENT_KEY"
  )
  if [[ "$VERIFY_STRICT_DATA_SCHEMA" == "1" || "$VERIFY_STRICT_DATA_SCHEMA" == "true" ]]; then
    verify_args+=(--strict-data-schema)
  fi
  if [[ -n "$VERIFY_OBS_COLUMNS" ]]; then
    IFS=',' read -r -a verify_obs_columns <<<"$VERIFY_OBS_COLUMNS"
    for column in "${verify_obs_columns[@]}"; do
      column="${column#"${column%%[![:space:]]*}"}"
      column="${column%"${column##*[![:space:]]}"}"
      [[ -n "$column" ]] || continue
      verify_args+=(--obs-column "$column")
    done
  fi
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "${verify_args[@]}"
