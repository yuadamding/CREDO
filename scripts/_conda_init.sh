#!/usr/bin/env bash

ensure_conda_available() {
  if [[ -n "${CONDA_BIN:-}" && -x "${CONDA_BIN}" ]]; then
    export PATH="$(dirname "$CONDA_BIN"):$PATH"
  fi

  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  if [[ -n "${MINIFORGE_ROOT:-}" && -f "$MINIFORGE_ROOT/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"
  fi

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    local conda_base=""
    conda_base="$("${CONDA_EXE}" info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$conda_base/etc/profile.d/conda.sh"
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  local candidates=(
    "$HOME/miniforge3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )
  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck source=/dev/null
      source "$candidate"
      break
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    export PATH="$(dirname "$CONDA_EXE"):$PATH"
  fi

  command -v conda >/dev/null 2>&1
}

resolve_conda_executable() {
  if [[ -n "${CONDA_BIN:-}" && -x "${CONDA_BIN}" ]]; then
    printf '%s\n' "${CONDA_BIN}"
    return 0
  fi

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
    return 0
  fi

  local conda_bin=""
  conda_bin="$(type -P conda 2>/dev/null || true)"
  if [[ -n "$conda_bin" && -x "$conda_bin" ]]; then
    printf '%s\n' "$conda_bin"
    return 0
  fi

  return 1
}

resolve_conda_base() {
  local conda_bin="${1:-}"
  if [[ -z "$conda_bin" ]]; then
    conda_bin="$(resolve_conda_executable 2>/dev/null || true)"
  fi
  if [[ -z "$conda_bin" ]]; then
    return 1
  fi
  "$conda_bin" info --base 2>/dev/null || return 1
}

resolve_solver_executable() {
  local conda_bin="${1:-}"
  local conda_base=""
  if [[ -z "$conda_bin" ]]; then
    conda_bin="$(resolve_conda_executable 2>/dev/null || true)"
  fi
  if [[ -z "$conda_bin" ]]; then
    return 1
  fi
  conda_base="$(resolve_conda_base "$conda_bin" 2>/dev/null || true)"
  if [[ -n "$conda_base" ]]; then
    if [[ -x "$conda_base/bin/mamba" ]]; then
      printf '%s\n' "$conda_base/bin/mamba"
      return 0
    fi
    if [[ -x "$conda_base/bin/micromamba" ]]; then
      printf '%s\n' "$conda_base/bin/micromamba"
      return 0
    fi
  fi
  printf '%s\n' "$conda_bin"
}
