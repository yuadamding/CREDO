"""Run manifest helpers shared by CREDO trainers and runners."""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def _git_value(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _dependency_versions() -> dict[str, str | None]:
    names = ["anndata", "geomloss", "numpy", "pandas", "pydantic", "scipy", "torch"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    supported_pids: Sequence[str],
    active_pids: Sequence[str] | None = None,
    stage: str | None = None,
    n_epochs: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact reproducibility manifest for a trainer run."""
    try:
        from credo import __version__ as credo_version
    except Exception:  # pragma: no cover
        credo_version = None
    training_cfg = dict(config.get("training", {})) if isinstance(config, Mapping) else {}
    model_cfg = dict(config.get("model", {})) if isinstance(config, Mapping) else {}
    config_payload = json.dumps(config, sort_keys=True, default=str)
    return {
        "manifest_schema_version": 2,
        "package_version": credo_version,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "dependency_versions": _dependency_versions(),
        "git_sha": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--short"])),
        "stage": stage,
        "n_epochs": n_epochs,
        "supported_perturbation_count": len(supported_pids),
        "supported_perturbation_ids": list(supported_pids),
        "active_perturbation_count": len(active_pids) if active_pids is not None else None,
        "active_perturbation_ids": list(active_pids) if active_pids is not None else None,
        "context_kind": model_cfg.get("context_kind"),
        "global_context_batching": training_cfg.get("global_context_batching"),
        "config_sha256": hashlib.sha256(config_payload.encode("utf-8")).hexdigest(),
        "ess_thresholds": {
            "ess_warn_frac": training_cfg.get("ess_warn_frac"),
            "ess_fail_frac": training_cfg.get("ess_fail_frac"),
            "ess_claim_grade_min_frac": training_cfg.get("ess_claim_grade_min_frac"),
            "ess_max_weight_frac_fail": training_cfg.get("ess_max_weight_frac_fail"),
        },
        "checkpoint_schema_version": 1,
        "config": dict(config),
    }


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest as deterministic JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return out


def append_run_manifest_record(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Append one manifest record as JSONL for staged trainer calls."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, default=str))
        handle.write("\n")
    return out


__all__ = ["append_run_manifest_record", "build_run_manifest", "write_run_manifest"]
