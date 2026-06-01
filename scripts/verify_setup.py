#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any


REQUIRED_IMPORTS = (
    "credo",
    "torch",
    "numpy",
    "pandas",
)

OPTIONAL_IMPORTS = (
    "anndata",
    "geomloss",
    "scipy",
)


def _module_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exercised by missing local deps
        return {"ok": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    return {"ok": True, "version": version, "error": None}


def _environment_report() -> dict[str, Any]:
    report = {
        "python": {
            "version": platform.python_version(),
            "executable": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "required_imports": {name: _module_status(name) for name in REQUIRED_IMPORTS},
        "optional_imports": {name: _module_status(name) for name in OPTIONAL_IMPORTS},
    }
    torch_status = report["required_imports"].get("torch", {})
    if torch_status.get("ok"):
        import torch

        report["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None
            ),
        }
    return report


def _data_report(data_path: str | None) -> dict[str, Any]:
    if not data_path:
        return {"checked": False, "ok": False, "error": "--check-data requires --data-path"}
    path = Path(data_path)
    if not path.exists():
        return {"checked": True, "ok": False, "path": str(path), "error": f"Missing data file: {path}"}
    try:
        import anndata as ad

        adata = ad.read_h5ad(path, backed="r")
        try:
            return {
                "checked": True,
                "ok": True,
                "path": str(path),
                "shape": list(adata.shape),
                "has_X_pca": "X_pca" in adata.obsm,
                "has_X_umap": "X_umap" in adata.obsm,
                "has_X_pca_latest_sct": "X_pca_latest_sct" in adata.obsm,
                "has_X_umap_latest": "X_umap_latest" in adata.obsm,
            }
        finally:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
    except Exception as exc:
        return {"checked": True, "ok": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _print_human(report: dict[str, Any]) -> None:
    print("python", report["environment"]["python"]["version"])
    print("platform", report["environment"]["python"]["platform"])
    for group_name in ("required_imports", "optional_imports"):
        for name, status in report["environment"][group_name].items():
            value = status["version"] if status["ok"] else status["error"]
            print(f"{name}", value)
    torch_report = report["environment"].get("torch")
    if torch_report:
        print("cuda_available", torch_report["cuda_available"])
        print("cuda_device_count", torch_report["cuda_device_count"])
        print("cuda_device_name", torch_report["cuda_device_name"])
    data = report.get("data", {"checked": False})
    if data.get("checked"):
        print("data_path", data.get("path"))
        print("data_ok", data.get("ok"))
        if data.get("ok"):
            print("shape", data.get("shape"))
            for key in ("has_X_pca", "has_X_umap", "has_X_pca_latest_sct", "has_X_umap_latest"):
                print(key, data.get(key))
        else:
            print("data_error", data.get("error"))
    else:
        print("data_check", "skipped")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = {"environment": _environment_report()}
    report["data"] = _data_report(args.data_path) if args.check_data else {"checked": False}

    required_ok = all(status["ok"] for status in report["environment"]["required_imports"].values())
    data_ok = (not args.check_data) or bool(report["data"].get("ok"))
    report["ok"] = bool(required_ok and data_ok)

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
