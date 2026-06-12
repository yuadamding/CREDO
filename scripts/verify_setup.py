#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import warnings
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
DATA_SCHEMA_CHOICES = ("custom", "endpoint", "minimal", "single_time", "trajectory")


def _module_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exercised by missing local deps
        return {
            "ok": False,
            "version": None,
            "distribution_version": None,
            "module_version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        distribution_version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module_version = getattr(module, "__version__", None)
    return {
        "ok": True,
        "version": module_version or distribution_version,
        "distribution_version": distribution_version,
        "module_version": module_version,
        "error": None,
    }


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


def _data_report(
    data_path: str | None,
    *,
    schema: str,
    latent_key: str,
    obs_columns: list[str] | None,
    column_map: dict[str, str | None] | None,
    strict: bool,
) -> dict[str, Any]:
    if not data_path:
        return {"checked": False, "ok": False, "error": "--check-data requires --data-path"}
    path = Path(data_path)
    if not path.exists():
        return {"checked": True, "ok": False, "path": str(path), "error": f"Missing data file: {path}"}
    try:
        from credo.data.schema import validate_anndata_schema

        report = validate_anndata_schema(
            path,
            schema=schema,
            latent_key=latent_key,
            obs_columns=obs_columns,
            column_map=column_map,
            strict=strict,
        )
        report["checked"] = True
        return report
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
            print("schema", data.get("schema"))
            print("strict", data.get("strict"))
            print("shape", data.get("shape"))
            print("latent_key", data.get("latent_key"))
            print("latent_shape", data.get("latent_shape"))
            print("obs_columns_required", data.get("obs_columns_required"))
            print("obs_columns_missing", data.get("obs_columns_missing"))
            print("required_columns_non_empty", data.get("required_columns_non_empty"))
        else:
            if data.get("error"):
                print("data_error", data.get("error"))
            for error in data.get("errors", []):
                print("data_error", error)
    else:
        print("data_check", "skipped")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument(
        "--data-schema",
        choices=DATA_SCHEMA_CHOICES,
        default="custom",
        help="AnnData schema profile for --check-data. Defaults to a package-surface smoke check.",
    )
    parser.add_argument("--strict-data-schema", action="store_true")
    parser.add_argument("--latent-key", default="X_pca")
    parser.add_argument("--perturbation-col", default="perturbation_id")
    parser.add_argument("--guide-col", default="guide_id")
    parser.add_argument("--target-gene-col", default="target_gene")
    parser.add_argument("--control-col", default="is_control")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--batch-col", default="batch_id")
    parser.add_argument(
        "--obs-column",
        action="append",
        default=None,
        help="Additional required obs column for --check-data. May be repeated.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = {"environment": _environment_report()}
    report["data"] = (
        _data_report(
            args.data_path,
            schema=args.data_schema,
            latent_key=args.latent_key,
            obs_columns=args.obs_column,
            column_map=(
                {
                    "perturbation": args.perturbation_col,
                    "guide": args.guide_col,
                    "target_gene": args.target_gene_col,
                    "control": args.control_col,
                    "sample": args.sample_col,
                    "batch": args.batch_col,
                }
                if args.data_schema == "single_time"
                else None
            ),
            strict=args.strict_data_schema,
        )
        if args.check_data
        else {"checked": False}
    )

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
