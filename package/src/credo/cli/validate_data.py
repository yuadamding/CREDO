from __future__ import annotations

import argparse
import json
import sys

from credo.data.schema import SCHEMA_OBS_COLUMNS, validate_anndata_schema


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generic CREDO AnnData inputs.")
    parser.add_argument("--data-path", required=True, help="Path to an .h5ad file.")
    parser.add_argument(
        "--schema",
        choices=sorted(SCHEMA_OBS_COLUMNS),
        default="minimal",
        help="Generic CREDO schema profile to validate.",
    )
    parser.add_argument("--latent-key", default="X_pca", help="Required obsm latent key.")
    parser.add_argument(
        "--obs-column",
        action="append",
        default=None,
        help=(
            "Additional required obs column. May be repeated. "
            "Use --schema custom to validate only the columns supplied here."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require the stricter package schema profile for endpoint/trajectory inputs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a short text report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_anndata_schema(
        args.data_path,
        schema=args.schema,
        latent_key=args.latent_key,
        obs_columns=args.obs_column or None,
        strict=args.strict,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ok"]:
        print(
            f"OK: {report['path']} schema={report['schema']} "
            f"shape={report['shape']} latent={report['latent_key']}"
        )
    else:
        print(f"FAILED: {report['path']}", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
