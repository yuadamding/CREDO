"""Explicit successor stages; the historical credo-v4 lifecycle is not redirected."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import CountRepresentationSpec
from .latent import compile_latent_cells
from .workflow import calibrate_representation, load_representation, refit_representation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--specification", type=Path, required=True)
    refit = sub.add_parser("refit")
    refit.add_argument("--calibration", type=Path, required=True)
    encode = sub.add_parser("encode")
    encode.add_argument("--fitted", type=Path, required=True)
    encode.add_argument("--role", choices=("fitting", "query"), required=True)
    for command in (calibrate, refit, encode):
        command.add_argument("--input-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "calibrate":
        spec = CountRepresentationSpec.model_validate_json(args.specification.read_text())
        record = calibrate_representation(args.input_root, spec, args.output)
    elif args.stage == "refit":
        record = refit_representation(args.input_root, args.calibration, args.output)
    else:
        _, spec, _ = load_representation(args.fitted)
        access = spec.fitting if args.role == "fitting" else spec.query
        record = compile_latent_cells(
            args.input_root, args.fitted, access, args.output, role=args.role
        )
    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
