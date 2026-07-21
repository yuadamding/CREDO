#!/usr/bin/env python3
"""Create a deterministic, non-claim GSE314342 smoke-test subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import anndata as ad
import pandas as pd


CREDO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CREDO_ROOT.parent
sys.path.insert(0, str(CREDO_ROOT / "package" / "src"))

from credo.data.gse314342 import (  # noqa: E402
    LateTimeResolution,
    build_support_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    input_root = WORKSPACE_ROOT / "inputs" / "GSE314342"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=input_root / "gse314342_credo_support.h5ad",
    )
    parser.add_argument("--input-root", type=Path, default=input_root)
    parser.add_argument("--output-dir", type=Path, default=input_root)
    parser.add_argument("--donor", default="D3")
    parser.add_argument("--target-genes", type=int, default=64)
    parser.add_argument("--controls", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _stable_order(values: set[str], *, seed: int, scope: str) -> list[str]:
    def key(value: str) -> bytes:
        return hashlib.sha256(f"{seed}|{scope}|{value}".encode("utf-8")).digest()

    return sorted(values, key=key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("false").astype(str).str.lower().isin({"1", "true", "yes"})


def _select_guides(
    manifest: pd.DataFrame,
    *,
    donor: str,
    target_genes: int,
    controls: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    required = {
        "sample_id",
        "guide_id",
        "embedding_id",
        "is_control",
        "has_Rest",
        "has_Stim8hr",
        "has_Stim48hr",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Measure manifest is missing columns: {missing}")
    complete = (
        _bool(manifest["has_Rest"])
        & _bool(manifest["has_Stim8hr"])
        & _bool(manifest["has_Stim48hr"])
    )
    donor_rows = manifest.loc[manifest["sample_id"].astype(str).eq(donor) & complete].copy()
    control_mask = _bool(donor_rows["is_control"])
    control_pool = set(donor_rows.loc[control_mask, "guide_id"].astype(str))
    gene_pool = set(donor_rows.loc[~control_mask, "embedding_id"].astype(str))
    if len(control_pool) < controls or len(gene_pool) < target_genes:
        raise ValueError(
            f"Requested {controls} controls and {target_genes} target genes, but "
            f"only {len(control_pool)} and {len(gene_pool)} are complete for {donor}."
        )
    selected_controls = _stable_order(control_pool, seed=seed, scope="control")[:controls]
    selected_genes = _stable_order(gene_pool, seed=seed, scope="target_gene")[:target_genes]
    selected = donor_rows.loc[
        donor_rows["guide_id"].astype(str).isin(selected_controls)
        | donor_rows["embedding_id"].astype(str).isin(selected_genes)
    ].copy()
    return selected, selected_controls, selected_genes


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.target_genes < 1 or args.controls < 1:
        raise ValueError("--target-genes and --controls must be positive.")
    manifest_path = args.input_root / "measure_manifest.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False)
    selected, selected_controls, selected_genes = _select_guides(
        manifest,
        donor=str(args.donor),
        target_genes=int(args.target_genes),
        controls=int(args.controls),
        seed=int(args.seed),
    )
    selected_guides = set(selected["guide_id"].astype(str))
    stem = f"pilot_{args.donor.lower()}"
    outputs = {
        "support": args.output_dir / f"gse314342_credo_{stem}.h5ad",
        "counts": args.output_dir / f"{stem}_guide_counts_and_masses.csv",
        "blocks": args.output_dir / f"{stem}_guide_count_blocks.csv",
        "measures": args.output_dir / f"{stem}_measure_manifest.csv",
        "manifest": args.output_dir / f"{stem}_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Pilot outputs already exist: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.input_path, backed="r")
    try:
        obs = source.obs
        mask = (
            obs["sample_id"].eq(str(args.donor))
            & obs["guide_id"].isin(selected_guides)
        ).to_numpy()
        pilot = source[mask].to_memory()
    finally:
        source.file.close()
    for column in pilot.obs.select_dtypes(["category"]).columns:
        pilot.obs[column] = pilot.obs[column].cat.remove_unused_categories()
    late_time = LateTimeResolution.load(
        CREDO_ROOT / "inputs" / "gse314342" / "late_time_resolution.json"
    )
    pilot.uns.update(
        build_support_metadata(
            latent_key="X_credo",
            latent_dim=int(pilot.obsm["X_credo"].shape[1]),
            support_atoms_cap=64,
            late_time=late_time,
            subset={
                "purpose": "numerical smoke test only",
                "donor": str(args.donor),
                "selection": "stable hash over complete guide annotations; no expression outcomes",
                "seed": int(args.seed),
                "n_target_genes": len(selected_genes),
                "n_controls": len(selected_controls),
                "masses_renormalized": False,
            },
        )
    )
    pilot.write_h5ad(outputs["support"], compression="lzf")

    counts = pd.read_csv(args.input_root / "guide_counts_and_masses.csv")
    counts = counts.loc[
        counts["sample_id"].astype(str).eq(str(args.donor))
        & counts["guide_id"].astype(str).isin(selected_guides)
    ].copy()
    blocks = pd.read_csv(args.input_root / "guide_count_blocks.csv")
    blocks = blocks.loc[
        blocks["context_group_id"].astype(str).eq(str(args.donor))
        & blocks["guide_id"].astype(str).isin(selected_guides)
    ].copy()
    blocks["n_total"] = blocks.groupby("time_label", observed=True)["count"].transform("sum")
    selected.to_csv(outputs["measures"], index=False)
    counts.to_csv(outputs["counts"], index=False)
    blocks.to_csv(outputs["blocks"], index=False)

    payload = {
        "accession": "GSE314342",
        "purpose": "numerical smoke test only",
        "donor": str(args.donor),
        "seed": int(args.seed),
        "selection_uses_expression_outcomes": False,
        "selection_requires_all_times": True,
        "n_target_genes": len(selected_genes),
        "n_targeting_guides": int((~_bool(selected["is_control"])).sum()),
        "n_control_guides": len(selected_controls),
        "n_support_atoms": int(pilot.n_obs),
        "target_genes": selected_genes,
        "control_guides": selected_controls,
        "masses_renormalized": False,
        "files": {
            key: {
                "path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"outputs": {key: str(path) for key, path in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
