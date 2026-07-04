from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from credo.data.hnscc import load_hnscc_expression


pytestmark = pytest.mark.unit


def _write_h5ad(path: Path, matrix: sp.spmatrix, var: pd.DataFrame | None = None) -> None:
    obs = pd.DataFrame({"cell_id": [f"cell_{idx}" for idx in range(matrix.shape[0])]})
    if var is None:
        var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(matrix.shape[1])])
    ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(path)


def test_parallel_expression_loading_matches_serial(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    matrix = sp.csr_matrix(rng.poisson(1.5, size=(48, 32)).astype(np.float32))
    var = pd.DataFrame(
        {"hv_gene": [idx % 2 == 0 for idx in range(matrix.shape[1])]},
        index=[f"gene_{idx}" for idx in range(matrix.shape[1])],
    )
    path = tmp_path / "synthetic.h5ad"
    _write_h5ad(path, matrix, var)

    rows = np.arange(3, 41, 2)
    _, expr_serial, genes_serial, meta_serial = load_hnscc_expression(
        str(path),
        gene_mask_col="hv_gene",
        row_indices=rows,
        n_workers=0,
        chunk_size=7,
    )
    _, expr_parallel, genes_parallel, meta_parallel = load_hnscc_expression(
        str(path),
        gene_mask_col="hv_gene",
        row_indices=rows,
        n_workers=2,
        chunk_size=7,
    )

    assert genes_parallel == genes_serial
    assert meta_parallel["selected_gene_indices"] == meta_serial["selected_gene_indices"]
    assert np.allclose(meta_parallel["full_library_totals"], meta_serial["full_library_totals"])
    assert (expr_parallel != expr_serial).nnz == 0


def test_empty_gene_mask_is_treated_as_missing_and_refuses_full_scan(tmp_path: Path) -> None:
    path = tmp_path / "wide.h5ad"
    _write_h5ad(path, sp.csr_matrix((4, 6001), dtype=np.float32))

    with pytest.raises(ValueError, match="Refusing to materialize the full transcriptome"):
        load_hnscc_expression(
            str(path),
            gene_mask_col="",
            top_genes=0,
            allow_full_gene_scan=False,
        )


def test_strict_counts_validate_selected_rows_not_head(tmp_path: Path) -> None:
    # First 256 rows are integer counts; rows 256+ are non-integer. Selecting the
    # non-integer rows must fail under strict_counts even though the matrix head is clean
    # (regression: validation previously always sampled source[:256], ignoring row_indices).
    rng = np.random.default_rng(0)
    head = rng.poisson(2.0, size=(256, 6)).astype(np.float32)
    tail = np.full((44, 6), 1.5, dtype=np.float32)
    matrix = sp.csr_matrix(np.vstack([head, tail]))
    path = tmp_path / "mixed_counts.h5ad"
    _write_h5ad(path, matrix)

    with pytest.raises(ValueError, match="near-integer"):
        load_hnscc_expression(
            str(path),
            row_indices=np.arange(256, 300),
            validate_counts=True,
            strict_counts=True,
            allow_full_gene_scan=True,
            n_workers=0,
        )

    # Selecting well-formed integer rows loads without error.
    load_hnscc_expression(
        str(path),
        row_indices=np.arange(0, 44),
        validate_counts=True,
        strict_counts=True,
        allow_full_gene_scan=True,
        n_workers=0,
    )
