import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encode_sources import EncodingError, authorize_sources, export_source


def fixture():
    guides = pd.DataFrame(dict(guide_index=[0, 1, 2], guide_id=["NTC", "a", "b"],
                               target_id=["NTC", "T", "T"], is_control=[True, False, False]))
    counts = sparse.csr_matrix(np.array([[1, 2], [0, 0], [3, 4]], dtype=np.uint32))
    cells = pd.DataFrame(dict(source_id=["D3_Stim8hr"] * 3, shard=[0] * 3,
                              row_in_shard=[0, 1, 2], cell_id=["c0", "c1", "c2"],
                              guide_index=[1, 0, 1], RNA_UMIs=[3, 0, 7]))
    lineage = {key: "a" * 64 for key in ("input_access_sha256", "representation_sha256",
                                        "model_sha256", "scaler_sha256", "gene_axis_sha256",
                                        "implementation_sha256")}
    return guides, counts, cells, lineage


def run(tmp_path, **override):
    guides, counts, cells, lineage = fixture()
    args = dict(source_id="D3_Stim8hr", phase="development", expected_rows=3, guides=guides,
                latent_dim=2, rna_features=2, blocks=[(counts, cells)],
                encode=lambda m: m.toarray().astype(np.float32), lineage=lineage)
    args.update(override)
    return export_source(tmp_path / "out", **args)


def test_individual_laws_zeros_complete_catalog_and_no_overwrite(tmp_path):
    result = run(tmp_path)
    support = pd.read_parquet(tmp_path / "out/support.parquet")
    assert support.captured_cells.tolist() == [1, 2, 0]
    assert support.geometry_cells.tolist() == [0, 2, 0]
    assert result["optimizer_updates"] == 0
    with h5py.File(tmp_path / "out/latent.h5") as f:
        assert f["population_offsets"][:].tolist() == [0, 0, 2, 2]
        assert f["population_rows"][:].tolist() == [0, 2]
        assert f["state"][:].tolist() == [[1., 2.], [0., 0.], [3., 4.]]
    with pytest.raises(FileExistsError):
        run(tmp_path)


@pytest.mark.parametrize("source", ["D1_Stim48hr", "D1_Stim8hr", "D3_Rest", "unknown"])
def test_role_rejects_before_payload_or_output(tmp_path, source):
    with pytest.raises(EncodingError, match="phase boundary"):
        run(tmp_path, source_id=source)
    assert not (tmp_path / "out").exists()


def test_release_strings_are_not_verified_custody():
    release = {name: "a" * 64 for name in ("selected_architecture_sha256",
               "terminal_checkpoints_sha256", "D1_forecast_seal_sha256")}
    with pytest.raises(EncodingError, match="lifecycle"):
        authorize_sources(("D1_Stim48hr",), "assessment", release)
    authorize_sources(("D1_Stim48hr",), "assessment", {**release, "verified_by_lifecycle": True})
    with pytest.raises(EncodingError, match="release"):
        authorize_sources(("D1_Stim48hr",), "final_refit", {**release, "verified_by_lifecycle": True})


def test_missing_rows_no_complete(tmp_path):
    with pytest.raises(EncodingError, match="omitted"):
        run(tmp_path, expected_rows=4)
    assert not (tmp_path / "out/COMPLETE.json").exists()


@pytest.mark.parametrize("field,value,match", [
    ("RNA_UMIs", [3, 1, 7], "depth"),
    ("guide_index", [1, 0, 3], "catalog"),
    ("cell_id", ["c0", "c1", "c0"], "duplicate"),
    ("row_in_shard", [0, 0, 2], "addresses"),
    ("source_id", ["D3_Stim8hr", "D1_Stim48hr", "D3_Stim8hr"], "Source mismatch"),
])
def test_metadata_corruption(tmp_path, field, value, match):
    _, counts, cells, _ = fixture()
    cells[field] = value
    with pytest.raises(EncodingError, match=match):
        run(tmp_path, blocks=[(counts, cells)])
    assert not (tmp_path / "out/COMPLETE.json").exists()


def test_nonfinite_observed_geometry_rejected(tmp_path):
    with pytest.raises(EncodingError, match="Nonfinite"):
        run(tmp_path, encode=lambda m: np.full((3, 2), np.nan, dtype=np.float32))


def test_wrong_precision_not_silently_downcast(tmp_path):
    with pytest.raises(EncodingError, match="float32"):
        run(tmp_path, encode=lambda m: np.zeros((3, 2), dtype=np.float64))


def test_budget_gate_precedes_write(tmp_path):
    with pytest.raises(EncodingError, match="budget"):
        run(tmp_path, maximum_output_bytes=1)
    assert not (tmp_path / "out").exists()
