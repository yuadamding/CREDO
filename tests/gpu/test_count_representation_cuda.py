"""Connected count-module CUDA qualification; synthetic, never cohort promotion."""

from __future__ import annotations

import json
import resource
import runpy
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from credo_count_sde_v4.canonical import sha256_file
from credo_count_sde_v4.count_representation.contracts import (
    RepresentationRules,
    numerical_settings,
)
from credo_count_sde_v4.count_representation.latent import (
    EmpiricalLatentStore,
    compile_latent_cells,
)
from credo_count_sde_v4.count_representation.network import count_loss, initialize, sparse_tensor
from credo_count_sde_v4.count_representation.stream import halves, iter_counts
from credo_count_sde_v4.count_representation.workflow import (
    calibrate_representation,
    load_representation,
    refit_representation,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


@pytest.mark.parametrize("features", [6, 18129])
def test_cuda_count_lifecycle_and_realistic_width(tmp_path, monkeypatch, features):
    """Same-device repeated refits, not CPU/CUDA byte equivalence or H100-by-request."""
    previous = numerical_settings()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        _exercise(tmp_path, features)
    finally:
        torch.use_deterministic_algorithms(
            previous["deterministic_algorithms"], warn_only=previous["deterministic_warn_only"]
        )
        torch.set_float32_matmul_precision(previous["matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = previous["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]


def _exercise(tmp_path: Path, features: int) -> None:
    build = runpy.run_path(
        str(Path(__file__).parents[1] / "unit/test_fold_count_representation.py")
    )["fixture"]
    rules = RepresentationRules(device="cuda", candidate_epochs=(1,), batch_rows=256)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    root, spec = build(tmp_path / "inputs", rna_features=features, rows_per_shard=256, rules=rules)
    counts, cells = next(iter_counts(root, spec, training=True))
    assert counts.shape == (256, features)
    first, second = halves(counts, cells, rules.seed)
    assert (first + second != counts).nnz == 0
    model = initialize(rules, features)
    assert model.device.type == "cuda"
    assert sparse_tensor(first, model.device, normalize=True).layout == torch.sparse_coo
    optimizer = torch.optim.AdamW(model.parameters(), lr=rules.learning_rate)
    before = model.first.weight.detach().clone()
    for observed, target in ((first, second), (second, first)):
        optimizer.zero_grad(set_to_none=True)
        scores, depths = count_loss(model.log_probabilities(observed), target)
        loss = scores[depths > 0].mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert model.first.weight.grad is not None and bool(
            torch.isfinite(model.first.weight.grad).all()
        )
        assert bool(model.first.weight.grad.abs().sum() > 0)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), rules.gradient_clip, error_if_nonfinite=True
        )
        assert bool(torch.isfinite(norm))
        optimizer.step()
    assert not torch.equal(before, model.first.weight)
    del model, optimizer, before, loss, scores

    calibrated = calibrate_representation(root, spec, tmp_path / "calibration")
    audit = json.loads((tmp_path / "calibration/calibration.json").read_text())
    assert all(
        np.isfinite(v["mean_cell_direction_CE"]) for v in audit["candidates"][0]["models"].values()
    )
    reason = "Explicit bounded CUDA engineering canary; no count or scientific promotion"
    fitted = refit_representation(
        root, tmp_path / "calibration", tmp_path / "fitted", diagnostic_reason=reason
    )
    model, saved_spec, saved = load_representation(tmp_path / "fitted")
    assert saved_spec == spec and saved.identity() == fitted.identity()
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    reference = model.encode(counts).detach().cpu()
    reloaded, _, _ = load_representation(tmp_path / "fitted")
    assert torch.equal(reference, reloaded.encode(counts).detach().cpu())
    del reloaded
    repeated = refit_representation(
        root, tmp_path / "calibration", tmp_path / "repeated", diagnostic_reason=reason
    )
    other, _, _ = load_representation(tmp_path / "repeated")
    candidate = other.encode(counts).detach().cpu()
    difference = float((reference - candidate).abs().max())
    assert torch.allclose(reference, candidate, rtol=1e-6, atol=1e-6)
    tensor_difference = max(
        float((model.state_dict()[k] - other.state_dict()[k]).abs().max())
        for k in model.state_dict()
    )
    assert all(
        torch.allclose(model.state_dict()[k], other.state_dict()[k], rtol=1e-6, atol=1e-6)
        for k in model.state_dict()
    )
    frozen_hash = sha256_file(tmp_path / "fitted/model.safetensors")
    latent = compile_latent_cells(
        root, tmp_path / "fitted", spec.query, tmp_path / "latent", role="query"
    )
    assert frozen_hash == sha256_file(tmp_path / "fitted/model.safetensors")
    store = EmpiricalLatentStore(tmp_path / "latent", representation_sha256=fitted.identity())
    population = list(store.population("query-source", 0, batch_rows=32))
    assert population and np.isfinite(np.concatenate([p[0] for p in population])).all()
    assert np.concatenate([p[1] for p in population]).sum() == pytest.approx(1)
    assert (
        latent.facts["retained_individual_cells"] == 512
        and latent.facts["encoded_weight_updates"] == 0
    )
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert 0 < peak_allocated <= peak_reserved < min(device.total_memory, 8 * 1024**3)
    result = dict(
        schema_version=1,
        status="passed_cuda_engineering_only",
        features=features,
        batch_rows=256,
        synthetic_sparsity="30_percent_beyond_first_six_genes"
        if features > 6
        else "six_gene_signal",
        GPU_model=device.name,
        GPU_total_bytes=device.total_memory,
        CUDA_version=torch.version.cuda,
        torch_version=torch.__version__,
        implementation_sha256=spec.implementation_sha256,
        environment=spec.environment,
        specification_sha256=spec.identity(),
        numerical_settings=spec.numerical_settings,
        process_environment=spec.process_environment,
        calibration_sha256=calibrated.identity(),
        fitted_sha256=fitted.identity(),
        repeated_fitted_sha256=repeated.identity(),
        encoded_sha256=latent.identity(),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        host_process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        elapsed_seconds=time.monotonic() - started,
        repeated_latent_max_abs_difference=difference,
        repeated_tensor_max_abs_difference=tensor_difference,
        same_device_tolerance=dict(rtol=1e-6, atol=1e-6),
        representation_qualified=False,
        cohort_training=False,
    )
    (tmp_path / "CUDA_QUALIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COUNT_REPRESENTATION_CUDA=" + json.dumps(result, sort_keys=True))
