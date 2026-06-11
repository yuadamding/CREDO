from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from credo.cli.validate_data import main as validate_data_main
from credo.config.schema import RunConfig
from credo.data import (
    SingleTimeProblem,
    TimeAxis,
    build_single_time_problem_from_anndata,
    validate_anndata_schema,
)
from credo.losses import (
    control_null_effect_loss,
    minimal_effect_action_loss,
    single_time_guide_concordance_loss,
)
from credo.models import FullDynamicsModel, SingleTimeCounterfactualEngine, WeightedParticleSimulator


pytestmark = pytest.mark.unit


def _single_time_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(8)],
            "perturbation_id": ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_b", "gene_b"],
            "is_control": [True, True, False, False, True, True, False, False],
            "sample_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
        },
        index=[f"cell_{i}" for i in range(8)],
    )
    data = ad.AnnData(X=np.ones((8, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.0, 1.0],
            [0.0, 1.1],
        ],
        dtype=np.float32,
    )
    return data


def _model() -> FullDynamicsModel:
    torch.manual_seed(0)
    return FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=1,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        control_mode="soft_ref",
    )


def test_single_time_does_not_relax_longitudinal_time_axis() -> None:
    with pytest.raises(ValueError, match="Need at least two time points"):
        TimeAxis(["snapshot"], [0.0])


def test_single_time_schema_profile_accepts_snapshot_with_sample_id(tmp_path) -> None:
    path = tmp_path / "single_time.h5ad"
    _single_time_adata().write_h5ad(path)

    report = validate_anndata_schema(path, schema="single_time")

    assert report["ok"] is True
    assert report["schema"] == "single_time"
    assert report["n_controls"] == 4
    assert report["n_non_controls"] == 4


def test_single_time_schema_profile_rejects_missing_control_flag(tmp_path) -> None:
    path = tmp_path / "missing_control.h5ad"
    data = _single_time_adata()
    del data.obs["is_control"]
    data.write_h5ad(path)

    report = validate_anndata_schema(path, schema="single_time")

    assert report["ok"] is False
    assert "is_control" in report["obs_columns_missing"]


def test_validate_data_cli_accepts_schema_profile_alias(tmp_path, capsys) -> None:
    path = tmp_path / "single_time.h5ad"
    _single_time_adata().write_h5ad(path)

    exit_code = validate_data_main([
        "--data-path",
        str(path),
        "--schema-profile",
        "single_time",
        "--json",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["schema"] == "single_time"


def test_single_time_problem_builds_nonphysical_effect_endpoint() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    endpoint = problem.to_effect_endpoint_problem()

    assert isinstance(problem, SingleTimeProblem)
    assert endpoint.time_axis.labels == ["control_reference", "observed_snapshot"]
    assert endpoint.metadata["problem_mode"] == "single_time"
    assert endpoint.metadata["effect_axis_is_physical_time"] is False
    assert endpoint.metadata["claim_level"] == "single_time_effect_path"
    assert endpoint.metadata["context_protocol"] == "observed_snapshot"
    assert endpoint.perturbation_ids == ["gene_a", "gene_b"]
    assert endpoint.initial["gene_a"].n_atoms == 2
    assert endpoint.terminal["gene_a"].n_atoms == 2


def test_single_time_problem_uses_batch_matching_without_sample_id() -> None:
    data = _single_time_adata()
    data.obs["batch_id"] = data.obs["sample_id"].replace({"s1": "b1", "s2": "b2"})
    del data.obs["sample_id"]

    problem = build_single_time_problem_from_anndata(data, reference_scope="batch")

    assert {view.batch_id for view in problem.views} == {"b1", "b2"}
    assert {view.reference_scope for view in problem.views} == {"batch"}


def test_single_time_unit_mass_disables_abundance_claims() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), mass_mode="unit_mass")

    assert problem.abundance_claims_allowed is False
    assert problem.to_effect_endpoint_problem().metadata["abundance_claims_allowed"] is False


def test_single_time_config_rejects_abundance_claims_with_unavailable_mass() -> None:
    with pytest.raises(ValueError, match="abundance_claims"):
        RunConfig(
            single_time={
                "enabled": True,
                "mass_mode": "unavailable",
                "abundance_claims": "enabled",
            }
        )


def test_single_time_counterfactual_uses_same_reference_source_and_noise() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    engine = SingleTimeCounterfactualEngine(
        model=_model(),
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=4,
    )

    result = engine.run(problem, ["gene_a"], seed=9, common_noise=True)[0]

    assert result.metadata["counterfactual_type"] == "single_time_effect_path"
    assert result.metadata["same_reference_source"] is True
    assert result.metadata["same_noise"] is True
    assert result.metadata["context_protocol"] == "observed_snapshot"
    assert torch.equal(result.rollout_perturb.z_steps[0], result.rollout_control.z_steps[0])
    assert torch.equal(result.rollout_perturb.logw_steps[0], result.rollout_control.logw_steps[0])
    assert torch.equal(result.rollout_perturb.noise_steps, result.rollout_control.noise_steps)


def test_single_time_regularizer_helpers() -> None:
    effects = torch.tensor([0.1, 2.0, 4.0])
    is_control = torch.tensor([True, False, False])

    assert torch.isclose(control_null_effect_loss(effects, is_control), torch.tensor(0.01))
    assert minimal_effect_action_loss(growth_steps=torch.ones(2, 3, 4)).item() == pytest.approx(1.0)
    assert single_time_guide_concordance_loss(
        torch.tensor([1.0, 3.0, 10.0]),
        ["gene_a", "gene_a", "gene_b"],
    ).item() == pytest.approx(1.0)
