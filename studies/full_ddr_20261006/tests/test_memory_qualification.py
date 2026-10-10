"""Synthetic-only checks of the actual qualification forward/backward path."""
# ruff: noqa: E402
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]
import pytest
from test_native_study import fixture, config
from native_study import build_native_study, compile_development
from memory_qualification import probe, next_batch, qualification_config


def test_actual_probe_runs_ddr_gradients_with_no_optimizer_updates():
    bundle = build_native_study(**fixture())
    problem, _ = compile_development(bundle, config())
    cfg = qualification_config({"hidden_dim": 8, "embedding_dim": 2})
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"particles": 4, "steps_per_interval": 2})})
    ids = tuple(problem.training.measures["Stim48hr"])
    row = probe(problem, cfg, ids, device="cpu")
    assert row["parameter_updates"] == 0
    assert row["groups"] == len(ids)
    assert row["peak_allocated_bytes"] is None
    for head in ("drift_", "sigma_", "growth_"):
        assert any(name.startswith(head) and value > 0 for name, value in row["gradients"].items())


def test_ladder_is_capped_and_stops_near_memory_target():
    assert next_batch([], available=100) == 16
    row = {"groups": 16, "peak_allocated_bytes": 1_000_000_000, "peak_reserved_bytes": 2_000_000_000}
    assert next_batch([row], available=100) == 32
    assert next_batch([dict(row, peak_allocated_bytes=70_000_000_000,
                           peak_reserved_bytes=71_000_000_000)], available=100) is None
    assert next_batch([row], available=16) is None
    heavy = dict(row, peak_reserved_bytes=70_000_000_000)
    assert next_batch([heavy], available=100) is None


def test_nonlinear_ladder_would_not_repeat_the_failed_linear_jump():
    rows = [dict(groups=512, peak_allocated_bytes=13_348_445_184, peak_reserved_bytes=13_402_898_432),
            dict(groups=1024, peak_allocated_bytes=35_929_519_616, peak_reserved_bytes=36_014_391_296)]
    proposal = next_batch(rows, available=4096)
    assert 1024 < proposal < 1500  # Former linear rule proposed about 1919.
    assert rows[-1]["peak_reserved_bytes"] * (proposal / 1024) ** 2 * 1.10 <= 78_000_000_000
    assert next_batch(rows, available=4096, reserved_ceiling_bytes=35_000_000_000) is None


def test_recovery_initial_size_and_invalid_records_are_explicit():
    assert next_batch([], available=4096, initial_groups=1024) == 1024
    with pytest.raises(ValueError, match="Invalid measured"):
        next_batch([dict(groups=16, peak_allocated_bytes=3, peak_reserved_bytes=2)], available=100)
    with pytest.raises(ValueError, match="increase"):
        next_batch([dict(groups=16, peak_allocated_bytes=1, peak_reserved_bytes=2)] * 2, available=100)


@pytest.mark.parametrize("model", [
    {"hidden_dim": 128, "embedding_dim": 8, "trunk_kind": "mlp", "trunk_depth": 2},
    {"hidden_dim": 256, "embedding_dim": 8, "trunk_kind": "residual_mlp", "trunk_depth": 3},
    {"hidden_dim": 256, "embedding_dim": 16, "trunk_kind": "residual_mlp", "trunk_depth": 3},
])
def test_qualification_policy_never_enables_ecology_or_validation_selection(model):
    cfg = qualification_config(model)
    assert cfg.model.context == "none"
    assert cfg.training.epochs.context == 0
    assert cfg.training.checkpoint_selection == "last"
    assert cfg.loss.mass == 0
    assert cfg.loss.mass_increment == "particle_log_mass"
    assert cfg.loss.count_likelihood == "categorical_ce"
