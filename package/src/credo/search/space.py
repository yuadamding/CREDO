"""Typed search space for CREDO setting search.

A :class:`CREDOTrialSpec` is a flat, hashable description of one CREDO training
configuration that an external optimizer (Optuna / Ray Tune / a controller) may
propose. It deliberately separates three classes of field:

* ``SEARCHABLE``    - ordinary hyperparameters safe to tune within one study.
* ``ABLATION_ONLY`` - switches that change the *variant* of the method; vary
  these across separate studies / ablations, not as ordinary hyperparameters.
* ``FROZEN``        - method-contract semantics (control reference, same-start
  counterfactuals, mass-faithful context, exact full-context batching). These
  are logged for provenance but must never be mutated by a search policy; doing
  so would break CREDO's documented Semantic Guarantees.

``spec_to_run_config`` maps a spec onto the real :class:`credo.config.schema`
``RunConfig`` (running its validators), so the search layer treats CREDO as a
pure black-box function of a validated config.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from ..config.schema import (
    EvalConfig,
    ModelConfig,
    RunConfig,
    SimulationConfig,
    TrainingConfig,
)


# Field-class registry -------------------------------------------------------
# Names refer to attributes of CREDOTrialSpec.
SEARCHABLE: frozenset[str] = frozenset(
    {
        "embedding_dim",
        "n_programs",
        "mediator_dim",
        "hidden_dim",
        "depth",
        "epochs",
        "lr_net",
        "lr_embed",
        "lr_transformer",
        "lr_causal_attention",
        "weight_decay",
        "lambda_end",
        "lambda_weak",
        "lambda_count",
        "lambda_aux",
        "lambda_reg_net",
        "lambda_reg_embed",
        "lambda_reg_growth_bias",
        "lambda_reg_diffusion",
        "sinkhorn_epsilon",
        "sinkhorn_tau",
        "n_particles",
        "n_steps",
        "eval_particles",
    }
)
ABLATION_ONLY: frozenset[str] = frozenset(
    {
        "context_kind",
        "ecological_growth",
        "training_schedule",
    }
)
# Frozen method semantics with their locked values.
FROZEN: dict[str, object] = {
    "control_mode": "soft_ref",
    "same_start_counterfactuals": True,
    "mass_faithful_context": True,
    "global_context_batching": "full_context_cache",
}


@dataclass(frozen=True)
class CREDOTrialSpec:
    """One CREDO training configuration proposed by a search policy.

    Note on names: the log-mass penalty weight is ``sinkhorn_tau`` (it multiplies
    ``(log|mu| - log|nu|)^2`` in the endpoint loss), and ``sinkhorn_epsilon`` is
    the Sinkhorn entropic blur - matching ``credo.config.schema.TrainingConfig``.
    There is no single ``lambda_mass``/``lambda_reg``: regularization is four
    independent terms (``lambda_reg_{net,embed,growth_bias,diffusion}``).
    Endpoint rollout uses a single ``n_steps`` for both train and eval; only the
    particle count differs at eval (``eval_particles``).
    """

    # --- dataset / problem identity (provided, not searched) ---
    dataset_kind: str = "endpoint"  # "endpoint" | "trajectory" | "single_time"
    data_id: str = "dataset"
    seed: int = 0
    fold_id: Optional[str] = None

    # --- model capacity (SEARCHABLE) ---
    embedding_dim: int = 8
    n_programs: int = 8
    mediator_dim: int = 8
    hidden_dim: int = 128
    depth: int = 3

    # --- context backend / variant (ABLATION_ONLY) ---
    context_kind: str = "mlp"
    ecological_growth: bool = True
    training_schedule: str = "staged"

    # --- optimization (SEARCHABLE) ---
    epochs: int = 300
    lr_net: float = 3e-4
    lr_embed: float = 1e-3
    lr_transformer: float = 5e-5
    lr_causal_attention: float = 5e-5
    weight_decay: float = 1e-6
    lambda_end: float = 1.0
    lambda_weak: float = 0.1
    lambda_count: float = 0.3
    lambda_aux: float = 0.05
    lambda_reg_net: float = 1e-4
    lambda_reg_embed: float = 1e-4
    lambda_reg_growth_bias: float = 1e-4
    lambda_reg_diffusion: float = 1e-4
    sinkhorn_epsilon: float = 0.1
    sinkhorn_tau: float = 1.0  # log-mass penalty weight (NOT a "lambda_mass")

    # --- rollout fidelity (SEARCHABLE) ---
    n_particles: int = 128
    n_steps: int = 24
    eval_particles: int = 384

    # --- FROZEN method semantics (logged, never searched) ---
    control_mode: str = "soft_ref"
    same_start_counterfactuals: bool = True
    mass_faithful_context: bool = True
    global_context_batching: str = "full_context_cache"


def _known_fields() -> set[str]:
    return {f.name for f in fields(CREDOTrialSpec)}


def assert_frozen_semantics(spec: CREDOTrialSpec) -> None:
    """Raise if any frozen method-contract field deviates from its locked value.

    This is the guardrail that keeps a search/RL policy from exploring CREDO's
    semantic invariants as if they were tunable hyperparameters.
    """
    violations = {
        name: getattr(spec, name)
        for name, locked in FROZEN.items()
        if getattr(spec, name) != locked
    }
    if violations:
        expected = {name: FROZEN[name] for name in violations}
        raise ValueError(
            "CREDOTrialSpec violates frozen method semantics (these are part of "
            f"the method contract and must not be searched): got {violations}, "
            f"expected {expected}."
        )


def spec_to_run_config(
    spec: CREDOTrialSpec,
    *,
    output_dir: str = "outputs",
    device: str = "cpu",
    latent_dim: Optional[int] = None,
) -> RunConfig:
    """Map a :class:`CREDOTrialSpec` onto a validated ``RunConfig``.

    Constructs each sub-config with keyword arguments so the schema's pydantic
    validators run (e.g. transformer head divisibility, ESS-threshold ordering,
    the multi-GPU/``lambda_count`` interaction). ``multi_gpu_devices`` is left
    empty on purpose: a CREDO trial with global ecological context (or
    ``lambda_count > 0``) is single-device by construction, so search must
    parallelize *across* trials, not shard one trial.
    """
    assert_frozen_semantics(spec)

    model = ModelConfig(
        embedding_dim=spec.embedding_dim,
        n_programs=spec.n_programs,
        mediator_dim=spec.mediator_dim,
        hidden_dim=spec.hidden_dim,
        depth=spec.depth,
        ecological_growth=spec.ecological_growth,
        control_mode=spec.control_mode,
        context_kind=spec.context_kind,
    )
    simulation = SimulationConfig(
        n_particles=spec.n_particles,
        n_steps=spec.n_steps,
    )
    eval_cfg = EvalConfig(n_eval_particles=spec.eval_particles)
    training = TrainingConfig(
        lr_net=spec.lr_net,
        lr_embed=spec.lr_embed,
        lr_transformer=spec.lr_transformer,
        lr_causal_attention=spec.lr_causal_attention,
        weight_decay=spec.weight_decay,
        lambda_end=spec.lambda_end,
        lambda_weak=spec.lambda_weak,
        lambda_count=spec.lambda_count,
        lambda_aux=spec.lambda_aux,
        lambda_reg_net=spec.lambda_reg_net,
        lambda_reg_embed=spec.lambda_reg_embed,
        lambda_reg_growth_bias=spec.lambda_reg_growth_bias,
        lambda_reg_diffusion=spec.lambda_reg_diffusion,
        sinkhorn_epsilon=spec.sinkhorn_epsilon,
        sinkhorn_tau=spec.sinkhorn_tau,
        epochs=spec.epochs,
        seed=spec.seed,
        training_schedule=spec.training_schedule,
        global_context_batching=spec.global_context_batching,
    )

    cfg = RunConfig(
        run_id=f"{spec.data_id}:{spec.fold_id or 'all'}:seed{spec.seed}",
        data_id=spec.data_id,
        output_dir=output_dir,
        device=device,
        multi_gpu_devices=[],
        model=model,
        simulation=simulation,
        training=training,
        eval=eval_cfg,
    )
    if latent_dim is not None:
        cfg.latent.dim = int(latent_dim)
    return cfg


__all__ = [
    "ABLATION_ONLY",
    "CREDOTrialSpec",
    "FROZEN",
    "SEARCHABLE",
    "assert_frozen_semantics",
    "spec_to_run_config",
]
