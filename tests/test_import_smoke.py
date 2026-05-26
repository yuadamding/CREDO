from __future__ import annotations


def test_credo_model_stack_imports() -> None:
    import credo
    from credo.config import DataConfig, EvalConfig, RunConfig, TrajectoryTrainingConfig
    from credo.data import EndpointProblem as PublicEndpointProblem
    from credo.data import TrajectoryView
    from credo.data.problems import EndpointProblem, SparseTrajectoryProblem, TrajectoryProblem
    from credo.losses import MultiTimeEndpointLoss
    from credo.losses.trajectory import make_observed_tau_grid
    from credo.models import FullDynamicsModel as PublicFullDynamicsModel
    from credo.models.full_model import FullDynamicsModel
    from credo.models.particles import (
        TrajectoryCounterfactualEngine,
        WeightedParticleSimulator,
        rollout_with_clamped_context,
    )
    from credo.training import Trainer as PublicTrainer
    from credo.training import TrajectoryTrainer
    from credo.training.trainer import Trainer

    assert credo.__version__ == "2.0.11"
    assert FullDynamicsModel.__name__ == "FullDynamicsModel"
    assert Trainer.__name__ == "Trainer"
    assert PublicFullDynamicsModel is FullDynamicsModel
    assert PublicTrainer is Trainer
    assert PublicEndpointProblem is EndpointProblem
    assert SparseTrajectoryProblem.__name__ == "SparseTrajectoryProblem"
    assert MultiTimeEndpointLoss.__name__ == "MultiTimeEndpointLoss"
    assert make_observed_tau_grid.__name__ == "make_observed_tau_grid"
    assert rollout_with_clamped_context.__name__ == "rollout_with_clamped_context"
    assert TrajectoryCounterfactualEngine.__name__ == "TrajectoryCounterfactualEngine"
    assert DataConfig.__name__ == "DataConfig"
    assert EvalConfig.__name__ == "EvalConfig"
    assert RunConfig.__name__ == "RunConfig"
    assert TrajectoryTrainingConfig.__name__ == "TrajectoryTrainingConfig"
    assert TrajectoryView.__name__ == "TrajectoryView"
    assert TrajectoryTrainer.__name__ == "TrajectoryTrainer"
