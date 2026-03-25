"""Training loops for the benchmark and full CAMFND models."""

from camfnd.training.single_screen_model import (
    SingleScreenTrainConfig,
    SingleScreenTrainingResult,
    Stage1TrainConfig,
    Stage1TrainingResult,
    initialize_single_screen_model_from_moments,
    train_stage1_model,
    train_single_screen_model,
    train_step3_model,  # backward-compatible alias
)
from camfnd.training.multiscreen_context_model import (
    MultiscreenContextTrainConfig,
    MultiscreenContextTrainingResult,
    Stage2TrainConfig,
    Stage2TrainingResult,
    initialize_multiscreen_context_model_from_moments,
    train_multiscreen_context_model,
    train_stage2_model,
    train_step4_model,  # backward-compatible alias
)
from camfnd.training.full_model import (
    FullModelTrainConfig,
    FullModelTrainingResult,
    train_full_model,
)

__all__ = [
    "SingleScreenTrainConfig",
    "SingleScreenTrainingResult",
    "initialize_single_screen_model_from_moments",
    "train_single_screen_model",
    "Stage1TrainConfig",
    "Stage1TrainingResult",
    "train_stage1_model",
    "train_step3_model",
    "MultiscreenContextTrainConfig",
    "MultiscreenContextTrainingResult",
    "initialize_multiscreen_context_model_from_moments",
    "train_multiscreen_context_model",
    "Stage2TrainConfig",
    "Stage2TrainingResult",
    "train_stage2_model",
    "train_step4_model",
    "FullModelTrainConfig",
    "FullModelTrainingResult",
    "train_full_model",
]
