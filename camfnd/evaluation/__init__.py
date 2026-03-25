"""Acceptance evaluators for the benchmark and comparison suites."""

from camfnd.evaluation.data_contract import (
    DataContractEvaluation,
    Step1Evaluation,
    evaluate_data_contract,
    evaluate_step1,
)
from camfnd.evaluation.simulator_validation import (
    SimulatorValidationEvaluation,
    Step2Evaluation,
    evaluate_simulator_validation,
    evaluate_step2,
)
from camfnd.evaluation.single_screen_model import (
    SingleScreenModelEvaluation,
    Step3Evaluation,
    evaluate_single_screen_model,
    evaluate_step3,
)
from camfnd.evaluation.multiscreen_context_model import (
    MultiscreenContextModelEvaluation,
    Step4Evaluation,
    evaluate_multiscreen_context_model,
    evaluate_step4,
)
from camfnd.evaluation.full_model import FullModelEvaluation, evaluate_full_model
from camfnd.evaluation.full_joint_sim_cases import FullJointSimCaseSuite, evaluate_full_joint_sim_cases
from camfnd.evaluation.scdiffeq_larry import ScDiffEqLarryEvaluation, evaluate_camfnd_on_scdiffeq_larry
from camfnd.evaluation.scdiffeq_larry_4to6_compare import (
    Larry4to6CVComparison,
    Larry4to6MethodComparison,
    ScDiffEqTuningResult,
    build_larry_4to6_celltype_benchmark,
    evaluate_camfnd_vs_scdiffeq_larry_4to6,
    evaluate_camfnd_vs_scdiffeq_larry_4to6_cv,
    tune_scdiffeq_on_larry_4to6,
)
from camfnd.evaluation.scdiffeq_additional_datasets import (
    ScDiffEqAdditionalDatasetSuite,
    ScDiffEqDatasetMethodComparison,
    build_human_hematopoiesis_celltype_benchmark,
    build_pancreas_overall_benchmark,
    evaluate_camfnd_vs_scdiffeq_additional_datasets,
    evaluate_camfnd_vs_scdiffeq_human_hematopoiesis,
    evaluate_camfnd_vs_scdiffeq_pancreas,
)

__all__ = [
    "DataContractEvaluation",
    "Step1Evaluation",
    "evaluate_data_contract",
    "evaluate_step1",
    "SimulatorValidationEvaluation",
    "Step2Evaluation",
    "evaluate_simulator_validation",
    "evaluate_step2",
    "SingleScreenModelEvaluation",
    "Step3Evaluation",
    "evaluate_single_screen_model",
    "evaluate_step3",
    "MultiscreenContextModelEvaluation",
    "Step4Evaluation",
    "evaluate_multiscreen_context_model",
    "evaluate_step4",
    "FullModelEvaluation",
    "evaluate_full_model",
    "FullJointSimCaseSuite",
    "evaluate_full_joint_sim_cases",
    "ScDiffEqLarryEvaluation",
    "evaluate_camfnd_on_scdiffeq_larry",
    "Larry4to6CVComparison",
    "Larry4to6MethodComparison",
    "ScDiffEqTuningResult",
    "build_larry_4to6_celltype_benchmark",
    "evaluate_camfnd_vs_scdiffeq_larry_4to6",
    "evaluate_camfnd_vs_scdiffeq_larry_4to6_cv",
    "tune_scdiffeq_on_larry_4to6",
    "ScDiffEqDatasetMethodComparison",
    "ScDiffEqAdditionalDatasetSuite",
    "build_human_hematopoiesis_celltype_benchmark",
    "build_pancreas_overall_benchmark",
    "evaluate_camfnd_vs_scdiffeq_human_hematopoiesis",
    "evaluate_camfnd_vs_scdiffeq_pancreas",
    "evaluate_camfnd_vs_scdiffeq_additional_datasets",
]
