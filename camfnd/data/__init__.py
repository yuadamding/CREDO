"""Data contract and benchmark generators."""

from camfnd.data.contract import (
    CellStateTable,
    EndpointProblem,
    FiniteMeasure,
    Key,
    LatentTransform,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    SimulationTruth,
    TimeAxis,
)
from camfnd.data.single_screen_benchmark import (
    SingleScreenBenchmarkConfig,
    SingleScreenTruthParams,
    Stage1BenchmarkConfig,
    Stage1TruthParams,
    build_single_screen_truth_params,
    build_stage1_truth_params,
    generate_single_screen_dataset,
    generate_stage1_dataset,
    ou_terminal_moments,
)
from camfnd.data.multiscreen_benchmark import (
    MultiscreenBenchmarkConfig,
    MultiscreenTruthParams,
    Stage2BenchmarkConfig,
    Stage2TruthParams,
    build_multiscreen_truth_params,
    build_stage2_truth_params,
    generate_multiscreen_dataset,
    generate_stage2_dataset,
)

__all__ = [
    # contract
    "CellStateTable",
    "EndpointProblem",
    "FiniteMeasure",
    "Key",
    "LatentTransform",
    "MassTable",
    "PerturbSeqDynamicsData",
    "PerturbationCatalog",
    "SimulationTruth",
    "TimeAxis",
    # single-screen benchmark
    "SingleScreenBenchmarkConfig",
    "SingleScreenTruthParams",
    "build_single_screen_truth_params",
    "generate_single_screen_dataset",
    "Stage1BenchmarkConfig",
    "Stage1TruthParams",
    "build_stage1_truth_params",
    "generate_stage1_dataset",
    "ou_terminal_moments",
    # multi-screen benchmark
    "MultiscreenBenchmarkConfig",
    "MultiscreenTruthParams",
    "build_multiscreen_truth_params",
    "generate_multiscreen_dataset",
    "Stage2BenchmarkConfig",
    "Stage2TruthParams",
    "build_stage2_truth_params",
    "generate_stage2_dataset",
]
