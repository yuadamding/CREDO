"""CAMFND: Control-Anchored Mean-Field Neural Differential Equations.

CAMFND learns perturbation dynamics from endpoint data using finite measures,
particle simulation, and transport-based training objectives.

Quick start
-----------
Run the complete benchmark pipeline:

    from camfnd.pipeline import run_full_pipeline
    result = run_full_pipeline(output_dir="./outputs")

Or use the semantic benchmark modules directly:

    from camfnd.data.single_screen_benchmark import SingleScreenBenchmarkConfig, generate_single_screen_dataset
    from camfnd.evaluation.data_contract import evaluate_data_contract
    from camfnd.evaluation.simulator_validation import evaluate_simulator_validation
    from camfnd.evaluation.single_screen_model import evaluate_single_screen_model
    from camfnd.evaluation.multiscreen_context_model import evaluate_multiscreen_context_model
"""

from camfnd.pipeline import PipelineResult, run_full_pipeline

__all__ = ["run_full_pipeline", "PipelineResult"]
