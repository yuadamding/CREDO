from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
for path in (str(SOURCE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    pd.set_option("future.infer_string", False)
except (KeyError, ValueError):
    pass

from credo.data import open_study  # noqa: E402
from credo.io import RunConfig, load_config, load_data  # noqa: E402
from credo.registry import get_recipe  # noqa: E402
from credo.runtime import TrainingEngine  # noqa: E402
from examples.synthetic.generate import generate  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config(tmp_path_factory: pytest.TempPathFactory) -> RunConfig:
    root = tmp_path_factory.mktemp("compact_credo")
    data_dir = root / "input"
    generate(data_dir, seed=7)
    config = load_config(ROOT / "examples" / "synthetic" / "config.yaml")
    data_config = config.data.model_copy(
        update={
            "support": data_dir / "support.h5ad",
            "measure_meta": data_dir / "measure_meta.parquet",
            "masses": data_dir / "masses.parquet",
            "counts": data_dir / "counts.parquet",
            "dataset": data_dir / "dataset.json",
        }
    )
    settings = config.recipe_config
    epochs = settings.training.epochs.model_copy(update={"state": 1, "mass": 1, "context": 1})
    training = settings.training.model_copy(
        update={
            "epochs": epochs,
            "particles": 6,
            "measures_per_batch": 12,
            "patience": 2,
        }
    )
    evaluation = settings.evaluation.model_copy(update={"particles": 8})
    settings = settings.model_copy(update={"training": training, "evaluation": evaluation})
    return config.model_copy(
        update={
            "data": data_config,
            "recipe_config": settings,
            "output": root / "run",
        }
    )


@pytest.fixture(scope="session")
def tiny_data(tiny_config: RunConfig):
    return load_data(tiny_config)


@pytest.fixture(scope="session")
def trained_run(tiny_config: RunConfig):
    study = open_study(tiny_config)
    trainer = TrainingEngine().fit(
        get_recipe(tiny_config.recipe), study.view(), tiny_config, device="cpu"
    )
    trainer.save()
    yield trainer
    study.close()
