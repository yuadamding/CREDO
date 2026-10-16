"""The 24-row registry. No outcome or runtime action occurs during import."""

from dataclasses import asdict, dataclass


DONORS = ("D1", "D2", "D3", "D4")
SEEDS = (0, 1, 2)
ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SPREAD_GRID = (1.0, 1.15, 1.3, 1.5)
STAGE_UPDATES = (1024, 3072, 9216)
FAMILIES = {
    "F0": {"name": "DIRECT-SHARED", "dynamics": "direct", "target_residual": False, "mixture_components": 1},
    "F1": {"name": "DIRECT-SHARED-SHAPE", "dynamics": "direct", "target_residual": False, "mixture_components": 1},
    "F2": {"name": "DIRECT-SHARED-MIXTURE", "dynamics": "direct", "target_residual": False, "mixture_components": 4},
    "F3": {"name": "DIRECT-SHRUNK-TARGET", "dynamics": "direct", "target_residual": True, "mixture_components": 1},
    "F4": {"name": "DDR-SHARED-SHAPE-MOLECULAR", "dynamics": "ddr", "target_residual": False, "mixture_components": 1},
    "F5": {"name": "DDR-SHRUNK-TARGET", "dynamics": "ddr", "target_residual": True, "mixture_components": 1},
}
VARIANTS = {
    "V0": (128, 0.001, 0.0),
    "V1": (128, 0.0001, 0.001),
    "V2": (256, 0.0001, 0.001),
    "V3": (256, 0.0003, 0.01),
}


@dataclass(frozen=True)
class Configuration:
    configuration_id: str
    family: str
    variant: str
    name: str
    dynamics: str
    width: int
    learning_rate: float
    weight_decay: float
    target_residual: bool
    mixture_components: int
    depth: int = 2
    latent_dim: int = 48
    embedding_dim: int = 8
    residual_rank: int = 4
    integration_steps: int = 8
    reaction_bound: float = 3.0
    diffusion_floor: float = 0.001
    optimizer: str = "Adam"
    adam_betas: tuple = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    gradient_clip: float = 10.0


def configurations():
    return tuple(
        Configuration(f"{family}-{variant}", family, variant,
                      **fields, width=values[0], learning_rate=values[1], weight_decay=values[2])
        for family, fields in FAMILIES.items()
        for variant, values in VARIANTS.items()
    )


def registry():
    return {item.configuration_id: asdict(item) for item in configurations()}


def get_configuration(identifier):
    for item in configurations():
        if item.configuration_id == identifier:
            return item
    raise ValueError(f"Unknown registered configuration: {identifier}")
