"""Shared atomic scientific scope used by results, evidence, and claims."""

from __future__ import annotations

from pydantic import Field, model_validator

from .models import StrictModel

__all__ = ("ScientificScope",)


class ScientificScope(StrictModel):
    """Exact scope that must agree across a result and every supporting claim."""

    subject: str = Field(min_length=1)
    entity_scope: tuple[str, ...]
    sample_scope: tuple[str, ...]
    time_scope: tuple[str, ...]
    population: str = Field(min_length=1)
    comparison: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope(self) -> ScientificScope:
        for name in ("entity_scope", "sample_scope", "time_scope"):
            values = getattr(self, name)
            if not values or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be nonempty, unique, and sorted.")
        return self
