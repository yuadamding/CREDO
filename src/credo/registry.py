"""Lazy immutable recipe registry."""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any

from .contracts import CapabilitySet
from .runtime import CREDORecipe


class RecipeUnavailableError(LookupError):
    pass


_BUILTINS = {
    "credo.compact_sde_v3@3.0": "credo.recipes.compact_v3:recipe",
    "credo.transformer_sde_v2@2.0": "credo.recipes.transformer_v2.recipe:recipe",
}
_ALIASES = {
    "credo.compact_sde_v3": "credo.compact_sde_v3@3.0",
    "credo.transformer_sde_v2": "credo.transformer_sde_v2@2.0",
}
_LOADED: dict[str, CREDORecipe] = {}


def _entry_point_key(name: str) -> str | None:
    """Decode a packaging-safe ``recipe_id__version`` entry-point name."""
    if "__" not in name:
        return None
    recipe_id, version = name.rsplit("__", 1)
    return f"{recipe_id}@{version}" if recipe_id and version else None


def _load_reference(reference: str) -> CREDORecipe:
    module_name, attribute = reference.split(":", 1)
    module = importlib.import_module(module_name)
    value: Any = getattr(module, attribute)
    return value() if isinstance(value, type) else value


def _recipe_key(recipe: CREDORecipe) -> str:
    if not isinstance(recipe, CREDORecipe):
        raise TypeError("Recipe registration requires the complete CREDORecipe protocol.")
    if not isinstance(recipe.capabilities, CapabilitySet):
        raise TypeError("Recipe capabilities must be a CapabilitySet.")
    recipe_id = str(recipe.recipe_id)
    recipe_version = str(recipe.recipe_version)
    if not recipe_id or not recipe_version or "@" in recipe_id:
        raise ValueError("Recipe IDs and versions must be nonempty; the ID cannot contain '@'.")
    return f"{recipe_id}@{recipe_version}"


def register_recipe(recipe: CREDORecipe) -> None:
    key = _recipe_key(recipe)
    existing = _LOADED.get(key)
    if existing is not None and existing is not recipe:
        raise ValueError(f"Recipe {key!r} is immutable and already registered.")
    _LOADED[key] = recipe


def get_recipe(identifier: str) -> CREDORecipe:
    key = _ALIASES.get(str(identifier), str(identifier))
    if key in _LOADED:
        return _LOADED[key]
    reference = _BUILTINS.get(key)
    if reference is not None:
        try:
            recipe = _load_reference(reference)
        except ImportError as exc:
            raise RecipeUnavailableError(
                f"Recipe {key!r} is installed but its optional dependencies are unavailable: {exc}."
            ) from exc
        if _recipe_key(recipe) != key:
            raise ValueError(f"Recipe reference {reference!r} resolved to the wrong ID/version.")
        register_recipe(recipe)
        return recipe
    for entry_point in metadata.entry_points(group="credo.recipes"):
        if _entry_point_key(entry_point.name) != key:
            continue
        recipe = entry_point.load()
        recipe = recipe() if isinstance(recipe, type) else recipe
        if _recipe_key(recipe) != key:
            raise ValueError(
                f"Recipe entry point {entry_point.name!r} resolved to the wrong ID/version."
            )
        register_recipe(recipe)
        return recipe
    available = ", ".join(sorted(_BUILTINS))
    raise RecipeUnavailableError(
        f"Recipe {identifier!r} is unavailable. Installed recipes: {available}."
    )


def available_recipes() -> tuple[str, ...]:
    plugins = tuple(
        key
        for entry in metadata.entry_points(group="credo.recipes")
        if (key := _entry_point_key(entry.name)) is not None
    )
    return tuple(sorted(set(_BUILTINS) | set(plugins)))
