"""Family-aware layer resolution and validation for distillation studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FamilyLayerSpec:
    family_name: str
    architecture: str
    alignment_side: str
    second_layer: str
    penultimate_layer: str


_FAMILY_SPECS: dict[str, FamilyLayerSpec] = {
    "pythia": FamilyLayerSpec(
        family_name="pythia",
        architecture="decoder_only",
        alignment_side="decoder",
        second_layer="transformer.h[1]",
        penultimate_layer="transformer.h[-2]",
    ),
    "qwen": FamilyLayerSpec(
        family_name="qwen",
        architecture="decoder_only",
        alignment_side="decoder",
        second_layer="model.layers[1]",
        penultimate_layer="model.layers[-2]",
    ),
    "t5": FamilyLayerSpec(
        family_name="t5",
        architecture="encoder_decoder",
        alignment_side="encoder",
        second_layer="encoder.block[1]",
        penultimate_layer="encoder.block[-2]",
    ),
}

_ALIASES = {
    "second",
    "second_layer",
    "penultimate",
    "penultimate_layer",
}


def get_family_layer_spec(family_name: str | None) -> FamilyLayerSpec | None:
    if family_name is None:
        return None
    return _FAMILY_SPECS.get(str(family_name).lower())


def resolve_layer_alias(layer: str, family_name: str | None) -> str:
    spec = get_family_layer_spec(family_name)
    if spec is None:
        return layer

    normalized = str(layer).strip().lower()
    if normalized in {"second", "second_layer"}:
        return spec.second_layer
    if normalized in {"penultimate", "penultimate_layer"}:
        return spec.penultimate_layer
    return layer


def resolve_layer_aliases(layers: Sequence[str] | None, family_name: str | None) -> list[str]:
    return [resolve_layer_alias(layer, family_name) for layer in (layers or [])]


def validate_family_layers(
    *,
    family_name: str | None,
    architecture: str | None,
    alignment_side: str | None,
    teacher_layers: Sequence[str],
    student_layers: Sequence[str],
) -> None:
    spec = get_family_layer_spec(family_name)
    if spec is None:
        return

    if architecture is not None and str(architecture) != spec.architecture:
        raise ValueError(
            f"Family '{family_name}' expects architecture '{spec.architecture}', got '{architecture}'"
        )
    if alignment_side is not None and str(alignment_side) != spec.alignment_side:
        raise ValueError(
            f"Family '{family_name}' expects alignment_side '{spec.alignment_side}', got '{alignment_side}'"
        )

    all_layers = list(teacher_layers) + list(student_layers)
    unresolved = [layer for layer in all_layers if str(layer).strip().lower() in _ALIASES]
    if unresolved:
        raise ValueError(
            f"Unresolved layer aliases for family '{family_name}': {unresolved}"
        )

    if spec.alignment_side == "encoder":
        invalid = [layer for layer in all_layers if not str(layer).startswith("encoder.")]
        if invalid:
            raise ValueError(
                f"Family '{family_name}' is configured for encoder-side alignment, "
                f"but found non-encoder layer paths: {invalid}"
            )

