"""The MDS solver is the one PHATE knob that changes the target itself.

Everything else `project_to_student_dim` exposes changes how a target is built
from the same embedding; `mds_solver` changes the embedding. A student trained
against the 'sgd' target is not aligned to the 'smacof' one, so the default here
is load-bearing: it must stay on PHATE's own default, and it must actually reach
the PHATE object rather than being accepted and dropped.
"""
from __future__ import annotations

import pytest
import torch

from manylatents.algorithms.snapshot_projection import project_to_student_dim
from manylatents.lightning.activation_snapshot import ActivationSnapshot


def _make_snapshot(n: int = 24, teacher_dim: int = 16) -> ActivationSnapshot:
    torch.manual_seed(0)
    return ActivationSnapshot(
        input_ids=torch.zeros(n, 8, dtype=torch.int64),
        attention_mask=torch.ones(n, 8, dtype=torch.int64),
        sample_ids=list(range(n)),
        activations={"layer.0": torch.randn(n, teacher_dim)},
        reduction="mean",
    )


def _module(**kwargs):
    from manylatents.algorithms.latent.phate import PHATEModule

    return PHATEModule(n_components=4, n_pca=None, n_landmark=None, **kwargs)


def test_defaults_are_phate_upstream_defaults() -> None:
    """Not merely 'a' default: these are the values every target built so far
    used, so drifting from them silently re-embeds every downstream artifact."""
    model = _module().model
    assert (model.mds, model.mds_solver) == ("metric", "sgd")


def test_solver_reaches_the_phate_object() -> None:
    assert _module(mds_solver="smacof").model.mds_solver == "smacof"


def test_mds_kind_reaches_the_phate_object() -> None:
    assert _module(mds="classic").model.mds == "classic"


def test_torchdr_backend_refuses_a_solver_it_cannot_honour() -> None:
    """torchdr has no separate MDS stage. Accepting the argument and returning
    an embedding built the other way is the failure mode worth preventing."""
    pytest.importorskip("torchdr")
    with pytest.raises(ValueError, match="mds_solver"):
        _module(backend="torchdr", mds_solver="smacof")


@pytest.mark.parametrize("requested,expected", [(None, "sgd"), ("smacof", "smacof")])
def test_project_to_student_dim_forwards_the_solver(monkeypatch, requested, expected) -> None:
    import manylatents.algorithms.latent.phate as phate_mod

    real = phate_mod.PHATEModule
    seen: dict = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(phate_mod, "PHATEModule", spy)

    extra = {} if requested is None else {"mds_solver": requested}
    project_to_student_dim(
        _make_snapshot(), student_hidden_dim=8, random_state=0, **extra,
    )
    assert seen["mds_solver"] == expected
