"""Pair means need a model-level interval, not a pair-level one.

A zoo of m models produces m(m-1)/2 pairs and each model appears in m-1 of them,
so resampling pairs treats one model's idiosyncrasy as m-1 independent
observations and reports an interval that is too narrow. These tests pin the
property that motivates the function: with one odd model in the zoo, the model
bootstrap must be visibly wider than the pair bootstrap.
"""
from __future__ import annotations

import math

import pytest

from manylatents.stats.intervals import (
    Interval,
    PairInterval,
    interval_payload,
    pairwise_cluster_bootstrap,
)


def zoo(names, value_of):
    return {(a, b): value_of(a, b)
            for i, a in enumerate(names) for b in names[i + 1:]}


def test_mean_is_the_plain_pair_mean() -> None:
    v = zoo(["a", "b", "c", "d"], lambda a, b: 1.0 if a < b else 2.0)
    pi = pairwise_cluster_bootstrap(v, n_boot=200)
    assert pi.mean == pytest.approx(sum(v.values()) / len(v))
    assert pi.n_pairs == 6
    assert pi.n_models == 4


def test_key_order_does_not_matter() -> None:
    a = pairwise_cluster_bootstrap({("x", "y"): 1.0, ("x", "z"): 3.0,
                                    ("y", "z"): 2.0}, n_boot=200, seed=1)
    b = pairwise_cluster_bootstrap({("y", "x"): 1.0, ("z", "x"): 3.0,
                                    ("z", "y"): 2.0}, n_boot=200, seed=1)
    assert a.mean == pytest.approx(b.mean)
    assert a.model_ci.lo == pytest.approx(b.model_ci.lo)


def test_model_bootstrap_is_wider_when_one_model_is_odd() -> None:
    """The property the function exists for.

    'odd' scores 1.0 against everyone; the other 11 score 0.0 among themselves.
    Resampling pairs sees 11 high values out of 66 and barely moves. Resampling
    models sometimes drops 'odd' entirely and sometimes draws it repeatedly, so
    the mean genuinely swings.
    """
    names = ["odd"] + [f"m{i}" for i in range(11)]
    v = zoo(names, lambda a, b: 1.0 if "odd" in (a, b) else 0.0)
    pi = pairwise_cluster_bootstrap(v, n_boot=2000, seed=0)

    pair_w = pi.pair_ci.hi - pi.pair_ci.lo
    model_w = pi.model_ci.hi - pi.model_ci.lo
    assert model_w > pair_w * 1.5, f"model {model_w:.4f} vs pair {pair_w:.4f}"


def test_both_intervals_bracket_the_mean() -> None:
    names = [f"m{i}" for i in range(6)]
    v = zoo(names, lambda a, b: float(int(a[1:]) + int(b[1:])))
    pi = pairwise_cluster_bootstrap(v, n_boot=1000, seed=3)
    for iv in (pi.pair_ci, pi.model_ci):
        assert iv.lo <= pi.mean <= iv.hi


def test_deterministic_for_a_seed() -> None:
    names = [f"m{i}" for i in range(5)]
    v = zoo(names, lambda a, b: float(int(a[1:]) * int(b[1:])))
    one = pairwise_cluster_bootstrap(v, n_boot=500, seed=7)
    two = pairwise_cluster_bootstrap(v, n_boot=500, seed=7)
    assert one == two
    assert pairwise_cluster_bootstrap(v, n_boot=500, seed=8) != one


def test_constant_zoo_has_zero_width() -> None:
    names = [f"m{i}" for i in range(5)]
    v = zoo(names, lambda a, b: 0.42)
    pi = pairwise_cluster_bootstrap(v, n_boot=300, seed=0)
    assert pi.model_ci.lo == pytest.approx(0.42)
    assert pi.model_ci.hi == pytest.approx(0.42)


def test_non_finite_values_are_dropped_not_propagated() -> None:
    v = {("a", "b"): 1.0, ("a", "c"): float("nan"), ("b", "c"): 3.0}
    pi = pairwise_cluster_bootstrap(v, n_boot=200, seed=0)
    assert math.isfinite(pi.mean)
    assert pi.n_pairs == 2


def test_single_pair_degrades_to_the_pair_interval() -> None:
    """Two models give one pair; there is no model-level spread to estimate."""
    pi = pairwise_cluster_bootstrap({("a", "b"): 5.0}, n_boot=100, seed=0)
    assert pi.mean == pytest.approx(5.0)
    assert pi.n_models == 2


def test_empty_input_does_not_raise() -> None:
    pi = pairwise_cluster_bootstrap({}, n_boot=50)
    assert pi.n_pairs == 0
    assert math.isnan(pi.mean)


class TestPayload:
    def test_pair_interval_payload_reports_the_model_ci_as_ci95(self) -> None:
        """The honest interval is the headline; the narrow one is labelled."""
        names = [f"m{i}" for i in range(5)]
        v = zoo(names, lambda a, b: float(int(a[1:]) + int(b[1:])))
        pi = pairwise_cluster_bootstrap(v, n_boot=400, seed=0)
        d = interval_payload(pi)
        assert d["ci95"] == [pi.model_ci.lo, pi.model_ci.hi]
        assert d["ci95_pair_bootstrap"] == [pi.pair_ci.lo, pi.pair_ci.hi]
        assert d["resampled"] == "models"
        assert d["n_models"] == 5 and d["n_pairs"] == 10

    def test_plain_interval_payload(self) -> None:
        d = interval_payload(Interval(1.0, 0.5, 1.5, 9))
        assert d == {"mean": 1.0, "ci95": [0.5, 1.5], "n": 9,
                     "resampled": "observations"}
