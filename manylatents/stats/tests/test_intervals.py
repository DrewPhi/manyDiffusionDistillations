import math
import numpy as np
from manylatents.stats.intervals import (
    Interval, bootstrap_ci, mean_se, paired_bootstrap_diff, format_interval,
)


def test_bootstrap_ci_mean_and_bounds():
    vals = [0.90, 0.92, 0.94, 0.96, 0.98]
    iv = bootstrap_ci(vals, n_boot=2000, seed=0)
    assert isinstance(iv, Interval)
    assert iv.n == 5
    assert math.isclose(iv.mean, 0.94, abs_tol=1e-9)
    assert iv.lo < iv.mean < iv.hi
    assert 0.90 <= iv.lo and iv.hi <= 0.98


def test_bootstrap_ci_is_deterministic_for_seed():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert bootstrap_ci(vals, n_boot=1000, seed=0) == bootstrap_ci(vals, n_boot=1000, seed=0)


def test_single_value_returns_point_with_n1():
    iv = bootstrap_ci([0.5], seed=0)
    assert iv.n == 1 and iv.mean == 0.5 and iv.lo == 0.5 and iv.hi == 0.5


def test_mean_se():
    m, se = mean_se([2.0, 4.0, 6.0])
    assert math.isclose(m, 4.0)
    assert math.isclose(se, np.std([2.0, 4.0, 6.0], ddof=1) / math.sqrt(3))


def test_paired_bootstrap_diff_sign():
    a = [0.9, 0.95, 0.92]   # student-vs-control (high)
    b = [0.4, 0.45, 0.42]   # student-vs-teacher (low)
    iv = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert iv.mean > 0 and iv.lo > 0   # control closer, CI excludes 0


def test_format_interval_variants():
    assert format_interval(Interval(0.96, 0.94, 0.98, 3)) == "0.96 [0.94, 0.98]"
    assert format_interval(Interval(0.5, 0.5, 0.5, 1)) == "0.50 (n=1)"
    assert "e-07" in format_interval(Interval(4e-7, 3e-7, 5e-7, 3)).replace("E", "e")


def test_empty_input_bootstrap_and_mean_se():
    iv = bootstrap_ci([])
    assert iv.n == 0 and math.isnan(iv.mean) and math.isnan(iv.lo) and math.isnan(iv.hi)
    m, se = mean_se([])
    assert math.isnan(m) and math.isnan(se)


def test_format_interval_zero_n_is_honest():
    s = format_interval(Interval(float("nan"), float("nan"), float("nan"), 0))
    assert "(n=0)" in s and "(n=1)" not in s
