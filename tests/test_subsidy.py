"""§3.5 — Subsidy burn-down retires for Model C but not Model A."""

import copy

import numpy as np
import pytest

from corridor_sim.calibrate import calibrate_and_apply
from corridor_sim.config import build_default_config
from corridor_sim.subsidy import run_subsidy, turnover_curve


@pytest.fixture(scope="module")
def cfg():
    c = build_default_config()
    c.calibration.trials = 4_000
    calibrate_and_apply(c)
    return c


def test_turnover_curve_endpoints(cfg):
    Ts = turnover_curve(cfg)
    assert Ts[0] == pytest.approx(cfg.subsidy.t_start, abs=1e-9)
    assert Ts[-1] == pytest.approx(cfg.subsidy.t_mature, abs=1e-9)
    assert np.all(np.diff(Ts) > 0)  # monotonically increasing ramp


def test_subsidy_retires_for_C_not_A(cfg):
    # Retirement is a property of the economics at *mature* turnover (DoD#3). The default
    # ramp is now an early-stage curve (≤1×/day) where nothing retires within the horizon,
    # so assert the retirement property against a maturity-reaching ramp.
    mature = copy.deepcopy(cfg)
    mature.subsidy.t_mature = 20.0
    res = run_subsidy(mature, "MX_MXN")
    summ = res.summary.set_index("model")

    assert not summ.loc["A", "retires"], "Model A subsidy must not retire"
    assert summ.loc["C_xborder", "retires"], "Model C subsidy must retire by maturity"
    # A's break-even stays above the wholesale target; C's settles well below it
    assert summ.loc["A", "final_breakeven_bps"] > res.target_bps
    assert summ.loc["C_xborder", "final_breakeven_bps"] < res.target_bps
    # A's cumulative subsidy dwarfs C's
    assert summ.loc["A", "total_subsidy"] > 10 * summ.loc["C_xborder", "total_subsidy"]


def test_early_stage_yield_offsets_subsidy(cfg):
    # On the default early-stage ramp, turnover is too thin for anyone to break even, but
    # the yield edge means the reserved ramp (C) is subsidised strictly less than
    # stable/stable (B), which is subsidised far less than the incumbent (A).
    res = run_subsidy(cfg, "MX_MXN")
    summ = res.summary.set_index("model")

    assert not summ.loc["C_xborder", "retires"]  # early stage: no retirement yet
    sub = summ["total_subsidy"]
    assert sub["A"] > sub["B"] > sub["C_same_denom"] > sub["C_xborder"] > 0
    # the yield edge saves a meaningful share of the stable/stable subsidy
    assert sub["C_xborder"] < sub["B"]


def test_subsidy_curve_shape(cfg):
    res = run_subsidy(cfg, "MX_MXN")
    a = res.curve[res.curve.model == "A"].sort_values("month")
    # break-even falls as turnover rises (front-loaded, declining gap)
    assert a["breakeven_bps"].iloc[0] >= a["breakeven_bps"].iloc[-1]
    # cumulative subsidy is non-decreasing
    assert np.all(np.diff(a["cumulative_subsidy"].values) >= -1e-6)
