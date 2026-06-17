"""§7.3 — Structural ordering, B−C ≈ ΔS, returns to scale, parity ceiling."""

import numpy as np
import pytest

from corridor_sim.calibrate import calibrate_and_apply
from corridor_sim.config import build_default_config
from corridor_sim.economics import blended_yield, spread_components, yield_subsidy_bps
from corridor_sim.models import MODELS
from corridor_sim.montecarlo import point_params, run_cell

EPS = 1e-6


@pytest.fixture(scope="module")
def cfg():
    c = build_default_config()
    c.calibration.trials = 6_000
    calibrate_and_apply(c)
    return c


def _pivot(cfg, corridor, regime, T, n=4000):
    res = run_cell(cfg, corridor, regime, T, n)
    return res.trials.pivot_table(index="trial", columns="model", values="c_total"), res


def test_ordering_A_ge_B_ge_C(cfg):
    piv, _ = _pivot(cfg, "MX_MXN", "persistent", 5.0)
    assert (piv["A"] >= piv["B"] - EPS).all()
    assert (piv["B"] >= piv["C_xborder"] - EPS).all()
    assert (piv["B"] >= piv["C_same_denom"] - EPS).all()
    # x-border carries more yield than same-denom, so settles lowest
    assert (piv["C_same_denom"] >= piv["C_xborder"] - EPS).all()


def test_B_minus_C_equals_delta_S(cfg):
    res = run_cell(cfg, "MX_MXN", "persistent", 5.0, 4000)
    t = res.trials
    b = t[t.model == "B"].set_index("trial")["c_total"]
    cx = t[t.model == "C_xborder"].set_index("trial")
    gap = b - cx["c_total"]
    delta_s = yield_subsidy_bps(cx["y_blend"].values, 5.0)
    assert np.allclose(gap.values, delta_s, atol=1e-4)


def test_returns_to_scale_A_floor(cfg):
    # As T -> infinity, A retains its venue fee term; B/C collapse toward ~0.
    big = 5000.0
    pers_beta_mode = cfg.regime("persistent").beta.args["mode"]
    pA1 = point_params(cfg, "MX_MXN", "persistent", big)
    pA2 = point_params(cfg, "MX_MXN", "persistent", 2 * big)
    a1 = spread_components(MODELS["A"], pA1)["s_fx"]
    a2 = spread_components(MODELS["A"], pA2)["s_fx"]
    b1 = spread_components(MODELS["B"], pA1)["s_fx"]

    venue_floor = cfg.params.f_venue.point * pers_beta_mode * 1e4
    # A's spread is ~flat in T at scale (carry term negligible)
    assert abs(a2 - a1) < 0.05
    # ...and still carries at least the venue fee on imbalanced flow
    assert a1 >= venue_floor
    # B has effectively no such floor: A exceeds B by at least the venue term
    assert a1 - b1 >= 0.5 * venue_floor


def test_parity_ceiling_B_never_below_C(cfg):
    res = run_cell(cfg, "MX_MXN", "intermittent", 10.0, 4000)
    t = res.trials
    b = t[t.model == "B"].set_index("trial")["s_fx"]
    cx = t[t.model == "C_xborder"].set_index("trial")["s_fx"]
    assert (b.values >= cx.values - EPS).all()
