"""§7.5 — C_total invariant to where the Model A venue fee is booked."""

import numpy as np

from corridor_sim.config import build_default_config
from corridor_sim.economics import spread_components
from corridor_sim.models import MODELS
from corridor_sim.montecarlo import cell_rng, sample_params


def test_venue_fee_bookkeeping_conserves_c_total():
    cfg = build_default_config()
    rng = cell_rng(cfg.seed, "MX_MXN", "persistent", 5.0)
    p = sample_params(cfg, "MX_MXN", "persistent", 5.0, 5000, rng)

    in_fx = spread_components(MODELS["A"], p, book_venue_in_ramp=False)
    in_ramp = spread_components(MODELS["A"], p, book_venue_in_ramp=True)

    assert np.allclose(in_fx["c_total"], in_ramp["c_total"], atol=1e-9)
    # but the split between S_ramp and S_fx genuinely differs
    assert not np.allclose(in_fx["s_fx"], in_ramp["s_fx"], atol=1e-6)
    assert np.allclose(in_fx["s_ramp"] + in_fx["s_fx"], in_ramp["s_ramp"] + in_ramp["s_fx"])
