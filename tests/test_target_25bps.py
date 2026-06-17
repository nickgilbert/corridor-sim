"""§7.4 — ≤25 bps wholesale target demonstrated probabilistically."""

import pytest

from corridor_sim.calibrate import calibrate_and_apply
from corridor_sim.config import build_default_config
from corridor_sim.montecarlo import run_cell


@pytest.fixture(scope="module")
def cfg():
    c = build_default_config()
    c.calibration.trials = 8_000
    calibrate_and_apply(c)
    return c


@pytest.mark.parametrize("T", [5.0, 10.0, 20.0])
def test_C_clears_25bps_majority(cfg, T):
    res = run_cell(cfg, "MX_MXN", "persistent", T, 12_000, target_bps=25.0)
    s = res.summary.set_index("model")
    p_c = s.loc["C_xborder", "p_le_target"]
    p_a = s.loc["A", "p_le_target"]
    assert p_c >= 0.5, f"T={T}: P(C<=25)={p_c:.3f}"
    assert p_a <= p_c - 0.2, f"T={T}: A not materially lower (A={p_a:.3f}, C={p_c:.3f})"
