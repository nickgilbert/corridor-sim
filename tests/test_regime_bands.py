"""§7.2 — Flow-regime bands: fitted k_risk lands every regime median in-band."""

from corridor_sim.calibrate import calibrate
from corridor_sim.config import build_default_config


def test_regime_bands_in_band():
    cfg = build_default_config()
    cfg.calibration.trials = 12_000
    res = calibrate(cfg)

    assert 0.0 < res.k_risk < 3.0
    for regime, in_band in res.regime_in_band.items():
        lo, hi = res.regime_bands_bps[regime]
        med = res.regime_medians_bps[regime]
        assert in_band, f"{regime}: median {med} bps outside band [{lo}, {hi}]"


def test_yield_subsidy_anchor_embedded():
    cfg = build_default_config()
    cfg.calibration.trials = 2_000
    res = calibrate(cfg)
    assert all(c["ok"] for c in res.yield_subsidy_check.values())
