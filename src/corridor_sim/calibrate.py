"""Calibrate the single free coefficient ``k_risk`` to the regime bands (spec §7.2).

The competitive on-chain FX spread (Model B: ``m≈0``, ``y_blend=0``, ``f_redeem`` only)
is **linear in ``k_risk``** — ``s_fx = base + k·coeff`` per trial — so each regime's
median is monotonic in ``k``. We fit one scalar so the three regime medians, each
evaluated at that regime's representative turnover, land inside the doc's illustrative
bands. If the analytic form drifts, the coefficient is refit; the bands never move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from .config import Config
from .economics import spread_components, yield_subsidy_bps
from .models import MODELS
from .montecarlo import cell_rng, sample_params

# §7.1 anchor: ΔS with y_blend = 5% must match this table to within 0.01 bps.
YIELD_SUBSIDY_ANCHORS = {0.5: 5.479, 1.0: 2.740, 5.0: 0.548, 10.0: 0.274, 20.0: 0.137}


@dataclass
class CalibrationResult:
    k_risk: float
    corridor: str
    regime_medians_bps: dict[str, float]
    regime_bands_bps: dict[str, tuple[float, float]]
    regime_in_band: dict[str, bool]
    yield_subsidy_check: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "k_risk": self.k_risk,
            "corridor": self.corridor,
            "regime_medians_bps": self.regime_medians_bps,
            "regime_bands_bps": {k: list(v) for k, v in self.regime_bands_bps.items()},
            "regime_in_band": self.regime_in_band,
            "yield_subsidy_check": self.yield_subsidy_check,
        }


def _base_coeff_for_regime(cfg: Config, corridor: str, regime: str, n: int):
    """Return (base_bps, coeff_bps) arrays so that s_fx(k) = base + k·coeff (Model B)."""

    reg = cfg.regime(regime)
    T = reg.calibration_turnover
    rng = cell_rng(cfg.seed, corridor, regime, T)
    p = sample_params(cfg, corridor, regime, T, n, rng)
    spec = MODELS["B"]

    p.k_risk = 0.0
    base = np.asarray(spread_components(spec, p)["s_fx"], dtype=float)
    p.k_risk = 1.0
    full = np.asarray(spread_components(spec, p)["s_fx"], dtype=float)
    coeff = full - base
    return base, coeff


def calibrate(cfg: Config) -> CalibrationResult:
    """Fit ``k_risk`` to the §7.2 regime bands for the calibration corridor."""

    corridor = cfg.calibration.corridor
    n = cfg.calibration.trials
    regimes = list(cfg.regimes.keys())

    base_coeff = {r: _base_coeff_for_regime(cfg, corridor, r, n) for r in regimes}
    bands = {r: cfg.regime(r).band_bps for r in regimes}

    def median_sfx(regime: str, k: float) -> float:
        base, coeff = base_coeff[regime]
        return float(np.median(base + k * coeff))

    def objective(k: float) -> float:
        # Minimax of each median's distance from its band centre, normalised by the
        # band half-width. This is convex in k (max of V-shaped terms), so the bounded
        # solver reliably lands every regime inside its band when a feasible k exists —
        # unlike a relative-error sum, which is dominated by the tight low-bps band.
        worst = 0.0
        for r in regimes:
            lo, hi = bands[r]
            centre = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            z = abs(median_sfx(r, k) - centre) / half
            worst = max(worst, z)
        return worst

    lo_k, hi_k = cfg.calibration.k_bounds
    res = minimize_scalar(objective, bounds=(lo_k, hi_k), method="bounded")
    k = float(res.x)

    medians = {r: median_sfx(r, k) for r in regimes}
    in_band = {r: bands[r][0] <= medians[r] <= bands[r][1] for r in regimes}

    # §7.1 yield-subsidy anchor check (exact table, y_blend = 5%)
    ys_check = {}
    for T, expected in YIELD_SUBSIDY_ANCHORS.items():
        got = float(yield_subsidy_bps(0.05, T))
        ys_check[str(T)] = {
            "expected": expected,
            "got": round(got, 4),
            "ok": abs(got - expected) <= 0.01,
        }

    return CalibrationResult(
        k_risk=k,
        corridor=corridor,
        regime_medians_bps={r: round(v, 4) for r, v in medians.items()},
        regime_bands_bps=bands,
        regime_in_band=in_band,
        yield_subsidy_check=ys_check,
    )


def calibrate_and_apply(cfg: Config) -> CalibrationResult:
    """Fit ``k_risk`` and write it back into ``cfg.params.k_risk.point`` in place."""

    result = calibrate(cfg)
    cfg.params.k_risk.point = result.k_risk
    cfg.params.k_risk.dist.args = {"value": result.k_risk}
    return result
