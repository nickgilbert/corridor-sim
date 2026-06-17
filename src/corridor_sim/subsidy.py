"""Front-loaded, declining subsidy model and burn-down (spec §3.5).

A corridor launches below equilibrium turnover and ramps up an S-curve. SDF subsidises
the gap between the target competitive spread (the lesser of the ≤25 bps wholesale floor
and the incumbent's spread) and the break-even spread each model can offer at the current
turnover. The engine must show Model C's subsidy **retires** (curve crosses zero) while
Model A's does **not**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .economics import spread_components
from .models import MODELS
from .montecarlo import point_params

DAYS_PER_MONTH = 365.0 / 12.0


def turnover_curve(cfg: Config) -> np.ndarray:
    """Logistic turnover ramp T(t) from t_start to t_mature over the horizon (spec §3.5)."""

    s = cfg.subsidy
    months = np.arange(s.horizon_months + 1)
    x = months / s.horizon_months
    logistic = 1.0 / (1.0 + np.exp(-s.steepness * (x - 0.5)))
    # normalise so endpoints hit t_start / t_mature exactly
    lo, hi = logistic[0], logistic[-1]
    logistic = (logistic - lo) / (hi - lo)
    return s.t_start + (s.t_mature - s.t_start) * logistic


def breakeven_bps(cfg: Config, corridor: str, regime: str, T: float, model_key: str) -> float:
    """Break-even all-in spread C_total (bps) for a model at turnover T (point params)."""

    p = point_params(cfg, corridor, regime, T)
    return float(spread_components(MODELS[model_key], p)["c_total"])


@dataclass
class SubsidyResult:
    curve: pd.DataFrame  # month, T, model, breakeven_bps, subsidy_per_day, cumulative
    target_bps: float
    summary: pd.DataFrame  # one row per model: retires?, month_to_retire, total subsidy


def run_subsidy(cfg: Config, corridor: str, models: list[str] | None = None) -> SubsidyResult:
    """Compute the subsidy burn-down for each model over the turnover S-curve."""

    models = models or cfg.grid.models
    regime = cfg.subsidy.regime
    Ts = turnover_curve(cfg)
    months = np.arange(len(Ts))
    K = cfg.params.K.point

    # Target = lesser of the wholesale floor and the incumbent's mature break-even (§3.5).
    incumbent_mature = breakeven_bps(cfg, corridor, regime, Ts[-1], "A")
    target = min(cfg.subsidy.target_bps, incumbent_mature)

    rows = []
    summary_rows = []
    for key in models:
        be = np.array([breakeven_bps(cfg, corridor, regime, float(T), key) for T in Ts])
        V = Ts * K
        # daily subsidy on the gap (bps -> fraction), accumulated per month
        gap = np.maximum(0.0, (be - target) / 1e4)
        subsidy_per_day = gap * V
        cumulative = np.cumsum(subsidy_per_day * DAYS_PER_MONTH)

        for m, T, b, spd, cum in zip(months, Ts, be, subsidy_per_day, cumulative):
            rows.append(
                {
                    "corridor": corridor,
                    "model": key,
                    "month": int(m),
                    "T": float(T),
                    "breakeven_bps": float(b),
                    "subsidy_per_day": float(spd),
                    "cumulative_subsidy": float(cum),
                }
            )

        retired_idx = np.where(be <= target)[0]
        retires = bool(retired_idx.size > 0)
        summary_rows.append(
            {
                "corridor": corridor,
                "model": key,
                "retires": retires,
                "month_to_retire": int(retired_idx[0]) if retires else None,
                "total_subsidy": float(cumulative[-1]),
                "final_breakeven_bps": float(be[-1]),
            }
        )

    return SubsidyResult(
        curve=pd.DataFrame(rows),
        target_bps=float(target),
        summary=pd.DataFrame(summary_rows),
    )
