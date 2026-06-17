"""Vectorised Monte Carlo trial/run/sweep engine and aggregation (spec §6).

One *trial* = one sampled parameter vector evaluated for all four models under the
**same** draws (common random numbers), so differences are attributable only to
structure. One *run* = a fixed (corridor, T, regime) cell × N trials. The *sweep* is
the full corridor × T × regime × model grid. No Python-level per-trial loop — every
cell is a single vectorised pass.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import distributions as dist
from .config import Config
from .economics import EconParams, mm_roc_annual, spread_components
from .models import MODELS, ModelSpec

# additive spread terms recorded per trial (bps)
TERM_COLUMNS = [
    "operating",
    "margin",
    "carry",
    "risk",
    "rebalance",
    "ramp_yield_credit",
    "delta_s",
    "mint_cost",
    "friction_cost",
    "s_fx",
    "s_ramp",
    "c_total",
]

PCTS = [5, 25, 50, 75, 95]


# --------------------------------------------------------------------------- #
# RNG factory (spec §6.4) — deterministic per-cell seeds, no global state
# --------------------------------------------------------------------------- #
def _stable_idx(s: str) -> int:
    """Process-stable hash of a string → int in [0, 2**31).

    ``hash()`` on str is salted per process (PYTHONHASHSEED), which would make per-cell
    seeds — and therefore every result — irreproducible across separate CLI invocations.
    A fixed digest keeps runs reproducible from the master seed alone (spec §6.4).
    """

    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big") % (2**31)


def cell_rng(master_seed: int, corridor: str, regime: str, T: float) -> np.random.Generator:
    """Derive a deterministic per-cell RNG from the master seed."""

    corr_idx = _stable_idx(corridor)
    reg_idx = _stable_idx(regime)
    t_int = int(round(float(T) * 1000))
    ss = np.random.SeedSequence([int(master_seed), corr_idx, reg_idx, t_int])
    return np.random.default_rng(ss)


# --------------------------------------------------------------------------- #
# Parameter sampling
# --------------------------------------------------------------------------- #
def sample_params(
    cfg: Config,
    corridor: str,
    regime: str,
    T: float,
    n: int,
    rng: np.random.Generator,
) -> EconParams:
    """Sample an ``EconParams`` vector of length ``n`` for one cell (spec §4, §5)."""

    p = cfg.params
    corr = cfg.corridor(corridor)
    reg = cfg.regime(regime)

    sigma_spec = corr.sigma or p.sigma.dist
    y_local_spec = corr.y_local or p.y_local.dist

    return EconParams(
        T=float(T),
        beta=dist.sample(reg.beta, rng, n),
        sigma=dist.sample(sigma_spec, rng, n),
        rho=dist.sample(p.rho.dist, rng, n),
        y_local=dist.sample(y_local_spec, rng, n),
        y_ustry=dist.sample(p.y_ustry.dist, rng, n),
        w=dist.sample(p.w.dist, rng, n),
        f_venue=dist.sample(p.f_venue.dist, rng, n),
        f_redeem=dist.sample(p.f_redeem.dist, rng, n),
        f_mint=dist.sample(p.f_mint.dist, rng, n),
        t_settle_offchain=dist.sample(p.t_settle_offchain.dist, rng, n),
        t_settle_offchain_hold=dist.sample(p.t_settle_offchain_hold.dist, rng, n),
        t_settle_onchain=dist.sample(p.t_settle_onchain.dist, rng, n),
        s_op=dist.sample(p.s_op.dist, rng, n),
        m_offchain=dist.sample(p.m_offchain.dist, rng, n),
        m_onchain=dist.sample(p.m_onchain.dist, rng, n),
        k_risk=float(p.k_risk.point),  # fixed after calibration (§7.2)
        K=dist.sample(p.K.dist, rng, n),
        var_charge=float(p.var_charge.point),
    )


def point_params(cfg: Config, corridor: str, regime: str, T: float, beta: float | None = None) -> EconParams:
    """Build a deterministic single-point ``EconParams`` (modes/means)."""

    p = cfg.params
    corr = cfg.corridor(corridor)
    reg = cfg.regime(regime)
    sigma_spec = corr.sigma or p.sigma.dist
    y_local_spec = corr.y_local or p.y_local.dist
    b = dist.point(reg.beta) if beta is None else beta
    return EconParams(
        T=float(T),
        beta=float(b),
        sigma=dist.point(sigma_spec),
        rho=p.rho.point,
        y_local=dist.point(y_local_spec),
        y_ustry=p.y_ustry.point,
        w=p.w.point,
        f_venue=p.f_venue.point,
        f_redeem=p.f_redeem.point,
        f_mint=p.f_mint.point,
        t_settle_offchain=p.t_settle_offchain.point,
        t_settle_offchain_hold=p.t_settle_offchain_hold.point,
        t_settle_onchain=p.t_settle_onchain.point,
        s_op=p.s_op.point,
        m_offchain=p.m_offchain.point,
        m_onchain=p.m_onchain.point,
        k_risk=p.k_risk.point,
        K=p.K.point,
        var_charge=p.var_charge.point,
    )


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
@dataclass
class CellResult:
    trials: pd.DataFrame  # long format, one row per (trial, model)
    summary: pd.DataFrame  # one row per model (+ paired advantage on C rows)


def run_cell(
    cfg: Config,
    corridor: str,
    regime: str,
    T: float,
    n: int,
    models: list[str] | None = None,
    target_bps: float | None = None,
) -> CellResult:
    """Run one (corridor, regime, T) cell × n trials over all requested models."""

    models = models or cfg.grid.models
    target = cfg.subsidy.target_bps if target_bps is None else target_bps
    rng = cell_rng(cfg.seed, corridor, regime, T)
    p = sample_params(cfg, corridor, regime, T, n, rng)

    V = np.asarray(p.T * p.K, dtype=float)  # daily volume per trial (USD/day)
    per_model: dict[str, dict[str, np.ndarray]] = {}
    frames = []
    for key in models:
        spec = MODELS[key]
        comp = spread_components(spec, p)
        roc = mm_roc_annual(spec, p, comp["s_fx"])
        c_total = np.asarray(comp["c_total"], dtype=float)
        # instantaneous subsidy rate at this cell (USD/day): the gap above target on flow
        subsidy_rate = np.maximum(0.0, (c_total - target) / 1e4) * V
        per_model[key] = {**comp, "roc": roc, "subsidy_rate": subsidy_rate}

        df = pd.DataFrame({col: np.asarray(comp[col], dtype=float) for col in TERM_COLUMNS})
        df.insert(0, "model", key)
        df.insert(0, "T", float(T))
        df.insert(0, "regime", regime)
        df.insert(0, "corridor", corridor)
        df["trial"] = np.arange(n)
        df["roc"] = np.asarray(roc, dtype=float)
        df["beta"] = np.asarray(p.beta, dtype=float)
        df["sigma"] = np.asarray(p.sigma, dtype=float)
        df["y_blend"] = np.asarray(comp["y_blend"], dtype=float)
        df["friction_breach"] = np.asarray(comp["friction_breach"], dtype=float)
        df["subsidy_rate"] = subsidy_rate
        frames.append(df)

    trials = pd.concat(frames, ignore_index=True)
    summary = _summarise_cell(corridor, regime, T, per_model, target)
    return CellResult(trials=trials, summary=summary)


def _summarise_cell(
    corridor: str,
    regime: str,
    T: float,
    per_model: dict[str, dict[str, np.ndarray]],
    target_bps: float,
) -> pd.DataFrame:
    rows = []
    a = per_model.get("A", {}).get("c_total")
    b = per_model.get("B", {}).get("c_total")

    for key, comp in per_model.items():
        c_total = np.asarray(comp["c_total"], dtype=float)
        s_fx = np.asarray(comp["s_fx"], dtype=float)
        subsidy_rate = np.asarray(comp["subsidy_rate"], dtype=float)
        row = {
            "corridor": corridor,
            "regime": regime,
            "T": float(T),
            "model": key,
            "c_total_mean": float(np.mean(c_total)),
            "c_total_std": float(np.std(c_total)),
            "c_total_min": float(np.min(c_total)),
            "c_total_max": float(np.max(c_total)),
            "s_fx_mean": float(np.mean(s_fx)),
            "s_fx_std": float(np.std(s_fx)),
            "s_fx_min": float(np.min(s_fx)),
            "s_fx_max": float(np.max(s_fx)),
            "roc_mean": float(np.mean(np.asarray(comp["roc"], dtype=float))),
            "p_le_target": float(np.mean(c_total <= target_bps)),
            "target_bps": float(target_bps),
            "p_friction_breach": float(np.mean(np.asarray(comp["friction_breach"], dtype=float))),
            "subsidy_rate_mean": float(np.mean(subsidy_rate)),
            "subsidy_rate_p50": float(np.percentile(subsidy_rate, 50)),
        }
        for q in PCTS:
            row[f"c_total_p{q}"] = float(np.percentile(c_total, q))
            row[f"s_fx_p{q}"] = float(np.percentile(s_fx, q))

        # paired advantage (common random numbers) on the C rows (§6.5)
        if key.startswith("C"):
            if a is not None:
                adv = a - c_total
                row["adv_vs_A_mean"] = float(np.mean(adv))
                row["adv_vs_A_p50"] = float(np.percentile(adv, 50))
                row["p_better_than_A"] = float(np.mean(c_total <= a))
            if b is not None:
                adv = b - c_total
                row["adv_vs_B_mean"] = float(np.mean(adv))
                row["adv_vs_B_p50"] = float(np.percentile(adv, 50))
                row["p_better_than_B"] = float(np.mean(c_total <= b))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Full sweep
# --------------------------------------------------------------------------- #
@dataclass
class SweepResult:
    trials: pd.DataFrame
    summary: pd.DataFrame


def run_sweep(cfg: Config, n: int | None = None, progress=None) -> SweepResult:
    """Run the full corridor × T × regime grid (spec §6.3)."""

    n = n or cfg.trials
    trial_frames = []
    summary_frames = []
    cells = [
        (c, r, t)
        for c in cfg.grid.corridors
        for r in cfg.grid.regimes
        for t in cfg.grid.turnovers
    ]
    for corridor, regime, T in cells:
        res = run_cell(cfg, corridor, regime, T, n)
        trial_frames.append(res.trials)
        summary_frames.append(res.summary)
        if progress is not None:
            progress(corridor, regime, T)

    trials = pd.concat(trial_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    return SweepResult(trials=trials, summary=summary)
