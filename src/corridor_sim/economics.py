"""The economic model: closed-form spread equations (spec §3).

All spreads are decimal internally (0.0025 = 25 bps); every public function that
reports a spread returns **basis points**. Time is in days; yields/rates are annual
unless suffixed ``_daily`` (``_daily = _annual / 365``).

The whole cost stack is defined **once** in :func:`_components_decimal`; the per-model
differences are entirely which terms are switched on and which inputs they receive
(see :mod:`corridor_sim.models`). This is what makes A/B/C provably comparable and the
waterfall chart trivial (spec §3.3 implementation note).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import ModelSpec

BPS = 1e4
DAYS_PER_YEAR = 365.0
VOL_TRADING_DAYS = 252.0  # σ_daily = σ_annual / sqrt(252)


@dataclass
class EconParams:
    """A (possibly vectorised) parameter vector fed to the cost stack.

    Scalars and NumPy arrays are both accepted; arithmetic is vectorised, so a single
    instance can carry an entire Monte Carlo run's worth of trials.
    """

    T: Any  # daily turnover (capital velocity) = V / K  [×/day]
    beta: Any  # flow imbalance = net/gross ∈ [0,1]
    sigma: Any  # annualised FX volatility (fraction/yr)
    rho: Any  # hurdle return on capital (annual fraction)
    y_local: Any  # local tokenized-treasury yield (annual)
    y_ustry: Any  # USD tokenized-treasury yield (annual)
    w: Any  # inventory weight on leg 1
    f_venue: Any  # off-chain FX venue fee (fraction)
    f_redeem: Any  # on-chain face-value redeem fee (fraction)
    t_settle_offchain: Any  # off-chain rebalance settlement time (days)
    t_settle_offchain_hold: Any  # off-chain minimum holding latency (days)
    t_settle_onchain: Any  # on-chain settlement / min latency (days)
    s_op: Any  # ramp operating spread (fraction)
    m_offchain: Any  # off-chain residual margin (fraction)
    m_onchain: Any  # on-chain residual margin (fraction, ~0)
    k_risk: Any  # risk-premium calibration coefficient
    K: Any  # working capital (USD)
    var_charge: Any = 0.0  # optional VaR charge against the yield edge (Appendix B.1)


# --------------------------------------------------------------------------- #
# Yield subsidy (the heart of the yield edge, spec §3.2 / §7.1)
# --------------------------------------------------------------------------- #
def yield_subsidy_bps(y_annual: Any, T: Any) -> Any:
    """ΔS = 2 · (y_annual / 365) / T, in basis points (spec §7.1)."""

    return 2.0 * (np.asarray(y_annual, dtype=float) / DAYS_PER_YEAR) / np.asarray(T, dtype=float) * BPS


def blended_yield(model: ModelSpec, p: EconParams) -> Any:
    """Blended annual FX-inventory yield per model (spec §3.3 last row)."""

    if model.yield_pair == "none":
        return np.zeros_like(np.asarray(p.w, dtype=float)) if np.ndim(p.w) else 0.0
    if model.yield_pair == "yield_yield":  # C-xborder: both legs yield
        return p.w * p.y_local + (1.0 - p.w) * p.y_ustry
    if model.yield_pair == "yield_stable":  # C-same-denom: one yield leg (USTRY) + stable
        return p.w * p.y_ustry
    raise ValueError(f"Unknown yield_pair {model.yield_pair!r}")


# --------------------------------------------------------------------------- #
# The cost stack (spec §3.2, §3.3) — defined once, decimal
# --------------------------------------------------------------------------- #
def _components_decimal(
    model: ModelSpec, p: EconParams, *, book_venue_in_ramp: bool = False
) -> dict[str, Any]:
    rho_d = p.rho / DAYS_PER_YEAR
    y_ann = blended_yield(model, p)
    # VaR charge erodes the usable yield edge (Appendix B.1); 0 by default.
    y_eff = y_ann - p.var_charge if model.ramp_yield else y_ann
    y_d = y_eff / DAYS_PER_YEAR
    sig_d = p.sigma / np.sqrt(VOL_TRADING_DAYS)

    onchain = model.onchain
    t_min = p.t_settle_onchain if onchain else p.t_settle_offchain_hold
    t_set = p.t_settle_onchain if onchain else p.t_settle_offchain
    f_reb = p.f_redeem if onchain else p.f_venue
    margin = p.m_onchain if onchain else p.m_offchain

    # holding window grows with imbalance, shrinks with turnover (§3.2)
    t_hold = t_min + p.beta / p.T

    carry = 2.0 * (rho_d - y_d) / p.T          # net cost-of-capital carry
    risk = p.k_risk * sig_d * np.sqrt(t_hold)  # price risk over the holding window
    rebal = f_reb * p.beta + p.k_risk * sig_d * np.sqrt(t_set)  # rebalance cost on imbalanced flow

    # The ramp's own yield credit: idle-treasury drag inverts into income (V6 §4.3).
    yield_credit = (2.0 * y_d / p.T) if model.ramp_yield else 0.0
    s_op = p.s_op

    # Bookkeeping choice (§3.3 note): for the off-chain mismatched ramp the venue
    # rebalance term may be booked under S_ramp instead of S_fx. C_total is invariant.
    s_fx = margin + carry + risk + rebal
    s_ramp = s_op - yield_credit
    if book_venue_in_ramp and not onchain:
        s_fx = margin + carry + risk
        s_ramp = s_ramp + rebal

    c_total = s_ramp + s_fx

    # Friction differential (Appendix B.4): the redeem/venue fee paid on imbalanced flow
    # must stay below the yield edge ΔS, or the structural advantage is eaten by friction.
    delta_s = 2.0 * y_d / p.T  # yield subsidy (decimal); 0 when y_d == 0
    friction_cost = f_reb * p.beta
    # Only meaningful where there is a yield edge to erode (the C models).
    friction_breach = (
        (np.asarray(friction_cost, dtype=float) > np.asarray(delta_s, dtype=float)).astype(float)
        if model.yield_pair != "none"
        else np.zeros_like(np.asarray(friction_cost, dtype=float))
    )

    return {
        "operating": s_op,
        "margin": margin,
        "carry": carry,
        "risk": risk,
        "rebalance": rebal,
        "ramp_yield_credit": yield_credit,
        "y_blend": y_ann,
        "delta_s": delta_s,
        "friction_cost": friction_cost,
        "friction_breach": friction_breach,
        "s_fx": s_fx,
        "s_ramp": s_ramp,
        "c_total": c_total,
    }


def spread_components(
    model: ModelSpec, p: EconParams, *, book_venue_in_ramp: bool = False
) -> dict[str, Any]:
    """Return every additive spread term plus totals, in **basis points** (spec §3.3).

    Keys: ``operating, margin, carry, risk, rebalance, ramp_yield_credit, delta_s,
    friction_cost, s_fx, s_ramp, c_total`` (all bps); ``y_blend`` (annual fraction);
    ``friction_breach`` (0/1 indicator, Appendix B.4).
    """

    d = _components_decimal(model, p, book_venue_in_ramp=book_venue_in_ramp)
    not_bps = {"y_blend", "friction_breach"}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = v if k in not_bps else v * BPS
    return out


# --------------------------------------------------------------------------- #
# Market-maker return on capital (diagnostic, spec §3.4)
# --------------------------------------------------------------------------- #
def mm_roc_annual(model: ModelSpec, p: EconParams, s_quote_bps: Any) -> Any:
    """Realised annualised ROC for a maker quoting ``s_quote_bps`` (spec §3.4).

    Built so that at ``s_quote = s_fx`` the ROC equals the hurdle ρ by construction.
    """

    d = _components_decimal(model, p)
    s_quote = np.asarray(s_quote_bps, dtype=float) / BPS
    y_ann = blended_yield(model, p)
    y_eff = y_ann - p.var_charge if model.ramp_yield else y_ann
    y_d = y_eff / DAYS_PER_YEAR
    V = p.T * p.K

    # cost side, consistent with the §3.2 break-even identity
    cost_spread = d["margin"] + d["risk"] + d["rebalance"]
    daily_profit = V * (s_quote / 2.0) + p.K * y_d - V * (cost_spread / 2.0)
    return DAYS_PER_YEAR * daily_profit / p.K
