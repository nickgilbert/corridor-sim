"""Network (hub-and-spoke) layer — Act II.

Routes corridors through a shared **USTRY** reserve (tokenized US Treasuries) and shows how
per-corridor cost and network subsidy change as corridors are added. Pools are `X ↔ USTRY`
(X = a local bond, or USDC the dollar-exit). The `USTRY ↔ USDC` hub is **same-currency** (USD
vs USD), so it carries no FX risk; its turnover aggregates every corridor's flow (carry → 0)
and its imbalance nets down as the network diversifies (β_hub → floor).

Per-hop pricing reuses :func:`economics.spread_components`; this module only adds the routing,
the hub aggregation, and the β-netting law.

Path types priced here (all-in = one ramp `s_op` + Σ hop `S_fx`):
  - ``A``              incumbent: one off-chain venue hop (baseline).
  - ``B_local``        stable→stable: one on-chain hop local-stable → USDC.
  - ``C_local_ustry``  one on-chain hop local-bond → USTRY (yield↔yield); recipient holds USTRY.
  - ``C_local_usdc``   local-bond → USTRY (spoke) + USTRY → USDC (shared hub).
  - ``B_xborder``      stable→stable cross-border: local → USDC → local (USDC as hub).
  - ``C_xborder``      yield cross-border: local → USTRY → local (USTRY as hub, no stablecoin).

Assumptions (all in :class:`NetworkConfig`, adjustable):
  - Capital efficiency [Q1, regime c]: one shared hub book, ``T_hub = min(ΣV / K_hub, ceiling)``.
  - Netting [Q2]: ``β_hub(N) = β_floor + (β_0 − β_floor) / N**netting_exp``.
  - Steelman B routes cross-border through USDC as a hub (same aggregation as C in V1).

Deferred (see docs/TODO.md): explicit spoke price-impact term; USDC-hub liquidity
fragmentation as stablecoins diversify (the dynamic that ultimately breaks B's aggregation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .economics import spread_components
from .models import MODELS
from .montecarlo import point_params

BPS = 1e4
DAYS_PER_MONTH = 365.0 / 12.0
# default network: corridors added left-to-right (Q: MXN → BRL → KRW → TRY)
DEFAULT_CORRIDORS = ["MX_MXN", "BR_BRL", "KR_KRW", "TR_TRY"]


@dataclass
class NetworkConfig:
    corridors: list[str] = field(default_factory=lambda: list(DEFAULT_CORRIDORS))
    t_corridor: float = 10.0  # per-corridor (spoke) turnover for the static N-comparison
    # Q1 — capital efficiency at the shared hub
    k_hub_ratio: float = 1.0  # K_hub / K ; 1.0 = one shared book the size of one corridor's
    t_ceiling: float = 5000.0  # block-rate × utilization cap on hub turnover
    # Q2 — netting law for hub imbalance
    beta_0: float = 0.8  # single same-direction corridor
    beta_floor: float = 0.2  # mature, netted hub (ASSUMPTION — adjustable)
    netting_exp: float = 0.5  # √N diversification law
    # spoke imbalance: single-currency, stays one-directional (no netting)
    beta_spoke: float = 0.8
    # TODO #2 — stablecoin fragmentation: B's USDC hub splits across this many stablecoins,
    # so its flow can't aggregate into one deep pool. C's USTRY reserve is always unified (1).
    b_frag_levels: tuple[int, ...] = (3, 10)
    # fragmentation that GROWS with the network: each new market spawns its own stablecoin,
    # so S(N) = round(b_frag_rate · N). At rate 1.0 every corridor adds a coin → B never
    # consolidates (turnover per pool stays flat, β never nets). Reality is between this and
    # the unified steelman (S=1).
    b_frag_rate: float = 1.0
    # subsidy-view ramp: the network burn-down runs to MATURE turnover so retirement is
    # visible (the §7 early-stage ramp stops at 1×/day, too thin for anyone to break even).
    sub_t_start: float = 0.01
    sub_t_mature: float = 20.0
    sub_horizon: int = 24


def frag_growing(n: int, nc: NetworkConfig) -> int:
    """Number of stablecoins B fragments across when fragmentation grows with the network."""
    return max(1, round(nc.b_frag_rate * max(1, int(n))))


def hub_beta(n: int, nc: NetworkConfig, fragments: int = 1) -> float:
    """Hub imbalance β_hub vs network size N (netting law, Q2).

    ``fragments`` > 1 (stablecoin fragmentation) splits flow across pools, so each pool nets
    over only N/fragments corridors — fragmented liquidity can't offset across coins.
    """
    n = max(1, int(n))
    n_eff = max(1.0, n / max(1, fragments))
    return nc.beta_floor + (nc.beta_0 - nc.beta_floor) / (n_eff ** nc.netting_exp)


def hub_turnover(n: int, nc: NetworkConfig, fragments: int = 1) -> float:
    """Aggregated hub turnover T_hub = min(ΣV / (K_hub·fragments), ceiling).

    ``fragments`` > 1 spreads the aggregate flow across that many shallow stablecoin pools,
    so each one runs at 1/fragments the turnover — the heart of why B can't compound.
    """
    n = max(1, int(n))
    aggregate = n * nc.t_corridor / (nc.k_hub_ratio * max(1, fragments))
    return min(aggregate, nc.t_ceiling)


# --------------------------------------------------------------------------- #
def _hop_sfx(cfg: Config, corridor: str, model_key: str, *, T: float, beta: float,
             sigma: float | None = None, y_local: float | None = None,
             y_ustry: float | None = None) -> float:
    """S_fx (bps) for one hop, reusing the per-hop economics with overrides."""
    p = point_params(cfg, corridor, "persistent", float(T))
    p.T = float(T)
    p.beta = float(beta)
    if sigma is not None:
        p.sigma = float(sigma)
    if y_local is not None:
        p.y_local = float(y_local)
    if y_ustry is not None:
        p.y_ustry = float(y_ustry)
    return float(spread_components(MODELS[model_key], p)["s_fx"])


def _sigma_of(cfg: Config, corridor: str) -> float:
    c = cfg.corridor(corridor)
    spec = c.sigma or cfg.params.sigma.dist
    return float(spec.args.get("mode", spec.args.get("value", cfg.params.sigma.point)))


def _y_local_of(cfg: Config, corridor: str) -> float:
    c = cfg.corridor(corridor)
    spec = c.y_local or cfg.params.y_local.dist
    return float(spec.args.get("mode", spec.args.get("value", cfg.params.y_local.point)))


def corridor_costs(cfg: Config, corridor: str, n: int, nc: NetworkConfig) -> dict:
    """All-in cost (bps) for `corridor` under each path type, with the network at size N.

    For cross-border, the second leg uses the next corridor in the list as the destination.
    """
    s_op = float(cfg.params.s_op.point) * BPS
    y_ustry = float(cfg.params.y_ustry.point)
    sig = _sigma_of(cfg, corridor)
    y_loc = _y_local_of(cfg, corridor)
    T0 = nc.t_corridor
    Th = hub_turnover(n, nc)
    Bh = hub_beta(n, nc)

    # destination corridor for cross-border (wrap around the active set)
    active = nc.corridors[:max(1, n)]
    dst = active[(active.index(corridor) + 1) % len(active)] if len(active) > 1 else corridor
    sig_dst = _sigma_of(cfg, dst)
    y_dst = _y_local_of(cfg, dst)

    # incumbent: single off-chain venue hop
    A = s_op + _hop_sfx(cfg, corridor, "A", T=T0, beta=nc.beta_spoke, sigma=sig)
    # stable→stable to USDC: single on-chain hop, no yield
    B_local = s_op + _hop_sfx(cfg, corridor, "B", T=T0, beta=nc.beta_spoke, sigma=sig)
    # local→USTRY (yield↔yield), deliver yield-bearing dollars; per-corridor spoke
    C_local_ustry = s_op + _hop_sfx(cfg, corridor, "C_xborder", T=T0, beta=nc.beta_spoke,
                                    sigma=sig, y_local=y_loc, y_ustry=y_ustry)
    # local→USTRY spoke + USTRY→USDC shared hub (same-currency: σ≈0)
    spoke = _hop_sfx(cfg, corridor, "C_xborder", T=T0, beta=nc.beta_spoke, sigma=sig,
                     y_local=y_loc, y_ustry=y_ustry)
    hub = _hop_sfx(cfg, corridor, "C_same_denom", T=Th, beta=Bh, sigma=0.0, y_ustry=y_ustry)
    C_local_usdc = s_op + spoke + hub
    # cross-border: both legs route through the aggregating hub.
    #   C: local→USTRY→local (yield legs, unified reserve — fragments=1)
    cx1 = _hop_sfx(cfg, corridor, "C_xborder", T=Th, beta=Bh, sigma=sig, y_local=y_loc, y_ustry=y_ustry)
    cx2 = _hop_sfx(cfg, dst, "C_xborder", T=Th, beta=Bh, sigma=sig_dst, y_local=y_dst, y_ustry=y_ustry)
    C_xborder = s_op + cx1 + cx2

    # incumbent cross-border: two off-chain venue hops (no hub)
    A_xborder = (s_op + _hop_sfx(cfg, corridor, "A", T=T0, beta=nc.beta_spoke, sigma=sig)
                 + _hop_sfx(cfg, dst, "A", T=T0, beta=nc.beta_spoke, sigma=sig_dst))

    row = {
        "corridor": corridor, "n": int(n), "T_hub": Th, "beta_hub": Bh,
        "A": A, "A_xborder": A_xborder, "B_local": B_local, "C_local_ustry": C_local_ustry,
        "C_local_usdc": C_local_usdc, "C_xborder": C_xborder,
    }
    #   B steelman: local→USDC→local, but the USDC hub fragments across S stablecoins.
    def _b_xborder(S: int) -> float:
        Th_b = hub_turnover(n, nc, fragments=S)
        Bh_b = hub_beta(n, nc, fragments=S)
        bx1 = _hop_sfx(cfg, corridor, "B", T=Th_b, beta=Bh_b, sigma=sig)
        bx2 = _hop_sfx(cfg, dst, "B", T=Th_b, beta=Bh_b, sigma=sig_dst)
        return s_op + bx1 + bx2

    for S in (1,) + tuple(nc.b_frag_levels):
        row[f"B_xborder_s{S}"] = _b_xborder(S)
    # fragmentation growing with the network (a stablecoin per market)
    row["B_xborder_grow"] = _b_xborder(frag_growing(n, nc))
    row["frag_grow"] = frag_growing(n, nc)
    return row


def network_sweep(cfg: Config, nc: NetworkConfig | None = None) -> pd.DataFrame:
    """Per-corridor costs as the network grows N = 1 … len(corridors) (corridors added in order)."""
    nc = nc or NetworkConfig()
    rows = []
    for n in range(1, len(nc.corridors) + 1):
        for corridor in nc.corridors[:n]:
            rows.append(corridor_costs(cfg, corridor, n, nc))
    return pd.DataFrame(rows)


def _xborder_cost(cfg: Config, corridor: str, dst: str, *, kind: str, t_spoke: float,
                  T_hub: float, beta_hub: float, nc: NetworkConfig) -> float:
    """Cross-border all-in cost (bps) given an explicit hub state. kind ∈ {A, B, C}."""
    s_op = float(cfg.params.s_op.point) * BPS
    y_ustry = float(cfg.params.y_ustry.point)
    sig, sigd = _sigma_of(cfg, corridor), _sigma_of(cfg, dst)
    yl, yld = _y_local_of(cfg, corridor), _y_local_of(cfg, dst)
    if kind == "A":  # incumbent: two off-chain venue hops
        return s_op + _hop_sfx(cfg, corridor, "A", T=t_spoke, beta=nc.beta_spoke, sigma=sig) \
            + _hop_sfx(cfg, dst, "A", T=t_spoke, beta=nc.beta_spoke, sigma=sigd)
    if kind == "C":  # yield legs through the unified USTRY hub
        return s_op + _hop_sfx(cfg, corridor, "C_xborder", T=T_hub, beta=beta_hub, sigma=sig, y_local=yl, y_ustry=y_ustry) \
            + _hop_sfx(cfg, dst, "C_xborder", T=T_hub, beta=beta_hub, sigma=sigd, y_local=yld, y_ustry=y_ustry)
    # B: stable legs through the (fragmented) USDC hub
    return s_op + _hop_sfx(cfg, corridor, "B", T=T_hub, beta=beta_hub, sigma=sig) \
        + _hop_sfx(cfg, dst, "B", T=T_hub, beta=beta_hub, sigma=sigd)


def network_subsidy(cfg: Config, nc: NetworkConfig | None = None):
    """Cumulative network subsidy over time as corridors launch staggered and ramp.

    Per-corridor subsidy summed at the network level (Q: per-corridor, sum). C routes the
    unified USTRY hub; B fragments a stablecoin per market (frag grows with N). Returns
    (curve DataFrame, launch schedule dict, retire summary dict).
    """
    import copy

    from .subsidy import turnover_curve

    nc = nc or NetworkConfig()
    target_bps = cfg.subsidy.target_bps
    K = cfg.params.K.point
    # run the burn-down on a maturity-reaching ramp (not the §7 early-stage ramp) so the
    # retire/not-retire contrast is visible.
    cfg_ramp = copy.deepcopy(cfg)
    cfg_ramp.subsidy.t_start = nc.sub_t_start
    cfg_ramp.subsidy.t_mature = nc.sub_t_mature
    cfg_ramp.subsidy.horizon_months = nc.sub_horizon
    horizon = nc.sub_horizon
    ramp = turnover_curve(cfg_ramp)  # per-corridor turnover by age
    corridors = nc.corridors
    gap = max(1, horizon // len(corridors))
    launch = {c: min(i * gap, horizon) for i, c in enumerate(corridors)}

    cum = {"A": 0.0, "C": 0.0, "B_grow": 0.0}
    rows = []
    for m in range(horizon + 1):
        live = [c for c in corridors if launch[c] <= m]
        n = len(live)
        t_spoke = {c: float(ramp[min(m - launch[c], len(ramp) - 1)]) for c in live}
        agg = sum(t_spoke.values()) / nc.k_hub_ratio  # = ΣV / K_hub (unified)
        Sg = frag_growing(n, nc)
        hub = {"C": (agg, hub_beta(n, nc, 1)), "B_grow": (agg / Sg, hub_beta(n, nc, Sg))}
        day = {k: 0.0 for k in cum}
        for i, c in enumerate(live):
            dst = live[(i + 1) % n] if n > 1 else c
            V = t_spoke[c] * K
            for kind, key in (("A", "A"), ("C", "C"), ("B", "B_grow")):
                Th, Bh = hub.get(key, (t_spoke[c], nc.beta_spoke))
                cost = _xborder_cost(cfg, c, dst, kind=kind, t_spoke=t_spoke[c], T_hub=Th, beta_hub=Bh, nc=nc)
                day[key] += max(0.0, (cost - target_bps) / BPS) * V
        for k in cum:
            cum[k] += day[k] * DAYS_PER_MONTH
            rows.append({"month": m, "model": k, "cumulative_subsidy": cum[k],
                         "daily_subsidy": day[k], "live": n})
    curve = pd.DataFrame(rows)
    # "retires" = daily subsidy has fallen ~to zero by the horizon
    summary = {}
    for k in cum:
        tail = curve[(curve.model == k) & (curve.month >= horizon - 1)]["daily_subsidy"]
        summary[k] = {"total_subsidy": cum[k], "retires": bool(tail.max() < 1.0)}
    return curve, launch, summary
