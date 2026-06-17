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

    row = {
        "corridor": corridor, "n": int(n), "T_hub": Th, "beta_hub": Bh,
        "A": A, "B_local": B_local, "C_local_ustry": C_local_ustry,
        "C_local_usdc": C_local_usdc, "C_xborder": C_xborder,
    }
    #   B steelman: local→USDC→local, but the USDC hub fragments across S stablecoins.
    for S in (1,) + tuple(nc.b_frag_levels):
        Th_b = hub_turnover(n, nc, fragments=S)
        Bh_b = hub_beta(n, nc, fragments=S)
        bx1 = _hop_sfx(cfg, corridor, "B", T=Th_b, beta=Bh_b, sigma=sig)
        bx2 = _hop_sfx(cfg, dst, "B", T=Th_b, beta=Bh_b, sigma=sig_dst)
        row[f"B_xborder_s{S}"] = s_op + bx1 + bx2
    return row


def network_sweep(cfg: Config, nc: NetworkConfig | None = None) -> pd.DataFrame:
    """Per-corridor costs as the network grows N = 1 … len(corridors) (corridors added in order)."""
    nc = nc or NetworkConfig()
    rows = []
    for n in range(1, len(nc.corridors) + 1):
        for corridor in nc.corridors[:n]:
            rows.append(corridor_costs(cfg, corridor, n, nc))
    return pd.DataFrame(rows)
