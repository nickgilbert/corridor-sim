"""The four architectures compared by the engine (spec §2).

Each :class:`ModelSpec` is just a set of flags telling :func:`economics.spread_components`
which terms are switched on and which inputs they receive. This keeps A/B/C provably
comparable — the *only* differences between models live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

YieldPair = Literal["none", "yield_yield", "yield_stable"]


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short id used in the grid / outputs
    label: str  # human-readable name
    onchain: bool  # FX settles on-chain (B, C) vs off-chain venue (A)
    ramp_yield: bool  # ramp's own inventory earns yield (drag -> income line)
    yield_pair: YieldPair  # how blended FX-inventory yield is computed
    description: str = ""


# Model A — traditional ramp, off-chain FX (incumbent / baseline). §2 Model A.
MODEL_A = ModelSpec(
    key="A",
    label="A: Traditional ramp, off-chain FX",
    onchain=False,
    ramp_yield=False,
    yield_pair="none",
    description="Denomination-mismatched fiat+USDC, off-chain venue FX, days-long "
    "rebalance, idle (0% yield) inventory. No returns to scale.",
)

# Model B — denomination-matched, stable-vs-stable on-chain FX, no yield. §2 Model B.
MODEL_B = ModelSpec(
    key="B",
    label="B: Denomination-matched, stable/stable on-chain FX",
    onchain=True,
    ramp_yield=False,
    yield_pair="none",
    description="Local stablecoin matched to fiat, on-chain DEX FX, seconds-long "
    "rebalance, idle stablecoins (0% yield). Returns to scale, capped at cash-FX parity.",
)

# Model C-xborder — reserved ramp, yield<->yield on-chain FX (CETES<->USTRY). §2 Model C.
#
# NOTE on ``ramp_yield``: the yield edge is modelled through the FX cost-of-capital carry
# (carry uses ``y_blend``), which is exactly the documented subsidy ΔS = 2·y_daily/T. The
# *additional* ramp_yield_credit on S_ramp (§3.3) is implemented in economics.py but kept
# OFF by default so the §7.3 acceptance invariant — C_total(B) − C_total(C) ≈ ΔS — holds
# exactly. Flip ``ramp_yield=True`` to book the extra idle-inventory income line; doing so
# widens C's edge beyond ΔS.
MODEL_C_XBORDER = ModelSpec(
    key="C_xborder",
    label="C (x-border): Reserved ramp, yield/yield on-chain FX",
    onchain=True,
    ramp_yield=False,
    yield_pair="yield_yield",
    description="Tokenized local treasury vs tokenized USD treasury (e.g. CETES<->USTRY). "
    "Both legs earn carry; yield funds a quote below cash-FX parity.",
)

# Model C-same-denom — reserved ramp, yield<->stable on-chain FX (USTRY<->USDC). §2 Model C.
MODEL_C_SAME_DENOM = ModelSpec(
    key="C_same_denom",
    label="C (same-denom): Reserved ramp, yield/stable on-chain FX",
    onchain=True,
    ramp_yield=False,
    yield_pair="yield_stable",
    description="Tokenized USD treasury vs USD stablecoin (USTRY<->USDC). Hub-and-spoke: "
    "the yield leg sits on one side of every trade (~50% of inventory always earning).",
)


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in (MODEL_A, MODEL_B, MODEL_C_XBORDER, MODEL_C_SAME_DENOM)
}
