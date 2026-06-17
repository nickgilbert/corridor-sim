"""Configuration models, defaults, loading/merging and validation (spec §4, §5, §6).

The default configuration is defined programmatically in :func:`build_default_config`
so the engine and its acceptance tests never depend on a YAML file being present.
``config/default.yaml`` is a serialized copy for transparency and overriding; loading
merges a YAML file (and an optional corridor preset) *over* these defaults.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
class DistSpec(BaseModel):
    """A sampling distribution for a parameter (spec §4)."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["fixed", "triangular", "normal", "lognormal"] = "fixed"
    # args interpreted per family:
    #   fixed       -> {value}
    #   triangular  -> {lo, mode, hi}
    #   normal      -> {mean, sd, lo?, hi?}   (truncated if lo/hi given)
    #   lognormal   -> {mean, sd}             (of the underlying normal, base e)
    args: dict[str, float] = Field(default_factory=dict)


class ParamSpec(BaseModel):
    """A parameter with a deterministic point default and an MC distribution."""

    model_config = ConfigDict(extra="forbid")

    point: float
    dist: DistSpec = Field(default_factory=DistSpec)

    @model_validator(mode="after")
    def _default_fixed_to_point(self) -> "ParamSpec":
        # A bare ``fixed`` distribution with no value samples the point default.
        if self.dist.family == "fixed" and "value" not in self.dist.args:
            self.dist.args = {"value": self.point}
        return self


# --------------------------------------------------------------------------- #
# Scenario structure
# --------------------------------------------------------------------------- #
class RegimeSpec(BaseModel):
    """A flow/balance regime (spec §5)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    beta: DistSpec
    calibration_turnover: float  # representative T used when fitting k_risk (§7.2)
    band_bps: tuple[float, float]  # illustrative competitive S_fx band the regime targets


class CorridorSpec(BaseModel):
    """A corridor preset (spec §4 'Corridor presets')."""

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str = ""
    sigma: DistSpec | None = None  # overrides params.sigma
    y_local: DistSpec | None = None  # overrides params.y_local
    representative_turnover: float = 5.0  # mature/representative T for headline outputs
    provisional: bool = False


class SubsidyConfig(BaseModel):
    """Front-loaded declining subsidy ramp (spec §3.5)."""

    model_config = ConfigDict(extra="forbid")

    t_start: float = 0.01  # near-zero turnover at launch (standing up a brand-new market)
    t_mature: float = 1.0  # early-stage maturity within the modelled horizon
    horizon_months: int = 12
    steepness: float = 6.0  # logistic steepness of the S-curve
    target_bps: float = 25.0  # wholesale floor; effective target = min(this, incumbent)
    regime: str = "persistent"  # corridors SDF subsidises are persistent-imbalance


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corridor: str = "MX_MXN"
    trials: int = 20_000
    k_bounds: tuple[float, float] = (0.01, 3.0)


class GridConfig(BaseModel):
    """The sweep grid (spec §6.3)."""

    model_config = ConfigDict(extra="forbid")

    corridors: list[str] = Field(default_factory=lambda: ["MX_MXN", "TR_TRY"])
    turnovers: list[float] = Field(default_factory=lambda: [0.5, 1.0, 5.0, 10.0, 20.0])
    regimes: list[str] = Field(default_factory=lambda: ["balanced", "intermittent", "persistent"])
    models: list[str] = Field(
        default_factory=lambda: ["A", "B", "C_xborder", "C_same_denom"]
    )


class Params(BaseModel):
    """All economic parameters (spec §4 table)."""

    model_config = ConfigDict(extra="forbid")

    sigma: ParamSpec
    rho: ParamSpec
    y_local: ParamSpec  # local tokenized treasury yield (CETES for MX)
    y_ustry: ParamSpec  # USD tokenized treasury yield
    w: ParamSpec  # inventory leg weight
    f_venue: ParamSpec
    f_redeem: ParamSpec  # on-chain redeem (asset->cash) fee, Models B/C
    f_mint: ParamSpec  # on-chain mint (cash->asset) fee; 0 by default — issuers monetise float
    t_settle_offchain: ParamSpec
    t_settle_offchain_hold: ParamSpec
    t_settle_onchain: ParamSpec
    s_op: ParamSpec
    m_offchain: ParamSpec
    m_onchain: ParamSpec
    k_risk: ParamSpec
    K: ParamSpec
    var_charge: ParamSpec  # optional VaR charge against the yield credit (Appendix B)


class Config(BaseModel):
    """Top-level resolved configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    trials: int = 10_000
    params: Params
    regimes: dict[str, RegimeSpec]
    corridors: dict[str, CorridorSpec]
    grid: GridConfig = Field(default_factory=GridConfig)
    subsidy: SubsidyConfig = Field(default_factory=SubsidyConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    # --------------------------------------------------------------------- #
    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def corridor(self, name: str) -> CorridorSpec:
        if name not in self.corridors:
            raise KeyError(f"Unknown corridor '{name}'. Known: {list(self.corridors)}")
        return self.corridors[name]

    def regime(self, name: str) -> RegimeSpec:
        if name not in self.regimes:
            raise KeyError(f"Unknown regime '{name}'. Known: {list(self.regimes)}")
        return self.regimes[name]


# --------------------------------------------------------------------------- #
# Defaults (the §4/§5 content, encoded once)
# --------------------------------------------------------------------------- #
def _p(point: float, family: str = "fixed", **args: float) -> ParamSpec:
    return ParamSpec(point=point, dist=DistSpec(family=family, args=args))


def build_default_config() -> Config:
    """Construct the canonical default configuration (spec §4, §5, §6)."""

    params = Params(
        sigma=_p(0.12, "triangular", lo=0.05, mode=0.12, hi=0.35),
        rho=_p(0.15, "triangular", lo=0.08, mode=0.15, hi=0.25),
        y_local=_p(0.095, "triangular", lo=0.06, mode=0.095, hi=0.115),  # CETES 6-11%
        y_ustry=_p(0.045, "triangular", lo=0.035, mode=0.045, hi=0.055),  # USD T-bill 4-5%
        w=_p(0.5, "triangular", lo=0.3, mode=0.5, hi=0.7),
        f_venue=_p(0.0025, "triangular", lo=0.0015, mode=0.0025, hi=0.0040),
        f_redeem=_p(0.0003, "triangular", lo=0.0001, mode=0.0003, hi=0.0008),
        f_mint=_p(0.0),  # mint at face value is free (V6 §4.3); issuer earns the float
        t_settle_offchain=_p(1.5, "triangular", lo=0.5, mode=1.5, hi=3.0),
        t_settle_offchain_hold=_p(1.5, "triangular", lo=0.5, mode=1.5, hi=3.0),
        t_settle_onchain=_p(1.16e-5),  # ~1 second, fixed
        s_op=_p(0.0005, "triangular", lo=0.0002, mode=0.0005, hi=0.0010),
        m_offchain=_p(0.0010, "triangular", lo=0.0, mode=0.0010, hi=0.0030),
        m_onchain=_p(0.0, "triangular", lo=0.0, mode=0.0, hi=0.0002),
        k_risk=_p(0.35),  # placeholder; fit by `calibrate` (§7.2)
        K=_p(10_000_000.0),
        var_charge=_p(0.0),  # Appendix B.1: off by default, documented honest unknown
    )

    regimes = {
        "balanced": RegimeSpec(
            name="balanced",
            beta=DistSpec(family="triangular", args={"lo": 0.00, "mode": 0.05, "hi": 0.15}),
            calibration_turnover=20.0,
            band_bps=(1.0, 3.0),
        ),
        "intermittent": RegimeSpec(
            name="intermittent",
            beta=DistSpec(family="triangular", args={"lo": 0.15, "mode": 0.35, "hi": 0.55}),
            calibration_turnover=5.0,
            band_bps=(10.0, 25.0),
        ),
        "persistent": RegimeSpec(
            name="persistent",
            beta=DistSpec(family="triangular", args={"lo": 0.60, "mode": 0.80, "hi": 1.00}),
            calibration_turnover=1.0,
            band_bps=(30.0, 60.0),
        ),
    }

    corridors = {
        "MX_MXN": CorridorSpec(
            name="MX_MXN",
            label="Mexico (MXN) — CETES, low FX vol",
            sigma=DistSpec(family="triangular", args={"lo": 0.05, "mode": 0.12, "hi": 0.35}),
            y_local=DistSpec(family="triangular", args={"lo": 0.06, "mode": 0.095, "hi": 0.115}),
            representative_turnover=10.0,
            provisional=False,
        ),
        "TR_TRY": CorridorSpec(
            name="TR_TRY",
            label="Turkey (TRY) — high FX vol stress",
            sigma=DistSpec(family="triangular", args={"lo": 0.15, "mode": 0.35, "hi": 0.60}),
            y_local=DistSpec(family="triangular", args={"lo": 0.30, "mode": 0.42, "hi": 0.50}),
            representative_turnover=10.0,
            provisional=False,
        ),
        "BR_BRL": CorridorSpec(
            name="BR_BRL",
            label="Brazil (BRL) — TESOURO, mid vol (provisional)",
            sigma=DistSpec(family="triangular", args={"lo": 0.08, "mode": 0.16, "hi": 0.40}),
            y_local=DistSpec(family="triangular", args={"lo": 0.09, "mode": 0.105, "hi": 0.135}),
            representative_turnover=10.0,
            provisional=True,
        ),
        "AR_ARS": CorridorSpec(
            name="AR_ARS",
            label="Argentina (ARS) — very high vol (provisional)",
            sigma=DistSpec(family="triangular", args={"lo": 0.25, "mode": 0.55, "hi": 0.90}),
            y_local=DistSpec(family="triangular", args={"lo": 0.30, "mode": 0.60, "hi": 0.90}),
            representative_turnover=10.0,
            provisional=True,
        ),
        "IN_INR": CorridorSpec(
            name="IN_INR",
            label="India (INR) — mid vol (provisional)",
            sigma=DistSpec(family="triangular", args={"lo": 0.04, "mode": 0.08, "hi": 0.20}),
            y_local=DistSpec(family="triangular", args={"lo": 0.06, "mode": 0.070, "hi": 0.085}),
            representative_turnover=10.0,
            provisional=True,
        ),
        "VN_VND": CorridorSpec(
            name="VN_VND",
            label="Vietnam (VND) — low/mid vol (provisional)",
            sigma=DistSpec(family="triangular", args={"lo": 0.03, "mode": 0.07, "hi": 0.18}),
            y_local=DistSpec(family="triangular", args={"lo": 0.03, "mode": 0.045, "hi": 0.060}),
            representative_turnover=10.0,
            provisional=True,
        ),
        "ID_IDR": CorridorSpec(
            name="ID_IDR",
            label="Indonesia (IDR) — mid vol (provisional)",
            sigma=DistSpec(family="triangular", args={"lo": 0.05, "mode": 0.10, "hi": 0.25}),
            y_local=DistSpec(family="triangular", args={"lo": 0.055, "mode": 0.065, "hi": 0.080}),
            representative_turnover=10.0,
            provisional=True,
        ),
    }

    return Config(
        seed=42,
        trials=10_000,
        params=params,
        regimes=regimes,
        corridors=corridors,
        grid=GridConfig(),
        subsidy=SubsidyConfig(),
        calibration=CalibrationConfig(),
    )


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Load a config, merging a YAML override (if any) over the programmatic defaults."""

    default = build_default_config().model_dump(mode="json")
    if path is None:
        return Config.model_validate(default)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        override = yaml.safe_load(fh) or {}

    # Merge an optional corridor preset directory reference, then the file's contents.
    merged = _deep_merge(default, override)
    return Config.model_validate(merged)
