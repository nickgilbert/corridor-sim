"""Self-contained HTML report with embedded charts (spec §8).

Every Monte Carlo chart renders from ``results_trials.parquet`` alone; the analytic
panels (yield-subsidy curve, subsidy burn-down, tornado) are cheap deterministic
recomputations from the resolved config, so the report is fully regenerable without
re-simulating.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from .config import Config, load_config  # noqa: E402
from .economics import spread_components, yield_subsidy_bps  # noqa: E402
from .models import MODELS  # noqa: E402
from .montecarlo import point_params  # noqa: E402
from .subsidy import DAYS_PER_MONTH, breakeven_bps, run_subsidy, turnover_curve  # noqa: E402

# consistent colour per model across every chart (spec §8)
MODEL_COLORS = {
    "A": "#c0392b",          # incumbent — red
    "B": "#2980b9",          # stable/stable — blue
    "C_xborder": "#27ae60",  # reserved x-border — green
    "C_same_denom": "#16a085",  # reserved same-denom — teal
}
MODEL_ORDER = ["A", "B", "C_xborder", "C_same_denom"]
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _nearest_T(trials: pd.DataFrame, target: float) -> float:
    ts = np.array(sorted(trials["T"].unique()))
    return float(ts[np.argmin(np.abs(ts - target))])


def _model_label(key: str) -> str:
    return MODELS[key].label


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_distributions(trials: pd.DataFrame, corridor: str, regime: str, T: float) -> str:
    sub = trials[(trials.corridor == corridor) & (trials.regime == regime) & (trials["T"] == T)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data, labels, colors = [], [], []
    for key in MODEL_ORDER:
        d = sub[sub.model == key]["c_total"].values
        if d.size:
            data.append(d)
            labels.append(key)
            colors.append(MODEL_COLORS[key])
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
    for key, col, d in zip(labels, colors, data):
        p5, p95 = np.percentile(d, [5, 95])
        idx = labels.index(key) + 1
        ax.plot([idx, idx], [p5, p95], color=col, lw=1.4, alpha=0.9)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("All-in cost C_total (bps)")
    ax.set_title(f"C_total distribution — {corridor}, {regime}, T={T:g}× (p5–p95 whiskers)")
    ax.axhline(25, color="grey", ls="--", lw=1, label="≤25 bps target")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def chart_turnover_curves(trials: pd.DataFrame, corridor: str, regime: str) -> str:
    sub = trials[(trials.corridor == corridor) & (trials.regime == regime)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in MODEL_ORDER:
        d = sub[sub.model == key]
        if d.empty:
            continue
        g = d.groupby("T")["s_fx"]
        Ts = np.array(sorted(d["T"].unique()))
        med = g.median().reindex(Ts).values
        p25 = g.quantile(0.25).reindex(Ts).values
        p75 = g.quantile(0.75).reindex(Ts).values
        ax.plot(Ts, med, "-o", color=MODEL_COLORS[key], label=key, lw=2, ms=4)
        ax.fill_between(Ts, p25, p75, color=MODEL_COLORS[key], alpha=0.15)
    ax.axhline(25, color="grey", ls="--", lw=1, label="≤25 bps target")
    ax.set_xlabel("Daily turnover T (×/day)")
    ax.set_ylabel("Competitive FX spread S_fx (bps)")
    ax.set_title(f"Returns to scale — {corridor}, {regime} (median, p25–p75 fan)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_b64(fig)


def chart_yield_subsidy() -> str:
    Ts = np.linspace(0.25, 20, 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y in (0.05, 0.07, 0.095):
        ax.plot(Ts, yield_subsidy_bps(y, Ts), label=f"y_blend = {y*100:.1f}%")
    ax.axhspan(0.20, 0.50, color="gold", alpha=0.25, label="structural-edge zone (0.20–0.50 bps)")
    ax.set_xlabel("Daily turnover T (×/day)")
    ax.set_ylabel("Yield subsidy ΔS = 2·y_daily/T (bps)")
    ax.set_title("Yield subsidy vs turnover (ΔS shrinks as 1/T)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_b64(fig)


def chart_waterfall(trials: pd.DataFrame, corridor: str, regime: str, T: float) -> str:
    sub = trials[(trials.corridor == corridor) & (trials.regime == regime) & (trials["T"] == T)]
    terms = ["operating", "margin", "carry", "risk", "rebalance"]
    term_colors = ["#7f8c8d", "#95a5a6", "#e67e22", "#9b59b6", "#34495e"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(MODEL_ORDER))
    bottoms = np.zeros(len(MODEL_ORDER))
    for term, col in zip(terms, term_colors):
        vals = np.array(
            [sub[sub.model == k][term].mean() if not sub[sub.model == k].empty else 0.0 for k in MODEL_ORDER]
        )
        ax.bar(x, vals, bottom=bottoms, label=term, color=col)
        bottoms += vals
    # yield credit is a negative (income) term
    credit = np.array(
        [-sub[sub.model == k]["ramp_yield_credit"].mean() if not sub[sub.model == k].empty else 0.0 for k in MODEL_ORDER]
    )
    ax.bar(x, credit, bottom=0, label="ramp_yield_credit (income)", color="#2ecc71")
    totals = np.array(
        [sub[sub.model == k]["c_total"].mean() if not sub[sub.model == k].empty else 0.0 for k in MODEL_ORDER]
    )
    for xi, t in zip(x, totals):
        ax.text(xi, t, f"{t:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Mean spread term (bps)")
    ax.set_title(f"Cost-stack waterfall — {corridor}, {regime}, T={T:g}×")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def chart_paired_advantage(trials: pd.DataFrame, corridor: str, regime: str, T: float) -> str:
    sub = trials[(trials.corridor == corridor) & (trials.regime == regime) & (trials["T"] == T)]
    piv = sub.pivot_table(index="trial", columns="model", values="c_total")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if {"A", "C_xborder"}.issubset(piv.columns):
        adv = (piv["A"] - piv["C_xborder"]).values
        ax.hist(adv, bins=60, color=MODEL_COLORS["A"], alpha=0.5,
                label=f"A − C (mean {adv.mean():.1f}, P>0 {(adv > 0).mean():.0%})")
    if {"B", "C_xborder"}.issubset(piv.columns):
        adv = (piv["B"] - piv["C_xborder"]).values
        ax.hist(adv, bins=60, color=MODEL_COLORS["B"], alpha=0.5,
                label=f"B − C (mean {adv.mean():.2f}, P>0 {(adv > 0).mean():.0%})")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Spread saved per unit flow (bps)")
    ax.set_ylabel("Trials")
    ax.set_title(f"Paired advantage (common random numbers) — {corridor}, {regime}, T={T:g}×")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_b64(fig)


def chart_subsidy(cfg: Config, corridor: str) -> tuple[str, "pd.DataFrame"]:
    res = run_subsidy(cfg, corridor)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in MODEL_ORDER:
        d = res.curve[res.curve.model == key]
        if d.empty:
            continue
        ax.plot(d["month"], d["cumulative_subsidy"] / 1e6, "-", color=MODEL_COLORS[key], label=key, lw=2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative subsidy (USD millions)")
    ax.set_title(f"Subsidy burn-down — {corridor} (target {res.target_bps:.1f} bps, persistent)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_b64(fig), res.summary


def chart_tornado(cfg: Config, corridor: str, regime: str, T: float) -> str:
    """One-at-a-time sensitivity of median C_total(C_xborder) (spec §8.8)."""

    p = cfg.params
    base_p = point_params(cfg, corridor, regime, T)
    spec = MODELS["C_xborder"]

    def c_total(modifier) -> float:
        pp = point_params(cfg, corridor, regime, T)
        modifier(pp)
        return float(spread_components(spec, pp)["c_total"])

    base = float(spread_components(spec, base_p)["c_total"])
    corr = cfg.corridor(corridor)
    reg = cfg.regime(regime)

    def rng_of(distspec):
        a = distspec.args
        if distspec.family == "triangular":
            return a["lo"], a["hi"]
        return None

    factors = {
        "sigma": (rng_of(corr.sigma or p.sigma.dist), lambda v: ("sigma", v)),
        "beta": (rng_of(reg.beta), lambda v: ("beta", v)),
        "rho": (rng_of(p.rho.dist), lambda v: ("rho", v)),
        "y_local": (rng_of(corr.y_local or p.y_local.dist), lambda v: ("y_local", v)),
        "y_ustry": (rng_of(p.y_ustry.dist), lambda v: ("y_ustry", v)),
        "w": (rng_of(p.w.dist), lambda v: ("w", v)),
        "f_redeem": (rng_of(p.f_redeem.dist), lambda v: ("f_redeem", v)),
        "T": (((min(cfg.grid.turnovers), max(cfg.grid.turnovers))), lambda v: ("T", v)),
    }

    swings = []
    for name, (bounds, _) in factors.items():
        if bounds is None:
            continue
        lo, hi = bounds
        low = c_total(lambda pp, n=name, v=lo: setattr(pp, n, v))
        high = c_total(lambda pp, n=name, v=hi: setattr(pp, n, v))
        swings.append((name, low, high))

    swings.sort(key=lambda s: abs(s[2] - s[1]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(swings))
    for i, (name, low, high) in enumerate(swings):
        lo_v, hi_v = sorted([low, high])
        col = "#e67e22" if name in {"sigma", "beta", "y_local", "y_ustry", "T"} else "#7f8c8d"
        ax.barh(i, hi_v - lo_v, left=lo_v, color=col, alpha=0.8)
    ax.axvline(base, color="k", ls="--", lw=1, label=f"baseline {base:.1f} bps")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in swings])
    ax.set_xlabel("C_total(C x-border) (bps)")
    ax.set_title(f"Sensitivity tornado — {corridor}, {regime}, T={T:g}×")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    return _fig_to_b64(fig)


def _total_subsidy_at_yield(cfg: Config, corridor: str, y_blend: float, target: float) -> float:
    """Total subsidy (USD) over the ramp for a reserved ramp whose blended yield is y_blend.

    Uses the yield/yield model with both legs forced to y_blend, so y_blend = 0 reproduces
    the stable/stable (no-yield) baseline exactly — the curve's intercept is Model B.
    """

    regime = cfg.subsidy.regime
    Ts = turnover_curve(cfg)
    K = cfg.params.K.point
    spec = MODELS["C_xborder"]
    cum = 0.0
    for T in Ts:
        p = point_params(cfg, corridor, regime, float(T))
        p.y_local = y_blend
        p.y_ustry = y_blend  # both legs = y_blend → blended_yield == y_blend
        be = float(spread_components(spec, p)["c_total"])
        gap = max(0.0, (be - target) / 1e4)
        cum += gap * (T * K) * DAYS_PER_MONTH
    return cum


def subsidy_yield_sensitivity(cfg: Config, corridor: str):
    """Total subsidy vs blended yield over the early-stage ramp (chart b64, table records)."""

    regime = cfg.subsidy.regime
    Ts = turnover_curve(cfg)
    target = min(cfg.subsidy.target_bps, breakeven_bps(cfg, corridor, regime, float(Ts[-1]), "A"))

    ys = np.linspace(0.0, 0.12, 25)
    subs = np.array([_total_subsidy_at_yield(cfg, corridor, float(y), target) for y in ys])
    baseline = subs[0]  # y = 0 → stable/stable (Model B)
    default_y = float(spread_components(MODELS["C_xborder"], point_params(cfg, corridor, regime, float(Ts[-1])))["y_blend"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ys * 100, subs / 1e6, "-", color=MODEL_COLORS["C_xborder"], lw=2, label="reserved ramp (C)")
    ax.axhline(baseline / 1e6, color=MODEL_COLORS["B"], ls="--", lw=1.4, label="stable/stable (B), y=0")
    ax.fill_between(ys * 100, subs / 1e6, baseline / 1e6, color="#2ecc71", alpha=0.18,
                    label="subsidy bought back by yield")
    ax.axvline(default_y * 100, color="#7f8c8d", ls=":", lw=1.2, label=f"default y_blend = {default_y*100:.1f}%")
    ax.set_xlabel("Blended inventory yield (% APY)")
    ax.set_ylabel(f"Total subsidy over {cfg.subsidy.horizon_months}mo (USD millions)")
    ax.set_title(f"Subsidy vs yield — {corridor}, early-stage ramp ({regime})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    img = _fig_to_b64(fig)

    rows = []
    for y in (0.0, 0.02, 0.045, 0.07, 0.095, 0.12):
        s = _total_subsidy_at_yield(cfg, corridor, y, target)
        rows.append({
            "yield_pct": round(y * 100, 1),
            "total_subsidy": round(s, 0),
            "saved_vs_b": round(baseline - s, 0),
            "saved_pct": round((baseline - s) / baseline * 100, 1) if baseline else 0.0,
        })
    return img, rows, round(target, 1)


def network_charts(cfg: Config):
    """Act II — hub-and-spoke compounding: 4 charts (b64) + summary tables."""
    from .network import NetworkConfig, hub_beta, network_subsidy, network_sweep

    nc = NetworkConfig()
    df = network_sweep(cfg, nc)
    ref = "MX_MXN" if "MX_MXN" in nc.corridors else nc.corridors[0]
    mx = df[df.corridor == ref].set_index("n").sort_index()
    Ns = list(mx.index)
    GREEN, BLUE, NAVY, RED = "#27ae60", "#2980b9", "#1f3a5f", "#c0392b"

    # Panel 1 — hero: cross-border cost vs N
    f1, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(Ns, mx["C_xborder"], "-o", color=GREEN, lw=2, label="C — USTRY hub (unified reserve)")
    ax.plot(Ns, mx["B_xborder_s10"], "-^", color=NAVY, lw=2, label="B — USDC hub, fixed 10 stablecoins")
    ax.plot(Ns, mx["B_xborder_grow"], "-s", color=BLUE, lw=2, label="B — a stablecoin per market (grows with N)")
    ax.plot(Ns, mx["A_xborder"], "--", color=RED, lw=1.4, label="A — incumbent (2 off-chain hops)")
    ax.axhline(25, color="grey", ls=":", lw=1, label="25 bps target")
    ax.set_xticks(Ns)
    ax.set_xlabel("Corridors on the network (N)")
    ax.set_ylabel(f"{ref} cross-border all-in cost (bps)")
    ax.set_title("Cross-border compounding: unified treasury hub vs fragmenting stablecoins")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    img1 = _fig_to_b64(f1)

    # Panel 2 — mechanism: effective hub imbalance β_hub vs N
    f2, ax = plt.subplots(figsize=(8, 4.5))
    for S, lab, col, mk in [(1, "C — USTRY (unified)", GREEN, "-o"), (3, "B — 3 stablecoins", BLUE, "-s"),
                            (10, "B — 10 stablecoins", NAVY, "-^")]:
        ax.plot(Ns, [hub_beta(n, nc, fragments=S) for n in Ns], mk, color=col, lw=2, label=lab)
    ax.axhline(nc.beta_floor, color="grey", ls=":", lw=1, label=f"β floor {nc.beta_floor}")
    ax.set_xticks(Ns)
    ax.set_xlabel("Corridors on the network (N)")
    ax.set_ylabel("Effective hub imbalance β_hub")
    ax.set_title("Why: a unified reserve nets toward the floor; fragmented liquidity cannot")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    img2 = _fig_to_b64(f2)

    # Panel 3 — Point 1: like-for-like dominance across corridors (deliver a dollar asset)
    dom = df[df.n == len(nc.corridors)].set_index("corridor")
    corrs = [c for c in nc.corridors if c in dom.index]
    x = np.arange(len(corrs))
    f3, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.25, [dom.loc[c, "A"] for c in corrs], 0.25, color=RED, label="A — incumbent")
    ax.bar(x, [dom.loc[c, "B_local"] for c in corrs], 0.25, color=BLUE, label="B — stable→USDC")
    ax.bar(x + 0.25, [dom.loc[c, "C_local_ustry"] for c in corrs], 0.25, color=GREEN, label="C — local→USTRY")
    ax.axhline(25, color="grey", ls=":", lw=1, label="25 bps target")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_")[-1] for c in corrs])
    ax.set_ylabel("All-in cost (bps)")
    ax.set_title("Point 1 — yield is ≤ B and ≪ A in every corridor (deliver a dollar asset)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    img3 = _fig_to_b64(f3)

    # Panel 4 — network subsidy over time (corridors launch staggered, ramp to maturity)
    curve, launch, summ = network_subsidy(cfg, nc)
    piv = curve.pivot_table(index="month", columns="model", values="cumulative_subsidy")
    f4, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(piv.index, piv["A"] / 1e6, "--", color=RED, lw=1.4, label="A — incumbent")
    ax.plot(piv.index, piv["B_grow"] / 1e6, "-s", color=BLUE, lw=2, label="B — fragmenting stablecoins")
    ax.plot(piv.index, piv["C"] / 1e6, "-o", color=GREEN, lw=2, label="C — unified USTRY hub")
    ax.set_xlabel("Month (corridors launch staggered, ramp to maturity)")
    ax.set_ylabel("Cumulative network subsidy (USD millions)")
    ax.set_title("Network subsidy: C retires; fragmented stable-to-stable and incumbent do not")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    img4 = _fig_to_b64(f4)

    rows = []
    for n in (Ns[0], Ns[-1]):
        r = mx.loc[n]
        rows.append({
            "n": int(n), "C": round(r["C_xborder"], 1), "B_s10": round(r["B_xborder_s10"], 1),
            "B_grow": round(r["B_xborder_grow"], 1),
            "gap_grow": round(r["B_xborder_grow"] - r["C_xborder"], 1),
        })
    sub_summary = {k: {"retires": v["retires"], "total_m": round(v["total_subsidy"] / 1e6, 1)}
                   for k, v in summ.items()}
    return ({"hero": img1, "beta": img2, "dominance": img3, "subsidy": img4},
            rows, ref, sub_summary)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_report(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "report.html"

    trials = pd.read_parquet(run_dir / "results_trials.parquet")
    summary = pd.read_csv(run_dir / "results_summary.csv")
    cfg = load_config(run_dir / "config_resolved.yaml")
    calib = {}
    calib_path = run_dir / "calibration.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())

    corridors = list(trials["corridor"].unique())
    primary = "MX_MXN" if "MX_MXN" in corridors else corridors[0]
    mature_T = _nearest_T(trials, cfg.corridor(primary).representative_turnover)

    # Executive summary table
    exec_rows = []
    for regime in trials["regime"].unique():
        sub = trials[(trials.corridor == primary) & (trials.regime == regime) & (trials["T"] == mature_T)]
        row = {"regime": regime}
        for key in MODEL_ORDER:
            d = sub[sub.model == key]["c_total"]
            row[f"{key}_median"] = round(float(d.median()), 1) if len(d) else None
        cx = sub[sub.model == "C_xborder"]["c_total"]
        row["C_p_le_25"] = round(float((cx <= 25).mean()), 2) if len(cx) else None
        exec_rows.append(row)

    # friction-differential breach rate for the reserved ramp (Appendix B.4)
    fr = summary[
        (summary.corridor == primary)
        & (summary.regime == "persistent")
        & (summary["T"] == mature_T)
        & (summary.model == "C_xborder")
    ]
    friction_pct = (
        round(float(fr["p_friction_breach"].iloc[0]) * 100, 1)
        if len(fr) and "p_friction_breach" in fr.columns
        else None
    )

    charts = {
        "yield_subsidy": chart_yield_subsidy(),
        "distributions": {},
        "turnover": {},
        "waterfall": chart_waterfall(trials, primary, "persistent", mature_T),
        "paired": chart_paired_advantage(trials, primary, "persistent", mature_T),
        "tornado": chart_tornado(cfg, primary, "persistent", mature_T),
    }
    for regime in trials["regime"].unique():
        charts["distributions"][regime] = chart_distributions(trials, primary, regime, mature_T)
        charts["turnover"][regime] = chart_turnover_curves(trials, primary, regime)

    subsidy_img, subsidy_summary = chart_subsidy(cfg, primary)
    charts["subsidy"] = subsidy_img

    # Turnover S-curve assumed by the burn-down (volume V = T·K), shown as a table so the
    # ramp assumption is explicit rather than buried in the chart.
    Ts = turnover_curve(cfg)
    K = cfg.params.K.point
    turnover_ramp = [
        {"month": int(m), "T": round(float(T), 3), "volume_m": round(float(T) * K / 1e6, 2)}
        for m, T in enumerate(Ts)
    ]

    sens_img, sens_rows, sens_target = subsidy_yield_sensitivity(cfg, primary)
    charts["subsidy_yield"] = sens_img

    net_charts, net_rows, net_ref, net_sub = network_charts(cfg)
    charts["network"] = net_charts
    # NaN month_to_retire (non-retiring models) -> None so the template shows "—"
    subsidy_records = [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        for rec in subsidy_summary.to_dict("records")
    ]

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")
    html = template.render(
        primary=primary,
        corridor_label=cfg.corridor(primary).label,
        mature_T=mature_T,
        exec_rows=exec_rows,
        model_order=MODEL_ORDER,
        model_labels={k: _model_label(k) for k in MODEL_ORDER},
        model_colors=MODEL_COLORS,
        model_descriptions={k: MODELS[k].description for k in MODEL_ORDER},
        charts=charts,
        calib=calib,
        subsidy_summary=subsidy_records,
        turnover_ramp=turnover_ramp,
        horizon_months=cfg.subsidy.horizon_months,
        ramp_t_start=cfg.subsidy.t_start,
        ramp_t_mature=cfg.subsidy.t_mature,
        working_capital_m=round(cfg.params.K.point / 1e6, 2),
        subsidy_yield_rows=sens_rows,
        subsidy_yield_target=sens_target,
        network_rows=net_rows,
        network_ref=net_ref,
        network_subsidy=net_sub,
        friction_pct=friction_pct,
        mature_T_label=f"{mature_T:g}",
        regimes=list(trials["regime"].unique()),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
