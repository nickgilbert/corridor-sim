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
from .subsidy import run_subsidy  # noqa: E402

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
        friction_pct=friction_pct,
        mature_T_label=f"{mature_T:g}",
        regimes=list(trials["regime"].unique()),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
