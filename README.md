# Corridor Economics Simulation Engine

A Monte Carlo engine that quantifies the all-in cross-currency spread a user pays on a
payment corridor, and shows why the **denomination-matched reserved ramp** (Model C)
structurally beats the two alternative ramp/FX architectures. It implements the economics
of the V6 *Distribution as Our Growth Engine* strategy doc as closed-form spread equations
evaluated per trial, with Monte Carlo sampling over uncertain inputs.

```
pip install -e ".[dev]"
corridor-sim all --config config/default.yaml --out runs/demo   # calibrate -> sweep -> report
open runs/demo/report.html
```

`corridor-sim all` runs in a few seconds; the full 30-cell × 10k-trial × 4-model sweep is
< 2 s. Re-running with the same `--seed` reproduces identical numbers.

## The three models

Every model prices **one unit of cross-currency flow**. The user's all-in cost is
`C_total = S_ramp + S_fx` (bps). The only thing that differs between models is the ramp's
balance sheet and where FX settles — so all differences in `C_total` are structural.

| Model | Ramp inventory | FX venue | Rebalance | Yield | Returns to scale |
|---|---|---|---|---|---|
| **A** — incumbent | fiat **+** USDC (mismatched) | off-chain venue | wire + ~25 bps venue fee, **days** | 0% | **none** — venue fee never retires |
| **B** — stable/stable | local stablecoin (matched) | on-chain DEX | mint/redeem at face value, **seconds** | 0% | yes, but only to **cash-FX parity** |
| **C** — reserved ramp | tokenized treasury (matched + yield) | on-chain DEX | mint/redeem at face value, **seconds** | 4–11% APY | yes, and **below** parity |

Model **C** is reported in two pair types: **C-xborder** (yield↔yield, e.g. CETES↔USTRY —
both legs earn carry) and **C-same-denom** (yield↔stable, e.g. USTRY↔USDC — one yield leg
per trade). C-xborder carries the most yield and settles lowest.

The yield edge is the **yield subsidy** `ΔS = 2·y_daily / T`: a maker whose inventory earns
yield can quote `ΔS` tighter than a cash desk at the same return on capital. It flows through
the FX cost-of-capital carry term, so by construction `C_total(B) − C_total(C) ≈ ΔS`.

> The ramp's *own* idle-inventory yield credit (§3.3) is implemented (`ModelSpec.ramp_yield`)
> but **off by default**, so the headline B−C gap equals exactly ΔS as the §7.3 acceptance
> invariant requires. Turn it on to book the extra income line; it widens C's edge.

## How to read the report

1. **Executive summary** — median `C_total` (bps) per model at the corridor's mature
   turnover, by flow regime, plus `P(C ≤ 25 bps)`. The headline numbers.
2. **Distribution of all-in cost** — violins of `C_total` for A/B/C at mature T, with p5–p95
   whiskers. The dashed line is the ≤25 bps wholesale target.
3. **Returns to scale (the money chart)** — `S_fx` (median + p25–p75 fan) vs turnover. **A is
   flat and high** (its venue fee never retires); **B declines to the cash-FX-parity floor**;
   **C declines below it**. This is the central claim, made visual.
4. **Yield-subsidy panel** — `ΔS = 2·y_daily/T` vs turnover. Shaded band is the 0.20–0.50 bps
   structural-edge zone. ΔS shrinks as 1/T, so the edge is largest in thin, early corridors.
5. **Cost-stack waterfall** — mean decomposition of `C_total` into operating, margin,
   cost-of-capital carry, risk premium, and rebalance, side by side per model.
6. **Paired advantage** — per-trial bps saved by C vs A and C vs B (common random numbers),
   with means and P(>0).
7. **Subsidy burn-down** — cumulative SDF subsidy over a 24-month turnover S-curve. **C's
   subsidy retires to zero; A's does not** (A's break-even never falls below the target).
8. **Sensitivity (tornado)** — one-at-a-time effect of each parameter on median `C_total(C)`,
   ranked by swing; σ, β, yields and T dominate.
9. **Assumptions & caveats** — the honest open questions: VaR vs the yield edge
   (`var_charge`), interest-rate risk, bond-market liquidity, the friction differential
   (`f_redeem·β` must stay below ΔS), and that imbalance — not turnover — is the adversary.

## CLI

```
corridor-sim calibrate --config config/default.yaml --out runs/latest   # fit k_risk (§7.2)
corridor-sim run        --config config/default.yaml --out runs/latest --trials 10000 --seed 42
corridor-sim report     --in runs/latest --out runs/latest/report.html
corridor-sim all        --config config/default.yaml --out runs/latest   # all three
```

`run` calibrates first by default (`--no-calibrate` to reuse a pre-fitted config, e.g. the
`config_resolved.yaml` written by `calibrate`). `examples/run_all.sh` shows the staged flow.

## Outputs (`--out <dir>`)

- `results_trials.parquet` — every trial (cell, model, all spread terms incl. `delta_s` /
  `friction_cost` / `friction_breach`, totals, ROC, per-trial subsidy rate).
- `results_summary.csv` — per cell × model aggregate stats (mean/std/min/max/percentiles of
  `C_total` and `S_fx`, P(≤target), paired advantage vs A and B, `p_friction_breach`,
  mean/median subsidy rate).
- `results_subsidy.csv` / `results_subsidy_summary.csv` — the 24-month burn-down curves and
  per-model retirement summary.
- `config_resolved.yaml` — the exact resolved config used (with the fitted `k_risk`).
- `calibration.json` — fitted `k_risk`, regime band checks, yield-subsidy anchor check.
- `report.html` — self-contained, regenerable from the parquet alone.

### Stress preset (Appendix B.3)

`corridor-sim all --config config/stress.yaml --out runs/stress` models a bond-market
liquidity crisis: a higher redeem fee, much longer on-chain settlement, and a non-zero
`var_charge` against the yield edge. It drives the friction differential
(`f_redeem·β > ΔS`) toward breach — the `friction_breach` flag (Appendix B.4) makes the
failure mode visible per trial.

## Calibration

The model has a single free coefficient, `k_risk`, that converts a 1-σ price move into the
charged risk premium. It is fit so the median competitive on-chain spread lands inside the
doc's illustrative bands at each regime's representative turnover (balanced ∈ [1,3], intermittent
∈ [10,25], persistent ∈ [30,60] bps). Everything else is grounded in the strategy doc and
overridable in `config/default.yaml`. If the analytic form drifts, `k_risk` is refit — the
bands (the acceptance anchors) never move.

## Tests

`pytest` encodes the §7 acceptance contract: the exact yield-subsidy table, regime-band
calibration, structural ordering / returns-to-scale / parity-ceiling invariants, the ≤25 bps
target probability, venue-fee bookkeeping conservation, subsidy retirement, and seed
reproducibility.

## Layout

```
config/            default.yaml + corridor presets
src/corridor_sim/  config, distributions, economics, models, montecarlo,
                   calibrate, subsidy, report, cli
templates/         report.html.j2
tests/             the §7 acceptance suite
examples/run_all.sh
```
