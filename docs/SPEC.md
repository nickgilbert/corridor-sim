# Corridor Economics Simulation Engine — Requirements & Build Spec

**Status:** Build-ready specification (v1)
**Audience:** Claude Code (single-pass implementation) + reviewing engineers/economists
**Source of record:** *V6 — Distribution as Our Growth Engine: Corridor Strategy Explainer* (Sections 4, 6, 7, 8.3, 9) and the supporting notes *Advantages to Making Markets with Yield Bearing Assets*, *Yield Subsidy*, *Reserved Ramps Model*, *On/Off Ramp Economics*.

---

## 0. How to use this document

This is a complete specification for a Monte Carlo simulation engine that demonstrates the economic performance of the **denomination-matched reserved ramp** value chain described in the V6 strategy doc, benchmarked against two alternative ramp/FX architectures. It is written to be handed to Claude Code and implemented in a single pass.

The reader should be able to build the engine **without re-deriving any economics**: every equation, parameter, default value, distribution, scenario, output, and acceptance test is specified below. Where the strategy doc gives illustrative numbers (e.g. the turnover table, the three flow-regime bands), those are reproduced here as **calibration anchors** and encoded as automated acceptance tests. If an equation and an anchor disagree, the anchor wins and the calibration coefficient is fit to it.

Deliverable decisions already made (do not re-litigate):

- **Form factor:** a configuration/CLI-driven **Python engine** that runs the Monte Carlo and emits (a) machine-readable results (Parquet/CSV/JSON) and (b) a self-contained **HTML report** with charts. No web app, no notebook required.
- **Modelling approach:** **closed-form analytic cost equations** evaluated per trial, with **Monte Carlo** sampling over the uncertain inputs to produce distributions of outcomes. Not agent-based; no discrete-event order book.
- **Language/stack:** Python 3.11+, NumPy, pandas, SciPy, Matplotlib, Pydantic, PyYAML, Jinja2, pytest. (Full list in §10.)

---

## 1. Background: the economics being simulated

The strategy reduces a corridor's price competitiveness to a single quantity: **the all-in spread, in basis points, that an end user pays to move value across currencies.** A user's all-in cost is the sum of two spreads (V6 §4.4):

```
C_total = S_ramp + S_fx
```

- `S_ramp` — the ramp's own spread on the local fiat leg (covers operating cost, residual margin, and — in the mismatched case — its FX/volatility buffer).
- `S_fx` — the spread incurred converting between currencies (off-chain at a venue, or on-chain on the DEX).

On an open order book, competition drives **margin → 0** (the "competitive ratchet"), so what remains in any competitive FX spread is **cost of capital + a risk premium** (V6 §7.3). The strategy's central claim is that the *structure* of the ramp's balance sheet and the *venue* of FX settlement determine how large those two terms are — and that the denomination-matched, yield-bearing "reserved ramp" structurally minimises both. The engine must make that claim quantitative and show its **distribution of outcomes** across realistic conditions.

Three economic facts from the doc drive everything (all encoded below):

1. **The two dominant ramp costs both come from rebalancing** (V6 §4.2): the off-chain FX venue fee (~25 bps on net flow) and the volatility buffer (`σ·√(holding window)`). Denomination matching deletes the venue fee (restock by mint/redeem at face value) and collapses the holding window from days to seconds (V6 §4.3).
2. **Turnover sets efficiency** (V6 §7.4): the faster and more two-sided the flow, the closer a market sits to the frictionless ideal and the tighter the competitive spread. Three **flow/balance regimes** — balanced, intermittent imbalance, persistent imbalance — produce progressively wider spreads.
3. **Yield-bearing inventory is a structural spread discount** (V6 §7.5, *Yield Subsidy*): a maker whose inventory earns yield can quote tighter than a cash desk by `ΔS ≈ 2 × DailyYield / DailyTurnover` while holding return on capital constant.

These combine into the **returns-to-scale** distinction the simulation must visibly reproduce (V6 §8.3): off-chain FX has *no* returns to scale (subsidy never retires); on-chain FX with non-yield stablecoins earns returns to scale but only to **cash-FX parity**; on-chain FX with yield-bearing inventory is the only model that settles **below incumbent FX economics**.

---

## 2. The three models to compare

The engine simulates one corridor "trade" (a unit of cross-currency flow) under three competing architectures. Each model is fully defined by (a) the ramp's inventory legs, (b) where FX settlement happens, (c) the rebalancing mechanism, and (d) whether inventory earns yield. The user-facing output for each is the **all-in spread `C_total` in bps** and, underneath it, the market maker's **return on capital** and **required subsidy**.

### Model A — Traditional ramp, off-chain FX (the incumbent / baseline)

> "Traditional ramps which match local off-chain assets against stables directly and use off-chain venues for FX settlement."

| Attribute | Value |
|---|---|
| Ramp inventory legs | Local fiat (e.g. MXN bank balance) **+** USD stablecoin (USDC) — **denomination-mismatched** |
| FX venue | **Off-chain** exchange/issuer desk, at rebalance time |
| Rebalancing | Wire accumulated local fiat to an FX venue; pay venue FX fee (~25 bps) on essentially every unit of **net** flow |
| Holding window | **Days** (wire + venue settlement) → large duration risk |
| Inventory yield | **0%** (idle fiat + idle USDC) |
| Returns to scale | **None** — venue fee scales with net flow; duration risk fixed in days |

Maps to V6 §4.3 "Incumbent" column and §8.3 case 1.

### Model B — Denomination-matched ramp, stable-vs-stable on-chain FX (no yield)

> "Denomination-matched ramps which use direct stable vs stable on-chain liquidity."

| Attribute | Value |
|---|---|
| Ramp inventory legs | Local **stablecoin** (e.g. MXNe) matched to local fiat liabilities — **denomination-matched** (no FX risk on the ramp) |
| FX venue | **On-chain** DEX: local-stable ↔ USDC |
| Rebalancing | On-chain swap + mint/redeem the local stable at face value with its issuer (small redemption fee) |
| Holding window | **Seconds** on-chain; rebalance via face-value redeem |
| Inventory yield | **0%** (idle stablecoins — "dead capital") |
| Returns to scale | **Yes, but capped at cash-FX parity** — the maker carries the same cost of capital as a cash desk |

Maps to V6 §8.3 case 2.

### Model C — Denomination-matched **reserved ramp**, yield-bearing on-chain FX (the strategy model)

> "The denomination-matched reserved ramp model, using on-chain markets between yield-bearing assets for cross-border payments **and** yield-bearing assets against stables for same-denomination stables (i.e. USTRY vs USDC)."

| Attribute | Value |
|---|---|
| Ramp inventory legs | Local **tokenized treasury** (e.g. CETES) matched to local fiat liabilities — **denomination-matched + yield-bearing** |
| FX venue | **On-chain** DEX, in one of two pair types (see below) |
| Rebalancing | On-chain swap + mint/redeem tokenized treasury at **face value** (small redemption fee) |
| Holding window | **Seconds**; risk transfers to on-chain makers at trade time |
| Inventory yield | **Positive carry** on both/one leg (4–11% APY depending on asset) |
| Returns to scale | **Yes, and below incumbent FX** — yield funds a quote below cash-FX parity |

Model C must support **two pair types**, both reported:

- **C-xborder (cross-border payments):** on-chain market between **two yield-bearing assets**, e.g. **CETES ↔ USTRY** (local MXN treasury vs USD treasury). Blended inventory yield uses both legs.
- **C-same-denom (same-denomination stables):** on-chain market between a **yield-bearing asset and a stablecoin**, e.g. **USTRY ↔ USDC**. One leg is yield-bearing (hub-and-spoke: the yield leg is on one side of every trade).

Maps to V6 §4.3 "Reserved" column, §7.1, §7.5, and §8.3 case 3.

---

## 3. The economic model (equations)

All spreads are expressed in **decimal** internally (e.g. 0.0025 = 25 bps) and converted to **bps** (`× 10_000`) only for reporting. Time is in **days**. Yields/rates are **annual** unless suffixed `_daily`. `_daily = _annual / 365`.

### 3.1 Notation

| Symbol | Meaning | Units |
|---|---|---|
| `K` | Market-maker (or ramp) working capital / inventory | USD |
| `V` | Daily traded volume across the pair | USD/day |
| `T` | **Capital velocity / daily turnover** = `V / K` | ×/day |
| `ρ` | Required (hurdle) return on capital, annual | fraction/yr |
| `y_blend` | Blended annual inventory yield (weighted across legs) | fraction/yr |
| `σ` | Annualised volatility of the FX rate between the two legs | fraction/yr |
| `β` | **Flow imbalance** = net flow / gross flow ∈ [0,1] | — |
| `f_venue` | Off-chain FX venue fee (Model A rebalancing) | fraction |
| `f_redeem` | On-chain mint/redeem (face-value) fee (Models B/C) | fraction |
| `t_settle` | Time inventory is exposed during a rebalance | days |
| `S_op` | Ramp operating spread (agents/servers/rails, amortised KYC) | fraction |
| `m` | Residual competitive margin (≈0 on open book; may persist off-chain) | fraction |
| `k_risk` | Risk-premium calibration coefficient (fit to anchors, §7) | — |
| `w` | Inventory weight on leg 1 (for blended yield) | — |

### 3.2 The competitive FX spread (unifying equation)

A market maker sets revenue equal to its required return at break-even (margin competed away). Using the doc's half-spread revenue convention (`Revenue = V · S/2`, V6 §7.5 / *Advantages* §1):

```
V · (S_fx / 2)  +  YieldIncome  −  RebalanceCost  −  RiskCost   =   ρ_daily · K
```

Solving for the competitive FX spread `S_fx` (all per-day quantities; divide $-costs by V to express as a spread):

```
S_fx = m
     + 2 · (ρ_daily − y_blend_daily) / T        # net cost-of-capital carry
     + RiskPremium                              # price risk over the holding window
     + RebalanceSpread                          # venue/redeem cost on imbalanced flow
```

Where:

- **Cost-of-capital carry** `= 2 · (ρ_daily − y_blend_daily) / T`. This is the heart of the yield edge. With `y_blend = 0` (Models A, B) it is `2·ρ_daily/T`. With yield (Model C) it is reduced by exactly the **Yield Subsidy**:

  ```
  ΔS  =  2 · y_blend_daily / T  =  2 · (y_blend_annual / 365) / T
  ```

  This term must reproduce the *Advantages* turnover table exactly (see §7 acceptance test).

- **Risk premium** (price risk while holding inventory between offsetting trades), V6 §7.4:

  ```
  RiskPremium = k_risk · σ_daily · sqrt(t_hold)
  σ_daily     = σ_annual / sqrt(252)
  ```

  `t_hold` is the expected time inventory sits before an offset arrives, increasing with imbalance and decreasing with turnover:

  ```
  t_hold = t_min + β · (1 / T)
  ```

  `t_min` is the minimum settlement/holding latency of the venue: **seconds (≈ 1e-5 days)** on-chain (Models B, C), **a configured number of days** off-chain (Model A, default 1.5 d). `k_risk` is fit once (§7) so the three regimes land in the doc's bands.

- **Rebalance spread** — cost of moving accumulated (imbalanced) inventory back, charged per unit of gross flow but scaled by imbalance `β`:

  ```
  RebalanceSpread = f_rebalance · β  +  k_risk · σ_daily · sqrt(t_settle)
  ```

  - Model A: `f_rebalance = f_venue` (~25 bps), `t_settle = t_settle_offchain` (days). **This term does not shrink with volume → no returns to scale.**
  - Models B, C: `f_rebalance = f_redeem` (small, face-value), `t_settle = t_settle_onchain` (seconds). The duration term is negligible.

### 3.3 The full all-in user cost per model

```
C_total = S_ramp + S_fx
```

`S_ramp` and the active terms of `S_fx` differ by model:

| Term | Model A (off-chain) | Model B (stable/stable) | Model C (reserved) |
|---|---|---|---|
| `S_ramp` | `S_op + ramp_fx_buffer` | `S_op` | `S_op − ramp_yield_credit` |
| `m` (FX margin) | `m_offchain` (>0 plausible) | `≈ 0` | `≈ 0` |
| Cost-of-capital carry | `2·ρ_daily/T` | `2·ρ_daily/T` | `2·(ρ_daily − y_blend_daily)/T` |
| Risk premium | `k·σ_daily·√(t_hold)`, on-chain? **no → off-chain `t_hold`** | `k·σ_daily·√(t_hold)`, `t_min` = seconds | same as B |
| Rebalance spread | `f_venue·β + k·σ_daily·√(t_settle_offchain)` | `f_redeem·β + ~0` | `f_redeem·β + ~0` |
| `y_blend` | 0 | 0 | C-xborder: `w·y_CETES+(1−w)·y_USTRY`; C-same-denom: `w·y_USTRY+(1−w)·0` |

Where:

- `ramp_fx_buffer` = the mismatched ramp's own volatility buffer + venue fee on its fiat→USD rebalance: `f_venue·β + k·σ_daily·√(t_settle_offchain)`. (In Model A the ramp **is** the FX-bearing party; whether you book the venue fee in `S_ramp` or `S_fx` must net to the same `C_total` — implement once, document the choice, and assert the invariant in tests.)
- `ramp_yield_credit` = yield earned on the ramp's own idle treasury inventory while it sits (`2·y_ramp_daily/T_ramp`), the "drag inverts into an income line" (V6 §4.3). Default: enabled for Model C, 0 for A and B.

> **Implementation note.** Define the cost stack **once** as a function `spread_components(model, params) -> dict` returning every additive term in bps, then sum. The per-model differences are entirely which terms are switched on and which inputs (`y_blend`, `t_min`, `f_rebalance`, `t_settle`, `m`) they receive. This keeps A/B/C provably comparable and makes the waterfall chart (§6) trivial.

### 3.4 Market-maker return on capital (diagnostic output)

For a *given* quoted spread `S_quote` (e.g. the corridor's competitive target), report the maker's realised annualised ROC:

```
ROC_annual = 365 · [ V·(S_quote/2) + K·y_blend_daily − V·RebalanceSpread/2... ] / K
```

Implement consistently with §3.2 (i.e. ROC at `S_quote = S_fx` equals `ρ` by construction). Use this to show that **at the same target spread, Model C clears its hurdle at lower turnover than B, and A may never clear it.**

### 3.5 Subsidy model (front-loaded, declining — V6 §3.3)

A corridor launches below equilibrium turnover and ramps up. SDF subsidises the gap between the **target competitive spread** `S_target` (default: the lesser of the ≤25 bps wholesale floor and the incumbent's spread) and the **break-even spread** the model can offer at the current turnover. Per day:

```
subsidy_per_day(t) = max(0, S_breakeven(T(t)) − S_target) · V(t)
```

Given a turnover ramp `T(t)` from `T_start` to `T_mature` over a horizon (default S-curve over 24 months), report **cumulative subsidy to self-sustaining** for each model. Expected result the engine must show: Model C's subsidy **retires** (curve crosses zero) and Model A's **does not** (V6 §3.3, §8.3).

---

## 4. Parameters: defaults, ranges, and sampling distributions

Every parameter below is a config field with a **point default** (for deterministic runs) and a **Monte Carlo distribution** (for stochastic runs). Distributions are specified as `(family, args)`; `triangular(lo, mode, hi)` and `lognormal`/`normal(truncated)` are the defaults. Values are grounded in V6 and the supporting notes; all are overridable in config.

| Parameter | Symbol | Point default | MC distribution | Source / rationale |
|---|---|---|---|---|
| Daily turnover (sweep axis) | `T` | {0.5, 1, 5, 10, 20} | swept, not sampled | *Advantages* efficiency matrix; V6 §7.5 |
| Flow imbalance | `β` | regime-set (§5) | `triangular` per regime | V6 §7.4 three regimes |
| FX volatility (annual) | `σ` | 0.12 | `triangular(0.05, 0.12, 0.35)` | EM FX vol; MXN≈low, ARS/TRY≈high |
| Hurdle ROC (annual) | `ρ` | 0.15 | `triangular(0.08, 0.15, 0.25)` | MM target ROE |
| Blended yield — CETES (MXN treasury) | `y_CETES` | 0.095 | `triangular(0.06, 0.095, 0.115)` | CETES range cited 6–11% |
| Blended yield — USTRY (USD treasury) | `y_USTRY` | 0.045 | `triangular(0.035, 0.045, 0.055)` | USD T-bill 4–5% |
| Inventory leg weight | `w` | 0.5 | `triangular(0.3, 0.5, 0.7)` | *Advantages* 50/50 example |
| Off-chain venue FX fee | `f_venue` | 0.0025 | `triangular(0.0015, 0.0025, 0.0040)` | V6 §4.2 "~25 bps" |
| On-chain redeem (face-value) fee | `f_redeem` | 0.0003 | `triangular(0.0001, 0.0003, 0.0008)` | "small redemption fee" V6 §4.3 |
| Off-chain rebalance settlement time | `t_settle_offchain` | 1.5 d | `triangular(0.5, 1.5, 3.0)` | "days" V6 §4.2 |
| On-chain settlement time | `t_settle_onchain` / `t_min` | 1.16e-5 d (≈1 s) | fixed | "seconds" V6 §4.3 |
| Ramp operating spread | `S_op` | 0.0005 | `triangular(0.0002, 0.0005, 0.0010)` | residual structural cost |
| Off-chain residual margin | `m_offchain` | 0.0010 | `triangular(0, 0.0010, 0.0030)` | isolated venue pricing power |
| On-chain margin | `m_onchain` | 0.0 | `triangular(0, 0, 0.0002)` | open-book ratchet → ~0 |
| Risk-premium coefficient | `k_risk` | **fit (§7)** | fixed after fit | calibration to regime bands |
| Capital (for $-denominated outputs) | `K` | 10_000_000 | `lognormal` opt. | *Advantages* $10M example |

**Corridor presets** (selectable; set `σ`, `y_local`, regime priors): `MX_MXN` (CETES yield, low σ), `BR_BRL` (TESOURO yield, mid σ), `TR_TRY` (high σ), `IN_INR`, `VN_VND`, `ID_IDR`, `AR_ARS` (very high σ). At minimum ship **MX_MXN** (lead market, full data) and **TR_TRY** (high-vol stress); others inherit MX defaults with σ overrides and are marked `provisional`.

---

## 5. Balance (flow) regimes

The three regimes from V6 §7.4 are first-class scenario dimensions. Each sets the imbalance prior `β` and is expected to reproduce the doc's illustrative spread band (the calibration anchors of §7).

| Regime | What the maker carries | `β` distribution | Doc's illustrative competitive spread |
|---|---|---|---|
| **Balanced, high turnover** | Inventory returns to flat after each pair | `triangular(0.00, 0.05, 0.15)` | **~1–3 bps** |
| **Intermittent imbalance** | A live position held until an offset arrives | `triangular(0.15, 0.35, 0.55)` | **~10–25 bps** |
| **Persistent imbalance** | One leg accumulates; must rebalance periodically | `triangular(0.60, 0.80, 1.00)` | **~30–60 bps** |

Remittance/savings corridors are dominated by **persistent imbalance** (almost everyone selling local currency for USD); the simulation must show that this is precisely the regime where Model A's off-chain rebalancing is most punishing and Model C's structure helps most.

---

## 6. Monte Carlo design

### 6.1 Trial

One **trial** = one sampled parameter vector → evaluate `spread_components` for **all three models (A, B, C-xborder, C-same-denom)** under the **same** sampled `σ, β, ρ, yields, fees, t_settle, w` so differences are attributable only to structure. Record every additive spread term (bps), `C_total` (bps), MM ROC, and subsidy-to-self-sustaining.

### 6.2 Run

One **run** = a fixed `(corridor, turnover T, balance regime)` cell × `N` trials (default `N = 10_000`). Turnover `T` is held at the swept value within a run (the sweep is the deterministic axis); all other params are sampled.

### 6.3 Sweep / scenario grid

Full grid = `corridors × T_values × regimes × models`. Defaults:

- corridors: `[MX_MXN, TR_TRY]` (extensible)
- `T_values`: `[0.5, 1, 5, 10, 20]`
- regimes: `[balanced, intermittent, persistent]`
- models: `[A, B, C_xborder, C_same_denom]`

→ 2 × 5 × 3 = 30 cells × 4 models × 10k trials. Must complete in **< 60 s** on a laptop (vectorise with NumPy; no Python-level per-trial loop).

### 6.4 Reproducibility

- Master `seed` in config; derive per-cell seeds deterministically (`numpy.random.default_rng(seed_sequence)`).
- Persist the exact resolved config (post-defaults) alongside results.
- All randomness flows through one RNG factory; no global `np.random`.

### 6.5 Aggregation / statistics

Per cell × model report: mean, std, **p5 / p25 / p50 / p75 / p95**, min, max of `C_total` and `S_fx` (bps); `P(C_total ≤ S_target)`; `P(Model C ≤ Model B)` and `P(Model C ≤ Model A)` (paired, same trials); mean MM ROC; mean/median subsidy-to-self-sustaining. Also compute the **paired spread advantage** distributions `A−C` and `B−C` (use common random numbers — same trial inputs — so the difference distribution is meaningful).

---

## 7. Calibration & acceptance tests (must pass)

These are automated `pytest` tests. They are the contract; if the analytic form drifts from them, fit the free coefficient (`k_risk`) rather than changing the anchors.

### 7.1 Yield-subsidy turnover table (exact)

With `y_blend = 5%`, `ΔS = 2·(0.05/365)/T` must reproduce *Advantages*' table to within 0.01 bps:

| `T` | Expected ΔS |
|---|---|
| 0.5× | 5.48 bps |
| 1.0× | 2.74 bps |
| 5.0× | 0.55 bps |
| 10.0× | 0.27 bps |
| 20.0× | 0.14 bps |

(Verified during spec authoring: 5.479, 2.740, 0.548, 0.274, 0.137.)

### 7.2 Flow-regime bands (calibrate `k_risk`)

Fit `k_risk` (single scalar) so that the **median** competitive on-chain spread (`m≈0`, `y_blend=0`, `f_redeem` only) at the corridor's representative turnover lands inside each band: balanced ∈ [1,3] bps, intermittent ∈ [10,25] bps, persistent ∈ [30,60] bps. Test asserts medians fall in-band for the `MX_MXN` preset. Document the fitted `k_risk` in the report.

### 7.3 Structural ordering (qualitative invariants — must always hold)

For any trial with `β > 0`, `σ > 0`, `y_blend > 0`:

- `C_total(A) ≥ C_total(B) ≥ C_total(C)` at equal turnover (off-chain ≥ stable/stable ≥ reserved).
- `C_total(B) − C_total(C) ≈ ΔS` (the gap between B and C is the yield subsidy, ± rounding).
- **Returns to scale:** as `T → ∞`, `S_fx(B)` and `S_fx(C)` → their floors while `S_fx(A)` retains the `f_venue·β` term (does **not** vanish). Assert `dS_fx/dT (A) ≈ 0` for the venue term and that A's floor `> 0` by at least `f_venue·β_persistent`.
- **Parity ceiling for B:** `S_fx(B)` never falls below `S_fx(C)`; B asymptotes to the cash-FX (zero-yield) floor.

### 7.4 ≤25 bps wholesale target (V6 §4.4)

For `MX_MXN`, persistent regime, at `T ≥ 5×`, assert `P(C_total(C) ≤ 25 bps) ≥ 0.5` while `P(C_total(A) ≤ 25 bps)` is materially lower. (This demonstrates the doc's headline claim probabilistically, not as a point estimate.)

### 7.5 Conservation

`C_total` is invariant to whether the Model A venue fee is booked under `S_ramp` or `S_fx` (§3.3 note). Assert equality across both bookkeeping paths.

---

## 8. Outputs & report

The engine writes to `--out <dir>`:

**Data artifacts**
- `results_trials.parquet` — every trial (cell, model, all spread terms, totals, ROC, subsidy).
- `results_summary.csv` — per cell × model aggregate stats (§6.5).
- `config_resolved.yaml` — the exact config used.
- `calibration.json` — fitted `k_risk`, anchor check results.

**HTML report** (`report.html`, self-contained, Jinja2 + embedded PNG/SVG) containing:

1. **Executive summary** — one paragraph + a headline table: median `C_total` (bps) by model, at the corridor's mature turnover, per regime; and `P(C ≤ 25 bps)`.
2. **Distribution plots** — for each (corridor, regime) at mature `T`: overlaid histograms / violin plots of `C_total` for A vs B vs C-xborder vs C-same-denom. Show p5–p95 bands.
3. **Turnover curves (returns to scale)** — `S_fx` (median, with p25–p75 fan) vs `T` for each model, on one axis per regime. This is the money chart: A flat/high, B declining to the cash-FX-parity line, C declining below it. Annotate the cash-FX-parity asymptote and the ≤25 bps target line.
4. **Yield-subsidy panel** — reproduce the `ΔS` vs turnover table/curve; mark the 0.20–0.50 bps "structural edge" zone and the 10–20% relative-discount note.
5. **Spread waterfall** — for a representative cell, stacked decomposition of `C_total` into its additive terms (operating, margin, cost-of-capital carry, yield credit, risk premium, rebalance) for each model side by side.
6. **Paired advantage** — distributions of `A − C` and `B − C` (bps saved per unit flow) with means and P(>0).
7. **Subsidy burn-down** — cumulative SDF subsidy vs months for each model over the turnover S-curve; show C retiring to zero and A not.
8. **Sensitivity / tornado** — one-at-a-time effect of each parameter on median `C_total(C)` (rank by swing), highlighting `σ`, `β`, `y_blend`, `T`.
9. **Assumptions & caveats** — the VaR open question (§Appendix), interest-rate risk on treasuries, bond-market hours/liquidity, friction differential (`f_redeem` must stay below the yield edge). State plainly that the yield edge is "an edge over a risk, not a riskless gift" (V6 §7.6).

All charts must render from `results_trials.parquet` alone (report is regenerable without re-simulating). Use a consistent colour per model across every chart.

---

## 9. Architecture & file layout

```
corridor-sim/
├── README.md                 # quickstart, the three models, how to read the report
├── pyproject.toml
├── config/
│   ├── default.yaml          # all params, distributions, scenario grid (the §4/§5/§6 content)
│   └── corridors/            # MX_MXN.yaml, TR_TRY.yaml, ...
├── src/corridor_sim/
│   ├── __init__.py
│   ├── config.py             # Pydantic models; load/merge/validate; resolve distributions
│   ├── distributions.py      # triangular/normal/lognormal samplers over an injected RNG
│   ├── economics.py          # spread_components(model, p) -> dict[str,float]; the §3 equations
│   ├── models.py             # Model A/B/C definitions: which terms/inputs are active
│   ├── montecarlo.py         # vectorised trial/run/sweep engine; RNG factory; aggregation
│   ├── calibrate.py          # fit k_risk to §7.2 bands; emit calibration.json
│   ├── subsidy.py            # §3.5 turnover S-curve + subsidy burn-down
│   ├── report.py             # Jinja2 HTML + Matplotlib figures
│   └── cli.py                # `corridor-sim run|calibrate|report` (argparse/typer)
├── templates/report.html.j2
├── tests/
│   ├── test_yield_subsidy.py     # §7.1 exact table
│   ├── test_regime_bands.py      # §7.2 calibration
│   ├── test_invariants.py        # §7.3 ordering + returns-to-scale + parity ceiling
│   ├── test_target_25bps.py      # §7.4
│   ├── test_conservation.py      # §7.5
│   └── test_reproducibility.py   # same seed → identical results
└── examples/
    └── run_all.sh            # end-to-end: calibrate → run sweep → build report
```

**CLI contract**

```
corridor-sim calibrate --config config/default.yaml --out runs/latest
corridor-sim run        --config config/default.yaml --out runs/latest --trials 10000 --seed 42
corridor-sim report     --in runs/latest --out runs/latest/report.html
# convenience:
corridor-sim all        --config config/default.yaml --out runs/latest
```

**`economics.py` core (pseudocode — implement vectorised over NumPy arrays):**

```python
def spread_components(model: ModelSpec, p: Params) -> dict:
    rho_d   = p.rho / 365
    y_d     = p.y_blend(model) / 365            # 0 for A and B
    sig_d   = p.sigma / np.sqrt(252)
    t_min   = p.t_settle_onchain if model.onchain else p.t_settle_offchain_hold
    t_hold  = t_min + p.beta / p.T
    carry   = 2 * (rho_d - y_d) / p.T
    risk    = p.k_risk * sig_d * np.sqrt(t_hold)
    f_reb   = p.f_redeem if model.onchain else p.f_venue
    t_set   = p.t_settle_onchain if model.onchain else p.t_settle_offchain
    rebal   = f_reb * p.beta + p.k_risk * sig_d * np.sqrt(t_set)
    margin  = p.m_onchain if model.onchain else p.m_offchain
    s_fx    = margin + carry + risk + rebal
    s_ramp  = p.s_op - (2*y_d/p.T if model.ramp_yield else 0.0)
    return dict(operating=p.s_op, ramp_yield_credit=(2*y_d/p.T if model.ramp_yield else 0),
                margin=margin, carry=carry, risk=risk, rebalance=rebal,
                s_fx=s_fx, s_ramp=s_ramp, c_total=s_ramp + s_fx)   # all ×1e4 for bps on output
```

(`ModelSpec` flags: `onchain: bool`, `ramp_yield: bool`, and a `yield_pair` selecting how `y_blend` is computed — `none` for A/B, `yield_yield` for C-xborder, `yield_stable` for C-same-denom.)

---

## 10. Tech stack & dependencies

- **Python** 3.11+
- **numpy**, **pandas**, **scipy** (calibration root-find/optimise), **pyarrow** (Parquet)
- **pydantic** v2 (config validation), **pyyaml**
- **matplotlib** (charts; no seaborn requirement), **jinja2** (report)
- **typer** or argparse (CLI)
- **pytest** (+ **pytest-cov**)
- Optional: **rich** for CLI progress

No network access required at runtime. Deterministic given seed. Target: full sweep + report < 60 s on a modern laptop.

---

## 11. Definition of done

1. `pip install -e .` then `corridor-sim all --config config/default.yaml --out runs/demo` produces `report.html` plus all data artifacts.
2. `pytest` passes, including all §7 acceptance tests; coverage on `economics.py`, `models.py`, `montecarlo.py`, `subsidy.py` ≥ 90%.
3. The report visibly demonstrates, for the persistent-imbalance MX_MXN corridor: Model C beating Model B by ≈ΔS, both beating Model A, A showing no returns to scale, C clearing ≤25 bps at scale with majority probability, and the subsidy burn-down retiring for C but not A.
4. `README.md` explains the three models and how to read every chart in ≤2 pages.
5. Re-running with the same seed reproduces identical numbers.

---

## Appendix A — Equation derivations & provenance

- **Yield subsidy** `ΔS = 2·DailyYield/DailyTurnover`. From revenue equality (V6 §7.5, *Advantages* §1): `Fiat: V·S_f/2 = ρ·K`; `Debt: V·S_d/2 + K·y_d = ρ·K`. Subtract: `S_f − S_d = 2·K·y_d/V = 2·y_d/T`. With 5% yield this gives the §7.1 table exactly.
- **Volatility-over-holding-window** `RiskPremium ∝ σ·√(t_hold)`. V6 §4.2/§7.4: a position held over duration `t` faces an expected move on the order of `σ·√t`; duration is set by turnover and imbalance. `k_risk` converts a 1-σ move into the *charged* premium (risk aversion / confidence level), fit to the doc's bands (a raw 1-σ daily move at 12% annual vol is ~76 bps, far above the intermittent band, so `k_risk < 1` is expected and required).
- **Rebalancing / no-returns-to-scale for off-chain FX.** V6 §4.2, §8.3 case 1: the ~25 bps venue fee is paid on essentially every unit of net flow regardless of size; bigger volume is simply more to settle off-chain, so the term is volume-invariant as a spread.
- **Cash-FX parity ceiling for non-yield on-chain FX.** V6 §8.3 case 2: a maker holding idle stablecoin inventory carries the same cost of capital as a cash desk, so moving FX on-chain earns returns to scale only down to parity, not below.
- **Hub-and-spoke / one yield leg per trade.** V6 §7.1, *Yield Subsidy*: using a shared yield-bearing numéraire puts the yield leg on one side of every trade → ~50% of inventory always earning the subsidy (basis for `w` default and C-same-denom).

## Appendix B — Open questions / risks the model must surface (not hide)

These come from V6 §7.6 and the *Yield Subsidy* caveats. The report's assumptions section must state them; where parameterisable, expose them.

1. **VaR vs the yield edge.** Does the sub-bps yield advantage cover the value-at-risk of holding securities instead of cash? Expose an optional `var_charge` term that subtracts from the yield credit; default 0 but documented as the key honest unknown.
2. **Interest-rate risk** on short-duration treasuries (low, not zero) — sensitivity to a rate-cycle shock (`y` falling sharply); include in the tornado.
3. **Bond-market liquidity / hours** — can seize in a crisis; reflected via a higher `f_redeem` and longer `t_settle_onchain` stress preset.
4. **Friction differential** — `f_redeem` (securities settlement) must remain below the yield edge for the model to hold; flag trials where `f_redeem·β > ΔS`.
5. **Imbalance is the adversary, not turnover alone** — show that even Model C's advantage compresses at very high turnover (ΔS shrinks as 1/T), so the win is largest exactly in the thin, imbalanced, early-corridor conditions SDF subsidises.

## Appendix C — Source map

| Claim in this spec | Primary source |
|---|---|
| Two-spread user cost; ≤25 bps wholesale target | V6 §4.4 |
| Ramp cost stack; rebalancing as the two dominant costs | V6 §4.1–4.2 |
| Reserved ramp line-by-line (incumbent vs reserved) | V6 §4.3 |
| Three flow regimes + illustrative bps bands | V6 §7.4 (table) |
| Yield-subsidy formula + turnover table | V6 §7.5; *Advantages to Making Markets with Yield Bearing Assets*; *Yield Subsidy* note |
| Returns-to-scale, three cases | V6 §8.3 |
| Denomination matching / reserved model balance sheet | *Reserved Ramps Model* note; V6 §4.3 |
| Duration fix (days → seconds) | *On/Off Ramp Economics* note; V6 §4.3 |
| Subsidy front-loaded & declining | V6 §3.3 |
| Corridors & yields (CETES 6–11%, USTRY 4–5%) | V6 §3.5, §6; *Advantages* |

---

*End of specification.*
