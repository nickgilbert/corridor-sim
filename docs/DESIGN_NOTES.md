# Design notes & decision log

A standalone narrative of *why* this model is shaped the way it is — the thesis it argues,
the decisions made, the insights that emerged, and the assumptions that are load-bearing.
Written for handoff into a broader strategy document and for the next person/agent picking
this up. For the *what* (equations, parameters, file layout) see `SPEC.md`, `CLAUDE.md`, and
the module docstrings; for the *latest state* see the committed `docs/report.html`.

---

## 1. The thesis this model argues

Two claims, in order of how the report makes them:

**Point 1 — yield-bearing rails are universally cheaper.** In every corridor and regime, the
reserved-ramp model (C, tokenized treasuries) is **≪ the incumbent (A)** and **≤ stable-to-stable
(B)**. Delivering a *yield-bearing dollar asset* (local→USTRY) beats delivering USDC (local→USDC)
by the yield subsidy ΔS, with no extra hop.

**Point 2 — it compounds across corridors; stable-to-stable does not.** Every corridor funnels
local value through **tokenized US Treasuries (USTRY) as a universal reserve hub**. Because USTRY
is common to all corridors, the shared `USTRY↔USDC` pool (and the USTRY reserve generally)
aggregates the whole network's volume: more corridors → higher hub turnover → tighter spread for
*everyone*. A stable-to-stable competitor cannot do this **once stablecoins fragment** — its
liquidity splits across many shallow pools that never aggregate.

The strategic punchline: **yield gets you competitive while small; the unified-reserve hub makes
you cheapest at scale.**

---

## 2. Why the single-corridor view undersells the case (the core reframe)

The original report was single-corridor, and on that view C beats B only by ΔS — a fraction of a
basis point at mature turnover. That is *real but unconvincing*. The reason is structural, not
cosmetic:

- **Yield and returns-to-scale are the same `1/T` lever.** Yield enters the cost stack *only*
  through the cost-of-capital carry, `carry = 2·(ρ−y)/T`, and the yield edge is `ΔS = 2·y/T`.
  Both vanish as turnover `T` rises. So a single mature corridor (high T) has almost no yield edge
  left — and almost no carry either; its cost floors at friction.
- Therefore the yield advantage is a **low-turnover / early-stage phenomenon** (thin corridors,
  spokes), while the compounding advantage is a **high-turnover phenomenon** (the mature shared
  hub). The single-corridor chart shows neither at its best. The network view is where the case
  is actually made.

This reframe is the reason Act II (the network layer) exists.

---

## 3. Decision log (chronological)

Each entry: what we decided, and why. Commit hashes in parentheses.

1. **Reproducibility fix** (`4f113c1`). `cell_rng` seeded each cell from Python's built-in
   `hash()` of the corridor/regime strings. Python *salts* string hashes per process, so every
   separate `corridor-sim` invocation produced different `k_risk` and different sweep numbers —
   the README's "same seed reproduces" was false across runs (the in-process tests couldn't catch
   it). Replaced with a stable `blake2b` digest. *Discovered while verifying the mint change was
   inert — the "drift" wasn't the change, it was this bug.*

2. **Mint is free; only redeem is fee-bearing** (`b79a98a`). Economic correction from the owner:
   minting a stable/yield asset from same-denomination cash is free at face value — the issuer
   monetises the float on minted assets and charges only on redemption. Added `f_mint` (default 0)
   and split the rebalance into redeem + mint legs; the friction-breach diagnostic now compares the
   **redeem leg only** (`f_redeem·β`) against ΔS. *Finding: the model already honored this
   implicitly (a single fee, no mint charge); the change makes it explicit, auditable, and
   stress-testable. Default numbers unchanged.* Off-chain venue fees (Model A) keep both directions
   — the mint exemption is an on-chain *issuer* property.

3. **Early-stage subsidy ramp** (`f97b7ef`). The burn-down ramp was 0.5→20×/day over 24 months —
   too high/fast for a brand-new corridor. Retuned to **0.01→1× over 12 months** to show the
   standup phase where nobody has reached break-even and the subsidy is still funding the gap.
   Finding: yield *offsets* the subsidy linearly (~$217K bought back per percentage point of
   blended yield, ~38% of the no-yield bill at the modelled 7% cross-border yield).

4. **Act II — hub-and-spoke network model** (`e4f138e`). Built the network layer (`network.py`):
   multi-hop routing through a USTRY star, hub turnover aggregation, β-netting. See §4 for the two
   structural decisions (Q1, Q2) that gate it.

5. **Growing fragmentation + network subsidy** (`478d4df`). Added fragmentation that grows with the
   network (a stablecoin per market) and the time-based network subsidy burn-down. See §5.

6. **Report glossary + readable labels** (`bc82e8d`). Glossary at the top; tornado relabeled to
   plain names.

---

## 4. The two structural decisions behind the network model (Q1 & Q2)

The compounding thesis hinges on two questions the model owner answered. Both are **assumptions**,
exposed as knobs in `NetworkConfig`.

**Q1 — capital efficiency at the hub.** Compounding requires hub turnover `T_hub = ΣV / K_hub` to
*rise* as corridors are added, which requires hub capital `K_hub` to grow **sub-linearly** with
volume. Decision: **one shared book serves everyone** (high but not perfect efficiency) —
rebalancing block-to-block, so capital depth bounds per-block transaction size but the book recycles
to serve the next customer. Model: `T_hub = min(ΣV / K_hub, ceiling)`, `K_hub` ≈ one corridor's
book. *Consequence (see §2): this drives `T_hub` high, which kills carry — and the yield edge with
it — so at the mature hub the cost floors at friction `f_redeem·β_hub`, making β_hub (Q2) the
decisive variable.*

**Q2 — does hub flow net, or all push the same way?** A shared book can only recycle if there's
offsetting flow — so high `T_hub` *requires* low `β_hub`. Decision: **flows net as the network
grows**, for two reasons: (a) diversification across corridors (netting law `β_hub = β_floor +
(β₀−β_floor)/√N_eff`), and (b) a *growing share of direct cross-border flow that bypasses
stablecoins entirely* — the Samsung archetype (sell TVs in pesos, repatriate to KRW: pesos→CETES→
USTRY→KTB→won, never touching USDC). These corporate flows are larger, more two-sided, and exactly
the flows a stablecoin rail serves worst. Anchors chosen: **β_floor = 0.2** (assumption, adjustable),
retail/corporate mix **90/10 → 50/50** over time.

---

## 5. The load-bearing finding: Point 2 requires reserve-asset fungibility

This is the most important insight and the easiest to miss.

**A V1 network model with a *unified* USDC hub shows C and B both compounding** — B's USDC hub
aggregates just like C's USTRY hub, and C wins only by the (scale-shrinking) yield. On that model,
"stable-to-stable can't compound" is **false**.

What actually breaks B's hub is **stablecoin fragmentation**: tokenized Treasuries are one fungible
reserve (every corridor's USTRY is the same deep pool), but stablecoins proliferate (USDT, PYUSD,
local coins…), so B's "USDC hub" splits across many shallow pools that never aggregate. Modelled as:
B's effective hub turnover = `T_hub / S`, and netting degraded to `N_eff = N/S`, for `S` stablecoins.

Results (cross-border MXN→KRW, all-in bps, as N grows 1→4):
- **C (unified USTRY):** 21.6 → 13.3 (compounds to a floor).
- **B, 1 unified stablecoin (steelman):** 22.3 → 13.5 (nearly matches C — loses only by yield).
- **B, fragments 10 ways:** 60.4 → 33.9 (stays ~2.5× above C).
- **B, a stablecoin per market (S=N):** flatlines ~24 (never aggregates; the gap to C *widens*).

**So Point 2 is a statement about reserve-asset fungibility, and it lives entirely in the
fragmentation assumption.** This is stated plainly in the report §9 caption and is the reason
fragmentation calibration is the top deferred item.

Network subsidy corollary (maturity-reaching ramp, staggered launches): **C's network subsidy
retires** (~$5.4M, plateaus as the hub aggregates); a fragmenting stable-to-stable network never
retires (~$39M and climbing); the incumbent runs away (~$2.1B).

---

## 6. Other findings worth carrying into the strategy doc

- **β (flow imbalance) is the adversary, not turnover.** Turnover helps every model; imbalance is
  what erodes C's edge (it lengthens the holding window *and* scales the redeem fee). Netting it
  down at the hub is the core compounding mechanism, alongside turnover aggregation.
- **B and C are otherwise identical on-chain.** The only modelled difference between stable/stable
  and reserved-ramp is the yield in the carry term — `margin`, `risk premium`, and `rebalance` are
  byte-identical. There is **no explicit DEX price-impact/liquidity term** (deferred, TODO #1); the
  model currently assumes equal pool depth for B and C.
- **The `USTRY↔USDC` hub is same-currency** (USD vs USD), so it carries *no FX risk* — its cost is
  carry (→0 at scale) + friction only. This is why the hub is cheap.
- **Spread ordering yield-yield < yield-stable < stable-stable** holds and is tested; it's driven
  by total blended yield across the pair's legs, applied to the *market-maker* spread `S_fx` (not
  the ramp).

---

## 7. Assumptions register (all adjustable)

| Assumption | Value | Where | Status |
|---|---|---|---|
| Yield subsidy is the only B-vs-C difference | — | `economics.py` | core model identity (tested) |
| Mint fee | 0 (face-value mint) | `f_mint` | owner decision |
| Early-stage ramp | 0.01→1× / 12mo | `SubsidyConfig` | for the standup narrative |
| Hub capital efficiency | one shared book, T_hub=min(ΣV/K_hub, ceiling) | `NetworkConfig` (Q1) | owner decision |
| Hub netting floor β_floor | **0.2** | `NetworkConfig` | **assumption, flagged adjustable** |
| Retail/corporate mix | 90/10 → 50/50 | (narrative; drives netting) | owner estimate |
| Stablecoin fragmentation | S = 3, 10, and grows with N | `NetworkConfig` | **load-bearing for Point 2; uncalibrated** |
| Equal pool depth for B and C | yes (no price-impact term) | — | deferred (TODO #1) |
| DEX has no explicit LP/slippage fee | yes | `economics.py` | structural simplification |

---

## 8. Open questions / deferred work

From `docs/TODO.md` plus design-level items:

1. **Spoke price-impact term** (TODO #1) — local-bond↔USTRY pools are thin and per-corridor; the
   model penalizes thinness only indirectly. An explicit depth/price-impact term would test whether
   thin spokes erode C's edge.
2. **Calibrate the fragmentation rate** (TODO #2) — Point 2's magnitude depends on how many
   stablecoins B realistically fragments across, and whether that grows with the network. This is
   the single most important number to ground against real stablecoin market structure.
3. **Validate β_floor and the mix trajectory** against real corridor flow data (currently owner
   estimates).
4. **Local→USD delivery: USTRY vs USDC.** C beats B with no extra hop only if the recipient holds
   USTRY (yield dollars). If USDC is mandatory, C adds the hub hop (a small penalty that shrinks
   with N). Worth pinning which is the product.
5. **VaR vs the yield edge** (from `SPEC.md` §Appendix) — `var_charge` is off by default; the
   sub-bps yield edge must cover the value-at-risk of holding securities vs cash.

---

## 9. Map: thesis → report section → code

| Thesis element | Report section | Code |
|---|---|---|
| Cost stack, B/C identity | §1, §5 waterfall | `economics.py` |
| Returns to scale (per corridor) | §3 | `montecarlo.py` |
| Yield subsidy ΔS | §4 | `economics.yield_subsidy_bps` |
| Subsidy + yield offset (early stage) | §7, §8 | `subsidy.py`, `report.subsidy_yield_sensitivity` |
| Point 1 — universal dominance | §9 (bars) | `network.corridor_costs` |
| Point 2 — compounding + fragmentation | §9 (hero, β, subsidy) | `network.py` |
| What drives C's cost | §10 tornado | `report.chart_tornado` |
| Honest caveats | §11 | `report.py`, `docs/TODO.md` |

Reproduce everything: `corridor-sim all --config config/default.yaml --out runs/<name>`
(deterministic from `seed`; ~2s sweep). Acceptance invariants live in `tests/`.
