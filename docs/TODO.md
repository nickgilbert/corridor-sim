# Modeling TODO

Deferred items, captured during the Act II (network / hub-and-spoke) design.

## 1. Spoke-pool price-impact / liquidity term  (deferred)
The local-bond ↔ USTRY spoke pools are thin and per-corridor; they do not aggregate the
way the shared hub does. The current model penalizes thin/imbalanced pools only indirectly
(risk premium `σ·√t_hold` + friction `f_redeem·β`). Add an explicit per-pool price-impact /
depth term keyed to pool liquidity, so the spoke cost isn't understated.
Status: **not modeled in V1.** Build after the network compounding shape is confirmed.

## 2. Stablecoin liquidity fragmentation  (deferred — load-bearing for Point 2)
The steelman competitor B routes cross-border through **USDC as a hub**. In V1 that hub
aggregates flow the same way C's USTRY hub does, so V1 shows C beating B only by the (scale-
shrinking) yield edge. The *structural* "B cannot compound" claim depends on a dynamic V1
omits: **as stablecoins diversify (USDT, PYUSD, local stables, …), B's USDC hub fragments
into many shallow pools**, while tokenized Treasuries remain one fungible, deep reserve.
Model B's hub depth as **declining/fragmenting** with stablecoin proliferation; this is what
breaks B's aggregation while C's USTRY hub stays unified.
Status: **not modeled in V1.** Likely required to make Point 2 fully.
