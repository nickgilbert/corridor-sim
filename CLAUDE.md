# corridor-sim

Monte Carlo simulation engine quantifying the all-in cross-currency spread on a payment
corridor, demonstrating why the **denomination-matched reserved ramp** (Model C) structurally
beats the alternative ramp/FX architectures (SDF V6 *Distribution as Our Growth Engine* strategy).

**This is a standalone project.** It is unrelated to the `~/Claude/Org Memory` wiki vault — do
not read, reference, or write to that vault when working here.

## Environment

- Python ≥3.11. A virtualenv lives at `.venv/` — activate with `source .venv/bin/activate`.
- Install (editable, with dev deps): `pip install -e ".[dev]"`
- Console entry point: `corridor-sim` (defined in `pyproject.toml`).

## Common commands

```bash
pytest                                                          # §7 acceptance suite
corridor-sim all --config config/default.yaml --out runs/demo   # calibrate → sweep → report
corridor-sim calibrate --config config/default.yaml --out runs/latest   # fit k_risk (§7.2)
corridor-sim run  --config config/default.yaml --out runs/latest --trials 10000 --seed 42
corridor-sim report --in runs/latest --out runs/latest/report.html
```

Re-running `run` with the same `--seed` reproduces identical numbers (enforced by a test).
The full 30-cell × 10k-trial × 4-model sweep runs in < 2 s.

## Architecture

The three models each price one unit of cross-currency flow: `C_total = S_ramp + S_fx` (bps).
Differences in `C_total` are structural (ramp balance sheet + FX venue). Model **C** is the
yield-bearing reserved ramp; its edge over Model B is the **yield subsidy** `ΔS = 2·y_daily/T`.

| Module (`src/corridor_sim/`) | Responsibility |
|---|---|
| `config.py` | Load/resolve `config/*.yaml`; Pydantic models for inputs |
| `distributions.py` | Sampling of uncertain inputs (σ, β, yields, …) |
| `economics.py` | Closed-form spread equations evaluated per trial |
| `models.py` | `ModelSpec` for A / B / C; ramp balance-sheet differences |
| `montecarlo.py` | The trial sweep (cells × trials × models) |
| `calibrate.py` | Fits the single free coefficient `k_risk` to the §7.2 regime bands |
| `subsidy.py` | 24-month subsidy burn-down / retirement curves |
| `report.py` | Renders `templates/report.html.j2` → self-contained HTML |
| `cli.py` | Typer app: `calibrate`, `run`, `report`, `all` |

`docs/SPEC.md` is the authoritative build spec; section refs (§3.3, §7.2, §7.3, App. B) in code
and tests point into it. The acceptance contract lives in `tests/` — keep those invariants green.

## Docs — read these first for context

- `docs/DESIGN_NOTES.md` — **the design narrative & decision log**: the thesis (yield is universally
  cheaper; it compounds via a unified treasury hub), why each decision was made, the key findings
  (yield and scale are the same 1/T lever; Point 2 rests on reserve-asset fungibility vs stablecoin
  fragmentation), the assumptions register, and open questions. Start here for the "why."
- `docs/SPEC.md` — the build spec (the "what": equations, parameters, acceptance criteria).
- `docs/TODO.md` — deferred modeling work (spoke price-impact; fragmentation calibration).
- `docs/report.html` — committed snapshot of the latest rendered report (regenerable).

## Conventions

- Calibration fits `k_risk` to land median spreads inside the doc's regime bands (balanced
  [1,3], intermittent [10,25], persistent [30,60] bps). If the analytic form drifts, refit
  `k_risk` — the bands are the fixed acceptance anchors and never move.
- The ramp's own idle-inventory yield credit (§3.3, `ModelSpec.ramp_yield`) is implemented but
  **off by default** so the headline B−C gap equals exactly ΔS (the §7.3 invariant).
- Outputs are written under `--out <dir>` (parquet trials, summary CSVs, resolved config,
  `calibration.json`, `report.html`). `runs/` is the conventional output location.
