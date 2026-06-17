"""Command-line interface: ``corridor-sim calibrate | run | report | all`` (spec §9)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console

import pandas as pd

from .calibrate import calibrate_and_apply
from .config import Config, load_config
from .montecarlo import run_sweep
from .report import build_report
from .subsidy import run_subsidy

app = typer.Typer(add_completion=False, help="Corridor Economics Simulation Engine")
console = Console()


def _write_config(cfg: Config, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.yaml").write_text(cfg.to_yaml(), encoding="utf-8")


def _write_results(result, out: Path) -> None:
    """Persist trials/summary, downcasting trial floats to float32 to keep Parquet lean."""

    out.mkdir(parents=True, exist_ok=True)
    trials = result.trials.copy()
    float_cols = trials.select_dtypes("float64").columns
    trials[float_cols] = trials[float_cols].astype("float32")
    trials.to_parquet(out / "results_trials.parquet", index=False)
    result.summary.to_csv(out / "results_summary.csv", index=False)


def _write_subsidy(cfg: Config, out: Path) -> None:
    """Write the subsidy burn-down curves + summary across all corridors (§3.5)."""

    curves, summaries = [], []
    for corridor in cfg.grid.corridors:
        res = run_subsidy(cfg, corridor)
        c = res.curve.copy()
        c["target_bps"] = res.target_bps
        curves.append(c)
        summaries.append(res.summary)
    pd.concat(curves, ignore_index=True).to_csv(out / "results_subsidy.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(out / "results_subsidy_summary.csv", index=False)


def _do_calibrate(cfg: Config, out: Path):
    result = calibrate_and_apply(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    console.print(f"[bold]Calibrated[/] k_risk = {result.k_risk:.4f}")
    for r, med in result.regime_medians_bps.items():
        lo, hi = result.regime_bands_bps[r]
        ok = "[green]✓[/]" if result.regime_in_band[r] else "[red]✗[/]"
        console.print(f"  {r:14s} median {med:6.2f} bps  band [{lo:g},{hi:g}]  {ok}")
    return result


@app.command()
def calibrate(
    config: Path = typer.Option(None, "--config", help="Config YAML (defaults to built-in)"),
    out: Path = typer.Option("runs/latest", "--out", help="Output directory"),
):
    """Fit k_risk to the regime bands (§7.2) and write calibration.json."""

    cfg = load_config(config)
    _do_calibrate(cfg, out)
    _write_config(cfg, out)


@app.command()
def run(
    config: Path = typer.Option(None, "--config", help="Config YAML (defaults to built-in)"),
    out: Path = typer.Option("runs/latest", "--out", help="Output directory"),
    trials: int = typer.Option(None, "--trials", help="Trials per cell (default from config)"),
    seed: int = typer.Option(None, "--seed", help="Master RNG seed"),
    calibrate_first: bool = typer.Option(True, "--calibrate/--no-calibrate"),
):
    """Run the full Monte Carlo sweep and write data artifacts (§6, §8)."""

    cfg = load_config(config)
    if trials is not None:
        cfg.trials = trials
    if seed is not None:
        cfg.seed = seed
    out = Path(out)

    if calibrate_first:
        _do_calibrate(cfg, out)

    t0 = time.time()
    n_cells = len(cfg.grid.corridors) * len(cfg.grid.regimes) * len(cfg.grid.turnovers)
    console.print(f"Running sweep: {n_cells} cells × {cfg.trials} trials × {len(cfg.grid.models)} models ...")
    result = run_sweep(cfg, n=cfg.trials)
    dt = time.time() - t0

    _write_results(result, out)
    _write_subsidy(cfg, out)
    _write_config(cfg, out)
    console.print(f"[green]Done[/] in {dt:.1f}s → {out} ({len(result.trials):,} trial rows)")


@app.command()
def report(
    in_dir: Path = typer.Option("runs/latest", "--in", help="Run directory with artifacts"),
    out: Path = typer.Option(None, "--out", help="Output HTML path"),
):
    """Build the self-contained HTML report from existing artifacts (§8)."""

    path = build_report(in_dir, out)
    console.print(f"[green]Report[/] → {path}")


@app.command()
def all(
    config: Path = typer.Option(None, "--config", help="Config YAML (defaults to built-in)"),
    out: Path = typer.Option("runs/latest", "--out", help="Output directory"),
    trials: int = typer.Option(None, "--trials"),
    seed: int = typer.Option(None, "--seed"),
):
    """Calibrate → run sweep → build report (end-to-end)."""

    cfg = load_config(config)
    if trials is not None:
        cfg.trials = trials
    if seed is not None:
        cfg.seed = seed
    out = Path(out)

    _do_calibrate(cfg, out)
    t0 = time.time()
    n_cells = len(cfg.grid.corridors) * len(cfg.grid.regimes) * len(cfg.grid.turnovers)
    console.print(f"Running sweep: {n_cells} cells × {cfg.trials} trials × {len(cfg.grid.models)} models ...")
    result = run_sweep(cfg, n=cfg.trials)
    _write_results(result, out)
    _write_subsidy(cfg, out)
    _write_config(cfg, out)
    console.print(f"sweep done in {time.time() - t0:.1f}s")
    path = build_report(out)
    console.print(f"[green]All done[/] → {path}")


if __name__ == "__main__":
    app()
