"""Smoke tests: distribution families and end-to-end report generation."""

import numpy as np
import pytest

from corridor_sim import distributions as dist
from corridor_sim.calibrate import calibrate_and_apply
from corridor_sim.config import DistSpec, build_default_config
from corridor_sim.montecarlo import run_sweep
from corridor_sim.report import build_report


def test_distribution_families():
    rng = np.random.default_rng(0)
    fixed = dist.sample(DistSpec(family="fixed", args={"value": 0.5}), rng, 100)
    assert np.all(fixed == 0.5)

    tri = dist.sample(DistSpec(family="triangular", args={"lo": 0, "mode": 1, "hi": 2}), rng, 1000)
    assert tri.min() >= 0 and tri.max() <= 2

    nrm = dist.sample(DistSpec(family="normal", args={"mean": 0, "sd": 1, "lo": -1, "hi": 1}), rng, 1000)
    assert nrm.min() >= -1 and nrm.max() <= 1

    ln = dist.sample(DistSpec(family="lognormal", args={"mean": 0, "sd": 0.5}), rng, 1000)
    assert np.all(ln > 0)

    assert dist.point(DistSpec(family="triangular", args={"lo": 0, "mode": 1, "hi": 2})) == 1.0
    assert dist.point(DistSpec(family="normal", args={"mean": 3, "sd": 1})) == 3.0
    assert dist.point(DistSpec(family="lognormal", args={"mean": 0, "sd": 0.5})) > 0


def test_report_end_to_end(tmp_path):
    cfg = build_default_config()
    cfg.grid.corridors = ["MX_MXN"]
    cfg.grid.turnovers = [1.0, 5.0, 10.0]
    cfg.grid.regimes = ["balanced", "intermittent", "persistent"]
    cfg.calibration.trials = 3_000
    calibrate_and_apply(cfg)

    result = run_sweep(cfg, n=1_500)
    out = tmp_path / "run"
    out.mkdir()
    result.trials.to_parquet(out / "results_trials.parquet", index=False)
    result.summary.to_csv(out / "results_summary.csv", index=False)
    (out / "config_resolved.yaml").write_text(cfg.to_yaml())

    report_path = build_report(out)
    assert report_path.exists()
    html = report_path.read_text()
    for marker in ["Executive summary", "Returns to scale", "Subsidy burn-down", "data:image/png;base64"]:
        assert marker in html
