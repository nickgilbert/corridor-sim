"""§6.4 / DoD#5 — same seed reproduces identical results."""

import pandas as pd

from corridor_sim.config import build_default_config
from corridor_sim.montecarlo import _stable_idx, run_cell, run_sweep


def _small_cfg():
    cfg = build_default_config()
    cfg.grid.corridors = ["MX_MXN"]
    cfg.grid.turnovers = [1.0, 5.0]
    cfg.grid.regimes = ["persistent"]
    cfg.trials = 800
    return cfg


def test_cell_seed_is_process_stable():
    """Per-cell seeds must not depend on Python's salted str hash (PYTHONHASHSEED).

    These constants are pinned: if the cell index ever reverts to the built-in ``hash()``,
    the value becomes per-process random and this fails — catching the reproducibility
    regression that in-process equality checks (below) cannot see.
    """
    assert _stable_idx("MX_MXN") == 778325625
    assert _stable_idx("persistent") == 300870985


def test_cell_reproducible():
    cfg = _small_cfg()
    a = run_cell(cfg, "MX_MXN", "persistent", 5.0, 800).trials
    b = run_cell(cfg, "MX_MXN", "persistent", 5.0, 800).trials
    pd.testing.assert_frame_equal(a, b)


def test_sweep_reproducible():
    a = run_sweep(_small_cfg(), n=800).trials
    b = run_sweep(_small_cfg(), n=800).trials
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    cfg1 = _small_cfg()
    cfg2 = _small_cfg()
    cfg2.seed = 999
    a = run_sweep(cfg1, n=800).trials["c_total"].sum()
    b = run_sweep(cfg2, n=800).trials["c_total"].sum()
    assert a != b
