"""§7.1 — Yield-subsidy turnover table (exact, within 0.01 bps)."""

import pytest

from corridor_sim.economics import yield_subsidy_bps

EXPECTED = {0.5: 5.479, 1.0: 2.740, 5.0: 0.548, 10.0: 0.274, 20.0: 0.137}


@pytest.mark.parametrize("T,expected", EXPECTED.items())
def test_yield_subsidy_table(T, expected):
    got = float(yield_subsidy_bps(0.05, T))
    assert abs(got - expected) <= 0.01, f"T={T}: got {got}, expected {expected}"
