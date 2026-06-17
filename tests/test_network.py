"""Act II — hub-and-spoke network model: aggregation, netting, fragmentation."""

import pytest

from corridor_sim.calibrate import calibrate_and_apply
from corridor_sim.config import build_default_config
from corridor_sim.network import (
    NetworkConfig,
    hub_beta,
    hub_turnover,
    network_subsidy,
    network_sweep,
)


@pytest.fixture(scope="module")
def cfg():
    c = build_default_config()
    c.calibration.trials = 4_000
    calibrate_and_apply(c)
    return c


def test_hub_aggregates_and_nets():
    nc = NetworkConfig()
    # turnover rises with N (more corridors → deeper shared book)
    assert hub_turnover(4, nc) > hub_turnover(1, nc)
    # imbalance falls with N (netting), bounded below by the floor
    assert hub_beta(1, nc) == pytest.approx(nc.beta_0)
    assert hub_beta(4, nc) < hub_beta(1, nc)
    assert hub_beta(10_000, nc) >= nc.beta_floor - 1e-9


def test_fragmentation_throttles_B_hub():
    nc = NetworkConfig()
    # more stablecoins → lower per-pool turnover and worse netting
    assert hub_turnover(4, nc, fragments=10) < hub_turnover(4, nc, fragments=3) < hub_turnover(4, nc, fragments=1)
    assert hub_beta(4, nc, fragments=10) > hub_beta(4, nc, fragments=3) > hub_beta(4, nc, fragments=1)


def test_point1_yield_dominates(cfg):
    # Deliver a dollar asset: C (local→USTRY) ≤ B (local→USDC) ≪ A, in every corridor.
    df = network_sweep(cfg, NetworkConfig())
    last = df[df.n == df.n.max()]
    for _, r in last.iterrows():
        assert r["C_local_ustry"] <= r["B_local"] + 1e-6
        assert r["C_local_ustry"] < r["A"]


def test_point2_compounding_needs_unfragmented_reserve(cfg):
    nc = NetworkConfig()
    df = network_sweep(cfg, nc)
    mx = df[df.corridor == "MX_MXN"].set_index("n")
    # C's cross-border cost falls as the network grows (unified reserve aggregates to a floor)
    assert mx["C_xborder"].iloc[-1] < mx["C_xborder"].iloc[0]
    # fragmented B stays a large multiple above C at every network size — it never reaches
    # the floor, because its liquidity can't aggregate into one deep pool.
    for n in mx.index:
        assert mx.loc[n, "B_xborder_s10"] > 1.5 * mx.loc[n, "C_xborder"]
        assert mx.loc[n, "B_xborder_s3"] > mx.loc[n, "C_xborder"]
    # the unified steelman (1 stablecoin) is the only B that nearly catches C — by yield alone
    assert abs(mx["B_xborder_s1"].iloc[-1] - mx["C_xborder"].iloc[-1]) < 1.0


def test_growing_fragmentation_gap_widens(cfg):
    # a stablecoin per market (S = N): B's cross-border cost flatlines (no aggregation),
    # while C keeps falling — so the gap to C *widens* with the network.
    nc = NetworkConfig()
    mx = network_sweep(cfg, nc)
    mx = mx[mx.corridor == "MX_MXN"].set_index("n")
    gap1 = mx["B_xborder_grow"].iloc[0] - mx["C_xborder"].iloc[0]
    gap_last = mx["B_xborder_grow"].iloc[-1] - mx["C_xborder"].iloc[-1]
    assert gap_last > gap1
    # B_grow barely moves once each market has its own coin (turnover per pool flat)
    assert abs(mx["B_xborder_grow"].iloc[-1] - mx["B_xborder_grow"].iloc[1]) < 1.0


def test_network_subsidy_C_retires_B_does_not(cfg):
    _, _, summ = network_subsidy(cfg, NetworkConfig())
    assert summ["C"]["retires"]            # unified hub aggregates → C reaches break-even
    assert not summ["B_grow"]["retires"]   # fragmenting stablecoins never aggregate
    assert not summ["A"]["retires"]
    # C's total network subsidy is a small fraction of the fragmented-B and incumbent bills
    assert summ["C"]["total_subsidy"] < summ["B_grow"]["total_subsidy"] < summ["A"]["total_subsidy"]
