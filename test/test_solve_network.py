# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""End-to-end toy tests for the solve driver (attach + constraints + solve + metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.cfe_constraints import dc_metrics, transition_targets
from scripts.data_center import attach_data_center
from scripts.solve_network import resolve_cfe0, resolve_target, solve_network
from scripts.tariff_periods import tariff_period


def _solve(n: pypsa.Network, costs: pd.DataFrame, policy: str, **kw):
    attach_data_center(n, dc_mw=100.0, costs=costs, policy=policy)
    return solve_network(n, policy=policy, **kw)


def _es_price(n: pypsa.Network) -> pd.Series:
    """Hourly Spanish price: nodal prices weighted by background load."""
    load = n.loads_t.p[["ES0 6AC demand", "ES0 7AC demand"]]
    load = load.set_axis(["ES0 6AC", "ES0 7AC"], axis=1)
    prices = n.buses_t.marginal_price[load.columns]
    return (prices * load).sum(axis=1) / load.sum(axis=1)


def test_resolve_target_from_policy_names() -> None:
    assert resolve_target("no_dc") == 0.0
    assert resolve_target("unmatched_dc", 80.0) == 0.0
    assert resolve_target("hourly_80_ppa_only") == pytest.approx(0.80)
    assert resolve_target("strict_hourly_90_grid_credit") == pytest.approx(0.90)
    assert resolve_target("annual_100_res") == pytest.approx(1.0)
    assert resolve_target("annual_80_res") == pytest.approx(0.80)
    assert resolve_target("annual_res") == pytest.approx(1.0)
    assert resolve_target("hourly_80_grid_credit", 80.0) == pytest.approx(0.80)
    assert resolve_target("transition_p1_70_p2_60") == pytest.approx(0.70)
    with pytest.raises(ValueError, match="embeds target"):
        resolve_target("hourly_80_grid_credit", 60.0)


def test_grid_cfe_on_other_snapshots_fails_fast(toy_base) -> None:
    other_year = pd.date_range("2013-01-01", periods=len(toy_base.snapshots), freq="h")
    with pytest.raises(ValueError, match="snapshots"):
        resolve_cfe0(toy_base, pd.Series(0.5, index=other_year))


def test_unmatched_dc_imports_everything(toy_base, toy_costs) -> None:
    n, conv, metrics = _solve(toy_base, toy_costs, "unmatched_dc")
    assert metrics["dc_ppa_total_mw"] == 0.0
    assert metrics["grid_import_mwh"] == pytest.approx(
        100.0 * len(n.snapshots), rel=1e-6
    )
    assert metrics["cfe_score"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["load_shedding_mwh"] == pytest.approx(0.0, abs=1e-6)


def test_unmatched_dc_pays_spanish_price_and_sees_grid_cfe(toy_base, toy_costs) -> None:
    # congest the hub with a surplus node so nodal prices split
    toy_base.lines.loc["hub link", "s_nom"] = 1.0
    toy_base.loads.loc["ES0 6AC demand", "p_set"] = 100.0
    n, conv, metrics = _solve(toy_base, toy_costs, "unmatched_dc")
    nodal = n.buses_t.marginal_price
    assert (nodal["ES0 6AC"] - nodal["ES0 7AC"]).abs().max() > 1.0
    w = n.snapshot_weightings["generators"]
    bill = float((w * 100.0 * _es_price(n)).sum())
    assert metrics["dc_grid_cost_eur"] == pytest.approx(bill, rel=1e-9)
    assert metrics["dc_sales_revenue_eur"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["dc_net_cost_eur_per_mwh"] == pytest.approx(
        bill / metrics["total_load_mwh"], rel=1e-9
    )
    # no PPAs: the flat DC load is covered exactly by the grid's own hourly CFE
    assert metrics["clean_coverage"] == pytest.approx(n._grid_cfe.mean(), rel=1e-9)


def test_net_cost_and_coverage_add_up(toy_base, toy_costs) -> None:
    n, conv, metrics = _solve(toy_base, toy_costs, "hourly_80_ppa_only")
    w = n.snapshot_weightings["generators"]
    links = n.links.index
    dis = links[links.str.contains("discharger")]
    cha = links[links.str.contains("charger") & ~links.str.contains("discharger")]
    ppa = n.generators.index[n.generators.index.str.startswith("DC")]
    # grid position from the physical balance: d - g - dis + cha = im - ex
    net_grid = (
        100.0
        - n.generators_t.p[ppa].sum(axis=1)
        - (n.links_t.p0[dis] * n.links.loc[dis, "efficiency"]).sum(axis=1)
        + n.links_t.p0[cha].sum(axis=1)
    )
    grid_bill = float((w * _es_price(n) * net_grid).sum())
    assets = metrics["dc_capital_cost_eur"] + metrics["dc_marginal_cost_eur"]
    assert metrics["dc_net_cost_eur"] == pytest.approx(assets + grid_bill, rel=1e-6)

    assert metrics["clean_coverage"] == pytest.approx(
        metrics["ppa_only_score"] + metrics["grid_clean_share"], rel=1e-9
    )
    assert metrics["clean_coverage"] >= metrics["cfe_score"] - 1e-9
    per_carrier = [
        v for k, v in metrics.items() if k.startswith("dc_ppa_") and k.endswith("_mwh")
    ]
    assert sum(per_carrier) == pytest.approx(metrics["ppa_generation_mwh"], rel=1e-9)


def test_grid_cfe_is_load_weighted(toy_base, toy_costs) -> None:
    rng = np.random.default_rng(3)
    toy_base.loads_t.p_set["ES0 6AC demand"] = pd.Series(
        rng.uniform(200.0, 800.0, len(toy_base.snapshots)), index=toy_base.snapshots
    )
    n, conv, metrics = _solve(toy_base, toy_costs, "unmatched_dc")
    load = n.loads_t.p[["ES0 6AC demand", "ES0 7AC demand"]].sum(axis=1)
    weighted = float((n._grid_cfe * load).sum() / load.sum())
    assert abs(weighted - n._grid_cfe.mean()) > 1e-4
    assert metrics["mean_grid_cfe_es"] == pytest.approx(weighted, rel=1e-9)


def test_hourly_80_ppa_only_meets_target(toy_base, toy_costs) -> None:
    n, conv, metrics = _solve(toy_base, toy_costs, "hourly_80_ppa_only")
    assert metrics["cfe_score"] >= 0.80 - 1e-6
    assert metrics["dc_ppa_total_mw"] > 0.0
    # PPA-only: imports carry no credit, so the score equals 1 - imports/load
    assert metrics["cfe_score"] == pytest.approx(
        1.0 - metrics["grid_import_mwh"] / metrics["total_load_mwh"], rel=1e-9
    )


def test_grid_credit_iterates_and_relaxes_build(toy_base, toy_costs) -> None:
    n2 = toy_base.copy()
    n, conv, metrics = _solve(
        toy_base,
        toy_costs,
        "hourly_80_grid_credit",
        cfe0_source=0.3,
        n_iterations=2,
    )
    assert len(conv) == 2
    assert metrics["max_delta"] < 0.05
    assert 0.0 <= metrics["mean_grid_cfe_es"] <= 1.0
    # grid credit makes the constraint easier than ppa_only at equal target
    attach_data_center(n2, dc_mw=100.0, costs=toy_costs, policy="hourly_80_ppa_only")
    _, _, metrics_ppa = solve_network(n2, policy="hourly_80_ppa_only")
    assert metrics["objective"] <= metrics_ppa["objective"] + 1e-3


def test_no_same_hour_buy_and_sell(toy_base, toy_costs) -> None:
    # uncapped sales leave the im/ex split degenerate: a free solver may buy and sell at once
    n, conv, metrics = _solve(
        toy_base, toy_costs, "strict_hourly_80_ppa_only", excess_share=None
    )
    im = n.model.solution["DC-im"].to_pandas()
    ex = n.model.solution["DC-ex"].to_pandas()
    assert float(np.minimum(im, ex).max()) < 1e-6


def test_strict_hourly_meets_every_hour(toy_base, toy_costs) -> None:
    n, conv, metrics = _solve(toy_base, toy_costs, "strict_hourly_80_ppa_only")
    # per-hour check from the solved primal: matched share >= 80% of load in every hour
    per_t = dc_metrics(n)["constraint_lhs_per_t"]
    assert (per_t >= 0.80 * 100.0 - 1e-4).all()


@pytest.mark.parametrize(
    ("policy", "floors"),
    [
        ("transition_p1_60", {"P1": 0.6}),
        ("transition_p1_70_p2_60", {"P1": 0.7, "P2": 0.6}),
    ],
)
def test_transition_targets(policy: str, floors: dict[str, float]) -> None:
    assert transition_targets(policy) == floors


@pytest.mark.parametrize("policy", ["transition", "transition_60", "transition_p1_60x"])
def test_transition_targets_reject_malformed_names(policy: str) -> None:
    with pytest.raises(ValueError, match="transition_p"):
        transition_targets(policy)


def test_transition_floors_only_the_named_periods(toy_base, toy_costs) -> None:
    # the toy's second day is a high-season weekday: 9 P1 and 7 P2 hours
    n, conv, metrics = _solve(toy_base, toy_costs, "transition_p1_70_p2_60")
    period = tariff_period(n.snapshots)
    per_t = dc_metrics(n)["constraint_lhs_per_t"]
    assert n.model.constraints["DC-transition-floor"].size == 16
    assert (per_t[period == "P1"] >= 70.0 - 1e-4).all()
    assert (per_t[period == "P2"] >= 60.0 - 1e-4).all()
    assert per_t[period == "P6"].min() < 60.0  # unconstrained hours buy from the grid
