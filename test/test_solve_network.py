# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""End-to-end toy tests for the solve driver (attach + constraints + solve + KPIs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.cfe_constraints import dc_kpis
from scripts.data_center import attach_data_center
from scripts.solve_network import resolve_target, solve_network

HOURS = 48


@pytest.fixture
def toy_costs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "investment": [1.4e6, 0.6e6, 0.7e6, 0.25e6, 0.29e6],
            "lifetime": [28.5, 37.5, 37.5, 22.5, 10.0],
            "FOM": [2.0, 2.0, 2.0, 1.5, 1.5],
            "VOM": [1.0, 0.0, 0.0, 0.0, 0.0],
            "efficiency": [1.0, 1.0, 1.0, 1.0, 0.955],
        },
        index=[
            "onwind",
            "solar-utility",
            "solar-hsat",
            "battery storage",
            "battery inverter",
        ],
    )


@pytest.fixture
def toy_base() -> pypsa.Network:
    """Two Spanish hubs with a cheap dirty and an expensive clean background fleet."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=HOURS, freq="h"))
    rng = np.random.default_rng(7)
    for bus in ("ES0 6AC", "ES0 7AC"):
        n.add("Bus", bus, carrier="AC", country="ES", x=-3.7, y=40.4)
        n.add("Load", f"{bus} demand", bus=bus, p_set=500.0)
        n.add(
            "Generator",
            f"{bus} CCGT",
            bus=bus,
            carrier="CCGT",
            p_nom=2000.0,
            efficiency=0.55,
            marginal_cost=70.0,
        )
        n.add(
            "Generator",
            f"{bus} onwind bg",
            bus=bus,
            carrier="onwind",
            p_nom=300.0,
            marginal_cost=1.0,
            p_max_pu=pd.Series(rng.uniform(0.05, 0.9, HOURS), index=n.snapshots),
        )
        n.add(
            "Generator",
            f"{bus} solar bg",
            bus=bus,
            carrier="solar",
            p_nom=300.0,
            p_max_pu=pd.Series(
                np.tile([0.0] * 7 + [0.6] * 10 + [0.0] * 7, HOURS // 24),
                index=n.snapshots,
            ),
        )
    n.add("Line", "hub link", bus0="ES0 6AC", bus1="ES0 7AC", x=0.1, s_nom=5000.0)
    n.add("Carrier", "CCGT", co2_emissions=0.2)
    for c in ("onwind", "solar", "solar-hsat", "AC"):
        n.add("Carrier", c)
    return n


def _solve(n: pypsa.Network, costs: pd.DataFrame, policy: str, **kw):
    attach_data_center(n, dc_mw=100.0, costs=costs, policy=policy)
    return solve_network(n, policy=policy, **kw)


def test_resolve_target_from_policy_names() -> None:
    assert resolve_target("no_dc") == 0.0
    assert resolve_target("unmatched_dc", 80.0) == 0.0
    assert resolve_target("hourly_80_ppa_only") == pytest.approx(0.80)
    assert resolve_target("strict_hourly_90_grid_credit") == pytest.approx(0.90)
    assert resolve_target("annual_100_res") == pytest.approx(1.0)
    assert resolve_target("annual_80_res") == pytest.approx(0.80)
    assert resolve_target("annual_res") == pytest.approx(1.0)
    assert resolve_target("hourly_80_grid_credit", 60.0) == pytest.approx(0.60)


def test_unmatched_dc_imports_everything(toy_base, toy_costs) -> None:
    n, conv, kpis = _solve(toy_base, toy_costs, "unmatched_dc")
    assert kpis["dc_ppa_total_mw"] == 0.0
    assert kpis["grid_import_mwh"] == pytest.approx(100.0 * HOURS, rel=1e-6)
    assert kpis["cfe_score"] == pytest.approx(0.0, abs=1e-9)
    assert kpis["load_shedding_mwh"] == pytest.approx(0.0, abs=1e-6)


def test_hourly_80_ppa_only_meets_target(toy_base, toy_costs) -> None:
    n, conv, kpis = _solve(toy_base, toy_costs, "hourly_80_ppa_only")
    assert kpis["cfe_score"] >= 0.80 - 1e-6
    assert kpis["dc_ppa_total_mw"] > 0.0
    # PPA-only: imports carry no credit, so the score equals 1 - imports/load
    assert kpis["cfe_score"] == pytest.approx(
        1.0 - kpis["grid_import_mwh"] / kpis["total_load_mwh"], rel=1e-9
    )


def test_grid_credit_iterates_and_relaxes_build(toy_base, toy_costs) -> None:
    n2 = toy_base.copy()
    n, conv, kpis = _solve(
        toy_base,
        toy_costs,
        "hourly_80_grid_credit",
        r0_source=0.3,
        n_iterations=2,
    )
    assert len(conv) == 2
    assert kpis["max_delta"] < 0.05
    assert 0.0 <= kpis["mean_grid_share_es"] <= 1.0
    # grid credit makes the constraint easier than ppa_only at equal target
    attach_data_center(n2, dc_mw=100.0, costs=toy_costs, policy="hourly_80_ppa_only")
    _, _, kpis_ppa = solve_network(n2, policy="hourly_80_ppa_only")
    assert kpis["objective"] <= kpis_ppa["objective"] + 1e-3


def test_strict_hourly_meets_every_hour(toy_base, toy_costs) -> None:
    n, conv, kpis = _solve(toy_base, toy_costs, "strict_hourly_80_ppa_only")
    # per-hour check from the solved primal: matched share >= 80% of load in every hour
    per_t = dc_kpis(n)["constraint_lhs_per_t"]
    assert (per_t >= 0.80 * 100.0 - 1e-4).all()
