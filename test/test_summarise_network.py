# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""End-to-end toy tests for scenario postprocessing (national and DC metrics)."""

from __future__ import annotations

import pandas as pd
import pypsa
import pytest

from scripts.calculate_grid_cfe import calculate_grid_emission_factor
from scripts.data_center import attach_data_center
from scripts.solve_network import solve_network
from scripts.summarise_network import dc_hourly, summarise_network


def _summarise(
    n: pypsa.Network, costs: pd.DataFrame, policy: str
) -> tuple[pypsa.Network, pd.Series]:
    attach_data_center(n, dc_mw=100.0, costs=costs, policy=policy)
    n, _, metrics = solve_network(n, policy=policy)
    return n, summarise_network(n, pd.Series(metrics))


@pytest.fixture
def toy_with_france(toy_base: pypsa.Network) -> pypsa.Network:
    """The toy plus a cheap French nuclear node behind a 200 MW interconnector."""
    toy_base.add("Bus", "FR0 0AC", carrier="AC", country="FR", x=2.0, y=47.0)
    toy_base.add("Load", "FR0 0AC demand", bus="FR0 0AC", p_set=100.0)
    toy_base.add(
        "Generator",
        "FR0 0AC nuclear",
        bus="FR0 0AC",
        carrier="nuclear",
        p_nom=1000.0,
        marginal_cost=5.0,
        efficiency=0.33,
    )
    toy_base.add("Line", "FR-ES", bus0="FR0 0AC", bus1="ES0 6AC", x=0.1, s_nom=200.0)
    toy_base.add("Carrier", "nuclear")
    return toy_base


def test_spanish_balance_closes(toy_with_france, toy_costs) -> None:
    n, s = _summarise(toy_with_france, toy_costs, "hourly_80_ppa_only")
    background = sum(v for k, v in s.items() if k.startswith("es_gen_"))
    supply = (
        background
        + s["ppa_generation_mwh"]
        + s["es_net_import_mwh"]
        + s["battery_discharge_mwh"]
        - s["battery_charge_mwh"]
    )
    demand = 2 * 500.0 * len(n.snapshots) + s["total_load_mwh"]
    assert s["es_net_import_mwh"] > 0.0
    assert supply == pytest.approx(demand, rel=1e-6)


def test_dc_cost_split_and_curtailment(toy_base, toy_costs) -> None:
    n, s = _summarise(toy_base, toy_costs, "strict_hourly_80_ppa_only")
    split = sum(
        v for k, v in s.items() if k.startswith("dc_ppa_") and k.endswith("_cost_eur")
    )
    assert split + s["dc_battery_cost_eur"] == pytest.approx(
        s["dc_capital_cost_eur"] + s["dc_marginal_cost_eur"], rel=1e-9
    )
    curtailed = [v for k, v in s.items() if k.endswith("_curtailed_mwh")]
    assert min(curtailed) >= -1e-6


def test_unmatched_dc_buys_at_grid_emission_factor(toy_base, toy_costs) -> None:
    n, s = _summarise(toy_base, toy_costs, "unmatched_dc")
    ef = calculate_grid_emission_factor(n, country="ES")
    # flat load bought entirely from the grid: its intensity is the mean factor
    assert s["dc_attributional_intensity_g_per_kwh"] == pytest.approx(
        1e3 * ef.mean(), rel=1e-9
    )
    gas = n.generators_t.p[["ES0 6AC CCGT", "ES0 7AC CCGT"]].sum().sum() / 0.55
    assert s["es_gas_fuel_mwh_th"] == pytest.approx(gas, rel=1e-9)


@pytest.mark.parametrize(
    "policy", ["unmatched_dc", "hourly_80_ppa_only", "strict_hourly_80_ppa_only"]
)
def test_hourly_series_average_to_annual_metrics(toy_base, toy_costs, policy) -> None:
    n, s = _summarise(toy_base, toy_costs, policy)
    hourly = dc_hourly(n)
    # flat DC load: the consumption-weighted mean is the snapshot-weighted one
    w = n.snapshot_weightings["generators"]
    mean = hourly.mul(w, axis=0).sum() / w.sum()
    assert mean["cfe_score"] == pytest.approx(s["clean_coverage"], rel=1e-9)
    assert mean["intensity_g_per_kwh"] == pytest.approx(
        s["dc_attributional_intensity_g_per_kwh"], rel=1e-9
    )


def test_annual_score_is_contracted_yield(toy_base, toy_costs) -> None:
    attach_data_center(toy_base, dc_mw=100.0, costs=toy_costs, policy="annual_100_res")
    n, _, metrics = solve_network(toy_base, policy="annual_100_res", excess_share=0.0)
    # a stale solve-time value of the same name must not survive postprocessing
    s = summarise_network(n, pd.Series({**metrics, "annual_res_score": 99.0}))
    assert s["ppa_yield_mwh"] == pytest.approx(s["total_load_mwh"], rel=1e-6)
    assert s["annual_res_score"] == pytest.approx(1.0, rel=1e-6)
    assert not s.index.duplicated().any()
    # no sales: surplus is curtailed, yet it counts, so the DC still buys from the grid
    losses = s["battery_charge_mwh"] - s["battery_discharge_mwh"]
    assert s["grid_import_mwh"] > losses + 1.0


def test_inconsistent_grid_purchases_fail_fast(toy_base, toy_costs) -> None:
    attach_data_center(toy_base, dc_mw=100.0, costs=toy_costs, policy="unmatched_dc")
    n, _, metrics = solve_network(toy_base, policy="unmatched_dc")
    metrics["grid_import_mwh"] += 1000.0
    with pytest.raises(ValueError, match="same-snapshot"):
        summarise_network(n, pd.Series(metrics))
