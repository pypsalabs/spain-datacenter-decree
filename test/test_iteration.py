# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Unit tests for the forward-pass grid-CFE iteration."""

import pandas as pd
import pypsa
import pytest

from scripts.cfe_constraints import add_dc_cfe_constraints, dc_kpis
from scripts.iterate import solve_with_grid_iteration


def _build_toy_iteration_network() -> pypsa.Network:
    """
    Build a 2-country (ES, FR), 6-snapshot toy network with exact fixed point r* = 0.60.

    Background system:
    - ``ES0`` has 750 MW background load, a 400 MW must-run wind generator (cost 0),
      and a dispatchable CCGT generator (capacity 600 MW, cost 50 EUR/MWh).
    - ``FR0`` has 100 MW background load, a 160 MW must-run wind generator (cost 0),
      and a 40 MW must-run CCGT generator (cost 10 EUR/MWh), so FR exports 100 MW
      to ES every hour over ``line_FR_ES`` with FR clean share ``160 / 200 = 0.80``.
    - DC load ``DC Madrid`` at ``ES0`` is 100 MW flat.
    - DC PPA wind generator ``DC ES0 onwind`` at ``ES0`` is extendable (capital_cost
      600 EUR/MW, marginal_cost 0 EUR/MWh, availability 1.0).

    Analytical fixed point for ``target = 0.80``:
    Let ``G`` be the hourly DC PPA wind dispatch (flat across the 6 snapshots):
    - The DC imports ``m = 100 - G`` MW from the grid.
    - Excluding DC assets, ES domestic + imported supply is ``750 + (100 - G)`` MW,
      and total clean supply (ES wind + FR clean import) is ``400 + 100 * 0.80 = 480`` MW.
    - Binding 80% hourly CFE constraint: ``100 - (1 - r) * (100 - G) = 80 => 100 - G = 20 / (1 - r)``.
    - Fixed point equation: ``r = 480 / (750 + 20 / (1 - r))``, with unique solution
      ``r* = 0.60`` and ``G* = 50.0 MW``.
    """
    snapshots = pd.date_range("2025-01-01", periods=6, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)

    n.add("Bus", "ES0", carrier="AC", country="ES")
    n.add("Bus", "FR0", carrier="AC", country="FR")
    n.add("Line", "line_FR_ES", bus0="FR0", bus1="ES0", x=0.01, r=0.001, s_nom=1000.0)

    # Background loads
    n.add("Load", "ES_grid_load", bus="ES0", p_set=750.0)
    n.add("Load", "FR_grid_load", bus="FR0", p_set=100.0)

    # Background generators
    n.add(
        "Generator",
        "ES_wind",
        bus="ES0",
        carrier="onwind",
        p_nom=400.0,
        p_min_pu=1.0,
        p_max_pu=1.0,
        marginal_cost=0.0,
    )
    n.add(
        "Generator",
        "ES_gas",
        bus="ES0",
        carrier="CCGT",
        p_nom=600.0,
        marginal_cost=50.0,
    )
    n.add(
        "Generator",
        "FR_wind",
        bus="FR0",
        carrier="onwind",
        p_nom=160.0,
        p_min_pu=1.0,
        p_max_pu=1.0,
        marginal_cost=0.0,
    )
    n.add(
        "Generator",
        "FR_gas",
        bus="FR0",
        carrier="CCGT",
        p_nom=40.0,
        p_min_pu=1.0,
        p_max_pu=1.0,
        marginal_cost=10.0,
    )

    # DC load and extendable PPA wind (capital_cost high enough per MWh that CFE constraint binds)
    n.add("Load", "DC Madrid", bus="ES0", p_set=100.0)
    n.add(
        "Generator",
        "DC ES0 onwind",
        bus="ES0",
        carrier="onwind",
        p_nom_extendable=True,
        capital_cost=600.0,
        marginal_cost=0.0,
    )
    return n


def test_forward_passes_reach_fixed_point() -> None:
    """Three forward passes land on the analytical fixed point r* = 0.60 (plain replacement)."""
    n = _build_toy_iteration_network()
    target = 0.80
    expected_r_star = 0.60
    expected_g_star = 50.0

    def build_fn(net: pypsa.Network, cfe_iter: pd.Series) -> None:
        add_dc_cfe_constraints(
            n=net,
            dc_cfg={"prefix": "DC"},
            cfe_t=cfe_iter,
            policy="hourly_80_grid_credit",
            target=target,
            excess_share=0.20,
        )

    solved_n, conv_df = solve_with_grid_iteration(
        n=n,
        build_fn=build_fn,
        r0=0.0,
        n_iterations=3,
        country="ES",
        clean_carriers="res",
    )

    expected_cols = {
        "iteration",
        "max_delta",
        "objective",
        "mean_cfe_in",
        "mean_cfe_out",
    }
    assert expected_cols.issubset(set(conv_df.columns))
    assert len(conv_df) == 3, "Exactly n_iterations forward passes, no early exit"
    # The delta is a trailing signal: the solution sits on the fixed point (checked
    # below) while max_delta still reports the previous pass's correction
    assert float(conv_df["max_delta"].iloc[-1]) < 0.05
    assert float(conv_df["max_delta"].iloc[-1]) < float(conv_df["max_delta"].iloc[0])

    # Verify the exact analytical fixed point r* = 0.60 and G* = 50.0 MW
    assert float(conv_df["mean_cfe_out"].iloc[-1]) == pytest.approx(
        expected_r_star, abs=1e-3
    )
    assert float(solved_n.generators.at["DC ES0 onwind", "p_nom_opt"]) == pytest.approx(
        expected_g_star, abs=0.5
    )

    kpis = dc_kpis(solved_n)
    assert kpis["cfe_score"] == pytest.approx(target, abs=1e-3)

    with pytest.raises(ValueError, match="n_iterations"):
        solve_with_grid_iteration(n=n, build_fn=build_fn, n_iterations=0)
