# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Unit tests for 24/7 CFE virtual balance, policy constraints, and unclipped KPIs."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.cfe_constraints import add_dc_cfe_constraints, dc_kpis

CFE_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _build_toy_3bus_network() -> tuple[pypsa.Network, pd.Series]:
    """
    Build a 3-bus, 24-snapshot toy network where DC battery grid-charging is economic.

    Topology:
    - ``ES_Madrid`` (DC load hub, 100 MW flat load)
    - ``ES_Aragon`` (remote wind PPA node)
    - ``ES_South`` (remote solar PPA node)

    During night hours (0..7), grid electricity at ``ES_Madrid`` is cheap (10 EUR/MWh)
    with moderate grid clean share ``r_t = 0.25``, while DC PPA solar/wind availability
    is zero. During evening hours (16..23), grid power is expensive (120 EUR/MWh) and
    dirty (``r_t = 0.0``), making it economic for the DC battery to charge from the
    grid at night and discharge in the evening.
    """
    snapshots = pd.date_range("2025-01-01", periods=24, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)

    for bus in ("ES_Madrid", "ES_Aragon", "ES_South"):
        n.add("Bus", bus, carrier="AC", country="ES")
    n.add("Bus", "DC_Madrid_battery_bus", carrier="battery", country="ES")

    n.add(
        "Line",
        "line_Madrid_Aragon",
        bus0="ES_Madrid",
        bus1="ES_Aragon",
        x=0.01,
        s_nom=5000.0,
    )
    n.add(
        "Line",
        "line_Madrid_South",
        bus0="ES_Madrid",
        bus1="ES_South",
        x=0.01,
        s_nom=5000.0,
    )

    # Background grid load and dispatchable generators
    n.add("Load", "grid_load_Madrid", bus="ES_Madrid", p_set=200.0)

    cheap_night_p_max = pd.Series(0.0, index=snapshots)
    cheap_night_p_max.iloc[0:8] = 1.0
    n.add(
        "Generator",
        "grid_cheap_night",
        bus="ES_Madrid",
        carrier="CCGT",
        p_nom=1000.0,
        p_max_pu=cheap_night_p_max,
        marginal_cost=10.0,
    )
    n.add(
        "Generator",
        "grid_peaker",
        bus="ES_Madrid",
        carrier="OCGT",
        p_nom=2000.0,
        marginal_cost=120.0,
    )

    # DC flat load: 100 MW at ES_Madrid
    n.add("Load", "DC Madrid", bus="ES_Madrid", p_set=100.0)

    # DC PPA solar at ES_South (available hours 8..15) and wind at ES_Aragon (available hours 8..19)
    solar_pu = pd.Series(0.0, index=snapshots)
    solar_pu.iloc[8:16] = 1.0
    wind_pu = pd.Series(0.0, index=snapshots)
    wind_pu.iloc[8:20] = 0.6

    n.add(
        "Generator",
        "DC ES_South solar",
        bus="ES_South",
        carrier="solar",
        p_nom_extendable=True,
        capital_cost=100.0,
        marginal_cost=0.0,
        p_max_pu=solar_pu,
    )
    n.add(
        "Generator",
        "DC ES_Aragon onwind",
        bus="ES_Aragon",
        carrier="onwind",
        p_nom_extendable=True,
        capital_cost=140.0,
        marginal_cost=0.0,
        p_max_pu=wind_pu,
    )

    # DC battery at Madrid with fixed existing capacity so grid arbitrage is immediately active
    n.add(
        "Store",
        "DC Madrid battery",
        bus="DC_Madrid_battery_bus",
        carrier="battery",
        e_nom=400.0,
        e_initial=0.0,
        e_cyclic=True,
    )
    n.add(
        "Link",
        "DC Madrid battery charger",
        bus0="ES_Madrid",
        bus1="DC_Madrid_battery_bus",
        carrier="battery charger",
        p_nom=80.0,
        efficiency=0.95,
    )
    n.add(
        "Link",
        "DC Madrid battery discharger",
        bus0="DC_Madrid_battery_bus",
        bus1="ES_Madrid",
        carrier="battery discharger",
        p_nom=80.0,
        efficiency=0.95,
    )

    r = pd.Series(0.0, index=snapshots)
    r.iloc[0:8] = 0.25
    r.iloc[8:16] = 0.60
    r.iloc[16:24] = 0.0
    return n, r


def test_kpi_equals_constraint_lhs() -> None:
    """Verify KPI score equals exact Linopy constraint LHS when battery grid-charges."""
    n, r = _build_toy_3bus_network()
    target = 0.65

    n.optimize.create_model()
    add_dc_cfe_constraints(
        n=n,
        dc_cfg={"prefix": "DC"},
        r=r,
        policy="hourly_80_grid_credit",
        target=target,
        excess_share=0.20,
    )
    status, condition = n.optimize.solve_model(solver_name="highs", solver="simplex")
    assert status == "ok", f"Expected optimal solve, got {status} ({condition})"

    kpis = dc_kpis(n)

    # Confirm the battery actually charged from the grid during night hours (0..7) when PPA gen is 0
    night_charge = n.links_t.p0["DC Madrid battery charger"].iloc[0:8].sum()
    night_ppa_gen = (
        n.generators_t.p[["DC ES_South solar", "DC ES_Aragon onwind"]]
        .iloc[0:8]
        .sum()
        .sum()
    )
    assert night_ppa_gen == pytest.approx(0.0, abs=1e-7)
    assert night_charge > 10.0, (
        "Expected DC battery to charge from cheap night grid power"
    )

    # During night hours with c_t > 0 and G_t = d_t = 0, virtual balance forces m_t = L_t + c_t > L_t
    m_sol = n.model.solution["DC-import"].to_pandas()
    assert (m_sol.iloc[0:8] > 100.0 - 1e-6).any()

    # Exact equality between primal formula, balance formula, and Linopy constraint LHS
    assert kpis["constraint_lhs"] == pytest.approx(
        kpis["model_constraint_lhs"], abs=1e-6
    )
    assert kpis["constraint_lhs"] == pytest.approx(
        kpis["constraint_lhs_from_balance"], abs=1e-6
    )
    assert kpis["cfe_score"] == pytest.approx(
        kpis["constraint_lhs"] / kpis["total_load_mwh"], abs=1e-6
    )
    assert kpis["cfe_score"] == pytest.approx(target, abs=1e-6)

    # Contrast with a naive clipped calculation that ignores grid-charging subtraction:
    # Unclipped PPA-only matched energy subtracts battery grid-charging c_t, whereas a
    # naive clipped formula min(G_t + d_t, L_t) counts grid-charged battery discharge as clean.
    ppa_gen_t = n.generators_t.p[["DC ES_South solar", "DC ES_Aragon onwind"]].sum(
        axis=1
    )
    dis_t = n.links_t.p0["DC Madrid battery discharger"] * 0.95
    naive_clipped_ppa_mwh = float(np.minimum(ppa_gen_t + dis_t, 100.0).sum())
    unclipped_ppa_mwh = kpis["ppa_only_score"] * kpis["total_load_mwh"]
    assert naive_clipped_ppa_mwh > unclipped_ppa_mwh + 10.0


def test_policies_ppa_only_strict_and_annual_res() -> None:
    """Verify PPA-only (r=0), strict_hourly, annual_res, and excess cap constraints."""
    # 1. PPA-only overrides non-zero r to 0
    n_ppa, r = _build_toy_3bus_network()
    n_ppa.optimize.create_model()
    add_dc_cfe_constraints(
        n=n_ppa,
        dc_cfg={"prefix": "DC"},
        r=r,
        policy="hourly_80_ppa_only",
        target=0.60,
        excess_share=0.10,
    )
    status, _ = n_ppa.optimize.solve_model(solver_name="highs", solver="simplex")
    assert status == "ok"
    kpis_ppa = dc_kpis(n_ppa)
    assert kpis_ppa["grid_clean_import_mwh"] == pytest.approx(0.0, abs=1e-6)
    assert kpis_ppa["cfe_score"] == pytest.approx(kpis_ppa["ppa_only_score"], abs=1e-6)
    assert kpis_ppa["cfe_score"] >= 0.60 - 1e-6
    assert kpis_ppa["excess_share"] <= 0.10 + 1e-6

    # 2. Strict hourly constraint enforces per-snapshot matching
    n_strict, r = _build_toy_3bus_network()
    r_strict = pd.Series(0.50, index=n_strict.snapshots)
    n_strict.optimize.create_model()
    add_dc_cfe_constraints(
        n=n_strict,
        dc_cfg={"prefix": "DC"},
        r=r_strict,
        policy="strict_hourly_80_grid_credit",
        target=0.50,
    )
    status, _ = n_strict.optimize.solve_model(solver_name="highs", solver="simplex")
    assert status == "ok"
    kpis_strict = dc_kpis(n_strict)
    per_t_ratio = kpis_strict["constraint_lhs_per_t"] / 100.0
    assert (per_t_ratio >= 0.50 - 1e-6).all()

    # 3. Annual RES volumetric constraint
    n_ann, r = _build_toy_3bus_network()
    n_ann.optimize.create_model()
    add_dc_cfe_constraints(
        n=n_ann,
        dc_cfg={"prefix": "DC", "annual_equality": True},
        r=r,
        policy="annual_res",
        target=1.00,
    )
    status, _ = n_ann.optimize.solve_model(solver_name="highs", solver="simplex")
    assert status == "ok"
    kpis_ann = dc_kpis(n_ann)
    assert kpis_ann["annual_res_score"] == pytest.approx(1.00, abs=1e-6)


def test_no_clip_in_kpis() -> None:
    """AST scan asserting no ``.clip(...)`` calls exist anywhere in ``scripts/``."""
    py_files = sorted(CFE_DIR.glob("*.py"))
    assert py_files, f"No Python files found in {CFE_DIR}"

    offenders: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clip"
            ):
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, f"Forbidden .clip() calls found in scripts: {offenders}"
