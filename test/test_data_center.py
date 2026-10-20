# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Unit tests for DC procurement network preparation (scripts/data_center.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.data_center import (
    attach_data_center,
    calculate_annuity,
    load_costs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_2025_PATH = (
    REPO_ROOT / "resources" / "wy2024" / "networks" / "base_2025_patched.nc"
)
COSTS_2025_PATH = (
    REPO_ROOT / "pypsa-eur" / "resources" / "wy2024" / "costs_2025_processed.csv"
)


@pytest.fixture(scope="module")
def base_sliced_network() -> pypsa.Network:
    """Load a 24-hour slice of the wy2024 patched base network once per module."""
    if not BASE_2025_PATH.exists():
        pytest.skip(f"Missing base network (run the workflow first): {BASE_2025_PATH}")
    n = pypsa.Network(str(BASE_2025_PATH))
    n.set_snapshots(n.snapshots[:24])
    return n


@pytest.fixture(scope="module")
def costs_2025() -> pd.DataFrame:
    """Load processed 2025 v0.15.0 technology costs."""
    if not COSTS_2025_PATH.exists():
        pytest.skip(
            f"Missing processed costs (run the workflow first): {COSTS_2025_PATH}"
        )
    return load_costs(COSTS_2025_PATH)


def test_no_dc_adds_zero_assets(
    base_sliced_network: pypsa.Network, costs_2025: pd.DataFrame
) -> None:
    """Policy 'no_dc' must add 0 DC loads, 0 DC generators, 0 DC stores, and 0 DC links."""
    n = base_sliced_network.copy()
    n_loads_before = len(n.loads)
    n_gens_before = len(n.generators)
    n_stores_before = len(n.stores)
    n_links_before = len(n.links)

    attach_data_center(
        n,
        dc_mw=1000.0,
        siting="spain_wide",
        costs=costs_2025,
        policy="no_dc",
    )

    assert len(n.loads) == n_loads_before
    assert len(n.generators) == n_gens_before
    assert len(n.stores) == n_stores_before
    assert len(n.links) == n_links_before
    assert not n.loads.index.astype(str).str.startswith("DC").any()
    assert not n.generators.index.astype(str).str.startswith("DC").any()
    assert not n.stores.index.astype(str).str.startswith("DC").any()
    for car in ("CHP", "csp", "boundary", "load"):
        assert car in n.carriers.index
    # Fleet composition is prepare_network's job: phantom upstream lignite is zeroed
    # by the national_capacities patch, never removed by the DC attach.
    assert (n.generators.loc[n.generators["carrier"] == "lignite", "p_nom"] == 0).all()


def test_unmatched_dc_adds_loads_only(
    base_sliced_network: pypsa.Network, costs_2025: pd.DataFrame
) -> None:
    """Policy 'unmatched_dc' adds one flat full-size DC load at the site and nothing else."""
    n = base_sliced_network.copy()
    bg_load_aragon = float(n.loads.loc[n.loads["bus"] == "ES0 7AC", "p_set"].sum())

    attach_data_center(
        n,
        dc_mw=1000.0,
        siting="spain_wide",
        costs=costs_2025,
        policy="unmatched_dc",
    )

    dc_loads = n.loads.index[n.loads.index.astype(str).str.startswith("DC")]
    assert list(dc_loads) == ["DC ES0 7AC load"]
    assert n.loads.at["DC ES0 7AC load", "bus"] == "ES0 7AC"
    assert n.loads.at["DC ES0 7AC load", "carrier"] == "DC load"
    assert np.isclose(n.loads.at["DC ES0 7AC load", "p_set"], 1000.0)

    # Background load at the site bus remains untouched (DC load is additional)
    bg_loads_after = n.loads[
        (n.loads["bus"] == "ES0 7AC")
        & (~n.loads.index.astype(str).str.startswith("DC"))
    ]
    assert np.isclose(float(bg_loads_after["p_set"].sum()), bg_load_aragon)

    assert not n.generators.index.astype(str).str.startswith("DC").any()
    assert not n.stores.index.astype(str).str.startswith("DC").any()
    assert not n.links.index.astype(str).str.startswith("DC").any()


def test_each_site_gets_full_dc_mw(
    base_sliced_network: pypsa.Network, costs_2025: pd.DataFrame
) -> None:
    """With two sites, each hosts the full dc_mw (no splitting)."""
    n = base_sliced_network.copy()
    attach_data_center(
        n,
        dc_mw=1000.0,
        site_buses=["ES0 7AC", "ES0 6AC"],
        siting="spain_wide",
        costs=costs_2025,
        policy="unmatched_dc",
    )
    assert np.isclose(n.loads.at["DC ES0 7AC load", "p_set"], 1000.0)
    assert np.isclose(n.loads.at["DC ES0 6AC load", "p_set"], 1000.0)


def test_spain_wide_adds_24_ppa_gens_and_site_battery(
    base_sliced_network: pypsa.Network, costs_2025: pd.DataFrame
) -> None:
    """Siting 'spain_wide' adds 24 PPA gens (8 mainland buses x 3 carriers), 1 battery, 1 load."""
    n = base_sliced_network.copy()
    attach_data_center(
        n,
        dc_mw=1000.0,
        siting="spain_wide",
        costs=costs_2025,
        policy="hourly_80_grid_credit",
    )

    dc_loads = n.loads.index[n.loads.index.astype(str).str.startswith("DC")]
    dc_gens = n.generators.index[n.generators.index.astype(str).str.startswith("DC")]
    dc_stores = n.stores.index[n.stores.index.astype(str).str.startswith("DC")]
    dc_links = n.links.index[n.links.index.astype(str).str.startswith("DC")]
    dc_chargers = [lk for lk in dc_links if "battery charger" in str(lk)]
    dc_dischargers = [lk for lk in dc_links if "battery discharger" in str(lk)]

    assert len(dc_loads) == 1
    assert len(dc_gens) == 24  # 8 mainland ES buses x 3 carriers
    assert list(dc_stores) == ["DC ES0 7AC battery"]
    assert dc_chargers == ["DC ES0 7AC battery charger"]
    assert dc_dischargers == ["DC ES0 7AC battery discharger"]

    assert n.generators.loc[dc_gens, "p_nom_extendable"].all()
    assert (n.generators.loc[dc_gens, "p_nom"] == 0.0).all()
    assert n.stores.loc[dc_stores, "e_nom_extendable"].all()
    assert n.stores.loc[dc_stores, "e_cyclic"].all()
    assert n.links.loc[dc_links, "p_nom_extendable"].all()

    # Regression (corrupted the first core-suite run): every PPA candidate must carry
    # a time-varying resource-class profile; a static p_max_pu means baseload "wind".
    for g in dc_gens:
        assert g in n.generators_t.p_max_pu.columns, (
            f"PPA candidate {g!r} has no time-varying availability profile"
        )
        assert float(n.generators_t.p_max_pu[g].mean()) < 1.0
        assert np.isfinite(float(n.generators.at[g, "p_nom_max"]))


@pytest.mark.parametrize(
    ("site", "expected"),
    [("ES0 7AC", {f"ES0 {i}AC" for i in range(8)}), ("ES1 0AC", {"ES1 0AC"})],
)
def test_spain_wide_stays_in_the_site_synchronous_area(
    base_sliced_network: pypsa.Network,
    costs_2025: pd.DataFrame,
    site: str,
    expected: set[str],
) -> None:
    """The HVDC-linked Balearics and the mainland never contract each other's PPAs."""
    n = base_sliced_network.copy()
    attach_data_center(
        n,
        dc_mw=1000.0,
        site_buses=[site],
        siting="spain_wide",
        costs=costs_2025,
        policy="hourly_80_grid_credit",
    )
    dc_gens = n.generators.index[n.generators.index.astype(str).str.startswith("DC")]
    assert set(n.generators.loc[dc_gens, "bus"]) == expected


def test_local_region_adds_3_ppa_gens(
    base_sliced_network: pypsa.Network, costs_2025: pd.DataFrame
) -> None:
    """Siting 'local_region' adds only 3 extendable PPA generators at the site bus."""
    n = base_sliced_network.copy()
    attach_data_center(
        n,
        dc_mw=1000.0,
        siting="local_region",
        costs=costs_2025,
        policy="hourly_80_grid_credit",
    )

    dc_gens = n.generators.index[n.generators.index.astype(str).str.startswith("DC")]
    expected_gens = {f"DC ES0 7AC {car}" for car in ("onwind", "solar", "solar-hsat")}
    assert set(dc_gens) == expected_gens


def _expected_best_class(n: pypsa.Network, bus: str, car: str) -> tuple[str, float]:
    """Reference implementation: highest-mean-CF profiled class with remaining potential."""
    bg = n.generators.index[
        (n.generators["bus"] == bus)
        & (n.generators["carrier"] == car)
        & (~n.generators.index.astype(str).str.startswith("DC"))
    ].intersection(n.generators_t.p_max_pu.columns)
    remaining = (
        n.generators.loc[bg, "p_nom_max"] - n.generators.loc[bg, "p_nom"]
    ).where(lambda s: s > 0.0, 0.0)
    mean_cf = n.generators_t.p_max_pu[bg].mean()
    pool = remaining[remaining > 0.0].index
    if pool.empty:
        pool = bg
    best = mean_cf.loc[pool].idxmax()
    return str(best), float(remaining.at[best])


def test_ppa_candidates_use_best_resource_class(
    base_sliced_network: pypsa.Network, costs_2025: pd.DataFrame
) -> None:
    """Each PPA candidate carries the best remaining resource class: its profile and potential."""
    n = base_sliced_network.copy()
    # Fill every onwind class at one bus to test the zero-remaining fallback
    full_bus_mask = (n.generators["bus"] == "ES0 0AC") & (
        n.generators["carrier"] == "onwind"
    )
    n.generators.loc[full_bus_mask, "p_nom"] = (
        n.generators.loc[full_bus_mask, "p_nom_max"] + 500.0
    )

    attach_data_center(
        n,
        dc_mw=1000.0,
        siting="spain_wide",
        costs=costs_2025,
        policy="hourly_80_grid_credit",
    )

    dc_gens = n.generators.index[n.generators.index.astype(str).str.startswith("DC")]
    assert (n.generators.loc[dc_gens, "p_nom_max"] >= 0.0).all()
    assert np.isclose(n.generators.at["DC ES0 0AC onwind", "p_nom_max"], 0.0)

    for g in dc_gens:
        bus = str(n.generators.at[g, "bus"])
        car = str(n.generators.at[g, "carrier"])
        ref_bg, expected_pot = _expected_best_class(n, bus, car)
        assert np.isclose(
            float(n.generators.at[g, "p_nom_max"]), expected_pot, rtol=1e-6, atol=1e-6
        )
        pd.testing.assert_series_equal(
            n.generators_t.p_max_pu[g],
            n.generators_t.p_max_pu[ref_bg],
            check_names=False,
        )


def test_capital_costs_match_v015_annuity_formula(
    base_sliced_network: pypsa.Network, costs_2025: pd.DataFrame
) -> None:
    """Verify capital_cost matches v0.15.0 annuity formula at 7% WACC and harmonized lifetimes."""
    n = base_sliced_network.copy()
    discount_rate = 0.07
    attach_data_center(
        n,
        dc_mw=1000.0,
        siting="local_region",
        costs=costs_2025,
        discount_rate=discount_rate,
        policy="hourly_80_grid_credit",
    )

    # Native v0.15.0 2025 lifetimes (no cross-year overwrites in a single-year study)
    expected_lifetimes = {
        ("onwind", "onwind"): 28.5,
        ("solar", "solar-utility"): 37.5,
        ("solar-hsat", "solar-hsat"): 37.5,
        ("battery storage", "battery storage"): 22.5,
        ("battery inverter", "battery inverter"): 10.0,
    }
    for (car, row_key), exp_lt in expected_lifetimes.items():
        actual_lt = float(costs_2025.at[row_key, "lifetime"])
        assert np.isclose(actual_lt, exp_lt), (
            f"Lifetime mismatch for {car}: {actual_lt} != {exp_lt}"
        )

    # Verify PPA generator capital_cost = (annuity(lifetime, r) + FOM/100) * investment
    for car, row_key in (
        ("onwind", "onwind"),
        ("solar", "solar-utility"),
        ("solar-hsat", "solar-hsat"),
    ):
        inv = float(costs_2025.at[row_key, "investment"])
        fom = float(costs_2025.at[row_key, "FOM"])
        lt = float(costs_2025.at[row_key, "lifetime"])
        expected_cap = (calculate_annuity(lt, discount_rate) + fom / 100.0) * inv
        gen_name = f"DC ES0 7AC {car}"
        assert np.isclose(
            float(n.generators.at[gen_name, "capital_cost"]),
            expected_cap,
            rtol=1e-6,
        )
        assert np.isclose(
            float(n.generators.at[gen_name, "capital_cost"]),
            float(costs_2025.at[car, "capital_cost"]),
            rtol=1e-6,
        )

    # Verify battery store and inverter capital_cost and one-way efficiency
    inv_store = float(costs_2025.at["battery storage", "investment"])
    fom_store = float(costs_2025.at["battery storage", "FOM"])
    lt_store = float(costs_2025.at["battery storage", "lifetime"])
    exp_store_cap = (
        calculate_annuity(lt_store, discount_rate) + fom_store / 100.0
    ) * inv_store
    assert np.isclose(
        float(n.stores.at["DC ES0 7AC battery", "capital_cost"]),
        exp_store_cap,
        rtol=1e-6,
    )

    inv_link = float(costs_2025.at["battery inverter", "investment"])
    fom_link = float(costs_2025.at["battery inverter", "FOM"])
    lt_link = float(costs_2025.at["battery inverter", "lifetime"])
    exp_link_cap = (
        calculate_annuity(lt_link, discount_rate) + fom_link / 100.0
    ) * inv_link
    assert np.isclose(
        float(n.links.at["DC ES0 7AC battery charger", "capital_cost"]),
        exp_link_cap,
        rtol=1e-6,
    )
    assert np.isclose(
        float(n.links.at["DC ES0 7AC battery discharger", "capital_cost"]),
        0.0,
    )

    eta_rt = float(costs_2025.at["battery inverter", "efficiency"])
    assert np.isclose(
        float(n.links.at["DC ES0 7AC battery charger", "efficiency"]),
        np.sqrt(eta_rt),
        rtol=1e-6,
    )
    assert np.isclose(
        float(n.links.at["DC ES0 7AC battery discharger", "efficiency"]),
        np.sqrt(eta_rt),
        rtol=1e-6,
    )
