# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Unit tests for DC procurement network preparation (scripts/data_center.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.data_center import attach_data_center, calculate_annuity

CARRIERS = ("onwind", "solar", "solar-hsat")
MAINLAND = ("ES0 6AC", "ES0 7AC")
ISLAND = "ES1 0AC"


def _profile(n: pypsa.Network, rng: np.random.Generator, high: float) -> pd.Series:
    return pd.Series(rng.uniform(0.0, high, len(n.snapshots)), index=n.snapshots)


@pytest.fixture
def toy_grid(toy_base: pypsa.Network) -> pypsa.Network:
    """
    The toy plus an HVDC-linked island, resource classes and a static plant.

    Every bus has a profiled background plant of each PPA carrier with finite
    potential. At ES0 6AC onwind has two extra classes: a windier one that is
    full and a weaker one with room. ES0 7AC also hosts an existing plant with a
    static availability, which must never anchor a PPA candidate.
    """
    n = toy_base
    rng = np.random.default_rng(11)
    n.add("Bus", ISLAND, carrier="AC", country="ES", x=2.9, y=39.6)
    n.add("Load", f"{ISLAND} demand", bus=ISLAND, p_set=400.0)
    n.add(
        "Link", "island link", bus0="ES0 7AC", bus1=ISLAND, p_nom=400.0, p_min_pu=-1.0
    )
    n.generators.loc[n.generators["carrier"].isin(CARRIERS), "p_nom_max"] = 1000.0
    for bus in (*MAINLAND, ISLAND):
        for car in CARRIERS:
            if not n.generators.index.isin([f"{bus} {car} bg"]).any():
                n.add(
                    "Generator",
                    f"{bus} {car} bg",
                    bus=bus,
                    carrier=car,
                    p_nom=100.0,
                    p_nom_max=1000.0,
                    p_max_pu=_profile(n, rng, 0.6),
                )
    n.add(
        "Generator",
        ["ES0 6AC onwind windy", "ES0 6AC onwind fair"],
        bus="ES0 6AC",
        carrier="onwind",
        p_nom=[200.0, 0.0],
        p_nom_max=[200.0, 800.0],
        p_max_pu=pd.DataFrame(
            {
                "ES0 6AC onwind windy": 0.95 * np.ones(len(n.snapshots)),
                "ES0 6AC onwind fair": 0.7 * np.ones(len(n.snapshots)),
            },
            index=n.snapshots,
        ),
    )
    n.add(
        "Generator", "ES0 7AC onwind static", bus="ES0 7AC", carrier="onwind", p_nom=4.0
    )
    return n


def _dc(index: pd.Index) -> pd.Index:
    return index[index.astype(str).str.startswith("DC")]


def test_no_dc_adds_zero_assets(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame
) -> None:
    """Policy 'no_dc' adds no DC loads, generators, stores or links."""
    before = {
        c: len(getattr(toy_grid, c)) for c in ("loads", "generators", "stores", "links")
    }
    attach_data_center(toy_grid, dc_mw=1000.0, costs=toy_costs, policy="no_dc")
    assert {c: len(getattr(toy_grid, c)) for c in before} == before


def test_unmatched_dc_adds_loads_only(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame
) -> None:
    """Policy 'unmatched_dc' adds one flat full-size DC load at the site and nothing else."""
    n = toy_grid
    background = float(n.loads.loc[n.loads["bus"] == "ES0 7AC", "p_set"].sum())
    attach_data_center(n, dc_mw=1000.0, costs=toy_costs, policy="unmatched_dc")

    assert list(_dc(n.loads.index)) == ["DC ES0 7AC load"]
    assert n.loads.at["DC ES0 7AC load", "bus"] == "ES0 7AC"
    assert n.loads.at["DC ES0 7AC load", "carrier"] == "DC load"
    assert np.isclose(n.loads.at["DC ES0 7AC load", "p_set"], 1000.0)
    # the DC load is additional: background demand at the site is untouched
    site = n.loads[(n.loads["bus"] == "ES0 7AC") & ~n.loads.index.str.startswith("DC")]
    assert np.isclose(float(site["p_set"].sum()), background)
    for c in ("generators", "stores", "links"):
        assert _dc(getattr(n, c).index).empty


def test_each_site_gets_full_dc_mw(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame
) -> None:
    """With two sites, each hosts the full dc_mw (no splitting)."""
    attach_data_center(
        toy_grid,
        dc_mw=1000.0,
        site_buses=list(MAINLAND),
        costs=toy_costs,
        policy="unmatched_dc",
    )
    for bus in MAINLAND:
        assert np.isclose(toy_grid.loads.at[f"DC {bus} load", "p_set"], 1000.0)


def test_spain_wide_adds_ppa_candidates_and_site_battery(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame
) -> None:
    """Siting 'spain_wide' adds one candidate per mainland bus and carrier, one battery."""
    n = toy_grid
    attach_data_center(n, dc_mw=1000.0, costs=toy_costs, policy="hourly_80_grid_credit")

    dc_gens = _dc(n.generators.index)
    dc_links = list(_dc(n.links.index))
    assert len(dc_gens) == len(MAINLAND) * len(CARRIERS)
    assert list(_dc(n.stores.index)) == ["DC ES0 7AC battery"]
    assert sorted(dc_links) == [
        "DC ES0 7AC battery charger",
        "DC ES0 7AC battery discharger",
    ]
    assert n.generators.loc[dc_gens, "p_nom_extendable"].all()
    assert (n.generators.loc[dc_gens, "p_nom"] == 0.0).all()
    assert n.stores.loc[_dc(n.stores.index), "e_cyclic"].all()
    assert n.links.loc[dc_links, "p_nom_extendable"].all()
    # regression: a static p_max_pu plant must not anchor baseload "wind"
    for g in dc_gens:
        assert g in n.generators_t.p_max_pu.columns
        assert float(n.generators_t.p_max_pu[g].mean()) < 1.0
        assert np.isfinite(float(n.generators.at[g, "p_nom_max"]))


@pytest.mark.parametrize(
    ("site", "expected"), [("ES0 7AC", set(MAINLAND)), (ISLAND, {ISLAND})]
)
def test_spain_wide_stays_in_the_site_synchronous_area(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame, site: str, expected: set[str]
) -> None:
    """An HVDC-linked island and the mainland never contract each other's PPAs."""
    attach_data_center(
        toy_grid,
        dc_mw=1000.0,
        site_buses=[site],
        costs=toy_costs,
        policy="hourly_80_grid_credit",
    )
    dc_gens = _dc(toy_grid.generators.index)
    assert set(toy_grid.generators.loc[dc_gens, "bus"]) == expected


def test_local_region_adds_candidates_at_the_site_only(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame
) -> None:
    """Siting 'local_region' adds one candidate per carrier at the site bus."""
    attach_data_center(
        toy_grid,
        dc_mw=1000.0,
        siting="local_region",
        costs=toy_costs,
        policy="hourly_80_grid_credit",
    )
    assert set(_dc(toy_grid.generators.index)) == {f"DC ES0 7AC {c}" for c in CARRIERS}


@pytest.mark.parametrize(
    ("full", "expected"),
    [(False, "ES0 6AC onwind fair"), (True, "ES0 6AC onwind windy")],
)
def test_ppa_candidates_use_best_resource_class(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame, full: bool, expected: str
) -> None:
    """The best class with remaining potential, or the best class once all are full."""
    n = toy_grid
    onwind = n.generators.index[
        (n.generators["bus"] == "ES0 6AC") & (n.generators["carrier"] == "onwind")
    ]
    if full:
        n.generators.loc[onwind, "p_nom"] = n.generators.loc[onwind, "p_nom_max"]
    attach_data_center(n, dc_mw=1000.0, costs=toy_costs, policy="hourly_80_grid_credit")

    room = n.generators.at[expected, "p_nom_max"] - n.generators.at[expected, "p_nom"]
    assert np.isclose(n.generators.at["DC ES0 6AC onwind", "p_nom_max"], max(room, 0.0))
    pd.testing.assert_series_equal(
        n.generators_t.p_max_pu["DC ES0 6AC onwind"],
        n.generators_t.p_max_pu[expected],
        check_names=False,
    )


def test_capital_costs_follow_annuity_formula(
    toy_grid: pypsa.Network, toy_costs: pd.DataFrame
) -> None:
    """PPA and battery annuities: (annuity(lifetime, r) + FOM/100) * investment."""
    r = 0.07
    attach_data_center(
        toy_grid,
        dc_mw=1000.0,
        siting="local_region",
        costs=toy_costs,
        discount_rate=r,
        policy="hourly_80_grid_credit",
    )

    def annuity(tech: str) -> float:
        row = toy_costs.loc[tech]
        return (calculate_annuity(row["lifetime"], r) + row["FOM"] / 100.0) * row[
            "investment"
        ]

    for car, tech in (
        ("onwind", "onwind"),
        ("solar", "solar-utility"),
        ("solar-hsat", "solar-hsat"),
    ):
        assert np.isclose(
            toy_grid.generators.at[f"DC ES0 7AC {car}", "capital_cost"], annuity(tech)
        )
    links = toy_grid.links
    assert np.isclose(
        toy_grid.stores.at["DC ES0 7AC battery", "capital_cost"],
        annuity("battery storage"),
    )
    assert np.isclose(
        links.at["DC ES0 7AC battery charger", "capital_cost"],
        annuity("battery inverter"),
    )
    assert np.isclose(links.at["DC ES0 7AC battery discharger", "capital_cost"], 0.0)
    eta = np.sqrt(toy_costs.at["battery inverter", "efficiency"])
    for link in ("charger", "discharger"):
        assert np.isclose(links.at[f"DC ES0 7AC battery {link}", "efficiency"], eta)
