# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Pytest configuration: project root on ``sys.path`` and the shared toy solve network."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    """Two Spanish nodes with a cheap dirty and an expensive clean background fleet."""
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
