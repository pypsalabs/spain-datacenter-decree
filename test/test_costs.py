# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT

"""
Guard tests for the DC technology-cost inputs.

Checks the technology-data v0.15.0 cost values, currency year, native 2025 lifetimes,
standalone battery costs, the carrier-to-technology mapping, and the sha256 of the
vendored cost CSV against the registry in doc/data_sources.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.data_center import (
    COST_TECH_MAP,
    _compute_tech_costs,
    _resolve_cost_row,
    calculate_annuity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_COSTS_2025 = (
    REPO_ROOT
    / "data"
    / "custom"
    / "sources"
    / "pypsa_technology_data_v0.15.0_costs_2025.csv"
)
PROCESSED_COSTS_2025 = (
    REPO_ROOT / "pypsa-eur" / "resources" / "wy2013" / "costs_2025_processed.csv"
)
CONFIG_PATH = REPO_ROOT / "config" / "config.spain.yaml"
DATA_SOURCES_DOC = REPO_ROOT / "doc" / "data_sources.md"


def _load_raw_costs(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _raw_param(df: pd.DataFrame, tech: str, param: str) -> float:
    sub = df[(df["technology"] == tech) & (df["parameter"] == param)]
    assert not sub.empty, f"Missing ({tech!r}, {param!r}) in raw costs CSV"
    return float(sub["value"].iloc[0])


def test_costs_match_v015_csv() -> None:
    """Processed costs and CFE cost mapping match technology-data v0.15.0."""
    # 'solar' must resolve to solar-utility, not the 50/50 rooftop/utility blend row
    assert COST_TECH_MAP["solar"][0] == "solar-utility"
    assert COST_TECH_MAP["solar-hsat"][0] == "solar-hsat"
    assert COST_TECH_MAP["onwind"] == ("onwind",)
    assert COST_TECH_MAP["battery storage"] == ("battery storage",)
    assert COST_TECH_MAP["battery inverter"] == ("battery inverter",)

    raw_2025 = _load_raw_costs(RAW_COSTS_2025)

    expected = {
        "onwind": ("onwind", 1438.8845, 1.2347, 1.9035),
        "solar-utility": ("solar-utility", 594.8646, 2.1982, 0.0),
        "solar-hsat": ("solar-utility single-axis tracking", 694.5658, 2.0365, 0.0),
        "battery storage": ("battery storage", 250.0283, 0.0, 0.0),
        "battery inverter": ("battery inverter", 287.4657, 0.2512, 0.0),
    }

    for proc_tech, (raw_tech, inv_kw, fom, vom) in expected.items():
        assert _raw_param(raw_2025, raw_tech, "investment") == pytest.approx(
            inv_kw, rel=1e-5
        )
        if fom > 0:
            assert _raw_param(raw_2025, raw_tech, "FOM") == pytest.approx(fom, rel=1e-4)
        if vom > 0:
            assert _raw_param(raw_2025, raw_tech, "VOM") == pytest.approx(vom, rel=1e-4)

    if PROCESSED_COSTS_2025.exists():
        proc_df = pd.read_csv(PROCESSED_COSTS_2025, index_col=0)
        assert _resolve_cost_row(proc_df, "solar") == "solar-utility"
        assert _resolve_cost_row(proc_df, "solar-hsat") == "solar-hsat"
        for proc_tech, (_, inv_kw, fom, _) in expected.items():
            row = proc_df.loc[proc_tech]
            assert float(row["investment"]) == pytest.approx(inv_kw * 1000.0, rel=1e-5)
            assert float(row["FOM"]) == pytest.approx(fom, abs=1e-4)

        # Verify data_center._compute_tech_costs uses solar-utility for 'solar'
        solar_costs = _compute_tech_costs(proc_df, "solar", discount_rate=0.07)
        solar_util_row = proc_df.loc["solar-utility"]
        expected_solar_cap = float(solar_util_row["investment"]) * (
            calculate_annuity(float(solar_util_row["lifetime"]), 0.07)
            + float(solar_util_row["FOM"]) / 100.0
        )
        assert solar_costs["capital_cost"] == pytest.approx(
            expected_solar_cap, rel=1e-6
        )
        mixed_solar_inv = float(proc_df.loc["solar", "investment"])
        assert float(solar_util_row["investment"]) < mixed_solar_inv


def test_currency_year_2025() -> None:
    """All cost inputs are inflated to EUR_2025 (never mixed EUR_2015 / EUR_2020)."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    assert cfg["data"]["costs"]["version"] == "v0.15.0"

    df = _load_raw_costs(RAW_COSTS_2025)
    for tech in (
        "onwind",
        "solar-utility",
        "solar-utility single-axis tracking",
        "battery storage",
        "battery inverter",
    ):
        inv_row = df[(df["technology"] == tech) & (df["parameter"] == "investment")]
        assert not inv_row.empty
        assert float(inv_row["currency_year"].iloc[0]) == 2025.0


def test_lifetimes_native_v015() -> None:
    """Native v0.15.0 2025 lifetimes are used; no lifetime overwrites in the config (single-year scope)."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    assert "overwrites" not in cfg.get("costs", {}), (
        "costs.overwrites was dropped with the 2025-only scope; it must not reappear silently."
    )

    raw_2025 = _load_raw_costs(RAW_COSTS_2025)
    expected_raw_lifetimes = {
        "onwind": 28.5,
        "solar-utility": 37.5,
        "solar-utility single-axis tracking": 37.5,
        "battery storage": 22.5,
        "battery inverter": 10.0,
    }
    for tech, exp_lt in expected_raw_lifetimes.items():
        assert _raw_param(raw_2025, tech, "lifetime") == pytest.approx(exp_lt)

    if PROCESSED_COSTS_2025.exists():
        p25 = pd.read_csv(PROCESSED_COSTS_2025, index_col=0)
        expected_proc_lifetimes = {
            "onwind": 28.5,
            "solar-utility": 37.5,
            "solar-hsat": 37.5,
            "battery storage": 22.5,
            "battery inverter": 10.0,
        }
        for tech, exp_lt in expected_proc_lifetimes.items():
            assert float(p25.loc[tech, "lifetime"]) == pytest.approx(exp_lt)


def test_battery_is_standalone() -> None:
    """Battery costs reflect standalone utility-scale BESS (DEA Ch. 180), not co-located PV+BESS."""
    raw_2025 = _load_raw_costs(RAW_COSTS_2025)

    e_2025 = _raw_param(raw_2025, "battery storage", "investment")
    p_2025 = _raw_param(raw_2025, "battery inverter", "investment")

    assert e_2025 == pytest.approx(250.0283, rel=1e-4)
    assert p_2025 == pytest.approx(287.4657, rel=1e-4)

    # Round-trip efficiency from one-way inverter efficiency (0.955^2)
    eff_2025 = _raw_param(raw_2025, "battery inverter", "efficiency")
    assert np.isclose(eff_2025**2, 0.912025, atol=1e-5)


def test_checksum() -> None:
    """Vendored v0.15.0 cost CSV matches the SHA-256 digest recorded in doc/data_sources.md."""
    doc_text = DATA_SOURCES_DOC.read_text()
    assert RAW_COSTS_2025.exists(), f"Missing vendored cost file: {RAW_COSTS_2025}"
    digest = hashlib.sha256(RAW_COSTS_2025.read_bytes()).hexdigest()
    assert digest in doc_text, (
        f"SHA-256 {digest} for {RAW_COSTS_2025.name} not found in doc/data_sources.md"
    )
