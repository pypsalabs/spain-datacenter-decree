# SPDX-FileCopyrightText: : 2026 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""Unit tests for summary baseline delta keying and calculations."""

import numpy as np
import pandas as pd
import pytest

from scripts.make_summary import (
    NO_DC_BASELINE_KEYS,
    UNMATCHED_DC_BASELINE_KEYS,
    attach_baseline_deltas,
)


def _make_suite3_summary_df() -> pd.DataFrame:
    """Build a synthetic multi-year, multi-resolution, multi-dc_mw scenario summary table."""
    rows: list[dict[str, object]] = []
    for planning_year in (2025, 2030):
        for weather_year in (2010, 2013, 2020):
            for resolution in ("1h", "3h"):
                base_co2 = 20.0 + (2030 - planning_year) * 0.8 + (weather_year - 2010) * 0.1
                base_cost = 1000.0 + (planning_year - 2025) * 50.0 + (0.0 if resolution == "1h" else 5.0)
                rows.append(
                    {
                        "planning_year": planning_year,
                        "weather_year": weather_year,
                        "resolution": resolution,
                        "dc_mw": 0,
                        "policy": "no_dc",
                        "emissions": base_co2,
                        "system_cost": base_cost,
                    }
                )
                for dc_mw in (500, 1000, 1500, 3500, 6000):
                    unm_co2 = base_co2 + 0.002 * dc_mw
                    unm_cost = base_cost + 0.5 * dc_mw
                    rows.append(
                        {
                            "planning_year": planning_year,
                            "weather_year": weather_year,
                            "resolution": resolution,
                            "dc_mw": dc_mw,
                            "policy": "unmatched_dc",
                            "emissions": unm_co2,
                            "system_cost": unm_cost,
                        }
                    )
                    rows.append(
                        {
                            "planning_year": planning_year,
                            "weather_year": weather_year,
                            "resolution": resolution,
                            "dc_mw": dc_mw,
                            "policy": "hourly_80_grid_credit",
                            "emissions": base_co2 + 0.0004 * dc_mw,
                            "system_cost": unm_cost + 0.1 * dc_mw,
                        }
                    )
                    rows.append(
                        {
                            "planning_year": planning_year,
                            "weather_year": weather_year,
                            "resolution": resolution,
                            "dc_mw": dc_mw,
                            "policy": "hourly_80_ppa_only",
                            "emissions": base_co2 + 0.0002 * dc_mw,
                            "system_cost": unm_cost + 0.15 * dc_mw,
                        }
                    )
    return pd.DataFrame(rows)


def test_unmatched_delta_zero() -> None:
    """Verify unmatched_dc rows have exact zero delta across all dc_mw levels."""
    df = _make_suite3_summary_df()
    out = attach_baseline_deltas(df)

    unm_rows = out.loc[out["policy"] == "unmatched_dc"]
    assert len(unm_rows) == 2 * 3 * 2 * 5  # 60 unmatched_dc rows across 5 dc_mw levels

    for col in (
        "delta_unmatched_dc_emissions",
        "emissions_delta_unmatched",
        "delta_unmatched_dc_system_cost",
        "system_cost_delta_unmatched",
        "dc_co2_reduction_vs_unmatched_pct",
    ):
        assert np.allclose(unm_rows[col].to_numpy(), 0.0, atol=1e-12), (
            f"Expected all zeros in {col} for unmatched_dc rows"
        )


def test_baseline_keys() -> None:
    """Verify no_dc is keyed on (planning_year, weather_year, resolution) and unmatched_dc adds dc_mw."""
    assert NO_DC_BASELINE_KEYS == ("planning_year", "weather_year", "resolution")
    assert UNMATCHED_DC_BASELINE_KEYS == (
        "planning_year",
        "weather_year",
        "resolution",
        "dc_mw",
    )

    df = _make_suite3_summary_df()
    out = attach_baseline_deltas(df)

    # Check a specific slice: (2025, 2013, '1h', dc_mw=3500, policy='hourly_80_grid_credit')
    row = out.loc[
        (out["planning_year"] == 2025)
        & (out["weather_year"] == 2013)
        & (out["resolution"] == "1h")
        & (out["dc_mw"] == 3500)
        & (out["policy"] == "hourly_80_grid_credit")
    ].iloc[0]

    # Expected vs no_dc: 0.0004 * 3500 = 1.4 Mt CO2, (0.5 + 0.1) * 3500 = 2100.0 cost
    assert row["delta_no_dc_emissions"] == pytest.approx(0.0004 * 3500, abs=1e-9)
    assert row["delta_no_dc_system_cost"] == pytest.approx(0.6 * 3500, abs=1e-9)

    # Expected vs unmatched_dc at dc_mw=3500: (0.0004 - 0.002) * 3500 = -5.6 Mt CO2, +0.1 * 3500 = +350.0 cost
    assert row["delta_unmatched_dc_emissions"] == pytest.approx(-0.0016 * 3500, abs=1e-9)
    assert row["delta_unmatched_dc_system_cost"] == pytest.approx(0.1 * 3500, abs=1e-9)

    # Abatement percentage vs unmatched_dc: (0.002 - 0.0004) / 0.002 * 100 = 80.0%
    assert row["dc_co2_reduction_vs_unmatched_pct"] == pytest.approx(80.0, abs=1e-9)

    # Verify missing key columns raise KeyError
    with pytest.raises(KeyError):
        attach_baseline_deltas(df.drop(columns=["weather_year"]))
    with pytest.raises(KeyError):
        attach_baseline_deltas(df.drop(columns=["dc_mw"]))
