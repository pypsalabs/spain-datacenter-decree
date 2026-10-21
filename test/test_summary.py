# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Unit tests for summary baseline delta keying and calculations."""

import numpy as np
import pandas as pd
import pytest

from scripts.make_summary import attach_baseline_deltas

VARIANT_OFFSET = {"raw": (3.0, 500.0), "patched": (0.0, 0.0)}  # CO2, cost


def _summary_df() -> pd.DataFrame:
    """
    A synthetic summary table over data years, weather years, resolutions,
    variants and DC sizes; the raw variant's baselines are offset.
    """
    rows: list[dict[str, object]] = []
    for planning_year in (2024, 2025):
        for weather_year in (2010, 2013, 2020):
            for resolution in ("1h", "3h"):
                for variant, (co2_offset, cost_offset) in VARIANT_OFFSET.items():
                    base_co2 = 20.0 + (weather_year - 2010) * 0.1 + co2_offset
                    base_cost = 1000.0 + (planning_year - 2024) * 50.0 + cost_offset
                    key = {
                        "planning_year": planning_year,
                        "weather_year": weather_year,
                        "resolution": resolution,
                        "variant": variant,
                        "excess_cap": 0.0,
                    }
                    rows.append(
                        {
                            **key,
                            "dc_mw": 0,
                            "policy": "no_dc",
                            "emissions": base_co2,
                            "objective": base_cost,
                        }
                    )
                    for dc_mw in (500, 1000, 2000, 4000, 6000):
                        unm_cost = base_cost + 0.5 * dc_mw
                        rows += [
                            {
                                **key,
                                "dc_mw": dc_mw,
                                "policy": "unmatched_dc",
                                "emissions": base_co2 + 0.002 * dc_mw,
                                "objective": unm_cost,
                            },
                            {
                                **key,
                                "dc_mw": dc_mw,
                                "policy": "hourly_80_grid_credit",
                                "emissions": base_co2 + 0.0004 * dc_mw,
                                "objective": unm_cost + 0.1 * dc_mw,
                            },
                        ]
    return pd.DataFrame(rows)


def test_unmatched_delta_zero() -> None:
    """unmatched_dc rows have exactly zero delta against themselves."""
    out = attach_baseline_deltas(_summary_df())
    unm = out.loc[out["policy"] == "unmatched_dc"]
    assert len(unm) == 2 * 3 * 2 * 2 * 5
    for col in (
        "delta_unmatched_dc_emissions",
        "delta_unmatched_dc_objective",
        "dc_co2_reduction_vs_unmatched_pct",
    ):
        assert np.allclose(unm[col].to_numpy(), 0.0, atol=1e-12), col


@pytest.mark.parametrize("variant", list(VARIANT_OFFSET))
def test_deltas_use_their_own_baselines(variant: str) -> None:
    """Each row is compared with the baselines of its own variant and DC size."""
    out = attach_baseline_deltas(_summary_df())
    assert "delta_no_dc_excess_cap" not in out.columns  # identifiers get no deltas
    row = out.loc[
        (out["planning_year"] == 2025)
        & (out["weather_year"] == 2013)
        & (out["resolution"] == "1h")
        & (out["variant"] == variant)
        & (out["dc_mw"] == 4000)
        & (out["policy"] == "hourly_80_grid_credit")
    ].iloc[0]
    # vs no_dc: 0.0004 * 4000 = 1.6 Mt CO2, (0.5 + 0.1) * 4000 = 2400 cost
    assert row["delta_no_dc_emissions"] == pytest.approx(1.6, abs=1e-9)
    assert row["delta_no_dc_objective"] == pytest.approx(2400.0, abs=1e-9)
    # vs unmatched_dc: (0.0004 - 0.002) * 4000 = -6.4 Mt CO2, 0.1 * 4000 = 400 cost
    assert row["delta_unmatched_dc_emissions"] == pytest.approx(-6.4, abs=1e-9)
    assert row["delta_unmatched_dc_objective"] == pytest.approx(400.0, abs=1e-9)
    # abatement vs unmatched_dc: (0.002 - 0.0004) / 0.002 = 80%
    assert row["dc_co2_reduction_vs_unmatched_pct"] == pytest.approx(80.0, abs=1e-9)


@pytest.mark.parametrize("key", ["weather_year", "variant", "dc_mw"])
def test_missing_key_column_fails_fast(key: str) -> None:
    with pytest.raises(KeyError):
        attach_baseline_deltas(_summary_df().drop(columns=[key]))
