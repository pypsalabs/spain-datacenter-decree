# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""End-to-end toy test for the study figures: solve, summarise, collect, plot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.data_center import attach_data_center
from scripts.make_summary import attach_baseline_deltas
from scripts.plot_summary import (
    FIGURES,
    FORMATS,
    POLICIES,
    main,
    reference_rows,
    select_rows,
)
from scripts.solve_network import solve_network
from scripts.summarise_network import summarise_network

COLORS = Path(__file__).resolve().parents[1] / "pypsa-eur/config/plotting.default.yaml"
META = {
    "planning_year": 2025,
    "weather_year": 2024,
    "resolution": "1h",
    "variant": "patched",
    "sites": "aragon",
}


def _split(key: str) -> tuple[str, str]:
    """(policy, siting) of a ``POLICIES`` key."""
    policy, _, siting = key.partition("@")
    return policy, siting or "spain_wide"


@pytest.fixture
def toy_summary(toy_base, toy_costs, tmp_path: Path) -> Path:
    rows = []
    for policy, siting in map(_split, ["no_dc", *POLICIES]):
        n = toy_base.copy()
        dc_mw = 0.0 if policy == "no_dc" else 100.0
        attach_data_center(
            n, dc_mw=dc_mw, siting=siting, costs=toy_costs, policy=policy
        )
        cfe0 = 0.3 if "grid_credit" in policy else None
        n, _, metrics = solve_network(
            n, policy=policy, excess_share=0.2, cfe0_source=cfe0
        )
        meta = {**META, "dc_mw": dc_mw, "siting": siting}
        rows.append(summarise_network(n, pd.Series({**meta, **metrics})))
    path = tmp_path / "scenarios.csv"
    attach_baseline_deltas(pd.DataFrame(rows)).to_csv(path, index=False)
    return path


@pytest.mark.skipif(not COLORS.exists(), reason="pypsa-eur submodule not checked out")
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_figure_renders(toy_summary: Path, tmp_path: Path, fmt: str) -> None:
    plots = tmp_path / "plots"
    main(
        [
            "--summary",
            str(toy_summary),
            "--colors",
            str(COLORS),
            "--excess-cap",
            "0.2",
            "--weather-year",
            "2024",
            "--dc-mw",
            "100",
            "--plots-dir",
            str(plots),
            "--format",
            fmt,
        ]
    )
    for name in FIGURES:
        assert (plots / f"{name}.{fmt}").stat().st_size > 0


def test_multi_run_suite_fails_fast() -> None:
    df = pd.DataFrame({"policy": ["no_dc", "no_dc"], "siting": "spain_wide"})
    with pytest.raises(ValueError, match="single-run"):
        reference_rows(df)


@pytest.mark.parametrize("sigma", [0.0, 0.2])
@pytest.mark.parametrize("dc_mw", [500.0, 2000.0])
def test_select_rows_takes_one_size_and_cap_and_shared_baselines(
    sigma: float, dc_mw: float
) -> None:
    keys = [k for k in POLICIES if k != "unmatched_dc"]
    rows = [("no_dc", "spain_wide", 0.0, 0.0)]
    for size in (500.0, 2000.0):
        rows += [("unmatched_dc", "spain_wide", 0.0, size)]
        rows += [(*_split(k), cap, size) for k in keys for cap in (0.0, 0.2)]
    df = pd.DataFrame(rows, columns=["policy", "siting", "excess_cap", "dc_mw"])
    df = df.assign(weather_year=2024)
    other_year = df.assign(weather_year=2013)
    selected = select_rows(pd.concat([df, other_year]), sigma, 2024, dc_mw)
    assert sorted(selected.index) == sorted(["no_dc", *POLICIES])
    assert (selected.loc[keys, "excess_cap"] == sigma).all()
    assert (selected.loc[list(POLICIES), "dc_mw"] == dc_mw).all()
