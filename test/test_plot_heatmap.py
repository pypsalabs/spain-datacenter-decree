# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Toy tests for the hourly heatmaps of DC consumption."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.data_center import attach_data_center
from scripts.plot_heatmap import main, time_of_day_grid
from scripts.solve_network import solve_network
from scripts.summarise_network import dc_hourly, summarise_network


def test_time_of_day_grid_places_each_snapshot() -> None:
    snapshots = pd.date_range("2024-01-01", periods=16, freq="3h")
    grid = time_of_day_grid(pd.Series(range(16), index=snapshots, dtype=float))
    assert grid.shape == (8, 2)
    assert list(grid.index) == list(range(0, 24, 3))
    assert grid.loc[6, pd.Timestamp("2024-01-02")] == 10.0


def test_heatmap_renders(toy_base, toy_costs, tmp_path: Path) -> None:
    policy = "strict_hourly_80_ppa_only"
    attach_data_center(toy_base, dc_mw=100.0, costs=toy_costs, policy=policy)
    n, _, metrics = solve_network(toy_base, policy=policy)
    row = summarise_network(
        n, pd.Series({"sites": "aragon", "dc_mw": 100.0, **metrics})
    )
    summary, hourly = tmp_path / "summary.csv", tmp_path / "hourly.csv"
    row.to_frame().T.to_csv(summary, index=False)
    dc_hourly(n).to_csv(hourly)
    output = tmp_path / "plots" / f"heatmap_{policy}.pdf"
    main(["--hourly", str(hourly), "--summary", str(summary), "--output", str(output)])
    assert output.stat().st_size > 0
