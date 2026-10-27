# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""Toy test for the cross-size supplementary figure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_plot_summary import toy_summary  # noqa: F401

from scripts.plot_by_size import FIGURES, main


def test_cross_size_figures_render(toy_summary: Path, tmp_path: Path) -> None:  # noqa: F811
    one = pd.read_csv(toy_summary)
    two = one[one["dc_mw"] > 0.0].assign(dc_mw=200.0)
    summary = tmp_path / "sizes.csv"
    pd.concat([one, two]).to_csv(summary, index=False)
    plots = tmp_path / "by_size"
    args = ["--summary", str(summary), "--excess-cap", "0.2", "--weather-year", "2024"]
    main([*args, "--plots-dir", str(plots)])
    for name in FIGURES:
        assert (plots / f"{name}.pdf").stat().st_size > 0
