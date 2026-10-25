# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Plot the hourly CFE score and CO2 intensity of one scenario's DC consumption.

Reads the hourly series from ``summarise_network.py`` and the scenario's summary
row, and writes one PDF with two heatmaps on the same time of day by day grid.
The colour scales are fixed, so heatmaps of different policies compare directly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.plot_summary import (  # noqa: E402
    DECREE,
    POLICIES,
    SPAIN_WIDE,
    STYLE,
    _caption,
    scenario_key,
    site_name,
)

logger = logging.getLogger(__name__)

# shared by every policy; above the highest hourly emission factor of Spanish supply
INTENSITY_MAX_G_PER_KWH = 250.0


def time_of_day_grid(s: pd.Series) -> pd.DataFrame:
    """Snapshot series as a table of time of day (rows, hours) by day (columns)."""
    return s.groupby([s.index.hour, s.index.normalize()]).mean().unstack()


def plot_heatmap(hourly: pd.DataFrame, row: pd.Series) -> plt.Figure:
    """CFE score (a) and attributional CO2 intensity (b) of DC consumption."""
    panels = {
        "cfe": ("cfe_score", 100.0, "CFE score (%)", "viridis", 100.0),
        "intensity": (
            "intensity_g_per_kwh",
            1.0,
            "Emission intensity\n(gCO$_2$/kWh)",
            "magma_r",
            INTENSITY_MAX_G_PER_KWH,
        ),
    }
    fig, axd = plt.subplot_mosaic(
        [[key] for key in panels], figsize=(7.2, 5.4), sharex=True, sharey=True
    )
    for letter, (key, (col, scale, label, cmap, vmax)) in zip(
        "ab", panels.items(), strict=True
    ):
        grid = time_of_day_grid(hourly[col] * scale)
        ax = axd[key]
        image = ax.imshow(
            grid,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(0, grid.shape[1], 0, 24),
        )
        extend = "max" if grid.to_numpy().max() > vmax else "neither"
        fig.colorbar(image, ax=ax, label=label, extend=extend)
        ax.grid(False)
        ax.set_yticks(range(0, 25, 6))
        ax.set_ylabel("Time of day (h, UTC)")
        ax.set_title(letter, loc="left", fontweight="bold")
    days = grid.columns
    starts = np.flatnonzero(days.day == 1)
    ax.set_xticks(starts, days[starts].strftime("%b"))

    key = scenario_key(row["policy"], row["siting"])
    name = ("Decree: " if key == DECREE else "") + POLICIES[key].name
    fig.suptitle(name, x=0.0, ha="left", fontweight="bold")
    step = 24 // len(grid.index)
    site = site_name(row["sites"])
    local = "" if row["siting"] == SPAIN_WIDE else f" Local: PPAs in {site} only."
    _caption(
        fig,
        f"Hourly CFE score (a) and emission intensity (b) of the consumption of a "
        f"{row['dc_mw']:.0f} MW data center in {site}, one cell per {step}-hour "
        f"snapshot (UTC).{local} CFE score: the share of consumption covered by the data center's "
        "PPAs and battery plus the renewable share of its grid purchases "
        "(import-traced, nuclear excluded); annual mean "
        f"{100.0 * row['clean_coverage']:.0f}%. Emission intensity: its grid "
        "purchases valued at the hourly "
        "emission factor of Spanish supply (imports traced to their exporter's mix), "
        "per kWh consumed; annual mean "
        f"{row['dc_attributional_intensity_g_per_kwh']:.0f} gCO2/kWh.",
    )
    return fig


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: write the heatmap of one scenario (format from the file suffix)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.style.use(STYLE)
    hourly = pd.read_csv(args.hourly, index_col=0, parse_dates=True)
    row = pd.read_csv(args.summary).iloc[0]
    fig = plot_heatmap(hourly, row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
