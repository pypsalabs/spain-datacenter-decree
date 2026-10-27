# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Supplementary figure across data center sizes: the system-impact panels by size.

Reads the scenario summary table of a suite with several DC sizes and writes the
Spanish system impact (system cost, CO2, gas for power, spot price) with the sizes
on the x-axis and, per size, no policy, hourly 80% on average and the decree side
by side. Policies, colours, cost split and caption style come from
``plot_summary.py``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scripts.plot_summary import (  # noqa: E402
    DC_COSTS,
    DECREE,
    FORMATS,
    GRID_COLOR,
    POLICIES,
    STYLE,
    _caption,
    _cost_parts,
    _percent,
    _signed,
    _zero_line,
    select_rows,
    site_name,
)

logger = logging.getLogger(__name__)

# policies shown per size, in plotting order; a second target of one family keeps
# the family colour and is hatched instead (the per-size figures use the axis tick)
SIZE_POLICIES = ["unmatched_dc", "hourly_80_ppa_only", "hourly_90_ppa_only", DECREE]
HATCH = {"hourly_90_ppa_only": "///"}
# highlighted policies: a light band in this colour behind their bars
HIGHLIGHT = {DECREE: POLICIES[DECREE].color, "hourly_90_ppa_only": "#009E73"}
SLOT_IN = 0.55  # figure width per bar slot (inches): room for the two-line labels
BAR = 0.8  # bar width in slots: the policies of one size sit close together
GAP = 0.3  # extra slots between sizes
SYSTEM = "Extra fuel and running costs of the Spanish electricity system"
# panels: key, axis label, summary column, scale, label decimals
# (labels mixing math write Delta as math: their text gets no font fallback)
PANELS: list[tuple[str, str, str, float, int]] = [
    ("cost", "Δ system cost\n(M€/yr)", "es_system_cost_eur", 1e-6, 0),
    ("co2", "$\\Delta$ CO$_2$ emissions\n(Mt/yr)", "es_total_co2_mt", 1.0, 2),
    ("gas", "$\\Delta$ gas for power\n(TWh$_{th}$/yr)", "es_gas_fuel_mwh_th", 1e-6, 2),
    ("price", "Δ spot price\n(€/MWh)", "es_mean_price_eur_mwh", 1.0, 2),
]


def _vs(v: float, ref: float) -> str:
    """Difference of ``v`` to ``ref`` in percent of ``ref``, e.g. ``(−23%)``."""
    return _percent(v - ref, abs(ref))


def _box(edge: str | list[str]) -> dict:
    """Uncoloured box hatched in ``edge``: the Spanish system's part of the cost."""
    return {"facecolor": "white", "edgecolor": edge, "hatch": "////", "linewidth": 0.6}


def _hatch(bars, hatches: list[str]) -> None:
    """White hatch on the bars whose policy asks for one."""
    for bar, hatch in zip(bars, hatches, strict=True):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")


def _label(ax: plt.Axes, x: float, y: float, up: bool, text: str, *notes: str) -> None:
    """Bar label at ``(x, y)``: ``text`` and, under it in grey, one line per note."""
    lines = [text, *notes]
    for k, line in enumerate(lines):
        offset = 7.0 * (len(lines) - 1 - k) if up else -7.0 * k
        ax.annotate(
            line,
            (x, y),
            xytext=(0.0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if up else "top",
            fontsize=6 if k == 0 else 5.5,
            color="#000000" if k == 0 else "#777777",
        )


def plot_system_impact_by_size(
    summary: pd.DataFrame, excess_cap: float, weather_year: int
) -> plt.Figure:
    """Spanish system impact against no DC by data center size, policies side by side."""
    sizes = sorted(s for s in summary["dc_mw"].unique() if s > 0.0)
    frames = {s: select_rows(summary, excess_cap, weather_year, s) for s in sizes}
    policies = [p for p in SIZE_POLICIES if all(p in f.index for f in frames.values())]
    n = len(policies)
    pos = np.array([i * (n + GAP) + j for i in range(len(sizes)) for j in range(n)])
    colors = [POLICIES[p].color for p in policies] * len(sizes)
    hatches = [HATCH.get(p, "") for p in policies] * len(sizes)
    is_ref = np.tile([p == "unmatched_dc" for p in policies], len(sizes))
    band_colors = [HIGHLIGHT.get(p) for p in policies] * len(sizes)
    fig, axd = plt.subplot_mosaic(
        [[p[0]] for p in PANELS],
        figsize=(1.1 + SLOT_IN * (pos[-1] + 1.0), 6.4),
        sharex=True,
    )
    for key, label, col, scale, decimals in PANELS:
        ax = axd[key]
        for x, color in zip(pos, band_colors, strict=True):
            if color:
                ax.axvspan(x - 0.5, x + 0.5, color=color, alpha=0.08, lw=0, zorder=0)
        vals = scale * np.concatenate(
            [
                frames[s].loc[policies, f"delta_no_dc_{col}"].to_numpy(float)
                for s in sizes
            ]
        )
        ref = np.repeat(vals[is_ref], n)
        notes = [
            [] if skip else [_vs(v, r)]
            for v, r, skip in zip(vals, ref, is_ref, strict=True)
        ]
        if key == "cost":
            # stacked: the Spanish system's part as a hatched box, the DC costs on top
            parts = pd.concat([_cost_parts(frames[s].loc[policies]) for s in sizes])
            system = parts.drop(columns=DC_COSTS).iloc[:, 0].to_numpy()
            _hatch(
                ax.bar(
                    pos, parts[DC_COSTS], BAR, bottom=system, color=colors, linewidth=0
                ),
                hatches,
            )
            ax.bar(pos, system, BAR, **_box(colors), zorder=2.2)
            cost_legend = [
                Patch(**_box(GRID_COLOR), label=SYSTEM),
                Patch(color=GRID_COLOR, label=DC_COSTS),
            ]
            tops, bottom = np.maximum(vals, system), system.min()
        else:
            _hatch(ax.bar(pos, vals, width=BAR, color=colors, linewidth=0), hatches)
            tops, bottom = vals, vals.min()
        _zero_line(ax)
        lo, hi = min(0.0, bottom), max(0.0, tops.max())
        pad = 0.05 * (hi - lo)
        for x, v, top, note in zip(pos, vals, tops, notes, strict=True):
            y = (top if v >= 0.0 else v) + np.sign(v) * pad
            _label(ax, x, y, v >= 0.0, _signed(v, decimals), *note)
        ax.set_ylim(
            lo - (0.4 * (hi - lo) if vals.min() < 0.0 else 0.0),
            hi + (0.4 * (hi - lo) if hi > 0.0 else pad),
        )
        ax.set_ylabel(label)
        ax.set_xlim(pos[0] - 0.55, pos[-1] + 0.55)
        top = ax.get_ylim()[1]
        # no ticks or gridlines below zero: negative bars hang below the baseline
        ax.set_yticks([t for t in ax.get_yticks() if 0.0 <= t <= top])
    centers = [pos[i * n : (i + 1) * n].mean() for i in range(len(sizes))]
    axd[PANELS[-1][0]].set_xticks(centers, [f"{s:,.0f} MW" for s in sizes])
    # one legend, left-aligned: the policies on the first row, the cost parts below
    policy_legend = [
        Patch(
            facecolor=POLICIES[p].color,
            edgecolor="white",
            hatch=HATCH.get(p, ""),
            label=POLICIES[p].name,
        )
        for p in policies
    ]
    # legends fill column by column: interleave the rows, padding the second one
    blank = Patch(alpha=0.0, label="")
    second = cost_legend + [blank] * (n - len(cost_legend))
    entries = [h for pair in zip(policy_legend, second, strict=True) for h in pair]
    fig.legend(handles=entries, loc="outside upper left", ncol=n)
    site = site_name(frames[sizes[0]].at["unmatched_dc", "sites"])
    _caption(
        fig,
        f"Change in the Spanish power system with a data center of {sizes[0]:,.0f} to "
        f"{sizes[-1]:,.0f} MW in {site} against a baseline without it, by size and "
        "procurement policy: each bar carries the change and, in grey, its difference "
        "to the no-policy case of the same size. System cost: fuel, CO2 (EU ETS) and "
        "running costs of the Spanish electricity system (hatched; imports excluded), "
        "then the data center's PPA and battery annuities on top (solid), ending at the "
        "total. CO2: fuel "
        "burnt in Spanish plants times its carbon content. Gas: fuel burnt in Spanish "
        "gas-fired power plants (CCGT, OCGT). Spot price: hourly nodal prices weighted "
        "by Spanish demand. PPAs anywhere in mainland Spain.",
        fontsize=float(plt.rcParams["legend.fontsize"]),
    )
    return fig


FIGURES: dict[str, Callable[[pd.DataFrame, float, int], plt.Figure]] = {
    "system_impact_by_size": plot_system_impact_by_size,
}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: write the cross-size figures of one run and sales cap."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--excess-cap", type=float, required=True)
    parser.add_argument("--weather-year", type=int, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--format", choices=FORMATS, default="pdf")
    args = parser.parse_args(argv)

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.style.use(STYLE)
    summary = pd.read_csv(args.summary)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in FIGURES.items():
        fig = plot(summary, args.excess_cap, args.weather_year)
        fig.savefig(args.plots_dir / f"{name}.{args.format}")
        plt.close(fig)
        logger.info("Wrote %s", args.plots_dir / name)


if __name__ == "__main__":
    main()
