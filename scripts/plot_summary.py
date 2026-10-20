# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Plot the study figures from the scenario summary table.

Reads ``results/scenarios.csv`` from ``make_summary.py`` and writes one PDF per
figure. The figures cover a single-run suite (one row per policy): hourly
clean coverage of DC consumption, the DC's PPA portfolio and net cost, the Spanish
system impact, the shift in Spanish supply, and the CO2 intensity of DC
consumption. Carrier colours come from PyPSA-Eur's ``tech_colors``.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

logger = logging.getLogger(__name__)

STYLE = Path(__file__).with_name("plots.mplstyle")
FORMATS = ("pdf",)

# policy -> (label, Okabe-Ito colour), in order of policy relevance
POLICIES: dict[str, tuple[str, str]] = {
    "unmatched_dc": ("No policy", "#999999"),
    "strict_hourly_80_ppa_only": ("Decree\nCFE 80%", "#D55E00"),
    "hourly_80_ppa_only": ("Hourly\n80%", "#0072B2"),
    "hourly_80_grid_credit": ("Hourly 80%\n+ grid credit", "#56B4E9"),
    "annual_100_res": ("Annual\n100%", "#009E73"),
}
DECREE = "strict_hourly_80_ppa_only"
BASELINE = "no_dc"

PPA_CARRIERS: dict[str, str] = {
    "onwind": "Onshore wind",
    "solar": "Solar PV",
    "solar-hsat": "Solar PV (tracking)",
}
PPA_COLOR = "#01665e"
GRID_RES_COLOR = "#80cdc1"
GRID_COLOR = "#7f7f7f"
SALES_COLOR = "#bdbdbd"
IMPORT_COLOR = "#8c8c8c"

# Spanish supply groups -> (network carriers, tech_colors key)
SUPPLY_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "Nuclear": (("nuclear",), "nuclear"),
    "Hydro": (("hydro", "ror"), "hydro"),
    "Pumped hydro": (("PHS",), "PHS"),
    "Wind": (("onwind", "offwind-ac", "offwind-dc", "offwind-float"), "onwind"),
    "Solar": (("solar", "solar-hsat", "csp"), "solar"),
    "Gas": (("CCGT", "OCGT"), "CCGT"),
    "CHP": (("CHP",), "CHP"),
    "Coal": (("coal", "lignite"), "coal"),
    "Oil": (("oil",), "oil"),
    "Biomass & waste": (("biomass", "waste"), "biomass"),
}
MIN_SUPPLY_SHIFT_TWH = 0.02
CAPTION_CHARS_PER_INCH = 21.0
HOURS_PER_YEAR = 8760.0
G_PER_KWH_PER_MT_PER_MWH = 1.0e12 / 1.0e3


def load_colors(path: Path) -> dict[str, str]:
    """PyPSA-Eur ``plotting.tech_colors`` from its plotting config."""
    with open(path) as f:
        return yaml.safe_load(f)["plotting"]["tech_colors"]


def reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a single-run suite indexed by policy; fails on multi-run suites."""
    if df["policy"].duplicated().any():
        raise ValueError(
            "plot_summary covers single-run suites (one row per policy); "
            "the full suite needs its own figures."
        )
    missing = (set(POLICIES) | {BASELINE}) - set(df["policy"])
    if missing:
        raise ValueError(f"Summary lacks policies {sorted(missing)}.")
    return df.set_index("policy").fillna(0.0)


def select_rows(df: pd.DataFrame, excess_cap: float, weather_year: int) -> pd.DataFrame:
    """One run's rows: the policies at ``excess_cap`` plus the shared baselines."""
    run = df[df["weather_year"] == weather_year]
    baseline = run["policy"].isin([BASELINE, "unmatched_dc"])
    return reference_rows(run[baseline | np.isclose(run["excess_cap"], excess_cap)])


def _policy_axis(ax: plt.Axes, policies: list[str], axis: str = "x") -> np.ndarray:
    """Categorical policy axis with the decree highlighted."""
    pos = np.arange(len(policies))
    labels = [POLICIES[p][0] for p in policies]
    i = policies.index(DECREE)
    band = {"color": POLICIES[DECREE][1], "alpha": 0.08, "linewidth": 0, "zorder": 0}
    if axis == "x":
        ax.set_xticks(pos, labels)
        ax.axvspan(i - 0.5, i + 0.5, **band)
        ticks = ax.get_xticklabels()
    else:
        ax.set_yticks(pos, labels)
        ax.axhspan(i - 0.5, i + 0.5, **band)
        ticks = ax.get_yticklabels()
    if ticks:  # inner panels of a shared axis carry no labels
        ticks[i].set_fontweight("bold")
    return pos


def _stack(
    ax: plt.Axes,
    pos: np.ndarray,
    parts: pd.DataFrame,
    styles: dict[str, dict],
    width: float = 0.62,
) -> np.ndarray:
    """Stack the columns of ``parts`` as bars, negatives downward; return the tops."""
    up = np.zeros(len(pos))
    down = np.zeros(len(pos))
    for col in parts.columns:
        v = parts[col].to_numpy(dtype=float)
        base = np.where(v >= 0.0, up, down)
        shown = np.where(v != 0.0, v, np.nan)  # zero-height bars render as hairlines
        ax.bar(
            pos, shown, width, bottom=base, linewidth=0, **{"label": col, **styles[col]}
        )
        up += np.where(v >= 0.0, v, 0.0)
        down += np.where(v < 0.0, v, 0.0)
    return up


def _span(values: pd.Series, decimals: int = 0) -> str:
    """'lo–hi' of ``values`` as rounded, or one number when both ends round alike."""
    lo, hi = f"{values.min():.{decimals}f}", f"{values.max():.{decimals}f}"
    return lo if lo == hi else f"{lo}–{hi}"


def _signed(v: float, decimals: int) -> str:
    return f"{v:+.{decimals}f}".replace("-", "−")


def _legend(fig: plt.Figure, handles: list, labels: list[str], ncol: int) -> None:
    """Framed multi-column legend above the plot area, bar series first."""
    entries = sorted(
        zip(handles, labels, strict=True),
        key=lambda e: not isinstance(e[0], BarContainer),
    )
    fig.legend(*zip(*entries, strict=True), loc="outside upper left", ncol=ncol)


def _caption(fig: plt.Figure, summary: str, points: list[str]) -> None:
    """Caption under the figure: what is shown, then the key mechanics as bullets."""
    width = int(CAPTION_CHARS_PER_INCH * fig.get_size_inches()[0])
    lines = textwrap.wrap(summary, width)
    for point in points:
        lines += textwrap.wrap(
            point, width, initial_indent="• ", subsequent_indent="   "
        )
    fig.supxlabel("\n".join(lines), x=0.0, ha="left", fontsize=6.5, linespacing=1.35)


def plot_clean_coverage(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Hourly renewable coverage of DC consumption: PPA-matched plus grid renewables."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    pos = _policy_axis(ax, policies)
    parts = pd.DataFrame(
        {"PPA matched": r["ppa_only_score"], "Grid renewables": r["grid_clean_share"]}
    ).mul(100.0)
    seg_colors = {"PPA matched": PPA_COLOR, "Grid renewables": GRID_RES_COLOR}
    tops = _stack(ax, pos, parts, {k: {"color": c} for k, c in seg_colors.items()})
    for x, top in zip(pos, tops, strict=True):
        ax.text(x, top + 1.5, f"{top:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.axhline(
        80.0, color="black", linewidth=0.6, linestyle=(0, (4, 3)), label="80% target"
    )
    ax.set_ylim(0.0, 110.0)
    ax.set_ylabel("Hourly renewable share\nof DC consumption (%)")
    _legend(fig, *ax.get_legend_handles_labels(), ncol=3)

    coverage = r["clean_coverage"].mul(100.0)
    sold = r.at["annual_100_res", "excess_share"]
    _caption(
        fig,
        "Share of the data center's consumption covered by renewables in the same "
        "hour: by its own PPAs and battery (dark), or by the renewable share of its "
        "grid purchases (light; import-traced, nuclear excluded).",
        [
            f"No policy: the data center still gets the grid's renewable share, "
            f"{coverage['unmatched_dc']:.0f}%.",
            "Decree: 80% is a floor in every hour, and surplus in one hour cannot "
            "offset a shortfall in another. The portfolio is sized for the worst hours "
            "(winter lulls in wind and sun), so it fully covers most other hours: "
            f"{coverage[DECREE]:.0f}% on average.",
            "Hourly 80%: matched energy is summed over the year, so surplus hours "
            "offset short ones and the rule lands exactly on 80%.",
            f"Annual 100%: a full year of PPA volume, but {sold:.0%} of it arrives when "
            "the data center does not need it and is sold: hourly coverage ends at "
            f"{coverage['annual_100_res']:.0f}%, exactly as under hourly 80%.",
            "Grid credit: the grid's renewable share counts toward the 80%, so the "
            "PPA share alone stays below the line.",
        ],
    )
    return fig


def plot_portfolio(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """DC PPA and battery capacity, and PPA output with its curtailment."""
    policies = [p for p in POLICIES if p != "unmatched_dc"]
    r = rows.loc[policies]
    fig, axd = plt.subplot_mosaic([["capacity", "energy"]], figsize=(7.2, 3.9))
    carriers = {c: label for c, label in PPA_CARRIERS.items() if f"dc_ppa_{c}_mw" in r}
    carrier_colors = {label: colors[c] for c, label in carriers.items()}

    ax = axd["capacity"]
    pos = _policy_axis(ax, policies)
    cap = pd.DataFrame({label: r[f"dc_ppa_{c}_mw"] for c, label in carriers.items()})
    cap["Battery"] = r["dc_battery_mw"]
    cap = cap / 1.0e3
    cap_colors = {**carrier_colors, "Battery": colors["battery"]}
    tops = _stack(ax, pos, cap, {k: {"color": c} for k, c in cap_colors.items()})
    for x, top, mwh in zip(pos, tops, r["dc_battery_mwh"], strict=True):
        if mwh > 1.0:
            ax.text(
                x, top, f"{mwh / 1e3:.1f} GWh", ha="center", va="bottom", fontsize=6
            )
    ax.axhline(
        float(r["dc_mw"].iloc[0]) / 1.0e3,
        color="black",
        linewidth=0.6,
        linestyle=(0, (4, 3)),
    )
    ax.set_ylim(0.0, 1.15 * tops.max())
    ax.set_ylabel("PPA and battery capacity (GW)")
    ax.set_title("a", loc="left", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()

    ax = axd["energy"]
    pos = _policy_axis(ax, policies)
    used = pd.DataFrame({label: r[f"dc_ppa_{c}_mwh"] for c, label in carriers.items()})
    curtailed = pd.DataFrame(
        {
            f"{label} curtailed": r[f"dc_ppa_{c}_curtailed_mwh"]
            for c, label in carriers.items()
        }
    )
    energy = pd.concat([used, curtailed], axis=1) / 1.0e6
    hatched = {"facecolor": "white", "hatch": "/////", "label": "_nolegend_"}
    styles = {k: {"color": c, "label": "_nolegend_"} for k, c in carrier_colors.items()}
    styles |= {
        f"{k} curtailed": {**hatched, "edgecolor": c} for k, c in carrier_colors.items()
    }
    tops = _stack(ax, pos, energy, styles)
    load = float(r["total_load_mwh"].iloc[0]) / 1.0e6
    consumption = ax.axhline(load, color="black", linewidth=0.6, linestyle=(0, (4, 3)))
    ax.set_ylim(0.0, 1.15 * tops.max())
    ax.set_ylabel("PPA output (TWh/yr)")
    ax.set_title("b", loc="left", fontweight="bold")

    handles += [Patch(facecolor="white", edgecolor="0.3", hatch="/////"), consumption]
    labels += ["Curtailed", "DC load"]
    _legend(fig, handles, labels, ncol=len(labels))

    curtailed_share = (
        100.0 * curtailed.sum(axis=1) / (used.sum(axis=1) + curtailed.sum(axis=1))
    )
    _caption(
        fig,
        "The data center's contracted assets (a) and their output (b); hatched: "
        "output curtailed; dashed: the data center's load, as capacity (a) and "
        "annual consumption (b).",
        [
            "Decree: covering every hour takes solar plus a battery that carries it "
            "into the night, overbuilt for winter lulls: "
            f"{r.at[DECREE, 'dc_ppa_total_mw'] / 1e3:.1f} GW of PPAs and "
            f"{r.at[DECREE, 'dc_battery_mwh'] / 1e3:.1f} GWh of storage for a "
            f"{r.at[DECREE, 'total_load_mwh'] / HOURS_PER_YEAR:.0f} MW load.",
            "The other rules are met with mostly onshore wind, whose output also spans "
            "the night.",
            f"Sales are capped at {r['excess_share'].max():.0%} of consumption; output "
            f"beyond the cap is curtailed ({_span(curtailed_share)}% of PPA output).",
        ],
    )
    return fig


def plot_dc_cost(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """DC net cost per MWh consumed: PPAs, battery, grid purchases, minus sales."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    mwh = r["total_load_mwh"]
    parts = pd.DataFrame(
        {
            label: r[f"dc_ppa_{c}_cost_eur"] / mwh
            for c, label in PPA_CARRIERS.items()
            if f"dc_ppa_{c}_cost_eur" in r
        }
    )
    parts["Battery"] = r["dc_battery_cost_eur"] / mwh
    parts["Grid purchases"] = r["dc_grid_cost_eur"] / mwh
    parts["Sales revenue"] = -r["dc_sales_revenue_eur"] / mwh
    net = r["dc_net_cost_eur_per_mwh"]
    if not np.allclose(parts.sum(axis=1), net, rtol=1e-6, atol=1e-3):
        raise ValueError("DC cost components do not add up to the net cost.")

    seg_colors = {label: colors[c] for c, label in PPA_CARRIERS.items()}
    seg_colors |= {
        "Battery": colors["battery"],
        "Grid purchases": GRID_COLOR,
        "Sales revenue": SALES_COLOR,
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    pos = _policy_axis(ax, policies)
    tops = _stack(ax, pos, parts, {k: {"color": seg_colors[k]} for k in parts})
    ax.hlines(
        net,
        pos - 0.31,
        pos + 0.31,
        color="black",
        linewidth=1.6,
        zorder=4,
        label="Net cost",
    )
    for x, v in zip(pos, net, strict=True):
        ax.text(x + 0.34, v, f"{v:.0f}", va="center", fontsize=6.5, fontweight="bold")
    ax.axhline(0.0, color="black", linewidth=0.6)
    bottom = parts.where(parts < 0.0, 0.0).sum(axis=1).min()
    ax.set_ylim(1.25 * bottom, 1.1 * tops.max())
    ax.set_ylabel("DC cost (€ per MWh consumed)")
    _legend(fig, *ax.get_legend_handles_labels(), ncol=4)

    others = net.drop(["unmatched_dc", DECREE])
    _caption(
        fig,
        "The data center's cost per MWh it consumes: PPA and battery annuities and "
        "grid purchases, minus revenue from selling PPA surplus (bold line: net cost).",
        [
            "No policy: the data center buys every MWh at the modelled hourly Spanish "
            f"spot price; {net['unmatched_dc']:.0f} €/MWh is the annual average of "
            "these hourly purchases.",
            "Annual and hourly rules: contracted wind costs less than the spot price, "
            f"so matching roughly halves the bill ({_span(others)} €/MWh).",
            "Decree: the solar overbuild and battery cost about as much as buying from "
            f"the grid ({net[DECREE]:.0f} €/MWh).",
        ],
    )
    return fig


def plot_system_impact(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Spanish system impact against no DC; hollow markers trace the whole model."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    panels = [
        ("cost", "Δ system cost\n(M€/yr)", "es_system_cost_eur", 1e-6, "objective", 0),
        ("co2", "Δ CO$_2$ emissions\n(Mt/yr)", "es_total_co2_mt", 1.0, "emissions", 2),
        (
            "intensity",
            "Δ CO$_2$ intensity\n(g/kWh)",
            "es_co2_intensity_g_per_kwh",
            1.0,
            None,
            1,
        ),
        (
            "gas",
            "Δ gas for power\n(TWh$_{th}$/yr)",
            "es_gas_fuel_mwh_th",
            1e-6,
            None,
            2,
        ),
        ("price", "Δ spot price\n(€/MWh)", "es_mean_price_eur_mwh", 1.0, None, 2),
    ]
    fig, axd = plt.subplot_mosaic(
        [[p[0] for p in panels]], figsize=(7.2, 3.9), sharey=True
    )
    bar_colors = [POLICIES[p][1] for p in policies]
    for key, label, col, scale, tracer, decimals in panels:
        ax = axd[key]
        pos = _policy_axis(ax, policies, axis="y")
        vals = r[f"delta_no_dc_{col}"].to_numpy(dtype=float) * scale
        ax.barh(pos, vals, height=0.62, color=bar_colors, linewidth=0)
        ax.axvline(0.0, color="black", linewidth=0.6)
        whole = vals
        if tracer is not None:
            whole = r[f"delta_no_dc_{tracer}"].to_numpy(dtype=float) * scale
            ax.scatter(
                whole,
                pos,
                marker="o",
                s=14,
                facecolor="none",
                edgecolor="black",
                linewidth=0.8,
                zorder=4,
            )
        lo = min(0.0, vals.min(), whole.min())
        hi = max(0.0, vals.max(), whole.max())
        pad = 0.05 * (hi - lo)
        for y, v, t in zip(pos, vals, whole, strict=True):
            edge = max(v, t) if v >= 0.0 else min(v, t)
            ax.text(
                edge + np.sign(v) * pad,
                y,
                _signed(v, decimals),
                va="center",
                ha="left" if v >= 0.0 else "right",
                fontsize=6,
            )
        ax.set_xlim(
            lo - (0.45 * (hi - lo) if lo < 0.0 else pad),
            hi + (0.45 * (hi - lo) if hi > 0.0 else pad),
        )
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    axd["cost"].invert_yaxis()
    handles = [
        Patch(facecolor=GRID_COLOR),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=4,
        ),
    ]
    _legend(fig, handles, ["Spanish system", "Whole model (ES, FR, PT)"], ncol=2)

    def delta(col: str, policy: str, scale: float = 1.0) -> float:
        return float(r.at[policy, f"delta_no_dc_{col}"]) * scale

    no = "unmatched_dc"
    _caption(
        fig,
        "Change in the Spanish power system against no data center (bars); hollow "
        "markers: the whole model, including France and Portugal.",
        [
            "No policy: the new load is served largely by gas: "
            f"{_signed(delta('es_gas_fuel_mwh_th', no, 1e-6), 1)} TWh of gas and "
            f"{_signed(delta('es_total_co2_mt', no), 2)} Mt CO2 a year in Spain; the "
            "whole model adds more, as part of the load is served by imports.",
            "Every matching rule more than offsets the data center's emissions: PPA "
            "output sold to the grid displaces gas.",
            "System cost: the annual and hourly rules add about half the cost of "
            "serving the load from the grid; the decree adds slightly more than that "
            f"({_signed(delta('es_system_cost_eur', DECREE, 1e-6), 0)} vs. "
            f"{_signed(delta('es_system_cost_eur', no, 1e-6), 0)} M€/yr).",
            "Spot price: only the unmatched load lifts it noticeably "
            f"({_signed(delta('es_mean_price_eur_mwh', no), 1)} €/MWh): new demand "
            "without new supply raises the price for every consumer.",
        ],
    )
    return fig


def plot_generation_shift(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Change in Spanish supply against no DC: what serves the DC's consumption."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    ppa = [f"dc_ppa_{c}_mwh" for c in PPA_CARRIERS if f"dc_ppa_{c}_mwh" in r]
    parts = pd.DataFrame({"DC PPAs": r[ppa].sum(axis=1)})
    seg_colors = {"DC PPAs": PPA_COLOR}
    for name, (carriers, key) in SUPPLY_GROUPS.items():
        cols = [
            f"delta_no_dc_es_gen_{c}_mwh"
            for c in carriers
            if f"delta_no_dc_es_gen_{c}_mwh" in r
        ]
        if cols:
            parts[name] = r[cols].sum(axis=1)
            seg_colors[name] = colors[key]
    parts["Net imports"] = r["delta_no_dc_es_net_import_mwh"]
    seg_colors["Net imports"] = IMPORT_COLOR
    parts = parts / 1.0e6
    parts = parts.loc[:, parts.abs().max() >= MIN_SUPPLY_SHIFT_TWH]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pos = _policy_axis(ax, policies)
    _stack(ax, pos, parts, {k: {"color": seg_colors[k]} for k in parts})
    ax.axhline(0.0, color="black", linewidth=0.6)
    load = float(r["total_load_mwh"].iloc[0]) / 1.0e6
    ax.axhline(
        load,
        color="black",
        linewidth=0.6,
        linestyle=(0, (4, 3)),
        label="DC consumption",
    )
    ax.set_ylabel("Δ Spanish supply vs. no DC (TWh/yr)")
    _legend(fig, *ax.get_legend_handles_labels(), ncol=5)

    no = parts.loc["unmatched_dc"]
    _caption(
        fig,
        "Change in Spanish electricity supply against no data center; dashed line: "
        "the data center's consumption.",
        [
            f"No policy: the extra load is served by gas ({no.get('Gas', 0.0):.1f} TWh), "
            f"more nuclear output ({no.get('Nuclear', 0.0):.1f} TWh) and imports "
            f"({no.get('Net imports', 0.0):.1f} TWh).",
            "With matching, the data center's PPAs supply its load. Bars above the "
            "dashed line exceed its load: that surplus is PPA output sold to the grid, "
            "which displaces gas (below zero).",
        ],
    )
    return fig


def plot_dc_emission_intensity(
    rows: pd.DataFrame, colors: dict[str, str]
) -> plt.Figure:
    """CO2 intensity of DC consumption: caused (consequential) vs. claimed (attributional)."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    caused = (
        r["delta_no_dc_emissions"] / r["total_load_mwh"] * G_PER_KWH_PER_MT_PER_MWH
    ).to_numpy(dtype=float)
    claimed = r["dc_attributional_intensity_g_per_kwh"].to_numpy(dtype=float)
    point_colors = [POLICIES[p][1] for p in policies]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    pos = _policy_axis(ax, policies, axis="y")
    for y, a, b, c in zip(pos, claimed, caused, point_colors, strict=True):
        ax.plot([a, b], [y, y], color=c, linewidth=1.2, zorder=2)
    ax.scatter(caused, pos, s=26, color=point_colors, zorder=3)
    ax.scatter(
        claimed,
        pos,
        s=26,
        facecolor="white",
        edgecolor=point_colors,
        linewidth=1.2,
        zorder=3,
    )
    ax.axvline(0.0, color="black", linewidth=0.6)
    grid = float(rows.loc[BASELINE, "es_grid_ef_g_per_kwh"])
    grid_line = ax.axvline(grid, color=GRID_COLOR, linewidth=0.6, linestyle=(0, (4, 3)))
    ax.invert_yaxis()
    ax.set_xlabel("CO$_2$ intensity of DC consumption (g/kWh)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    dot = {"marker": "o", "linestyle": "none", "markersize": 5}
    handles = [
        Line2D([], [], color="black", **dot),
        Line2D([], [], markerfacecolor="white", markeredgecolor="black", **dot),
        grid_line,
    ]
    labels = [
        "Caused (consequential)",
        "Claimed (attributional)",
        "Spanish grid average",
    ]
    _legend(fig, handles, labels, ncol=3)

    i = policies.index(DECREE)
    _caption(
        fig,
        "CO2 per kWh the data center consumes. Filled: the emissions it causes "
        "(whole-model change against no data center). Hollow: what it claims by "
        "valuing its grid purchases at the grid's hourly average intensity, with PPA "
        "supply at zero.",
        [
            f"No policy: the data center causes {caused[0]:.0f} g/kWh but would claim "
            f"{claimed[0]:.0f} g/kWh, {caused[0] / claimed[0]:.1f}× less: new load is "
            "served by gas at the margin, not by the average mix.",
            "With matching, the emissions caused turn negative: PPA output sold to the "
            "grid displaces gas.",
            f"Decree: it claims nearly zero ({claimed[i]:.0f} g/kWh) and causes the "
            f"largest reduction ({_signed(caused[i], 0)} g/kWh).",
        ],
    )
    return fig


FIGURES: dict[str, Callable[[pd.DataFrame, dict[str, str]], plt.Figure]] = {
    "clean_coverage": plot_clean_coverage,
    "portfolio": plot_portfolio,
    "dc_cost": plot_dc_cost,
    "system_impact": plot_system_impact,
    "generation_shift": plot_generation_shift,
    "dc_emission_intensity": plot_dc_emission_intensity,
}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: write every study figure as PDF."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--colors", type=Path, required=True)
    parser.add_argument("--excess-cap", type=float, required=True)
    parser.add_argument("--weather-year", type=int, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.style.use(STYLE)
    rows = select_rows(pd.read_csv(args.summary), args.excess_cap, args.weather_year)
    colors = load_colors(args.colors)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in FIGURES.items():
        fig = plot(rows, colors)
        for ext in FORMATS:
            fig.savefig(args.plots_dir / f"{name}.{ext}")
        plt.close(fig)
        logger.info("Wrote %s", args.plots_dir / name)


if __name__ == "__main__":
    main()
