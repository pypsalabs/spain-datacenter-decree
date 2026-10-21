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
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

logger = logging.getLogger(__name__)

STYLE = Path(__file__).with_name("plots.mplstyle")
FORMATS = ("pdf", "png")

SPAIN_WIDE = "spain_wide"
LOCAL = "hourly_80_ppa_only@local_region"
# scenario key (see scenario_key) -> (label, Okabe-Ito colour), in order of relevance
POLICIES: dict[str, tuple[str, str]] = {
    "unmatched_dc": ("No policy", "#999999"),
    "strict_hourly_80_ppa_only": ("Decree:\n80% every hour", "#D55E00"),
    "hourly_80_ppa_only": ("Hourly 80%\non average", "#0072B2"),
    "hourly_80_grid_credit": ("Hourly 80%\navg. + credit", "#56B4E9"),
    LOCAL: ("Hourly 80%\navg., local", "#CC79A7"),
    "annual_80_res": ("Annual\n80%", "#009E73"),
}
DECREE = "strict_hourly_80_ppa_only"
ANNUAL = "annual_80_res"
SUM_RULES = ("hourly_80_ppa_only", "hourly_80_grid_credit", ANNUAL)
BASELINE = "no_dc"

PPA_CARRIERS: dict[str, str] = {
    "onwind": "Onshore wind",
    "solar": "Solar PV",
    "solar-hsat": "Solar PV (tracking)",
}
PPA_COLOR = "#2f6b4f"
GRID_RES_COLOR = "#b9d3c3"
GRID_COLOR = "#8c8c8c"
SALES_COLOR = "#cfcfcf"
IMPORT_COLOR = "#a0a0a0"

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
CAPTION_FONTSIZE = 6.5
CAPTION_WIDTH = 0.98  # share of the figure width
HOURS_PER_YEAR = 8760.0
G_PER_KWH_PER_MT_PER_MWH = 1.0e12 / 1.0e3


def load_colors(path: Path) -> dict[str, str]:
    """PyPSA-Eur ``plotting.tech_colors`` from its plotting config."""
    with open(path) as f:
        return yaml.safe_load(f)["plotting"]["tech_colors"]


def scenario_key(policy: str, siting: str) -> str:
    """Key into ``POLICIES``: the policy, suffixed '@siting' unless PPAs are Spain-wide."""
    return policy if siting == SPAIN_WIDE else f"{policy}@{siting}"


def site_name(sites: str) -> str:
    """Display name of the data center site(s), e.g. 'aragon' -> 'Aragon'."""
    return str(sites).replace("_", " ").title()


def reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a single-run suite indexed by scenario key; fails on multi-run suites."""
    keys = pd.Index(map(scenario_key, df["policy"], df["siting"]), name="scenario")
    if keys.duplicated().any():
        raise ValueError(
            "plot_summary covers single-run suites (one row per scenario); "
            "the full suite needs its own figures."
        )
    missing = (set(POLICIES) | {BASELINE}) - set(keys)
    if missing:
        raise ValueError(f"Summary lacks scenarios {sorted(missing)}.")
    return df.set_index(keys).fillna(0.0)


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


def _zero_line(ax: plt.Axes) -> None:
    """Zero baseline drawn as the axis line, replacing the bottom spine."""
    ax.axhline(0.0, color="#222222", linewidth=0.8, zorder=2.5)
    ax.spines["bottom"].set_visible(False)


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


def _curtailed_share(r: pd.DataFrame) -> pd.Series:
    """Share of the contracted PPA yield curtailed (%), per policy."""
    curtailed = r.filter(regex=r"^dc_ppa_.*_curtailed_mwh$").sum(axis=1)
    return 100.0 * curtailed / r["ppa_yield_mwh"]


def _siting_note(rows: pd.DataFrame) -> str:
    site = site_name(rows.at[LOCAL, "sites"])
    return f" PPAs anywhere in mainland Spain unless marked local ({site} only)."


def _caption(fig: plt.Figure, summary: str, points: list[str]) -> None:
    """
    Caption under the figure: what is shown, then the key mechanics as bullets.

    Lines wrap at the figure width, measured from the caption's own characters
    in the current font.
    """
    text = summary + "".join(points)
    text_pt, _, _ = TextToPath().get_text_width_height_descent(
        text, FontProperties(size=CAPTION_FONTSIZE), ismath=False
    )
    width = int(CAPTION_WIDTH * fig.get_size_inches()[0] * 72.0 * len(text) / text_pt)
    lines = textwrap.wrap(summary, width)
    for point in points:
        lines += textwrap.wrap(
            point, width, initial_indent="• ", subsequent_indent="   "
        )
    fig.supxlabel(
        "\n".join(lines), x=0.0, ha="left", fontsize=CAPTION_FONTSIZE, linespacing=1.35
    )


def plot_clean_coverage(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Hourly renewable coverage of DC consumption: PPA-matched plus grid renewables."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
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
    ppa = r["ppa_only_score"].mul(100.0)
    curtailed = _curtailed_share(r.loc[[ANNUAL]]).iloc[0]
    annual = 100.0 * r.at[ANNUAL, "target"]
    _caption(
        fig,
        "Share of the data center's consumption covered by renewables in the same "
        "hour: by its own PPAs and battery (dark), or by the renewable share of its "
        "grid purchases (light; import-traced, nuclear excluded)." + _siting_note(r),
        [
            f"No policy: the data center still gets the grid's renewable share, "
            f"{coverage['unmatched_dc']:.0f}%.",
            "Decree: 80% is a floor in every hour, and surplus in one hour cannot "
            "offset a shortfall in another. The portfolio is sized for the worst hours "
            "(winter lulls in wind and sun), so it fully covers most other hours: "
            f"{coverage[DECREE]:.0f}% on average.",
            "Hourly 80%: matched energy is summed over the year, so surplus hours "
            "offset short ones and the rule lands exactly on 80%.",
            f"Annual {annual:.0f}%: the PPAs yield {annual:.0f}% of the energy the data "
            f"center consumes, but {curtailed:.0f}% of it arrives when it is not needed "
            f"and is curtailed: they match {ppa[ANNUAL]:.0f}% of consumption hour by "
            "hour.",
            "Grid credit: the grid's renewable share counts toward the 80%, and its "
            f"PPAs alone match {ppa['hourly_80_grid_credit']:.0f}%.",
            f"Annual {annual:.0f}% and grid credit ask for little more than a data "
            "center contracts without any rule, as PPAs cost less than grid power: "
            "grid credit does not bind (score "
            f"{100 * r.at['hourly_80_grid_credit', 'cfe_score']:.0f}%), annual "
            f"{annual:.0f}% barely does.",
        ],
    )
    return fig


def plot_portfolio(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """DC PPA and battery capacity, and PPA output with its curtailment."""
    policies = [p for p in POLICIES if p != "unmatched_dc"]
    r = rows.loc[policies]
    fig, axd = plt.subplot_mosaic(
        [["capacity"], ["energy"]], figsize=(5.4, 6.6), sharex=True
    )
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
    _legend(fig, handles, labels, ncol=3)

    yield_share = r["annual_res_score"].mul(100.0)
    others = list(SUM_RULES)
    _caption(
        fig,
        "The data center's contracted assets (a) and their output (b); hatched: "
        "output curtailed, as surplus cannot be sold; dashed: the data center's load, "
        "as capacity (a) and annual consumption (b)." + _siting_note(r),
        [
            "Decree: covering every hour takes solar plus a battery that carries it "
            "into the night, overbuilt for winter lulls: "
            f"{r.at[DECREE, 'dc_ppa_total_mw'] / 1e3:.1f} GW of PPAs and "
            f"{r.at[DECREE, 'dc_battery_mwh'] / 1e3:.1f} GWh of storage for a "
            f"{r.at[DECREE, 'total_load_mwh'] / HOURS_PER_YEAR:.0f} MW load, with "
            f"{_curtailed_share(r).loc[DECREE]:.0f}% of their output curtailed.",
            "The other Spain-wide rules are met with onshore wind plus tracking solar "
            "and no storage: wind also blows at night, and their targets need not hold "
            "in every hour.",
            "Local: without Spain's windiest sites, the same hourly 80% on average "
            f"takes {r.at[LOCAL, 'dc_ppa_total_mw'] / 1e3:.1f} GW of PPAs, mostly solar, "
            f"and {r.at[LOCAL, 'dc_battery_mwh'] / 1e3:.1f} GWh of storage, with "
            f"{_curtailed_share(r).loc[LOCAL]:.0f}% of their output curtailed.",
            f"Contracted yield: {yield_share[DECREE]:.0f}% of consumption under the "
            f"decree, {_span(yield_share[others])}% under the other Spain-wide rules.",
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
    parts = parts.loc[:, parts.ne(0.0).any()]
    net = r["dc_net_cost_eur_per_mwh"]
    if not np.allclose(parts.sum(axis=1), net, rtol=1e-6, atol=1e-3):
        raise ValueError("DC cost components do not add up to the net cost.")

    seg_colors = {label: colors[c] for c, label in PPA_CARRIERS.items()}
    seg_colors |= {
        "Battery": colors["battery"],
        "Grid purchases": GRID_COLOR,
        "Sales revenue": SALES_COLOR,
    }
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
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
    _zero_line(ax)
    bottom = parts.where(parts < 0.0, 0.0).sum(axis=1).min()
    ax.set_ylim(1.25 * bottom, 1.1 * tops.max())
    ax.set_ylabel("DC cost (€ per MWh consumed)")
    _legend(fig, *ax.get_legend_handles_labels(), ncol=4)

    others = net[list(SUM_RULES)]
    _caption(
        fig,
        "The data center's cost per MWh it consumes: PPA and battery annuities plus "
        "grid purchases (bold line: net cost). Surplus PPA output cannot be sold, so "
        "no revenue offsets it." + _siting_note(r),
        [
            "No policy: the data center buys every MWh at the modelled hourly Spanish "
            f"spot price; {net['unmatched_dc']:.0f} €/MWh is the annual average of "
            "these hourly purchases.",
            "Spain-wide annual and hourly rules: wind and solar PPAs cost less than the "
            f"grid even with {_span(_curtailed_share(r.loc[others.index]))}% of their "
            f"output curtailed, so the bill falls to {_span(others)} €/MWh.",
            f"Decree: the solar overbuild and battery raise it to {net[DECREE]:.0f} "
            "€/MWh, above buying from the grid and "
            f"{_span(net[DECREE] - others)} €/MWh above these rules.",
            f"Local: the same hourly 80% on average costs {net[LOCAL]:.0f} €/MWh with "
            f"PPAs in {site_name(r.at[LOCAL, 'sites'])} only, "
            f"{net[LOCAL] - net['hourly_80_ppa_only']:.0f} €/MWh more than with "
            "Spain-wide PPAs.",
        ],
    )
    return fig


def plot_system_impact(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Spanish system impact against no DC, absolute (left) and relative (right)."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    # labels mixing math write Delta as math: their text gets no font fallback
    panels = [
        ("cost", "Δ system cost\n(M€/yr)", "es_system_cost_eur", 1e-6, 0),
        ("co2", "$\\Delta$ CO$_2$ emissions\n(Mt/yr)", "es_total_co2_mt", 1.0, 2),
        (
            "intensity",
            "$\\Delta$ CO$_2$ intensity\n(g/kWh)",
            "es_co2_intensity_g_per_kwh",
            1.0,
            1,
        ),
        (
            "gas",
            "$\\Delta$ gas for power\n(TWh$_{th}$/yr)",
            "es_gas_fuel_mwh_th",
            1e-6,
            2,
        ),
        ("price", "Δ spot price\n(€/MWh)", "es_mean_price_eur_mwh", 1.0, 2),
    ]
    fig, axd = plt.subplot_mosaic(
        [[p[0]] for p in panels], figsize=(6.4, 9.4), sharex=True
    )
    bar_colors = [POLICIES[p][1] for p in policies]
    for key, label, col, scale, decimals in panels:
        ax = axd[key]
        pos = _policy_axis(ax, policies)
        vals = r[f"delta_no_dc_{col}"].to_numpy(dtype=float) * scale
        ax.bar(pos, vals, width=0.62, color=bar_colors, linewidth=0)
        _zero_line(ax)
        lo, hi = min(0.0, vals.min()), max(0.0, vals.max())
        pad = 0.05 * (hi - lo)
        for x, v in zip(pos, vals, strict=True):
            ax.text(
                x,
                v + np.sign(v) * pad,
                _signed(v, decimals),
                ha="center",
                va="bottom" if v >= 0.0 else "top",
                fontsize=6,
            )
        ax.set_ylim(
            lo - (0.3 * (hi - lo) if lo < 0.0 else 0.0),
            hi + (0.3 * (hi - lo) if hi > 0.0 else pad),
        )
        ax.set_ylabel(label)
        top = ax.get_ylim()[1]
        # no ticks or gridlines below zero: negative bars hang below the baseline
        ax.set_yticks([t for t in ax.get_yticks() if 0.0 <= t <= top])
        base = float(rows.at[BASELINE, col]) * scale
        relative = ax.secondary_yaxis(
            "right",
            functions=(
                lambda v, b=base: 100.0 * v / b,
                lambda q, b=base: q * b / 100.0,
            ),
        )
        relative.set_ylabel("% of baseline")
        relative.spines["right"].set_visible(False)
        top_pct = 100.0 * top / base
        pct_ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(
            0.0, top_pct
        )
        relative.set_yticks([q for q in pct_ticks if 0.0 <= q <= top_pct])

    no = "unmatched_dc"
    site = site_name(r.at[no, "sites"])
    _caption(
        fig,
        f"Change in the Spanish power system with a {r.at[no, 'dc_mw']:.0f} MW data "
        f"center in {site} against a baseline without it, absolute (left axes) and in "
        "percent of the baseline level (right axes). System cost: operating cost of "
        "Spanish plants and storage plus the data center's annuities. CO2: fuel burnt "
        "in Spanish plants times its carbon content. CO2 intensity: Spanish CO2 per "
        "kWh of Spanish generation, the data center's PPAs included. Gas: fuel burnt "
        "in Spanish gas-fired power plants (CCGT, OCGT). Spot price: hourly nodal "
        "prices weighted by Spanish demand, the data center excluded."
        + _siting_note(r),
        [],
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
    shown = parts.loc[:, parts.abs().max() >= MIN_SUPPLY_SHIFT_TWH]

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    pos = _policy_axis(ax, policies)
    _stack(ax, pos, shown, {k: {"color": seg_colors[k]} for k in shown})
    _zero_line(ax)
    load = float(r["total_load_mwh"].iloc[0]) / 1.0e6
    ax.axhline(
        load,
        color="black",
        linewidth=0.6,
        linestyle=(0, (4, 3)),
        label="DC consumption",
    )
    ax.set_ylabel("Δ Spanish supply vs. no DC (TWh/yr)")
    _legend(fig, *ax.get_legend_handles_labels(), ncol=4)

    no = parts.loc["unmatched_dc"]
    _caption(
        fig,
        "Change in Spanish electricity supply against no data center; dashed line: "
        "the data center's consumption. Surplus cannot be sold, so the data center's "
        "PPAs supply at most its own consumption (plus battery losses)."
        + _siting_note(r),
        [
            f"No policy: the extra load is served by gas ({no.get('Gas', 0.0):.1f} TWh), "
            f"more nuclear output ({no.get('Nuclear', 0.0):.1f} TWh) and imports "
            f"({no['Net imports']:.1f} TWh).",
            "Spain-wide annual and hourly rules: the PPAs supply "
            f"{_span(parts.loc[list(SUM_RULES), 'DC PPAs'], 1)} TWh of the data center's "
            f"{load:.1f} TWh; gas serves most of the rest, in hours of low wind and sun.",
            "Decree: its PPAs and battery supply the whole load, so the rest of the "
            "system barely changes.",
        ],
    )
    return fig


def plot_dc_emission_intensity(
    rows: pd.DataFrame, colors: dict[str, str]
) -> plt.Figure:
    """Emission intensity of DC consumption: consequential vs. attributional."""
    policies = list(POLICIES)
    r = rows.loc[policies]
    consequential = (
        r["delta_no_dc_emissions"] / r["total_load_mwh"] * G_PER_KWH_PER_MT_PER_MWH
    ).to_numpy(dtype=float)
    attributional = r["dc_attributional_intensity_g_per_kwh"].to_numpy(dtype=float)
    point_colors = [POLICIES[p][1] for p in policies]

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    pos = _policy_axis(ax, policies)
    for x, a, b, c in zip(pos, attributional, consequential, point_colors, strict=True):
        ax.plot([x, x], [a, b], color=c, linewidth=1.2, zorder=2)
    ax.scatter(pos, consequential, s=26, color=point_colors, zorder=3)
    ax.scatter(
        pos,
        attributional,
        s=26,
        facecolor="white",
        edgecolor=point_colors,
        linewidth=1.2,
        zorder=3,
    )
    _zero_line(ax)
    grid = float(rows.loc[BASELINE, "es_grid_ef_g_per_kwh"])
    grid_line = ax.axhline(grid, color=GRID_COLOR, linewidth=0.6, linestyle=(0, (4, 3)))
    ax.set_xlim(-0.5, len(policies) - 0.5)
    ax.set_ylabel("Emission intensity of DC\nconsumption (gCO$_2$/kWh)")
    dot = {"marker": "o", "linestyle": "none", "markersize": 5}
    handles = [
        Line2D([], [], color="black", **dot),
        Line2D([], [], markerfacecolor="white", markeredgecolor="black", **dot),
        grid_line,
    ]
    _legend(fig, handles, ["Consequential", "Attributional", "Spanish grid average"], 3)

    i = policies.index(DECREE)
    sums = [policies.index(p) for p in SUM_RULES]
    loc, sw = policies.index(LOCAL), policies.index("hourly_80_ppa_only")
    imports = r["delta_no_dc_es_net_import_mwh"] / 1e6
    _caption(
        fig,
        "CO2 per kWh the data center consumes, measured two ways. Consequential "
        "(filled): the change in whole-model emissions against no data center, per "
        "kWh it consumes. Attributional (hollow): the emissions of its grid purchases, "
        "valued at the hourly emission factor of Spanish supply (imports traced to "
        "their exporter's mix), divided by all its consumption." + _siting_note(r),
        [
            f"No policy: consequential {consequential[0]:.0f} g/kWh against "
            f"attributional {attributional[0]:.0f} g/kWh, "
            f"{consequential[0] / attributional[0]:.1f}× higher: new load is served by "
            "gas at the margin, not by the average mix.",
            "Spain-wide annual and hourly rules: consequential "
            f"{_span(consequential[sums])} g/kWh "
            f"against attributional {_span(attributional[sums])} g/kWh: their residual "
            "purchases fall in hours when gas sets the margin.",
            "Decree: attributional nearly zero "
            f"({attributional[i]:.0f} g/kWh), consequential slightly negative "
            f"({_signed(consequential[i], 0)} g/kWh): it buys only "
            f"{r.at[DECREE, 'grid_import_mwh'] / 1e6:.2f} TWh from the grid, and with "
            f"the shifted grid flows Spain imports {imports[DECREE]:.2f} TWh more and "
            f"burns {-r.at[DECREE, 'delta_no_dc_es_gas_fuel_mwh_th'] / 1e6:.2f} TWh "
            "less gas.",
            f"Local: attributional {attributional[loc]:.0f} g/kWh, below the "
            f"Spain-wide {attributional[sw]:.0f}, yet consequential "
            f"{consequential[loc]:.0f} against {consequential[sw]:.0f} g/kWh: Spain "
            f"imports {imports[LOCAL]:.2f} TWh more, so part of the extra emissions "
            "lands in France and Portugal.",
        ],
    )
    return fig


FIGURES: dict[str, Callable[[pd.DataFrame, dict[str, str]], plt.Figure]] = {
    "dc_clean_coverage": plot_clean_coverage,
    "dc_portfolio": plot_portfolio,
    "dc_cost": plot_dc_cost,
    "system_impact": plot_system_impact,
    "system_generation_shift": plot_generation_shift,
    "dc_emission_intensity": plot_dc_emission_intensity,
}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: write every study figure as PDF (study) or PNG (showcase)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--colors", type=Path, required=True)
    parser.add_argument("--excess-cap", type=float, required=True)
    parser.add_argument("--weather-year", type=int, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--format", choices=FORMATS, default="pdf")
    args = parser.parse_args(argv)

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.style.use(STYLE)
    rows = select_rows(pd.read_csv(args.summary), args.excess_cap, args.weather_year)
    if args.excess_cap > 0.0:
        logger.warning("Captions describe the no-sales case; review them for sales.")
    colors = load_colors(args.colors)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in FIGURES.items():
        fig = plot(rows, colors)
        fig.savefig(args.plots_dir / f"{name}.{args.format}")
        plt.close(fig)
        logger.info("Wrote %s", args.plots_dir / name)


if __name__ == "__main__":
    main()
