# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Plot the study figures from the scenario summary table.

Reads ``results/scenarios.csv`` from ``make_summary.py`` and writes one file per
figure (PDF, or PNG with ``--format``) for one run, DC size and sales cap: hourly
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
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402

logger = logging.getLogger(__name__)

STYLE = Path(__file__).with_name("plots.mplstyle")
FORMATS = ("pdf", "png")

SPAIN_WIDE = "spain_wide"


class Policy(NamedTuple):
    group: str  # axis label shared by the targets of one policy
    target: int | None
    name: str
    color: str  # Okabe-Ito
    tick: str | None = None  # axis tick when it is not the target share


TARGETS = (80, 90)  # the decree's 80%, and 90% as a sensitivity
# policy key template -> (axis group label, name template, colour), in plotting
# order; the decree's every-hour rule closes the axis
FAMILIES: dict[str, tuple[str, str, str]] = {
    "hourly_{t}_ppa_only": ("Hourly\non average", "Hourly {t}% on average", "#0072B2"),
    "hourly_{t}_grid_credit": (
        "Hourly avg.\n+ credit",
        "Hourly {t}% avg. + credit",
        "#56B4E9",
    ),
    "hourly_{t}_ppa_only@local_region": (
        "Hourly avg.,\nlocal",
        "Hourly {t}% avg., local",
        "#CC79A7",
    ),
    "annual_{t}_res": ("Annual", "Annual {t}%", "#009E73"),
}
TRANSITION_GROUP = "Transition rules\n(tariff periods P1/P2)"  # the revised draft's DT1
TRANSITION_NOTE = (
    " Transition rules: the revised draft's phase-in for data centers in the"
    " automatic load-reduction scheme, a floor in tariff periods P1 and P2 only,"
    " applied hour by hour as in Art. 9:"
    " 60% in P1 hours under the 2026 rules (to 2028), then 70% in P1 and 60% in P2"
    " hours under the 2029 rules; all other hours are unconstrained"
    " (P1: 9–14 h and 18–22 h on weekdays in January, February, July and December;"
    " P2: 8–9, 14–18 and 22–24 h on those days, and 9–14 h and 18–22 h on weekdays"
    " in March and November; local time, national holidays excluded)."
)


def _targets(key: str, group: str, name: str, color: str) -> dict[str, Policy]:
    """One ``Policy`` per target share of a policy family."""
    return {key.format(t=t): Policy(group, t, name.format(t=t), color) for t in TARGETS}


# scenario key (see scenario_key) -> Policy, grouped by family
POLICIES: dict[str, Policy] = (
    {"unmatched_dc": Policy("No policy", None, "No policy", "#999999")}
    | {k: p for key, spec in FAMILIES.items() for k, p in _targets(key, *spec).items()}
    | {
        "transition_p1_60": Policy(
            TRANSITION_GROUP,
            None,
            "Transition, 2026 rules (60% in P1 hours)",
            "#E69F00",
            "2026",
        ),
        "transition_p1_70_p2_60": Policy(
            TRANSITION_GROUP,
            None,
            "Transition, 2029 rules (70% in P1, 60% in P2 hours)",
            "#E69F00",
            "2029",
        ),
    }
    | _targets(
        "strict_hourly_{t}_ppa_only",
        "Every hour\n(decree: 80%)",
        "{t}% every hour",
        "#D55E00",
    )
)
DECREE = "strict_hourly_80_ppa_only"
# every figure shows no policy, the 80% average and the decree; others join when solved
REQUIRED = ["unmatched_dc", "hourly_80_ppa_only", DECREE]
LOCAL = "hourly_80_ppa_only@local_region"
BASELINE = "no_dc"
GROUP_GAP = 0.4  # extra space between policy groups, in bar slots
SLOT_IN = 0.5  # figure width per bar slot (inches)

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
OPERATING = "Extra fuel and running costs of Spanish plants"
DC_COSTS = "Data center PPAs and battery (annuities)"

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
CAPTION_WIDTH = 0.95  # share of the figure width
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
    missing = (set(REQUIRED) | {BASELINE}) - set(keys)
    if missing:
        raise ValueError(f"Summary lacks scenarios {sorted(missing)}.")
    return df.set_index(keys).fillna(0.0)


def shown_policies(rows: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """``POLICIES`` keys solved in ``rows``, in plotting order."""
    return [p for p in POLICIES if p in rows.index and p not in exclude]


def select_rows(
    df: pd.DataFrame, excess_cap: float, weather_year: int, dc_mw: float
) -> pd.DataFrame:
    """One run and DC size: the policies at ``excess_cap`` plus the baselines."""
    run = df[df["weather_year"] == weather_year]
    size = run[(run["policy"] == BASELINE) | np.isclose(run["dc_mw"], dc_mw)]
    baseline = size["policy"].isin([BASELINE, "unmatched_dc"])
    return reference_rows(size[baseline | np.isclose(size["excess_cap"], excess_cap)])


def _positions(policies: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Bar positions of ``policies``, a gap between groups, and each one's group index."""
    groups = pd.Series([POLICIES[p].group for p in policies])
    group_id = ((groups != groups.shift()).cumsum() - 1).to_numpy()
    return np.arange(len(policies)) + GROUP_GAP * group_id, group_id


def _width(policies: list[str], margin: float = 1.0) -> float:
    """Figure width (inches): ``margin`` for the axis labels plus one slot per bar."""
    pos, _ = _positions(policies)
    return margin + SLOT_IN * (pos[-1] + 1.0)


def _tick(p: Policy) -> str:
    if p.tick is not None:
        return p.tick
    return "" if p.target is None else f"{p.target}%"


def _policy_axis(ax: plt.Axes, policies: list[str]) -> np.ndarray:
    """Policy axis: targets as ticks, policy groups below them, the decree highlighted."""
    pos, group_id = _positions(policies)
    ax.set_xticks(pos, [_tick(POLICIES[p]) for p in policies])
    i = policies.index(DECREE)
    band = {"color": POLICIES[DECREE].color, "alpha": 0.08, "linewidth": 0}
    ax.axvspan(pos[i] - 0.5, pos[i] + 0.5, zorder=0, **band)
    ticks = ax.get_xticklabels()
    if not ticks:  # inner panels of a shared axis carry no labels
        return pos
    ticks[i].set_fontweight("bold")
    for g in np.unique(group_id):
        members = np.flatnonzero(group_id == g)
        ax.annotate(
            POLICIES[policies[members[0]]].group,
            xy=(pos[members].mean(), 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(0.0, -16.0),
            textcoords="offset points",
            ha="center",
            va="top",
        )
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


def _tint(color: str | tuple, share: float = 0.55) -> tuple[float, float, float]:
    """``color`` blended towards white by ``share``."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * share)


def _cost_parts(r: pd.DataFrame) -> pd.DataFrame:
    """Change in Spanish system cost (M€/yr): background operating cost and DC costs."""
    dc = r.filter(regex=r"^dc_(ppa_.+|battery)_cost_eur$").fillna(0.0).sum(axis=1)
    operating = r["delta_no_dc_es_system_cost_eur"] - dc
    return pd.DataFrame({OPERATING: operating, DC_COSTS: dc}) * 1e-6


def _signed(v: float, decimals: int) -> str:
    return f"{v:+.{decimals}f}".replace("-", "−")


def _percent(v: float, base: float) -> str:
    """``v`` as a signed percent of ``base``, one decimal below 2%."""
    pct = 100.0 * v / base
    text = f"{pct:+.1f}%" if abs(pct) < 2.0 else f"{pct:+.0f}%"
    return f"({text})".replace("-", "−")


def _legend(fig: plt.Figure, handles: list, labels: list[str], ncol: int) -> None:
    """Frameless multi-column legend above the plot area, bar series first."""
    entries = sorted(
        zip(handles, labels, strict=True),
        key=lambda e: not isinstance(e[0], BarContainer),
    )
    fig.legend(*zip(*entries, strict=True), loc="outside upper left", ncol=ncol)


def _notes(rows: pd.DataFrame) -> str:
    """Caption tail: PPA siting, and the transition rules when those scenarios are shown."""
    if LOCAL in rows.index:
        site = site_name(rows.at[LOCAL, "sites"])
        note = f" PPAs anywhere in mainland Spain unless marked local ({site} only)."
    else:
        note = " PPAs anywhere in mainland Spain."
    if any(POLICIES[p].group == TRANSITION_GROUP for p in shown_policies(rows)):
        note += TRANSITION_NOTE
    return note


def _caption(fig: plt.Figure, summary: str, fontsize: float = CAPTION_FONTSIZE) -> None:
    """
    Caption under the figure: what is shown and how each quantity is defined.

    Lines wrap at the figure width, measured from the caption's own characters
    in the current font.
    """
    text_pt, _, _ = TextToPath().get_text_width_height_descent(
        summary, FontProperties(size=fontsize), ismath=False
    )
    width = int(
        CAPTION_WIDTH * fig.get_size_inches()[0] * 72.0 * len(summary) / text_pt
    )
    lines = textwrap.wrap(summary, width)
    fig.supxlabel(
        "\n".join(lines), x=0.0, ha="left", fontsize=fontsize, linespacing=1.35
    )


def plot_clean_coverage(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Hourly renewable coverage of DC consumption: PPA-matched plus grid renewables."""
    policies = shown_policies(rows)
    r = rows.loc[policies]
    fig, ax = plt.subplots(figsize=(_width(policies), 4.3))
    pos = _policy_axis(ax, policies)
    parts = pd.DataFrame(
        {"PPA matched": r["ppa_only_score"], "Grid renewables": r["grid_clean_share"]}
    ).mul(100.0)
    seg_colors = {"PPA matched": PPA_COLOR, "Grid renewables": GRID_RES_COLOR}
    tops = _stack(ax, pos, parts, {k: {"color": c} for k, c in seg_colors.items()})
    for x, top in zip(pos, tops, strict=True):
        ax.text(x, top + 1.5, f"{top:.0f}%", ha="center", va="bottom", fontsize=7)

    targets = np.array([POLICIES[p].target or np.nan for p in policies], dtype=float)
    ax.hlines(
        targets,
        pos - 0.31,
        pos + 0.31,
        color="black",
        linewidth=0.8,
        linestyle=(0, (2, 1)),
        label="Target",
    )
    ax.set_ylim(0.0, 110.0)
    ax.set_ylabel("Hourly renewable share\nof DC consumption (%)")
    _legend(fig, *ax.get_legend_handles_labels(), ncol=3)

    _caption(
        fig,
        "Share of the data center's consumption covered by renewables in the same "
        "hour: by its own PPAs and battery (dark), or by the renewable share of its "
        "grid purchases (light; import-traced, nuclear excluded)." + _notes(r),
    )
    return fig


def plot_portfolio(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """DC PPA and battery capacity, and PPA output with its curtailment."""
    policies = shown_policies(rows, exclude=("unmatched_dc",))
    r = rows.loc[policies]
    fig, axd = plt.subplot_mosaic(
        [["capacity"], ["energy"]], figsize=(_width(policies), 6.6), sharex=True
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

    _caption(
        fig,
        "The data center's contracted assets (a) and their output (b); hatched: "
        "output curtailed, as surplus cannot be sold; dashed: the data center's load, "
        "as capacity (a) and annual consumption (b)." + _notes(r),
    )
    return fig


def plot_dc_cost(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """DC net cost per MWh consumed: PPAs, battery, grid purchases, minus sales."""
    policies = shown_policies(rows)
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
    fig, ax = plt.subplots(figsize=(_width(policies), 4.6))
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

    _caption(
        fig,
        "The data center's cost per MWh it consumes: PPA and battery annuities plus "
        "grid purchases (bold line: net cost). Surplus PPA output cannot be sold, so "
        "no revenue offsets it." + _notes(r),
    )
    return fig


def plot_system_impact(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Spanish system impact against no DC, absolute (left) and relative (right)."""
    policies = shown_policies(rows)
    r = rows.loc[policies]
    # labels mixing math write Delta as math: their text gets no font fallback
    panels = [
        ("cost", "Δ system cost\n(M€/yr)", "es_system_cost_eur", 1e-6, 0),
        ("co2", "$\\Delta$ CO$_2$ emissions\n(Mt/yr)", "es_total_co2_mt", 1.0, 2),
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
        [[p[0]] for p in panels], figsize=(_width(policies, 2.0), 7.6), sharex=True
    )
    bar_colors = [POLICIES[p].color for p in policies]
    for key, label, col, scale, decimals in panels:
        ax = axd[key]
        pos = _policy_axis(ax, policies)
        vals = r[f"delta_no_dc_{col}"].to_numpy(dtype=float) * scale
        label_x = pos
        if key == "cost":
            # waterfall: operating cost, then the DC costs on top, ending at the total
            parts = _cost_parts(r)
            operating = parts[OPERATING].to_numpy()
            w = 0.31
            light = [_tint(c) for c in bar_colors]
            ax.bar(pos - w / 2, operating, w, color=light, linewidth=0)
            ax.bar(
                pos + w / 2,
                parts[DC_COSTS],
                w,
                bottom=operating,
                color=bar_colors,
                linewidth=0,
            )
            ax.hlines(operating, pos, pos + w, color="#444444", linewidth=0.6)
            ax.legend(
                handles=[
                    Patch(color=_tint(GRID_COLOR), label=OPERATING),
                    Patch(color=GRID_COLOR, label=DC_COSTS),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=2,
            )
            label_x = pos + w / 2
            tops, bottom = np.maximum(vals, operating), operating.min()
        else:
            ax.bar(pos, vals, width=0.62, color=bar_colors, linewidth=0)
            tops, bottom = vals, vals.min()
        _zero_line(ax)
        lo, hi = min(0.0, bottom), max(0.0, tops.max())
        pad = 0.05 * (hi - lo)
        base = float(rows.at[BASELINE, col]) * scale
        for x, v, top in zip(label_x, vals, tops, strict=True):
            # the change, and under it the same change in percent of the baseline
            y = (top if v >= 0.0 else v) + np.sign(v) * pad
            up = v >= 0.0
            ax.annotate(
                _signed(v, decimals),
                (x, y),
                xytext=(0.0, 7.0 if up else 0.0),
                textcoords="offset points",
                ha="center",
                va="bottom" if up else "top",
                fontsize=6,
            )
            ax.annotate(
                _percent(v, base),
                (x, y),
                xytext=(0.0, 0.0 if up else -7.0),
                textcoords="offset points",
                ha="center",
                va="bottom" if up else "top",
                fontsize=6,
                color="#777777",
            )
        ax.set_ylim(
            lo - (0.4 * (hi - lo) if vals.min() < 0.0 else 0.0),
            hi + (0.4 * (hi - lo) if hi > 0.0 else pad),
        )
        ax.set_ylabel(label)
        top = ax.get_ylim()[1]
        # no ticks or gridlines below zero: negative bars hang below the baseline
        ax.set_yticks([t for t in ax.get_yticks() if 0.0 <= t <= top])

    no = "unmatched_dc"
    site = site_name(r.at[no, "sites"])
    _caption(
        fig,
        f"Change in the Spanish power system with a {r.at[no, 'dc_mw']:.0f} MW data "
        f"center in {site} against a baseline without it: each bar carries the change "
        "and, in grey, the same change in percent of the baseline level. System cost: "
        "fuel, CO2 (EU ETS) "
        "and running costs of Spanish power plants and storage (light; imports "
        "excluded), then the data center's PPA and battery annuities on top (dark), "
        "ending at the total. CO2: fuel burnt "
        "in Spanish plants times its carbon content. Gas: fuel burnt "
        "in Spanish gas-fired power plants (CCGT, OCGT). Spot price: hourly nodal "
        "prices weighted by Spanish demand, the data center excluded." + _notes(r),
    )
    return fig


def plot_generation_shift(rows: pd.DataFrame, colors: dict[str, str]) -> plt.Figure:
    """Change in Spanish supply against no DC: what serves the DC's consumption."""
    policies = shown_policies(rows)
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

    fig, ax = plt.subplots(figsize=(_width(policies), 4.4))
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

    _caption(
        fig,
        "Change in Spanish electricity supply against no data center; dashed line: "
        "the data center's consumption. Surplus cannot be sold, so the data center's "
        "PPAs supply at most its own consumption (plus battery losses)." + _notes(r),
    )
    return fig


def plot_dc_emission_intensity(
    rows: pd.DataFrame, colors: dict[str, str]
) -> plt.Figure:
    """Emission intensity of DC consumption: consequential vs. attributional."""
    policies = shown_policies(rows)
    r = rows.loc[policies]
    consequential = (
        r["delta_no_dc_emissions"] / r["total_load_mwh"] * G_PER_KWH_PER_MT_PER_MWH
    ).to_numpy(dtype=float)
    attributional = r["dc_attributional_intensity_g_per_kwh"].to_numpy(dtype=float)
    point_colors = [POLICIES[p].color for p in policies]

    fig, ax = plt.subplots(figsize=(_width(policies), 4.4))
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
    ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.5)
    ax.set_ylabel("Emission intensity of DC\nconsumption (gCO$_2$/kWh)")
    dot = {"marker": "o", "linestyle": "none", "markersize": 5}
    handles = [
        Line2D([], [], color="black", **dot),
        Line2D([], [], markerfacecolor="white", markeredgecolor="black", **dot),
        grid_line,
    ]
    _legend(fig, handles, ["Consequential", "Attributional", "Spanish grid average"], 3)

    _caption(
        fig,
        "CO2 per kWh the data center consumes, measured two ways. Consequential "
        "(filled): the change in whole-model emissions against no data center, per "
        "kWh it consumes. Attributional (hollow): the emissions of its grid purchases, "
        "valued at the hourly emission factor of Spanish supply (imports traced to "
        "their exporter's mix), divided by all its consumption." + _notes(r),
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
    parser.add_argument("--dc-mw", type=float, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--format", choices=FORMATS, default="pdf")
    args = parser.parse_args(argv)

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.style.use(STYLE)
    summary = pd.read_csv(args.summary)
    rows = select_rows(summary, args.excess_cap, args.weather_year, args.dc_mw)
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
