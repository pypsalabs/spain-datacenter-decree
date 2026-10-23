# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Plot the graphical validation report of the 2024 back-cast.

One figure of the raw PyPSA-Eur build and the patched model against published
2024 actuals, with its caption: PyPSA-Eur's OSM grid and power plants; the
clustered raw and patched models; capacity factors of the raw PyPSA-Eur build per
resource class; and Spanish capacity, generation and power-sector CO2.

The maps read the upstream build in the ``pypsa-eur`` submodule (OSM base
network, power plants, clustered and resource-class regions) and the two base
networks; the bar charts read the validation reports. Style, technology groups
and colours follow the study figures (``plot_summary.py``).
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Wedge  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402
from shapely.geometry import box  # noqa: E402

from scripts.plot_summary import (  # noqa: E402
    CAPTION_WIDTH,
    STYLE,
    SUPPLY_GROUPS,
    _stack,
    _zero_line,
    load_colors,
)
from scripts.prepare_network import AVERAGE_OPERATING_CARRIERS  # noqa: E402

logger = logging.getLogger(__name__)

YEAR = 2024
# national_capacities scopes making up the modelled Spanish system
SPAIN_SCOPES = ("ES_peninsular", "ES_baleares")
VARIANTS = {"raw": "Raw", "patched": "Patched", "actual": "Actual"}

# map window: Iberia and the Balearics (lon0, lat0, lon1, lat1), in web-mercator metres
WINDOW = (
    gpd.GeoSeries([box(-10.3, 35.6, 5.0, 44.3)], crs=4326).to_crs(3857).total_bounds
)
LAND = "#f2f2f2"
HVAC_COLOR, HVDC_COLOR, HVAC_LV_COLOR = "mediumorchid", "dodgerblue", "orange"
PLANT_PT2_PER_MW = 1 / 20
PIE_M_PER_SQRT_GW = 2.0e4
# capacity factor maps: carrier -> colour map; wind in blues (YlGnBu without its
# palest end, which would merge with the land), apart from solar
CF_CMAPS = {
    "solar": "viridis",
    "solar-hsat": "viridis",
    "onwind": ListedColormap(plt.get_cmap("YlGnBu")(np.linspace(0.2, 1.0, 256))),
}
CF_CARRIERS = tuple(CF_CMAPS)
# colour range of each capacity factor map: these area-weighted quantiles of its
# Iberian classes
CF_QUANTILES = (0.05, 0.95)
# technology group -> Redeia CO2 split in the validation reports; Oil takes the rest
# (oil, gas turbines, waste)
CO2_PARTS: dict[str, str] = {"Gas": "co2_ccgt", "CHP": "co2_chp", "Coal": "co2_coal"}
# text sizes of the 10-inch figure, scaled up from the 6.4-inch study style
FONTS = {
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    # bold caption lead-in as mathtext, in the figure typeface
    "mathtext.fontset": "custom",
    "mathtext.rm": "Libre Franklin",
    "mathtext.bf": "Libre Franklin:bold",
}
CAPTION_FONTSIZE = 9.0

# technology group (SUPPLY_GROUPS) -> published national_capacities metrics, summed
PUBLISHED: dict[str, tuple[str, ...]] = {
    "Nuclear": ("nuclear",),
    "Hydro": ("run-of-river/small hydro", "reservoir hydro (incl. mixed PHS)"),
    "Pumped hydro": ("pumped hydro (pure)",),
    "Wind": ("onwind", "offwind"),
    "Solar": ("solar PV", "CSP/solar thermal"),
    "Gas": ("CCGT", "OCGT/gas turbines"),
    "CHP": ("cogeneration",),
    "Coal": ("coal", "lignite"),
    "Oil": ("oil/diesel/fuel",),
    "Biomass & waste": ("biomass/waste",),
}
# technology group -> validation report metrics (Redeia counts gas turbines with oil)
GENERATION: dict[str, tuple[str, ...]] = {
    "Nuclear": ("gen_nuclear",),
    "Hydro": ("gen_hydro",),
    "Pumped hydro": ("gen_phs_turbining",),
    "Wind": ("gen_onwind",),
    "Solar": ("gen_solar_pv", "gen_solar_thermal_csp"),
    "Gas": ("gen_ccgt",),
    "CHP": ("gen_chp",),
    "Coal": ("gen_coal",),
    "Oil": ("gen_oil_diesel_gt_st",),
    "Biomass & waste": ("gen_biomass_waste",),
}
# powerplantmatching fuel type -> technology group; storage is left out
PLANT_GROUPS: dict[str, str] = {
    "Nuclear": "Nuclear",
    "Hydro": "Hydro",
    "Wind": "Wind",
    "Solar": "Solar",
    "Natural Gas": "Gas",
    "Hard Coal": "Coal",
    "Lignite": "Coal",
    "Oil": "Oil",
    "Bioenergy": "Biomass & waste",
    "Solid Biomass": "Biomass & waste",
    "Waste": "Biomass & waste",
}


def _bold(text: str) -> str:
    """Bold mathtext that line wrapping keeps whole (no-break spaces and hyphens)."""
    return "$\\bf{" + text.replace(" ", "\u00a0").replace("-", "\u2011") + "}$"


CALIBRATION = (
    _bold("Calibration on top of PyPSA-Eur raw network")
    + ": (i) installed capacity per technology "
    "set to published totals in Spain, Portugal and France, wind and solar PV at the "
    "2024 average operating capacity, missing units added; (ii) onshore wind "
    "corrected for the residual ERA5 bias by national factor; (iii) reservoir inflow "
    "scaled to the observed 2024 hydro output; pumped hydro given its own storage, "
    "pumping power and efficiency; (iv) CHP and CSP output follow their realized 2024 "
    "profiles, biomass and waste capped at their realized capacity factor; (v) demand "
    "scaled to the published 2024 busbar demand; (vi) marginal costs from realized "
    "2024 fuel and carbon prices; also factored gas network premium for Portuguese "
    "gas-fired plants; (vii) corridors capped at their net transfer capacities, "
    "exports to Morocco and Andorra fixed, and French nuclear availability is shaped "
    "by its 2024 monthly output."
)
COAL_NOTE = (
    _bold("Why the patched model burns no coal")
    + ": Aboño I in Asturias burns the waste gases of the ArcelorMittal steelworks "
    "in Gijón. They are a by-product that has to be burned anyway, so the plant runs "
    "with the steelworks, not with the power price."
)


# ------------------------------------------------------------------ data
def _group_of(carrier: pd.Series) -> pd.Series:
    """Technology group (SUPPLY_GROUPS) of each network carrier; NaN if none."""
    groups = {c: g for g, (cs, _) in SUPPLY_GROUPS.items() for c in cs}
    return carrier.map(groups)


def capacity_by_group(n: pypsa.Network) -> pd.DataFrame:
    """Generator and storage capacity (GW) per bus and technology group."""
    assets = pd.concat([n.generators, n.storage_units])
    gw = assets["p_nom"].groupby([assets["bus"], _group_of(assets["carrier"])]).sum()
    return gw.unstack(fill_value=0.0).reindex(columns=list(SUPPLY_GROUPS)) / 1e3


def corridors(n: pypsa.Network) -> tuple[pd.Series, pd.Series]:
    """
    Transfer capacity (GW) per bus pair: AC lines and HVDC links.

    Counts ``s_nom x s_max_pu`` and ``p_nom x p_max_pu``, so the NTC caps of the
    patched model show.
    """

    def per_pair(df: pd.DataFrame, mw: pd.Series) -> pd.Series:
        pairs = [tuple(sorted(ab)) for ab in zip(df["bus0"], df["bus1"])]
        return mw.groupby(pd.Series(pairs, index=df.index)).sum() / 1e3

    links = n.links[(n.links["carrier"] == "DC") & (n.links["p_nom"] > 0)]
    return (
        per_pair(n.lines, n.lines["s_nom"] * n.lines["s_max_pu"]),
        per_pair(links, links["p_nom"] * links["p_max_pu"]),
    )


def class_cf(n: pypsa.Network, carrier: str) -> pd.Series:
    """Mean available capacity factor of each generator of a carrier."""
    gens = n.generators.index[n.generators["carrier"] == carrier]
    w = n.snapshot_weightings["generators"]
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu")[gens]
    return p_max_pu.mul(w, axis=0).sum() / w.sum()


def published_capacity(capacities: pd.DataFrame, scopes: tuple[str, ...]) -> pd.Series:
    """
    Published 2024 capacity (MW) per metric, summed over scopes.

    Wind and solar PV take the average operating capacity (mean of the 2023 and
    2024 year-ends) where both are published, as the patched model does.
    """
    df = capacities[capacities["country"].isin(scopes)]
    by_year = df.groupby(["year", "carrier_or_metric"])["value"].sum()
    out = by_year.loc[YEAR].copy()
    for metric in AVERAGE_OPERATING_CARRIERS & set(out.index):
        if (YEAR - 1, metric) in by_year.index:
            out[metric] = (out[metric] + by_year[(YEAR - 1, metric)]) / 2
    return out


def spanish_capacity(
    raw: pypsa.Network, patched: pypsa.Network, capacities: pd.DataFrame
) -> pd.DataFrame:
    """Spanish capacity (GW) per variant (rows) and technology group (columns)."""
    out = {}
    for name, n in (("raw", raw), ("patched", patched)):
        gw = capacity_by_group(n)
        out[name] = gw[gw.index.str.startswith("ES")].sum()
    published = published_capacity(capacities, SPAIN_SCOPES)
    out["actual"] = pd.Series(
        {g: published.reindex(ms).sum() / 1e3 for g, ms in PUBLISHED.items()}
    )
    return pd.DataFrame(out).T


def load_reports(raw: Path, patched: Path) -> pd.DataFrame:
    """Model values of both validation reports and their shared actuals, by metric."""
    r, p = (pd.read_csv(f, index_col="metric") for f in (raw, patched))
    if not r["actual"].equals(p["actual"]):
        raise ValueError(
            "The raw and patched reports compare against different actuals."
        )
    return pd.DataFrame(
        {"raw": r["model"], "patched": p["model"], "actual": p["actual"]}
    )


# ------------------------------------------------------------------ maps
def _mercator(lon, lat) -> list[tuple[float, float]]:
    """Longitude/latitude points in web-mercator metres."""
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(3857)
    return list(zip(pts.x, pts.y))


def _map_axes(ax: plt.Axes, regions: gpd.GeoSeries, letter: str) -> None:
    """Window, land, model regions dashed, countries solid."""
    countries = gpd.GeoSeries(
        regions.groupby(regions.index.str[:2]).agg(lambda s: s.union_all()),
        crs=regions.crs,
    )
    countries.plot(ax=ax, color=LAND, zorder=0)
    regions.boundary.plot(
        ax=ax, color="#9a9a9a", linewidth=0.3, linestyle=(0, (3, 2)), zorder=1
    )
    countries.boundary.plot(ax=ax, color="#444444", linewidth=0.5, zorder=1)
    x0, y0, x1, y1 = WINDOW
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xticks=[], yticks=[], aspect="equal")
    ax.set_title(letter, loc="left", fontweight="bold")
    ax.spines[:].set_visible(False)
    ax.grid(False)


def _map_keys() -> list[Line2D]:
    """Keys of the network maps, one legend column each: lines, plants, transfer."""
    lines = [
        Line2D([], [], color=HVAC_COLOR, lw=0.8, label="380-400 kV"),
        Line2D([], [], color=HVAC_LV_COLOR, lw=0.45, label="220-225 kV"),
        Line2D([], [], color=HVDC_COLOR, lw=0.9, ls=(0, (3, 1.5)), label="HVDC"),
    ]
    plants = [
        Line2D([], [], ls="", marker="o", ms=np.sqrt(mw * PLANT_PT2_PER_MW),
               color="#777777", alpha=0.6, mew=0, label=f"{mw:,} MW plant")
        for mw in (100, 500, 1000)
    ]  # fmt: skip
    transfer = [
        Line2D([], [], color=HVAC_COLOR, lw=_transfer_width(gw), label=f"{gw} GW transfer")
        for gw in (1, 3, 6)
    ]  # fmt: skip
    return [*lines, *plants, *transfer]


def _osm_map(ax, full, plants, regions, colors) -> None:
    """Upstream OSM grid by voltage and power plants by technology and size."""
    _map_axes(ax, regions, "a")
    lines = full.lines[~full.lines["under_construction"].astype(bool)]
    geoms = gpd.GeoSeries.from_wkt(lines["geometry"], crs=4326).to_crs(3857)
    hv = (lines["v_nom"] >= 380).to_numpy()
    geoms[~hv].plot(ax=ax, color=HVAC_LV_COLOR, linewidth=0.45, alpha=0.7, zorder=2)
    geoms[hv].plot(ax=ax, color=HVAC_COLOR, linewidth=0.8, alpha=0.7, zorder=3)
    dc = full.links[full.links["carrier"] == "DC"]
    gpd.GeoSeries.from_wkt(dc["geometry"], crs=4326).to_crs(3857).plot(
        ax=ax, color=HVDC_COLOR, linewidth=0.9, linestyle=(0, (3, 1.5)), zorder=3
    )
    group = plants["Fueltype"].map(PLANT_GROUPS)
    plants = plants[group.notna()].assign(group=group).sort_values("Capacity")
    xy = _mercator(plants["lon"], plants["lat"])
    ax.scatter(*zip(*xy), s=plants["Capacity"] * PLANT_PT2_PER_MW, alpha=0.6,
               c=plants["group"].map(lambda g: colors[SUPPLY_GROUPS[g][1]]),
               linewidth=0, zorder=4)  # fmt: skip


def _transfer_width(gw: float) -> float:
    return 0.3 + 0.3 * gw


def _pie(ax: plt.Axes, xy: tuple[float, float], gw: pd.Series, colors) -> None:
    """Capacity pie of one node, its area proportional to the total."""
    gw = gw[gw > 0]
    radius, start = PIE_M_PER_SQRT_GW * np.sqrt(gw.sum()), 90.0
    for group, value in gw.items():
        sweep = 360 * value / gw.sum()
        color = colors[SUPPLY_GROUPS[group][1]]
        ax.add_patch(Wedge(xy, radius, start - sweep, start, color=color, zorder=4))
        start -= sweep
    ax.add_patch(Circle(xy, radius, fill=False, edgecolor="white", lw=0.4, zorder=4.1))


def _cluster_map(ax, n, regions, colors, letter) -> None:
    """Clustered model: transfer capacity per corridor, capacity pie per node."""
    _map_axes(ax, regions, letter)
    ac = n.buses[n.buses["carrier"] == "AC"]
    pos = dict(zip(ac.index, _mercator(ac.x, ac.y)))
    lines, links = corridors(n)
    for (a, b), gw in lines.items():
        ax.plot(*zip(pos[a], pos[b]), color=HVAC_COLOR, lw=_transfer_width(gw),
                alpha=0.8, solid_capstyle="round", zorder=2)  # fmt: skip
    for (a, b), gw in links.items():
        ax.plot(*zip(pos[a], pos[b]), color=HVDC_COLOR, lw=_transfer_width(gw),
                ls=(0, (3, 1.5)), zorder=2)  # fmt: skip
    for bus, gw in capacity_by_group(n).reindex(list(pos), fill_value=0).iterrows():
        _pie(ax, pos[bus], gw, colors)


def _pie_key(ax: plt.Axes) -> None:
    """Reference pie sizes on one baseline in the Mediterranean, in map units."""
    [(x, y)] = _mercator([1.0], [35.9])
    sizes = (20, 5)
    label_x = x + PIE_M_PER_SQRT_GW * np.sqrt(max(sizes))
    for gw in sizes:
        r = PIE_M_PER_SQRT_GW * np.sqrt(gw)
        ax.add_patch(Circle((x, y + r), r, fill=False, edgecolor="#444444", lw=0.5))
        ax.plot([x, label_x], [y + 2 * r] * 2, color="#9a9a9a", lw=0.4)
        ax.annotate(f"{gw} GW", (label_x, y + 2 * r), xytext=(2, 0),
                    textcoords="offset points", va="center")  # fmt: skip


def _class_cf(n, class_regions, carrier) -> gpd.GeoDataFrame:
    """Resource-class regions of a carrier with their capacity factor."""
    cf = class_cf(n, carrier)
    cls = class_regions[carrier]
    names = cls["bus"] + " " + cls["bin"].astype(int).astype(str) + f" {carrier}"
    return cls.assign(cf=cf.reindex(names).to_numpy()).dropna(subset=["cf"])


def _fitted_norm(cls: gpd.GeoDataFrame) -> Normalize:
    """Colour scale over the area-weighted ``CF_QUANTILES`` of the Iberian classes."""
    iberia = cls[cls["bus"].str[:2].isin(["ES", "PT"])].sort_values("cf")
    area = iberia.to_crs(3035).area
    share = area.cumsum() / area.sum()
    return Normalize(*np.round(np.interp(CF_QUANTILES, share, iberia["cf"]), 2))


def _cf_map(fig, ax, n, regions, class_regions, carrier, letter) -> None:
    """Capacity factor per resource class, on a colour scale fitted to the map."""
    _map_axes(ax, regions, letter)
    cls = _class_cf(n, class_regions, carrier)
    norm = _fitted_norm(cls)
    cmap = CF_CMAPS[carrier]
    cls.to_crs(3857).plot(ax=ax, column="cf", cmap=cmap, norm=norm, zorder=0.5,
                          edgecolor="face", linewidth=0.2)  # fmt: skip
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
        location="bottom", shrink=0.7, aspect=25, pad=0.02, extend="both",
    )  # fmt: skip
    bar.ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}")
    bar.set_label("Capacity factor (%)")
    bar.outline.set_visible(False)


# ------------------------------------------------------------------ bars
def _variant_axis(ax: plt.Axes, letter: str, ylabel: str) -> np.ndarray:
    """Categorical axis of the three variants, with panel letter and zero line."""
    pos = np.arange(len(VARIANTS))
    ax.set_xticks(pos, list(VARIANTS.values()))
    ax.set_ylabel(ylabel)
    ax.set_title(letter, loc="left", fontweight="bold")
    _zero_line(ax)
    return pos


def _totals(ax: plt.Axes, pos: np.ndarray, tops: np.ndarray) -> None:
    """Value labels on top of the bars."""
    for x, top in zip(pos, tops):
        ax.annotate(f"{top:.0f}", (x, max(top, 0.0)), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9)  # fmt: skip


def _supply_bars(ax, df: pd.DataFrame, colors, letter: str, ylabel: str) -> None:
    """Stacked bars by technology group, one per variant."""
    pos = _variant_axis(ax, letter, ylabel)
    styles = {g: {"color": colors[c]} for g, (_, c) in SUPPLY_GROUPS.items()}
    groups = [g for g in SUPPLY_GROUPS if g in df]
    tops = _stack(ax, pos, df.loc[list(VARIANTS), groups], styles)
    _totals(ax, pos, tops)


# ------------------------------------------------------------------ figure
def _caption(fig: plt.Figure, paragraphs: list[str]) -> None:
    """
    Caption under the figure, one or more paragraphs.

    Lines wrap at the figure width, measured from the caption's own characters
    in the current font.
    """
    text = " ".join(paragraphs)
    text_pt, _, _ = TextToPath().get_text_width_height_descent(
        text, FontProperties(size=CAPTION_FONTSIZE), ismath=False
    )
    width = int(CAPTION_WIDTH * fig.get_size_inches()[0] * 72.0 * len(text) / text_pt)
    lines = [line for par in paragraphs for line in textwrap.wrap(par, width)]
    fig.supxlabel(
        "\n".join(lines), x=0.0, ha="left", fontsize=CAPTION_FONTSIZE, linespacing=1.35
    )


def plot_report(
    full: pypsa.Network,
    plants: pd.DataFrame,
    raw: pypsa.Network,
    patched: pypsa.Network,
    regions: gpd.GeoSeries,
    class_regions: dict[str, gpd.GeoDataFrame],
    v: pd.DataFrame,
    capacity: pd.DataFrame,
    colors: dict[str, str],
) -> plt.Figure:
    """The validation figure: maps of grid, clusters and capacity factors, then bars."""
    fig = plt.figure(figsize=(10.0, 10.6), layout="constrained")
    networks, cfs, bars = fig.subfigures(3, 1, height_ratios=[2.6, 3.0, 3.1])

    ax_osm, ax_raw, ax_patched = networks.subplots(1, 3)
    _osm_map(ax_osm, full, plants, regions, colors)
    _cluster_map(ax_raw, raw, regions, colors, "b")
    _cluster_map(ax_patched, patched, regions, colors, "c")
    _pie_key(ax_patched)

    for ax, carrier, letter in zip(cfs.subplots(1, 3), CF_CARRIERS, "def"):
        _cf_map(cfs, ax, raw, regions, class_regions, carrier, letter)

    generation = pd.DataFrame(
        {g: v.loc[list(ms)].sum() for g, ms in GENERATION.items()}
    )
    ax_cap, ax_gen, ax_co2 = bars.subplots(1, 3)
    _supply_bars(ax_cap, capacity, colors, "g", "Installed capacity (GW)")
    _supply_bars(ax_gen, generation, colors, "h", "Generation (TWh/yr)")
    co2 = pd.DataFrame({g: v.loc[m] for g, m in CO2_PARTS.items()})
    co2["Oil"] = v.loc["co2_power_sector_total"] - co2.sum(axis=1)
    _supply_bars(ax_co2, co2, colors, "i", "Power-sector CO$_2$ (Mt/yr)")

    fig.legend(
        handles=[
            Patch(color=colors[c], label=g) for g, (_, c) in SUPPLY_GROUPS.items()
        ],
        loc="outside upper left",
        ncol=4,
    )
    fig.legend(handles=_map_keys(), loc="outside upper right", ncol=3)
    _caption(
        fig,
        [
            "Validation of the 2024 back-cast for mainland Spain and the Balearics: "
            "the raw PyPSA-Eur build (raw) and the patched model (patched), both on "
            "2024 weather, against published 2024 actuals (actual; Redeia). (a) Input "
            "of the upstream PyPSA-Eur build: the OpenStreetMap grid by voltage and "
            "the powerplantmatching plants by technology, sized by capacity. (b, c) "
            "The clustered raw and patched models: capacity per node and transfer "
            "capacity per corridor. (d-f) Capacity factor of the raw PyPSA-Eur build "
            "per resource class, 2024 average, for solar PV with fixed tilt (d), solar "
            "PV with single-axis tracking (e) and onshore wind before its bias "
            "correction (f), each on its own colour scale (atlite on ERA5 weather, "
            "SARAH-3 irradiance for solar). "
            "(g-i) Spanish installed capacity, generation and power-sector CO2 by "
            "technology; oil includes gas turbines and waste.",
            CALIBRATION,
            COAL_NOTE,
        ],
    )
    return fig


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: write the validation figure."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-network", type=Path, required=True)
    parser.add_argument("--powerplants", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--class-regions", type=Path, nargs=3, required=True)
    parser.add_argument("--networks", type=Path, nargs=2, required=True)
    parser.add_argument("--reports", type=Path, nargs=2, required=True)
    parser.add_argument("--capacities", type=Path, required=True)
    parser.add_argument("--colors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    for noisy in ("pypsa", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    plt.style.use(STYLE)
    raw, patched = (pypsa.Network(f) for f in args.networks)
    regions = gpd.read_file(args.regions).set_index("name").geometry.to_crs(3857)
    class_regions = {
        f.stem.removeprefix("regions_by_class_"): gpd.read_file(f)
        for f in args.class_regions
    }
    with plt.rc_context(FONTS):
        fig = plot_report(
            pypsa.Network(args.base_network),
            pd.read_csv(args.powerplants, index_col=0),
            raw,
            patched,
            regions,
            class_regions,
            load_reports(*args.reports),
            spanish_capacity(raw, patched, pd.read_csv(args.capacities)),
            load_colors(args.colors),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=250, bbox_inches="tight")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
