# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Prepare a base network: apply cited patches to the composed PyPSA-Eur network.

Patch engine for the study's background system. Reads the composed network, applies an
ordered list of cited corrections controlled by ``spain_cfe.patches`` (each individually
switchable; the ``raw`` variant keeps only the France boundary closure), freezes the
background fleet, and writes the base network that ``solve_network.py`` consumes.
The philosophy is: build fresh, validate, correct only what the validation report
demands, cite every correction (inputs in ``data/custom``, sources in
``doc/data_sources.md``).

Two year dimensions: the **data year** (the ``{year}`` wildcard: 2024 rows for the
back-cast calibration, 2025 rows for the case study) selects the cited data vintage
for fleet, demand, prices and NTCs, while the **weather year** (from the run) keys
the weather-measurement patches (``hydro_inflow``, and the back-cast derivation in
``wind_correction``).

Patches
-------
- ``boundary_flows``: France's external borders (BE, DE, CH, IT, GB) fixed to measured
  hourly 2024 flows (RTE eCO2mix), attached as fixed generators (imports) and loads
  (exports) at the nearest French node. MWh only; no clean shares are traced beyond
  France. Model closure, not calibration: on in every variant.
- ``national_capacities``: set national fleet totals per carrier to the published
  actuals in ``national_capacities.csv``, pro rata over the existing nodal
  distribution. For the 2024 back-cast, fast-growing carriers (wind, solar)
  use the average operating capacity (mean of year-end 2023 and 2024).
- ``wind_correction``: residual ERA5 wind-bias factor for ES and PT; derived at run
  time on the 2024 back-cast, applied as the fixed cited factors from
  ``wind_correction.csv`` in case-study runs.
- ``phs_split``: pure pumped-storage energy, pumping capacity and round-trip
  efficiency per the cited sources; mixed pumped storage stays inside reservoir
  hydro per Redeia's convention (its capacity enters via ``national_capacities``).
- ``chp_profile``: Spanish cogeneration as a fixed profile (monthly shape derived at
  run time from the vendored Redeia monthly cogeneration CO2 series) scaled to the
  published annual output.
- ``csp``: concentrated solar power, capacity from ``national_capacities.csv``, with
  the nodal solar profile rescaled to the published annual CSP output.
- ``prices``: thermal marginal costs recomputed as VOM + fuel/eta + ETS x intensity/eta
  from ``prices.csv`` (fuel and CO2 prices) and the processed technology-data costs
  (VOM). The composed network carries no CO2 price at all.
- ``ntc_caps``: cap the ES-FR, ES-PT and Peninsula-Balearics corridors to published
  NTC values by scaling ``s_max_pu`` (lines) and ``p_max_pu`` (links); ``s_nom`` and
  impedances stay physical, so KVL is unaffected.
- ``demand``: scale each country's load to the cited annual busbar demand.
- ``hydro_inflow``: wy2024 runs only; scale ES/PT reservoir inflow to the observed
  2024 hydro energy (the upstream EIA normalization data ends before 2024).
- ``ma_ad_exports``: fixed ES exports to Morocco and Andorra as flat loads at the
  nearest Spanish nodes.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# Border coordinates used to pick the nearest node for fixed border flows.
FR_NEIGHBOUR_COORDS: dict[str, tuple[float, float]] = {
    "BE": (4.35, 50.50),
    "DE": (7.80, 48.80),
    "DE_BE": (5.90, 49.90),
    "CH": (6.80, 46.80),
    "IT": (7.00, 45.10),
    "GB": (0.50, 50.80),
    "LU": (6.13, 49.61),
}
ES_BORDER_COORDS: dict[str, tuple[float, float]] = {
    "MA": (-5.60, 36.00),
    "AD": (1.52, 42.51),
}

# national_capacities.csv carrier_or_metric -> network generator carriers
NATCAP_GEN_MAP: dict[str, tuple[str, ...]] = {
    "onwind": ("onwind",),
    "solar PV": ("solar", "solar-hsat"),
    "nuclear": ("nuclear",),
    "CCGT": ("CCGT",),
    "OCGT/gas turbines": ("OCGT",),
    "coal": ("coal",),
    "lignite": ("lignite",),
    "oil/diesel/fuel": ("oil",),
    "biomass/waste": ("biomass", "waste"),
    "run-of-river/small hydro": ("ror",),
}
NATCAP_SU_MAP: dict[str, str] = {
    "reservoir hydro (incl. mixed PHS)": "hydro",
    "pumped hydro (pure)": "PHS",
}
# scopes in national_capacities.csv -> bus prefix in the 15-node network
NATCAP_SCOPES: dict[str, str] = {
    "ES_peninsular": "ES0",
    "ES_baleares": "ES1",
    "PT": "PT",
    "FR": "FR",
}
# carriers whose 2024 validation capacity is the mean of year-end 2023 and 2024
AVERAGE_OPERATING_CARRIERS: frozenset[str] = frozenset({"onwind", "solar PV"})

# generator carrier -> fuel price metric family in prices.csv
FUEL_METRIC: dict[str, str] = {
    "CCGT": "gas",
    "OCGT": "gas",
    "coal": "coal",
    "lignite": "lignite",
    "oil": "oil",
    "nuclear": "nuclear",
    "biomass": "biomass",
}
# generator carrier -> VOM row in the processed technology-data costs CSV
VOM_TECH: dict[str, str] = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "coal": "coal",
    "lignite": "lignite",
    "oil": "oil",
    "nuclear": "nuclear",
    "biomass": "biomass",
}

PATCH_ORDER: tuple[str, ...] = (
    "boundary_flows",
    "national_capacities",
    "wind_correction",
    "phs_split",
    "chp_profile",
    "csp",
    "biomass_availability",
    "prices",
    "ntc_caps",
    "demand",
    "hydro_inflow",
    "ma_ad_exports",
)


def _ensure_carrier(n: pypsa.Network, carrier: str, co2: float = 0.0) -> None:
    """Add ``carrier`` to ``n.carriers`` if missing."""
    if carrier not in n.carriers.index:
        n.add("Carrier", carrier, co2_emissions=co2)


def _custom(ctx: Mapping[str, Any], name: str) -> pd.DataFrame:
    """Read a CSV from ``data/custom``."""
    return pd.read_csv(Path(ctx["custom_dir"]) / name)


def _value(df: pd.DataFrame, country: str, year: int, metric: str) -> float | None:
    """Return the value of one (country, year, carrier_or_metric) row, or None."""
    sel = df.query(
        "country == @country and year == @year and carrier_or_metric == @metric"
    )
    return float(sel["value"].iloc[0]) if len(sel) else None


def _country_buses(n: pypsa.Network, prefix: str) -> pd.Index:
    """AC buses whose name starts with ``prefix``."""
    return n.buses.index[
        n.buses.index.str.startswith(prefix) & (n.buses["carrier"] == "AC")
    ]


def _nearest_bus(n: pypsa.Network, buses: pd.Index, lonlat: tuple[float, float]) -> str:
    """Bus in ``buses`` nearest to ``lonlat``."""
    xy = n.buses.loc[buses, ["x", "y"]]
    return str(((xy["x"] - lonlat[0]) ** 2 + (xy["y"] - lonlat[1]) ** 2).idxmin())


def attach_boundary_flows(
    n: pypsa.Network,
    boundary_flows: pd.DataFrame | str | Path,
) -> None:
    """
    Attach fixed hourly external border flows (MWh only) at the nearest French nodes.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify in-place.
    boundary_flows : pd.DataFrame | str | Path
        Measured flows with columns ``['snapshot', 'neighbour', 'flow_mw_into_FR']``
        (positive = import into France), e.g. ``data/custom/fr_boundary_flows_2024.csv``.
    """
    if isinstance(boundary_flows, (str, Path)):
        df = pd.read_csv(boundary_flows)
    else:
        df = boundary_flows.copy()

    required = {"snapshot", "neighbour", "flow_mw_into_FR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Boundary flows table is missing columns: {sorted(missing)}")

    _ensure_carrier(n, "boundary")
    fr_buses = _country_buses(n, "FR")
    if fr_buses.empty:
        raise ValueError("No French AC buses found in network.")

    df["snapshot"] = pd.to_datetime(df["snapshot"]).dt.tz_localize(None)

    for nb, grp in df.groupby("neighbour", sort=False):
        nb_str = str(nb).strip().upper()
        bus = _nearest_bus(n, fr_buses, FR_NEIGHBOUR_COORDS[nb_str])

        hourly = grp.set_index("snapshot")["flow_mw_into_FR"].astype(float)
        step_h = int(n.snapshot_weightings["generators"].iloc[0])
        flow = hourly.resample(f"{step_h}h").mean().reindex(n.snapshots)
        if flow.isna().any():
            raise ValueError(
                f"Boundary flows for {nb_str} do not cover all network snapshots."
            )
        import_mw = flow.where(flow > 0.0, 0.0)
        export_mw = (-flow).where(flow < 0.0, 0.0)

        max_imp = float(import_mw.max())
        if max_imp > 0.0:
            name = f"{bus} boundary import {nb_str}"
            imp_pu = import_mw / max_imp
            n.add(
                "Generator",
                name,
                bus=bus,
                carrier="boundary",
                p_nom=max_imp,
                p_nom_extendable=False,
                p_min_pu=imp_pu,
                p_max_pu=imp_pu,
                marginal_cost=0.0,
            )

        if float(export_mw.max()) > 0.0:
            name = f"{bus} boundary export {nb_str}"
            n.add("Load", name, bus=bus, carrier="boundary", p_set=0.0)
            n.loads_t.p_set[name] = export_mw

        logger.info(
            "Attached FR boundary flow %s at '%s': import %.2f TWh, export %.2f TWh.",
            nb_str,
            bus,
            float((import_mw * n.snapshot_weightings["generators"]).sum()) / 1e6,
            float((export_mw * n.snapshot_weightings["generators"]).sum()) / 1e6,
        )


def freeze_fleet(n: pypsa.Network) -> None:
    """Set every background asset non-extendable (only the DC's assets are buildable)."""
    for comp, attr in (
        ("generators", "p_nom_extendable"),
        ("storage_units", "p_nom_extendable"),
        ("stores", "e_nom_extendable"),
        ("links", "p_nom_extendable"),
        ("lines", "s_nom_extendable"),
    ):
        df = getattr(n, comp)
        if not df.empty and df[attr].any():
            logger.warning(
                "Freezing %d extendable %s from the composed network.",
                int(df[attr].sum()),
                comp,
            )
            df[attr] = False


def _patch_national_capacities(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Set national fleet totals per carrier from national_capacities.csv, pro rata."""
    df = _custom(ctx, "national_capacities.csv")
    year = int(ctx["data_year"])
    for scope, prefix in NATCAP_SCOPES.items():
        buses = set(_country_buses(n, prefix))
        for metric, carriers in NATCAP_GEN_MAP.items():
            target = _value(df, scope, year, metric)
            if target is None:
                continue
            if year == 2024 and metric in AVERAGE_OPERATING_CARRIERS:
                prev = _value(df, scope, year - 1, metric)
                if prev is not None:
                    target = (target + prev) / 2.0
            sel = n.generators.index[
                n.generators["carrier"].isin(carriers) & n.generators["bus"].isin(buses)
            ]
            existing = float(n.generators.loc[sel, "p_nom"].sum())
            if existing > 0.0:
                n.generators.loc[sel, "p_nom"] *= target / existing
                n.generators.loc[sel, "p_nom_min"] = 0.0
            elif target > 0.0:
                load_cols = n.loads.index[n.loads["bus"].isin(buses)].intersection(
                    n.loads_t.p_set.columns
                )
                bus = (
                    n.loads_t.p_set[load_cols]
                    .mean()
                    .groupby(n.loads["bus"])
                    .sum()
                    .idxmax()
                )
                _ensure_carrier(n, carriers[0])
                costs = pd.read_csv(ctx["costs"], index_col=0)
                eta = (
                    float(costs.at[carriers[0], "efficiency"])
                    if carriers[0] in costs.index
                    else 1.0
                )
                n.add(
                    "Generator",
                    f"{bus} {carriers[0]}",
                    bus=bus,
                    carrier=carriers[0],
                    p_nom=target,
                    p_nom_extendable=False,
                    efficiency=eta,
                )
                logger.info(
                    "Created %.0f MW %s at %s (no existing units; efficiency %.3f).",
                    target,
                    carriers[0],
                    bus,
                    eta,
                )
            logger.info("%s %s: %.0f -> %.0f MW", scope, metric, existing, target)
        for metric, su_carrier in NATCAP_SU_MAP.items():
            target = _value(df, scope, year, metric)
            if target is None:
                continue
            sel = n.storage_units.index[
                (n.storage_units["carrier"] == su_carrier)
                & n.storage_units["bus"].isin(buses)
            ]
            existing = float(n.storage_units.loc[sel, "p_nom"].sum())
            if existing > 0.0:
                n.storage_units.loc[sel, "p_nom"] *= target / existing
                logger.info("%s %s: %.0f -> %.0f MW", scope, metric, existing, target)


def _patch_wind_correction(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """
    Scale ES and PT onshore wind profiles for the residual ERA5 wind-speed bias.

    Applied after ``national_capacities`` and the resource-class spatial treatment:
    the remaining deficit is the ERA5 wind-speed bias in Iberian terrain (plus poor
    GEM plant coordinates in Portugal). One multiplicative factor per country;
    profiles are capped at 1.0. France is untouched (its fleet CF validates as is).
    New-build PPA candidates inherit the corrected profiles by construction.

    In the 2024 back-cast (data year = weather year = 2024) the factor is derived
    at run time: cited 2024 generation over the fleet's modelled availability.
    Case-study runs (any other data year) apply the fixed factors recorded in
    ``wind_correction.csv``, which were derived on that back-cast.
    """
    year = int(ctx["data_year"])
    weather_year = int(ctx["weather_year"])
    backcast = year == 2024
    if backcast and weather_year != 2024:
        raise ValueError(
            "The 2024 back-cast derivation requires weather year 2024 "
            f"(got wy{weather_year}); cited generation and modelled availability "
            "must share the weather year."
        )

    w = n.snapshot_weightings["generators"]
    for prefix in ("ES", "PT"):
        gens = n.generators[
            (n.generators["carrier"] == "onwind")
            & n.generators["bus"].isin(set(_country_buses(n, prefix)))
        ]
        cols = gens.index.intersection(n.generators_t.p_max_pu.columns)
        avail_twh = (
            float(
                n.generators_t.p_max_pu[cols]
                .mul(gens.loc[cols, "p_nom"], axis=1)
                .mul(w, axis=0)
                .sum()
                .sum()
            )
            / 1e6
        )
        if backcast:
            targets = {
                "ES": _value(
                    _custom(ctx, "validation_2024_es.csv"),
                    "ES_mainland_baleares",
                    year,
                    "gen_onwind",
                ),
                "PT": _value(_custom(ctx, "pt_actuals.csv"), "PT", year, "gen_onwind"),
            }
            target_twh = targets[prefix]
            if target_twh is None:
                raise ValueError(f"No cited {prefix} wind generation for {year}.")
            factor = target_twh / avail_twh
            source = f"derived from {target_twh:.1f} TWh cited"
        else:
            factor = _value(
                _custom(ctx, "wind_correction.csv"),
                prefix,
                year,
                "wind_correction_factor",
            )
            if factor is None:
                raise ValueError(
                    f"No wind correction factor for {prefix}/{year} in "
                    "wind_correction.csv (derive it on the 2024 back-cast first)."
                )
            source = "fixed back-cast factor"
        scaled = n.generators_t.p_max_pu[cols] * factor
        n.generators_t.p_max_pu[cols] = scaled.where(scaled <= 1.0, 1.0)
        achieved = (
            float(
                n.generators_t.p_max_pu[cols]
                .mul(gens.loc[cols, "p_nom"], axis=1)
                .mul(w, axis=0)
                .sum()
                .sum()
            )
            / 1e6
        )
        logger.info(
            "%s onwind profiles scaled by %.3f (%s): %.1f -> %.1f TWh available.",
            prefix,
            factor,
            source,
            avail_twh,
            achieved,
        )


def _patch_phs_split(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Fix Spanish pure-PHS energy, pumping ratio and round-trip efficiency from cited data."""
    df = _custom(ctx, "national_capacities.csv")
    year = int(ctx["data_year"])
    scope = "ES_mainland_baleares"
    energy_gwh = _value(df, scope, year, "pumped hydro pure energy")
    pumping_mw = _value(df, scope, year, "pumped hydro (pure) pumping")
    rt_eff = _value(df, scope, year, "pumped hydro round-trip efficiency")
    if None in (energy_gwh, pumping_mw, rt_eff):
        raise ValueError(
            f"Missing pure-PHS rows for {scope}/{year} in national_capacities.csv"
        )

    sel = n.storage_units.index[
        (n.storage_units["carrier"] == "PHS")
        & n.storage_units["bus"].isin(set(_country_buses(n, "ES")))
    ]
    p_nom = float(n.storage_units.loc[sel, "p_nom"].sum())
    n.storage_units.loc[sel, "max_hours"] = energy_gwh * 1e3 / p_nom
    n.storage_units.loc[sel, "p_min_pu"] = -pumping_mw / p_nom
    one_way = rt_eff**0.5
    n.storage_units.loc[sel, "efficiency_store"] = one_way
    n.storage_units.loc[sel, "efficiency_dispatch"] = one_way
    logger.info(
        "ES pure PHS: %.0f MW, %.1f GWh (%.1f h), pumping %.0f MW, round-trip %.2f.",
        p_nom,
        energy_gwh,
        energy_gwh * 1e3 / p_nom,
        pumping_mw,
        rt_eff,
    )


def _monthly_chp_shape(ctx: Mapping[str, Any]) -> dict[int, float]:
    """Monthly CHP shape (mean 1.0) from the vendored Redeia monthly cogeneration CO2 series."""
    path = (
        Path(ctx["custom_dir"]) / "sources" / "redata_2024_emisiones_co2_mensual.json"
    )
    data = json.loads(path.read_text())
    series = next(
        item
        for item in data["included"]
        if "ogenera" in item.get("attributes", {}).get("title", "")
    )["attributes"]["values"]
    total = sum(v["value"] for v in series)
    return {
        pd.Timestamp(v["datetime"].split("T")[0]).month: v["value"]
        / (total / len(series))
        for v in series
    }


def _patch_chp_profile(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Spanish cogeneration as a fixed, seasonally shaped profile scaled to Redeia output."""
    df = _custom(ctx, "chp.csv")
    year = int(ctx["data_year"])
    scope = "ES_mainland_baleares"
    cap_mw = _value(df, scope, year, "chp_capacity_mw")
    gen_twh = _value(df, scope, year, "chp_generation_twh")
    em_factor = _value(df, "ES", year, "chp_emission_factor")
    if None in (cap_mw, gen_twh, em_factor):
        raise ValueError(f"Missing CHP rows for {scope}/{year} in chp.csv")

    w = n.snapshot_weightings["generators"]
    monthly = _monthly_chp_shape(ctx)
    shape = pd.Series([monthly[ts.month] for ts in n.snapshots], index=n.snapshots)
    profile = shape * (gen_twh * 1e6 / cap_mw / float((shape * w).sum()))
    if float(profile.max()) >= 1.0:
        raise ValueError(
            "CHP profile exceeds 1.0; check capacity and generation inputs."
        )

    _ensure_carrier(n, "CHP", co2=em_factor)
    load_cols = n.loads.index[
        n.loads["bus"].isin(set(_country_buses(n, "ES")))
        & (n.loads["carrier"] != "boundary")
    ].intersection(n.loads_t.p_set.columns)
    shares = n.loads_t.p_set[load_cols].mean().groupby(n.loads["bus"]).sum()
    shares = shares / shares.sum()
    for bus, share in shares.items():
        name = f"{bus} CHP"
        # fixed dispatch: marginal cost is an objective constant, so it is left at 0
        n.add(
            "Generator",
            name,
            bus=bus,
            carrier="CHP",
            p_nom=cap_mw * share,
            p_nom_extendable=False,
            efficiency=1.0,
            marginal_cost=0.0,
        )
        n.generators_t.p_min_pu[name] = profile
        n.generators_t.p_max_pu[name] = profile
    logger.info(
        "Attached %.0f MW CHP, fixed profile, %.2f TWh/yr, %.3f tCO2/MWh_el.",
        cap_mw,
        gen_twh,
        em_factor,
    )


def _patch_csp(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Concentrated solar power: cited capacity, nodal solar profile rescaled to cited output."""
    natcap = _custom(ctx, "national_capacities.csv")
    val = _custom(ctx, "validation_2024_es.csv")
    year = int(ctx["data_year"])
    scope = "ES_mainland_baleares"
    cap_mw = _value(natcap, scope, year, "CSP/solar thermal")
    gen_2024 = _value(val, scope, 2024, "gen_solar_thermal_csp")
    cap_2024 = _value(natcap, scope, 2024, "CSP/solar thermal")
    if None in (cap_mw, gen_2024, cap_2024):
        raise ValueError("Missing CSP capacity/generation rows in data/custom.")
    target_twh = gen_2024 * cap_mw / cap_2024  # realized 2024 CSP capacity factor

    _ensure_carrier(n, "csp")
    w = n.snapshot_weightings["generators"]
    es_solar = n.generators[
        (n.generators["carrier"] == "solar")
        & n.generators["bus"].isin(set(_country_buses(n, "ES0")))
    ]
    shares = es_solar.groupby("bus")["p_nom"].sum()
    shares = shares / shares.sum()
    profiles = {
        bus: n.generators_t.p_max_pu[es_solar.index[es_solar["bus"] == bus][0]]
        for bus in shares.index
    }
    avail_mwh = sum(
        float((profiles[bus] * w).sum()) * cap_mw * share
        for bus, share in shares.items()
    )
    k = target_twh * 1e6 / avail_mwh
    for bus, share in shares.items():
        name = f"{bus} csp"
        n.add(
            "Generator",
            name,
            bus=bus,
            carrier="csp",
            p_nom=cap_mw * share,
            p_nom_extendable=False,
            marginal_cost=0.0,
        )
        scaled = profiles[bus] * k
        n.generators_t.p_max_pu[name] = scaled.where(scaled <= 1.0, 1.0)
    logger.info(
        "Attached %.0f MW CSP, solar-shaped profile, %.2f TWh/yr target.",
        cap_mw,
        target_twh,
    )


def _patch_biomass_availability(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """
    Cap biomass/waste availability at the realized capacity factor from cited rows.

    Spanish biomass and waste plants run fuel-limited at roughly 40-50% of capacity;
    an unconstrained LP would dispatch them near-baseload. The cap is derived from
    the cited annual output and installed capacity, not tuned.
    """
    year = int(ctx["data_year"])
    natcap = _custom(ctx, "national_capacities.csv")
    val = _custom(ctx, "validation_2024_es.csv")
    es_gen = sum(
        _value(val, "ES_mainland_baleares", 2024, m) or 0.0
        for m in (
            "gen_other_renewables",
            "gen_renewable_waste",
            "gen_nonrenewable_waste",
        )
    )
    cases = {
        "ES": (_value(natcap, "ES_mainland_baleares", year, "biomass/waste"), es_gen),
        "PT": (
            _value(natcap, "PT", year, "biomass/waste"),
            _value(_custom(ctx, "pt_actuals.csv"), "PT", 2024, "gen_biomass_waste"),
        ),
    }
    hours = float(n.snapshot_weightings["generators"].sum())
    for prefix, (cap, gen_twh) in cases.items():
        if cap is None or gen_twh is None or cap <= 0.0:
            logger.info("No biomass capacity/generation rows for %s; skipped.", prefix)
            continue
        availability = gen_twh * 1e6 / (cap * hours)
        sel = n.generators.index[
            n.generators["carrier"].isin(("biomass", "waste"))
            & n.generators["bus"].isin(set(_country_buses(n, prefix)))
        ]
        n.generators.loc[sel, "p_max_pu"] = availability
        logger.info(
            "%s biomass/waste availability capped at %.3f (%.2f TWh / %.0f MW).",
            prefix,
            availability,
            gen_twh,
            cap,
        )


def _patch_prices(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Recompute thermal marginal costs: VOM + fuel/eta + ETS x thermal intensity/eta."""
    prices = _custom(ctx, "prices.csv")
    year = int(ctx["data_year"])
    costs = pd.read_csv(ctx["costs"], index_col=0)

    def price(metrics: tuple[str, ...]) -> float:
        for y in (year, 2025):
            for metric in metrics:
                for country in ("EU", "ES"):
                    v = _value(prices, country, y, metric)
                    if v is not None:
                        return v
        raise ValueError(f"No price row found for {metrics}.")

    ets = price(("co2_ets_price",))
    fuel = {
        "gas": price(("gas_ttf_price", "gas_fuel_price")),
        "coal": price(("coal_fuel_price",)),
        "lignite": price(("lignite_fuel_price",)),
        "oil": price(("oil_fuel_price",)),
        "nuclear": price(("nuclear_fuel_price",)),
        "biomass": price(("biomass_fuel_price",)),
    }
    co2 = n.carriers["co2_emissions"]
    for carrier, fuel_key in FUEL_METRIC.items():
        sel = n.generators.index[n.generators["carrier"] == carrier]
        if sel.empty:
            continue
        eta = n.generators.loc[sel, "efficiency"].replace(0.0, 1.0)
        vom = (
            float(costs.at[VOM_TECH[carrier], "VOM"])
            if VOM_TECH[carrier] in costs.index
            else 0.0
        )
        n.generators.loc[sel, "marginal_cost"] = (
            vom + (fuel[fuel_key] + ets * float(co2.get(carrier, 0.0))) / eta
        )
        logger.info(
            "%s marginal cost -> %.1f-%.1f EUR/MWh (fuel %.2f, ETS %.2f).",
            carrier,
            float(n.generators.loc[sel, "marginal_cost"].min()),
            float(n.generators.loc[sel, "marginal_cost"].max()),
            fuel[fuel_key],
            ets,
        )


def _cap_corridor(
    n: pypsa.Network, buses_a: set[str], buses_b: set[str], ntc_mw: float, label: str
) -> None:
    """Scale border line s_max_pu and link p_max_pu so corridor capability equals the NTC."""
    lines = n.lines.index[
        (n.lines["bus0"].isin(buses_a) & n.lines["bus1"].isin(buses_b))
        | (n.lines["bus0"].isin(buses_b) & n.lines["bus1"].isin(buses_a))
    ]
    links = n.links.index[
        (
            (n.links["bus0"].isin(buses_a) & n.links["bus1"].isin(buses_b))
            | (n.links["bus0"].isin(buses_b) & n.links["bus1"].isin(buses_a))
        )
        & (n.links["carrier"] != "boundary")
    ]
    capability = float(
        (n.lines.loc[lines, "s_nom"] * n.lines.loc[lines, "s_max_pu"]).sum()
        + (n.links.loc[links, "p_nom"] * n.links.loc[links, "p_max_pu"]).sum()
    )
    if capability <= 0.0:
        logger.warning("Corridor %s has no capacity; skipping NTC cap.", label)
        return
    factor = ntc_mw / capability
    if factor >= 1.0:
        logger.info(
            "Corridor %s capability %.0f MW <= NTC %.0f MW; no cap.",
            label,
            capability,
            ntc_mw,
        )
        return
    n.lines.loc[lines, "s_max_pu"] *= factor
    n.links.loc[links, "p_max_pu"] *= factor
    n.links.loc[links, "p_min_pu"] *= factor
    logger.info(
        "Corridor %s capped: %.0f -> %.0f MW (factor %.3f).",
        label,
        capability,
        ntc_mw,
        factor,
    )


def _patch_ntc_caps(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Cap ES-FR, ES-PT and Peninsula-Balearics corridors to published NTC values."""
    df = _custom(ctx, "ntc.csv")
    year = int(ctx["data_year"])
    es0 = set(_country_buses(n, "ES0"))
    corridors = [
        (
            es0,
            set(_country_buses(n, "FR")),
            _value(df, "ES-FR", year, "ntc_es_to_fr"),
            "ES-FR",
        ),
        (
            es0,
            set(_country_buses(n, "PT")),
            _value(df, "ES-PT", year, "ntc_es_to_pt"),
            "ES-PT",
        ),
        (
            es0,
            set(_country_buses(n, "ES1")),
            _value(df, "ES1-ES4", year, "ntc_peninsula_to_baleares"),
            "ES1-ES4",
        ),
    ]
    for buses_a, buses_b, ntc, label in corridors:
        if ntc is None:
            raise ValueError(f"Missing NTC row for {label}/{year} in ntc.csv")
        _cap_corridor(n, buses_a, buses_b, ntc, label)


def _patch_demand(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Scale each country's non-boundary load to the cited annual busbar demand."""
    df = _custom(ctx, "annual_demand.csv")
    year = int(ctx["data_year"])
    targets = {
        "ES": _value(df, "ES_mainland_baleares", year, "electricity_demand_busbar"),
        "PT": _value(df, "PT", year, "electricity_demand_busbar"),
        "FR": _value(df, "FR", year, "electricity_demand_corrected"),
    }
    w = n.snapshot_weightings["generators"]
    for prefix, target_twh in targets.items():
        if target_twh is None:
            raise ValueError(
                f"Missing demand row for {prefix}/{year} in annual_demand.csv"
            )
        cols = n.loads.index[
            n.loads["bus"].isin(set(_country_buses(n, prefix)))
            & (n.loads["carrier"] != "boundary")
        ].intersection(n.loads_t.p_set.columns)
        current = float(n.loads_t.p_set[cols].mul(w, axis=0).sum().sum()) / 1e6
        n.loads_t.p_set[cols] *= target_twh / current
        logger.info("%s demand scaled %.1f -> %.1f TWh.", prefix, current, target_twh)


def _patch_hydro_inflow(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """
    Scale ES/PT reservoir inflow to the hydro energy observed in the weather year.

    A weather-measurement closure, keyed on the run's weather year (not the data
    vintage): the upstream EIA inflow normalization ends before 2024, so wy2024 runs
    are rescaled to the observed wet-2024 energy whatever fleet year they carry.
    Other weather years are covered by EIA and are skipped.
    """
    weather_year = int(ctx["weather_year"])
    targets = {
        "ES": _value(
            _custom(ctx, "validation_2024_es.csv"),
            "ES_mainland_baleares",
            weather_year,
            "gen_hydro",
        ),
        "PT": _value(_custom(ctx, "pt_actuals.csv"), "PT", weather_year, "gen_hydro"),
    }
    w = n.snapshot_weightings["generators"]
    for prefix, target in targets.items():
        if target is None:
            logger.info(
                "No observed hydro energy for %s/wy%d; skipped.", prefix, weather_year
            )
            continue
        buses = set(_country_buses(n, prefix))
        su = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") & n.storage_units["bus"].isin(buses)
        ].intersection(n.storage_units_t.inflow.columns)
        inflow_twh = (
            float(n.storage_units_t.inflow[su].mul(w, axis=0).sum().sum()) / 1e6
        )
        ror = n.generators[
            (n.generators["carrier"] == "ror") & n.generators["bus"].isin(buses)
        ]
        ror_twh = (
            float(
                n.generators_t.p_max_pu[
                    ror.index.intersection(n.generators_t.p_max_pu.columns)
                ]
                .mul(ror["p_nom"], axis=1)
                .mul(w, axis=0)
                .sum()
                .sum()
            )
            / 1e6
        )
        factor = (target - ror_twh) / inflow_twh
        n.storage_units_t.inflow[list(su)] *= factor
        logger.info(
            "%s reservoir inflow scaled by %.3f: %.1f (+%.1f ror) -> %.1f TWh observed.",
            prefix,
            factor,
            inflow_twh,
            ror_twh,
            target,
        )


def _patch_ma_ad_exports(n: pypsa.Network, ctx: Mapping[str, Any]) -> None:
    """Fixed ES exports to Morocco and Andorra as flat loads at the nearest nodes."""
    val = _custom(ctx, "validation_2024_es.csv")
    _ensure_carrier(n, "boundary")
    hours = float(n.snapshot_weightings["generators"].sum())
    es_buses = _country_buses(n, "ES0")
    for code, metric in (("MA", "net_export_ma"), ("AD", "net_export_ad")):
        twh = _value(val, "ES", 2024, metric)
        if twh is None:
            raise ValueError(f"Missing {metric} row in validation_2024_es.csv")
        bus = _nearest_bus(n, es_buses, ES_BORDER_COORDS[code])
        name = f"{bus} boundary export {code}"
        n.add("Load", name, bus=bus, carrier="boundary", p_set=twh * 1e6 / hours)
        logger.info("Attached fixed ES->%s export %.2f TWh at %s.", code, twh, bus)


PATCH_FUNCTIONS: dict[str, Callable[[pypsa.Network, Mapping[str, Any]], None]] = {
    "boundary_flows": lambda n, ctx: attach_boundary_flows(n, ctx["boundary_flows"]),
    "national_capacities": _patch_national_capacities,
    "wind_correction": _patch_wind_correction,
    "phs_split": _patch_phs_split,
    "chp_profile": _patch_chp_profile,
    "csp": _patch_csp,
    "biomass_availability": _patch_biomass_availability,
    "prices": _patch_prices,
    "ntc_caps": _patch_ntc_caps,
    "demand": _patch_demand,
    "hydro_inflow": _patch_hydro_inflow,
    "ma_ad_exports": _patch_ma_ad_exports,
}


def prepare_network(
    n: pypsa.Network,
    patches: Mapping[str, bool],
    ctx: Mapping[str, Any],
) -> pypsa.Network:
    """Apply enabled patches in ``PATCH_ORDER``, then freeze the background fleet."""
    unknown = set(patches) - set(PATCH_ORDER)
    if unknown:
        raise ValueError(f"Unknown patches: {sorted(unknown)}")
    for name in PATCH_ORDER:
        if patches.get(name, False):
            logger.info("Applying patch '%s'.", name)
            PATCH_FUNCTIONS[name](n, ctx)
    freeze_fleet(n)
    return n


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network", type=Path, required=True, help="Composed network (.nc)."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Base network output (.nc)."
    )
    parser.add_argument(
        "--patches",
        type=str,
        default="boundary_flows",
        help="Comma-separated list of enabled patches.",
    )
    parser.add_argument(
        "--boundary-flows",
        type=Path,
        default=Path("data/custom/fr_boundary_flows_2024.csv"),
        help="Measured FR border flows CSV.",
    )
    parser.add_argument("--custom-dir", type=Path, default=Path("data/custom"))
    parser.add_argument(
        "--costs", type=Path, default=None, help="Processed technology costs CSV."
    )
    parser.add_argument(
        "--data-year",
        type=int,
        default=2025,
        help="Vintage of the cited data rows (2024 = back-cast, 2025 = case study).",
    )
    parser.add_argument(
        "--weather-year",
        type=int,
        default=2024,
        help="Weather year of the run (keys the weather-measurement patches).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point supporting both Snakemake execution and standalone CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if "snakemake" in globals():
        smk = globals()["snakemake"]
        net_in = Path(smk.input.network)
        net_out = Path(smk.output.network)
        enabled = [p.strip() for p in str(smk.params.patches).split(",") if p.strip()]
        ctx: dict[str, Any] = {
            "boundary_flows": Path(smk.input.boundary_flows),
            "custom_dir": Path(smk.params.custom_dir),
            "costs": Path(smk.input.costs),
            "data_year": int(smk.wildcards.year),
            "weather_year": int(smk.params.weather_year),
        }
    else:
        args = _parse_args(argv)
        net_in = args.network
        net_out = args.output
        enabled = [p.strip() for p in args.patches.split(",") if p.strip()]
        ctx = {
            "boundary_flows": args.boundary_flows,
            "custom_dir": args.custom_dir,
            "costs": args.costs,
            "data_year": args.data_year,
            "weather_year": args.weather_year,
        }

    n = pypsa.Network(str(net_in))
    prepare_network(n, patches=dict.fromkeys(enabled, True), ctx=ctx)
    net_out.parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(str(net_out))
    logger.info("Exported base network to %s", net_out)


if __name__ == "__main__":
    main()
