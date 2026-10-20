# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Postprocess one solved scenario into the national and data center metrics.

Appends to the scenario's solve metrics row what the study figures need, read
from the solved network without re-solving:

- Spanish system: generation by carrier, net imports from France and Portugal,
  fuel use of gas-fired power plants (CCGT, OCGT; cogeneration runs a fixed
  profile and is excluded), generation-based CO2 intensity, and system cost
  (operating cost of Spanish assets plus the data center's annuities; the
  frozen background fleet's capital cost is constant and left out).
- Data center: PPA cost, output and curtailment by carrier, battery cost, and the
  attributional CO2 intensity of its consumption: grid purchases at the
  import-traced hourly grid emission factor, PPA supply at zero.
- Hourly series of the data center: CFE score and attributional CO2 intensity
  of its consumption, for the heatmaps.

Grid purchases are rebuilt from the physical balance,
``im_t = max(d_t - g_t - dis_t + cha_t, 0)``. This is exact because the sales
cost in ``cfe_constraints`` rules out buying and selling in the same snapshot;
the rebuilt total is checked against the solved one.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pypsa

from scripts.calculate_grid_cfe import (
    calculate_grid_cfe,
    calculate_grid_emission_factor,
)
from scripts.cfe_constraints import yield_per_mw

logger = logging.getLogger(__name__)

GAS_POWER_CARRIERS: tuple[str, ...] = ("CCGT", "OCGT")
STORAGE_CARRIERS: frozenset[str] = frozenset({"PHS", "battery"})
NON_GENERATION_CARRIERS: frozenset[str] = frozenset({"load", "boundary"})
G_PER_KWH: float = 1.0e3  # per tCO2/MWh


def _es_buses(n: pypsa.Network) -> pd.Index:
    return n.buses.index[(n.buses["country"] == "ES") & (n.buses["carrier"] == "AC")]


def _is_dc(index: pd.Index) -> pd.Series:
    return pd.Series(index.astype(str).str.startswith("DC"), index=index)


def _weighted_sum(df: pd.DataFrame, w: pd.Series) -> pd.Series:
    return df.mul(w, axis=0).sum()


def spanish_generation(n: pypsa.Network) -> pd.Series:
    """Annual Spanish background generation (MWh) by carrier; storage units net."""
    w = n.snapshot_weightings["generators"]
    gens = n.generators.index[
        n.generators["bus"].isin(_es_buses(n))
        & ~n.generators["carrier"].isin(NON_GENERATION_CARRIERS)
        & ~_is_dc(n.generators.index)
    ]
    sus = n.storage_units.index[n.storage_units["bus"].isin(_es_buses(n))]
    gen = _weighted_sum(n.generators_t.p[gens], w)
    su = _weighted_sum(n.storage_units_t.p[sus], w)
    by_carrier = pd.concat(
        [
            gen.groupby(n.generators.loc[gens, "carrier"]).sum(),
            su.groupby(n.storage_units.loc[sus, "carrier"]).sum(),
        ]
    )
    return by_carrier.groupby(level=0).sum()


def spanish_net_imports(n: pypsa.Network) -> float:
    """Annual net imports into Spain (MWh) over all cross-border lines and links."""
    w = n.snapshot_weightings["generators"]
    es = set(_es_buses(n))
    ac = set(n.buses.index[n.buses["carrier"] == "AC"])
    total = 0.0
    for df, dyn in ((n.lines, n.lines_t), (n.links, n.links_t)):
        cross = df["bus0"].isin(ac) & df["bus1"].isin(ac)
        into = df.index[cross & df["bus1"].isin(es) & ~df["bus0"].isin(es)]
        out = df.index[cross & df["bus0"].isin(es) & ~df["bus1"].isin(es)]
        total -= float(_weighted_sum(dyn.p1[into], w).sum())
        total -= float(_weighted_sum(dyn.p0[out], w).sum())
    return total


def spanish_system_metrics(n: pypsa.Network, row: pd.Series) -> dict[str, float]:
    """Spanish generation mix, net imports, gas fuel, CO2 intensity and system cost."""
    w = n.snapshot_weightings["generators"]
    es = _es_buses(n)
    m: dict[str, float] = {}

    background = spanish_generation(n)
    for carrier, mwh in background.items():
        m[f"es_gen_{carrier}_mwh"] = float(mwh)
    m["es_net_import_mwh"] = spanish_net_imports(n)

    gas = n.generators.index[
        n.generators["bus"].isin(es) & n.generators["carrier"].isin(GAS_POWER_CARRIERS)
    ]
    fuel = n.generators_t.p[gas] / n.generators.loc[gas, "efficiency"]
    m["es_gas_fuel_mwh_th"] = float(_weighted_sum(fuel, w).sum())

    generation = background.drop(STORAGE_CARRIERS, errors="ignore").sum()
    generation += float(row["ppa_generation_mwh"])
    m["es_generation_mwh"] = float(generation)
    m["es_co2_intensity_g_per_kwh"] = (
        float(row["es_total_co2_mt"]) * 1.0e12 / (generation * 1.0e3)
    )

    gens = n.generators.index[n.generators["bus"].isin(es)]
    sus = n.storage_units.index[n.storage_units["bus"].isin(es)]
    gen_mc = n.get_switchable_as_dense("Generator", "marginal_cost")[gens]
    su_mc = n.get_switchable_as_dense("StorageUnit", "marginal_cost")[sus]
    gen_cost = n.generators_t.p[gens] * gen_mc
    su_cost = n.storage_units_t.p[sus] * su_mc
    operating = float(
        _weighted_sum(gen_cost, w).sum() + _weighted_sum(su_cost, w).sum()
    )
    m["es_system_cost_eur"] = operating + float(row["dc_capital_cost_eur"])
    return m


def dc_portfolio_metrics(n: pypsa.Network, row: pd.Series) -> dict[str, float]:
    """
    PPA cost, curtailment and contracted yield, and battery cost.

    ``annual_res_score`` is the contracted PPA yield (output whether used, sold or
    curtailed) per unit of DC consumption, the volume the annual policy constrains.
    """
    w = n.snapshot_weightings["generators"]
    gens = n.generators[
        _is_dc(n.generators.index) & (n.generators["carrier"] != "load")
    ]
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu")
    marginal_cost = n.get_switchable_as_dense("Generator", "marginal_cost")
    m: dict[str, float] = {}
    for carrier, grp in gens.groupby("carrier"):
        p = n.generators_t.p[grp.index]
        available = p_max_pu[grp.index] * grp["p_nom_opt"]
        energy_cost = float(_weighted_sum(p * marginal_cost[grp.index], w).sum())
        m[f"dc_ppa_{carrier}_cost_eur"] = (
            float((grp["p_nom_opt"] * grp["capital_cost"]).sum()) + energy_cost
        )
        m[f"dc_ppa_{carrier}_curtailed_mwh"] = float(
            _weighted_sum(available - p, w).sum()
        )
    m["ppa_yield_mwh"] = float(
        (yield_per_mw(n, list(gens.index)) * gens["p_nom_opt"]).sum()
    )
    load = float(row["total_load_mwh"])
    m["annual_res_score"] = m["ppa_yield_mwh"] / load if load > 0.0 else 0.0
    stores = n.stores[_is_dc(n.stores.index)]
    links = n.links[_is_dc(n.links.index)]
    m["dc_battery_cost_eur"] = float(
        (stores["e_nom_opt"] * stores["capital_cost"]).sum()
        + (links["p_nom_opt"] * links["capital_cost"]).sum()
    )
    return m


def dc_grid_purchases(n: pypsa.Network) -> tuple[pd.Series, pd.Series]:
    """Hourly DC load ``d_t`` and grid purchases ``im_t`` rebuilt from the balance."""
    loads = n.loads.index[_is_dc(n.loads.index)]
    gens = n.generators.index[
        _is_dc(n.generators.index) & (n.generators["carrier"] != "load")
    ]
    links = n.links.index[_is_dc(n.links.index)].astype(str)
    dischargers = links[links.str.contains("discharger")]
    chargers = links[links.str.contains("charger") & ~links.str.contains("discharger")]
    d = n.loads_t.p[loads].sum(axis=1)
    discharge = n.links_t.p0[dischargers] * n.links.loc[dischargers, "efficiency"]
    net = (
        d
        - n.generators_t.p[gens].sum(axis=1)
        - discharge.sum(axis=1)
        + n.links_t.p0[chargers].sum(axis=1)
    )
    return d, net.where(net > 0.0, 0.0)


def dc_attributional_metrics(n: pypsa.Network, row: pd.Series) -> dict[str, float]:
    """Attributional CO2 intensity of DC consumption and the Spanish grid average."""
    w = n.snapshot_weightings["generators"]
    ef = calculate_grid_emission_factor(n, country="ES")
    d, im = dc_grid_purchases(n)

    rebuilt = float((w * im).sum())
    solved = float(row["grid_import_mwh"])
    if abs(rebuilt - solved) > 1.0e-6 * max(solved, 1.0) + 1.0:
        raise ValueError(
            f"Rebuilt grid purchases {rebuilt:.1f} MWh differ from the solved "
            f"{solved:.1f} MWh: same-snapshot buying and selling in the solution?"
        )

    es_load = n.loads_t.p[
        n.loads.index[
            n.loads["bus"].isin(_es_buses(n))
            & (n.loads["carrier"] != "boundary")
            & ~_is_dc(n.loads.index)
        ]
    ].sum(axis=1)
    total_load = float((w * d).sum())
    intensity = float((w * im * ef).sum()) / total_load if total_load > 0.0 else 0.0
    return {
        "dc_attributional_intensity_g_per_kwh": intensity * G_PER_KWH,
        "es_grid_ef_g_per_kwh": float((w * es_load * ef).sum() / (w * es_load).sum())
        * G_PER_KWH,
    }


def dc_hourly(n: pypsa.Network) -> pd.DataFrame:
    """
    Hourly CFE score and attributional CO2 intensity of DC consumption.

    The CFE score is the share of consumption covered by PPA and battery supply
    plus the renewable share of grid purchases, ``1 - im_t (1 - CFE_t) / d_t``;
    the intensity values the purchases at the import-traced grid emission factor.
    Their consumption-weighted means are ``clean_coverage`` and
    ``dc_attributional_intensity_g_per_kwh``. Both are NaN without a data center.
    """
    d, im = dc_grid_purchases(n)
    bought = im / d
    cfe = calculate_grid_cfe(n, country="ES")
    ef = calculate_grid_emission_factor(n, country="ES")
    return pd.DataFrame(
        {
            "cfe_score": 1.0 - bought * (1.0 - cfe),
            "intensity_g_per_kwh": bought * ef * G_PER_KWH,
        }
    )


def summarise_network(n: pypsa.Network, row: pd.Series) -> pd.Series:
    """
    Return the solve metrics ``row`` extended by the postprocessing metrics.

    Postprocessing owns its metric names: where a solve row from an older run
    carries the same name, the definition computed here replaces it.
    """
    extra = {
        **spanish_system_metrics(n, row),
        **dc_portfolio_metrics(n, row),
        **dc_attributional_metrics(n, row),
    }
    return pd.concat([row.drop(list(extra), errors="ignore"), pd.Series(extra)])


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: one solved scenario into a summary row and hourly DC series."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-hourly", type=Path, required=True)
    args = parser.parse_args(argv)

    n = pypsa.Network(str(args.network))
    row = pd.read_csv(args.metrics).iloc[0]
    summary = summarise_network(n, row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_frame().T.to_csv(args.output, index=False)
    args.output_hourly.parent.mkdir(parents=True, exist_ok=True)
    dc_hourly(n).to_csv(args.output_hourly)
    logger.info("Summarised %s -> %s", args.network, args.output)


if __name__ == "__main__":
    main()
