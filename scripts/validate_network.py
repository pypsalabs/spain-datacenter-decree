# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT
"""
Validate a solved no-DC network against published actuals in one report.

Compares the solved 2024 ``no_dc`` network with Redeia actuals
(``data/custom/validation_2024_es.csv``) for the modelled Spanish scope
(peninsula + Balearics): generation per carrier, demand, installed wind/solar
capacity factors, pumped-hydro use, power-sector CO2 and average emission
intensity, cross-border flows, and prices. Writes one tidy CSV
(metric / unit / model / actual / diff / gate / status) and a YAML summary of
the headline gates. The same report runs on the raw and the patched variants,
so the diff between reports attributes each patch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pypsa
import yaml

from scripts.calculate_grid_res import _corridor_import

logger = logging.getLogger(__name__)

SCOPE = "ES_mainland_baleares"

# model carriers contributing to each Redeia generation metric
GEN_CARRIERS: dict[str, tuple[str, ...]] = {
    "gen_onwind": ("onwind", "offwind-ac", "offwind-dc", "offwind-float"),
    "gen_solar_pv": ("solar", "solar-hsat"),
    "gen_solar_thermal_csp": ("csp",),
    "gen_nuclear": ("nuclear",),
    "gen_hydro": ("hydro", "ror"),
    "gen_ccgt": ("CCGT",),
    "gen_chp": ("CHP",),
    "gen_coal": ("coal", "lignite"),
    "gen_oil_diesel_gt_st": ("oil", "OCGT"),
    "gen_biomass_waste": ("biomass", "waste"),
}
# Redeia metrics summed into the model's single biomass+waste bucket
BIOMASS_WASTE_ACTUALS = (
    "gen_other_renewables",
    "gen_renewable_waste",
    "gen_nonrenewable_waste",
)

# (metric, relative tolerance) headline gates; net exports additionally checks the sign
GATES: dict[str, float] = {
    "gen_onwind": 0.15,
    "gen_solar_pv": 0.15,
    "gen_nuclear": 0.10,
    "gen_hydro": 0.20,
    "gen_ccgt_plus_chp": 0.25,
    "co2_power_sector_total": 0.20,
    "net_export_total": 0.50,
    "load_shedding_share": 1.0e-4,
}


def _es_generation_twh(n: pypsa.Network) -> dict[str, float]:
    """Return annual ES generation per Redeia metric (TWh) from generators and storage units."""
    w = n.snapshot_weightings["generators"]
    es_buses = n.buses.index[
        (n.buses["country"] == "ES") & (n.buses["carrier"] == "AC")
    ]
    out: dict[str, float] = {}

    gen = n.generators[n.generators["bus"].isin(es_buses)]
    gen_twh = (
        n.generators_t.p[gen.index.intersection(n.generators_t.p.columns)]
        .mul(w, axis=0)
        .sum()
        .groupby(gen["carrier"])
        .sum()
        / 1e6
    )
    su = n.storage_units[n.storage_units["bus"].isin(es_buses)]
    su_p = n.storage_units_t.p[su.index.intersection(n.storage_units_t.p.columns)]
    su_supply_twh = (
        su_p.where(su_p > 0.0, 0.0).mul(w, axis=0).sum().groupby(su["carrier"]).sum()
        / 1e6
    )
    su_withdraw_twh = (
        su_p.where(su_p < 0.0, 0.0).mul(w, axis=0).sum().groupby(su["carrier"]).sum()
        / 1e6
    )
    supply = gen_twh.add(su_supply_twh, fill_value=0.0)

    for metric, carriers in GEN_CARRIERS.items():
        out[metric] = float(supply.reindex(list(carriers)).fillna(0.0).sum())
    out["gen_ccgt_plus_chp"] = out["gen_ccgt"] + out["gen_chp"]
    out["gen_phs_turbining"] = float(su_supply_twh.get("PHS", 0.0))
    out["demand_phs_pumping"] = float(su_withdraw_twh.get("PHS", 0.0))
    out["load_shedding"] = float(gen_twh.get("load", 0.0))
    return out


def _es_co2_mt(n: pypsa.Network) -> float:
    """Power-sector CO2 of the Spanish fleet (Mt), thermal input x carrier intensity."""
    w = n.snapshot_weightings["generators"]
    co2 = n.carriers["co2_emissions"]
    gens = n.generators[
        n.generators["bus"].str.startswith("ES")
        & (n.generators["carrier"].map(co2).fillna(0.0) > 0.0)
    ]
    cols = gens.index.intersection(n.generators_t.p.columns)
    gen_mwh = n.generators_t.p[cols].mul(w, axis=0).sum()
    eff = gens.loc[cols, "efficiency"].replace(0.0, 1.0)
    return float((gen_mwh / eff * gens.loc[cols, "carrier"].map(co2)).sum()) / 1e6


def _es_demand_twh(n: pypsa.Network) -> float:
    """Annual busbar demand of ES loads (TWh), excluding fixed boundary export loads."""
    w = n.snapshot_weightings["generators"]
    es_buses = n.buses.index[
        (n.buses["country"] == "ES") & (n.buses["carrier"] == "AC")
    ]
    loads = n.loads.index[
        n.loads["bus"].isin(es_buses) & (n.loads["carrier"] != "boundary")
    ].intersection(n.loads_t.p_set.columns)
    return float(n.loads_t.p_set[loads].mul(w, axis=0).sum().sum()) / 1e6


def _net_exports_twh(n: pypsa.Network) -> dict[str, float]:
    """Net ES exports (TWh, positive = export) to FR and PT plus fixed MA/AD boundary loads."""
    w = n.snapshot_weightings["generators"]
    out: dict[str, float] = {}
    for nb in ("FR", "PT"):
        imp = _corridor_import(n, exporter=nb, importer="ES", exclude_assets_set=set())
        exp = _corridor_import(n, exporter="ES", importer=nb, exclude_assets_set=set())
        out[f"net_export_{nb.lower()}"] = float(((exp - imp) * w).sum()) / 1e6
    es_buses = n.buses.index[n.buses["country"] == "ES"]
    b_loads = n.loads.index[
        n.loads["bus"].isin(es_buses) & (n.loads["carrier"] == "boundary")
    ]
    ts = b_loads.intersection(n.loads_t.p_set.columns)
    static = b_loads.difference(ts)
    out["net_export_ma_ad"] = (
        float(n.loads_t.p_set[ts].mul(w, axis=0).sum().sum())
        + float(n.loads.loc[static, "p_set"].sum()) * float(w.sum())
    ) / 1e6
    out["net_export_total"] = sum(out.values())
    return out


def _es_mean_price(n: pypsa.Network) -> float:
    """Load-weighted mean Spanish electricity price (EUR/MWh)."""
    es_buses = n.buses.index[
        (n.buses["country"] == "ES") & (n.buses["carrier"] == "AC")
    ]
    loads = n.loads.index[
        n.loads["bus"].isin(es_buses) & (n.loads["carrier"] != "boundary")
    ].intersection(n.loads_t.p_set.columns)
    per_bus = n.loads_t.p_set[loads].T.groupby(n.loads.loc[loads, "bus"]).sum().T
    prices = n.buses_t.marginal_price[per_bus.columns]
    return float((prices * per_bus).sum().sum() / per_bus.sum().sum())


def _wind_solar_cf(n: pypsa.Network) -> dict[str, float]:
    """Fleet capacity factors of Spanish wind and solar (available, pre-curtailment)."""
    w = n.snapshot_weightings["generators"]
    hours = float(w.sum())
    out = {}
    for name, carriers in (
        ("cf_onwind", ("onwind",)),
        ("cf_solar_pv", ("solar", "solar-hsat")),
    ):
        gens = n.generators[
            n.generators["bus"].str.startswith("ES")
            & n.generators["carrier"].isin(carriers)
        ]
        cols = gens.index.intersection(n.generators_t.p_max_pu.columns)
        p_nom = gens.loc[cols, "p_nom"]
        avail_mwh = float(
            n.generators_t.p_max_pu[cols].mul(p_nom, axis=1).mul(w, axis=0).sum().sum()
        )
        out[name] = avail_mwh / (float(p_nom.sum()) * hours) if p_nom.sum() > 0 else 0.0
    return out


def validate(n: pypsa.Network, actuals: pd.DataFrame) -> pd.DataFrame:
    """Build the tidy validation table: model vs Redeia, with gates."""
    act = actuals.query("country == @SCOPE").set_index("carrier_or_metric")["value"]
    act_nat = actuals.query("country == 'ES'").set_index("carrier_or_metric")["value"]

    model: dict[str, float] = {}
    model.update(_es_generation_twh(n))
    model["demand_b_c"] = _es_demand_twh(n)
    model["co2_power_sector_total"] = _es_co2_mt(n)
    model["emission_intensity_kg_mwh"] = (
        model["co2_power_sector_total"] / model["demand_b_c"] * 1e3
    )
    model.update(_net_exports_twh(n))
    model["day_ahead_price_omie_avg"] = _es_mean_price(n)
    model.update(_wind_solar_cf(n))
    model["load_shedding_share"] = (
        model.pop("load_shedding") / model["demand_b_c"]
        if model["demand_b_c"] > 0
        else 0.0
    )

    actual: dict[str, float] = {}
    for metric in model:
        if metric == "gen_biomass_waste":
            actual[metric] = float(act.reindex(BIOMASS_WASTE_ACTUALS).fillna(0.0).sum())
        elif metric == "gen_ccgt_plus_chp":
            actual[metric] = float(act.get("gen_ccgt", 0.0) + act.get("gen_chp", 0.0))
        elif metric in ("net_export_ma_ad",):
            actual[metric] = float(
                act_nat.get("net_export_ma", 0.0) + act_nat.get("net_export_ad", 0.0)
            )
        elif metric in (
            "gen_phs_turbining",
            "demand_phs_pumping",
            "day_ahead_price_omie_avg",
        ):
            actual[metric] = float(act_nat.get(metric, float("nan")))
        else:
            actual[metric] = float(act.get(metric, act_nat.get(metric, float("nan"))))
    actual["load_shedding_share"] = 0.0
    actual["emission_intensity_kg_mwh"] = (
        actual["co2_power_sector_total"] / actual["demand_b_c"] * 1e3
    )

    rows = []
    for metric, m_val in model.items():
        a_val = actual[metric]
        diff = m_val - a_val
        pct = diff / abs(a_val) * 100.0 if a_val and pd.notna(a_val) else float("nan")
        gate = GATES.get(metric)
        if gate is None:
            status = "INFO"
        elif metric == "load_shedding_share":
            status = "PASS" if m_val < gate else "FAIL"
        elif metric == "net_export_total":
            sign_ok = (m_val >= 0) == (a_val >= 0)
            status = "PASS" if sign_ok and abs(pct) <= gate * 100.0 else "FAIL"
        else:
            status = "PASS" if abs(pct) <= gate * 100.0 else "FAIL"
        rows.append(
            {
                "metric": metric,
                "model": round(m_val, 4),
                "actual": round(a_val, 4) if pd.notna(a_val) else a_val,
                "diff": round(diff, 4) if pd.notna(a_val) else float("nan"),
                "pct_diff": round(pct, 2) if pd.notna(pct) else pct,
                "gate_rel_tol": gate if gate is not None else "",
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Entry point supporting both Snakemake execution and standalone CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if "snakemake" in globals():
        smk = globals()["snakemake"]
        net_in, act_in = Path(smk.input.network), Path(smk.input.actuals)
        report_out, gates_out = Path(smk.output.report), Path(smk.output.gates)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--network", type=Path, required=True)
        parser.add_argument(
            "--actuals", type=Path, default=Path("data/custom/validation_2024_es.csv")
        )
        parser.add_argument("--output-report", type=Path, required=True)
        parser.add_argument("--output-gates", type=Path, required=True)
        args = parser.parse_args(argv)
        net_in, act_in = args.network, args.actuals
        report_out, gates_out = args.output_report, args.output_gates

    n = pypsa.Network(str(net_in))
    report = validate(n, pd.read_csv(act_in))

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_out, index=False)

    gated = report[report["status"].isin(["PASS", "FAIL"])]
    summary = {
        "network": str(net_in),
        "gates_passed": int((gated["status"] == "PASS").sum()),
        "gates_total": int(len(gated)),
        "failed": {
            r[
                "metric"
            ]: f"model {r['model']} vs actual {r['actual']} ({r['pct_diff']:+.1f}%)"
            for _, r in gated[gated["status"] == "FAIL"].iterrows()
        },
    }
    gates_out.write_text(yaml.safe_dump(summary, sort_keys=False))
    logger.info(
        "Validation: %d/%d gates pass -> %s",
        summary["gates_passed"],
        summary["gates_total"],
        report_out,
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
