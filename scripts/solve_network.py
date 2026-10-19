# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Solve a 24/7 CFE procurement scenario on a prepared base network.

Takes the base network from ``prepare_network.py``, attaches the data center
(flat load at the Madrid/Aragon hubs, PPA candidates at the Spanish nodes,
standalone battery; see ``scripts/data_center.py``), adds the policy constraint
from ``scripts/cfe_constraints.py`` as linopy extra functionality, and solves
with HiGHS dual simplex. Grid-credit policies resolve the bilinear grid-share
term with the fixed-point iteration in ``scripts/iterate.py``, starting from the
solved ``unmatched_dc`` network's import-traced hourly clean share
(``scripts/calculate_grid_res.py``) and reusing the simplex basis across
iterations. Writes the solved network, a one-row KPI table, and the iteration
convergence log.

Policies: ``no_dc``, ``unmatched_dc``, ``annual_100_res``, ``hourly_80_ppa_only``,
``hourly_80_grid_credit``, ``strict_hourly_80_ppa_only``,
``strict_hourly_80_grid_credit``; the ``_cfe`` suffix counts nuclear as clean.
"""

from __future__ import annotations

import argparse
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypsa

from scripts.calculate_grid_res import calculate_grid_share
from scripts.cfe_constraints import add_dc_cfe_constraints, dc_kpis
from scripts.data_center import DEFAULT_DISCOUNT_RATE, attach_data_center, load_costs
from scripts.iterate import (
    DEFAULT_DAMPING,
    DEFAULT_MAX_ITER,
    DEFAULT_TOL,
    solve_with_grid_iteration,
)

logger = logging.getLogger(__name__)

DEFAULT_EXCESS_SHARE: float = 0.20
DEFAULT_TARGET: float = 0.80
DEFAULT_LOAD_SHEDDING_COST: float = 10000.0
DEFAULT_SOLVER_OPTIONS: dict[str, Any] = {"solver": "simplex", "threads": 2}
NO_TARGET_POLICIES: frozenset[str] = frozenset({"no_dc", "unmatched_dc"})


def add_battery_constraints(n: pypsa.Network) -> None:
    """Enforce ``charger_p_nom == efficiency * discharger_p_nom`` for extendable DC batteries."""
    if n.links.empty or not n.links["p_nom_extendable"].any():
        return
    is_charger = n.links.index.astype(str).str.contains("battery charger")
    chargers = n.links.index[is_charger & n.links["p_nom_extendable"]]
    if chargers.empty:
        return

    link_p_nom = n.model["Link-p_nom"]
    for i, charger in enumerate(chargers):
        discharger = str(charger).replace("charger", "discharger")
        eff = float(n.links.at[discharger, "efficiency"])
        n.model.add_constraints(
            link_p_nom.loc[charger] - eff * link_p_nom.loc[discharger] == 0.0,
            name=f"DC-battery-symmetry-{i}",
        )


def ensure_load_shedding(
    n: pypsa.Network,
    cost: float = DEFAULT_LOAD_SHEDDING_COST,
) -> None:
    """Add high-cost diagnostic load-shedding generators (carrier 'load') on every AC bus."""
    if "load" not in n.carriers.index:
        n.add("Carrier", "load", co2_emissions=0.0)
    ac_buses = n.buses.index[n.buses["carrier"] == "AC"]
    covered = set(n.generators.loc[n.generators["carrier"] == "load", "bus"])
    missing = pd.Index([b for b in ac_buses if b not in covered])
    if missing.empty:
        return
    n.add(
        "Generator",
        missing,
        suffix=" load shedding",
        bus=missing,
        carrier="load",
        marginal_cost=float(cost),
        p_nom=1.0e6,
        p_nom_extendable=False,
    )
    logger.info(
        "Added load shedding on %d AC buses (%.0f EUR/MWh).", len(missing), cost
    )


def resolve_target(policy: str, target: float | str | None = None) -> float:
    """Resolve the matching target ``tau`` in [0, 1] from the policy name or ``target``."""
    policy_norm = str(policy).strip().lower()
    if policy_norm in NO_TARGET_POLICIES:
        return 0.0
    if target is not None and float(target) > 0.0:
        val = float(target)
        return val / 100.0 if val > 1.0 else val
    if policy_norm.startswith("annual"):
        return 1.0
    m = re.search(r"(\d+(?:\.\d+)?)", policy_norm)
    if m:
        val = float(m.group(1))
        return val / 100.0 if val > 1.0 else val
    return DEFAULT_TARGET


def resolve_r0(
    n: pypsa.Network,
    r0_source: str | Path | pypsa.Network | pd.Series | float | None,
    clean_carriers: str | list[str] = "res",
) -> pd.Series | None:
    """Resolve the initial grid clean share ``r^(0)``, e.g. from a solved unmatched_dc network."""
    if r0_source is None:
        return None
    if isinstance(r0_source, (int, float, np.floating)):
        return pd.Series(float(r0_source), index=n.snapshots, dtype=float)
    if isinstance(r0_source, pd.Series):
        return r0_source.reindex(n.snapshots).fillna(0.0).astype(float)
    if isinstance(r0_source, (str, Path)):
        logger.info("Computing r0 from solved network %s", r0_source)
        r0_source = pypsa.Network(str(r0_source))
    r_s = calculate_grid_share(r0_source, country="ES", clean_carriers=clean_carriers)
    return r_s.reindex(n.snapshots).fillna(0.0).astype(float)


def _dc_portfolio_metrics(n: pypsa.Network, total_load_mwh: float) -> dict[str, float]:
    """DC portfolio capacities and cost, system/ES CO2, load shedding, ES mean price."""
    m: dict[str, float] = {}
    w = n.snapshot_weightings["generators"]

    dc_gens = n.generators.index[
        n.generators.index.astype(str).str.startswith("DC")
        & (n.generators["carrier"] != "load")
    ]
    cap_cost = 0.0
    marg_cost = 0.0
    for carrier, grp in n.generators.loc[dc_gens].groupby("carrier"):
        m[f"dc_ppa_{carrier}_mw"] = float(grp["p_nom_opt"].sum())
    m["dc_ppa_total_mw"] = float(n.generators.loc[dc_gens, "p_nom_opt"].sum())
    for g in dc_gens:
        cap_cost += float(
            n.generators.at[g, "p_nom_opt"] * n.generators.at[g, "capital_cost"]
        )
        if g in n.generators_t.p.columns:
            marg_cost += float(
                (n.generators_t.p[g] * w).sum() * n.generators.at[g, "marginal_cost"]
            )

    dc_stores = n.stores.index[n.stores.index.astype(str).str.startswith("DC")]
    m["dc_battery_mwh"] = float(n.stores.loc[dc_stores, "e_nom_opt"].sum())
    cap_cost += float(
        (
            n.stores.loc[dc_stores, "e_nom_opt"]
            * n.stores.loc[dc_stores, "capital_cost"]
        ).sum()
    )
    dc_links = n.links.index[n.links.index.astype(str).str.startswith("DC")]
    chargers = [lk for lk in dc_links if "charger" in lk and "discharger" not in lk]
    m["dc_battery_mw"] = float(n.links.loc[chargers, "p_nom_opt"].sum())
    cap_cost += float(
        (
            n.links.loc[dc_links, "p_nom_opt"] * n.links.loc[dc_links, "capital_cost"]
        ).sum()
    )

    m["dc_capital_cost_eur"] = cap_cost
    m["dc_marginal_cost_eur"] = marg_cost
    m["dc_cost_eur_per_mwh"] = (
        (cap_cost + marg_cost) / total_load_mwh if total_load_mwh > 0.0 else 0.0
    )

    # CO2 (thermal input per generator x carrier intensity), total and Spain
    co2 = n.carriers["co2_emissions"]
    emitters = n.generators.index[
        n.generators["carrier"].map(co2).fillna(0.0) > 0.0
    ].intersection(n.generators_t.p.columns)
    gen_mwh = n.generators_t.p[emitters].mul(w, axis=0).sum()
    eff = n.generators.loc[emitters, "efficiency"].replace(0.0, 1.0)
    em_t = gen_mwh / eff * n.generators.loc[emitters, "carrier"].map(co2)
    m["emissions"] = float(em_t.sum()) / 1.0e6
    es_mask = n.generators.loc[emitters, "bus"].str.startswith("ES")
    m["es_total_co2_mt"] = float(em_t[es_mask].sum()) / 1.0e6

    shed = n.generators.index[n.generators["carrier"] == "load"].intersection(
        n.generators_t.p.columns
    )
    shed_mwh = n.generators_t.p[shed].mul(w, axis=0).sum()
    m["load_shedding_mwh"] = float(shed_mwh.sum())
    es_shed = [g for g in shed if str(n.generators.at[g, "bus"]).startswith("ES")]
    m["es_load_shedding_mwh"] = float(shed_mwh.loc[es_shed].sum()) if es_shed else 0.0

    # Load-weighted Spanish electricity price
    es_buses = n.buses.index[
        (n.buses["country"] == "ES") & (n.buses["carrier"] == "AC")
    ]
    load_cols = n.loads.index[
        n.loads["bus"].isin(es_buses) & (n.loads["carrier"] != "boundary")
    ].intersection(n.loads_t.p_set.columns)
    if not load_cols.empty and not n.buses_t.marginal_price.empty:
        load_per_bus = (
            n.loads_t.p_set[load_cols].T.groupby(n.loads.loc[load_cols, "bus"]).sum().T
        )
        prices = n.buses_t.marginal_price[load_per_bus.columns]
        m["es_mean_price_eur_mwh"] = float(
            (prices * load_per_bus).sum().sum() / load_per_bus.sum().sum()
        )
    return m


def solve_network(
    n: pypsa.Network,
    policy: str,
    target: float | None = None,
    excess_share: float = DEFAULT_EXCESS_SHARE,
    r0_source: str | Path | pd.Series | float | None = None,
    clean_carriers: str = "res",
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    damping: float = DEFAULT_DAMPING,
    solver_options: dict[str, Any] | None = None,
    load_shedding_cost: float = DEFAULT_LOAD_SHEDDING_COST,
) -> tuple[pypsa.Network, pd.DataFrame, dict[str, Any]]:
    """
    Solve one policy scenario and return ``(network, convergence_df, kpi_row)``.

    The network must already carry the DC components (``attach_data_center``).
    """
    ensure_load_shedding(n, cost=load_shedding_cost)

    policy_norm = str(policy).strip().lower()
    tau = resolve_target(policy_norm, target)
    sigma = 0.0 if policy_norm in NO_TARGET_POLICIES else float(excess_share)
    constraint_policy = (
        "annual_res" if policy_norm.startswith("annual") else policy_norm
    )
    dc_cfg = {"prefix": "DC"}

    opts = dict(DEFAULT_SOLVER_OPTIONS)
    opts.update(solver_options or {})
    r0_series = resolve_r0(n, r0_source, clean_carriers=clean_carriers)

    t_start = time.perf_counter()
    if "grid_credit" in policy_norm:

        def build_fn(net: pypsa.Network, r_iter: pd.Series) -> None:
            add_battery_constraints(net)
            add_dc_cfe_constraints(
                n=net,
                dc_cfg=dc_cfg,
                r=r_iter,
                policy=constraint_policy,
                target=tau,
                excess_share=sigma,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            bas = Path(tmpdir) / "highs_iter.bas"
            iter_opts = {
                "io_api": "direct",
                "basis_fn": bas,
                "warmstart_fn": bas,
                **opts,
            }
            n, convergence_df = solve_with_grid_iteration(
                n=n,
                build_fn=build_fn,
                r0=r0_series,
                max_iter=max_iter,
                tol=tol,
                damping=damping,
                solver_options=iter_opts,
                country="ES",
                clean_carriers=clean_carriers,
            )
    else:
        r_single = r0_series if r0_series is not None else 0.0

        def extra_functionality(net: pypsa.Network, snapshots: pd.Index) -> None:
            add_battery_constraints(net)
            add_dc_cfe_constraints(
                n=net,
                dc_cfg=dc_cfg,
                r=r_single,
                policy=constraint_policy,
                target=tau,
                excess_share=sigma,
            )

        status, condition = n.optimize(
            solver_name="highs",
            solver_options=opts,
            extra_functionality=extra_functionality,
            io_api="direct",
        )
        if status != "ok":
            raise RuntimeError(
                f"Optimization failed for policy='{policy}': {status} / {condition}"
            )
        r_out = calculate_grid_share(n, country="ES", clean_carriers=clean_carriers)
        n._grid_share = r_out
        convergence_df = pd.DataFrame(
            [
                {
                    "iteration": 0,
                    "max_delta": 0.0,
                    "objective": float(n.objective),
                    "mean_r_in": float(np.mean(r_single)),
                    "mean_r_out": float(r_out.mean()),
                    "converged": True,
                }
            ]
        )
    solve_time_s = time.perf_counter() - t_start

    raw_kpis = dc_kpis(n)
    total_load_mwh = float(raw_kpis["total_load_mwh"])
    kpi_row: dict[str, Any] = {
        "policy": str(policy),
        "target": tau,
        "excess_cap": sigma,
        "objective": float(n.objective),
        "solve_time_s": solve_time_s,
        "iterations": len(convergence_df),
        "converged": bool(convergence_df["converged"].iloc[-1]),
        "max_delta": float(convergence_df["max_delta"].iloc[-1]),
        "mean_grid_share_es": float(n._grid_share.mean()),
    }
    for k, v in raw_kpis.items():
        if not isinstance(v, pd.Series):
            kpi_row[k] = float(v)
    kpi_row.update(_dc_portfolio_metrics(n, total_load_mwh))
    return n, convergence_df, kpi_row


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network", type=Path, required=True, help="Base network (.nc)."
    )
    parser.add_argument(
        "--costs", type=Path, required=True, help="Processed costs CSV."
    )
    parser.add_argument("--output-network", type=Path, required=True)
    parser.add_argument("--output-kpis", type=Path, required=True)
    parser.add_argument("--output-convergence", type=Path, required=True)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--dc-mw", type=float, default=1000.0)
    parser.add_argument("--siting", type=str, default="spain_wide")
    parser.add_argument("--r0-network", type=Path, default=None)
    parser.add_argument("--excess-share", type=float, default=DEFAULT_EXCESS_SHARE)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument(
        "--meta", type=str, default="", help="key=value,... appended to the KPI row."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point supporting both Snakemake execution and standalone CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if "snakemake" in globals():
        smk = globals()["snakemake"]
        cfg = smk.config.get("spain_cfe", {})
        w = smk.wildcards
        net_in, costs_in = Path(smk.input.network), Path(smk.input.costs)
        r0_in = Path(smk.input.r0_network[0]) if smk.input.r0_network else None
        net_out = Path(smk.output.network)
        kpi_out, conv_out = Path(smk.output.kpis), Path(smk.output.convergence)
        policy, siting = str(w.policy), str(w.siting)
        target, dc_mw = float(w.target), float(w.dc_mw)
        meta = {
            "planning_year": int(w.year),
            "weather_year": int(str(w.run).removeprefix("wy")),
            "variant": str(w.variant),
            "dc_mw": dc_mw,
            "siting": siting,
        }
    else:
        args = _parse_args(argv)
        net_in, costs_in = args.network, args.costs
        r0_in = args.r0_network
        net_out = args.output_network
        kpi_out, conv_out = args.output_kpis, args.output_convergence
        policy, siting = args.policy, args.siting
        target, dc_mw = args.target, args.dc_mw
        meta = dict(kv.split("=") for kv in args.meta.split(",") if "=" in kv)
        meta["dc_mw"] = dc_mw
        meta["siting"] = siting
        cfg = {
            "accounting": {"excess_cap": args.excess_share},
            "iteration": {
                "max_iter": args.max_iter,
                "tol": args.tol,
                "damping": args.damping,
            },
        }

    iter_cfg = cfg.get("iteration", {})
    dc_cfg = cfg.get("dc", {})

    n = pypsa.Network(str(net_in))
    meta["resolution"] = f"{int(n.snapshot_weightings['generators'].iloc[0])}h"
    attach_data_center(
        n,
        dc_mw=dc_mw,
        hubs=dc_cfg.get("hubs"),
        siting=siting,
        ppa_carriers=cfg.get("ppa", {}).get("carriers"),
        costs=load_costs(costs_in),
        discount_rate=float(dc_cfg.get("discount_rate", DEFAULT_DISCOUNT_RATE)),
        policy=policy,
    )
    n, conv_df, kpi_row = solve_network(
        n,
        policy=policy,
        target=target,
        excess_share=float(
            cfg.get("accounting", {}).get("excess_cap", DEFAULT_EXCESS_SHARE)
        ),
        r0_source=r0_in,
        clean_carriers="cfe" if policy.endswith("_cfe") else "res",
        max_iter=int(iter_cfg.get("max_iter", DEFAULT_MAX_ITER)),
        tol=float(iter_cfg.get("tol", DEFAULT_TOL)),
        damping=float(iter_cfg.get("damping", DEFAULT_DAMPING)),
    )
    kpi_row = {**meta, **kpi_row}

    for path in (net_out, kpi_out, conv_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(str(net_out))
    pd.DataFrame([kpi_row]).to_csv(kpi_out, index=False)
    conv_df.to_csv(conv_out, index=False)
    logger.info(
        "Solved %s: objective %.4e, KPIs -> %s", policy, float(n.objective), kpi_out
    )


if __name__ == "__main__":
    main()
