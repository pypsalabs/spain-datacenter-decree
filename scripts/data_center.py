# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Prepare a PyPSA-Eur network for 24/7 CFE data center procurement.

Adds flat data center loads at the configured Spanish site buses (additional to
existing background electricity demand), extendable PPA renewable generator
candidates at the best resource class across Spain (or in the local region only), and standalone DC battery
storage (``Bus`` + ``Store`` + charger/discharger ``Link``s) at each site.

Processed technology costs from ``PyPSA/technology-data v0.15.0`` are stored by
PyPSA-Eur v2026.09.0 in ``pypsa-eur/resources/{run}/costs_{planning_year}_processed.csv``
(loaded via ``load_costs``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# Default DC site: Aragon (the hyperscale pipeline); overridden via config spain_cfe.dc
DEFAULT_SITE_BUSES: tuple[str, ...] = ("ES0 7AC",)
DEFAULT_PPA_CARRIERS: tuple[str, ...] = ("onwind", "solar", "solar-hsat")
DEFAULT_DISCOUNT_RATE: float = 0.07

# Explicit mapping from PyPSA carrier to v0.15.0 technology-data row for investment/FOM/lifetime
COST_TECH_MAP: dict[str, tuple[str, ...]] = {
    "onwind": ("onwind",),
    "solar": ("solar-utility", "solar"),
    "solar-hsat": ("solar-hsat", "solar-utility single-axis tracking"),
    "battery storage": ("battery storage",),
    "battery inverter": ("battery inverter",),
}


def calculate_annuity(lifetime: float, discount_rate: float) -> float:
    """Compute the capital recovery factor (annuity) for ``lifetime`` and ``discount_rate``."""
    n_yr = float(lifetime)
    r = float(discount_rate)
    if n_yr <= 0.0:
        raise ValueError(f"Asset lifetime must be positive; got {lifetime}.")
    if abs(r) < 1e-12:
        return 1.0 / n_yr
    return r / (1.0 - (1.0 + r) ** (-n_yr))


def load_costs(cost_file: str | Path) -> pd.DataFrame:
    """
    Load processed PyPSA-Eur technology costs CSV (index = ``technology``).

    In PyPSA-Eur v2026.09.0, processed costs are written by ``process_cost_data.py``
    to ``pypsa-eur/resources/{run}/costs_{planning_year}_processed.csv``.
    """
    return pd.read_csv(cost_file, index_col=0)


def _resolve_cost_row(costs: pd.DataFrame, carrier: str) -> str:
    """Return the matching technology index name in ``costs`` for ``carrier``."""
    candidates = COST_TECH_MAP.get(carrier, (carrier,))
    for cand in candidates:
        if cand in costs.index:
            return cand
    if carrier in costs.index:
        return carrier
    raise KeyError(
        f"Technology '{carrier}' (candidates {candidates}) not found in costs index."
    )


def _compute_tech_costs(
    costs: pd.DataFrame,
    carrier: str,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> dict[str, float]:
    """Compute annualized ``capital_cost``, ``marginal_cost``, ``lifetime``, and ``efficiency``."""
    tech_row = _resolve_cost_row(costs, carrier)
    row = costs.loc[tech_row]

    lifetime = (
        float(row["lifetime"])
        if "lifetime" in row and pd.notna(row["lifetime"])
        else 25.0
    )
    investment = (
        float(row["investment"])
        if "investment" in row and pd.notna(row["investment"])
        else np.nan
    )
    fom_pct = float(row["FOM"]) if "FOM" in row and pd.notna(row["FOM"]) else 0.0

    if np.isfinite(investment):
        ann = calculate_annuity(lifetime, discount_rate)
        # In technology-data v0.15.0, FOM is given in %/year of overnight investment
        capital_cost = (ann + fom_pct / 100.0) * investment
    elif "capital_cost" in row and pd.notna(row["capital_cost"]):
        capital_cost = float(row["capital_cost"])
    else:
        raise ValueError(
            f"Cannot determine capital_cost for '{carrier}' from costs row '{tech_row}'."
        )

    # Marginal cost: check direct carrier row first (e.g. 'solar' has 0.010 VOM overwrite), then tech_row
    if (
        carrier in costs.index
        and "marginal_cost" in costs.columns
        and pd.notna(costs.at[carrier, "marginal_cost"])
    ):
        marginal_cost = float(costs.at[carrier, "marginal_cost"])
    elif "marginal_cost" in row and pd.notna(row["marginal_cost"]):
        marginal_cost = float(row["marginal_cost"])
    elif "VOM" in row and pd.notna(row["VOM"]):
        marginal_cost = float(row["VOM"])
    else:
        marginal_cost = 0.0

    efficiency = (
        float(row["efficiency"])
        if "efficiency" in row and pd.notna(row["efficiency"])
        else 1.0
    )

    return {
        "capital_cost": capital_cost,
        "marginal_cost": marginal_cost,
        "lifetime": lifetime,
        "efficiency": efficiency,
    }


def _best_class_gens(
    n: pypsa.Network, bus: str, carrier: str
) -> tuple[str, float] | None:
    """
    Pick the PPA reference generator at ``bus`` for ``carrier``: the resource class
    (CF bin) with the highest mean availability among those with remaining potential.

    Returns ``(reference_generator, remaining_potential_mw)`` or ``None`` if the
    bus has no background generator of that carrier.
    """
    bg_mask = (
        (n.generators["bus"] == bus)
        & (n.generators["carrier"] == carrier)
        & (~n.generators.index.astype(str).str.startswith("DC"))
    )
    bg_gens = n.generators.index[bg_mask]
    if bg_gens.empty:
        return None

    remaining = (
        n.generators.loc[bg_gens, "p_nom_max"] - n.generators.loc[bg_gens, "p_nom"]
    ).where(lambda s: s > 0.0, 0.0)
    t_cols = bg_gens.intersection(n.generators_t.p_max_pu.columns)
    mean_cf = pd.Series(
        n.generators.loc[bg_gens, "p_max_pu"].astype(float), index=bg_gens
    )
    if not t_cols.empty:
        mean_cf.loc[t_cols] = n.generators_t.p_max_pu[t_cols].mean()

    with_room = remaining[remaining > 0.0]
    pool = with_room.index if not with_room.empty else bg_gens
    best = mean_cf.loc[pool].idxmax()
    return str(best), float(remaining.at[best])


def _ensure_carriers(n: pypsa.Network, carriers: Sequence[str]) -> None:
    """Add any missing carrier names to ``n.carriers``."""
    missing = [c for c in carriers if c not in n.carriers.index]
    if missing:
        n.add("Carrier", missing)


def _ensure_base_carriers(n: pypsa.Network) -> None:
    """Ensure study carriers (CHP, csp, boundary, load) exist."""
    for car, co2 in (("CHP", 0.369), ("csp", 0.0), ("boundary", 0.0), ("load", 0.0)):
        if car not in n.carriers.index:
            n.add("Carrier", car, co2_emissions=co2)


def attach_data_center(
    n: pypsa.Network,
    dc_mw: float,
    site_buses: Sequence[str] | None = None,
    siting: str = "spain_wide",
    ppa_carriers: Sequence[str] | None = None,
    costs: pd.DataFrame | str | Path | None = None,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    policy: str = "hourly_80_grid_credit",
) -> pypsa.Network:
    """
    Prepare ``n`` in-place with DC loads, extendable PPA candidates, and DC batteries.

    Parameters
    ----------
    n : pypsa.Network
        Prepared base network (15 buses: ES 9 / FR 4 / PT 2).
    dc_mw : float
        Flat baseload demand in MW hosted at **each** site bus.
    site_buses : Sequence[str] | None, optional
        Network buses hosting a data center, each with the full ``dc_mw``.
        Defaults to ``('ES0 7AC',)`` (Aragon); the site labels and their node
        mapping live in the config under ``spain_cfe.dc``.
    siting : str, default 'spain_wide'
        PPA candidate scope: ``'spain_wide'`` (the best resource class at every
        ES AC bus; the optimizer chooses where to contract) or ``'local_region'``
        (the best class in the DC's own region only).
    ppa_carriers : Sequence[str] | None, optional
        Extendable PPA generator carriers. Defaults to
        ``['onwind', 'solar', 'solar-hsat']``.
    costs : pd.DataFrame | str | Path | None
        Processed technology costs DataFrame (or path to CSV) from
        ``pypsa-eur/resources/{run}/costs_{year}_processed.csv``.
    discount_rate : float, default 0.07
        WACC / discount rate used for DC PPA and battery annuities.
    policy : str, default 'hourly_80_grid_credit'
        Procurement policy name. ``'no_dc'`` adds no DC assets at all;
        ``'unmatched_dc'`` adds DC loads only (no PPA candidates, no DC battery).

    Returns
    -------
    pypsa.Network
        The modified network ``n``.
    """
    _ensure_base_carriers(n)

    policy_norm = str(policy).strip().lower()
    if policy_norm == "no_dc":
        logger.info(
            "Policy is 'no_dc': adding no DC loads, PPA generators, or batteries."
        )
        return n

    siting_norm = str(siting).strip().lower()
    if siting_norm not in {"spain_wide", "local_region"}:
        raise ValueError(
            f"Unsupported siting '{siting}'. Expected 'spain_wide' or 'local_region'."
        )

    sites = list(site_buses) if site_buses is not None else list(DEFAULT_SITE_BUSES)
    for site_bus in sites:
        if site_bus not in n.buses.index:
            raise KeyError(f"DC site bus '{site_bus}' not found in network buses.")

    # 1. Add a flat DC load of the full dc_mw at each site bus
    _ensure_carriers(n, ["DC load"])
    for site_bus in sites:
        load_name = f"DC {site_bus} load"
        if load_name in n.loads.index:
            n.remove("Load", load_name)
        n.add(
            "Load",
            load_name,
            bus=site_bus,
            carrier="DC load",
            p_set=float(dc_mw),
        )
        logger.info(
            "Added flat DC load '%s' at bus '%s': %.2f MW",
            load_name,
            site_bus,
            float(dc_mw),
        )

    if policy_norm == "unmatched_dc":
        logger.info(
            "Policy is 'unmatched_dc': added DC loads only (no PPA candidates, no DC battery)."
        )
        return n

    if costs is None:
        raise ValueError(
            f"Processed technology costs must be provided for policy='{policy}'."
        )
    costs_df = load_costs(costs) if isinstance(costs, (str, Path)) else costs

    carriers_list = (
        list(ppa_carriers) if ppa_carriers is not None else list(DEFAULT_PPA_CARRIERS)
    )
    _ensure_carriers(
        n,
        [
            *carriers_list,
            "DC battery",
            "DC battery charger",
            "DC battery discharger",
        ],
    )

    # 2. Determine candidate buses for PPA generators
    if siting_norm == "spain_wide":
        es_mask = (n.buses["country"].astype(str) == "ES") & (
            n.buses["carrier"].astype(str) == "AC"
        )
        not_dc_bus = ~n.buses.index.astype(str).str.startswith("DC")
        candidate_buses = list(n.buses.index[es_mask & not_dc_bus])
    else:
        candidate_buses = list(sites)

    # 3. Add extendable PPA candidates 'DC {bus} {carrier}': the best resource class
    # (highest-CF bin with remaining potential) at each candidate bus
    added_ppas = 0
    for bus in candidate_buses:
        for car in carriers_list:
            picked = _best_class_gens(n, bus, car)
            if picked is None:
                logger.warning(
                    "No background generator found at bus '%s' for carrier '%s'; skipping PPA candidate.",
                    bus,
                    car,
                )
                continue
            ref_gen, rem_potential = picked

            if (
                not n.generators_t.p_max_pu.empty
                and ref_gen in n.generators_t.p_max_pu.columns
            ):
                p_max_pu = n.generators_t.p_max_pu[ref_gen].copy()
            else:
                p_max_pu = float(n.generators.at[ref_gen, "p_max_pu"])

            c_info = _compute_tech_costs(costs_df, car, discount_rate=discount_rate)
            ppa_name = f"DC {bus} {car}"
            if ppa_name in n.generators.index:
                n.remove("Generator", ppa_name)

            n.add(
                "Generator",
                ppa_name,
                bus=bus,
                carrier=car,
                p_nom=0.0,
                p_nom_extendable=True,
                p_nom_max=rem_potential,
                p_max_pu=p_max_pu,
                capital_cost=c_info["capital_cost"],
                marginal_cost=c_info["marginal_cost"],
                efficiency=c_info["efficiency"],
                lifetime=c_info["lifetime"],
            )
            added_ppas += 1

    logger.info(
        "Added %d extendable DC PPA generator candidates across %d buses (siting='%s').",
        added_ppas,
        len(candidate_buses),
        siting_norm,
    )

    # 4. Add a standalone DC battery at each site (Bus + Store + charger/discharger Links)
    store_cost = _compute_tech_costs(
        costs_df, "battery storage", discount_rate=discount_rate
    )
    inv_cost = _compute_tech_costs(
        costs_df, "battery inverter", discount_rate=discount_rate
    )
    one_way_eff = float(np.sqrt(inv_cost["efficiency"]))

    for hub_bus in sites:
        bat_bus_name = f"DC {hub_bus} battery"
        bat_store_name = f"DC {hub_bus} battery"
        charger_name = f"DC {hub_bus} battery charger"
        discharger_name = f"DC {hub_bus} battery discharger"

        for comp, df_idx, name in (
            ("Link", n.links.index, charger_name),
            ("Link", n.links.index, discharger_name),
            ("Store", n.stores.index, bat_store_name),
            ("Bus", n.buses.index, bat_bus_name),
        ):
            if name in df_idx:
                n.remove(comp, name)

        hub_x = float(n.buses.at[hub_bus, "x"]) if "x" in n.buses.columns else 0.0
        hub_y = float(n.buses.at[hub_bus, "y"]) if "y" in n.buses.columns else 0.0
        hub_ct = (
            str(n.buses.at[hub_bus, "country"])
            if "country" in n.buses.columns
            else "ES"
        )

        n.add(
            "Bus",
            bat_bus_name,
            carrier="DC battery",
            location=hub_bus,
            country=hub_ct,
            x=hub_x,
            y=hub_y,
        )
        n.add(
            "Store",
            bat_store_name,
            bus=bat_bus_name,
            carrier="DC battery",
            e_nom=0.0,
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=store_cost["capital_cost"],
            marginal_cost=store_cost["marginal_cost"],
            lifetime=store_cost["lifetime"],
        )
        n.add(
            "Link",
            charger_name,
            bus0=hub_bus,
            bus1=bat_bus_name,
            carrier="DC battery charger",
            efficiency=one_way_eff,
            capital_cost=inv_cost["capital_cost"],
            marginal_cost=inv_cost["marginal_cost"],
            p_nom=0.0,
            p_nom_extendable=True,
            lifetime=inv_cost["lifetime"],
        )
        n.add(
            "Link",
            discharger_name,
            bus0=bat_bus_name,
            bus1=hub_bus,
            carrier="DC battery discharger",
            efficiency=one_way_eff,
            capital_cost=0.0,
            marginal_cost=inv_cost["marginal_cost"],
            p_nom=0.0,
            p_nom_extendable=True,
            lifetime=inv_cost["lifetime"],
        )
        logger.info(
            "Added DC battery at site '%s': Store cap_cost=%.2f EUR/MWh/a, "
            "Inverter cap_cost=%.2f EUR/MW/a, one_way_eff=%.4f",
            hub_bus,
            store_cost["capital_cost"],
            inv_cost["capital_cost"],
            one_way_eff,
        )

    return n
