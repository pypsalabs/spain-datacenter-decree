# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT
"""
Import-traced hourly grid renewable / CFE share calculation.

Adapts ``PyPSA/247-cfe``'s ``calculate_grid_cfe`` for a multi-node network with
cross-border lines and HVDC links. Storage discharge (PHS, battery), the data
center's own PPA assets and fixed ``boundary`` flow components (France's measured
external border flows, MWh only) are excluded from both numerator and denominator:
a neighbour's clean share is its domestic generation mix.
"""

import logging
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

DEFAULT_RES_CARRIERS: tuple[str, ...] = (
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "solar",
    "solar-hsat",
    "ror",
    "hydro",
    "biomass",
    "csp",
)
DEFAULT_CFE_EXTRA_CARRIERS: tuple[str, ...] = ("nuclear",)
DEFAULT_CFE_CARRIERS: tuple[str, ...] = (
    DEFAULT_RES_CARRIERS + DEFAULT_CFE_EXTRA_CARRIERS
)
DEFAULT_EXCLUDED_CARRIERS: tuple[str, ...] = (
    "PHS",
    "battery",
    "battery charger",
    "battery discharger",
    "battery storage",
)
BOUNDARY_CARRIER: str = "boundary"
NON_GENERATION_CARRIERS: frozenset[str] = frozenset(
    {"AC", "DC", BOUNDARY_CARRIER, "load", "virtual_link", "dsm", ""}
)


def get_clean_carriers(
    accounting_cfg: Mapping[str, Any] | None = None,
    mode: str = "res",
) -> list[str]:
    """Return the configured clean carrier list for ``'res'`` or ``'cfe'`` mode."""
    cfg = accounting_cfg or {}
    res_list = list(cfg.get("res_carriers", DEFAULT_RES_CARRIERS))
    mode_norm = mode.strip().lower()
    if mode_norm == "res":
        return res_list
    if mode_norm == "cfe":
        extra = list(cfg.get("cfe_extra_carriers", DEFAULT_CFE_EXTRA_CARRIERS))
        return res_list + [c for c in extra if c not in res_list]
    raise ValueError(f"Unsupported accounting mode '{mode}'; expected 'res' or 'cfe'.")


def _normalize_clean_carriers(
    clean_carriers: Iterable[str] | str | None,
    accounting_cfg: Mapping[str, Any] | None = None,
) -> set[str]:
    """Resolve ``clean_carriers`` from an iterable of names or a mode string."""
    if clean_carriers is None:
        return set(get_clean_carriers(accounting_cfg, mode="res"))
    if isinstance(clean_carriers, str):
        return set(get_clean_carriers(accounting_cfg, mode=clean_carriers))
    return set(clean_carriers)


def _is_excluded_index(
    index: pd.Index,
    exclude_assets: Collection[str],
) -> pd.Series:
    """Return boolean mask over ``index`` for assets in ``exclude_assets``."""
    if not exclude_assets:
        return pd.Series(False, index=index)
    exact_match = index.isin(exclude_assets)
    prefixes = tuple(str(a) for a in exclude_assets if a)
    prefix_match = index.astype(str).str.startswith(prefixes)
    return pd.Series(exact_match | prefix_match, index=index)


def _country_buses(
    n: pypsa.Network,
    country: str,
    exclude_assets: Collection[str],
) -> pd.Index:
    """Return bus names belonging to ``country`` excluding DC / excluded buses."""
    if n.buses.empty:
        return pd.Index([])
    idx_str = n.buses.index.astype(str)
    prefix_mask = idx_str.str.startswith(country)
    if "country" in n.buses.columns:
        col_mask = n.buses["country"].astype(str) == country
        bus_mask = col_mask | prefix_mask
    else:
        bus_mask = prefix_mask
    excl_mask = _is_excluded_index(n.buses.index, exclude_assets)
    return n.buses.index[bus_mask & ~excl_mask]


def _positive_part(series: pd.Series) -> pd.Series:
    """Return ``max(series, 0.0)`` without calling ``.clip``."""
    return series.where(series > 0.0, 0.0)


def _domestic_supply(
    n: pypsa.Network,
    country: str,
    clean_set: set[str],
    excluded_set: set[str],
    exclude_assets_set: set[str],
) -> tuple[pd.Series, pd.Series]:
    """Compute (clean_supply, total_supply) for ``country`` from domestic generation."""
    snapshots = n.snapshots
    clean_supply = pd.Series(0.0, index=snapshots, dtype=float)
    total_supply = pd.Series(0.0, index=snapshots, dtype=float)

    c_buses = _country_buses(n, country, exclude_assets_set)
    if c_buses.empty:
        return clean_supply, total_supply

    all_ac_buses = n.buses.index[
        ~_is_excluded_index(n.buses.index, exclude_assets_set)
        & (
            (n.buses["carrier"] == "AC")
            if "carrier" in n.buses.columns
            else pd.Series(True, index=n.buses.index)
        )
    ]

    # 1. Generators (primary domestic + optional boundary imports)
    if not n.generators.empty and not n.generators_t.p.empty:
        gen_df = n.generators
        in_country = gen_df["bus"].isin(c_buses)
        not_excl_asset = ~_is_excluded_index(gen_df.index, exclude_assets_set)
        valid_gens = gen_df.index[in_country & not_excl_asset].intersection(
            n.generators_t.p.columns
        )

        for g in valid_gens:
            carrier = str(gen_df.at[g, "carrier"])
            if carrier in excluded_set or carrier in {
                "load",
                "virtual_link",
                "dsm",
                BOUNDARY_CARRIER,
            }:
                continue
            p_series = _positive_part(n.generators_t.p[g])
            total_supply = total_supply + p_series
            if carrier in clean_set:
                clean_supply = clean_supply + p_series

    # 2. StorageUnits (natural-inflow hydro/ror/csp counted; PHS and battery excluded)
    if not n.storage_units.empty:
        su_df = n.storage_units
        in_country = su_df["bus"].isin(c_buses)
        not_excl_asset = ~_is_excluded_index(su_df.index, exclude_assets_set)
        su_p_table = (
            n.storage_units_t.p
            if not n.storage_units_t.p.empty
            else n.storage_units_t.p_dispatch
        )
        if not su_p_table.empty:
            valid_sus = su_df.index[in_country & not_excl_asset].intersection(
                su_p_table.columns
            )
            for su in valid_sus:
                carrier = str(su_df.at[su, "carrier"])
                if carrier in excluded_set or carrier in NON_GENERATION_CARRIERS:
                    continue
                p_series = _positive_part(su_p_table[su])
                total_supply = total_supply + p_series
                if carrier in clean_set:
                    clean_supply = clean_supply + p_series

    # 3. Links whose bus1 is in-country for primary conversion carriers
    if not n.links.empty:
        link_df = n.links
        in_country = link_df["bus1"].isin(c_buses)
        not_excl_asset = ~_is_excluded_index(link_df.index, exclude_assets_set)
        not_grid_transfer = ~link_df["bus0"].isin(all_ac_buses)
        candidate_links = link_df.index[in_country & not_excl_asset & not_grid_transfer]

        for lk in candidate_links:
            carrier = str(link_df.at[lk, "carrier"])
            if (
                carrier in excluded_set
                or carrier in NON_GENERATION_CARRIERS
                or "charger" in carrier
                or "discharger" in carrier
                or "battery" in carrier
                or "PHS" in carrier
            ):
                continue

            if (
                not n.links_t.p1.empty
                and lk in n.links_t.p1.columns
                and (n.links_t.p1[lk] != 0.0).any()
            ):
                p_out = _positive_part(-n.links_t.p1[lk])
            elif not n.links_t.p0.empty and lk in n.links_t.p0.columns:
                eff = (
                    float(link_df.at[lk, "efficiency"])
                    if "efficiency" in link_df.columns
                    else 1.0
                )
                p_out = _positive_part(n.links_t.p0[lk] * eff)
            else:
                continue

            total_supply = total_supply + p_out
            if carrier in clean_set:
                clean_supply = clean_supply + p_out

    return clean_supply, total_supply


def _corridor_import(
    n: pypsa.Network,
    exporter: str,
    importer: str,
    exclude_assets_set: set[str],
) -> pd.Series:
    """
    Return hourly positive imports (MW) from ``exporter`` into ``importer``.

    Handles both orientations (``bus0`` in ``exporter`` / ``bus1`` in ``importer``
    and ``bus0`` in ``importer`` / ``bus1`` in ``exporter``) across ``n.lines``
    and cross-border ``n.links``.
    """
    snapshots = n.snapshots
    imp_total = pd.Series(0.0, index=snapshots, dtype=float)

    exp_buses = _country_buses(n, exporter, exclude_assets_set)
    imp_buses = _country_buses(n, importer, exclude_assets_set)
    if exp_buses.empty or imp_buses.empty:
        return imp_total

    # 1. AC Lines
    if not n.lines.empty:
        lines_df = n.lines
        not_excl = ~_is_excluded_index(lines_df.index, exclude_assets_set)

        # Subset A: bus0 in importer, bus1 in exporter (import means flow bus1 -> bus0, p1 > 0 / -p0 > 0)
        sub_a = lines_df.index[
            not_excl
            & lines_df["bus0"].isin(imp_buses)
            & lines_df["bus1"].isin(exp_buses)
        ]
        for ln in sub_a:
            if (
                not n.lines_t.p1.empty
                and ln in n.lines_t.p1.columns
                and (n.lines_t.p1[ln] != 0.0).any()
            ):
                imp_total = imp_total + _positive_part(n.lines_t.p1[ln])
            elif not n.lines_t.p0.empty and ln in n.lines_t.p0.columns:
                imp_total = imp_total + _positive_part(-n.lines_t.p0[ln])

        # Subset B: bus0 in exporter, bus1 in importer (import means flow bus0 -> bus1, p0 > 0 / -p1 > 0)
        sub_b = lines_df.index[
            not_excl
            & lines_df["bus0"].isin(exp_buses)
            & lines_df["bus1"].isin(imp_buses)
        ]
        for ln in sub_b:
            if (
                not n.lines_t.p0.empty
                and ln in n.lines_t.p0.columns
                and (n.lines_t.p0[ln] != 0.0).any()
            ):
                imp_total = imp_total + _positive_part(n.lines_t.p0[ln])
            elif not n.lines_t.p1.empty and ln in n.lines_t.p1.columns:
                imp_total = imp_total + _positive_part(-n.lines_t.p1[ln])

    # 2. Cross-border Links (DC interconnectors)
    if not n.links.empty:
        links_df = n.links
        not_excl = ~_is_excluded_index(links_df.index, exclude_assets_set)

        # Subset A: bus0 in importer, bus1 in exporter
        sub_a = links_df.index[
            not_excl
            & links_df["bus0"].isin(imp_buses)
            & links_df["bus1"].isin(exp_buses)
        ]
        for lk in sub_a:
            if (
                not n.links_t.p1.empty
                and lk in n.links_t.p1.columns
                and (n.links_t.p1[lk] != 0.0).any()
            ):
                imp_total = imp_total + _positive_part(n.links_t.p1[lk])
            elif not n.links_t.p0.empty and lk in n.links_t.p0.columns:
                imp_total = imp_total + _positive_part(-n.links_t.p0[lk])

        # Subset B: bus0 in exporter, bus1 in importer
        sub_b = links_df.index[
            not_excl
            & links_df["bus0"].isin(exp_buses)
            & links_df["bus1"].isin(imp_buses)
        ]
        for lk in sub_b:
            if (
                not n.links_t.p0.empty
                and lk in n.links_t.p0.columns
                and (n.links_t.p0[lk] != 0.0).any()
            ):
                imp_total = imp_total + _positive_part(n.links_t.p0[lk])
            elif not n.links_t.p1.empty and lk in n.links_t.p1.columns:
                imp_total = imp_total + _positive_part(-n.links_t.p1[lk])

    return imp_total


def _infer_neighbours(
    n: pypsa.Network,
    country: str,
    exclude_assets_set: set[str],
) -> list[str]:
    """Infer neighbouring country codes connected to ``country`` by lines or links."""
    c_buses = _country_buses(n, country, exclude_assets_set)
    if c_buses.empty:
        return []

    other_buses: set[str] = set()
    if not n.lines.empty:
        for _, row in n.lines.iterrows():
            if row["bus0"] in c_buses and row["bus1"] not in c_buses:
                other_buses.add(str(row["bus1"]))
            elif row["bus1"] in c_buses and row["bus0"] not in c_buses:
                other_buses.add(str(row["bus0"]))

    if not n.links.empty:
        for idx, row in n.links.iterrows():
            if _is_excluded_index(pd.Index([idx]), exclude_assets_set).iloc[0]:
                continue
            if row["bus0"] in c_buses and row["bus1"] not in c_buses:
                other_buses.add(str(row["bus1"]))
            elif row["bus1"] in c_buses and row["bus0"] not in c_buses:
                other_buses.add(str(row["bus0"]))

    neighbours: list[str] = []
    for b in sorted(other_buses):
        if b not in n.buses.index:
            continue
        if _is_excluded_index(pd.Index([b]), exclude_assets_set).iloc[0]:
            continue
        if (
            "country" in n.buses.columns
            and pd.notna(n.buses.at[b, "country"])
            and str(n.buses.at[b, "country"])
        ):
            ct = str(n.buses.at[b, "country"])
        else:
            ct = b[:2]
        if ct and ct != country and ct not in neighbours:
            neighbours.append(ct)
    return neighbours


def calculate_grid_share(
    n: pypsa.Network,
    country: str = "ES",
    clean_carriers: Iterable[str] | str | None = None,
    excluded_carriers: Iterable[str] | None = None,
    exclude_assets: Iterable[str] | None = None,
    neighbours: Iterable[str] | None = None,
    accounting_cfg: Mapping[str, Any] | None = None,
) -> pd.Series:
    """
    Compute the import-traced hourly clean share ``r_t`` of grid supply for ``country``.

    Parameters
    ----------
    n : pypsa.Network
        Solved (or dispatch-populated) PyPSA network.
    country : str, default "ES"
        Two-letter country code of the target country.
    clean_carriers : Iterable[str] | str | None, optional
        Set/list of clean primary carriers, or mode string ``"res"`` / ``"cfe"``.
        Defaults to ``DEFAULT_RES_CARRIERS``.
    excluded_carriers : Iterable[str] | None, optional
        Storage/secondary carriers excluded from both numerator and denominator.
        Defaults to ``DEFAULT_EXCLUDED_CARRIERS``.
    exclude_assets : Iterable[str] | None, optional
        Asset names or prefixes (e.g. data center PPA generators and batteries)
        excluded from both numerator and denominator.
    neighbours : Iterable[str] | None, optional
        Neighbouring country codes from which imports are traced. If ``None``,
        inferred automatically from cross-border lines and links.
    accounting_cfg : Mapping[str, Any] | None, optional
        Optional config dictionary with ``res_carriers`` / ``cfe_extra_carriers``.

    Returns
    -------
    pd.Series
        Hourly clean share ``r_t`` indexed by ``n.snapshots`` in ``[0, 1]``.
    """
    clean_set = _normalize_clean_carriers(clean_carriers, accounting_cfg)
    excluded_set = (
        set(excluded_carriers)
        if excluded_carriers is not None
        else set(DEFAULT_EXCLUDED_CARRIERS)
    )
    exclude_assets_set = (
        set(exclude_assets) if exclude_assets is not None else {"DC ", "DC_", "DC-"}
    )

    neighbour_list = (
        list(neighbours)
        if neighbours is not None
        else _infer_neighbours(n, country, exclude_assets_set)
    )

    clean_num, total_den = _domestic_supply(
        n=n,
        country=country,
        clean_set=clean_set,
        excluded_set=excluded_set,
        exclude_assets_set=exclude_assets_set,
    )

    for nb in neighbour_list:
        if nb == country:
            continue
        imp_nb = _corridor_import(
            n, exporter=nb, importer=country, exclude_assets_set=exclude_assets_set
        )
        if (imp_nb > 0.0).any():
            nb_clean, nb_total = _domestic_supply(
                n=n,
                country=nb,
                clean_set=clean_set,
                excluded_set=excluded_set,
                exclude_assets_set=exclude_assets_set,
            )
            nb_share = pd.Series(
                np.where(nb_total > 0.0, nb_clean / nb_total, 0.0),
                index=n.snapshots,
                dtype=float,
            )
            clean_num = clean_num + imp_nb * nb_share
            total_den = total_den + imp_nb

    r_values = np.where(total_den > 0.0, clean_num / total_den, 0.0)
    r_series = pd.Series(
        r_values, index=n.snapshots, name=f"grid_share_{country}", dtype=float
    )
    logger.info(
        f"Computed grid share for {country} (neighbours={neighbour_list}): mean={r_series.mean():.4f}"
    )
    return r_series


calculate_grid_res = calculate_grid_share
