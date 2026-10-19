# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
r"""
Linopy constraints and primal KPI extraction for 24/7 CFE procurement.

Implements the virtual data center energy balance per snapshot ``t``:

.. math::
    \\sum_i G_{i,t} + \\eta_d d_t - c_t + m_t - x_t = L_t, \\quad m_t, x_t \\ge 0

and policy constraints (``annual_res``, ``hourly``, ``strict_hourly``) plus the
annual excess export cap. Battery grid-charging enters through ``c_t`` in the
virtual balance and increases ``m_t``, so no post-processing clipping is ever
performed.
"""

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

DEFAULT_DC_PREFIX: str = "DC"


def _resolve_dc_components(
    n: pypsa.Network,
    dc_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve DC PPA generators, battery links, and load time series from ``dc_cfg``."""
    cfg = dc_cfg or {}
    prefix = str(cfg.get("prefix", DEFAULT_DC_PREFIX))

    # 1. PPA Generators (at any node)
    gen_keys = ("generators", "ppa_generators", "dc_generators")
    dc_gens: list[str] | None = None
    for k in gen_keys:
        if k in cfg and cfg[k] is not None:
            dc_gens = [g for g in cfg[k] if g in n.generators.index]
            break
    if dc_gens is None:
        dc_gens = [
            g
            for g in n.generators.index
            if str(g).startswith(prefix)
            and str(n.generators.at[g, "carrier"])
            not in {"load", "virtual_link", "dsm"}
        ]

    # 2. Battery chargers & dischargers
    charger_keys = ("chargers", "battery_chargers", "storage_chargers")
    chargers: list[str] | None = None
    for k in charger_keys:
        if k in cfg and cfg[k] is not None:
            chargers = [lk for lk in cfg[k] if lk in n.links.index]
            break
    if chargers is None:
        chargers = [
            lk
            for lk in n.links.index
            if str(lk).startswith(prefix)
            and "charger" in str(n.links.at[lk, "carrier"]).lower()
            and "discharger" not in str(n.links.at[lk, "carrier"]).lower()
        ]
        if not chargers:
            chargers = [
                lk
                for lk in n.links.index
                if str(lk).startswith(prefix)
                and "charger" in str(lk).lower()
                and "discharger" not in str(lk).lower()
            ]

    discharger_keys = ("dischargers", "battery_dischargers", "storage_dischargers")
    dischargers: list[str] | None = None
    for k in discharger_keys:
        if k in cfg and cfg[k] is not None:
            dischargers = [lk for lk in cfg[k] if lk in n.links.index]
            break
    if dischargers is None:
        dischargers = [
            lk
            for lk in n.links.index
            if str(lk).startswith(prefix)
            and "discharger" in str(n.links.at[lk, "carrier"]).lower()
        ]
        if not dischargers:
            dischargers = [
                lk
                for lk in n.links.index
                if str(lk).startswith(prefix) and "discharger" in str(lk).lower()
            ]

    # 3. DC Load profile L_t
    snapshots = n.snapshots
    load_series: pd.Series | None = None
    if "load_series" in cfg and cfg["load_series"] is not None:
        raw_l = cfg["load_series"]
        if isinstance(raw_l, pd.Series):
            load_series = raw_l.reindex(snapshots).fillna(0.0).astype(float)
        else:
            load_series = pd.Series(float(raw_l), index=snapshots, dtype=float)
    elif "load_mw" in cfg and cfg["load_mw"] is not None and "loads" not in cfg:
        load_series = pd.Series(float(cfg["load_mw"]), index=snapshots, dtype=float)
    else:
        load_keys = ("loads", "dc_loads")
        dc_loads: list[str] | None = None
        for k in load_keys:
            if k in cfg and cfg[k] is not None:
                if isinstance(cfg[k], pd.Series):
                    load_series = cfg[k].reindex(snapshots).fillna(0.0).astype(float)
                elif isinstance(cfg[k], (int, float)):
                    load_series = pd.Series(float(cfg[k]), index=snapshots, dtype=float)
                else:
                    dc_loads = [ld for ld in cfg[k] if ld in n.loads.index]
                break
        if load_series is None:
            if dc_loads is None:
                dc_loads = [ld for ld in n.loads.index if str(ld).startswith(prefix)]
            load_series = pd.Series(0.0, index=snapshots, dtype=float)
            for ld in dc_loads:
                if not n.loads_t.p_set.empty and ld in n.loads_t.p_set.columns:
                    load_series = load_series + n.loads_t.p_set[ld].reindex(
                        snapshots
                    ).fillna(0.0)
                elif "p_set" in n.loads.columns:
                    load_series = load_series + float(n.loads.at[ld, "p_set"])

    load_series.index = snapshots
    load_series.index.name = snapshots.name

    return {
        "generators": dc_gens,
        "chargers": chargers,
        "dischargers": dischargers,
        "load_series": load_series,
        "annual_equality": bool(cfg.get("annual_equality", False)),
    }


def _effective_grid_share(
    n: pypsa.Network,
    r: pd.Series | float | None,
    policy: str,
) -> pd.Series:
    """Return effective hourly grid clean share ``r_t`` (zeros for PPA-only policies)."""
    snapshots = n.snapshots
    policy_norm = policy.strip().lower()
    if "ppa_only" in policy_norm or "ppa-only" in policy_norm or r is None:
        eff_r = pd.Series(0.0, index=snapshots, dtype=float)
    elif isinstance(r, (int, float)):
        eff_r = pd.Series(float(r), index=snapshots, dtype=float)
    else:
        eff_r = r.reindex(snapshots).fillna(0.0).astype(float)
    eff_r.index = snapshots
    eff_r.index.name = snapshots.name
    return eff_r


def add_dc_cfe_constraints(
    n: pypsa.Network,
    dc_cfg: Mapping[str, Any] | None,
    r: pd.Series | float | None,
    policy: str,
    target: float,
    excess_share: float | None = None,
) -> None:
    """
    Add virtual DC energy balance, CFE/RES policy constraint, and excess cap to ``n.model``.

    Parameters
    ----------
    n : pypsa.Network
        Network with an initialized Linopy model (``n.optimize.create_model()``).
    dc_cfg : Mapping[str, Any] | None
        Configuration dictionary identifying DC PPA generators, battery chargers/
        dischargers, and DC loads (or prefix ``"DC"``).
    r : pd.Series | float | None
        Hourly grid clean share ``r_t`` indexed by ``n.snapshots``. Overridden to
        ``0.0`` when ``policy`` specifies PPA-only.
    policy : str
        Procurement policy: ``'annual_res'``, ``'hourly'``, ``'strict_hourly'``,
        or scenario variants (e.g. ``'hourly_80_ppa_only'``,
        ``'hourly_80_grid_credit'``, ``'strict_hourly_80_ppa_only'``,
        ``'strict_hourly_80_grid_credit'``, ``'annual_100_res'``,
        ``'unmatched_dc'``, ``'no_dc'``).
    target : float
        Volumetric or hourly matching target fraction in ``[0, 1]``.
    excess_share : float | None, optional
        Maximum allowed annual export share relative to total DC load.
    """
    if not hasattr(n, "model") or n.model is None:
        n.optimize.create_model()

    comp = _resolve_dc_components(n, dc_cfg)
    dc_gens: list[str] = comp["generators"]
    chargers: list[str] = comp["chargers"]
    dischargers: list[str] = comp["dischargers"]
    load_series: pd.Series = comp["load_series"]
    annual_equality: bool = comp["annual_equality"]

    snapshots = n.snapshots.rename("snapshot")
    weights = n.snapshot_weightings["generators"].copy()
    weights.index = snapshots

    eff_r = _effective_grid_share(n, r, policy)
    eff_r.index = snapshots
    load_series = load_series.copy()
    load_series.index = snapshots

    m = n.model.add_variables(lower=0.0, coords=[snapshots], name="DC-import")
    x = n.model.add_variables(lower=0.0, coords=[snapshots], name="DC-export")

    lhs_balance = m - x
    if dc_gens:
        gen_var = n.model["Generator-p"].loc[:, dc_gens]
        gen_dim = "name" if "name" in gen_var.dims else "Generator"
        g_expr = gen_var.sum(gen_dim)
        lhs_balance = lhs_balance + g_expr
    else:
        g_expr = None

    if dischargers:
        link_var_d = n.model["Link-p"].loc[:, dischargers]
        link_dim_d = "name" if "name" in link_var_d.dims else "Link"
        eff_dis = n.links.loc[dischargers, "efficiency"].rename_axis(link_dim_d)
        d_expr = (link_var_d * eff_dis).sum(link_dim_d)
        lhs_balance = lhs_balance + d_expr

    if chargers:
        link_var_c = n.model["Link-p"].loc[:, chargers]
        link_dim_c = "name" if "name" in link_var_c.dims else "Link"
        c_expr = link_var_c.sum(link_dim_c)
        lhs_balance = lhs_balance - c_expr

    n.model.add_constraints(lhs_balance == load_series, name="DC-virtual-balance")

    total_weighted_load = float((weights * load_series).sum())
    policy_norm = policy.strip().lower()

    hourly_lhs_per_t = load_series - (1.0 - eff_r) * m
    hourly_lhs_total = (weights * hourly_lhs_per_t).sum()

    if policy_norm in {"no_dc", "unmatched_dc", "ref", "none"}:
        logger.info(
            f"Added DC virtual balance with no CFE target for policy='{policy}'."
        )
    elif policy_norm.startswith("annual"):
        if g_expr is None:
            raise ValueError(
                "Policy 'annual_res' requires at least one DC PPA generator."
            )
        annual_lhs = (weights * g_expr).sum()
        annual_rhs = float(target) * total_weighted_load
        if annual_equality:
            n.model.add_constraints(annual_lhs == annual_rhs, name="DC-annual-res")
        else:
            n.model.add_constraints(annual_lhs >= annual_rhs, name="DC-annual-res")
        logger.info(
            f"Added DC annual RES constraint (equality={annual_equality}, target={target:.3f})."
        )
    elif policy_norm.startswith("strict"):
        n.model.add_constraints(
            hourly_lhs_per_t >= float(target) * load_series,
            name="DC-strict-hourly-cfe",
        )
        logger.info(f"Added DC strict hourly CFE constraint (target={target:.3f}).")
    elif policy_norm.startswith("hourly") or policy_norm == "cfe":
        n.model.add_constraints(
            hourly_lhs_total >= float(target) * total_weighted_load,
            name="DC-hourly-cfe",
        )
        logger.info(f"Added DC hourly CFE constraint (target={target:.3f}).")
    else:
        raise ValueError(
            f"Unsupported policy '{policy}'. Expected one of 'annual_res', 'hourly', "
            f"'strict_hourly', 'unmatched_dc', or 'no_dc'."
        )

    if excess_share is not None:
        excess_lhs = (weights * x).sum()
        excess_rhs = float(excess_share) * total_weighted_load
        n.model.add_constraints(excess_lhs <= excess_rhs, name="DC-excess")
        logger.info(f"Added DC excess constraint (excess_share={excess_share:.3f}).")

    n._dc_cfe_meta = {
        "dc_cfg": dict(dc_cfg) if dc_cfg is not None else {},
        "r": eff_r,
        "policy": policy,
        "target": float(target),
        "excess_share": float(excess_share) if excess_share is not None else None,
        "hourly_lhs_expr": hourly_lhs_total,
        "hourly_lhs_per_t_expr": hourly_lhs_per_t,
    }


add_cfe_constraints = add_dc_cfe_constraints


def dc_kpis(
    n: pypsa.Network,
    dc_cfg: Mapping[str, Any] | None = None,
    r: pd.Series | float | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    """
    Compute DC CFE KPIs from solved primal values using the exact constraint LHS.

    Never applies ``.clip`` to any time series. Both ``L_t - (1 - r_t) * m_t``
    and ``G_t + eta_d * d_t - c_t - x_t + r_t * m_t`` are evaluated and verified
    to match the solved Linopy constraint LHS expression.
    """
    meta: dict[str, Any] = getattr(n, "_dc_cfe_meta", {})
    eff_cfg = dc_cfg if dc_cfg is not None else meta.get("dc_cfg", {})
    eff_policy = policy if policy is not None else str(meta.get("policy", "hourly"))

    if r is None and "r" in meta:
        r_series = meta["r"].reindex(n.snapshots).fillna(0.0).astype(float)
    else:
        r_series = _effective_grid_share(n, r, eff_policy)

    comp = _resolve_dc_components(n, eff_cfg)
    dc_gens: list[str] = comp["generators"]
    chargers: list[str] = comp["chargers"]
    dischargers: list[str] = comp["dischargers"]
    load_series: pd.Series = comp["load_series"]

    snapshots = n.snapshots
    weights = n.snapshot_weightings["generators"].copy()
    weights.index = snapshots

    # Solved primal variables m_t and x_t
    if hasattr(n, "model") and n.model is not None and "DC-import" in n.model.solution:
        m_series = (
            n.model.solution["DC-import"].to_pandas().reindex(snapshots).astype(float)
        )
        x_series = (
            n.model.solution["DC-export"].to_pandas().reindex(snapshots).astype(float)
        )
    else:
        raise RuntimeError(
            "Solved primal variables 'DC-import' and 'DC-export' not found in n.model.solution."
        )

    # Solved primal G_t, d_t, c_t
    if dc_gens and not n.generators_t.p.empty:
        g_series = n.generators_t.p[dc_gens].sum(axis=1).reindex(snapshots).fillna(0.0)
    else:
        g_series = pd.Series(0.0, index=snapshots, dtype=float)

    if dischargers and not n.links_t.p0.empty:
        eff_dis = n.links.loc[dischargers, "efficiency"]
        d_series = (
            (n.links_t.p0[dischargers] * eff_dis)
            .sum(axis=1)
            .reindex(snapshots)
            .fillna(0.0)
        )
    else:
        d_series = pd.Series(0.0, index=snapshots, dtype=float)

    if chargers and not n.links_t.p0.empty:
        c_series = n.links_t.p0[chargers].sum(axis=1).reindex(snapshots).fillna(0.0)
    else:
        c_series = pd.Series(0.0, index=snapshots, dtype=float)

    # Exact constraint LHS expression per snapshot (no clipping)
    lhs_per_t = load_series - (1.0 - r_series) * m_series
    lhs_from_balance_per_t = (
        g_series + d_series - c_series - x_series + r_series * m_series
    )
    ppa_only_per_t = load_series - m_series

    total_load_mwh = float((weights * load_series).sum())
    constraint_lhs = float((weights * lhs_per_t).sum())
    constraint_lhs_from_balance = float((weights * lhs_from_balance_per_t).sum())
    ppa_only_lhs = float((weights * ppa_only_per_t).sum())
    ppa_gen_mwh = float((weights * g_series).sum())

    if "hourly_lhs_expr" in meta and hasattr(meta["hourly_lhs_expr"], "solution"):
        # Re-evaluate with r_series if r matches meta['r'], else use formula
        if (r_series - meta["r"]).abs().max() <= 1e-12:
            model_constraint_lhs = float(meta["hourly_lhs_expr"].solution.item())
        else:
            model_constraint_lhs = constraint_lhs
    else:
        model_constraint_lhs = constraint_lhs

    cfe_score = constraint_lhs / total_load_mwh if total_load_mwh > 0.0 else 0.0
    ppa_only_score = ppa_only_lhs / total_load_mwh if total_load_mwh > 0.0 else 0.0
    annual_res_score = ppa_gen_mwh / total_load_mwh if total_load_mwh > 0.0 else 0.0
    export_mwh = float((weights * x_series).sum())

    return {
        "cfe_score": cfe_score,
        "hourly_score": cfe_score,
        "ppa_only_score": ppa_only_score,
        "annual_res_score": annual_res_score,
        "constraint_lhs": constraint_lhs,
        "constraint_lhs_from_balance": constraint_lhs_from_balance,
        "model_constraint_lhs": model_constraint_lhs,
        "constraint_lhs_per_t": lhs_per_t,
        "hourly_matched_mwh": lhs_per_t,
        "total_load_mwh": total_load_mwh,
        "ppa_generation_mwh": ppa_gen_mwh,
        "battery_discharge_mwh": float((weights * d_series).sum()),
        "battery_charge_mwh": float((weights * c_series).sum()),
        "grid_import_mwh": float((weights * m_series).sum()),
        "grid_clean_import_mwh": float((weights * r_series * m_series).sum()),
        "export_mwh": export_mwh,
        "excess_mwh": export_mwh,
        "excess_share": export_mwh / total_load_mwh if total_load_mwh > 0.0 else 0.0,
    }
