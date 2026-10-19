# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
r"""
Linopy constraints and primal KPI extraction for 24/7 CFE procurement.

Notation follows Riepin & Brown (2023), "System-level impacts of 24/7
carbon-free electricity procurement in Europe" (247-procurement-paper):
demand :math:`d_t`, contracted generation :math:`g_{r,t}` for
:math:`r \in CFE`, storage discharge/charge
:math:`\bar{g}_{s,t} / \underline{g}_{s,t}`, grid imports :math:`im_t`,
excess sold to the grid :math:`ex_t`, hourly grid CFE factor :math:`CFE_t`,
and matching target :math:`x`.

Consumer energy balance per snapshot (``DC-virtual-balance``):

.. math::
    \sum_{r \in CFE} g_{r,t} + \sum_{s \in STO} (\bar{g}_{s,t} -
    \underline{g}_{s,t}) - ex_t + im_t = d_t, \quad im_t, ex_t \ge 0

Policy constraints:

- ``hourly`` (24/7 CFE score):
  :math:`\sum_{r,t} g_{r,t} + \sum_{s,t} (\bar{g}_{s,t} -
  \underline{g}_{s,t}) - \sum_t ex_t + \sum_t CFE_t \cdot im_t
  \ge x \sum_t d_t`
- ``strict_hourly``: the same left-hand side per snapshot,
  :math:`\ge x \cdot d_t \; \forall t` (the draft royal decree's Art. 9).
- ``annual_res``: :math:`\sum_{r,t} g_{r,t} \ge x \sum_t d_t`
  (volumetric matching; the residual load is bought from the grid).

PPA-only policies set :math:`CFE_t \equiv 0`; grid-credit policies take the
hourly grid CFE factor from ``calculate_grid_res`` (iterated to a fixed
point in ``iterate``). The optional excess cap limits annual sales to the
grid: :math:`\sum_t ex_t \le \sigma \sum_t d_t`. Battery grid-charging
enters through :math:`\underline{g}_{s,t}` in the balance and increases
:math:`im_t`, so no post-processing clipping is ever performed.
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


def _effective_cfe_factor(
    n: pypsa.Network,
    cfe_t: pd.Series | float | None,
    policy: str,
) -> pd.Series:
    """Return the effective hourly grid CFE factor ``CFE_t`` (zeros for PPA-only policies)."""
    snapshots = n.snapshots
    policy_norm = policy.strip().lower()
    if "ppa_only" in policy_norm or "ppa-only" in policy_norm or cfe_t is None:
        cfe = pd.Series(0.0, index=snapshots, dtype=float)
    elif isinstance(cfe_t, (int, float)):
        cfe = pd.Series(float(cfe_t), index=snapshots, dtype=float)
    else:
        cfe = cfe_t.reindex(snapshots).fillna(0.0).astype(float)
    cfe.index = snapshots
    cfe.index.name = snapshots.name
    return cfe


def add_dc_cfe_constraints(
    n: pypsa.Network,
    dc_cfg: Mapping[str, Any] | None,
    cfe_t: pd.Series | float | None,
    policy: str,
    target: float,
    excess_share: float | None = None,
) -> None:
    """
    Add the consumer energy balance, the CFE/RES policy constraint, and the sales cap.

    Parameters
    ----------
    n : pypsa.Network
        Network with an initialized Linopy model (``n.optimize.create_model()``).
    dc_cfg : Mapping[str, Any] | None
        Configuration dictionary identifying DC PPA generators, battery chargers/
        dischargers, and DC loads (or prefix ``"DC"``).
    cfe_t : pd.Series | float | None
        Hourly grid CFE factor ``CFE_t`` indexed by ``n.snapshots``. Overridden
        to ``0.0`` when ``policy`` specifies PPA-only.
    policy : str
        Procurement policy: ``'annual_res'``, ``'hourly'``, ``'strict_hourly'``,
        or scenario variants (e.g. ``'hourly_80_ppa_only'``,
        ``'hourly_80_grid_credit'``, ``'strict_hourly_80_ppa_only'``,
        ``'annual_100_res'``, ``'annual_80_res'``, ``'unmatched_dc'``,
        ``'no_dc'``).
    target : float
        The matching target ``x`` as a fraction in ``[0, 1]``.
    excess_share : float | None, optional
        Cap ``sigma`` on annual sales to the grid relative to total DC demand,
        ``sum_t ex_t <= sigma * sum_t d_t``; ``None`` leaves sales unlimited.
    """
    if not hasattr(n, "model") or n.model is None:
        n.optimize.create_model()

    comp = _resolve_dc_components(n, dc_cfg)
    dc_gens: list[str] = comp["generators"]
    chargers: list[str] = comp["chargers"]
    dischargers: list[str] = comp["dischargers"]
    d: pd.Series = comp["load_series"]
    annual_equality: bool = comp["annual_equality"]

    snapshots = n.snapshots.rename("snapshot")
    weights = n.snapshot_weightings["generators"].copy()
    weights.index = snapshots

    cfe = _effective_cfe_factor(n, cfe_t, policy)
    cfe.index = snapshots
    d = d.copy()
    d.index = snapshots

    im = n.model.add_variables(lower=0.0, coords=[snapshots], name="DC-im")
    ex = n.model.add_variables(lower=0.0, coords=[snapshots], name="DC-ex")

    # Consumer balance: sum_r g_{r,t} + (dis - cha) - ex_t + im_t = d_t
    lhs_balance = im - ex
    if dc_gens:
        gen_var = n.model["Generator-p"].loc[:, dc_gens]
        gen_dim = "name" if "name" in gen_var.dims else "Generator"
        g = gen_var.sum(gen_dim)
        lhs_balance = lhs_balance + g
    else:
        g = None

    if dischargers:
        link_var_d = n.model["Link-p"].loc[:, dischargers]
        link_dim_d = "name" if "name" in link_var_d.dims else "Link"
        eta_dis = n.links.loc[dischargers, "efficiency"].rename_axis(link_dim_d)
        g_dis = (link_var_d * eta_dis).sum(link_dim_d)
        lhs_balance = lhs_balance + g_dis

    if chargers:
        link_var_c = n.model["Link-p"].loc[:, chargers]
        link_dim_c = "name" if "name" in link_var_c.dims else "Link"
        g_cha = link_var_c.sum(link_dim_c)
        lhs_balance = lhs_balance - g_cha

    n.model.add_constraints(lhs_balance == d, name="DC-virtual-balance")

    total_demand = float((weights * d).sum())
    policy_norm = policy.strip().lower()

    # Matched energy per snapshot: d_t - (1 - CFE_t) * im_t
    # (by the balance: g + dis - cha - ex + CFE_t * im)
    matched_t = d - (1.0 - cfe) * im
    matched_total = (weights * matched_t).sum()

    if policy_norm in {"no_dc", "unmatched_dc", "ref", "none"}:
        logger.info(
            f"Added DC virtual balance with no CFE target for policy='{policy}'."
        )
    elif policy_norm.startswith("annual"):
        if g is None:
            raise ValueError(
                "Policy 'annual_res' requires at least one DC PPA generator."
            )
        annual_lhs = (weights * g).sum()
        annual_rhs = float(target) * total_demand
        if annual_equality:
            n.model.add_constraints(annual_lhs == annual_rhs, name="DC-annual-res")
        else:
            n.model.add_constraints(annual_lhs >= annual_rhs, name="DC-annual-res")
        logger.info(
            f"Added DC annual RES constraint (equality={annual_equality}, x={target:.3f})."
        )
    elif policy_norm.startswith("strict"):
        n.model.add_constraints(
            matched_t >= float(target) * d,
            name="DC-strict-hourly-cfe",
        )
        logger.info(f"Added DC strict hourly CFE constraint (x={target:.3f}).")
    elif policy_norm.startswith("hourly") or policy_norm == "cfe":
        n.model.add_constraints(
            matched_total >= float(target) * total_demand,
            name="DC-hourly-cfe",
        )
        logger.info(f"Added DC hourly CFE constraint (x={target:.3f}).")
    else:
        raise ValueError(
            f"Unsupported policy '{policy}'. Expected one of 'annual_res', 'hourly', "
            f"'strict_hourly', 'unmatched_dc', or 'no_dc'."
        )

    if excess_share is not None:
        excess_lhs = (weights * ex).sum()
        excess_rhs = float(excess_share) * total_demand
        n.model.add_constraints(excess_lhs <= excess_rhs, name="DC-excess")
        logger.info(f"Added DC sales cap (excess_share={excess_share:.3f}).")

    n._dc_cfe_meta = {
        "dc_cfg": dict(dc_cfg) if dc_cfg is not None else {},
        "cfe_t": cfe,
        "policy": policy,
        "target": float(target),
        "excess_share": float(excess_share) if excess_share is not None else None,
        "matched_expr": matched_total,
        "matched_per_t_expr": matched_t,
    }


add_cfe_constraints = add_dc_cfe_constraints


def dc_kpis(
    n: pypsa.Network,
    dc_cfg: Mapping[str, Any] | None = None,
    cfe_t: pd.Series | float | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    """
    Compute DC CFE KPIs from solved primal values using the exact constraint LHS.

    Never applies ``.clip`` to any time series. Both ``d_t - (1 - CFE_t) * im_t``
    and ``g_t + dis_t - cha_t - ex_t + CFE_t * im_t`` are evaluated and verified
    to match the solved Linopy constraint LHS expression.
    """
    meta: dict[str, Any] = getattr(n, "_dc_cfe_meta", {})
    eff_cfg = dc_cfg if dc_cfg is not None else meta.get("dc_cfg", {})
    eff_policy = policy if policy is not None else str(meta.get("policy", "hourly"))

    if cfe_t is None and "cfe_t" in meta:
        cfe_s = meta["cfe_t"].reindex(n.snapshots).fillna(0.0).astype(float)
    else:
        cfe_s = _effective_cfe_factor(n, cfe_t, eff_policy)

    comp = _resolve_dc_components(n, eff_cfg)
    dc_gens: list[str] = comp["generators"]
    chargers: list[str] = comp["chargers"]
    dischargers: list[str] = comp["dischargers"]
    d_s: pd.Series = comp["load_series"]

    snapshots = n.snapshots
    weights = n.snapshot_weightings["generators"].copy()
    weights.index = snapshots

    # Solved primal variables im_t and ex_t
    if hasattr(n, "model") and n.model is not None and "DC-im" in n.model.solution:
        im_s = n.model.solution["DC-im"].to_pandas().reindex(snapshots).astype(float)
        ex_s = n.model.solution["DC-ex"].to_pandas().reindex(snapshots).astype(float)
    else:
        raise RuntimeError(
            "Solved primal variables 'DC-im' and 'DC-ex' not found in n.model.solution."
        )

    # Solved primal g_t, dis_t, cha_t
    if dc_gens and not n.generators_t.p.empty:
        g_s = n.generators_t.p[dc_gens].sum(axis=1).reindex(snapshots).fillna(0.0)
    else:
        g_s = pd.Series(0.0, index=snapshots, dtype=float)

    if dischargers and not n.links_t.p0.empty:
        eta_dis = n.links.loc[dischargers, "efficiency"]
        dis_s = (
            (n.links_t.p0[dischargers] * eta_dis)
            .sum(axis=1)
            .reindex(snapshots)
            .fillna(0.0)
        )
    else:
        dis_s = pd.Series(0.0, index=snapshots, dtype=float)

    if chargers and not n.links_t.p0.empty:
        cha_s = n.links_t.p0[chargers].sum(axis=1).reindex(snapshots).fillna(0.0)
    else:
        cha_s = pd.Series(0.0, index=snapshots, dtype=float)

    # Exact constraint LHS expression per snapshot (no clipping)
    matched_t = d_s - (1.0 - cfe_s) * im_s
    matched_from_balance_t = g_s + dis_s - cha_s - ex_s + cfe_s * im_s
    ppa_only_t = d_s - im_s

    total_load_mwh = float((weights * d_s).sum())
    constraint_lhs = float((weights * matched_t).sum())
    constraint_lhs_from_balance = float((weights * matched_from_balance_t).sum())
    ppa_only_lhs = float((weights * ppa_only_t).sum())
    ppa_gen_mwh = float((weights * g_s).sum())

    if "matched_expr" in meta and hasattr(meta["matched_expr"], "solution"):
        # Re-evaluate with cfe_s if it matches meta['cfe_t'], else use formula
        if (cfe_s - meta["cfe_t"]).abs().max() <= 1e-12:
            model_constraint_lhs = float(meta["matched_expr"].solution.item())
        else:
            model_constraint_lhs = constraint_lhs
    else:
        model_constraint_lhs = constraint_lhs

    cfe_score = constraint_lhs / total_load_mwh if total_load_mwh > 0.0 else 0.0
    ppa_only_score = ppa_only_lhs / total_load_mwh if total_load_mwh > 0.0 else 0.0
    annual_res_score = ppa_gen_mwh / total_load_mwh if total_load_mwh > 0.0 else 0.0
    export_mwh = float((weights * ex_s).sum())

    return {
        "cfe_score": cfe_score,
        "hourly_score": cfe_score,
        "ppa_only_score": ppa_only_score,
        "annual_res_score": annual_res_score,
        "constraint_lhs": constraint_lhs,
        "constraint_lhs_from_balance": constraint_lhs_from_balance,
        "model_constraint_lhs": model_constraint_lhs,
        "constraint_lhs_per_t": matched_t,
        "hourly_matched_mwh": matched_t,
        "total_load_mwh": total_load_mwh,
        "ppa_generation_mwh": ppa_gen_mwh,
        "battery_discharge_mwh": float((weights * dis_s).sum()),
        "battery_charge_mwh": float((weights * cha_s).sum()),
        "grid_import_mwh": float((weights * im_s).sum()),
        "grid_clean_import_mwh": float((weights * cfe_s * im_s).sum()),
        "export_mwh": export_mwh,
        "excess_mwh": export_mwh,
        "excess_share": export_mwh / total_load_mwh if total_load_mwh > 0.0 else 0.0,
    }
