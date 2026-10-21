# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
r"""
Linopy constraints and primal metrics extraction for 24/7 CFE procurement.

Notation follows Riepin & Brown (2024), "On the means, costs, and system-level
impacts of 24/7 carbon-free energy procurement" (doi:10.1016/j.esr.2024.101488):
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
- ``annual_res``: :math:`\sum_{r,t} a_{r,t} G_r = x \sum_t d_t`, the contracted
  PPA yield (availability :math:`a_{r,t}` times capacity :math:`G_r`, whether the
  output is used, sold or curtailed) equals the target volume, as in Riepin &
  Brown (2024); the residual load is bought from the grid.

PPA-only policies set :math:`CFE_t \equiv 0`; grid-credit policies take the
hourly grid CFE factor from ``calculate_grid_cfe`` (resolved with fixed forward
passes in ``iterate``), which counts **renewables only** by default — nuclear
enters only under the ``_cfe`` policy suffix. The optional excess cap limits annual sales to the
grid: :math:`\sum_t ex_t \le \sigma \sum_t d_t`. A tiny objective cost on
:math:`ex_t` rules out buying and selling in the same snapshot.
Battery grid-charging
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
# EUR/MWh on sales ex_t: rules out buying and selling in the same snapshot (a cost
# on one leg suffices) while staying negligible next to the policy costs
EX_COST: float = 0.01


def _resolve_dc_components(
    n: pypsa.Network,
    dc_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve DC PPA generators, battery links, and load series by name prefix."""
    cfg = dc_cfg or {}
    prefix = str(cfg.get("prefix", DEFAULT_DC_PREFIX))

    dc_gens = [
        g
        for g in n.generators.index
        if str(g).startswith(prefix)
        and str(n.generators.at[g, "carrier"]) not in {"load"}
    ]

    def _links_with(token: str, without: str | None = None) -> list[str]:
        out = []
        for lk in n.links.index:
            if not str(lk).startswith(prefix):
                continue
            carrier = str(n.links.at[lk, "carrier"]).lower()
            name = str(lk).lower()
            hit = token in carrier or token in name
            miss = without is not None and (without in carrier or without in name)
            if hit and not miss:
                out.append(lk)
        return out

    chargers = _links_with("charger", without="discharger")
    dischargers = _links_with("discharger")

    snapshots = n.snapshots
    load_series = pd.Series(0.0, index=snapshots, dtype=float)
    for ld in n.loads.index:
        if not str(ld).startswith(prefix):
            continue
        if not n.loads_t.p_set.empty and ld in n.loads_t.p_set.columns:
            load_series += n.loads_t.p_set[ld].reindex(snapshots).fillna(0.0)
        elif "p_set" in n.loads.columns:
            load_series += float(n.loads.at[ld, "p_set"])

    load_series.index = snapshots
    load_series.index.name = snapshots.name

    return {
        "generators": dc_gens,
        "chargers": chargers,
        "dischargers": dischargers,
        "load_series": load_series,
    }


def yield_per_mw(n: pypsa.Network, gens: list[str]) -> pd.Series:
    """Annual energy (MWh) each generator yields per MW of capacity, used or not."""
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu")[gens]
    return p_max_pu.mul(n.snapshot_weightings["generators"], axis=0).sum()


def _effective_cfe_factor(
    n: pypsa.Network,
    cfe_t: pd.Series | float | None,
    policy: str,
) -> pd.Series:
    """Return the effective hourly grid CFE factor ``CFE_t`` (zeros for PPA-only policies)."""
    snapshots = n.snapshots
    policy_norm = policy.strip().lower()
    if "ppa_only" in policy_norm or cfe_t is None:
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

    snapshots = n.snapshots.rename("snapshot")
    weights = n.snapshot_weightings["generators"].copy()
    weights.index = snapshots

    cfe = _effective_cfe_factor(n, cfe_t, policy)
    cfe.index = snapshots
    d = d.copy()
    d.index = snapshots

    im = n.model.add_variables(lower=0.0, coords=[snapshots], name="DC-im")
    ex = n.model.add_variables(lower=0.0, coords=[snapshots], name="DC-ex")
    obj_weights = n.snapshot_weightings["objective"].copy()
    obj_weights.index = snapshots
    n.model.objective += (obj_weights * EX_COST * ex).sum()

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

    if policy_norm in {"no_dc", "unmatched_dc"}:
        logger.info(
            f"Added DC virtual balance with no CFE target for policy='{policy}'."
        )
    elif policy_norm.startswith("annual"):
        if g is None:
            raise ValueError(
                "Policy 'annual_res' requires at least one DC PPA generator."
            )
        capacity = n.model["Generator-p_nom"].loc[dc_gens]
        per_mw = yield_per_mw(n, dc_gens).rename_axis(capacity.dims[0])
        n.model.add_constraints(
            (capacity * per_mw).sum() == float(target) * total_demand,
            name="DC-annual-res",
        )
        logger.info(
            f"Added DC annual RES constraint on contracted yield (x={target:.3f})."
        )
    elif policy_norm.startswith("strict"):
        n.model.add_constraints(
            matched_t >= float(target) * d,
            name="DC-strict-hourly-cfe",
        )
        logger.info(f"Added DC strict hourly CFE constraint (x={target:.3f}).")
    elif policy_norm.startswith("hourly"):
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
    }


def dc_metrics(
    n: pypsa.Network,
    dc_cfg: Mapping[str, Any] | None = None,
    cfe_t: pd.Series | float | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    """
    Compute DC CFE metrics from solved primal values using the exact constraint LHS.

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
    export_mwh = float((weights * ex_s).sum())

    return {
        "cfe_score": cfe_score,
        "ppa_only_score": ppa_only_score,
        "constraint_lhs": constraint_lhs,
        "constraint_lhs_from_balance": constraint_lhs_from_balance,
        "model_constraint_lhs": model_constraint_lhs,
        "constraint_lhs_per_t": matched_t,
        "im_t": im_s,
        "ex_t": ex_s,
        "total_load_mwh": total_load_mwh,
        "ppa_generation_mwh": ppa_gen_mwh,
        "battery_discharge_mwh": float((weights * dis_s).sum()),
        "battery_charge_mwh": float((weights * cha_s).sum()),
        "grid_import_mwh": float((weights * im_s).sum()),
        "grid_clean_import_mwh": float((weights * cfe_s * im_s).sum()),
        "export_mwh": export_mwh,
        "excess_share": export_mwh / total_load_mwh if total_load_mwh > 0.0 else 0.0,
    }
