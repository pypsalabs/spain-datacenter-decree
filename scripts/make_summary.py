# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT
"""
Scenario summary baseline delta calculations.

Computes deltas relative to two distinct baselines:
- ``no_dc``: matched on ``(planning_year, weather_year, resolution)``
- ``unmatched_dc``: matched on ``(planning_year, weather_year, resolution, dc_mw)``
"""

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NO_DC_BASELINE_KEYS: tuple[str, ...] = (
    "planning_year",
    "weather_year",
    "resolution",
)
UNMATCHED_DC_BASELINE_KEYS: tuple[str, ...] = (
    "planning_year",
    "weather_year",
    "resolution",
    "dc_mw",
)
IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {
        "planning_year",
        "weather_year",
        "resolution",
        "dc_mw",
        "dc_load_mw",
        "policy",
        "policy_type",
        "target",
        "siting",
        "suite",
        "scenario",
        "excess_share",
    }
)
MIN_INDUCED_DENOMINATOR: float = 1.0e-9


def _resolve_key_columns(df: pd.DataFrame) -> tuple[list[str], list[str], str]:
    """Return ``(no_dc_keys, unmatched_keys, policy_col)`` validated against ``df``."""
    policy_col = "policy" if "policy" in df.columns else "policy_type"
    if policy_col not in df.columns:
        raise KeyError("DataFrame must contain a 'policy' (or 'policy_type') column.")

    no_dc_keys = [k for k in NO_DC_BASELINE_KEYS if k in df.columns]
    if len(no_dc_keys) != len(NO_DC_BASELINE_KEYS):
        missing = set(NO_DC_BASELINE_KEYS) - set(no_dc_keys)
        raise KeyError(
            f"DataFrame is missing required no_dc key columns: {sorted(missing)}"
        )

    dc_mw_col = "dc_mw" if "dc_mw" in df.columns else "dc_load_mw"
    if dc_mw_col not in df.columns:
        raise KeyError("DataFrame must contain a 'dc_mw' (or 'dc_load_mw') column.")

    unmatched_keys = [*no_dc_keys, dc_mw_col]
    return no_dc_keys, unmatched_keys, policy_col


def attach_baseline_deltas(
    df: pd.DataFrame,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Attach deltas against ``no_dc`` and ``unmatched_dc`` baselines to ``df``.

    - ``no_dc`` baseline matches on ``(planning_year, weather_year, resolution)``.
    - ``unmatched_dc`` baseline matches on
      ``(planning_year, weather_year, resolution, dc_mw)``.

    For each numeric metric column ``col``, creates:
    - ``delta_no_dc_{col}`` and ``{col}_delta_no_dc``
    - ``delta_unmatched_dc_{col}``, ``delta_unmatched_{col}``, and ``{col}_delta_unmatched``

    If an emissions column (e.g. ``emissions`` or ``es_total_co2_mt``) is present,
    also computes ``dc_co2_reduction_vs_unmatched_pct`` where every ``unmatched_dc``
    row is guaranteed to have ``0.0`` delta and ``0.0%`` abatement.
    """
    out = df.copy()
    if out.empty:
        return out

    no_dc_keys, unmatched_keys, policy_col = _resolve_key_columns(out)

    if metrics is not None:
        metric_cols = [c for c in metrics if c in out.columns]
    else:
        metric_cols = [
            c
            for c in out.select_dtypes(include=[np.number]).columns
            if c not in IDENTIFIER_COLUMNS
            and not c.startswith("delta_")
            and "_delta_" not in c
        ]

    no_dc_df = (
        out.loc[out[policy_col] == "no_dc"]
        .drop_duplicates(subset=no_dc_keys, keep="first")
        .set_index(no_dc_keys)
    )
    unm_df = (
        out.loc[out[policy_col] == "unmatched_dc"]
        .drop_duplicates(subset=unmatched_keys, keep="first")
        .set_index(unmatched_keys)
    )

    row_no_dc_idx = pd.MultiIndex.from_frame(out[no_dc_keys])
    row_unm_idx = pd.MultiIndex.from_frame(out[unmatched_keys])

    for col in metric_cols:
        base_no_dc = no_dc_df[col].reindex(row_no_dc_idx).to_numpy(dtype=float)
        base_unm = unm_df[col].reindex(row_unm_idx).to_numpy(dtype=float)
        vals = out[col].to_numpy(dtype=float)

        delta_no_dc = vals - base_no_dc
        delta_unm = vals - base_unm

        out[f"delta_no_dc_{col}"] = delta_no_dc
        out[f"{col}_delta_no_dc"] = delta_no_dc
        out[f"delta_unmatched_dc_{col}"] = delta_unm
        out[f"delta_unmatched_{col}"] = delta_unm
        out[f"{col}_delta_unmatched"] = delta_unm
        out[f"{col}_delta_unmatched_dc"] = delta_unm

    # Compute DC-induced emissions abatement percentage if an emissions column exists
    co2_candidates = ("emissions", "es_total_co2_mt", "co2_mt", "co2_emissions")
    for co2_col in co2_candidates:
        if co2_col in metric_cols:
            base_no_dc_co2 = (
                no_dc_df[co2_col].reindex(row_no_dc_idx).to_numpy(dtype=float)
            )
            base_unm_co2 = unm_df[co2_col].reindex(row_unm_idx).to_numpy(dtype=float)
            vals_co2 = out[co2_col].to_numpy(dtype=float)

            dc_induced = base_unm_co2 - base_no_dc_co2
            reduction = base_unm_co2 - vals_co2
            pct = np.where(
                np.abs(dc_induced) > MIN_INDUCED_DENOMINATOR,
                (reduction / dc_induced) * 100.0,
                0.0,
            )
            out["dc_co2_reduction_vs_unmatched_pct"] = pct
            break

    logger.info(
        f"Attached baseline deltas for {len(metric_cols)} metric columns across {len(out)} rows."
    )
    return out
