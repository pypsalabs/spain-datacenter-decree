# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Collect scenario metrics rows into one summary table with baseline deltas.

Reads the per-scenario summary rows written by ``summarise_network.py``,
concatenates them and attaches deltas relative to two distinct baselines:
- ``no_dc``: matched on ``(planning_year, weather_year, resolution, variant)``
- ``unmatched_dc``: matched on the same keys plus ``dc_mw``
"""

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NO_DC_BASELINE_KEYS: tuple[str, ...] = (
    "planning_year",
    "weather_year",
    "resolution",
    "variant",
)
UNMATCHED_DC_BASELINE_KEYS: tuple[str, ...] = (*NO_DC_BASELINE_KEYS, "dc_mw")
# Scenario identifiers and run bookkeeping: excluded from delta metrics
IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {
        "planning_year",
        "weather_year",
        "variant",
        "sites",
        "resolution",
        "dc_mw",
        "policy",
        "target",
        "siting",
        "excess_cap",
        "solve_time_s",
        "iterations",
        "max_delta",
    }
)
MIN_INDUCED_DENOMINATOR: float = 1.0e-9


def _resolve_key_columns(df: pd.DataFrame) -> tuple[list[str], list[str], str]:
    """Return ``(no_dc_keys, unmatched_keys, policy_col)`` validated against ``df``."""
    policy_col = "policy"
    if policy_col not in df.columns:
        raise KeyError("DataFrame must contain a 'policy' column.")

    no_dc_keys = [k for k in NO_DC_BASELINE_KEYS if k in df.columns]
    if len(no_dc_keys) != len(NO_DC_BASELINE_KEYS):
        missing = set(NO_DC_BASELINE_KEYS) - set(no_dc_keys)
        raise KeyError(
            f"DataFrame is missing required no_dc key columns: {sorted(missing)}"
        )

    if "dc_mw" not in df.columns:
        raise KeyError("DataFrame must contain a 'dc_mw' column.")

    unmatched_keys = [*no_dc_keys, "dc_mw"]
    return no_dc_keys, unmatched_keys, policy_col


def attach_baseline_deltas(
    df: pd.DataFrame,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Attach deltas against ``no_dc`` and ``unmatched_dc`` baselines to ``df``.

    - ``no_dc`` baseline matches on ``NO_DC_BASELINE_KEYS``.
    - ``unmatched_dc`` baseline matches on ``UNMATCHED_DC_BASELINE_KEYS``.

    For each numeric metric column ``col``, creates ``delta_no_dc_{col}`` and
    ``delta_unmatched_dc_{col}``.

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
            if c not in IDENTIFIER_COLUMNS and not c.startswith("delta_")
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
        out[f"delta_unmatched_dc_{col}"] = delta_unm

    # Compute DC-induced emissions abatement percentage if an emissions column exists
    co2_candidates = ("emissions", "es_total_co2_mt")
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


def collect_metrics(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Concatenate the one-row metrics CSVs of all scenarios into one table."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("No metrics files given; define the scenario suite first.")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: collect metrics CSVs, attach baseline deltas, write the summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = attach_baseline_deltas(collect_metrics(args.metrics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    logger.info("Wrote %d scenario rows to %s", len(summary), args.output)


if __name__ == "__main__":
    main()
