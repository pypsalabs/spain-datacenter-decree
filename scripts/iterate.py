# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Forward-pass iteration for the endogenous grid CFE factor ``CFE_t``.

Resolves the bilinear coupling between the data center's grid imports ``im_t``
and the grid CFE factor ``CFE_t`` the way Riepin & Brown (2024) do: solve with
``CFE_t`` fixed, recompute it from the solved dispatch (excluding DC assets),
and repeat for a **fixed number of forward passes** (default 2) with plain
replacement — no damping, no stopping criterion. One forward pass converges
well in practice; the per-iteration objective and ``max_t |ΔCFE_t|`` are
reported in the convergence table as the signal that a scenario (e.g. a very
large DC) might need more passes.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd
import pypsa

from scripts.calculate_grid_cfe import align_to_snapshots, calculate_grid_cfe

logger = logging.getLogger(__name__)

DEFAULT_N_ITERATIONS: int = 2
DEFAULT_SOLVER_NAME: str = "highs"


def _coerce_initial_cfe(
    n: pypsa.Network,
    cfe0: pd.Series | float | None,
    grid_cfe_kwargs: Mapping[str, Any],
) -> pd.Series:
    """Return the initial hourly grid CFE series ``CFE^(0)`` indexed by ``n.snapshots``."""
    snapshots = n.snapshots
    if cfe0 is None:
        if not n.generators_t.p.empty:
            cfe_init = calculate_grid_cfe(n, **grid_cfe_kwargs)
        else:
            cfe_init = pd.Series(0.0, index=snapshots, dtype=float)
    elif isinstance(cfe0, (int, float)):
        cfe_init = pd.Series(float(cfe0), index=snapshots, dtype=float)
    else:
        cfe_init = align_to_snapshots(cfe0, snapshots)

    cfe_init.index = snapshots
    cfe_init.index.name = snapshots.name
    return cfe_init


def solve_with_grid_iteration(
    n: pypsa.Network,
    build_fn: Callable[[pypsa.Network, pd.Series], Any],
    r0: pd.Series | float | None = None,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    solver_options: Mapping[str, Any] | None = None,
    country: str = "ES",
    clean_carriers: Iterable[str] | str | None = None,
) -> tuple[pypsa.Network, pd.DataFrame]:
    """
    Solve ``n_iterations`` forward passes for ``CFE_t``.

    Parameters
    ----------
    n : pypsa.Network
        Network to optimize.
    build_fn : Callable[[pypsa.Network, pd.Series], Any]
        Callback ``build_fn(n, cfe_t)`` that adds the CFE constraints for the
        current grid CFE series (and optionally creates the model).
    r0 : pd.Series | float | None, optional
        Initial hourly grid CFE series ``CFE^(0)`` (e.g. from the solved
        ``unmatched_dc`` network; zeros if the network is unsolved).
    n_iterations : int, default 2
        Number of forward passes. Each pass solves once and recomputes
        ``CFE_t`` from the solved dispatch by plain replacement.
    solver_options : Mapping[str, Any] | None, optional
        Solver options for ``n.optimize.solve_model``; ``solver_name`` may be
        included (default ``"highs"``). The caller supplies per-solver defaults.
    country : str, default "ES"
        Country code for ``calculate_grid_cfe``.
    clean_carriers : Iterable[str] | str | None, optional
        Clean carrier set or mode (``"res"`` / ``"cfe"``) for ``calculate_grid_cfe``.

    Returns
    -------
    tuple[pypsa.Network, pd.DataFrame]
        Solved ``(n, convergence_df)`` with one row per pass and columns
        ``["iteration", "max_delta", "objective", "mean_cfe_in", "mean_cfe_out"]``.
    """
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1; got {n_iterations}.")

    grid_kwargs: dict[str, Any] = {
        "country": country,
        "clean_carriers": clean_carriers,
    }

    cfe = _coerce_initial_cfe(n, r0, grid_kwargs)

    opts: dict[str, Any] = dict(solver_options or {})
    solver_name = str(opts.pop("solver_name", DEFAULT_SOLVER_NAME))

    records: list[dict[str, Any]] = []

    for iteration in range(n_iterations):
        n.optimize.create_model()
        build_fn(n, cfe)

        status, condition = n.optimize.solve_model(
            solver_name=solver_name,
            **opts,
        )
        if status != "ok":
            raise RuntimeError(
                f"Optimization failed at iteration {iteration}: status={status}, condition={condition}"
            )

        cfe_new = calculate_grid_cfe(n, **grid_kwargs)

        cfe_new.index = n.snapshots
        cfe_new.index.name = n.snapshots.name

        max_delta = float((cfe_new - cfe).abs().max())
        objective = float(n.objective)

        records.append(
            {
                "iteration": iteration,
                "max_delta": max_delta,
                "objective": objective,
                "mean_cfe_in": float(cfe.mean()),
                "mean_cfe_out": float(cfe_new.mean()),
            }
        )

        logger.info(
            f"Grid CFE pass {iteration}: max_delta={max_delta:.6f}, "
            f"objective={objective:.4f}, mean_cfe={cfe_new.mean():.4f}"
        )

        n._grid_cfe = cfe_new
        cfe = cfe_new

    convergence_df = pd.DataFrame(records)
    return n, convergence_df
