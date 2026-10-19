# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Fixed-point iteration loop for endogenous grid CFE share ``r_t``.

Resolves the bilinear coupling between the data center's grid import ``m_t``
and the background grid clean share ``r_t`` by iteratively building the
Linopy model with ``r^(k)``, solving with HiGHS Simplex, and recomputing
``r^(k+1)`` from the solved dispatch (excluding DC assets) until
``max_t |r_new - r| < tol``.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd
import pypsa

from scripts.calculate_grid_res import calculate_grid_share

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITER: int = 6
DEFAULT_TOL: float = 1.0e-3
DEFAULT_DAMPING: float = 1.0
DEFAULT_SOLVER_NAME: str = "highs"
DEFAULT_SOLVER_OPTIONS: dict[str, Any] = {"solver": "simplex"}


def _coerce_initial_r(
    n: pypsa.Network,
    r0: pd.Series | float | None,
    grid_share_kwargs: Mapping[str, Any],
) -> pd.Series:
    """Return initial hourly grid share series ``r^(0)`` indexed by ``n.snapshots``."""
    snapshots = n.snapshots
    if r0 is None:
        if not n.generators_t.p.empty:
            r_init = calculate_grid_share(n, **grid_share_kwargs)
        else:
            r_init = pd.Series(0.0, index=snapshots, dtype=float)
    elif isinstance(r0, (int, float)):
        r_init = pd.Series(float(r0), index=snapshots, dtype=float)
    else:
        r_init = r0.reindex(snapshots).fillna(0.0).astype(float)

    r_init.index = snapshots
    r_init.index.name = snapshots.name
    return r_init


def solve_with_grid_iteration(
    n: pypsa.Network,
    build_fn: Callable[[pypsa.Network, pd.Series], Any],
    r0: pd.Series | float | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    damping: float = DEFAULT_DAMPING,
    solver_options: Mapping[str, Any] | None = None,
    country: str = "ES",
    clean_carriers: Iterable[str] | str | None = None,
    excluded_carriers: Iterable[str] | None = None,
    exclude_assets: Iterable[str] | None = None,
    neighbours: Iterable[str] | None = None,
    grid_share_fn: Callable[[pypsa.Network], pd.Series] | None = None,
) -> tuple[pypsa.Network, pd.DataFrame]:
    """
    Run the fixed-point iteration loop for ``r_t`` with HiGHS Simplex.

    Parameters
    ----------
    n : pypsa.Network
        Network to optimize.
    build_fn : Callable[[pypsa.Network, pd.Series], Any]
        Callback ``build_fn(n, r)`` that adds CFE constraints for the current
        grid share series ``r`` (and optionally creates the model).
    r0 : pd.Series | float | None, optional
        Initial hourly grid clean share ``r^(0)``.
    max_iter : int, default 6
        Maximum number of fixed-point iterations.
    tol : float, default 1e-3
        Convergence threshold on ``max_t |r_new - r|``.
    damping : float, default 1.0
        Damping factor ``alpha`` in ``(0, 1]`` for ``r = alpha * r_new + (1 - alpha) * r``.
    solver_options : Mapping[str, Any] | None, optional
        Additional options passed to ``n.optimize.solve_model(solver_name="highs", ...)``.
        Defaults to ``{"solver": "simplex"}``.
    country : str, default "ES"
        Country code for ``calculate_grid_share``.
    clean_carriers : Iterable[str] | str | None, optional
        Clean carrier set or mode (``"res"`` / ``"cfe"``) for ``calculate_grid_share``.
    excluded_carriers : Iterable[str] | None, optional
        Excluded storage carriers for ``calculate_grid_share``.
    exclude_assets : Iterable[str] | None, optional
        Excluded DC assets or prefixes for ``calculate_grid_share``.
    neighbours : Iterable[str] | None, optional
        Neighbouring country codes for import tracing.
    grid_share_fn : Callable[[pypsa.Network], pd.Series] | None, optional
        Optional custom function ``grid_share_fn(n) -> pd.Series`` overriding
        ``calculate_grid_share``.

    Returns
    -------
    tuple[pypsa.Network, pd.DataFrame]
        Solved ``(n, convergence_df)`` with columns
        ``["iteration", "max_delta", "objective", "mean_r_in", "mean_r_out", "converged"]``.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}.")
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1]; got {damping}.")

    eff_exclude = set(exclude_assets) if exclude_assets is not None else {"DC"}
    eff_exclude.add("DC")

    grid_kwargs: dict[str, Any] = {
        "country": country,
        "clean_carriers": clean_carriers,
        "excluded_carriers": excluded_carriers,
        "exclude_assets": eff_exclude,
        "neighbours": neighbours,
    }

    r = _coerce_initial_r(n, r0, grid_kwargs)

    opts: dict[str, Any] = dict(DEFAULT_SOLVER_OPTIONS)
    if solver_options is not None:
        opts.update(solver_options)
    solver_name = str(opts.pop("solver_name", DEFAULT_SOLVER_NAME))

    records: list[dict[str, Any]] = []

    for iteration in range(max_iter):
        n.optimize.create_model()
        build_fn(n, r)

        status, condition = n.optimize.solve_model(
            solver_name=solver_name,
            **opts,
        )
        if status != "ok":
            raise RuntimeError(
                f"Optimization failed at iteration {iteration}: status={status}, condition={condition}"
            )

        if grid_share_fn is not None:
            r_new = grid_share_fn(n).reindex(n.snapshots).fillna(0.0).astype(float)
        else:
            r_new = calculate_grid_share(n, **grid_kwargs)

        r_new.index = n.snapshots
        r_new.index.name = n.snapshots.name

        max_delta = float((r_new - r).abs().max())
        objective = float(n.objective)
        converged = bool(max_delta < tol)

        records.append(
            {
                "iteration": iteration,
                "max_delta": max_delta,
                "objective": objective,
                "mean_r_in": float(r.mean()),
                "mean_r_out": float(r_new.mean()),
                "converged": converged,
            }
        )

        logger.info(
            f"Grid CFE iteration {iteration}: max_delta={max_delta:.6f}, "
            f"objective={objective:.4f}, mean_r={r_new.mean():.4f}, converged={converged}"
        )

        n._grid_share = r_new
        if converged:
            break

        r = float(damping) * r_new + (1.0 - float(damping)) * r
        r.index = n.snapshots
        r.index.name = n.snapshots.name

    convergence_df = pd.DataFrame(records)
    return n, convergence_df
