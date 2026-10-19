# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT
"""
Plot the study figures from the scenario summary table.

Reads ``results/scenarios.csv`` from ``make_summary.py`` and produces the study
figures (costs, CFE scores, portfolio composition, emissions and price impacts
across policies, targets and DC sizes).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; placeholder until the reporting stage defines the figures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "plot_summary is a placeholder: no figures produced yet from %s "
        "(the figure set is defined with the reporting stage).",
        args.summary,
    )


if __name__ == "__main__":
    main()
