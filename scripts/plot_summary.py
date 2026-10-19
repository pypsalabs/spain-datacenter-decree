# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT
"""
Plot the study figures from the scenario summary table.

Reads ``results/scenarios.csv`` from ``make_summary.py`` and produces the study
figures (costs, CFE scores, portfolio composition, emissions and price impacts
across policies, targets and DC sizes).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point supporting both Snakemake execution and standalone CLI."""
    raise NotImplementedError("plot_summary: to be written with the reporting stage.")


if __name__ == "__main__":
    main()
