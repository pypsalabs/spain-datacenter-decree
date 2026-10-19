# SPDX-FileCopyrightText: : 2026 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""Top-level Snakemake workflow for the Spain 80% hourly matching study.

The upstream PyPSA-Eur network build runs inside the ``pypsa-eur/`` submodule with
``--configfile ../config/config.spain.yaml`` (see README); the rules here take over
from the composed networks: prepare (patch engine), solve (DC + 24/7 CFE),
validate, summarize, plot. See README for the architecture.
"""

configfile: "pypsa-eur/config/config.default.yaml"
configfile: "config/config.spain.yaml"


include: "rules/study.smk"
