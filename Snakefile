# SPDX-FileCopyrightText: 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH)
#
# SPDX-License-Identifier: MIT

"""Snakemake workflow for the Spain 80% hourly matching study.

The upstream PyPSA-Eur network build runs inside the ``pypsa-eur/`` submodule with
``--configfile ../config/config.spain.yaml`` (see README); the rules here take over
from the composed networks: prepare (patch engine), solve (DC + 24/7 CFE),
validate, summarize, plot. See README for the architecture.
"""

configfile: "pypsa-eur/config/config.default.yaml"
configfile: "config/config.spain.yaml"


def resolved_patches(wildcards):
    """Return the enabled patch names for a variant (config defaults + variant overrides)."""
    cfg = config.get("spain_cfe", {})
    patches = dict(cfg.get("patches", {}))
    patches.update(cfg.get("variants", {}).get(wildcards.variant) or {})
    return ",".join(name for name, on in patches.items() if on)


def solve_r0_input(wildcards):
    """Return the solved unmatched_dc network for *_grid_credit policies, else nothing."""
    if "grid_credit" in wildcards.policy:
        return [
            f"results/{wildcards.run}/networks/"
            f"solved_{wildcards.year}_{wildcards.variant}_unmatched_dc_t0_{wildcards.dc_mw}mw_{wildcards.siting}.nc"
        ]
    return []


wildcard_constraints:
    run=r"wy\d{4}",
    year=r"\d{4}",
    variant=r"[a-z0-9_]+",
    policy=r"no_dc|unmatched_dc|annual_100_res|(?:strict_)?hourly_\d+_(?:ppa_only|grid_credit)(?:_cfe)?",
    target=r"\d+(?:\.\d+)?",
    dc_mw=r"\d+(?:\.\d+)?",
    siting=r"spain_wide|host_node",


rule prepare_network:
    """Applies the cited patches of a variant to the composed network and freezes the fleet."""
    input:
        network="pypsa-eur/resources/{run}/networks/composed_{year}.nc",
        costs="pypsa-eur/resources/{run}/costs_{year}_processed.csv",
        boundary_flows="data/custom/fr_boundary_flows_2024.csv",
        custom_data=[
            "data/custom/national_capacities.csv",
            "data/custom/chp.csv",
            "data/custom/prices.csv",
            "data/custom/ntc.csv",
            "data/custom/annual_demand.csv",
            "data/custom/validation_2024_es.csv",
            "data/custom/pt_actuals.csv",
        ],
    output:
        network="resources/{run}/networks/base_{year}_{variant}.nc",
    params:
        patches=resolved_patches,
        data_year=lambda w: "2024" if w.run == "wy2024" else w.year,
    threads: 1
    resources:
        mem_mb=8000,
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.prepare_network \
            --network "{input.network}" \
            --costs "{input.costs}" \
            --output "{output.network}" \
            --patches "{params.patches}" \
            --boundary-flows "{input.boundary_flows}" \
            --data-year "{params.data_year}"
        """


rule solve_network:
    """Attaches the data center, adds the 24/7 CFE policy constraint and solves with HiGHS."""
    input:
        network="resources/{run}/networks/base_{year}_{variant}.nc",
        costs="pypsa-eur/resources/{run}/costs_{year}_processed.csv",
        r0_network=solve_r0_input,
    output:
        network="results/{run}/networks/solved_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}.nc",
        kpis="results/{run}/kpis/kpi_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}.csv",
        convergence="results/{run}/convergence/conv_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}.csv",
    params:
        excess_share=lambda w: config["spain_cfe"]["accounting"]["excess_cap"],
        max_iter=lambda w: config["spain_cfe"]["iteration"]["max_iter"],
        tol=lambda w: config["spain_cfe"]["iteration"]["tol"],
        damping=lambda w: config["spain_cfe"]["iteration"]["damping"],
        meta=lambda w: (
            f"planning_year={w.year},weather_year={str(w.run).removeprefix('wy')},variant={w.variant}"
        ),
        r0_arg=lambda w, input: (
            f'--r0-network "{input.r0_network[0]}"' if input.r0_network else ""
        ),
    threads: 2
    resources:
        mem_mb=8000,
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.solve_network \
            --network "{input.network}" \
            --costs "{input.costs}" \
            --output-network "{output.network}" \
            --output-kpis "{output.kpis}" \
            --output-convergence "{output.convergence}" \
            --policy "{wildcards.policy}" \
            --target "{wildcards.target}" \
            --dc-mw "{wildcards.dc_mw}" \
            --siting "{wildcards.siting}" \
            --excess-share "{params.excess_share}" \
            --max-iter "{params.max_iter}" \
            --tol "{params.tol}" \
            --damping "{params.damping}" \
            --meta "{params.meta}" \
            {params.r0_arg}
        """


rule validate_network:
    """Compares the solved no-DC 2024 network with Redeia actuals in one report."""
    input:
        network="results/wy2024/networks/solved_2025_{variant}_no_dc_t0_0mw_spain_wide.nc",
        actuals="data/custom/validation_2024_es.csv",
    output:
        report="results/validation/validation_{variant}.csv",
        gates="results/validation/validation_{variant}.yaml",
    threads: 1
    resources:
        mem_mb=4000,
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.validate_network \
            --network "{input.network}" \
            --actuals "{input.actuals}" \
            --output-report "{output.report}" \
            --output-gates "{output.gates}"
        """


rule make_summary:
    """Collects all scenario KPI rows into one tidy table with baseline deltas."""
    input:
        kpis=lambda w: [],  # filled by the scenario definition once suites are set
    output:
        summary="results/scenarios.csv",
    threads: 1
    script:
        "scripts/make_summary.py"


rule plot_summary:
    """Produces the study figures from the scenario summary table."""
    input:
        summary="results/scenarios.csv",
    output:
        touch("results/plots/.done"),
    threads: 1
    script:
        "scripts/plot_summary.py"
