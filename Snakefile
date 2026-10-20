# SPDX-FileCopyrightText: 2026 Iegor Riepin
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


import re

HORIZON = config["planning_horizons"][0]


rule all:
    """Full case study: upstream build, scenario solves, summary, figures."""
    default_target: True
    input:
        "results/scenarios.csv",
        "results/plots/.done",


rule validation:
    """2024 back-cast and its validation reports (raw and patched variants)."""
    input:
        "results/validation/validation_raw.csv",
        "results/validation/validation_patched.csv",


rule build_upstream:
    """Builds the composed network and processed costs inside the pypsa-eur submodule."""
    output:
        network=f"pypsa-eur/resources/{{run}}/networks/composed_{HORIZON}.nc",
        costs=f"pypsa-eur/resources/{{run}}/costs_{HORIZON}_processed.csv",
    threads: workflow.cores
    params:
        horizon=HORIZON,
    shell:
        """
        cd pypsa-eur && snakemake \
            resources/{wildcards.run}/networks/composed_{params.horizon}.nc \
            resources/{wildcards.run}/costs_{params.horizon}_processed.csv \
            -j {threads} --configfile ../config/config.spain.yaml
        """


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
    policy=r"no_dc|unmatched_dc|annual_\d+_res|(?:strict_)?hourly_\d+_(?:ppa_only|grid_credit)(?:_cfe)?",
    target=r"\d+(?:\.\d+)?",
    dc_mw=r"\d+(?:\.\d+)?",
    siting=r"spain_wide|local_region",


rule prepare_network:
    """Applies the cited patches of a variant to the composed network and freezes the fleet.

    ``{year}`` is the data vintage (2024 = back-cast calibration, 2025 = case study);
    the planning horizon of the composed network is fixed. Weather-measurement patches
    (hydro inflow, wind-factor derivation) key on the run's weather year instead.
    """
    input:
        network=lambda w: (
            f"pypsa-eur/resources/{w.run}/networks/composed_{config['planning_horizons'][0]}.nc"
        ),
        costs=lambda w: (
            f"pypsa-eur/resources/{w.run}/costs_{config['planning_horizons'][0]}_processed.csv"
        ),
        boundary_flows="data/custom/fr_boundary_flows_2024.csv",
        custom_data=[
            "data/custom/national_capacities.csv",
            "data/custom/chp.csv",
            "data/custom/prices.csv",
            "data/custom/ntc.csv",
            "data/custom/annual_demand.csv",
            "data/custom/validation_2024_es.csv",
            "data/custom/pt_actuals.csv",
            "data/custom/wind_correction.csv",
            "data/custom/fr_nuclear_monthly_2024.csv",
        ],
    output:
        network="resources/{run}/networks/base_{year}_{variant}.nc",
    threads: 1
    resources:
        mem_mb=8000,
    params:
        patches=resolved_patches,
        weather_year=lambda w: w.run.removeprefix("wy"),
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.prepare_network \
            --network "{input.network}" \
            --costs "{input.costs}" \
            --output "{output.network}" \
            --patches "{params.patches}" \
            --boundary-flows "{input.boundary_flows}" \
            --data-year "{wildcards.year}" \
            --weather-year "{params.weather_year}"
        """


rule solve_network:
    """Attaches the data center, adds the 24/7 CFE policy constraint and solves with HiGHS."""
    input:
        network="resources/{run}/networks/base_{year}_{variant}.nc",
        costs=lambda w: (
            f"pypsa-eur/resources/{w.run}/costs_{config['planning_horizons'][0]}_processed.csv"
        ),
        r0_network=solve_r0_input,
    output:
        network="results/{run}/networks/solved_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}.nc",
        kpis="results/{run}/kpis/kpi_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}.csv",
        convergence="results/{run}/convergence/conv_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}.csv",
    threads: 2
    resources:
        mem_mb=8000,
    params:
        excess_share=lambda w: (
            "none"
            if config["spain_cfe"]["accounting"]["excess_cap"] is None
            else config["spain_cfe"]["accounting"]["excess_cap"]
        ),
        n_iterations=lambda w: config["spain_cfe"]["iteration"]["n_iterations"],
        solver=lambda w: config["spain_cfe"]["solving"]["solver"],
        meta=lambda w: (
            f"planning_year={w.year},weather_year={str(w.run).removeprefix('wy')},variant={w.variant},"
            f"sites={'+'.join(config['spain_cfe']['dc']['sites'])}"
        ),
        dc_nodes=lambda w: ",".join(
            config["spain_cfe"]["dc"]["nodes"][s]
            for s in config["spain_cfe"]["dc"]["sites"]
        ),
        r0_arg=lambda w, input: (
            f'--r0-network "{input.r0_network[0]}"' if input.r0_network else ""
        ),
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
            --dc-nodes "{params.dc_nodes}" \
            --siting "{wildcards.siting}" \
            --excess-share "{params.excess_share}" \
            --n-iterations "{params.n_iterations}" \
            --solver "{params.solver}" \
            --meta "{params.meta}" \
            {params.r0_arg}
        """


rule validate_network:
    """Compares the solved no-DC 2024 network with Redeia actuals in one report."""
    input:
        network="results/wy2024/networks/solved_2024_{variant}_no_dc_t0_0mw_spain_wide.nc",
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


def policy_target(policy):
    """Matching target embedded in the policy name (0 for untargeted policies)."""
    digits = re.search(r"(\d+)", policy)
    return int(digits.group(1)) if digits else 0


def suite_kpis(wildcards):
    """KPI files of the selected suite: its cross product plus a no_dc row per run."""
    year = config["planning_horizons"][0]
    suite = config["spain_cfe"]["suites"][config["spain_cfe"]["suite"]]

    def kpi(run, policy, target, dc_mw, siting):
        return (
            f"results/{run}/kpis/kpi_{year}_patched_"
            f"{policy}_t{target}_{dc_mw}mw_{siting}.csv"
        )

    files = [kpi(run, "no_dc", 0, 0, "spain_wide") for run in suite["runs"]]
    files += [
        kpi(run, policy, policy_target(policy), dc_mw, siting)
        for run in suite["runs"]
        for policy in suite["policies"]
        for dc_mw in suite["dc_mw"]
        for siting in suite["siting"]
    ]
    return files


rule make_summary:
    """Collects all scenario KPI rows into one tidy table with baseline deltas."""
    input:
        kpis=suite_kpis,
    output:
        summary="results/scenarios.csv",
    threads: 1
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.make_summary \
            --kpis {input.kpis} \
            --output "{output.summary}"
        """


rule plot_summary:
    """Produces the study figures from the scenario summary table."""
    input:
        summary="results/scenarios.csv",
    output:
        touch("results/plots/.done"),
    threads: 1
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.plot_summary \
            --summary "{input.summary}" \
            --plots-dir "results/plots"
        """
