# SPDX-FileCopyrightText: 2026 Iegor Riepin
#
# SPDX-License-Identifier: MIT

"""Snakemake workflow for the Spain 80% hourly matching study.

The upstream PyPSA-Eur network build runs inside the ``pypsa-eur/`` submodule with
``--configfile ../config/config.spain.yaml`` (see README); the rules here take over
from the composed networks: prepare (patch engine), solve (DC procurement
policies), validate, summarise, plot.
"""


configfile: "pypsa-eur/config/config.default.yaml"
configfile: "config/config.spain.yaml"


import re

from scripts.plot_summary import FIGURES
from scripts.plot_validation import CF_CARRIERS

HORIZON = config["planning_horizons"][0]
SUITE = config["spain_cfe"]["suites"][config["spain_cfe"]["suite"]]
# (policy, PPA siting) pairs of the selected suite
SCENARIOS = [(p, siting) for siting, ps in SUITE["policies"].items() for p in ps]
NO_SURPLUS_POLICIES = ("no_dc", "unmatched_dc")
PLOTS = list(FIGURES)  # file names of the summary figures


def ex_tag(sigma):
    """Sales-cap wildcard value: sigma as a whole percentage (0.20 -> '20')."""
    return f"{round(100 * sigma)}"


def suite_plots(wildcards):
    """Figure files of the selected suite: one folder per run, DC size and sales cap."""
    plots = PLOTS + [f"heatmap_{policy}_{siting}" for policy, siting in SCENARIOS]
    return [
        f"results/{run}/plots/{dc_mw}mw_ex{ex_tag(sigma)}/{plot}.pdf"
        for run in SUITE["runs"]
        for dc_mw in SUITE["dc_mw"]
        for sigma in SUITE["excess_cap"]
        for plot in plots
    ]


rule all:
    """Full case study: upstream build, scenario solves, summary, figures."""
    default_target: True
    input:
        "results/scenarios.csv",
        suite_plots,


rule validation:
    """2024 back-cast and its validation reports (raw and patched variants)."""
    input:
        "results/validation/validation_raw.csv",
        "results/validation/validation_patched.csv",
        "results/validation/validation_report.png",


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


rule upstream_map_inputs:
    """Upstream build files the validation maps read, built on the way to the composed network."""
    input:
        ancient(f"pypsa-eur/resources/{{run}}/networks/composed_{HORIZON}.nc"),
    output:
        base_network="pypsa-eur/resources/{run}/networks/base.nc",
        powerplants="pypsa-eur/resources/{run}/powerplants.csv",
        regions="pypsa-eur/resources/{run}/onshore_regions.geojson",
        class_regions=expand(
            "pypsa-eur/resources/{{run}}/regions_by_class_{carrier}.geojson",
            carrier=CF_CARRIERS,
        ),
    threads: 1
    params:
        targets=lambda w, output: [f.removeprefix("pypsa-eur/") for f in output],
    shell:
        """
        cd pypsa-eur && snakemake {params.targets} \
            -j {threads} --configfile ../config/config.spain.yaml
        """


def resolved_patches(wildcards):
    """Return the enabled patch names for a variant (config defaults + variant overrides)."""
    cfg = config["spain_cfe"]
    if wildcards.variant not in cfg["variants"]:
        raise ValueError(f"Unknown variant '{wildcards.variant}'.")
    patches = dict(cfg["patches"])
    patches.update(cfg["variants"][wildcards.variant] or {})
    return ",".join(name for name, on in patches.items() if on)


def solve_cfe0_input(wildcards):
    """Return the solved unmatched_dc network for *_grid_credit policies, else nothing."""
    if "grid_credit" in wildcards.policy:
        return [
            f"results/{wildcards.run}/networks/"
            f"solved_{wildcards.year}_{wildcards.variant}_unmatched_dc_t0_{wildcards.dc_mw}mw_{wildcards.siting}_ex0.nc"
        ]
    return []


wildcard_constraints:
    run=r"wy\d{4}",
    year=r"\d{4}",
    variant="|".join(config["spain_cfe"]["variants"]),
    policy=r"no_dc|unmatched_dc|annual_\d+_res|(?:strict_)?hourly_\d+_(?:ppa_only|grid_credit)(?:_cfe)?",
    target=r"\d+(?:\.\d+)?",
    dc_mw=r"\d+(?:\.\d+)?",
    siting=r"spain_wide|local_region",
    excess=r"\d+",
    ext=r"pdf|png",


rule prepare_network:
    """Applies the cited patches of a variant to the composed network and freezes the fleet.

    ``{year}`` is the data vintage (2024 = back-cast calibration, 2025 = case study);
    the planning horizon of the composed network is fixed. Weather-measurement patches
    (hydro inflow, wind-factor derivation) key on the run's weather year instead.
    """
    input:
        code=["scripts/prepare_network.py"],
        network=lambda w: f"pypsa-eur/resources/{w.run}/networks/composed_{HORIZON}.nc",
        costs=lambda w: f"pypsa-eur/resources/{w.run}/costs_{HORIZON}_processed.csv",
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
            "data/custom/sources/redata_2024_emisiones_co2_mensual.json",
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
    """Attaches the data center, adds the 24/7 CFE policy constraint and solves."""
    input:
        code=[
            "scripts/solve_network.py",
            "scripts/calculate_grid_cfe.py",
            "scripts/cfe_constraints.py",
            "scripts/data_center.py",
            "scripts/iterate.py",
        ],
        network="resources/{run}/networks/base_{year}_{variant}.nc",
        costs=lambda w: f"pypsa-eur/resources/{w.run}/costs_{HORIZON}_processed.csv",
        cfe0_network=solve_cfe0_input,
    output:
        network="results/{run}/networks/solved_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}_ex{excess}.nc",
        metrics="results/{run}/metrics/metrics_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}_ex{excess}.csv",
        convergence="results/{run}/convergence/conv_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}_ex{excess}.csv",
    threads: 2
    resources:
        mem_mb=8000,
    params:
        excess_share=lambda w: int(w.excess) / 100,
        n_iterations=lambda w: config["spain_cfe"]["iteration"]["n_iterations"],
        solver=lambda w: config["spain_cfe"]["solving"]["solver"],
        ppa_carriers=lambda w: ",".join(config["spain_cfe"]["ppa"]["carriers"]),
        discount_rate=lambda w: config["spain_cfe"]["dc"]["discount_rate"],
        meta=lambda w: (
            f"planning_year={w.year},weather_year={str(w.run).removeprefix('wy')},variant={w.variant},"
            f"sites={'+'.join(config['spain_cfe']['dc']['sites'])}"
        ),
        dc_nodes=lambda w: ",".join(
            config["spain_cfe"]["dc"]["nodes"][s]
            for s in config["spain_cfe"]["dc"]["sites"]
        ),
        cfe0_arg=lambda w, input: (
            f'--cfe0-network "{input.cfe0_network[0]}"' if input.cfe0_network else ""
        ),
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.solve_network \
            --network "{input.network}" \
            --costs "{input.costs}" \
            --output-network "{output.network}" \
            --output-metrics "{output.metrics}" \
            --output-convergence "{output.convergence}" \
            --policy "{wildcards.policy}" \
            --target "{wildcards.target}" \
            --dc-mw "{wildcards.dc_mw}" \
            --dc-nodes "{params.dc_nodes}" \
            --siting "{wildcards.siting}" \
            --excess-share "{params.excess_share}" \
            --n-iterations "{params.n_iterations}" \
            --solver "{params.solver}" \
            --ppa-carriers "{params.ppa_carriers}" \
            --discount-rate "{params.discount_rate}" \
            --meta "{params.meta}" \
            {params.cfe0_arg}
        """


rule validate_network:
    """Compares the solved no-DC 2024 network with Redeia actuals in one report."""
    input:
        code=["scripts/validate_network.py", "scripts/calculate_grid_cfe.py"],
        network="results/wy2024/networks/solved_2024_{variant}_no_dc_t0_0mw_spain_wide_ex0.nc",
        actuals="data/custom/validation_2024_es.csv",
        capacities="data/custom/national_capacities.csv",
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
            --capacities "{input.capacities}" \
            --output-report "{output.report}" \
            --output-gates "{output.gates}"
        """


rule plot_validation:
    """Plots the validation figure of the 2024 back-cast next to its reports."""
    input:
        code=[
            "scripts/plot_validation.py",
            "scripts/plot_summary.py",
            "scripts/prepare_network.py",
        ],
        base_network="pypsa-eur/resources/wy2024/networks/base.nc",
        powerplants="pypsa-eur/resources/wy2024/powerplants.csv",
        regions="pypsa-eur/resources/wy2024/onshore_regions.geojson",
        class_regions=expand(
            "pypsa-eur/resources/wy2024/regions_by_class_{carrier}.geojson",
            carrier=CF_CARRIERS,
        ),
        networks=expand(
            "resources/wy2024/networks/base_2024_{variant}.nc",
            variant=["raw", "patched"],
        ),
        reports=expand(
            "results/validation/validation_{variant}.csv", variant=["raw", "patched"]
        ),
        capacities="data/custom/national_capacities.csv",
        colors="pypsa-eur/config/plotting.default.yaml",
        style="scripts/plots.mplstyle",
    output:
        "results/validation/validation_report.png",
    threads: 1
    resources:
        mem_mb=4000,
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.plot_validation \
            --base-network "{input.base_network}" \
            --powerplants "{input.powerplants}" \
            --regions "{input.regions}" \
            --class-regions {input.class_regions} \
            --networks {input.networks} \
            --reports {input.reports} \
            --capacities "{input.capacities}" \
            --colors "{input.colors}" \
            --output "{output}"
        """


def policy_target(policy):
    """Matching target embedded in the policy name (0 for untargeted policies)."""
    digits = re.search(r"(\d+)", policy)
    return int(digits.group(1)) if digits else 0


rule summarise_network:
    """Postprocesses one solved scenario into national and data center metrics."""
    input:
        code=[
            "scripts/summarise_network.py",
            "scripts/calculate_grid_cfe.py",
            "scripts/cfe_constraints.py",
        ],
        network="results/{run}/networks/solved_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}_ex{excess}.nc",
        metrics="results/{run}/metrics/metrics_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}_ex{excess}.csv",
    output:
        summary="results/{run}/summaries/summary_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}_ex{excess}.csv",
        hourly="results/{run}/hourly/hourly_{year}_{variant}_{policy}_t{target}_{dc_mw}mw_{siting}_ex{excess}.csv",
    threads: 1
    resources:
        mem_mb=4000,
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.summarise_network \
            --network "{input.network}" \
            --metrics "{input.metrics}" \
            --output "{output.summary}" \
            --output-hourly "{output.hourly}"
        """


def scenario_stem(policy, dc_mw, siting, sigma):
    """Case-study scenario part of the result file names.

    The baselines have no surplus to sell, so they are solved once, under ex0.
    """
    if policy in NO_SURPLUS_POLICIES:
        sigma = 0.0
    return (
        f"{HORIZON}_patched_{policy}_t{policy_target(policy)}_{dc_mw}mw_{siting}"
        f"_ex{ex_tag(sigma)}"
    )


def suite_summaries(wildcards):
    """Summary files of the selected suite: its cross product plus a no_dc row per run."""

    def summary_file(run, policy, dc_mw, siting, sigma):
        stem = scenario_stem(policy, dc_mw, siting, sigma)
        return f"results/{run}/summaries/summary_{stem}.csv"

    def caps(policy):
        return [0.0] if policy in NO_SURPLUS_POLICIES else SUITE["excess_cap"]

    files = [summary_file(run, "no_dc", 0, "spain_wide", 0.0) for run in SUITE["runs"]]
    files += [
        summary_file(run, policy, dc_mw, siting, sigma)
        for run in SUITE["runs"]
        for policy, siting in SCENARIOS
        for dc_mw in SUITE["dc_mw"]
        for sigma in caps(policy)
    ]
    return files


rule make_summary:
    """Collects all scenario summary rows into one tidy table with baseline deltas."""
    input:
        code=["scripts/make_summary.py"],
        metrics=suite_summaries,
    output:
        summary="results/scenarios.csv",
    threads: 1
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.make_summary \
            --metrics {input.metrics} \
            --output "{output.summary}"
        """


rule plot_summary:
    """Produces the study figures of one run, DC size and sales cap from the summary table."""
    input:
        code=["scripts/plot_summary.py"],
        summary="results/scenarios.csv",
        colors="pypsa-eur/config/plotting.default.yaml",
        style="scripts/plots.mplstyle",
    output:
        expand(
            "results/{{run}}/plots/{{dc_mw}}mw_ex{{excess}}/{plot}.{{ext}}", plot=PLOTS
        ),
    threads: 1
    params:
        excess_cap=lambda w: int(w.excess) / 100,
        weather_year=lambda w: w.run.removeprefix("wy"),
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.plot_summary \
            --summary "{input.summary}" \
            --colors "{input.colors}" \
            --excess-cap {params.excess_cap} \
            --weather-year {params.weather_year} \
            --dc-mw {wildcards.dc_mw} \
            --plots-dir "results/{wildcards.run}/plots/{wildcards.dc_mw}mw_ex{wildcards.excess}" \
            --format {wildcards.ext}
        """


def heatmap_inputs(wildcards):
    """Summary row and hourly series of one scenario."""
    sigma = int(wildcards.excess) / 100
    stem = scenario_stem(wildcards.policy, wildcards.dc_mw, wildcards.siting, sigma)
    return {
        "summary": f"results/{wildcards.run}/summaries/summary_{stem}.csv",
        "hourly": f"results/{wildcards.run}/hourly/hourly_{stem}.csv",
    }


rule plot_heatmap:
    """Plots the hourly CFE score and CO2 intensity of one scenario's DC consumption."""
    input:
        unpack(heatmap_inputs),
        code=["scripts/plot_heatmap.py", "scripts/plot_summary.py"],
        style="scripts/plots.mplstyle",
    output:
        "results/{run}/plots/{dc_mw}mw_ex{excess}/heatmap_{policy}_{siting}.{ext}",
    threads: 1
    shell:
        """
        PYTHONPATH="{workflow.basedir}" python -m scripts.plot_heatmap \
            --hourly "{input.hourly}" \
            --summary "{input.summary}" \
            --output "{output}"
        """
