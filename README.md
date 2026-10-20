# Spain Hourly Matching

A study of Spain's proposed 80% hourly matching (24/7 CFE) requirement for new data
centers: a PyPSA model of today's Spanish electricity system (2025/26 fleet),
built on [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur), with datacenter PPA & 24/7-CFE procurement
model following [PyPSA/247-cfe](https://github.com/PyPSA/247-cfe).

## Why this study?

The next decade of digital infrastructure will be built somewhere and powered by
something. Spain is deciding how. A 
[draft royal decree](https://www.miteco.gob.es/es/energia/participacion/2026/detalle-participacion-publica-k-851.html)
would require new data centers (from 1 MW of grid access) to back, in every hour, at
least 80% of their consumption with renewable electricity generated that same hour by
new Spanish installations, with forfeiture of grid access on non-compliance. The stakes
are measured in gigawatts: data centers already hold
~6.3 GW of granted transmission access, about a third of all granted demand access, with
a further ~16 GW of requests pending.

The policy's logic: additional demand must bring additional clean supply, matched at the
hours it actually consumes. Done right, this makes new load a system asset. It finances
new wind, solar and storage, caps its own marginal emissions, and hedges the operator
against wholesale price volatility. Without it, every new flat gigawatt is served by the
gas margin: higher emissions, deeper gas import dependency, higher consumer prices, and
data centers exposed to every price swing. Bad for Spanish society, and bad for the data
center operators themselves.

This study provides a transparent, quantitative assessment of the policy's impact. The
repository contains a dispatch model of today's Spanish electricity system, validated
against actual data, in which a new data center procures 24/7 carbon-free energy under
the proposed 80% hourly matching rule. It measures system costs, emissions, prices and
the procurement portfolio against an unmatched baseline, so that both the Spanish
government and the data center industry can see what the policy does before it takes
effect.

We do this work in the open, for the public record. Every input number is cited and its
primary source vendored in this repository; every correction to the model is documented,
switchable and re-validated; every figure and table in the report can be reproduced from
scratch with the included scripts. Anyone, from a ministry analyst to an independent research
group, can check the numbers, challenge them, and build on them.

## Model

Every modelling choice favors verifiability: anyone can diff this repository against
vanilla PyPSA-Eur and see exactly what the study adds.

- Base model: PyPSA-Eur v2026.09.0 as a pinned git submodule. No upstream file is
  modified; every study choice lives in the `config/` overlay and this repository.
- Geography: 15 nodes, with 9 in Spain (8 mainland + Balearics), 4 in France and
  2 in Portugal.
- France's outer borders (BE, DE, CH, IT, GB, LU): fixed to measured hourly 2024
  flows (RTE/ENTSO-E).
- Temporal resolution: full year at 3-hourly resolution (2920 snapshots,
  averaged from hourly inputs upstream; each
  snapshot carries a weight of 3 hours in dispatch, storage and cost accounting).
- Techno-economic assumptions:
  [technology-data v0.15.0](https://github.com/PyPSA/technology-data/tree/v0.15.0) in EUR2025.
- Renewable resources: five
  [resource classes](https://pypsa-eur.readthedocs.io/en/latest/configuration/#renewable)
  per modelled region for onshore wind and utility solar PV (split into fixed optimally
  tilted and single-axis tracking), each with its own profile and potential. Existing
  plants are spatially joined to the class where they stand. The data center procurement model contracts renewable assets in merit order of their
  generation profiles, subject to remaining land availability, with an optional switch between two
  procurement scopes: `local_region` (contract only in the data center's own region) and
  `spain_wide` (contract anywhere in Spain, letting the optimizer choose the locations).
- Fleet: frozen background fleet, calibrated to cited national data
  (`data/custom/*.csv`; every number sourced in `doc/data_sources.md`). Only the data
  center's PPA assets are extendable.
- Solver: HiGHS by default, so anyone can reproduce a study without a license; Gurobi via `spain_cfe.solving.solver`. One full-year dispatch solves in ~2 min (HiGHS) or ~25 s (Gurobi) at ~4 GB RAM.

## Workflow architecture

```
composed_{year}.nc  (pypsa-eur submodule, config-driven: clustering, profiles, 3h averaging)
   └► prepare_network.py     + data/custom + config.patches        →  base.nc (per run, per variant)
        └► solve_network.py  attach DC, 24/7 CFE constraints,
                             grid-share iteration (grid-credit)    →  solved.nc, kpis, convergence
             ├► validate_network.py  solved no_dc wy2024 vs Redeia →  one validation report
             ├► make_summary.py      all KPIs                      →  scenarios.csv (+ baseline deltas)
             └► plot_summary.py                                    →  figures
```

`prepare_network.py` is a patch engine: it applies a config-controlled list of
cited corrections, each individually switchable. The philosophy is (i) build electricity system with pypsa-eur workflow, (ii) validate it against actual data (2024 back-cast), (iii) apply necessary data improvements for this study, and (iv) re-validate.

The patch set (fleet capacities, wind correction, PHS split, CHP, CSP, biomass
availability, NTC caps, demand, hydro inflow, MA/AD exports) is defined in
`config/config.spain.yaml` under `spain_cfe.patches`; what each patch does, why it
exists and what it changed in the 2024 back-cast is documented in
[doc/validation_log.md](doc/validation_log.md).

## Study stages

The study keeps what we measure strictly separate from what we assume. The `{year}`
wildcard is the data vintage; the run is the weather year. Fleet, demand, price and NTC
patches read rows for `{year}`, while weather-measurement closures (France border flows,
hydro inflow, the wind-bias derivation) key on the run's weather year.

| Stage | run (weather) | `{year}` (data) | variant | artifacts |
|---|---|---|---|---|
| (i) raw + validation | wy2024 | 2024 | raw | `base_2024_raw.nc` → `validation_raw` |
| (ii) back-cast calibration + validation | wy2024 | 2024 | patched | `base_2024_patched.nc` → `validation_patched` |
| (iii) case study | wy2024 | 2025 | patched | `base_2025_patched.nc` → DC + 24/7 CFE solves → `scenarios.csv` |
| (iv) weather-year sensitivities | wy2010 / wy2013 / wy2020 | 2025 | patched | planned; the 2024-measured closures and the wind factor become out-of-sample transfers (stated caveat) |

On the wy2024 reference the case study uses every weather-dependent closure in-sample:
measured 2024 border flows with 2024 weather, observed 2024 hydro energy, and the
wind-bias factor derived on the back-cast (stage ii) and applied as a fixed cited
constant (`data/custom/wind_correction.csv`) in stage (iii).

## Setup

Install [pixi](https://pixi.sh), then:

```bash
git clone --recursive <this-repo>
cd spain-hourly-matching
```

The study deliberately has no environment of its own: it runs entirely in the submodule's
pixi environment (`pypsa-eur/pixi.toml` + lock file), so the upstream build and the study
workflow always use exactly the package versions the pinned PyPSA-Eur release was tested
with. The path must contain no spaces.

## Workflow

Two commands run everything. The first invocation also installs the pinned environment.

```bash
# full case study: upstream build, scenario solves, summary, figures
pixi run --manifest-path pypsa-eur/pixi.toml snakemake -c all

# 2024 back-cast and its validation reports (not needed for the case study)
pixi run --manifest-path pypsa-eur/pixi.toml snakemake validation -c all
```

France's measured border flows ship with the repository
(`data/custom/fr_boundary_flows_2024.csv`, regenerable with
`scripts/retrieve_fr_boundary_flows.py --download`). Every intermediate target (prepared
network, single scenario solve, report) stays addressable by its filename; see
`Snakefile`. For interactive work, activate the environment once with
`pixi shell --manifest-path pypsa-eur/pixi.toml`.

## Validation preview

*placeholder*

## License

MIT © 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH); see [LICENSES/](LICENSES/),
[REUSE](https://reuse.software/)-compliant. Vendored
third-party primary sources under `data/custom/sources/` keep their providers' licences,
registered per file in [doc/data_sources.md](doc/data_sources.md).
