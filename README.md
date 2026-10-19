# Spain Hourly Matching

PyPSA study of Spain's proposed 80% hourly matching (24/7 CFE) requirement for new data
centers: a 15-node ES/FR/PT electricity model built on PyPSA-Eur, with 24/7 CFE procurement
accounting following [PyPSA/247-cfe](https://github.com/PyPSA/247-cfe).

**Scope:** today's Spanish system (2025/26 fleet), dispatch under three weather years
(2013 reference, 2010, 2020), validated against Redeia 2024 actuals. No capacity-expansion
year in the current scope.

## Model

- **PyPSA-Eur v2026.09.0** as a git submodule; zero upstream files modified, everything is
  overlaid via `config/` and the rules in this repository.
- 15 nodes: ES 9 (8 mainland + Balearics) / FR 4 / PT 2. France's external borders (BE, DE,
  CH, IT, GB, LU) are fixed to measured hourly flows (RTE/ENTSO-E, 2024); no pan-European
  model. A neighbour's clean share in the import tracing is its domestic generation mix.
- Electricity-only, 3-hourly averaged snapshots, technology-data v0.15.0 (EUR2025).
- Background fleet frozen and calibrated to cited national data (`data/custom/*.csv`, every
  number sourced in `doc/data_sources.md`); only the data center's PPA assets are extendable.
- Solver: HiGHS dual simplex. No commercial solver required.

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

`prepare_network.py` is a patch engine: it applies an ordered, config-controlled list of
cited corrections, each individually switchable. The philosophy is **build fresh, validate,
correct only what fails, cite every correction**: the raw variant (only the France boundary
closure on) is validated first, patches are turned on one by one where the report demands
it, and each patched variant is re-validated through the same pipeline.

The patch set (fleet capacities, wind correction, PHS split, CHP, CSP, biomass
availability, prices, NTC caps, demand, hydro inflow, MA/AD exports) is defined in
`config/config.spain.yaml` under `spain_cfe.patches`; what each patch does, why it
exists and what it changed in the 2024 back-cast is documented in
[doc/validation_log.md](doc/validation_log.md).

## Layout

| Path | Content |
|---|---|
| `Snakefile` | Study workflow: prepare (patch engine), solve (DC + 24/7 CFE), validate, summarize, plot |
| `config/` | PyPSA-Eur overlay (`config.spain.yaml`) and weather-year scenarios |
| `scripts/` | Study code (calibration, audit, CFE constraints, grid-share tracing, iteration, validation) |
| `data/custom/` | Cited calibration inputs and vendored primary sources |
| `test/` | pytest suite |
| `doc/` | Data sources, decision log, validation log |
| `networks/`, `resources/`, `results/` | Build artifacts (gitignored) |

## Setup

Install [pixi](https://pixi.sh), then:

```bash
git clone --recursive <this-repo>
cd spain-hourly-matching/pypsa-eur && pixi install && cd ..
```

The study deliberately has no environment of its own: it runs entirely in the submodule's
pixi environment (`pypsa-eur/pixi.toml` + lock file), so the upstream build and the study
workflow always use exactly the package versions the pinned PyPSA-Eur release was tested
with. The path must contain no spaces. All commands below use
`PX="pixi run --manifest-path pypsa-eur/pixi.toml"`.

## Workflow

1. **Upstream network build** (inside the submodule; validation year first):
   ```bash
   cd pypsa-eur
   pixi run snakemake resources/wy2024/networks/composed_2025.nc -j 12 \
     --configfile ../config/config.spain.yaml
   cd ..
   # then wy2013 (reference); wy2010, wy2020 after validation passes
   ```
2. **Prepare, solve, validate** from the repository root. France's measured border flows
   ship with the repo (`data/custom/fr_boundary_flows_2024.csv`, regenerable with
   `scripts/retrieve_fr_boundary_flows.py --download`):
   ```bash
   $PX snakemake resources/wy2024/networks/base_2025_raw.nc -j 4   # patch engine
   $PX snakemake results/validation/validation_raw.csv -j 4       # no_dc solve + report
   ```
3. **Scenarios, summary, figures**: `solve_network` targets per
   (run, variant, policy, target, DC size, siting), collected by `make_summary` and
   plotted by `plot_summary` (see `Snakefile`).

## Tests

```bash
$PX pytest test -q
```

## Documentation

Scope and architecture decisions are recorded in [doc/decision_log.md](doc/decision_log.md);
the 2024 validation story (raw findings, every applied correction with its cited source,
gate results and open items) in [doc/validation_log.md](doc/validation_log.md); every input
number's provenance in [doc/data_sources.md](doc/data_sources.md).

## License

MIT, © 2026 Iegor Riepin (TU Berlin, PyPSA Labs gGmbH); see [LICENSES/](LICENSES/),
[REUSE](https://reuse.software/)-compliant. Vendored
third-party primary sources under `data/custom/sources/` keep their providers' licences,
registered per file in [doc/data_sources.md](doc/data_sources.md).
