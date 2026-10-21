# Spain Hourly Matching

A study of Spain's proposed 80% hourly matching (24/7 CFE) requirement for new data
centers: a PyPSA model of today's Spanish electricity system (2025 fleet), built on
[PyPSA-Eur](https://github.com/PyPSA/pypsa-eur), with data center PPA & 24/7-CFE
procurement model following [PyPSA/247-cfe](https://github.com/PyPSA/247-cfe) and the
previous research of [Riepin & Brown (2024)](https://doi.org/10.1016/j.esr.2024.101488).

## Why this study?

The next decade of digital infrastructure will be built somewhere and powered by
something. Spain is deciding how. A
[draft royal decree](https://www.miteco.gob.es/es/energia/participacion/2026/detalle-participacion-publica-k-851.html)
would require new data centers (from 1 MW of grid access) to back, in every hour, at
least 80% of their consumption with renewable electricity generated that same hour by
new Spanish installations, with forfeiture of grid access on non-compliance. The stakes
are measured in gigawatts: data centers hold 6.3 GW of granted transmission access
([Red Eléctrica access map via pv magazine](https://www.pv-magazine.es/2026/10/02/red-electrica-asigna-633-gw-de-acceso-a-centros-de-datos-el-34-de-los-1852-gw-otorgados-a-la-demanda/), Oct 2026), and
more than 12 GW of access and connection rights granted since 2021 ([MITECO](https://www.miteco.gob.es/content/dam/miteco/es/prensa/20260827_NdP_El_Gobierno_plantea_impulsar_los_centros_de_datos_sostenibles.pdf),
Aug 2026).

The policy's logic: additional demand must bring additional clean supply, matched at the
hours it actually consumes. Done right, this makes new load a system asset. It finances
new wind, solar and storage, caps its own marginal emissions, and hedges the operator
against wholesale price volatility. Without it, every new flat gigawatt is served by the
gas margin: higher emissions, deeper gas import dependency, higher consumer prices, and
data centers exposed to every price swing. Bad for Spanish society, and bad for the data
center operators themselves.

This study provides a transparent, quantitative assessment of the policy's impact. The
repository contains a dispatch model of today's Spanish electricity system, validated
against actual data, in which a new data center procures renewable electricity under the
proposed 80% hourly matching rule and alternative procurement policies. It measures system
costs, emissions, prices and the procurement portfolio against a baseline without the data
center, so that both the Spanish government and the data center industry can see what the
policy does before it takes effect.

This work is done in the open and in the public interest. Every input number is cited and
linked to its primary source; every correction to the model is documented, switchable and
re-validated; every figure and table in this repository can be reproduced from scratch
with the included scripts. Anyone, from a ministry analyst to an independent research
group, can check the numbers, challenge them, and build on them.

## Results

A showcase snapshot of the study figures and the raw summary csv is in
[doc/showcase](doc/showcase). Running the [workflow](#workflow) writes the full set to
`results/`: solved networks, summary CSVs and plots for every scenario.

## Model

Every modelling choice favors verifiability:

- Base model: PyPSA-Eur v2026.09.0 as a pinned git submodule. No upstream file is
  modified; every study choice lives in the `config/` overlay and this repository.
- Geography: 15 nodes, with 9 in Spain (8 mainland + Balearics), 4 in France and
  2 in Portugal.
- France's outer borders (GB, IT, CH, DE+BE combined): fixed to measured hourly
  2024 flows (RTE eCO2mix via ODRE).
- Temporal resolution: full year at 3-hourly resolution (2920 snapshots,
  averaged from hourly inputs upstream; each
  snapshot carries a weight of 3 hours in dispatch, storage and cost accounting).
- Techno-economic assumptions:
  [technology-data v0.15.0](https://github.com/PyPSA/technology-data/tree/v0.15.0) in EUR2025.
- Renewable resources: five
  [resource classes](https://pypsa-eur.readthedocs.io/en/latest/configuration/#renewable)
  per modelled region for onshore wind and utility solar PV (split into fixed optimally tilted
  and single-axis tracking), each with its own profile and potential. Existing plants are
  spatially joined to the class where they stand. PPA candidates take the best class with
  remaining potential at each node; the optimizer chooses what to contract.
- Fleet: frozen background fleet, calibrated to cited national data
  (`data/custom/*.csv`; every number sourced in `doc/data_sources.md`). Only the data
  center's PPA and battery assets are extendable.
- Solver: HiGHS by default, so anyone can reproduce a study without a license; Gurobi via
  `spain_cfe.solving.solver`. One full-year dispatch solves in ~2 min (HiGHS) or ~25 s
  (Gurobi) at ~4 GB RAM; a data center policy scenario takes ~2–4 min with Gurobi.

## Scenario options

Each scenario suite in `config/config.spain.yaml` mixes and matches these ready-made
options:

- Procurement policy: no policy at all (`unmatched_dc`); the decree, 80% in
  every hour (`strict_hourly_80_ppa_only`); 80% hourly matching on average over the year
  (`hourly_80_ppa_only`), optionally with credit for the grid's renewable share
  (`hourly_80_grid_credit`); or annual volume matching (`annual_80_res`). Need a different
  score? Change the number in the name: `hourly_90_ppa_only` asks for 90%,
  `annual_100_res` for 100%.
- PPA location: the data center signs its PPAs anywhere in
  mainland Spain (`spain_wide`) or only in its own region (`local_region`).
- Data center cluster size: a flat load of any size. The full suite runs 500,
  1000, 2000, 4000 and 6000 MW.
- Data center site: any of the 9 Spanish regions, Aragón by
  default.
- Sales of surplus electricity: off by default, so the data center stores or
  curtails what it cannot use. Switch them on to let it sell surplus back to the grid.
- Weather year: this version of the project uses wy2024. The PyPSA-Eur and
  [atlite](https://github.com/PyPSA/atlite) workflow can build any other weather year from
  [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) reanalysis data,
  with SARAH-3 satellite data for solar.

## Study stages

The study keeps what we measure strictly separate from what we assume. The `{year}`
wildcard is the data vintage; the run is the weather year. Fleet, demand, price and NTC
patches read rows for `{year}`; hydro inflow keys on the run's weather year; the France
border flows are the measured 2024 series in every run; the wind-bias factor is derived on
the back-cast and applied to other data years as fixed cited constants.

| Stage | run (weather) | `{year}` (data) | variant | artifacts |
|---|---|---|---|---|
| (i) raw + validation | wy2024 | 2024 | raw | `base_2024_raw.nc` → `validation_raw` |
| (ii) back-cast calibration + validation | wy2024 | 2024 | patched | `base_2024_patched.nc` → `validation_patched` |
| (iii) case study | wy2024 | 2025 | patched | `base_2025_patched.nc` → DC procurement-policy solves → `scenarios.csv` |
| (iv) weather-year sensitivities (open for extension) | wy2010 / wy2013 / wy2020 | 2025 | patched | `base_2025_patched.nc` per run → DC procurement-policy solves → `scenarios.csv` |

On the wy2024 reference the case study uses every weather-dependent closure in-sample:
measured 2024 border flows with 2024 weather, observed 2024 hydro energy, and the
wind-bias factor derived on the back-cast (stage ii) and applied as a fixed cited
constant (`data/custom/wind_correction.csv`) in stage (iii).

## Setup

Install [pixi](https://pixi.sh), then:

```bash
git clone --recursive https://github.com/pypsalabs/spain-datacenter-decree.git
cd spain-datacenter-decree
```

The study deliberately has no environment of its own: it runs entirely in the submodule's
pixi environment (`pypsa-eur/pixi.toml` + lock file), so the upstream build and the study
workflow always use exactly the package versions the pinned PyPSA-Eur release was tested
with.

## Workflow

Two commands run everything (incl. installation of the pinned environment).

```bash
# full case study: upstream build, scenario solves, summary, figures
pixi run --manifest-path pypsa-eur/pixi.toml snakemake -c all

# 2024 back-cast and its validation reports (not needed for the case study)
pixi run --manifest-path pypsa-eur/pixi.toml snakemake validation -c all
```

Every intermediate target (prepared network, single scenario solve, report) stays
addressable by its filename; see `Snakefile`. For interactive work, activate the
environment with `pixi shell --manifest-path pypsa-eur/pixi.toml`.

## Validation

`prepare_network.py` corrects the PyPSA-Eur network with cited data, each patch switchable
under `spain_cfe.patches`: build the system, validate it against 2024 actuals, correct
only what the validation demands, and re-validate through the same pipeline.

Raw PyPSA-Eur ([validation_raw.csv](results/validation/validation_raw.csv)) and the model
patched with cited data corrections
([validation_patched.csv](results/validation/validation_patched.csv)) are compared against
Redeia's reported 2024 numbers. Both reports are tracked and regenerated with `snakemake
validation` whenever the model changes.

## License

MIT © 2026 Iegor Riepin (PyPSA Labs gGmbH); see [LICENSES/](LICENSES/),
[REUSE](https://reuse.software/)-compliant. Vendored
third-party primary sources under `data/custom/sources/` keep their providers' licences,
registered per file in [doc/data_sources.md](doc/data_sources.md).

## Independence

This project is independent work: Iegor Riepin and PyPSA Labs gGmbH received no funding,
no grant and no direction for it, from any government, company or interest group. The
modelling work was done because Spain is about to take a decision of lasting consequence,
and a decision of that weight deserves open, verifiable evidence that belongs to everyone.
