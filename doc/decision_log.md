# Decision Log

Scope and architecture decisions for the study, agreed 2026-10-18.
Validation findings, patches and their status live in [validation_log.md](validation_log.md).

## A. Scope

| Topic | Decision |
|---|---|
| Study question | Impact of adding data-center load to **today's Spanish system** under the proposed 80% hourly matching (24/7 CFE) policy. No capacity-expansion year: the background fleet is frozen (no co-optimization world — the system has not yet adapted to the new load) |
| Fleet year | 2025/26 actuals (Redeia; mid-2026 values where already published). No 2030 horizon: a future-year expansion case can be added later, but it answers a different question and is out of scope here |
| Geography | 15 nodes: ES 9 (8 mainland + Balearics) / FR 4 / PT 2. France's non-Iberian borders closed with measured flows (see B) |
| Weather years | 2013 (reference), 2010, 2020 — pre-built cutouts; validation back-cast on 2024 (the only year with both a cutout and full actuals) |
| Temporal resolution | 3-hourly **averaged** snapshots (`clustering.temporal.averaging: "3h"`, 2,920 steps), applied natively by PyPSA-Eur upstream of `composed_*.nc` — one network per run, so coarser resolutions are never built separately |
| Policies | `no_dc`, `unmatched_dc`, `annual_100_res`, `hourly_80_ppa_only`, `hourly_80_grid_credit` (iterative grid share), `strict_hourly_80_*`; the `_cfe` suffix counts nuclear as clean (sensitivity) |
| Data center | Flat load, 60% Madrid (`ES0 6AC`) / 40% Aragón (`ES0 7AC`); PPA candidates (onwind, solar, solar-hsat) at all 9 ES nodes with a host-node-only sensitivity; standalone battery; 7% WACC |

## B. Architecture

| Topic | Decision |
|---|---|
| Base model | **PyPSA-Eur v2026.09.0 as a pinned git submodule** (commit c8209014), zero upstream files modified. Overlay pattern, not a soft fork: anyone can diff this repo against vanilla PyPSA-Eur and see that every study choice lives in the overlay. The pin is recorded as the gitlink in the commit tree (`.gitmodules` carries only path + URL) |
| Workflow | Two steps by design. (1) The upstream network build runs *inside* the submodule with `--configfile ../config/config.spain.yaml`. (2) Our own DAG (`rules/study.smk`) takes over from `composed_*.nc`: `prepare_network` (patch engine) → `solve_network` (DC + 24/7 CFE constraints + grid-share iteration) → `validate_network` / `make_summary` / `plot_summary` |
| Validation design | Dispatch validation **reuses the study pipeline**: it is the solved `no_dc` scenario on the wy2024 network compared against Redeia actuals in one report (capacities, generation, FLH, CO2 intensity, flows, prices). No separate validation codebase, so the validated path is exactly the path the scenarios run on |
| Background corrections | `prepare_network.py` applies an ordered list of cited corrections ("patches"), each switchable in `spain_cfe.patches`; named `variants` (`raw`, `patched`) select sets via the `{variant}` wildcard. Philosophy: **build fresh, validate, correct only what the validation report demands, cite every correction.** The raw variant keeps only the France boundary closure, so the raw and patched validation reports together attribute the effect of every patch |
| France boundary | **No pan-European model.** France's external borders (GB, IT, CH, DE+BE combined) are fixed to measured hourly 2024 flows (RTE eCO2mix via ODRE, vendored + sha256-registered), attached as fixed generators/loads at the nearest FR node. MWh only, no clean shares: the Spanish CFE tracing covers Spain plus direct neighbours, and a neighbour's clean share is its domestic generation mix — tracing a second hop beyond France is third-order behind a 2.8 GW border into a ~95%-clean system. For the validation year, measured flows are strictly more accurate than any coarse Europe-wide model; they also avoid maintaining a second model whose assumptions must be kept permanently consistent with the main one. Stated limitations: FR external trade does not respond to the weather year or to the data center |
| Interconnectors | Linearized power flow (KVL) with physical impedances everywhere; thermal × `s_max_pu` line limits. Cross-border corridors capped to published commercial NTC only where validation demanded it (see validation log): per-element `s_max_pu`/`p_max_pu` scaling in a patch, with `s_nom` and impedances untouched so the flow physics is unaffected |
| Technology costs | technology-data v0.15.0 (EUR2025), native 2025 values including lifetimes (onwind 28.5 a, solar 37.5 a, battery store 22.5 a / inverter 10 a). A single cost year needs no cross-year harmonization |
| Wind representation | `renewable.onwind.resource_classes: 5`: each region split into 5 capacity-factor bins with own profiles and potentials, and existing plants spatially joined to the bin where they stand. This fixes the layout bias (a single regional profile averages over all eligible land, while real turbines occupy the windy sites) at the config level, before any correction factor. The residual ERA5 bias is handled in validation (see validation log) |
| Grid-share iteration | Fixed point on the hourly grid clean share with damping 0.5 and HiGHS simplex basis warm-start. Damping guards against oscillation from degenerate dispatch ties (e.g. equal-cost hydro); toy traces show the solution itself is stable from iteration 2, so a mean-Δr/objective stopping criterion is a candidate refinement |
| Solver | HiGHS dual simplex; no commercial solver. A 15-node 3H full-year dispatch solves in ~2 min |
| Snakemake style | All rules invoke scripts via `shell:` + `PYTHONPATH` and a CLI (the `script:` directive breaks on `from __future__` imports because its wrapper prepends code). Every `data/custom` CSV a script reads is declared as a rule input so data edits retrigger the DAG |
| Data policy | Every number a script uses comes from `data/custom/*.csv` with a `source_id` registered in `doc/data_sources.md` (URL, retrieval date, licence, sha256 of the vendored primary file). One-off retrievals (eCO2mix flows, Ember extract) are vendored together with a rerunnable converter script, and deliberately not wired into Snakemake |

## C. Open items

| # | Item | Status |
|---|---|---|
| C1 | Final validation gate (net exports): FR seasonal nuclear availability patch vs document-and-accept vs hourly NTCs — see validation log §5 | open |
| C2 | Out-of-sample derivation of the wind correction factors (train on 2022/23 profile builds, test on 2024; PT actuals already vendored) to replace the in-sample 2024 factors for publication | open |
| C3 | Re-check `national_capacities.csv` 2025 rows against latest Redeia published capacities (mid-2026 where available) before the scenario runs | open |
| C4 | `attach_data_center`: PPA candidates must become resource-class-aware (best bin with remaining potential per node) before scenario runs | open |
| C5 | Suite target lists, `make_summary` collector, `plot_summary` figures | open |
