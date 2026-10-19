# Validation Log — 2024 Back-cast (wy2024)

Status 2026-10-18. The solved `no_dc` network for weather year 2024 is compared against
Redeia actuals in one report (`results/validation/validation_{variant}.csv` + `.yaml`,
produced by `scripts/validate_network.py`). The **raw** variant is the fresh PyPSA-Eur
build with only the France boundary closure; the **patched** variant adds the cited
corrections below. Rule: *build fresh, validate, correct only what the report demands,
cite every correction* — no output is ever pinned to its historical value except where
explicitly stated ("by construction").

## 1. Raw-network findings (what a fresh PyPSA-Eur v2026.09.0 build gets wrong for Spain)

| Finding | Raw vs actual | Cause |
|---|---|---|
| Wind −21% | 46.8 vs 59.5 TWh, fleet CF 16.8% vs ~22% realized | (i) fleet profile = area-weighted average over eligible land, not where turbines stand; (ii) ERA5 underestimates Iberian ridge winds; (iii) GEM regional allocation off (10.2 GW in Aragón vs ~5.5 real) |
| Solar +47% | 41.3 GW in model vs 32.5 GW grid-connected | powerplantmatching inventory ≈ total PV incl. ~9 GW self-consumption (41.9 GW per Redeia), which does not belong over busbar demand |
| Hydro −30% | inflow 25.6 vs 34.9 TWh observed | upstream normalizes inflow to EIA annual statistics, whose vendored series ends before 2024 (a wet year) → average-year energy. Affects the validation year only; 2010/13/20 are covered by EIA |
| No CO2 price | CCGT marginal cost ≈ 81 EUR/MWh = gas + VOM only | composed network carries no ETS price at all (lignite at 28 EUR/MWh would run baseload) |
| CHP missing | 0 vs 16.4 TWh, ~6 Mt CO2 | electricity-only PyPSA-Eur has no cogeneration representation |
| CSP missing | 0 vs 4.1 TWh | carrier not built by default |
| Borders over-trade | France pushes 27.5 TWh into Spain | ES-FR corridor capability 5.2 GW (3 AC lines × 0.7 + 2×1 GW INELFE) vs 2.8 GW commercial NTC |
| PHS/hydro artifacts | PHS 7.4 GW @ 295 h; reservoir 1,402 h; RoR 0.5 vs 2.2 GW | JRC hydro database misclassification (river-cascade dams counted as storage) |
| Biomass/waste unconstrained | runs ~98% CF vs ~39% real | fuel-limited plants, no availability cap |
| Ghost lignite ES 0.51 GW + FR 0.98 GW; mainland oil | lignite runs 4.1 TWh baseload at 28 EUR/MWh (coal group 4.1 vs 3.0 TWh actual) | stale inventory (no operating lignite in Spain since 2022 or in France); zeroed by `national_capacities` |
| Demand −3.4% | ENTSO-E vs Redeia busbar definition | closure, not error |
| Portugal (found later) | hydro 10.3 vs 14.9 TWh; biomass ~100% CF; wind CF 14.1% vs 29.2% real; 4.2 of 5.6 GW in the worst CF bin | same EIA gap + same biomass issue + ERA5 bias compounded by bad GEM plant coordinates |

## 2. Config-level fix (before any patch): wind resource classes

`renewable.onwind.resource_classes: 5` splits each region into 5 CF bins; existing plants
are **spatially joined** to the bin where they stand, so the fleet profile reflects actual
siting. ES fleet CF moved 16.8% → 18.6% (the siting share of the bias, roughly a quarter);
FR lands at 24.1% (validates as is), PT stays at 14.1% (ERA5 + coordinates). This is the
principled part of the wind treatment; the factor below covers only the residual.

## 3. The patch set (`prepare_network.py`, all switchable, all inputs cited)

| Patch | What it does | Data |
|---|---|---|
| `boundary_flows` | FR external borders (GB, IT, CH, DE+BE) fixed to measured hourly 2024 flows; on in every variant (closure, not calibration) | RTE eCO2mix extract, vendored; −86.1 TWh net checks against RTE's ≈ −89 total minus ES |
| `national_capacities` | Fleet totals per carrier and scope (ES peninsular / Balearics, PT, FR) set pro rata to published actuals; creates missing units with technology-data efficiency (ES coal, Balearic OCGT); zeroes the phantom lignite units; for 2024, wind/solar use **average operating capacity** (mean of year-end 2023/2024 — ~6 GW of PV commissioned during 2024) | `national_capacities.csv` (Redeia ISE, RTE, REN/Ember) |
| `wind_correction` | ES and PT onwind profiles scaled so fleet availability matches cited annual generation; factors derived **at run time** (ES ×1.187, PT ×2.053), capped at 1.0; FR untouched. New-build PPA candidates inherit corrected profiles by construction | ES: Redeia; PT: `pt_actuals.csv` (Ember extract, vendored; capacity cross-checks Redeia/REN) |
| `phs_split` | Pure PHS energy 120 GWh (Huclin et al.), pumping 3,306 MW, round-trip 0.75; mixed PHS stays inside reservoir hydro per Redeia convention | `national_capacities.csv` |
| `chp_profile` | Cogeneration as a fixed profile: monthly shape derived at run time from the vendored Redeia monthly cogeneration CO2 series, scaled to published annual output (16.42 TWh, 0.366 tCO2/MWh_el), distributed by nodal load share. By construction | `chp.csv` + vendored REData JSON |
| `csp` | 2.3 GW CSP, nodal solar profile rescaled to the realized 2024 CSP capacity factor. By construction (no storage dispatch modelled; stated assumption) | `national_capacities.csv`, `validation_2024_es.csv` |
| `biomass_availability` | Biomass/waste `p_max_pu` capped at the realized capacity factor (ES 0.391, PT 0.563) — fuel-limited plants | cited generation ÷ capacity |
| `prices` | Thermal marginal costs = VOM + fuel/η + ETS × intensity/η (2024: TTF 34.45, ETS 65.24) | `prices.csv`, processed technology-data (VOM) |
| `ntc_caps` | ES-FR 2.8 GW, ES-PT 4.2 GW, Peninsula-Balearics 0.4 GW via per-element `s_max_pu`/`p_max_pu` scaling (impedances untouched; pro-rata caps are mildly conservative under KVL) | `ntc.csv` |
| `demand` | Country loads scaled to cited busbar demand (ES 239.7, PT 51.4, FR 449.2 TWh) | `annual_demand.csv` |
| `hydro_inflow` | Validation year only: ES and PT reservoir inflow scaled to observed 2024 hydro energy (ES ×1.143, PT ×2.177), closing the EIA data gap. FR needs none (model hits the actual 74.7 TWh) | `validation_2024_es.csv`, `pt_actuals.csv` |
| `ma_ad_exports` | Fixed flat ES exports to Morocco (2.54 TWh) and Andorra (0.24 TWh) at the nearest nodes | `validation_2024_es.csv` |

## 4. Current result: 7 of 8 gates pass

| Gate | Raw | Patched | Status |
|---|---|---|---|
| Wind (±15%) | −21.3% | **−2.5%** | PASS |
| Solar PV (±15%) | +46.6% | **+3.7%** | PASS |
| Nuclear (±10%) | +5.4% | −3.0% | PASS |
| Hydro (±20%) | −30.4% | −8.4% | PASS (by construction via inflow) |
| CCGT+CHP (±25%) | −54.3% | −5.7% | PASS |
| CO2 (±20%) | −54.4% | −17.1% | PASS |
| Load shedding (< 0.01%) | 0 | 0 | PASS |
| **Net exports (sign + ±50%)** | −194% | **+0.96 vs +10.2 TWh** | **FAIL** |

Informational: CSP, CHP, demand, MA/AD exact; biomass −6.5%; coal ≈ 0 and oil ≈ 0
(the agreed limitation: 2024 coal ran for non-market reasons — steel gases, grid security —
≈ 3.7 TWh and ≈ 2.9 Mt CO2, which is most of the CO2 gate's −17%); PHS cycling −27%/−39%
(LP arbitrage without reserves); average price +21% (€76 vs €63; cost-based LP without
bidding behaviour, and see §5).

## 5. The remaining gate, diagnosed

Net exports decompose into ES→FR −6.9 (actual −3.0) and ES→PT +5.1 (actual +10.5);
MA/AD exact. Three structural causes, none of them Spanish resource errors:

1. **France's flat nuclear availability.** Upstream applies the correct *annual* factor
   (0.715 for FR-2024) but flat over the year, while real availability is seasonal
   (~78% winter / ~60% summer maintenance). The model therefore has artificial French
   scarcity hours in winter (FR mean shadow price €237), distorting the hourly pattern
   of ES↔FR trade even though FR nuclear *energy* is right (369 vs 362 TWh).
2. **ES↔PT price-tie degeneracy.** Spanish and Portuguese gas plants carry identical
   marginal costs, so in ~55% of hours the LP is indifferent between Portugal burning
   its own gas and importing from Spain; the allocation is arbitrary and under-delivers
   ES→PT. Evidence: giving PT its correct (wet) 2024 hydro made the gate *worse* — the
   residual is not a resource error.
3. **Flat symmetric NTC** vs reality's hourly-varying capability (secondary).

Options on the table (user decision):
- (i) **FR seasonal nuclear availability patch**: a cited RTE monthly availability shape
  on FR nuclear `p_max_pu`, preserving the 0.715 annual mean. Small, cited, also removes
  the FR price artifact. Recommended.
- (ii) **Accept and document**: every gross quantity validates within ~10%; the failing
  number is the net of two large gross flows, ~4% of demand.
- (iii) **Hourly NTCs** from ENTSO-E (heavy data work).

## 6. Known caveats to carry into the paper

- Wind correction factors are **in-sample** (derived from 2024). Out-of-sample derivation
  (train on 2022/23 profiles, test on 2024) is queued; PT actuals for 2022/23 are already
  vendored, ES needs two more profile builds.
- Hydro inflow (validation year), CHP output and CSP output match actuals **by construction**;
  they are input choices, not validation successes, and are labelled as such above.
- Coal and oil dispatch are not reproduced (non-market operation); their ≈ 2.9 Mt CO2 are
  the dominant share of the CO2 gate's residual.
- The average price level is informational only; grid-purchase costs reported from the
  model inherit its +21% bias.
