# Validation Log — 2024 Back-cast (wy2024)

Status 2026-10-19. The solved `no_dc` network for weather year 2024 is compared against
Redeia actuals in one report (`results/validation/validation_{variant}.csv` + `.yaml`,
produced by `scripts/validate_network.py`). The **raw** variant is the fresh PyPSA-Eur
build with only the France boundary closure; the **patched** variant adds the cited
corrections below. Rule: *build fresh, validate, correct only what the report demands,
cite every correction* — no output is ever pinned to its historical value except where
explicitly stated ("by construction").

## 1. Raw-network findings (what a fresh PyPSA-Eur v2026.09.0 build gets wrong for Spain)

| Finding | Raw vs actual | Cause |
|---|---|---|
| Wind −21% (pre-resource-classes; −13% after, §2) | 46.8 vs 59.5 TWh, fleet CF 16.8% vs ~22% realized | (i) fleet profile = area-weighted average over eligible land, not where turbines stand; (ii) ERA5 underestimates Iberian ridge winds; (iii) GEM regional allocation off (10.2 GW in Aragón vs ~5.5 real) |
| Solar +46% | 41.3 GW in model vs 32.5 GW grid-connected | powerplantmatching inventory ≈ total PV incl. ~9 GW self-consumption (41.9 GW per Redeia), which does not belong over busbar demand |
| Hydro −30% | inflow 25.6 vs 34.9 TWh observed | upstream normalizes inflow to EIA annual statistics, whose vendored series ends before 2024 (a wet year) → average-year energy. Affects the validation year only; 2010/13/20 are covered by EIA |
| No CO2 price | CCGT marginal cost ≈ 81 EUR/MWh = gas + VOM only | composed network carries no ETS price at all (lignite at 28 EUR/MWh would run baseload) |
| CHP missing | 0 vs 16.4 TWh, ~6 Mt CO2 | electricity-only PyPSA-Eur has no cogeneration representation |
| CSP missing | 0 vs 4.1 TWh | carrier not built by default |
| Borders over-trade | France pushes 28.0 TWh into Spain | ES-FR corridor capability 5.2 GW (3 AC lines × 0.7 + 2×1 GW INELFE) vs 2.8 GW commercial NTC |
| PHS/hydro artifacts | PHS 7.4 GW @ 295 h; reservoir 1,402 h; RoR 0.5 vs 2.2 GW | JRC hydro database misclassification (river-cascade dams counted as storage) |
| Biomass/waste unconstrained | runs ~98% CF vs ~39% real | fuel-limited plants, no availability cap |
| Ghost lignite ES 0.51 GW + FR 0.98 GW; mainland oil | lignite runs 4.1 TWh baseload at 28 EUR/MWh (coal group 4.1 vs 3.0 TWh actual) | stale inventory (no operating lignite in Spain since 2022 or in France); zeroed by `national_capacities` |
| Demand −3.4% | ENTSO-E vs Redeia busbar definition | closure, not error |
| Portugal (found later) | hydro 10.3 vs 14.9 TWh; biomass ~100% CF; wind CF 14.1% vs 29.2% real; 4.2 of 5.6 GW in the worst CF bin | same EIA gap + same biomass issue + ERA5 bias compounded by bad GEM plant coordinates |

## 2. Config-level fix (before any patch): wind resource classes

`renewable.onwind.resource_classes: 5` splits each region into 5 CF bins; existing plants
are **spatially joined** to the bin where they stand, so the fleet profile reflects actual
siting. ES fleet CF moved 16.8% → 18.6% (the siting share of the bias, roughly a third);
FR lands at 24.1% (validates as is), PT stays at 14.1% (ERA5 + coordinates). This is the
principled part of the wind treatment; the factor below covers only the residual.

## 3. The patch set (`prepare_network.py`, all switchable, all inputs cited)

| Patch | What it does | Data |
|---|---|---|
| `boundary_flows` | FR external borders (GB, IT, CH, DE+BE) fixed to measured hourly 2024 flows; on in every variant (closure, not calibration) | RTE eCO2mix extract, vendored; −86.1 TWh net checks against RTE's ≈ −89 total minus ES |
| `national_capacities` | Fleet totals per carrier and scope (ES peninsular / Balearics, PT, FR) set pro rata to published actuals; creates missing units with technology-data efficiency (ES coal, Balearic OCGT); zeroes the phantom lignite units; PT CCGT is the **market-dispatched** fleet (2,839 MW: Tapada do Outeiro 990 MW in security-of-supply reserve since Q1 2024, ERSE); for 2024, wind/solar use **average operating capacity** (mean of year-end 2023/2024 — ~6 GW of PV commissioned during 2024) | `national_capacities.csv` (Redeia ISE, MITECO PNIEC, JRC, IIT/Huclin, RTE, REN, ERSE) |
| `wind_correction` | ES and PT onwind profiles scaled for the residual ERA5 bias, capped at 1.0; FR untouched. In the 2024 back-cast the factor is derived **at run time** (cited generation / modelled fleet availability: ES ×1.187, PT ×2.053); case-study runs apply those factors as fixed cited constants from `wind_correction.csv`. New-build PPA candidates inherit corrected profiles by construction | ES: Redeia; PT: `pt_actuals.csv` (Ember extract, vendored; capacity cross-checks Redeia/REN); factors: `wind_correction.csv` |
| `phs_split` | Pure PHS energy 120 GWh (Huclin et al.), pumping 3,306 MW, round-trip 0.75; mixed PHS stays inside reservoir hydro per Redeia convention | `national_capacities.csv` |
| `chp_profile` | Cogeneration as a fixed profile: monthly shape derived at run time from the vendored Redeia monthly cogeneration CO2 series, scaled to published annual output (16.42 TWh, 0.366 tCO2/MWh_el), distributed by nodal load share. By construction | `chp.csv` + vendored REData JSON |
| `csp` | 2.3 GW CSP, nodal solar profile rescaled to the realized 2024 CSP capacity factor. By construction (no storage dispatch modelled; stated assumption) | `national_capacities.csv`, `validation_2024_es.csv` |
| `biomass_availability` | Biomass/waste `p_max_pu` capped at the realized capacity factor (ES 0.391, PT 0.563) — fuel-limited plants | cited generation ÷ capacity |
| `prices` | Thermal marginal costs = VOM + fuel/η + ETS × intensity/η (2024: MIBGAS PVB gas 38.08, CIF-ARA coal 14.83, ETS 65.24; realized market prices at the hubs the plants buy at, not scenario inputs); PT gas plants additionally pay the Portuguese gas network premium (+1.10 EUR/MWh_th, ERSE: UGS parcela I + firm entry/exit URT), breaking the Iberian gas near-tie in the observed direction | `prices.csv`, processed technology-data (VOM), ERSE TEP gás 2025-26 |
| `ntc_caps` | ES-FR 2.8 GW, ES-PT 4.2 GW, Peninsula-Balearics 0.4 GW via per-element `s_max_pu`/`p_max_pu` scaling (impedances untouched; pro-rata caps are mildly conservative under KVL) | `ntc.csv` |
| `demand` | Country loads scaled to cited demand (ES 239.7 and PT 51.4 TWh busbar; FR 449.2 TWh RTE weather-corrected consumption) | `annual_demand.csv` |
| `hydro_inflow` | wy2024 runs only (back-cast and case study): ES and PT reservoir inflow scaled to observed 2024 hydro energy (ES ×1.143, PT ×2.177), closing the EIA data gap. FR needs none (model hits the actual 74.7 TWh) | `validation_2024_es.csv`, `pt_actuals.csv` |
| `fr_nuclear_profile` | French nuclear availability seasonalized with the measured 2024 monthly output shape (36.6 GW June – 49.3 GW December), normalized to preserve the upstream annual factor (0.715); applied in every weather year (planned-outage calendar assumed to persist). Kept for physical realism although it does not help the net-exports gate (see §5) | `fr_nuclear_monthly_2024.csv` (eCO2mix via ODRE, vendored) |
| `ma_ad_exports` | Fixed flat ES exports to Morocco (2.54 TWh) and Andorra (0.24 TWh) at the nearest nodes | `validation_2024_es.csv` |

## 4. Current result: 8 of 8 gates pass

| Gate | Raw | Patched | Status |
|---|---|---|---|
| Wind (±15%) | −13.1% | **−2.5%** | PASS |
| Solar PV (±15%) | +46.2% | **+3.4%** | PASS |
| Nuclear (±10%) | +5.4% | −2.2% | PASS |
| Hydro (±20%) | −30.4% | −8.4% | PASS (by construction via inflow) |
| CCGT+CHP (±25%) | −62.3% | +10.0% | PASS |
| CO2 (±20%) | −37.6% | **−5.1%** | PASS |
| Load shedding (< 0.01%) | 0 | 0 | PASS |
| Net exports (sign + ±50%) | −196% | **+9.08 vs +10.23 TWh (−11.2%)** | PASS |

OCGT carbon price (2026-10-20): `national_capacities` creates France's 1.5 GW of OCGT
from scratch (no upstream units), and the created carrier carried no CO2 intensity, so
it ran ~0.18 TWh/yr without a carbon price and uncounted. Created carriers now take the
technology-data CO2 intensity (OCGT 0.198 t/MWh_th, as CCGT). Effect on the back-cast,
same solver before and after: net exports −0.03 TWh, nuclear −0.03 TWh, CO2 +0.002 Mt;
every gate unchanged. The table reflects a Gurobi solve; HiGHS and Gurobi reach different
equally optimal dispatches of this LP, ~0.1 TWh apart on nuclear and the French exchange.


## 5. The net-exports gate: diagnosis, experiments, resolution

Net exports decompose into ES→FR and ES→PT; MA/AD are exact by construction. Each
candidate cause was implemented or measured rather than argued (all numbers wy2024,
patched, no_dc):

1. **France's flat nuclear availability — tested, hypothesis falsified, patch kept.**
   Upstream applies the correct annual factor (0.715) flat over the year. The
   `fr_nuclear_profile` patch replaces it with the measured 2024 monthly output shape
   (eCO2mix: 36.6 GW June to 49.3 GW December), annual mean preserved. The experiment
   *worsened* the gate (net exports +0.96 → −2.49 TWh; ES→FR −6.9 → −8.9) and pushed
   CO2 out of gate (−17.1% → −23.6%): more winter nuclear hands France more cheap
   energy to export into Spain, and those imports displace Spanish CCGT. The winter
   scarcity hypothesis is dead. The patch stays **on** regardless — it is better
   physics, and it is applied in every weather year (the planned-outage calendar is
   assumed to persist), so stage-iv sensitivities change weather, not the calendar.
2. **Transmission capacity — measured, rejected.** The model touches its 2.8 GW cap in
   only ~3% of hours per direction. The measured 2024 commercial exchanges (vendored
   eCO2mix raw) reached 3.6 GW in both directions, above the published NTC, with gross
   FR→ES of 10.6 TWh: capacity limits neither reality nor the model. A directional cap
   tight enough to force the totals (~1.7 GW) would contradict the measured envelope —
   gate-fitting, not a correction.
3. **Gross decomposition.** Model ES–FR gross import 14.1 TWh vs 10.1 actual (+4.0),
   gross export 5.2 vs 7.1 (−1.9): the model trickle-imports at ~2.2 GW whenever France
   is marginally cheaper, which in the model is ~70% of all hours (mean spread ~4
   EUR/MWh).
4. **PT gas misordering — found and fixed with cited inputs.** Portugal's cheapest CCGT
   (87.87 EUR/MWh) undercut Spain's cheapest (88.24; fuel prices of that run, now 96.07 vs
   94.56 with the fix and the realized MIBGAS prices) by an inventory-noise efficiency
   edge, so the model ran PT gas at 12.9 TWh (CF 38%) against ~3 TWh real, suppressing
   both ES→PT exports and Spanish CCGT/CO2. Two corrections from the ERSE gas tariff
   decision 2025–26 (`erse_tep_gas_2025_2026`): (a) the Portuguese **gas network
   premium** (+1.10 EUR/MWh_th: UGS parcela I 0.71 + annual firm entry/exit URT ≈ 0.39)
   that PT plants pay on top of the PVB/MIBGAS hub price — a conservative lower bound,
   since PT CCGTs are supplied predominantly via the Sines LNG terminal and its
   regasification charge is deliberately not counted;
   (b) **Tapada do Outeiro (990 MW)** ended its CAE in Q1 2024 and is held in
   security-of-supply reserve outside the market until 2026-03-31 — removed from the
   market-dispatched PT fleet.
5. **Residual — accepted and documented.** What remains is price-duration flatness: a
   15-node, 3-hourly, cost-based LP has too few very cheap hours on either side of the
   border (no negative bids, hydro at opportunity value, 3h averaging smoothing the
   midday solar troughs), so bidirectional gross flows under- and over-shoot even when
   every gross energy quantity validates within ~10%. This is a structural property of
   the model class, stated as a limitation rather than patched.

**Resolution (2026-10-19):** with `fr_nuclear_profile` and the two PT corrections on,
the gate **passes**: net exports +9.08 vs +10.23 TWh actual (−11.2%; current solve,
see the OCGT note in §4), and CO2 moves to −5.1%. The pass carries a
stated caveat: the border components compensate (ES→PT +16.0 vs +10.5; ES→FR −9.7 vs
−3.0 from the persistent import excess in item 5), so the
component-level numbers are reported alongside the total in §4.

## 6. Known caveats to carry into the paper

- Wind correction factors are derived on the 2024 back-cast. On the wy2024 reference
  weather year the case study uses them **in-sample by design**; applying them to the
  planned wy2010/13/20 sensitivities is an out-of-sample transfer whose check is queued
  (train on 2022/23 profiles, test on 2024; PT actuals for 2022/23 are already vendored,
  ES needs two more profile builds).
- Hydro inflow (wy2024 runs), CHP output and CSP output match actuals **by construction**;
  they are input choices, not validation successes, and are labelled as such above.
- Coal and oil dispatch are not reproduced (non-market operation); their ≈ 2.9 Mt CO2
  are partially offset in the CO2 total by the model's CCGT overshoot (+15%), so the
  small total CO2 deviation (−5.1%) rests on compensating technology-level errors.
- The net-exports gate passes on the total; the ES→FR and ES→PT components carry
  compensating errors (±~6 TWh) rooted in the price-duration flatness of §5.5.
- The average price level is informational only; grid-purchase costs reported from the
  model inherit its bias: +33% in the 2024 back-cast (€84.0 vs €63 OMIE average), +13% in
  the 2025 case study (€73.8 vs €65.3). Decomposition on the solved
  back-cast: the weighting convention is immaterial (arithmetic 83.4 vs load-weighted
  84.0) and there are no scarcity spikes (max 95); the entire gap is a cheap-hour
  deficit — the model prices below 1 EUR in 0.3% of snapshots against ~8.5% of real
  2024 hours, and its median (94.6) sits at the CCGT marginal cost. Causes: hydro priced at fossil opportunity value and no zero/negative bidding
