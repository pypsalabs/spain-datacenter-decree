# Validation Log — 2024 Back-cast (wy2024)

The solved `no_dc` network for weather year 2024 is compared against Redeia actuals in one
report (`results/validation/validation_{variant}.csv` and `.yaml`, from
`scripts/validate_network.py`). The **raw** variant is the fresh PyPSA-Eur build with only
the France boundary closure; the **patched** variant adds the cited corrections below.
The rule: *build fresh, validate, correct only what the report demands, cite every
correction*. No output is pinned to its historical value, except where marked "by
construction".

## 1. What a fresh PyPSA-Eur v2026.09.0 build gets wrong for Spain

| Finding | Raw vs actual | Cause |
|---|---|---|
| Wind −21% (−13% with resource classes, §2) | 46.8 vs 59.5 TWh; fleet CF 16.8% vs ~22% | (i) the fleet profile averages over all eligible land, not where turbines stand; (ii) ERA5 underestimates Iberian ridge winds; (iii) GEM regional allocation is off (10.2 GW in Aragón vs ~5.5 real) |
| Solar +46% | 41.3 GW vs 32.5 GW grid-connected | the inventory counts total PV including ~9 GW of self-consumption, which does not belong above busbar demand |
| Hydro −30% | inflow 25.6 vs 34.9 TWh | upstream normalizes inflow to EIA statistics, which end before 2024 (a wet year), giving an average year |
| No CO2 price | CCGT marginal cost ≈ 81 EUR/MWh (gas + VOM only) | the composed network has no ETS price (lignite at 28 EUR/MWh would run baseload) |
| Nuclear fuel 3× too expensive | nuclear marginal cost 27.3 EUR/MWh_el, above biomass (20.0) | technology-data v0.15.0 labels its nuclear fuel price (7.45) EUR/MWh_th, but the EIA source reports it per kWh of net generation, so dividing by η = 0.326 counts the efficiency twice |
| CHP missing | 0 vs 16.4 TWh, ~6 Mt CO2 | electricity-only PyPSA-Eur has no cogeneration |
| CSP missing | 0 vs 4.1 TWh | carrier not built by default |
| Borders over-trade | France pushes 28.0 TWh into Spain | ES-FR corridor capability 5.2 GW (3 AC lines × 0.7 + 2 × 1 GW INELFE) vs 2.8 GW commercial NTC |
| PHS and hydro artifacts | PHS 7.4 GW at 295 h; reservoir 1,402 h; run-of-river 0.5 vs 2.2 GW | JRC hydro database counts river-cascade dams as storage |
| Biomass and waste unconstrained | ~98% CF vs ~39% real | fuel-limited plants without an availability cap |
| Phantom lignite (ES 0.51 GW, FR 0.98 GW) and mainland oil | lignite runs 4.1 TWh baseload (coal group 4.1 vs 3.0 TWh) | stale inventory: no operating lignite in Spain since 2022, nor in France |
| Demand −3.4% | ENTSO-E vs Redeia busbar definition | a closure, not an error |
| Portugal | hydro 10.3 vs 14.9 TWh; biomass ~100% CF; wind CF 14.1% vs 29.2%, with 4.2 of 5.6 GW in the worst CF bin | the same EIA gap and biomass issue; the ERA5 bias compounded by wrong GEM plant coordinates |

## 2. Config-level fix first: wind resource classes

`renewable.onwind.resource_classes: 5` splits each region into 5 capacity-factor bins,
and existing plants are joined to the bin where they stand, so the fleet profile reflects
actual siting. Spain's fleet CF moves from 16.8% to 18.6%, about a third of the bias;
France lands at 24.1% and validates as is; Portugal stays at 14.1% (ERA5 and plant
coordinates). The wind factor below covers only the residual.

## 3. The patch set (`prepare_network.py`; each switchable, all inputs cited)

| Patch | What it does | Data |
|---|---|---|
| `boundary_flows` | France's external borders (GB, IT, CH, DE+BE) fixed to measured hourly 2024 flows; on in every variant (a closure, not a calibration) | RTE eCO2mix extract, vendored; −86.1 TWh net checks against RTE's ≈ −89 TWh total minus Spain |
| `national_capacities` | Fleet totals per carrier and scope (ES peninsular, Balearics, PT, FR) set pro rata to published actuals; creates missing units with technology-data efficiency and CO2 intensity (ES coal, Balearic OCGT, France's 1.5 GW OCGT); zeroes the phantom lignite. Portugal's CCGT is the market-dispatched fleet (2,839 MW: Tapada do Outeiro, 990 MW, sits in security-of-supply reserve since Q1 2024, ERSE). For 2024, wind and solar use average operating capacity (the mean of year-end 2023 and 2024; ~6 GW of PV came online during 2024) | `national_capacities.csv` (Redeia, MITECO PNIEC, JRC, Chaves, Huclin & Ramos, RTE, REN, ERSE) |
| `wind_correction` | Spanish and Portuguese onwind profiles scaled for the residual ERA5 bias, capped at 1.0; France untouched. The 2024 back-cast derives the factor at run time (cited generation ÷ modelled availability: ES ×1.187, PT ×2.053); the case study applies these as fixed cited constants. PPA candidates inherit the corrected profiles | ES: Redeia; PT: `pt_actuals.csv` (Ember extract, vendored); factors: `wind_correction.csv` |
| `phs_split` | Pure PHS: 120 GWh of storage and 0.75 round-trip efficiency (Chaves, Huclin & Ramos), 3,306 MW of pumping (JRC); mixed PHS stays inside reservoir hydro, as Redeia reports it | `national_capacities.csv` |
| `chp_profile` | Cogeneration as a fixed profile: the monthly shape of the vendored Redeia cogeneration CO2 series, scaled to published annual output (16.42 TWh, 0.366 tCO2/MWh_el) and distributed by nodal load. By construction | `chp.csv`, vendored REData JSON |
| `csp` | 2.3 GW of CSP with a solar profile rescaled to the realized 2024 capacity factor. By construction (no storage dispatch modelled) | `national_capacities.csv`, `validation_2024_es.csv` |
| `biomass_availability` | Biomass and waste `p_max_pu` capped at the realized capacity factor (ES 0.391, PT 0.563) | cited generation ÷ capacity |
| `prices` | Thermal marginal cost = VOM + fuel/η + ETS × intensity/η, with realized hub prices (2024: MIBGAS PVB gas 38.08, CIF-ARA coal 14.83, ETS 65.24). Nuclear fuel 2.43 EUR/MWh_th, the technology-data figure read per MWh_el as its EIA source reports it (marginal cost 11.9 instead of 27.3 EUR/MWh_el). Portuguese gas plants also pay the gas network premium (+1.10 EUR/MWh_th, ERSE), which breaks the Iberian gas near-tie in the observed direction | `prices.csv`, technology-data (VOM), ERSE gas tariffs 2025–26, EIA Table 8.4 |
| `ntc_caps` | ES→FR and FR→ES 2.8 GW, ES→PT 4.2 GW and PT→ES 3.5 GW, peninsula–Balearics 0.4 GW, by per-element `s_max_pu`/`p_max_pu` scaling (impedances untouched) | `ntc.csv` |
| `demand` | Country loads scaled to cited demand (ES 239.7 and PT 51.4 TWh busbar; FR 449.2 TWh, RTE weather-corrected) | `annual_demand.csv` |
| `hydro_inflow` | wy2024 runs only: Spanish and Portuguese reservoir inflow scaled to observed 2024 hydro energy (ES ×1.143, PT ×2.177), closing the EIA gap. France needs none (the model hits the actual 74.7 TWh) | `validation_2024_es.csv`, `pt_actuals.csv` |
| `fr_nuclear_profile` | French nuclear availability shaped by the measured 2024 monthly output (36.6 GW in June to 49.3 GW in December), preserving the annual factor (0.715); applied in every weather year, assuming the outage calendar persists. Kept for physical realism, although it does not help the net-exports gate (§5) | `fr_nuclear_monthly_2024.csv` (eCO2mix via ODRE, vendored) |
| `ma_ad_exports` | Fixed flat Spanish exports to Morocco (2.54 TWh) and Andorra (0.24 TWh) at the nearest nodes | `validation_2024_es.csv` |

## 4. Result: 8 of 8 gates pass

| Gate | Raw | Patched | Status |
|---|---|---|---|
| Wind (±15%) | −13.2% | **−3.0%** | PASS |
| Solar PV (±15%) | +46.2% | **+2.3%** | PASS |
| Nuclear (±10%) | +4.3% | −0.9% | PASS |
| Hydro (±20%) | −30.4% | −8.4% | PASS (by construction through inflow) |
| CCGT+CHP (±25%) | −60.5% | +11.7% | PASS |
| CO2 (±20%) | −36.5% | **−4.0%** | PASS |
| Load shedding (< 0.01%) | 0 | 0 | PASS |
| Net exports (sign and ±50%) | −196% | **+9.49 vs +10.23 TWh (−7.2%)** | PASS |

The table comes from an hourly solve. This LP has multiple optima: at 3h, a second solver run
reached a different, equally optimal dispatch, about 0.1 TWh apart on nuclear and on the French exchange.

## 5. The net-exports gate

Net exports split into ES→FR and ES→PT; Morocco and Andorra are exact by construction.
Each candidate cause was tested in the model (wy2024, patched, `no_dc`):

1. **France's flat nuclear availability: hypothesis rejected, patch kept.** Shaping French
   nuclear by the measured monthly output made the gate *worse* (net exports +0.96 →
   −2.49 TWh; ES→FR −6.9 → −8.9) and pushed CO2 out of gate (−17.1% → −23.6%): more winter
   nuclear lets France export more cheap energy into Spain, displacing Spanish CCGT. The
   patch stays on as better physics.
2. **Transmission capacity: rejected.** The model reaches its 2.8 GW cap in only ~3% of
   hours per direction. Measured 2024 exchanges reached 3.6 GW both ways, above the
   published NTC, with 10.6 TWh gross FR→ES, so capacity limits neither reality nor the
   model. A cap tight enough to force the totals (~1.7 GW) would contradict the measured
   flows.
3. **Gross flows.** The model imports 14.1 TWh gross from France against 10.1 actual
   (+4.0) and exports 5.2 against 7.1 (−1.9): it trickle-imports ~2.2 GW whenever France
   is marginally cheaper, which in the model is ~70% of hours (mean spread ~4 EUR/MWh).
4. **Portuguese gas misordering: fixed with cited inputs.** Portugal's cheapest CCGT
   (87.87 EUR/MWh) undercut Spain's cheapest (88.24) through an inventory-noise efficiency
   edge, at the fuel prices of that run, so the model ran Portuguese gas at 12.9 TWh
   against ~3 TWh real, suppressing ES→PT exports and Spanish CCGT and CO2. Two corrections
   from the ERSE gas tariff decision 2025–26 (`erse_tep_gas_2025_2026`): (a) the Portuguese
   **gas network premium**, +1.10 EUR/MWh_th on top of the MIBGAS hub price (UGS parcela I
   0.71 + firm entry/exit URT ≈ 0.39), a lower bound since Sines LNG regasification is not
   counted; (b) **Tapada do Outeiro (990 MW)** ended its power purchase agreement in Q1 2024
   and is held in security-of-supply reserve until 2026-03-31, so it leaves the
   market-dispatched fleet. With the fix and the realized MIBGAS prices, the cheapest CCGTs
   now sit at 94.56 (ES) and 96.07 EUR/MWh (PT).
5. **Residual: accepted as a model limitation.** A 15-node, cost-based LP has
   too few very cheap hours on either side of the border (no negative bids, hydro at its
   opportunity value), so gross flows in both directions over- and undershoot even when
   every gross energy quantity validates within ~10%.

**Result:** with the nuclear profile and the two Portuguese corrections, the gate passes:
net exports +9.49 vs +10.23 TWh actual (−7.2%), and CO2 moves to −4.0%. The border
components compensate (ES→PT +16.1 vs +10.5; ES→FR −9.4 vs −3.0 from the import excess in
item 3), so they are reported alongside the total.

## 6. Known caveats

- The wind factors are derived on the 2024 back-cast and used in-sample on the wy2024
  reference. Other weather years would be an out-of-sample transfer.
- Hydro inflow (wy2024 runs), CHP output and CSP output match actuals **by construction**:
  input choices, not validation successes.
- Coal and oil dispatch is not reproduced (non-market operation). Their ≈ 2.9 Mt CO2 are
  partly offset by the model's CCGT overshoot (+18%), so the small total CO2 deviation
  (−4.0%) rests on compensating errors.
- The net-exports gate passes on the total; the ES→FR and ES→PT components carry
  compensating errors (±~6 TWh) from the cheap-hour shortage of §5.5.
- The price level is informational only, and reported purchase costs inherit its bias:
  +31% in the 2024 back-cast (82.5 vs 63 EUR/MWh OMIE), +10% in the 2025 case study (71.9
  vs 65.3). On the back-cast, the weighting convention is immaterial (arithmetic 82.0 vs
  load-weighted 82.5) and there are no scarcity spikes (maximum 96): the whole gap is a
  shortage of cheap hours. The model prices below 1 EUR/MWh in 0.6% of hours against
  ~8.5% of real 2024 hours, and its median (94.6) sits at the CCGT marginal cost. The
  causes are hydro priced at its fossil opportunity value and the absence of zero or
  negative bids; temporal resolution is ruled out (3-hourly and hourly runs agree).
