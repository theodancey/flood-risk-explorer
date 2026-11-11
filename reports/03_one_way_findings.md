# Report 03: One-Way Analysis of Flood Risk Factors (Virginia, 2010–2025)

*Task 5, 2026-11-11. Source: `notebooks/04_one_way_analysis.ipynb` (code in `src/one_way.py`). Figures: `reports/figures/04_*.png`. All dollars are 2025 dollars. Area-level analysis of aggregated public data; nothing here describes an individual property.*

## In one paragraph
Looking at one factor at a time across 1.48 million policy-years and 9,557 paid claims, the clearest and most **stable** frequency drivers are **building age** (pre-1980 buildings claim about 3× as often as 1980–99 buildings) and the closely linked **post-FIRM** flag (0.35×), followed by **flood zone** (B/C/X 0.41× A/V). Flood zone matters far more in hurricane years than in normal years. Severity is driven by **occupancy** (multi-unit 2.3×, non-residential 3.1× the single-family average claim), **elevation** (0.54×), and **second homes** (1.7×). Several one-way effects are **confounded**. Elevated buildings look riskier on their own (1.08×) but are safer within every flood zone (0.86×), because they sit in the riskiest places. This is why the next step is a GLM, which adjusts for all factors at once. For "coastal," the **Tidewater Virginia** definition clearly beats the Census coastline list. But Virginia has two flood regimes (hurricane surge on the coast; river and flash flooding inland), and coastal effects flip direction between them.

## How to read the numbers
- **Relativity** = a level's frequency (or severity) ÷ the base level's. The base is the level with the most exposure. 0.41 means 59% lower than the base.
- **95% CI**: widened for **over-dispersion**. Flood claims arrive in clumps (one storm floods thousands of homes), so plain Poisson intervals are far too narrow. The widening factor √φ comes from the year-by-level table.
- **Importance F**: the improvement in fit (deviance drop per parameter) divided by the over-dispersion. F ≈ 1 is noise level.
- **Stability**: the relativity in the 3 catastrophe years (2011 Irene, 2012 Sandy, 2016 Matthew; 66% of claims) vs the other 13 years, and the share of years pointing the same way.
- **Zone-adjusted**: the relativity compared within each flood zone and then pooled (Mantel–Haenszel). A big change from the one-way value signals confounding.
- Statewide: frequency **6.45** paid claims per 1,000 policy-years, severity **$30,625**, pure premium **$197**.

## Frequency: ranked
| Rank | Feature | Key relativities (95% CI) vs base | F | Zone-adjusted | Cat yrs / other yrs | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Post-FIRM** | yes 0.35 (0.31–0.40) vs no | 272 | 0.38 (0.86 also adjusting for year built) | 0.36 / 0.33 | **Stable**, same direction in all 16 years. Mostly a proxy for building age. |
| 2 | **Year built** | before 1960 2.98 (2.46–3.60); 1960–79 2.86 (2.38–3.44); 2000+ 0.60 (0.44–0.81) vs 1980–99 | 106 | 2.74 / 2.67 / 0.60 | 3.09 / 2.77 (pre-1960) | **Stable**, same direction in all 16 years |
| 3 | **Flood zone** | B/C/X 0.41 (0.30–0.56) vs A/V | 33 | n/a | 0.29 / 0.75 | **Storm-dependent**: real in both regimes, but 2.6× stronger in hurricanes |
| 4 | **Building coverage** | under $100K 1.90 (1.41–2.55); $100K–249K 1.79 (1.45–2.21); $0 2.03 (1.03–4.02, *low credibility*) vs $250K+ | 13 | 1.74 / 1.52 / 2.17 | 0.94 / 3.42 (under $100K) | **Storm-dependent**. Partly geography: low limits are common inland. |
| 5 | **Occupancy** | multi-unit 1.43 (1.12–1.82); non-residential 1.13 (0.84–1.53) vs single-unit | 4.0 | 1.31 / 0.96 | 1.31 / 1.46 (multi-unit) | Mostly stable. Modest. |
| 6 | **Primary residence** | not primary 1.14 (0.99–1.30) vs primary | 3.5 | **0.98** | 0.96 / 1.28 | Weak. Largely explained by flood zone. |
| 7 | **County group** (35 levels) | from York County 0.21 to Roanoke County 2.17 vs Virginia Beach. Most CIs cross 1. | 1.4 | n/a | n/a | High-frequency inland cluster: Roanoke city/County, Salem; the non-tidal small-county pool is 1.34 |
| 8 | **Floors** | 3+ 1.11 (0.91–1.36); 1 0.99 vs 2 | 0.7 | 1.14 / 0.95 | 0.86 / 1.33 | Weak and flips |
| 9 | **Tidal (Tidewater)** | non-tidal 1.42 (0.60–3.34) vs tidal | 0.6 | 1.32 | **0.25 / 5.00** | **Flips** between regimes (see coastal section) |
| 10 | **Elevated** | yes 1.08 (0.72–1.60) vs no | 0.1 | **0.86** | 1.42 / 0.70 | **Confounded** with zone, and flips |
| 11 | **Coastal (Census)** | inland 1.04 (0.58–1.89) vs coastal | 0.02 | 1.03 | 0.41 / 3.43 | Separates nothing overall |

## Severity (average paid claim): ranked
| Rank | Feature | Severity relativity (95% CI) vs base | F | Without cat years | Comment |
|---|---|---|---|---|---|
| 1 | **Occupancy** | multi-unit 2.35 (2.11–2.61); non-residential 3.10 (2.72–3.54) vs single-unit ($25,088) | 319 | 3.09 (multi-unit) | Bigger buildings and contents; the strongest severity factor |
| 2 | **Elevated** | 0.54 (0.50–0.57) vs not elevated ($38,290) | 287 | 0.64 | Water reaches less of the building. Direction is stable. |
| 3 | **Primary residence** | not primary 1.72 (1.60–1.85) vs primary ($25,128) | 235 | 2.71 | Overlaps with occupancy (V = 0.46), since rentals, condos, and businesses are mostly "not primary" |
| 4 | **Year built** | 1960–79 1.43 (1.28–1.60); before 1960 0.86 (0.77–0.96); 2000+ 0.85 (0.71–1.02) vs 1980–99 | 49 | 0.74 (pre-1960) | Not monotonic; worth testing in the GLM |
| 5 | **Building coverage** | $0 0.45; under $100K 0.57; $100K–249K 0.82 vs $250K+ ($35,515) | 39 | 0.49 (under $100K) | Largely mechanical: a claim can't exceed the limit |
| 6 | **Flood zone** | B/C/X 0.78 (0.72–0.85) vs A/V ($32,637) | 30 | 0.58 | Shallower flooding outside the floodplain |
| 7 | **Floors** | 1 floor 1.22 (1.11–1.33); 3+ 0.81 (0.73–0.90) vs 2 | 25 | 1.16 (3+) | A single-storey home has more of its value near the ground |
| 8 | **Tidal (Tidewater)** | non-tidal 1.28 (1.15–1.42) | 23 | 1.75 | Inland river floods are bigger per claim |
| 9 | **County group** | most localities 0.3–0.8 vs Virginia Beach | 12 | n/a | Virginia Beach's high severity is mostly Matthew |
| 10 | **Post-FIRM** | 0.86 (0.79–0.94) | 11 | 0.83 | Small |
| 11 | **Coastal (Census)** | 1.03 (0.94–1.12) | 0.4 | 1.64 | Nothing overall |

**Pure premium** (frequency × severity), for the headline story: A/V $296 vs B/C/X $95 per policy-year (0.32×); pre-FIRM $300 vs post-FIRM $91 (0.30×); built 1960–79 $380 vs 1980–99 $93 (4.1×); multi-unit $524 and non-residential $551 vs single-unit $157 (about 3.4×); second homes $305 vs primary $156 (1.95×); non-tidal $329 vs Tidewater $181 (1.81×).

## Overlap and confounding
- **Strongest overlaps (Cramér's V, 0 to 1):** post-FIRM ↔ year built **0.81**; county group ↔ coastal/tidal **≈1.0** (each county sits on one side of the line); occupancy ↔ primary residence **0.46**; county group ↔ flood zone 0.40; flood zone ↔ elevated 0.27. Everything else is below 0.25.
- **Elevated (Simpson's paradox):** one-way 6.72 vs 6.25 claims per 1,000 (1.08×). But within A/V it's 8.46 vs 9.78, and within B/C/X 3.23 vs 3.91, so elevated is lower in both zones. 67% of elevated exposure is in A/V vs 40% for non-elevated, because floodplain rules require elevation there. Zone-adjusted: **0.86**.
- **Post-FIRM is mostly building age:** 0.35 one-way, but **0.86** after adjusting for zone and year built.
- **Primary residence frequency** almost vanishes after adjusting for zone (1.14 → 0.98).
- **Coverage band** shrinks after adjusting for zone and year built (under $100K: 1.90 → 1.39).

## The coastal question (Question 6)
| | Census coastline list | Tidewater Virginia (Code of Virginia § 62.1-44.15:68) |
|---|---|---|
| Localities / exposure share | 25 / 74% | 46 / 89% |
| Inland frequency relativity, all years | 1.04 | **1.42** |
| Zone-adjusted | 1.03 | **1.32** |
| Inland pure premium relativity | 1.07 | **1.81** |
| Excluding cat years (one-way) | 3.43 | **5.00** |
| Poisson GLM deviance gain over zone + year (all years / excl. cat) | 0.4 / 1,098 | **38.8 / 1,426** |

**Recommendation: use `coastal_tidal` (Tidewater).** The Census list's "inland" group mixes tidal cities with *low* frequency (Chesapeake 4.7 per 1,000) and mountain localities with *high* frequency (Roanoke city 14.5, Tazewell 18.9), and they cancel out. Tidewater draws the line between tidal and non-tidal Virginia, and it wins on every comparison. It also has a statutory source.

**Caveats.**
1. **The coastal effect flips by regime.** The coast is far riskier in hurricane years (non-tidal 0.25×), while inland is far riskier in normal years (5.0×). After over-dispersion, the all-years effect is not statistically significant (quasi χ² 0.56).
2. **The three-way split fits better still** (Census coastal / Tidewater-only / non-tidal; deviance gain 75 all years, 1,559 excluding cat years). Both flags are in the table, so it's a cheap week-3 test.
3. **Overlap with county:** with `county_group` in a model, the flag is redundant for the 32 credible counties.

**Acted on:** small counties are now pooled by Tidewater (`Other tidal (small)` / `Other non-tidal (small)`). The old `Other inland (small)` pool mixed 5.5 and 9.3 claims per 1,000.

## Caveats
1. **One-way ≠ causal.** Every number above mixes a factor's own effect with the factors it travels with. The GLM will separate them.
2. **Over-dispersion.** φ for relativities ranges from 8 (post-FIRM) to 230 (tidal). Plain Poisson tests would call almost everything significant. The frequency GLM needs a negative binomial or quasi-Poisson.
3. **Catastrophe dependence.** Two-thirds of claims come from 3 storms. Factors tied to *where* storms hit (zone, coastal, elevated, coverage) change with the storm mix. Results are reported with and without cat years for this reason.
4. **Credibility.** Only the `$0 (contents only)` coverage level is flagged as low credibility (1.4% of exposure, 199 claims). Most single-county relativities have wide intervals (e.g. Roanoke County 2.17, CI 0.75–6.24).
5. **Unmatched claims.** These tables include the 1% of paid claims with no matching exposure; the frequency GLM will drop them. The 2021–2025 window has few claims (677), so the "2021–2025" relativities are noisy.
6. **Peeking.** This analysis used all 16 years. From week 3 on, model choices use only 2010–2020 (see `docs/validation_plan.md`).

## Shortlist for the week 3 GLMs
**Frequency (Poisson → negative binomial, log(exposure) offset):**
- **Core:** `flood_zone_group`, `construction_band`, `building_coverage_band`, `occupancy_group`, `elevated` (expected to come out about 0.86 once zone is in the model).
- **Test, keep only if it adds fit in development LOYO:** `post_firm` (overlaps year built; about 0.86 after adjustment), `primary_residence` (about 0.98 after zone), `floors_group`.
- **Geography, compared head-to-head in LOYO:** (A) `coastal_tidal`; (B) `coastal` + `coastal_tidal` (three-way); (C) `county_group` (35 levels, the territory factor the app needs). Drop Census `coastal` on its own.
- **Year:** fit year as a factor so storm years don't distort the relativities, then predict at an average-year level. Consider a `flood_zone_group × coastal_tidal` interaction (the zone effect differs by regime).

**Severity (Gamma, log link, weighted by claim count):**
- **Core:** `occupancy_group`, `elevated`, `primary_residence`, `construction_band`, `building_coverage_band`, `flood_zone_group`, `floors_group`.
- **Geography:** `coastal_tidal`, then test `county_group`.
- **Drop:** Census `coastal`. `post_firm` is optional (0.86, small).

## Figures
- `04_oneway_zone_occupancy_floors.png`, `04_oneway_yearbuilt_coverage.png`, `04_oneway_yes_no.png`: one-way exhibits
- `04_oneway_county_group.png`: county relativities with CIs
- `04_stability_by_year.png`: each factor's relativity year by year
- `04_feature_association.png`: Cramér's V heatmap
- `04_coastal_three_way.png`: Census coastal / Tidewater-only / non-tidal, with and without cat years
- `04_validation_folds.png`: what each validation fold contains
