# The Modelling Dataset: `data/processed/va_model_data.parquet`

*Built by `src/build_dataset.py` (run: `.venv\Scripts\python.exe src\build_dataset.py`, about 20 seconds). Virginia, calendar years 2010–2025. Summary and charts: `notebooks/03_dataset_summary.ipynb`.*

This is the table the frequency and severity models (weeks 3–5) will be trained on. It's built only from FEMA's public NFIP policies and claims, plus two small official reference files (Census coastline counties and BLS inflation data).

## What one row means
**One row = one rating cell in one calendar year.** A **rating cell** is a group of policies that look identical on every rating feature: for example, "Virginia Beach, 2018, A/V flood zone, single-unit residential, 2 floors, not elevated, primary residence, pre-FIRM, built 1960–1979, $250K+ building coverage." For each cell the table adds up how much insurance was in force (exposure) and the claims and dollars that came from it.

Why cells instead of one row per policy?
- Policies and claims are **separate files with no shared ID**, so a claim can't be linked to its exact policy. What we *can* do is put both into the same cell using the features they both record, and then compare claims with exposure inside each cell.
- For a Poisson GLM with categorical features, adding up rows that share the same features loses no information: the fitted coefficients depend only on those sums. So the cell table gives the same frequency model as policy-level data, in fewer rows. The Gamma severity model, fitted on average claim size weighted by claim count, gives the same coefficients too (only its spread estimate can differ).

| | Value |
|---|---|
| Rows (cells) | **252,531**: 252,420 with exposure + 111 with claims but no exposure (see "Unmatched claims") |
| Distinct feature combinations (ignoring year) | 29,087 |
| Exposure | **1,482,412.77** policy-years |
| Claims | **12,545** (all), **9,557** paid (total paid > $0) |
| Paid losses | **$220,183,315** nominal = **$292,685,981** in 2025 dollars |
| Statewide measures (2025 $) | frequency 6.45 per 1,000 policy-years (6.38 using only cells with exposure) · severity $30,625 · pure premium $197 |

These totals equal notebook 02's exactly (that's checked on every run; see "Data checks").

## Columns
| Column | Role | Meaning |
|---|---|---|
| `year` | **key + feature** | Calendar year (2010–2025). For exposure, it's the year the cover was in force. For claims, `yearOfLoss`. |
| `county` | **key** | 5-digit county FIPS code. Policies: first 5 digits of `censusGeoid`. Claims: `countyCode`. Same rule as notebook 02. Blank → `UNKNOWN`. |
| `county_name` | label | Census Bureau name (e.g. "Virginia Beach city") |
| `county_group` | **feature** | Modelling version of county: the 32 counties with ≥5,000 policy-years keep their own level. The rest are pooled into `Other coastal (small)` / `Other inland (small)`, plus `Other/unknown county` (blank or out-of-state codes). |
| `coastal` | **feature** | County is on the Census Bureau's coastline-county list |
| `flood_zone_group` | **key + feature** | `A/V high risk` or `B/C/X moderate-low risk` |
| `occupancy_group` | **key + feature** | `single-unit residential`, `multi-unit residential`, `non-residential` |
| `floors_group` | **key + feature** | `1`, `2`, `3+` |
| `elevated` | **key + feature** | Building is elevated (posts, piles, walls) |
| `primary_residence` | **key + feature** | The owner lives there most of the year |
| `post_firm` | **key + feature** | Built after the community's first flood map (FIRM), so built to floodplain rules |
| `construction_band` | **key + feature** | Year built: `before 1960`, `1960-1979`, `1980-1999`, `2000 or later` |
| `building_coverage_band` | **key + feature** | Building coverage limit: `$0 (contents only)`, `under $100K`, `$100K-$249K`, `$250K or more` |
| `exposure` | **target denominator (offset)** | Earned policy-years. Each policy term counts 1 per year in force, not weighted by condo units. |
| `claims` | **frequency target** | Paid claims (total paid > $0) |
| `claims_all` | alt. target | All claims, including those closed without payment |
| `losses_nominal` | check | Total paid (building + contents + ICC), dollars of the year of loss |
| `cpi_factor` | helper | CPI(2025) ÷ CPI(year) |
| `losses_2025` | **severity / pure-premium target** | `losses_nominal × cpi_factor` |
| `exposure_matched` | filter | `False` for the 111 cells whose claims found no exposure (exposure = 0). Drop them for frequency models, keep them for severity. |

**Features vs targets.** A **feature** (or predictor) is something you'd know *before* a loss, which the model uses to make its prediction. A **target** is the outcome the model tries to predict. Exposure is neither: it's the amount of insurance at risk, used as an **offset** (the model predicts claims per policy-year, then scales by exposure).

## Grouping rules
The same function (`make_features()`) is applied to policies and to claims, so both sides of each cell use identical definitions.

| Feature | Rule | Why |
|---|---|---|
| Flood zone | Zones starting **A or V** are the high-risk Special Flood Hazard Area (the "100-year floodplain") → `A/V high risk`. **B, C, X** → `moderate-low`. **D** (undetermined) and blank (114 policy terms, 1 claim) also go to moderate-low, because they're outside the SFHA. | This is the main regulatory split (inside the SFHA, flood insurance is usually mandatory for mortgages). Numbered zones (A01–A30) are just older map labels for the same risk class. |
| Occupancy | Single-unit residential = codes {1, 11, 14, 16}; multi-unit residential = {2, 3, 12, 13, 15}; non-residential = {4, 6, 17, 18, 19} | FEMA switched to 2-digit codes with Risk Rating 2.0 (Oct 2021). Before then, condo units (now 16) and mobile homes (now 14) were coded 1 = single family, so grouping them together keeps the meaning the same across 2010–2025. |
| Floors | 1 = {1, 5 mobile home}; 2 = {2, 4 split-level}; 3+ = {3, 6 townhouse 3+}; blank (237 terms, 2 claims) → 2 | Split-level (0.6% of exposure) and mobile home (0.5%) are below the 1% rule. A mobile home is single-storey, and a split-level sits between 1 and 2 floors. |
| Elevated / primary residence / post-FIRM | as recorded (no blanks) | Complete in both files |
| Construction year | 4 bands. Blank or before 1700 (728 terms, 3 claims; e.g. a year of "202") → `1980-1999`, the most common band | Vintage (year built) instead of age, so a building stays in the same band every year |
| Building coverage | $0 / under $100K / $100K–$249,999 / $250K+, in **nominal** dollars | The NFIP residential limit has been $250,000 since 1994, so "at the limit" means the same thing every year. Restating limits in 2025 dollars would move buildings between bands just because time passed. |
| County group | Own level if ≥5,000 policy-years in 2010–2025 (the credibility floor from notebook 02); otherwise pooled by coastal/inland | Small counties have too few claims to estimate their own effect. Pooling small counties with similar ones is a simple form of **credibility** (trust a group's own experience only when there's enough of it). `county` itself stays in the key, so the pooling can be changed later without rebuilding. |

**Rare-category rule.** Every level of every feature must hold **at least 1% of 2010–2025 exposure** (about 15,000 policy-years). A level below that is merged into the most similar named level (V → A, split-level → 2 floors, mobile home → 1 floor). The script checks this on every run. Smallest resulting level: `$0 (contents only)` coverage, 1.4%.

### Exposure share of each level
| Feature | Levels (share of exposure) |
|---|---|
| flood_zone_group | A/V high risk 51.0% · B/C/X moderate-low 49.0% |
| occupancy_group | single-unit 89.3% · multi-unit 6.0% · non-residential 4.7% |
| floors_group | 1: 32.6% · 2: 50.5% · 3+: 16.9% |
| elevated | yes 41.3% · no 58.7% |
| primary_residence | yes 72.2% · no 27.8% |
| post_firm | yes 49.0% · no 51.0% |
| construction_band | before 1960 21.1% · 1960–79 28.9% · 1980–99 30.8% · 2000+ 19.1% |
| building_coverage_band | $0 1.4% · under $100K 9.4% · $100K–$249K 28.4% · $250K+ 60.8% |
| coastal | yes 74.0% · no 26.0% |

### Features considered and rejected
| Candidate | Why not (for now) |
|---|---|
| Basement/enclosure type | 54–58% blank in both files |
| CRS class (community discount) | 63% blank in claims vs 18% in policies. The blanks don't line up, so cells wouldn't match. |
| Foundation type | Mostly "0 = not reported" before Risk Rating 2.0 |
| Deductible | The code list changed with RR2.0 (new letter codes). A good candidate for later, once mapped to dollar amounts. |
| `floodZoneCurrent` | The current map can be redrawn *after* the loss, which is a leakage risk. We use `ratedFloodZone` (the zone the policy was priced on). |
| ZIP code / census tract | Too sparse for a first model. ZIP can come later through the app. |
| Contents coverage, condo type | Kept the cell count manageable. Partly captured by occupancy and the $0-building band. |

## Inflation adjustment (2025 dollars)
- **Source:** CPI-U (Consumer Price Index for All Urban Consumers, all items, US city average, not seasonally adjusted, 1982–84 = 100), BLS series **CUUR0000SA0**, from the free BLS Public Data API v1 (no key needed). Downloaded by `src\download_data.py cpi` to `data/raw/cpi_u_bls.csv`. (FRED's copy, CPIAUCNS, is the same numbers, but its download link timed out for Python scripts.)
- **Method:** annual average of the 12 monthly values; `cpi_factor = CPI(2025) / CPI(year of loss)`; `losses_2025 = losses_nominal × cpi_factor`. Example: a $10,000 claim in 2010 is $10,000 × 322.15 / 218.06 = **$14,774** in 2025 dollars.
- **October 2025 is missing.** BLS didn't collect it ("Data unavailable due to the 2025 lapse in appropriations", the federal shutdown). I filled it with a straight line between September (324.800) and November (324.122). The 2025 average is 322.153 (vs 321.943 using the 11 published months, a 0.07% difference).
- **Check:** the 2010 average equals BLS's published 218.056.

| Year | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI-U avg | 218.06 | 229.59 | 236.74 | 240.01 | 251.11 | 258.81 | 270.97 | 292.65 | 304.70 | 313.69 | 322.15 |
| Factor to 2025 $ | 1.477 | 1.403 | 1.361 | 1.342 | 1.283 | 1.245 | 1.189 | 1.101 | 1.057 | 1.027 | 1.000 |

Only **losses** are adjusted. Exposure is in policy-years (not dollars), and coverage bands are deliberately nominal (see above).

## Unmatched claims
Every claim is assigned to a cell, built from the claim's own features. A claim is **unmatched** when no policy with the same county, year, and 8 features was in force, so its cell has exposure = 0.

| | Count | Share |
|---|---|---|
| Unmatched cells | 111 | |
| Claims in them | 122 (95 paid) | 0.99% of paid claims |
| Paid losses in them | $2,698,593 nominal | 1.23% of losses |

**Why?** For each unmatched cell, the script tests whether ignoring one feature would let it match:
- **No unmatched claim is in a county-year with zero exposure.** Every one is a *rare combination*.
- Ignoring **county** would match 118 of the 122 (the same building type is insured elsewhere in Virginia), and ignoring **coverage band** would match 105. Other features: floors 52, construction 48, zone 44, elevated 35, primary residence 28, occupancy 21, post-FIRM 7.
- Only 2 claims (1 cell) differ from all policies on exactly one feature. The rest would match if any one of several features were ignored.

Likely causes: (1) a feature recorded on the claim differs slightly from the policy file (e.g. coverage changed mid-term, or the floors code was corrected by the adjuster); (2) the matching policy term is missing from the policy file, or is in a different county code.

**How they're handled:** they're kept as rows with `exposure = 0` and `exposure_matched = False`, so totals reconcile exactly with notebook 02. **Frequency** models must drop them (a rate needs exposure, and log(0) is undefined), which undercounts frequency by about 1%. **Severity** models can keep them, since a paid claim's size is valid on its own.

## Leakage
**Data leakage** means a feature contains information that wouldn't be known at the time of prediction, usually because it comes from the outcome itself. It makes a model look great in testing and fail in real use. Here:
- Every cell feature describes the **policy or building before the loss** (zone the policy was rated on, occupancy, floors, elevated, primary residence, post-FIRM, year built, coverage limit).
- **Not used:** anything produced by the claim: damage amounts, water depth, flood duration, cause of damage, event name, non-payment reason, ICC, adjuster's property value or replacement cost, and `floodZoneCurrent` (maps can be redrawn after a flood).
- County-level features (`coastal`, `county_group`) come from an external list and from **exposure** only, never from claim counts or losses.
- For the year-holdout tests in weeks 3–6, the county pooling threshold uses all 2010–2025 *exposure*. That uses no claim information, so it isn't leakage, but it's noted here for completeness.

## Data checks (run every time; any failure stops the script before saving)
1. Exposure, claims (all and paid), and nominal losses equal both the raw inputs and notebook 02's totals: 1,482,412.77 / 12,545 / 9,557 / $220,183,315.29 (tolerances 0.01 policy-years, $1), plus `va_county_year.parquet` when present.
2. No negative values. No missing values in any column. No duplicate cells. Paid claims ≤ all claims. Losses only where there are paid claims.
3. Every claim is counted in exactly one row (claim totals equal the input). Unmatched claims are counted and explained (printed, and saved to `data/processed/va_model_data_checks.json`).
4. CPI: 2025 factor = 1, factors ≥ 1 and falling over time, 2010 CPI = 218.056.
5. Rare-category rule: every feature level ≥ 1% of exposure. 25 Virginia coastline counties loaded.
6. A self-test of the grouping rules on made-up rows with known answers (plus `actuarial.py`'s exposure self-test).

## Known limitations
- **Condo one-per-policy-term simplification.** Exposure counts each policy term as 1, even a condo association master policy (RCBAP) covering dozens of units. Claims are counted the same way (one claim record per policy), so frequency is consistent. But a cell containing a big condo building looks like one "policy-year." Weighting by `policyCount` would add about 13% to exposure (notebook 02).
- **Rare combinations can't be matched exactly:** 1% of paid claims (above).
- **CPI isn't construction-cost inflation.** Repair costs (materials, labour) rose faster than general prices in 2021–2023, so CPI-adjusted severity may still show some upward trend. A proper actuarial **trend** study would also look at claim-cost indices.
- **Coastal definition.** The Census list covers counties touching the ocean or a major estuary. **Chesapeake city (131K policy-years, a third of "inland" exposure) is not on it**, even though it has tidal waterways, and neither are some tidal-river counties. This mutes the coastal/inland difference (paid frequency 6.4 vs 6.6 per 1,000). Week 3 can test a broader "tidal" definition.
- **Risk Rating 2.0 (Oct 2021)** changed how several fields are coded. The groups are designed to be consistent across the change, but subtle recording differences may remain.
- **County coding:** FEMA warns that a claim's county may not always be exact. 0.45% of exposure has no county, and 52 policy-years carry out-of-state census codes (pooled into `Other/unknown county`).
- **Years:** 2009 is excluded (its exposure is only an estimate) and 2026 is incomplete. Recent years may still gain late-reported claims.
- **Who buys NFIP isn't random.** Results describe the insured population, not all buildings.
- **Area-level only.** Cells are statistical groups, not properties. Model outputs must never be presented as a prediction for an individual home.
