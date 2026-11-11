# Report 05: The Claim-Severity GLM (Virginia, 2010–2025)

*Milestone 8, 2026-11-11. Source: `notebooks/06_severity_glm.ipynb` (code: `src/models/severity_glm.py`). Figures: `reports/figures/06_*.png`. Tables: `reports/05_severity_relativities.csv`, `reports/05_severity_model_steps.csv`, `reports/05_severity_second_pass.csv`, `reports/05_frequency_vs_severity.csv`. Saved model: `models/severity_glm.json`. All dollars are 2025 dollars. Area-level analysis of aggregated public data; nothing here is a prediction for an individual property.*

## In one paragraph
The severity model is a **Gamma GLM with a log link**, fitted to **9,553 individual paid claims** (building + contents paid, $290M), with calendar year as a factor. It contains **occupancy, year built, building coverage, flood zone, floors, primary residence and elevation**. Claim size is driven mostly by **what is exposed to the water**. Multi-unit buildings' claims are **2.1×** and non-residential buildings' **3.1×** a single-family home's. Low coverage limits mean smaller claims (under $100K: **0.69×**). Elevated buildings' claims are **0.76×**, and B/C/X zones' **0.75×**. Geography (Tidewater, county group) was tested again and **didn't hold up on unseen years**, the same result as for frequency. Comparing the two models, **flood zone and building age drive both** how often and how big. **Occupancy, coverage, elevation and second homes drive only severity.** **Floors push frequency and severity in opposite directions** and cancel out in the expected loss cost. Large losses are **not capped**: NFIP coverage limits already cut off the tail, and capping didn't help on unseen years. On held-out years the model beats a no-features baseline in **14 of 16 years** (mean **15.7%** deviance improvement; **16.7%** on the 2021–2025 holdout).

## 1. The modelling unit and the amount
- **One row per paid claim.** Every feature is recorded on the claim itself, and the same grouping code as the rating cells turns it into model features. Rebuilding the cell table's claim counts and losses from the individual claims matches **cell by cell** (9,557 paid claims, $292,685,981). A Gamma GLM on claim-weighted cell averages gives **identical coefficients** (difference 7×10⁻¹²). Individual claims are needed for the percentiles, capping, limit flags, residuals and the lognormal comparison.
- **Amount = building + contents paid, CPI-adjusted to 2025 dollars.** **ICC** (Increased Cost of Compliance: up to $30,000 to elevate, move or demolish a building the community declares substantially damaged) is **left out** of the modelled amount. It's triggered by a local decision rather than by the flood damage, it's paid later, and it's small: **0.96% of dollars** (78 claims, 4 of them ICC-only). It goes back in as a flat **ICC bridge factor of 1.0092**, which also accounts for the 4 ICC-only claims that the frequency model counts. Including ICC in the Gamma would move one relativity by 8% (built 2000+, 0.90 → 0.97) and the rest by under 2.5%.
- **The 95 "unmatched" claims stay in.** Their rating cell had no exposure, which matters for frequency but not for the size of a paid claim.

## 2. What claim sizes look like (`06_severity_distribution.png`, `06_policy_limits.png`)
| Median | Mean | P90 | P95 | P99 | Max | Skewness |
|---|---|---|---|---|---|---|
| $14,003 | $30,345 | $75,036 | $110,537 | $241,051 | $683,152 | 4.7 |

- **Heavy right skew:** the mean is 2.2× the median. The **largest 1% of claims (96) hold 11.3% of dollars**, the largest 5% hold **31.3%**, and the largest 10% hold 46.2%.
- **Policy limits:** **178 claims (1.9%) paid 99% or more of a coverage limit**, holding 7.2% of dollars. That's 85 building limits (16 at the $250K statutory maximum, 69 at a lower limit the owner chose) and 104 contents limits. Limit hits are most common for non-residential buildings (5.9% of their claims). These claims are **censored**: the damage was at least the limit, but we only see the limit.

## 3. Large losses: no cap (`06_large_losses.png`)
Compared on the 7 core features, development years only, with each cap set from that fold's training years:

| Approach | Load | Unseen-year improvement | Fold-to-fold spread of relativities | Multi-unit / non-residential |
|---|---|---|---|---|
| **No cap (chosen)** | 1.00 | **19.2%** | 0.040 | 2.12 / 2.98 |
| Cap at P99 ($234K) | 1.03 | 18.7% | 0.041 | 2.05 / 2.60 |
| Cap at P95 ($111K) | 1.14 | 17.5% | 0.038 | 1.70 / 2.11 |

**Recommendation: model uncapped amounts.**
1. Uncapped is best on unseen years, and P99 capping doesn't make the relativities any more stable.
2. NFIP limits already cap the tail.
3. The Gamma judges errors as *ratios*, so one huge claim has little pull.

**Capping at P95 is actively harmful.** The load is flat, but the capped-off dollars come from big buildings: 39% of non-residential dollars sit above P95, vs 4.6% for single-family homes. So capping at P95 underprices multi-unit and commercial buildings and overprices homes. If a cap is wanted later (e.g. for XGBoost), use P99. Its load is 1.03 overall but swings from 1.00 to 1.21 by year, which is why a load must be averaged over many years.

## 4. How the model was built (development years 2010–2020)
Same keep rule as frequency: **cluster-robust Wald p < 0.05** (storm-aware: claims in the same county group and year may share a surprise) **and** better **leave-one-year-out** Gamma deviance. Full tables: `reports/05_severity_model_steps.csv`, `05_severity_second_pass.csv`. Chart: `06_model_build_steps.png`.

| Step | ΔAIC | F-test p | Cluster Wald p | Δ unseen-year improvement (% pts) | Decision |
|---|---|---|---|---|---|
| Occupancy (vs year only) | −1,225 | <0.0001 | <0.0001 | +15.12 (10/11 years) | start |
| + elevated | −120 | <0.0001 | 0.0067 | −0.21 | drop (1st pass) |
| + primary residence | −8 | 0.0088 | 0.150 | +0.19 | drop (1st pass) |
| + year built | −189 | <0.0001 | <0.0001 | +0.17 | keep |
| + building coverage | −240 | <0.0001 | <0.0001 | **+2.11** (10/11) | keep |
| + flood zone | −93 | <0.0001 | <0.0001 | +0.85 | keep |
| + floors | −66 | <0.0001 | 0.0011 | +0.46 | keep |
| + post-FIRM | −14 | 0.0009 | 0.059 | −0.06 | drop |
| Geography A: + Tidewater | −19 | 0.0001 | 0.084 | +0.46 | drop (not significant after clumping) |
| Geography B: + Census coastal | +2 | 0.79 | 0.91 | −0.33 | drop |
| Geography C: county group | −412 | <0.0001 | <0.0001 | **−0.91** (5/11) | drop (worse on unseen years) |
| 2nd pass: + primary residence | | | 0.047 | +0.28 (9/11) | **keep** |
| 2nd pass: + elevated | | | 0.0007 | +0.18 (6/11) | **keep** |

- **AIC and the F-test would have kept 9 of 10 candidates.** As with frequency, the storm-aware test and unseen years are much stricter.
- **The second pass (order effects).** Elevated was tested second, before flood zone and building age were in the model, and failed. Re-tested against the forward-pass model, it passes clearly, and so does primary residence (narrowly: p = 0.047). *I added the second pass after a dry run showed this order effect.* The keep rule itself didn't change. The frequency model had no second pass, which is a follow-up for the model-comparison milestone.
- **Geography: dropped again.** Tidewater (non-tidal claims 1.30× bigger) helps a little on unseen years, but it isn't significant once storms are allowed for. County group cuts training deviance by 496 but makes unseen years worse. The out-of-fold county A/E (below) shows real but storm-dependent county differences.

## 5. Final relativities (refit on all 16 years)
Relativity = e^β, the multiplier on expected claim size versus the **base level**, holding the other features fixed. **Base levels are the frequency model's**: single-unit residential, built 1980–1999, $250K+ building coverage, A/V zone, 2 floors, primary residence, not elevated. **Base claim: $28,498** in an average year. Dispersion φ = 1.53: the coefficient of variation of a claim around its prediction is about 1.24 (the typical spread is as big as the prediction itself), which is normal for claim sizes. *Storm-aware* CIs are cluster-robust by county group × year (372 clusters), on average 1.7× wider (log scale) than plain Gamma CIs.

| Feature | Level | Claims | Actual avg claim | Relativity | 95% CI, storm-aware | 95% CI, plain Gamma |
|---|---|---|---|---|---|---|
| Occupancy | **single-unit residential (base)** | 8,262 | $24,769 | 1.00 | | |
| | multi-unit residential | 799 | $58,782 | **2.10** | 1.66–2.66 | 1.90–2.32 |
| | non-residential | 492 | $77,799 | **3.07** | 2.57–3.66 | 2.70–3.48 |
| Building coverage | **$250K or more (base)** | 4,386 | $35,262 | 1.00 | | |
| | $100K–$249K | 3,674 | $28,938 | 0.85 | 0.79–0.92 | 0.81–0.90 |
| | under $100K | 1,294 | $19,865 | **0.69** | 0.63–0.76 | 0.64–0.75 |
| | $0 (contents only) | 199 | $16,077 | **0.28** | 0.21–0.37 | 0.23–0.34 |
| Elevated | **no (base)** | 5,436 | $38,241 | 1.00 | | |
| | yes | 4,117 | $19,918 | **0.76** | 0.65–0.89 | 0.72–0.80 |
| Flood zone | **A/V high risk (base)** | 6,853 | $32,255 | 1.00 | | |
| | B/C/X moderate-low risk | 2,700 | $25,497 | **0.75** | 0.68–0.82 | 0.70–0.79 |
| Year built | **1980–1999 (base)** | 1,567 | $26,917 | 1.00 | | |
| | before 1960 | 3,196 | $22,955 | 1.02 | 0.85–1.23 | 0.95–1.10 |
| | 1960–1979 | 4,210 | $38,449 | **1.28** | 1.14–1.44 | 1.19–1.38 |
| | 2000 or later | 580 | $21,504 | 0.90 | 0.79–1.02 | 0.80–1.01 |
| Floors | **2 (base)** | 4,753 | $29,400 | 1.00 | | |
| | 1 | 3,030 | $35,629 | 1.12 | 0.97–1.30 | 1.06–1.19 |
| | 3+ | 1,770 | $23,835 | 0.89 | 0.82–0.98 | 0.83–0.96 |
| Primary residence | **yes (base)** | 6,645 | $24,848 | 1.00 | | |
| | no (second home, rental, business) | 2,908 | $42,905 | 1.12 | 1.00–1.25 | 1.05–1.19 |

*Example:* a multi-unit, A/V-zone, 1960s building with $250K+ coverage, 2 floors, primary residence, not elevated: $28,498 × 2.10 × 1.28 ≈ **$76,600** expected claim in an average year.

**One-way vs GLM:** the one-way severity numbers (report 03) shrink a lot once other factors are held fixed. Elevated goes 0.54 → 0.76: part of the one-way gap belongs to the other features elevated buildings travel with, and it's not zone, since 84% of elevated claims are in the higher-severity A/V zones vs 62% of the rest. Second homes go 1.72 → 1.12 (mostly occupancy and coverage). Year built before 1960 goes 0.86 → 1.02.

**Year factors** (vs the average year): 2016 Matthew 1.40, 2024 (Helene remnants inland) 1.46, 2019 1.31, but 2011 Irene 0.80 and 2010 0.57. Severity varies by year much less than frequency (Irene 4.2× in the frequency model), and a big claim *count* doesn't mean big claims.

## 6. Frequency vs severity: what drives what (`06_frequency_vs_severity.png`)
A factor *drives* frequency or severity if it's in that final model and moves the relativity by at least 10%. Pure premium = frequency × severity relativity.

| Factor | Frequency | Severity | Pure premium | Drives |
|---|---|---|---|---|
| Flood zone B/C/X | 0.46 | 0.75 | **0.34** | **both** (same direction) |
| Year built 1960–79 / before 1960 / 2000+ | 2.75 / 2.73 / 0.64 | 1.28 / 1.02 / 0.90 | **3.52 / 2.79 / 0.58** | **both**, frequency-dominated |
| Floors 1 / 3+ | 0.89 / 1.13 | 1.12 / 0.89 | **1.00 / 1.01** | both, **offsetting** |
| Occupancy multi-unit / non-residential | (1.39 / 0.94)* | 2.10 / 3.07 | 2.91 / 2.88 | **severity only** |
| Building coverage under $100K / $0 | (1.14 / 1.78)* | 0.69 / 0.28 | 0.79 / 0.50 | **severity only** |
| Elevated | (0.99)* | 0.76 | 0.75 | **severity only** |
| Not primary residence | (0.83)* | 1.12 | 0.93 | **severity only** |
| Post-FIRM | (0.79)* | (0.80)* | 0.63 | neither (absorbed by year built) |
| Non-tidal (Tidewater) | (1.09)* | (1.30)* | 1.42 | neither (geography failed both) |

*\* Not in that final model. The number is from the 9-feature frequency GLM, or the severity model with the factor added, shown for context.*

**Why the drivers differ.** **Frequency** asks *does water reach the building often enough to cause a paid claim?* That's about **hazard and vintage**: the flood zone, and whether the building was built before modern floodplain rules (lower floors relative to flood levels). **Severity** asks *given that water got in, how much was damaged?* That's about **what's in harm's way**:
- the size and use of the building (occupancy);
- how much cover was bought (a claim can't exceed the limit);
- how high the living space sits (elevation keeps water below the main floor, so floods still produce claims for foundations, equipment and enclosures, but smaller ones);
- how much of the value is at ground level (single-storey homes).

Two factors work in opposite directions:
- **Floors:** single-storey buildings claim a little less often but bigger, because all their value is at ground level.
- **Second homes:** they claim less often once zone is held fixed, but bigger.

## 7. Fit checks (`06_residuals.png`)
- **Residuals vs fitted:** roughly flat binned means, so there's no systematic bias for small vs large predicted claims.
- **Gamma vs lognormal:**

| | Gamma GLM | Lognormal (OLS on log dollars) |
|---|---|---|
| AIC, dollar scale (development years) | **197,754** | 197,886 |
| Unseen-year improvement (development LOYO) | **19.2%** | 17.6% |
| Predicted total ÷ actual (training) | 0.99 | 0.54 uncorrected · 1.12 with σ²/2 · 0.99 with smearing |
| Tail fit (Q–Q) | top ~2% of claims bigger than the Gamma expects | top tail too heavy; too few tiny claims |

**The Gamma is preferred.** It has the better AIC and does better on unseen years. It predicts the mean directly, with no back-transform, and its relativities don't depend on the exact distribution shape (only on the mean structure and on the variance growing with the mean). The lognormal's mean needs a correction, and the textbook correction over-shoots by 12%, because the spread differs between groups.

## 8. Performance on held-out data
Following `docs/validation_plan.md`. Baseline = "average claim size for everyone." *Re-based* = predictions scaled to each year's actual average, so the score measures *which* claims are bigger rather than the year's overall level.

| | Final severity GLM |
|---|---|
| LOYO, 16 years: mean re-based Gamma-deviance improvement over baseline | **15.7%** |
| … 13 non-catastrophe years / 3 catastrophe years | 17.1% / 9.8% |
| … median year / claim-weighted mean / pooled over all claims | 14.1% / 11.6% / 13.5% |
| Years beating the baseline | **14 of 16** (worse: 2021 −6.6%, 2022 −6.3%; 151 and 110 claims) |
| Mean Gini per year / pooled | 0.29 / 0.29 |
| Raw A/E summed over all 16 LOYO years | 1.03 |
| **Final holdout** (train 2010–2020, test 2021–2025): improvement / A/E / Gini | **16.7%** / 1.02 / 0.30 |

- **Raw A/E by year** ranges from **0.49 (2010) to 1.77 (2016 Matthew)**. The 2021–2025 holdout years run 0.58–1.40. Like frequency, an average-year severity can't know which storm is coming. Matthew's claims were unusually large, Irene's (0.71) smaller than average.
- **A/E by level** (out of fold, re-based) is within 0.92–1.05 for almost every level. The exceptions: coverage **under $100K (0.81)** and **$0 (0.87)** are over-predicted on unseen years (their claim sizes vary by storm). Non-tidal areas show 1.08 (not in the model).
- **County groups:** 9 of 29 with at least 30 claims have A/E outside 0.8–1.25. **Newport News 1.67** and Virginia Beach 1.16 are higher; Norfolk 0.79, Hampton 0.75, Portsmouth 0.75, Chesapeake 0.77 and York 0.57 lower. This is the main signal left over, and it's why the app will show a credibility-weighted county factor (Theo's Question 7 decision).
- **Lift chart** (`06_lift_chart.png`): the top decile's actual average claim is **4.7× the bottom decile's** (LOYO and holdout alike), and the top decile averages $81K vs a $30K baseline. The model separates the **top** well, but deciles 1–4 are nearly flat (about $16–19K). The bottom decile is **under-predicted** ($17.3K actual vs $12.0K predicted). It's 71% elevated buildings, and the shortfall is in its $100K+ coverage claims (about $19–21K actual vs $12–13K predicted). Contents-only claims there are predicted well. Several below-1 relativities (elevated, B/C/X, newer, 3+ floors) multiply together and undershoot. That suggests an interaction (e.g. elevated × zone) to test in the XGBoost comparison.

## 9. Putting it together (preview of the pure-premium milestone)
Expected loss cost per policy-year = frequency × severity × ICC bridge. For the base building (A/V zone, single-family, 1980–99, 2 floors, $250K+, primary, not elevated): 4.82 per 1,000 × $28,498 × 1.0092 ≈ **$139 per policy-year** in an average year. The statewide average is $197, because the statewide mix includes older buildings and bigger occupancies.

## Limitations
1. **Storm dependence.** Three storms hold two-thirds of claims. Year factors and storm-aware CIs absorb some of this, but the relativities still rest on a small number of hurricanes.
2. **Censoring at policy limits** (1.9% of claims) isn't modelled explicitly. The coverage-band factor captures most of its effect on the average.
3. **Tail shape.** The largest ~2% of claims are bigger than the Gamma expects. That's fine for average severity, not for pricing layers (deductibles, reinsurance).
4. **Order dependence and the second pass**, added after a dry run. Primary residence is borderline (Wald p = 0.047, CI 1.00–1.25).
5. **Geography left out** although out-of-fold county A/E differs (Newport News 1.67, Norfolk 0.79). The app milestone will add a credibility-weighted county factor, clearly labelled.
6. **CPI isn't construction-cost inflation.** The 2024–2025 year factors (1.46, 0.98) may include post-2021 repair-cost inflation as well as storm mix.
7. **Selection peeking:** the Milestone 6 shortlist used all 16 years. All Milestone 8 choices used 2010–2020 only.
8. **Area-level only.** These are averages for groups of similar insured buildings, never a prediction for a specific property.

## Interview-ready sentences
- "For severity I fitted a Gamma GLM with a log link to about 9,500 individual paid claims in 2025 dollars. It gives the same coefficients as weighting cell averages by claim count, but individual claims let me check percentiles, policy limits and residuals."
- "Frequency and severity have different drivers. Flood zone and building age decide how *often* water gets in. Occupancy, coverage limit and elevation decide how *much* damage it does. Floors even work in opposite directions and cancel out in the loss cost."
- "I tested capping large losses at the 95th and 99th percentile with a flat large-loss load. Capping didn't help on unseen years, and at the 95th percentile it shifted cost from big commercial buildings onto single-family homes. NFIP coverage limits already cap the tail, so I modelled uncapped amounts."
- "I chose the Gamma over a lognormal because it predicts the mean directly. The lognormal's back-transformed mean was 46% too low without a correction and 12% too high with the textbook one. The Gamma also had a better AIC and did better on held-out years."
