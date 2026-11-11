# Report 04: The Claim-Frequency GLM (Virginia, 2010–2025)

*Milestone 7, 2026-11-11. Source: `notebooks/05_frequency_glm.ipynb` (code: `src/models/frequency_glm.py`). Figures: `reports/figures/05_*.png`. Tables: `reports/04_frequency_relativities.csv`, `reports/04_frequency_model_steps.csv`, `reports/04_oneway_vs_glm.csv`. Saved model: `models/frequency_glm.json`. Area-level analysis of aggregated public data; nothing here is a prediction for an individual property.*

## In one paragraph
The frequency model is a **Poisson GLM** with a log link and log(earned exposure) as the offset, fitted to 173,363 rating cells (1.48 million policy-years, 9,462 paid claims). It contains **flood zone, year built and number of floors**, plus calendar year as a factor that absorbs the hurricane years. Moderate/low-risk zones (B/C/X) claim **0.46×** as often as A/V zones. Buildings from before 1980 claim about **2.7×** as often as 1980–99 buildings, and buildings from 2000 on about **0.64×**. Most of the other shortlist features look "significant" by AIC and plain likelihood-ratio tests, but fail once we allow for storms, or make unseen years worse. Several one-way effects (post-FIRM, elevation, second homes) shrink or reverse once zone and building age are held fixed. The data are strongly **over-dispersed at the storm level**, so the confidence intervals are cluster-robust: about 3.6× wider than plain Poisson's. On held-out years the model beats a no-features baseline in **14 of 16 years**. The gain is modest (a few percent of deviance), and largest in hurricane years.

## The model in plain English
For each group of similar policies in a given year:

> expected paid claims = policy-years × base rate × (zone factor) × (year-built factor) × (floors factor) × (year factor)

- **Base rate:** 4.82 paid claims per 1,000 policy-years, for an A/V-zone, 1980–99, 2-storey building in an *average* year.
- **Year factor:** soaks up the storms (2011 Irene 4.2×, 2016 Matthew 3.9×, 2012 Sandy 1.8× the average-year level), so storm years don't distort the building factors. A future year gets the **average-year level** (factor 1), because nobody knows in advance whether a hurricane will come.
- **Why only three features:** see "How the model was built" below.

## Relativities (final model, fitted on all 16 years)
Relativity = e^β, the multiplier on the claim rate versus the **base level**, holding the other features fixed. Base levels are the levels with the most exposure (as in Milestone 6).

| Feature | Level | Exposure share | Paid claims | Relativity | 95% CI, storm-aware (recommended) | 95% CI, plain Poisson |
|---|---|---|---|---|---|---|
| Flood zone | **A/V high risk (base)** | 51.0% | 6,798 | 1.00 | | |
| | B/C/X moderate-low risk | 49.0% | 2,664 | **0.46** | 0.34–0.61 | 0.44–0.48 |
| Year built | **1980–1999 (base)** | 30.8% | 1,554 | 1.00 | | |
| | before 1960 | 21.1% | 3,172 | **2.73** | 2.04–3.66 | 2.57–2.90 |
| | 1960–1979 | 28.9% | 4,164 | **2.75** | 2.41–3.13 | 2.59–2.91 |
| | 2000 or later | 19.1% | 572 | **0.64** | 0.51–0.80 | 0.58–0.71 |
| Floors | **2 floors (base)** | 50.5% | 4,724 | 1.00 | | |
| | 1 floor | 32.6% | 2,985 | 0.89 | 0.81–0.99 | 0.85–0.93 |
| | 3+ floors | 16.9% | 1,753 | 1.13 | 0.92–1.40 | 1.07–1.20 |

*Storm-aware* = cluster-robust standard errors by county group × year. *Example:* a B/C/X-zone building from the 1960s with 2 floors: 4.82 × 0.46 × 2.75 ≈ **6.1 claims per 1,000 policy-years** in an average year.

## How the model was built (development years 2010–2020 only)
Starting from flood zone alone, each shortlist candidate was tested against the current model. It was **kept only if** (1) a cluster-robust Wald test (which lets a whole county-year move together, as in a storm) gave p < 0.05, **and** (2) it improved leave-one-year-out (LOYO) deviance on 2010–2020. Full table: `reports/04_frequency_model_steps.csv`; chart: `05_model_build_steps.png`.

| Step | ΔAIC | LRT p | Cluster Wald p | Δ held-out improvement (% pts) | Decision |
|---|---|---|---|---|---|
| Flood zone (vs year only) | −1,644 | <0.0001 | <0.0001 | +0.91 | start |
| + year built | −2,391 | <0.0001 | <0.0001 | **+4.58** (11/11 years better) | **keep** |
| + building coverage | −58 | <0.0001 | 0.081 | −0.42 | drop |
| + occupancy | −82 | <0.0001 | 0.024 | −0.12 | drop |
| + elevated | +2 | 0.81 | 0.98 | −0.91 | drop |
| + post-FIRM | −22 | <0.0001 | 0.31 | −0.02 | drop |
| + primary residence | −36 | <0.0001 | 0.072 | −0.17 | drop |
| + floors | −69 | <0.0001 | 0.020 | +0.08 (8/11) | **keep** (narrowly) |
| Geography A: + Tidewater | −10 | 0.0006 | 0.74 | −1.71 | drop |
| Geography B: + Tidewater + Census coastal | −11 | 0.0005 | 0.93 | −2.03 | drop |
| Geography C: + county group (35 levels) | −1,102 | <0.0001 | <0.0001 | **−6.66** (1/11 better) | drop |
| + Tidewater + zone × Tidewater | −157 | <0.0001 | <0.0001 | −1.43 | drop |

**AIC alone would have kept 10 of 11 candidates; the plain LRT, 10 of 11.** Both assume claims are independent, which is false for flood. Geography is the clearest case. County group cuts training deviance by 1,170, but makes 10 of 11 unseen years worse, because *which* counties flood depends on where each storm goes.

## What changed versus the one-way view (Milestone 6)
| Factor | One-way | GLM (all 9 shortlist features) | Final GLM | Why |
|---|---|---|---|---|
| B/C/X zone | 0.41 | 0.46 | 0.46 | Barely confounded; the strongest factor |
| Built before 1960 | 2.98 | 2.21 | 2.73 | Shares its effect with post-FIRM when both are in |
| Post-FIRM | 0.35 | 0.79 | not in model | Mostly a proxy for building age (Cramér's V 0.81) |
| Elevated | 1.08 | 0.99 | not in model | Simpson's paradox: elevated buildings sit in A/V zones |
| Not primary residence | 1.13 | 0.83 | not in model | Second homes cluster in beach zones; after zone, they claim *less* often (but bigger claims) |
| Coverage under $100K | 1.89 | 1.14 | not in model | Low limits are common inland and on older homes |
| Non-tidal (Tidewater) | 1.37 | 1.09 | not in model | Zone and building mix explain most of it |

*(One-way values are recomputed without the 95 unmatched claims, so a few differ slightly from report 03. Chart: `05_oneway_vs_glm.png`.)*

## Over-dispersion and the choice of intervals
- **Present and strong.** Cameron–Trivedi test p < 0.0001. Pearson φ is about 2.0 on rating cells, but rises to 3.5 when cells are combined by county group and 12 by tidal/non-tidal area, with identical coefficients. Variance that grows when cells are combined means the cells move together: storms.
- **Negative binomial** (α = 3.6): much better AIC (41,863 vs 49,301), and slightly better on unseen development years (6.2% vs 5.6% mean improvement, 8 of 11 folds). But it weakens the zone relativity to 0.61 (it gives big, storm-hit cells less weight), depends on how cells are cut, and its intervals are about as narrow as quasi-Poisson's.
- **Decision:** Poisson (= quasi-Poisson) relativities, which balance exactly to actual claims in every level, with **cluster-robust (county group × year) CIs**. Average CI width vs plain Poisson: quasi-Poisson 1.4×, NB 1.4×, cluster 3.6×, year-cluster 4.5×. NB stays as the challenger for the model-comparison milestone.
- **Effect on conclusions:** zone and year built stay clearly significant under every method. Floors becomes doubtful (3+ floors CI 0.92–1.40).

## Performance on held-out data
Following `docs/validation_plan.md`. Baseline = "average frequency for everyone." Re-based = predictions scaled to each year's actual total, so the score measures *where* claims land rather than *whether* a storm came.

| | Final GLM (3 features) | Full GLM (9 features) |
|---|---|---|
| LOYO, 16 years: mean re-based deviance improvement over baseline | **4.7%** | 3.7% |
| … 13 non-catastrophe years | 2.5% | 2.9% |
| … 3 catastrophe years | **14.1%** | 7.3% |
| … median year | 3.0% | 3.3% |
| Years beating the baseline | 14 of 16 | |
| Mean Gini per year | 0.30 | 0.29 |
| Final holdout (train 2010–2020, test 2021–2025): pooled improvement | **2.2%** | 1.4% |
| Final holdout Gini | 0.26 | 0.24 |
| Raw A/E summed over all 16 LOYO years | 1.00 | |

- **Lift** (`05_lift_chart.png`, ranked by each cell's product of relativities): the top three deciles (essentially A/V buildings from before 1980) claim 11–16 paid claims per 1,000 policy-years in LOYO, against about 2 in the bottom two (pooled LOYO Gini 0.36; 0.28 in the 13 non-cat years). The holdout has the same shape (decile 10 ÷ decile 1 = 4.2×, Gini 0.26). The very top decile does a little worse than decile 9, which is where 3+ floors decides the order.
- **A/E by feature** (out of sample, re-based): about 1.0 for every level of zone, year built and floors. Non-tidal areas show a leftover gap (Tidewater isn't in the model). 16 of 35 county groups have A/E outside 0.67–1.5: geography is the main unexplained signal.
- **Catastrophe years honestly:** raw A/E was **5.6 (2011 Irene), 1.9 (2012 Sandy) and 4.8 (2016 Matthew)**, and 0.1–0.7 in most quiet years (2021–2025 holdout: 0.20). That is what an average-year rate should do. The model can't know a hurricane is coming, only how claims split across cells when one does. Re-based, the model ranks cells *best* in cat years (Gini 0.40 vs 0.28 in normal years).
- **Full vs final:** the 9-feature model is slightly better in 10 of 16 individual years (mostly ordinary ones), but much worse in 2011 and 2016 and on the holdout. The extra features help a little in normal years and hurt when a storm comes.

## Limitations
1. **Storm dependence.** Two-thirds of claims come from three storms. Relativities and validation scores are a small sample of hurricanes; the storm-aware CIs show how small.
2. **Modest discrimination.** A few percent of deviance out of sample. Eight building fields and county can't capture address-level elevation or distance to water.
3. **No geography in the frequency model.** County effects are real in-sample but unstable between storms. The app's area differences will come from each county's mix of zones and building ages until a credibility-weighted geography factor (or XGBoost) is shown to help on unseen years.
4. **Floors is fragile:** kept by the development rule, but not clearly significant with storm-aware intervals.
5. **Average-year pricing** under-predicts storm years and over-predicts quiet ones by design. Catastrophe modelling is out of scope.
6. **Unmatched claims:** 95 paid claims (1%) without matching exposure are excluded, so modelled frequency is about 1% low.
7. **Selection peeking:** the Milestone 6 shortlist used all 16 years (`docs/validation_plan.md`); all Milestone 7 choices used 2010–2020 only.
8. **Area-level only.** Cells are statistical groups, never individual properties.

## Interview-ready sentences
- "I modelled flood claim frequency with a Poisson GLM, using log exposure as an offset so it predicts claims per policy-year, with calendar year as a factor so hurricanes don't distort the rating factors."
- "AIC and likelihood-ratio tests wanted to keep almost every feature, because they assume claims are independent. With cluster-robust tests and leave-one-year-out validation, only flood zone, building age and floors survived. Geography actually made unseen years worse, because which counties flood depends on where the storm goes."
- "The GLM corrected several one-way illusions: elevated buildings looked riskier only because they're concentrated in high-risk zones, and the post-FIRM discount mostly reflected building age."
- "Over-dispersion here is at the storm level, not the cell level. Phi grows when cells are combined. So I reported cluster-robust confidence intervals, about 3.6 times wider than plain Poisson, and kept negative binomial as a challenger rather than a replacement."
