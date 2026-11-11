# Project Plan: Flood Risk Explorer

*Written 2026-11-10, end of week 1. A living plan; changes are logged in `PROGRESS.md`.*

## The problem and who cares
Flood is the most common and costly natural disaster in the US, yet most homeowners policies don't cover it. Most flood insurance comes from FEMA's National Flood Insurance Program (NFIP). This project asks: **for a given area, how often do flood claims happen, how big are they, what is the expected annual loss per policy, and why?**
- **Insurers and reinsurers** price and manage flood risk by area. Private flood insurance is growing alongside the NFIP.
- **Homeowners and buyers** want to understand the flood risk of a neighborhood.
- **Local governments and planners** decide where to invest in flood protection.

## Data
FEMA OpenFEMA, free and public: **NFIP claims v3** (2.7M US claims since 1978) and **NFIP policies v3** (1.67M Virginia policy terms, 2009 onward). We start with **Virginia, 2010–2025**, aggregated to **county × year** (later ZIP / census tract). Exposure = earned policy-years. Losses = total paid (building + contents + ICC). See `docs/data_dictionary.md` and `reports/01`–`02`.

## Modeling approach
1. **Frequency GLM.** A **GLM** (generalized linear model) is a regression adapted to non-normal outcomes. We model paid claims per county-year with a **Poisson** GLM using **log(exposure) as an offset**, so the model predicts a *rate* per policy-year. We then compare it with a **negative binomial** model, because notebook 02 found the counts are about 28× more variable than Poisson allows.
2. **Severity GLM.** A **Gamma** GLM (log link) for the average paid claim, weighted by the number of claims.
3. **Pure premium** = predicted frequency × predicted severity. We'll also try a single **Tweedie** GLM as a cross-check.
4. **XGBoost comparison.** Gradient-boosted trees with Poisson (frequency), Gamma (severity), and Tweedie objectives, using the same features. The question: does the extra flexibility beat the GLMs on unseen years, and is the gain worth the lost transparency?
5. **Explainability.** For the GLMs, **relativities**: e.g. "V-zone share +10 points → frequency ×1.3." For XGBoost, **SHAP** values (how much each feature pushes a prediction up or down). Each area gets a plain-English "why."
- **Candidate features** (area level): flood-zone mix, share of pre-FIRM / elevated / primary-residence buildings, coastal vs inland, average coverage and deductible, Community Rating System class, past claim history.

## Evaluation
- **Hold out years, not random rows.** Train on **2010–2020** and test on **2021–2025**, so we test the way pricing is used: predicting the future. Because a single storm can dominate a test period, we'll also run **leave-one-year-out** checks and report results with and without the catastrophe years.
- **Metrics:** Poisson / Gamma **deviance** (a goodness-of-fit score for these distributions; lower is better); **actual vs expected** totals overall and by county; **lift charts and Gini** (does the model rank risky areas above safe ones?).
- **Sanity checks:** predicted statewide losses match actual on average; relativities make sense (high-risk zones > low-risk zones); small counties are not over-fit.

## The web app (Streamlit)
Enter a **county** (later a ZIP code) to see the area's claim frequency, average claim size, expected annual loss per policy (pure premium), how it compares with the state, the top drivers (relativities / SHAP), and its loss history by year with storms labeled. A prominent **disclaimer**: area-level estimates from aggregated public data, *not* a quote or a prediction for any individual property.

## Limitations
- **Area-level only.** The data is redacted and aggregated, so results describe areas, never individual homes.
- **Catastrophe years dominate.** The worst 3 of 17 years produced 69% of losses, so any 15-year window is a small sample of storms. Estimates are uncertain, especially for rare big events. Proper catastrophe modelling (hurricane simulations) is out of scope.
- **Data quality:** exposure only from 2009 (2009 estimated); nominal dollars; FEMA's county coding can be imperfect; many building fields blank; rating codes changed with Risk Rating 2.0 (Oct 2021); recent years may still gain late claims.
- **Who buys NFIP** isn't random. Insured homes are concentrated in high-risk zones, and the policy count has fallen about 23% since 2013. Results describe the *insured* population.

## Week-by-week plan (weeks 2–9, a few hours per week)
| Week | Goal | Deliverable |
|---|---|---|
| **2** ✅ | Clean data and build the modelling table: rating cells (county × year × 8 building/policy features), losses in 2025 dollars (CPI-U), coastal flag, data checks | `src/build_dataset.py`, `data/processed/va_model_data.parquet`, `docs/modeling_dataset.md`, notebook 03 |
| **3** | Frequency GLM: Poisson, then negative binomial; interpret relativities | notebook 04, frequency section of report 03 |
| **4** | Severity Gamma GLM; combine into pure premium; Tweedie cross-check | notebook 05, report 03 |
| **5** | XGBoost frequency/severity/Tweedie with the same features and year split; light tuning | notebook 06 |
| **6** | Evaluation and explainability: holdout years, leave-one-year-out, lift/Gini, SHAP vs relativities | notebook 07, report 04 (model comparison) |
| **7** | Streamlit app MVP: county picker, metrics, history chart, drivers, disclaimer | `app/app.py` running locally |
| **8** | App polish: ZIP lookup, explanations, tests; free hosting (e.g. Streamlit Community Cloud, *needs Theo's OK: account sign-up*) | public app link (if approved) |
| **9** | Write-up: README results, final report, 2-minute demo script, interview Q&A review | finished portfolio repo |

*Stretch goals:* extend to more states (download more policy data), census-tract level, catastrophe-adjusted pricing (e.g. capping storm-year losses and adding a catastrophe load).
