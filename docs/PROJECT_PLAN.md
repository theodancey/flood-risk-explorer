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
- **Candidate features** (rating-cell level, see `docs/modeling_dataset.md`): flood zone, occupancy, floors, elevated, primary residence, post-FIRM, year built, coverage band, Tidewater (tidal) flag, county group. The week 3 shortlist is in `reports/03_one_way_findings.md`.

## Evaluation
- **Hold out years, not random rows** (full design: `docs/validation_plan.md`, code: `src/validation.py`). **Main comparison: leave-one-year-out cross-validation** (16 folds), scored per year and summarized with and without the 3 catastrophe years. Model development uses only 2010–2020. **Final one-time check:** train 2010–2020, test 2021–2025. *Changed in Task 5:* the plan originally made the 2021–2025 holdout the headline, but that window has no hurricane year, only 7% of claims, and an unusually inland mix of floods.
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
| **2b** ✅ | One-way analysis of risk factors, Tidewater coastal flag, validation design | notebook 04, `reports/03_one_way_findings.md`, `docs/validation_plan.md`, `src/one_way.py`, `src/validation.py` |
| **3** ✅ | Frequency GLM: Poisson, then negative binomial; interpret relativities (shortlist in report 03) | notebook 05, `reports/04_frequency_glm.md`, `src/models/frequency_glm.py`, `models/frequency_glm.json` |
| **4a** ✅ | Severity Gamma GLM on individual claims; large-loss (cap) analysis; frequency vs severity drivers | notebook 06, `reports/05_severity_glm.md`, `src/models/severity_glm.py`, `models/severity_glm.json` |
| **4b** ✅ | Combine into pure premium (frequency × severity × ICC bridge, off-balance); **credibility-weighted county factor** (frequency and severity; moved here from the app milestone at Theo's request); Tweedie cross-check; loss ratios vs NFIP premiums; rating worksheets | notebook 07, report 06, `src/models/pure_premium.py`, `models/pure_premium.json` |
| **5** ✅ | XGBoost frequency (`count:poisson`, exposure + year level in `base_margin`), severity (`reg:gamma`) and Tweedie (p = 1.7) on the same features, rows and year split; randomized search on development LOYO with early stopping on the fold mean; geography re-tested (each variant tuned separately); monotonic constraints | notebook 08, report 07, `src/models/xgb_models.py`, `models/xgb_settings.json` |
| **6** ✅ | Model comparison (Milestone 11): GLM vs XGBoost on 16-year leave-one-year-out and the final holdout, lift/double lift/Gini, segment checks, bootstrap CIs, stability, catastrophe years, explainability checks; second-pass check of the frequency selection (nothing added); NB challenger (not significantly better); hybrid GLM + interactions chosen as the production model. Then explainability (Milestone 12): SHAP vs relativities | notebook 09, report 08, `src/models/compare.py`, `models/production_model.json` + `src/models/production.py` (docs/production_model.md); SHAP notebook 10 |
| **7** | Streamlit app MVP: county picker, metrics, history chart, drivers, *display* the credibility-weighted county factor from `models/production_model.json` (labelled less certain), disclaimer | `app/app.py` running locally |
| **8** | App polish: ZIP lookup, explanations, tests; free hosting (e.g. Streamlit Community Cloud, *needs Theo's OK: account sign-up*) | public app link (if approved) |
| **9** | Write-up: README results, final report, 2-minute demo script, interview Q&A review | finished portfolio repo |

*Stretch goals:* extend to more states (download more policy data), census-tract level, catastrophe-adjusted pricing (e.g. capping storm-year losses and adding a catastrophe load).
