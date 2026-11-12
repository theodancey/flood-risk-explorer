# Flood Risk Explorer
I'm building a flood insurance risk model using FEMA's public data on 2.7 million National Flood Insurance Program claims, starting with Virginia. I calculate the core actuarial measures (claim frequency, severity, and pure premium) and am developing actuarial GLMs and an XGBoost model to compare how well each predicts flood losses by area. So far I've found that Virginia's expected flood loss is about $149 per insured home per year, and that just three storm years caused 69% of all losses since 2010, a pattern that makes flood insurance so hard to price. 

**How often do flood insurance claims happen in an area, how big are they, and what's the expected annual loss per policy?** This project answers those questions with the classic actuarial toolkit, using FEMA's public National Flood Insurance Program (NFIP) data.

- **Data:** OpenFEMA NFIP claims (2.7 million US claims since 1978) and policies (1.67 million Virginia policy terms since 2009).
- **Actuarial foundations:** earned exposure (policy-years), claim **frequency**, claim **severity**, and **pure premium** (expected loss per policy-year), by year and by county.
- **Models (complete):** a Poisson GLM for frequency and a Gamma GLM for severity (with three interactions and a credibility-weighted county factor), compared against XGBoost, with relativities and SHAP to explain what drives risk. Models are tested on held-out years.
- **App (planned):** a Streamlit app where you pick a county or ZIP code and see its area-level flood risk, expected claim cost, and the reasons behind it.

**Status:** modelling complete (weeks 1–6 of 9: data, exploration, actuarial basics, modelling dataset, one-way analysis, frequency and severity GLMs, pure premium with credibility county factors, XGBoost, model comparison, explainability). Everything the app needs is precomputed in `app/data/`. Next: the Streamlit app. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

> **Disclaimer:** All results are **area-level** estimates built from aggregated public data. They are not predictions, quotes, or risk ratings for any individual property.

## Key findings so far (Virginia, 2010–2025)
- **Average loss cost: about $149 per policy per year.** That's 6.5 paid claims per 1,000 policy-years × a $23,039 average paid claim.
- **Catastrophe-driven:** the worst 3 years (2009 nor'easter, 2016 Matthew, 2011 Irene) produced **69%** of 2009–2025 losses. The *median* year's loss cost ($79) is about half the average.
- **Storms drive frequency, not severity:** claim frequency swings about 40-fold between years, while the average claim stays roughly between $10K and $44K.
- **Risk per policy ≠ total losses:** Salem, Newport News, and the Roanoke area have the highest pure premiums (about $300–$455, inland river flooding in three of them). Virginia Beach has the most dollars: **30%** of losses from **20%** of exposure.
- **Counts are overdispersed** (about 28× Poisson variance), so the frequency model will need a negative binomial or similar.
- **In 2025 dollars** (CPI-U), the loss cost is about **$197** per policy-year. First one-way views of the modelling table: A/V flood zones $292 vs $94 elsewhere; pre-FIRM buildings $297 vs $89 post-FIRM ([`docs/modeling_dataset.md`](docs/modeling_dataset.md), notebook 03).
- **What drives risk (one factor at a time):** building age is the most stable frequency driver (pre-1980 buildings claim about 3× as often as 1980–99 ones, in every one of 16 years). Occupancy, elevation, and second homes drive claim *size*. One-way views can mislead: elevated buildings look riskier overall but are safer within every flood zone (Simpson's paradox), which is why the models are GLMs. A **Tidewater** (tidal) coastal flag from Virginia law separates risk far better than the Census coastline list, revealing two flood regimes: coastal hurricanes vs inland river and flash floods ([`reports/03_one_way_findings.md`](reports/03_one_way_findings.md)).
- **Frequency GLM** (Poisson, log(exposure) offset, year factor): **flood zone, year built and floors** survive. B/C/X zones claim 0.46× as often as A/V zones (storm-aware 95% CI 0.34–0.61), and pre-1980 buildings about 2.7× as often as 1980–99 ones. AIC and plain likelihood-ratio tests would keep almost every feature. With storm-aware tests and leave-one-year-out validation, geography and most other factors fail, because their effects change from storm to storm. Out of sample the model beats a no-features baseline in 14 of 16 years ([`reports/04_frequency_glm.md`](reports/04_frequency_glm.md)).
- **Severity GLM** (Gamma, log link, 9,553 individual paid claims, 2025 dollars): claim *size* is driven by what's in harm's way. Multi-unit claims are 2.1× and non-residential 3.1× a single-family home's; low coverage limits mean smaller claims (under $100K: 0.69×); elevated buildings 0.76×; B/C/X zones 0.75×. **Flood zone and building age drive both frequency and severity. Occupancy, coverage and elevation drive only severity. Floors push the two in opposite directions.** Large losses aren't capped: NFIP limits already cap the tail, and capping at the 95th percentile shifted cost from commercial buildings onto homes. Geography failed on unseen years again. Held out, the model beats a no-features baseline in 14 of 16 years (16.7% deviance improvement on the 2021–2025 holdout) ([`reports/05_severity_glm.md`](reports/05_severity_glm.md)).
- **Pure premium** (frequency × severity × ICC bridge × off-balance 0.993 × county factor): the base building costs **$138 per policy-year**. Buildings from 1960–79 cost **3.5×** that, because age raises both how often and how big. Commercial buildings cost 3.1×, B/C/X zones 0.34×. **Bühlmann–Straub credibility** found that county *frequency* differences are pure storm noise (Z = 0), while county *severity* differences are credible (k ≈ 33 claims). The resulting county factors (0.73 York County to 1.78 Newport News) cut county-level error by a third on unseen years. A one-step **Tweedie GLM** scored slightly better overall, but the **double-lift chart** showed frequency × severity was right where they disagreed, so the app will use frequency × severity. Virginia's NFIP **loss ratio was 19%** over 2010–2025 (107% in 2016). Relative to modelled risk, pre-1980 buildings look under-priced ([`reports/06_pure_premium.md`](reports/06_pure_premium.md)).
- **XGBoost challengers** (gradient-boosted trees, tuned by leave-one-year-out on 2010–2020 only, with log(exposure) and the year's level as the offset) beat the GLMs on unseen years:
  - **frequency:** 6.8% vs 5.6% deviance improvement (2021–25 holdout 4.3% vs 2.2%);
  - **severity:** 25.4% vs 19.2% (holdout 23.7% vs 16.7%).

  Frequency XGBoost is still worse in hurricane years. Geography again doesn't help predict claim *frequency* (county group was worse in 10 of 11 unseen years), but it does help predict claim *size*. That matches the credibility result. The XGBoost Tweedie scores best overall but fails the double-lift check, like the GLM Tweedie, so frequency × severity stays the pricing structure ([`reports/07_xgboost.md`](reports/07_xgboost.md)).

- **GLM vs XGBoost, head to head** (identical folds, loss treatment and yardstick; bootstrap confidence intervals):
  - **XGBoost is genuinely more accurate** on unseen years: pure-premium deviance improvement 15.0% vs 10.7%, better in 15 of 16 years (95% CI for the gap: 2.9–5.7 points). Most of the gain is in claim *size*.
  - **The app still uses a GLM**, by a rule written before the results. XGBoost over-predicted newer buildings by about 30% and mis-sized two county groups. Its predictions moved the wrong way for elevation in 78% of cells and changed with the random seed. And it can't produce a rating worksheet.
  - The **production model is a hybrid GLM** that adds the interactions XGBoost found: coverage × occupancy (highly significant), year built × zone, and elevated × zone. It keeps a relativity table and closes about a third of the gap on the 2021–2025 holdout (10.3% vs 8.9%); its 16-year gain (+0.4 points) is small.
  - See [`reports/08_model_comparison.md`](reports/08_model_comparison.md) and [`docs/production_model.md`](docs/production_model.md).

- **Explainability** (SHAP for XGBoost; exact relativities and SHAP for the GLM):
  - Across Virginia's insured buildings, **year built and flood zone** drive expected loss cost most (a typical building's factor moves its prediction ×1.95 and ×1.63 from average), then occupancy and the county factor. Floors barely matter.
  - XGBoost and the GLM **agree** on the top drivers, the weight of geography and the two key interactions. XGBoost's frequency model gets **elevation backwards** for 76% of elevated exposure: it learned *where* elevated buildings are, not what elevation does.
  - Explanations describe the **model, not the world**: coverage band stands in for building value; only the policy-limit cap is causal.
  - **Local explanations** for the app: an exact waterfall from the Virginia average ($197 per policy-year) for any building profile or county, a plain-English sentence and a certainty label, precomputed for all 133 Virginia localities. Property mix explains most county differences; the county factor explains, for example, Newport News's 1.6×.
  - See [`reports/09_explainability.md`](reports/09_explainability.md) and [`app/data/README.md`](app/data/README.md).

Details: [`reports/01_data_overview.md`](reports/01_data_overview.md) · [`reports/02_actuarial_basics.md`](reports/02_actuarial_basics.md) · [`reports/03_one_way_findings.md`](reports/03_one_way_findings.md) · [`reports/04_frequency_glm.md`](reports/04_frequency_glm.md) · [`reports/05_severity_glm.md`](reports/05_severity_glm.md) · [`reports/06_pure_premium.md`](reports/06_pure_premium.md) · [`reports/07_xgboost.md`](reports/07_xgboost.md) · [`reports/08_model_comparison.md`](reports/08_model_comparison.md) · [`reports/09_explainability.md`](reports/09_explainability.md)

![Frequency, severity, and pure premium by year](reports/figures/02_measures_by_year.png)

## Project structure
```
src/download_data.py        download FEMA data (+ Census county names) into data/raw/
src/actuarial.py            earned exposure, frequency, severity, pure premium
src/build_dataset.py        modelling table: rating cells, 2025 dollars, data checks
src/one_way.py              one-way tables: relativities, confidence intervals, credibility, confounding checks
src/validation.py           year-based train/test splits (leave-one-year-out, time holdout)
src/models/frequency_glm.py claim-frequency GLM: fitting, forward selection, over-dispersion, LOYO scoring
src/models/severity_glm.py  claim-severity Gamma GLM: claim table, capping/loads, lognormal check, LOYO scoring
src/models/pure_premium.py  pure premium: freq x sev, off-balance, county credibility, Tweedie, lift/double lift/Gini, loss ratios
src/models/xgb_models.py    XGBoost challengers: encoding, base_margin offset, leakage-free tuning, monotone constraints
src/models/compare.py       GLM vs XGBoost on identical folds: bootstrap CIs, stability, hybrid interaction tests
src/models/production.py    loads the production model (models/production_model.json): predict + rating worksheet
src/explain.py              local explanations for the app: explain_profile, explain_county, sentences, waterfalls; builds app/data
src/global_explain.py       SHAP values: exact for the GLM, TreeSHAP for XGBoost (grouped to features)
tests/test_explain.py       runs every county + random profiles from committed files only
notebooks/01_first_look     data exploration
notebooks/02_actuarial_basics   Virginia actuarial measures by year and county
notebooks/03_dataset_summary    the modelling table: cell sizes, zero-claim cells, one-way views
notebooks/04_one_way_analysis   risk factors one at a time, stability, confounding, coastal definitions
notebooks/05_frequency_glm      frequency GLM: model build, relativities, over-dispersion, held-out evaluation
notebooks/06_severity_glm       severity GLM: distribution, large losses, model build, freq vs sev, held-out evaluation
notebooks/07_pure_premium       pure premium, county credibility, Tweedie comparison, loss ratios, rating worksheets
notebooks/08_xgboost            XGBoost challengers: leakage-free tuning, geography, monotone constraints, preview vs GLMs
notebooks/09_model_comparison   GLM vs XGBoost head to head, significance, stability, explainability, hybrid GLM, production model
notebooks/10_explainability     global drivers (relativities, SHAP beeswarm/dependence), GLM vs XGBoost explanations, causation caveat, waterfalls
models/                     saved models (production_model.json = what the app loads; frequency_glm.json, severity_glm.json, pure_premium.json, xgb_settings.json; XGBoost .ubj files rebuilt with xgb_models.py fit-final)
reports/                    written findings + figures
docs/                       data dictionary, modelling dataset, project plan, validation plan
app/data/                   small precomputed files the app reads (county summary, rating tables, history; committed)
app/                        Streamlit app (coming)
```
Data files live in `data/` and are never committed (`app/data/` holds only small aggregates and model output). Re-create them with the download script.

## How to run (Windows, PowerShell)
```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\check_setup.py
.venv\Scripts\python.exe src\download_data.py all      # about 35 minutes, ~270 MB
.venv\Scripts\python.exe src\actuarial.py              # self-test + county-year table
.venv\Scripts\python.exe src\build_dataset.py          # modelling table + data checks (~20 s)
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace notebooks\02_actuarial_basics.ipynb
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace notebooks\03_dataset_summary.ipynb
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace notebooks\04_one_way_analysis.ipynb
.venv\Scripts\python.exe src\explain.py build       # rebuild app/data from the modelling table (~10 s)
.venv\Scripts\python.exe tests\test_explain.py      # explanations for every county, committed files only (~2 min)
```

*This product uses FEMA's OpenFEMA API but is not endorsed by FEMA. FEMA may update the data at any time.*
