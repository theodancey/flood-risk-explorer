# Flood Risk Explorer
I'm building a flood insurance risk model using FEMA's public data on 2.7 million National Flood Insurance Program claims, for Virginia, North Carolina and Maryland. I calculate the core actuarial measures (claim frequency, severity, and pure premium) and am developing actuarial GLMs and an XGBoost model to compare how well each predicts flood losses by area. So far I've found that Virginia's expected flood loss is about $149 per insured home per year, and that just three storm years caused 69% of all losses in 2009–2025, a pattern that makes flood insurance so hard to price. 

**How often do flood insurance claims happen in an area, how big are they, and what's the expected annual loss per policy?** This project answers those questions with the classic actuarial toolkit, using FEMA's public National Flood Insurance Program (NFIP) data.

- **Data:** OpenFEMA NFIP claims (2.7 million US claims since 1978) and policies (4.6 million policy terms since 2009 in Virginia, North Carolina and Maryland; Virginia first, the other two added in Milestone 13).
- **Actuarial foundations:** earned exposure (policy-years), claim **frequency**, claim **severity**, and **pure premium** (expected loss per policy-year), by year and by county.
- **Models (complete):** a Poisson GLM for frequency and a Gamma GLM for severity (with three interactions and a state factor), compared against XGBoost, with relativities and SHAP to explain what drives risk. Models are tested on held-out years and on a held-out state.
- **App (Milestones 14–15, on Streamlit Community Cloud):** a Streamlit app with a county map, an address search, a county explorer and a building-type calculator with a "what if?" section. Each shows area-level expected flood losses, how they compare with the state, how much local data backs them up, and the reasons behind them. See [Try the app](#try-the-app).

**Status:** modelling complete for three states (data, exploration, actuarial basics, modelling dataset, one-way analysis, frequency and severity GLMs, pure premium with credibility, XGBoost, model comparison, explainability, and the Milestone 13 extension to North Carolina and Maryland). The Streamlit app (Milestone 14) covers all 257 counties; Milestone 15 added the map, address lookup, "what if?" calculator and a plain-language pass. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

> **Disclaimer:** All results are **area-level** estimates built from aggregated public data. They are not predictions, quotes, or risk ratings for any individual property.

## Try the app
**Live app:** *(link to be added: the Streamlit Community Cloud address)*

**Screenshots:** *(placeholders: add `reports/figures/app_home.png`, `app_county.png` and `app_calculator.png`)*

Or run it on your own computer. From the project folder, in PowerShell:

```powershell
.venv\Scripts\python.exe -m streamlit run app\app.py
```

Then open <http://localhost:8501> if it doesn't open by itself (stop it with Ctrl+C).

**Features**
- **County map (first screen):** every county in Virginia, North Carolina and Maryland, shaded by expected yearly loss per insured building or by how it compares with its state average. Hover for details; click a county to open it.
- **Find my county:** type a street address to jump to its county (U.S. Census Geocoder).
- **County Explorer:** the model's estimate for a typical year next to the actual 2010–2025 average (with the storms behind any big gap), how much local claims data backs it up, the mix of insured buildings, the county's own claims history with major storms labelled, and why it differs from the state. Also at `/county`.
- **Property Profile Calculator:** describe a kind of building (or look up an address to fill in the state, county, flood zone from FEMA's flood map, and pre/post-FIRM from the area's first flood map date). See the expected yearly loss vs the state average and vs a typical lower-risk building, a chart of what raises or lowers it, and a **"What if?"** estimate of raising the building.
- **How It Works:** a three-line summary, data sources, methods with every term defined, key findings, how the certainty signals work, limitations and future work.
- Plain language on the main pages (technical detail in "Learn more" sections), a short disclaimer at the top of every page and the full one in the footer.

The app reads only the small files in `app/data/` (about 1.7 MB) and needs only streamlit, pandas, numpy, pyarrow, plotly and requests (`app/requirements.txt`). The address lookup uses only free public services with no sign-up. Details: [`app/README.md`](app/README.md). Tests: `.venv\Scripts\python.exe tests\test_app.py` and `tests\test_lookup.py` (no internet needed).

## Key findings so far (Virginia 2010–2025; three states since Milestone 13)
- **Three states** (Milestone 13: Virginia, North Carolina, Maryland; 4.1 million policy-years, 51,894 paid claims):
  - **North Carolina is a different kind of flood state:** a loss cost of **$988** per policy-year vs about **$198** in Virginia and Maryland (claims 3x as often, 1.7x as big), driven by Irene, Matthew, Florence and Helene.
  - **The Virginia model, applied unchanged, ranks risk almost as well out of state** (Gini 0.49–0.51 vs 0.53), but it underestimated NC's level **6.6-fold** (Florence 2018: 46x). Maryland's level was right (0.99).
  - **Refit with a state factor:** NC 6.2x Virginia, Maryland 1.0x. The building relativities barely move, and the big ones (zone, age, occupancy, elevation) point the same way in every state, which is evidence they're real. State-specific effects and the county credibility factor both made unseen years worse, so neither is used: in NC a county's history is mostly one storm's footprint.
  - **XGBoost** is about 4 points more accurate on average but fails the pre-stated rule on catastrophe years and segment balance, so **the GLM stays in production**.
  - See [`reports/10_multistate.md`](reports/10_multistate.md).

*The findings below are the Virginia milestones (1–12):*
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

Details: [`reports/01_data_overview.md`](reports/01_data_overview.md) · [`reports/02_actuarial_basics.md`](reports/02_actuarial_basics.md) · [`reports/03_one_way_findings.md`](reports/03_one_way_findings.md) · [`reports/04_frequency_glm.md`](reports/04_frequency_glm.md) · [`reports/05_severity_glm.md`](reports/05_severity_glm.md) · [`reports/06_pure_premium.md`](reports/06_pure_premium.md) · [`reports/07_xgboost.md`](reports/07_xgboost.md) · [`reports/08_model_comparison.md`](reports/08_model_comparison.md) · [`reports/09_explainability.md`](reports/09_explainability.md) · [`reports/10_multistate.md`](reports/10_multistate.md)

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
src/multistate.py           Milestone 13: transfer test, 3-state refit + validation, consistency by state, XGBoost check, production save
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
notebooks/11_multistate         VA + NC + MD: transfer test, state factor, validation by state, consistency of relativities, county factor, XGBoost check
models/                     saved models (production_model.json = what the app loads; frequency_glm.json, severity_glm.json, pure_premium.json, xgb_settings.json; XGBoost .ubj files rebuilt with xgb_models.py fit-final)
reports/                    written findings + figures
docs/                       data dictionary, modelling dataset, project plan, validation plan
app/data/                   small precomputed files the app reads (county summary, rating tables, history; committed)
app/app.py, app/views/      Streamlit app (Milestone 14): County Explorer, Profile Calculator, How It Works (see app/README.md)
tests/test_app.py           app tests (AppTest): every county, profiles, edge cases, map, address lookup (mocked), what if, no raw/processed reads
tests/test_lookup.py        address lookup tests with fake web answers (no internet)
```
Data files live in `data/` and are never committed (`app/data/` holds only small aggregates and model output). Re-create them with the download script.

## Future work
- **Bring geography back through storm-independent features** rather than county claims history: for example ground elevation, distance to the coast, and each area's share of exposure in FEMA high-risk zones. County history failed on unseen years (in North Carolina a county's history is mostly one hurricane's footprint), while features describing where water can go shouldn't depend on which storms happened to hit.
- ZIP-code or census-tract detail, more states, and catastrophe-adjusted pricing (a catastrophe load instead of raw storm years).

## How to run (Windows, PowerShell)
```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\check_setup.py
.venv\Scripts\python.exe src\download_data.py all      # claims + VA/NC/MD policies, about 1-1.5 hours, ~400 MB
.venv\Scripts\python.exe src\actuarial.py              # self-test + county-year table
.venv\Scripts\python.exe src\build_dataset.py          # Virginia modelling table + data checks (~20 s)
.venv\Scripts\python.exe src\build_dataset.py --states VA NC MD   # 3-state table (~2 min)
.venv\Scripts\python.exe src\multistate.py transfer   # Milestone 13, one step at a time (each prints memory use)
.venv\Scripts\python.exe src\multistate.py refit main
.venv\Scripts\python.exe src\multistate.py refit consistency
.venv\Scripts\python.exe src\multistate.py refit candidate
.venv\Scripts\python.exe src\multistate.py refit county
.venv\Scripts\python.exe src\multistate.py refit validate
.venv\Scripts\python.exe src\multistate.py xgb        # ~10 min
.venv\Scripts\python.exe src\multistate.py xgb-decision
.venv\Scripts\python.exe src\multistate.py production
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace notebooks\02_actuarial_basics.ipynb
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace notebooks\03_dataset_summary.ipynb
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace notebooks\04_one_way_analysis.ipynb
.venv\Scripts\python.exe src\explain.py build       # rebuild app/data from the modelling table (~10 s)
.venv\Scripts\python.exe tests\test_explain.py      # explanations for every county, committed files only (~2 min)
.venv\Scripts\python.exe tests\test_app.py          # Streamlit app tests (~1 min)
.venv\Scripts\python.exe -m streamlit run app\app.py  # the app, at http://localhost:8501
```

*This product uses FEMA's OpenFEMA API but is not endorsed by FEMA. FEMA may update the data at any time.*
