# Flood Risk Explorer
I'm building a flood insurance risk model using FEMA's public data on 2.7 million National Flood Insurance Program claims, starting with Virginia. I calculate the core actuarial measures (claim frequency, severity, and pure premium) and am developing actuarial GLMs and an XGBoost model to compare how well each predicts flood losses by area. So far I've found that Virginia's expected flood loss is about $149 per insured home per year, and that just three storm years caused 69% of all losses since 2010, a pattern that makes flood insurance so hard to price. 

**How often do flood insurance claims happen in an area, how big are they, and what's the expected annual loss per policy?** This project answers those questions with the classic actuarial toolkit, using FEMA's public National Flood Insurance Program (NFIP) data.

- **Data:** OpenFEMA NFIP claims (2.7 million US claims since 1978) and policies (1.67 million Virginia policy terms since 2009).
- **Actuarial foundations:** earned exposure (policy-years), claim **frequency**, claim **severity**, and **pure premium** (expected loss per policy-year), by year and by county.
- **Models (in progress):** Poisson / negative binomial GLMs for frequency, a Gamma GLM for severity, compared against XGBoost, with relativities and SHAP to explain what drives risk. Models are tested on held-out years.
- **App (planned):** a Streamlit app where you pick a county or ZIP code and see its area-level flood risk, expected claim cost, and the reasons behind it.

**Status:** weeks 1–2 of 9 done (data, exploration, actuarial basics, modelling dataset, one-way factor analysis, validation design). See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

> **Disclaimer:** All results are **area-level** estimates built from aggregated public data. They are not predictions, quotes, or risk ratings for any individual property.

## Key findings so far (Virginia, 2010–2025)
- **Average loss cost: about $149 per policy per year.** That's 6.5 paid claims per 1,000 policy-years × a $23,039 average paid claim.
- **Catastrophe-driven:** the worst 3 years (2009 nor'easter, 2016 Matthew, 2011 Irene) produced **69%** of 2009–2025 losses. The *median* year's loss cost ($79) is about half the average.
- **Storms drive frequency, not severity:** claim frequency swings about 40-fold between years, while the average claim stays roughly between $10K and $44K.
- **Risk per policy ≠ total losses:** Salem, Newport News, and the Roanoke area have the highest pure premiums (about $300–$455, inland river flooding in three of them). Virginia Beach has the most dollars: **30%** of losses from **20%** of exposure.
- **Counts are overdispersed** (about 28× Poisson variance), so the frequency model will need a negative binomial or similar.
- **In 2025 dollars** (CPI-U), the loss cost is about **$197** per policy-year. First one-way views of the modelling table: A/V flood zones $292 vs $94 elsewhere; pre-FIRM buildings $297 vs $89 post-FIRM ([`docs/modeling_dataset.md`](docs/modeling_dataset.md), notebook 03).
- **What drives risk (one factor at a time):** building age is the most stable frequency driver (pre-1980 buildings claim about 3× as often as 1980–99 ones, in every one of 16 years). Occupancy, elevation, and second homes drive claim *size*. One-way views can mislead: elevated buildings look riskier overall but are safer within every flood zone (Simpson's paradox), which is why the models are GLMs. A **Tidewater** (tidal) coastal flag from Virginia law separates risk far better than the Census coastline list, revealing two flood regimes: coastal hurricanes vs inland river and flash floods ([`reports/03_one_way_findings.md`](reports/03_one_way_findings.md)).

Details: [`reports/01_data_overview.md`](reports/01_data_overview.md) · [`reports/02_actuarial_basics.md`](reports/02_actuarial_basics.md) · [`reports/03_one_way_findings.md`](reports/03_one_way_findings.md)

![Frequency, severity, and pure premium by year](reports/figures/02_measures_by_year.png)

## Project structure
```
src/download_data.py        download FEMA data (+ Census county names) into data/raw/
src/actuarial.py            earned exposure, frequency, severity, pure premium
src/build_dataset.py        modelling table: rating cells, 2025 dollars, data checks
src/one_way.py              one-way tables: relativities, confidence intervals, credibility, confounding checks
src/validation.py           year-based train/test splits (leave-one-year-out, time holdout)
notebooks/01_first_look     data exploration
notebooks/02_actuarial_basics   Virginia actuarial measures by year and county
notebooks/03_dataset_summary    the modelling table: cell sizes, zero-claim cells, one-way views
notebooks/04_one_way_analysis   risk factors one at a time, stability, confounding, coastal definitions
reports/                    written findings + figures
docs/                       data dictionary, modelling dataset, project plan, validation plan
app/                        Streamlit app (coming)
```
Data files live in `data/` and are never committed. Re-create them with the download script.

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
```

*This product uses FEMA's OpenFEMA API but is not endorsed by FEMA. FEMA may update the data at any time.*
