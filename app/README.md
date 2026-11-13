# Flood Risk Explorer: the Streamlit app

*Milestone 14 (2026-11-13). Runs locally; public hosting is planned for Milestone 16.*

A three-page web app showing **area-level** expected flood insurance losses for every county and
independent city in Virginia (133), North Carolina (100) and Maryland (24), built from public FEMA
NFIP data, 2010-2025.

> Results are model-based, area-level averages from public FEMA NFIP data. They are not a prediction
> for any specific property, not an insurance quote and not financial advice. This project is not
> affiliated with or endorsed by FEMA.

## Run it locally (Windows, PowerShell)
From the project folder (`flood-risk-explorer`), using the project's virtual environment:

```powershell
.venv\Scripts\python.exe -m streamlit run app\app.py
```

Your browser should open at <http://localhost:8501> (if not, open that address yourself). Stop the
app with **Ctrl+C** in the PowerShell window.

- Run it **from the project folder** so Streamlit picks up the theme in `.streamlit/config.toml`.
  (The app finds its data from its own location, so it works from anywhere, just without the theme.)
- The project's `requirements.txt` already includes everything. To set up a fresh environment with
  **only** what the app needs: `.venv\Scripts\python.exe -m pip install -r app\requirements.txt`.

Run the automated tests (about 1 minute):

```powershell
.venv\Scripts\python.exe tests\test_app.py
```

## The pages
| Page | What it does |
|---|---|
| **County Explorer** | Pick a state, then a county (type to search). Shows the expected annual loss per policy-year and how it compares with the state average, claim frequency and severity, a plain-English summary, the **certainty label** and why, the county's **mix of insured buildings** vs the state, its **own claims history by year** (major storms labelled), and a **"why" waterfall**: how each part of the property mix moves the county away from the state average. |
| **Property Profile Calculator** | Describe a type of building (state, county, flood zone, year built, post-FIRM, elevated, occupancy, building coverage, floors, primary residence; each with help text). Shows expected claim frequency, claim severity and pure premium for the average insured building of that type, and an interactive waterfall of each characteristic's effect from the state average (exact Shapley values). Unusual combinations get a warning, rare ones a note, and impossible ones are prevented (a non-residential building can't be a primary residence). |
| **How It Works** | Data sources (with links and download dates), the methods in plain English (exposure, frequency, severity, pure premium, GLMs vs XGBoost, leave-one-year-out and multistate validation), key findings, limitations, future work, and an About section. |

Every page shows the disclaimer at the top, and the sidebar repeats it.

## Files
```
app/app.py                       entry point: page setup + menu (st.navigation)
app/common.py                    shared: data paths, cached loading, formatting, colours, disclaimer, waterfall chart
app/views/county_explorer.py     page 1
app/views/profile_calculator.py  page 2
app/views/how_it_works.py        page 3
app/requirements.txt             only the app's libraries: streamlit, pandas, numpy, pyarrow, plotly
app/data/                        the precomputed data (below)
.streamlit/config.toml           theme (project root, where Streamlit looks for it)
tests/test_app.py                automated tests (streamlit.testing.v1.AppTest)
```
The app also imports `src/explain.py`, which turns the rating tables into county and profile
explanations. When the app loads it, `explain.py` uses numpy and pandas only (the full model code,
which needs statsmodels, is imported only when rebuilding `app/data/`).

## Data the app reads (only these, about 1.3 MB)
| File | Used for |
|---|---|
| `app/data/county_summary.csv` | county list, summary sentence, certainty label and note, history vs model |
| `app/data/county_history.csv` | the claims-history chart (county × year: policy-years, paid claims, losses in 2025 $) |
| `app/data/portfolio.parquet` | each state's insured buildings (the starting point for averages and waterfalls) and each county's property mix |
| `app/data/rating_factors.json` | the production GLM as lookup tables (base rates, multipliers, state factor, loads) |
| `app/data/state_averages.json` | state averages used on the How It Works page |

The app **never** reads `data/raw/` or `data/processed/` (they're not in Git and won't exist on a
host), and never loads model pickles or XGBoost files. `tests/test_app.py` logs every file the app
opens and checks this. Column definitions are in [`data/README.md`](data/README.md); rebuild the files
with `.venv\Scripts\python.exe src\explain.py build` whenever the model changes.
