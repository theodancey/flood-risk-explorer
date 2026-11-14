# Flood Risk Explorer: the Streamlit app

*Milestone 14 (2026-11-13), extended in Milestone 15 (2026-11-13): county map, address lookup, "what if?" calculator, plain-language pass. Milestone 16 (2026-11-13): fixes from testing the live app (arrows, map scrolling and borders, address hand-over). Deployed on Streamlit Community Cloud (every push to `main` redeploys it).*

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
- To set up a fresh environment with **only** what the app needs:
  `.venv\Scripts\python.exe -m pip install -r app\requirements.txt`.

Run the automated tests (no internet needed; about 3-6 minutes for the app, seconds for the lookup):

```powershell
.venv\Scripts\python.exe tests\test_app.py
.venv\Scripts\python.exe tests\test_lookup.py
```

## The pages
| Page | URL | What it does |
|---|---|---|
| **County Explorer** (default) | `/` and `/county` | **First screen:** headline, how-to, the **county map** (opens comparing each county with its own state, or shade by expected yearly loss; thick state borders and VA/NC/MD labels; the scroll wheel scrolls the page, not the map; hover; click a county for a small summary card with a "See details below" link), **Find my county** address search, the short disclaimer and a credit line. **Then:** state + searchable county list, the **hero box** (model estimate for a typical year vs the state average, next to the actual 2010-2025 average, with a sentence naming the storms when they differ by more than 1.5×), a one-line summary, and three tabs: **Overview** (chance of a claim, typical claim size, two plain certainty signals, the mix of insured buildings), **Claims history** (by year, big storms labelled), **Why it differs** (waterfall). |
| **Property Profile Calculator** | `/calculator` | Optional **address lookup** that fills in state, county, flood zone and post-FIRM (and the age group from the year built). Building inputs with help text. Result: expected yearly loss vs the state average **and** vs a typical lower-risk (B/C/X) insured building, chance of a claim, typical claim size, local-data signals, a waterfall of what raises or lowers it, and **"What if?"** (raising the building). |
| **How It Works** | `/how-it-works` | "In short", data sources, methods with every term defined (policy-year, pure premium, GLM, interactions, Shapley values), key findings (NC numbers labelled), how the certainty signals work (credibility Z, Bühlmann-Straub, A/E), limitations, future work, About. |

Every page has the wordmark, a **short disclaimer at the top** and a **footer** with the full
disclaimer, Theo Dancey's name, the GitHub link, the data source and the last-updated date.

## Address lookup (free public services, no sign-up)
| Step | Service | What it gives |
|---|---|---|
| 1 | [U.S. Census Geocoder](https://geocoding.geo.census.gov/geocoder/) `geographies/onelineaddress` | coordinates, state, county, town/city |
| 2 | FEMA National Flood Hazard Layer [MapServer](https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer), layer 28 "Flood Hazard Zones" (point query) | `FLD_ZONE`, `ZONE_SUBTY` → A*/V* = high risk, X/B/C = lower risk; D, unmapped or nothing → the user chooses |
| 3 | Same service, layer 22 "Political Jurisdictions" | the NFIP community ID (CID) at the point |
| 4 | `app/data/firm_dates.csv` (OpenFEMA NFIP Community Status Book snapshot) | the community's initial FIRM date; fallback match by town/county name |

**Pre/post-FIRM rule (FEMA):** post-FIRM = built after Dec 31, 1974, or on/after the initial FIRM
date, whichever is later. With only a year, the cutoff year itself is "not sure" (unless the cutoff
is Jan 1) and the user chooses.

**Safeguards:** 6-second timeouts and 2 retries; results cached (failures never cached; geocodes
only 10 minutes, in memory); every failure falls back to manual inputs with a plain message; a clear
message outside VA/NC/MD. The address is never logged, printed, written to disk or put in the app's
URL, and error messages never repeat it. The page says it's not an official flood zone determination
(link to msc.fema.gov) and that the result is still an average for similar insured buildings.

## Files
```
app/app.py                       entry point: page setup + menu (st.navigation); /county alias
app/common.py                    shared: data loading, formatting, plain labels, certainty signals, waterfall,
                                 county map, lookup wrappers (cached), disclaimer/footer, CSS + wordmark
app/lookup.py                    address lookup (Census Geocoder, FEMA NFHL, FIRM rule); no Streamlit inside
app/views/county_explorer.py     page 1 (default)
app/views/profile_calculator.py  page 2
app/views/how_it_works.py        page 3
app/requirements.txt             streamlit, pandas, numpy, pyarrow, plotly, requests
app/data/                        the precomputed data (below)
.streamlit/config.toml           theme (navy accent) and settings
tests/test_app.py                AppTest: every county, profiles, map, address flow (mocked), what if, routes
tests/test_lookup.py             lookup module with fake web answers; real connections blocked
```

## Data the app reads (only these, about 1.7 MB)
| File | Used for |
|---|---|
| `app/data/county_summary.csv` | county list, estimates, certainty inputs (Z, history vs model), mix, actual 2010-2025 average |
| `app/data/county_history.csv` | the claims-history chart and the storm sentence |
| `app/data/portfolio.parquet` | each state's insured buildings (averages, waterfalls, the lower-risk comparison) |
| `app/data/rating_factors.json` | the production GLM as lookup tables |
| `app/data/state_averages.json` | state averages and mix shares |
| `app/data/counties_va_nc_md.geojson` | county outlines for the map (Census 2023 cartographic boundaries, simplified; 0.34 MB) |
| `app/data/states_va_nc_md.geojson` | state outlines for the thicker state borders (same source and scale; 0.12 MB; Milestone 16) |
| `app/data/firm_dates.csv` | each NFIP community's initial FIRM date (OpenFEMA Community Status Book; 76 KB) |

The app **never** reads `data/raw/` or `data/processed/`; `tests/test_app.py` logs every file the app
opens and checks this. Column definitions and how to rebuild each file: [`data/README.md`](data/README.md).
