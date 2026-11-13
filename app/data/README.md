# app/data: precomputed files for the Streamlit app

*Built by `src/explain.py build` (Milestone 12; three states since Milestone 13), plus two Milestone 15 files built by their own scripts (below). About 1.7 MB in total, committed to Git.*

The app needs everything here plus `models/production_model.json`. It does **not** need `data/processed/`, which isn't in Git and won't exist on a free host. Each file is either an aggregate (county, county × year, or building-type × county exposure) or model output. There's no policy-level or claim-level row.

**Area level only:** every number is an average for a group of insured buildings in an average year, never a prediction or quote for an individual property. Show the disclaimer (`state_averages.json` → `disclaimer`) next to every number.

**States:** Virginia (133 counties/cities), North Carolina (100 counties), Maryland (24: 23 counties + Baltimore City). Each county is compared with **its own state's** average.

**No county factor (since Milestone 13).** The production model no longer adjusts any county by its own claim history, because on three states that made unseen years less accurate (`reports/10_multistate.md`). A county's estimate = its mix of insured buildings × its state's level. All `county_factor` values are 1.0. The county's credibility and how its history compares with the model are still given, and they set the **certainty** label.

## Files
| File | Size | One row per | What it's for |
|---|---|---|---|
| `county_summary.csv` | ~320 KB | county/city with insured exposure (257) | the county page: expected frequency, severity, pure premium, ratio to its state, certainty, history vs model, property mix, why it differs, plain-English summary |
| `rating_factors.json` | ~53 KB | n/a | the production model as lookup tables (incl. the state factor), so the app can rate a custom profile |
| `portfolio.parquet` | ~650 KB | building type (7 model features + post-FIRM) × county | insured policy-years 2010–2025: each state's "background" for the profile waterfall, and each county's mix |
| `county_history.csv` | ~200 KB | county × year (2010–2025) | the loss-history chart: state, policy-years, paid claims, losses in 2025 $ (actual experience) |
| `state_averages.json` | ~7 KB | n/a | averages for each state and all three, the state factor, comparison rule, certainty rule, disclaimer |
| `counties_va_nc_md.geojson` | ~340 KB | county with a Census outline (257) | the county map: outlines, `id` = 5-digit FIPS. Built by `src/build_county_geojson.py` |
| `firm_dates.csv` | ~76 KB | NFIP community (1,084) | the address lookup: each community's initial FIRM date. Built by `src/build_firm_dates.py` |

`virginia_averages.json` (Milestone 12) was replaced by `state_averages.json`.

## Columns
**`county_summary.csv`** (sorted by state, then expected pure premium, highest first)
- **Identity:**
  - `county`: 5-digit FIPS as text (VA "51…", NC "37…", MD "24…");
  - `state`;
  - `name`: Census name. 51515 is the former Bedford city, part of Bedford County since 2013.
- **Experience 2010–2025:**
  - `policy_years_2010_2025`;
  - `avg_policies_per_year`;
  - `paid_claims_2010_2025`;
  - `actual_loss_cost_2010_2025`: noisy, dominated by whether a storm hit.
- **Model (average year, 2025 $):**
  - `expected_frequency`, `expected_severity`, `expected_pure_premium`;
  - `ratio_to_state`: vs the county's own state average;
  - `ratio_to_three_states`.
- **Credibility and certainty:**
  - `county_factor`: 1.0 everywhere (not applied, see above);
  - `Z_sev`: the Bühlmann–Straub severity credibility the county's own claim history earns, 0–1;
  - `Z_freq`: 0;
  - `history_vs_model`: the county's 2010–2025 claims vs the model, each year at its own storm level (1.0 = agrees);
  - `certainty`:
    - High = Z ≥ 0.8 and history within 0.8–1.25× the model;
    - Low = no claims, Z < 0.5, or history outside 0.5–2.0×;
    - Medium = everything else.
- **Why it differs:**
  - `ratio_to_state` = `mix_effect` × `county_factor_effect`, exactly; the county factor effect is 1.0 now.
  - `mix_effect` = the product of the seven `mix_index_<feature>` values × `overlap_effect`.
- **Mix (% of the county's insured policy-years):** `pct_av_zone`, `pct_elevated`, `pct_pre_firm`, `pct_built_pre_1980`, `pct_multi_or_nonres`, `pct_not_primary`, `pct_cover_250k_plus`, `pct_cover_under_100k`, `pct_one_floor`.
- **Text:**
  - `summary`: a plain-English sentence that names the state;
  - `uncertainty_note`: explains the certainty label and, where the history disagrees with the model, by how much.

**`rating_factors.json`**
- `levels`, `bases`, `frequency.base_per_policy_year`, `severity.base_claim_size` (includes the large-loss load, 1.000).
- `frequency/severity.log_relativity[feature]`, and `interactions` (two-way log tables).
- `state`: the levels and the frequency and severity **log state factors** vs Virginia.
- `loads`: ICC bridge, `off_balance_by_state`.
- `county_factors` (all 1.0), `county_factor_applied` (false), `county_credibility` (Z, claims, A/E, names).
- `labels`, `disclaimer`.

**How to rate a profile from this file:**
1. Pure premium = base frequency × base severity.
2. Multiply by exp(sum of the frequency and severity log relativities, including interactions and the state's log state factors).
3. Multiply by the ICC bridge × **that state's** off-balance × the county factor (1.0).

`tests/test_explain.py` checks that this equals `production.predict` exactly.

**`state_averages.json`** (2010–2025 mix; average year; 2025 $):

| | Expected pure premium per policy-year | On the 2025 mix | Actual 2010–2025 |
|---|---:|---:|---:|
| Virginia | $197.33 | $217.52 | $198.13 |
| North Carolina | $987.85 | $1,027.61 | $991.47 |
| Maryland | $198.30 | $211.24 | $199.12 |
| All three | $572.26 | | $574.40 |

Each state entry also holds its mix shares, mean relativities, the certainty rule and the disclaimer.

## Using it
```python
import sys; sys.path.insert(0, "src")
import explain
r = explain.explain_county("Dare County, NC")     # FIPS, "Name, ST", or a name unique across the states
r["sentence"], r["waterfall"], r["uncertainty_note"], r["certainty"]
p = explain.explain_profile({"county": "51810", "flood_zone_group": "A/V high risk",
                             "construction_band": "1960-1979", "floors_group": "1"})
q = explain.explain_profile({"flood_zone_group": "A/V high risk"}, state="NC")   # anywhere in NC
explain.plot_waterfall(p)                          # matplotlib figure
```
- Missing profile features default to the model's base levels.
- Ambiguous names raise an error listing the options: for example "Montgomery County" exists in VA, NC and MD, and Fairfax and Richmond are each both a county and a city. Use "Name, ST" or `state=` to pick one.

**`counties_va_nc_md.geojson`** (Milestone 15)
- Source: U.S. Census Bureau, 2023 Cartographic Boundary File, counties, 1:5,000,000 (`cb_2023_us_county_5m.zip`, downloaded 2026-11-13 from www2.census.gov/geo/tiger/GENZ2023/shp/), public domain.
- Kept VA (51), NC (37), MD (24); coordinates rounded to 3 decimals (about 100 m); duplicate points and specks of islands dropped.
- 257 outlines. Galax city, VA (51640) has an outline but no insured exposure in the data; the former Bedford city (51515, part of Bedford County since 2013) has data but no outline, so it isn't on the map (it's still in the county list).

**`firm_dates.csv`** (Milestone 15)
- Source: OpenFEMA API, NFIP Community Status Book (`NfipCommunityStatusBook` v1), downloaded 2026-11-13. Full download kept in `data/raw/community_status_book/` (not in Git).
- Columns: `cid` (6-digit NFIP community ID), `state`, `community` (e.g. "RICHMOND, CITY OF"), `county`, `initial_firm` (date of the community's first Flood Insurance Rate Map), `initial_fhbm` (first Flood Hazard Boundary Map), `participating`.
- One community (of 1,084) has no FIRM date; the app then says "not sure".

## Rebuild and test
```
.venv\Scripts\python.exe src\explain.py build      # needs data/processed/model_data_3states.parquet
.venv\Scripts\python.exe src\explain.py            # self-test
.venv\Scripts\python.exe tests\test_explain.py     # every county in all three states, committed files only
```
Rebuild whenever `models/production_model.json` or the modelling table changes.
