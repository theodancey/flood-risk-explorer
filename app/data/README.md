# app/data: precomputed files for the Streamlit app

*Built in Milestone 12 by `src/explain.py build`. About 0.55 MB in total, committed to Git.*

The app needs everything here plus `models/production_model.json`. It does **not** need `data/processed/`, which isn't in Git and won't exist on a free host. Each file is either an aggregate (county, county × year, or building-type × county exposure) or model output. There's no policy-level or claim-level row.

**Area level only:** every number is an average for a group of insured buildings in an average year, never a prediction or quote for an individual property. Show the disclaimer (`virginia_averages.json` → `disclaimer`) next to every number.

## Files
| File | Size | One row per | What it's for |
|---|---|---|---|
| `county_summary.csv` | ~140 KB | Virginia county/city with insured exposure (133) | the county page: expected frequency, severity, pure premium, ratio to Virginia, county factor, Z, certainty, property mix, why it differs, plain-English summary |
| `rating_factors.json` | ~33 KB | n/a | the production model as lookup tables, so the app can rate a custom profile |
| `portfolio.parquet` | ~260 KB | building type (7 model features + post-FIRM) × county | insured policy-years 2010–2025: the "Virginia background" for the profile waterfall and the source of each county's mix |
| `county_history.csv` | ~100 KB | county × year (2010–2025) | the loss-history chart: policy-years, paid claims, losses in 2025 $ (actual experience) |
| `virginia_averages.json` | ~2 KB | n/a | statewide averages, mix shares, basis, certainty rule, disclaimer |

## Columns
**`county_summary.csv`** (sorted by expected pure premium, highest first)
- **Identity:**
  - `county`: 5-digit FIPS as text (keep the leading "51");
  - `name`: Census name. 51515 is the former Bedford city, part of Bedford County since 2013.
- **Experience 2010–2025:**
  - `policy_years_2010_2025`;
  - `avg_policies_per_year` (= policy-years ÷ 16);
  - `paid_claims_2010_2025`;
  - `actual_loss_cost_2010_2025`: actual losses ÷ policy-years. It's noisy: it's dominated by whether a storm hit.
- **Model (average year, 2025 $):**
  - `expected_frequency`: claims per policy-year;
  - `expected_severity`: expected losses ÷ expected claims, $ per claim, without the county factor;
  - `expected_pure_premium`: $ per policy-year, county factor included;
  - `ratio_to_virginia`.
- **Credibility:**
  - `county_factor`: Bühlmann–Straub, labelled less certain;
  - `Z_sev`: severity credibility, 0–1;
  - `Z_freq`: always 0, because county frequency differences are storm noise;
  - `certainty`: High (Z ≥ 0.8), Medium (0.5 ≤ Z < 0.8), or Low (Z < 0.5 or no claims).
- **Why it differs:**
  - `ratio_to_virginia` = `mix_effect` × `county_factor_effect`, exactly.
  - `mix_effect` = the product of the seven `mix_index_<feature>` values × `overlap_effect`.
  - A mix index = the county's average relativity for that feature ÷ Virginia's.
  - The overlap effect covers interactions (mostly coverage × occupancy) and how the factors combine.
- **Mix (% of the county's insured policy-years):**
  - `pct_av_zone`, `pct_elevated`, `pct_pre_firm`, `pct_built_pre_1980`;
  - `pct_multi_or_nonres`, `pct_not_primary`;
  - `pct_cover_250k_plus`, `pct_cover_under_100k`;
  - `pct_one_floor`.
- **Text:**
  - `summary`: a plain-English sentence;
  - `uncertainty_note`.

**`rating_factors.json`**
- `levels`, `bases`.
- `frequency.base_per_policy_year`.
- `severity.base_claim_size`: includes the large-loss load (1.000).
- `frequency/severity.log_relativity[feature]`: natural-log relativity per level, in `levels` order. 0 = base, or the feature isn't in that part.
- `interactions`: two-way log tables, `[level of a][level of b]`.
- `loads`: ICC bridge, off-balance.
- `county_factors` and `county_credibility` (Z, claims, A/E, names).
- `labels`, `disclaimer`.

**How to rate a profile from this file:**
1. Pure premium = base frequency × base severity.
2. Multiply by exp(sum of the frequency and severity log relativities, including interactions).
3. Multiply by the ICC bridge × the off-balance × the county factor.

`tests/test_explain.py` checks that this equals `production.predict` exactly.

**`portfolio.parquet`:** columns are the 7 model features, `post_firm`, `county`, `county_name` and `exposure` (policy-years 2010–2025). Levels are text ("True"/"False" for yes/no).

**`virginia_averages.json`:**
- Basis = exposure-weighted average over all insured Virginia buildings, 2010–2025 mix.
- Expected pure premium **$197.21** per policy-year; on the 2025 mix it's $209.05; actual 2010–2025 is $198.13.
- Also holds the mix shares, mean relativities, average county factor (≈ 1.000), the certainty rule and the disclaimer.

## Using it
```python
import sys; sys.path.insert(0, "src")
import explain
r = explain.explain_county("Norfolk")             # FIPS or name ("Norfolk city" / "Norfolk")
r["sentence"], r["waterfall"], r["uncertainty_note"]
p = explain.explain_profile({"county": "51810", "flood_zone_group": "A/V high risk",
                             "construction_band": "1960-1979", "floors_group": "1"})
p["pure_premium"], p["waterfall"], p["worksheet"], p["sentence"]
explain.plot_waterfall(p)                          # matplotlib figure
```
Missing profile features default to the model's base levels. Ambiguous names (Fairfax, Franklin, Richmond and Roanoke are each both a county and a city) raise an error listing the options.

## Rebuild and test
```
.venv\Scripts\python.exe src\explain.py build      # needs data/processed/va_model_data.parquet (about 10 s)
.venv\Scripts\python.exe src\explain.py            # self-test
.venv\Scripts\python.exe tests\test_explain.py     # every county + random profiles, committed files only (about 2 min)
```
Rebuild whenever `models/production_model.json` or the modelling table changes.
