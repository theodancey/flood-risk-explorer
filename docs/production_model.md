# Production model: how to load and use it

*Chosen in Milestone 11 (`reports/08_model_comparison.md`).*

**What it is:** a hybrid GLM. That's a Poisson frequency GLM (with a year built × flood zone interaction) × a Gamma severity GLM (with building coverage × occupancy and elevated × flood zone interactions) × large-loss load (1.000) × ICC bridge × off-balance × Bühlmann–Straub county factor.
- Fitted on 2010–2025.
- 2025 dollars, average-year level.

## Files
| File | In Git? | What |
|---|---|---|
| `models/production_model.json` | yes (≈ 76 KB) | everything needed to predict: both GLMs' specs and coefficients, loads, county factors + county table (names, A/E, credibility Z), model card |
| `src/models/production.py` | yes | loader: `load()`, `predict()`, `explain()`, self-test |

No pickles or tree files are needed.

## Use
```python
import sys; sys.path.insert(0, "src/models")
import pandas as pd
import production

m = production.load()
m.features          # columns the input needs, plus 'county' (5-digit FIPS as text)
m.card              # model card: structure, years, data hash, validation scores
m.disclaimer        # the area-level disclaimer the app must show

cells = pd.DataFrame([{
    "county": "51810", "flood_zone_group": "A/V high risk", "construction_band": "1960-1979",
    "floors_group": "1", "occupancy_group": "single-unit residential",
    "building_coverage_band": "$250K or more", "elevated": "False", "primary_residence": "True"}])
m.predict(cells)    # frequency (claims per policy-year), severity ($ per claim), county_factor,
                    # pure_premium ($ per policy-year)  -> about $598 for this row
m.explain(cells.iloc[0])   # factor-by-factor worksheet; .attrs holds frequency, severity, pure_premium
```
- Level names are exactly those in the modelling table (`docs/modeling_dataset.md`).
- An unseen level is treated as the base level.
- A county not in the table gets factor 1.0. Counties with no claims carry the re-centring scale (1.06).

## Explanations for the app (Milestone 12)
`src/explain.py` wraps this model for the app and reads only committed files (`app/data/*` + this JSON):
- `explain_profile(features)`: the result above plus an exact waterfall from the Virginia average, the worksheet, a sentence and a certainty label.
- `explain_county(county)`: county averages, property mix vs county factor, the uncertainty note and a sentence.

See `app/data/README.md` and `reports/09_explainability.md`.

## Rules for the app
- **Area level only.** Show `m.disclaimer` next to every number. A prediction is the average for insured buildings with those characteristics in that county, in an average year. It's never a quote for a property.
- **Label the county factor "less certain"** (Theo's Question 7 decision). `m.county_table` has each county's A/E and credibility Z for display.
- **Show the worksheet** (`explain`) as "the reasons behind it". Interaction rows appear only when the row's levels activate them.

## Rebuild
1. Run `notebooks/09_model_comparison.ipynb` (section 8). It refits on all years and rewrites the JSON.
2. Check with `.venv\Scripts\python.exe src\models\production.py`. The self-test checks that the worksheet equals `predict()` and the stored Virginia Beach check value.
