# Production model: how to load and use it

*Chosen in Milestone 11 (`reports/08_model_comparison.md`). Refitted on Virginia, North Carolina and Maryland in Milestone 13 (`reports/10_multistate.md`).*

**What it is:** a hybrid GLM, with these parts multiplied together:
- a Poisson frequency GLM (with a year built × flood zone interaction);
- a Gamma severity GLM (with building coverage × occupancy and elevated × flood zone interactions);
- a **state factor**;
- large-loss load (1.000) × ICC bridge × **per-state off-balance**.

Details:
- **State factor:** each state's average-year level vs Virginia, from state × year effects. North Carolina 6.22×, Maryland 1.01×, including each state's off-balance.
- **Same everywhere:** the building relativities are the same in all three states. Factors whose effect differs by state were tested and flagged, not adopted.
- **No county factor** since Milestone 13. The pooled Bühlmann–Straub county factor made unseen years worse, badly so in North Carolina. That reverses Theo's Milestone 9 request; see PROGRESS.md, Questions for Theo 9.
- Fitted on 2010–2025, 2025 dollars, average-year level.

## Files
| File | In Git? | What |
|---|---|---|
| `models/production_model.json` | yes (≈ 145 KB) | the 3-state model: both GLMs' specs and coefficients (incl. state factors), loads, per-state off-balance, county table (names, A/E, credibility Z; factors all 1.0), model card |
| `models/production_model_va.json` | yes | the Milestone 11 Virginia-only model (the transfer-test model and the record) |
| `src/models/production.py` | yes | loader: `load()`, `predict()`, `explain()`, self-test |
| `src/multistate.py` | yes | `production` command: refits and saves the 3-state model |

No pickles or tree files are needed.

## Use
```python
import sys; sys.path.insert(0, "src/models")
import pandas as pd
import production

m = production.load()                      # production.load(production.VA_PATH) = the Virginia-only model
m.features          # columns the input needs, plus 'county' (5-digit FIPS as text); the state comes from the code
m.card              # model card: structure, state factors, years, data hash, validation scores by state
m.disclaimer        # the area-level disclaimer the app must show

cells = pd.DataFrame([{
    "county": "51810", "flood_zone_group": "A/V high risk", "construction_band": "1960-1979",
    "floors_group": "1", "occupancy_group": "single-unit residential",
    "building_coverage_band": "$250K or more", "elevated": "False", "primary_residence": "True"}])
m.predict(cells)    # frequency, severity, county_factor (1.0), pure_premium -> about $454 for this row
m.explain(cells.iloc[0])   # factor-by-factor worksheet (incl. the state factor)
```
- Level names are exactly those in the modelling table (`docs/modeling_dataset.md`).
- An unseen level is treated as the base level.
- **Stored check values** (`card["check_worksheets"]`), for a 1960s one-floor house in an A/V zone:
  - Virginia Beach, VA: $453.69. The Virginia-only model gave $598.07: its county factor of 1.21 is gone, and the pooled relativity for 1960–79 buildings is smaller.
  - Dare County, NC: $2,820.24.
  - Worcester County, MD: $458.67.

## Explanations for the app
`src/explain.py` wraps this model for the app and reads only committed files (`app/data/*` + this JSON):
- `explain_profile(features, state=None)`: the prediction plus an exact waterfall from **the state's** average, the worksheet, a sentence and a certainty label.
- `explain_county(county, state=None)`: county averages, property mix vs the state, the uncertainty note and a sentence.

See `app/data/README.md` and `reports/09_explainability.md`.

## Rules for the app
- **Area level only.** Show `m.disclaimer` next to every number. A prediction is the average for insured buildings with those characteristics in that county, in an average year. It's never a quote for a property.
- **Compare each county with its own state**, not with all three states. North Carolina's level is about six times the others'.
- **Show the certainty label and note.** Where a county's 2010–2025 history differs a lot from the model (e.g. Craven County, NC at 3.6×, Brunswick County, NC at 0.35×), the note says so. Such county differences didn't repeat reliably in unseen years, so the model doesn't follow them.
- **Show the worksheet** (`explain`) as "the reasons behind it".

## Rebuild
1. `.venv\Scripts\python.exe src\multistate.py production` (needs `data/processed/model_data_3states.parquet`; it reads the chosen county-factor form from `reports/10_county_form_summary.csv`).
2. `.venv\Scripts\python.exe src\explain.py build`, then `tests\test_explain.py`.
3. Check with `.venv\Scripts\python.exe src\models\production.py`. The self-test checks the worksheets against `predict()`, the stored check values, and that the Virginia-only model still gives $598.07.
