# Report 09: Explainability: What Drives Expected Flood Losses, Globally and Locally

*Milestone 12, 2026-11-12.*
- *Source:* `notebooks/10_explainability.ipynb`.
- *Code:* `src/global_explain.py` (SHAP for both models) and `src/explain.py` (local explanations, app data), both with self-tests, plus `tests/test_explain.py`.
- *Figures:* `reports/figures/10_*.png`.
- *Tables:* `reports/09_*.csv`.
- *App data:* `app/data/` (see `app/data/README.md`).

*Area-level analysis of aggregated public data: every number describes the average for a group of similar insured buildings in an average year, never a prediction or quote for an individual property.*

## In one paragraph
The production model (the hybrid GLM chosen in Milestone 11) is **explainable by design**: a base rate times one multiplier per factor.

**Global drivers.** Across Virginia's insured buildings, **year built** and **flood zone** drive expected loss cost most. A typical building's year-built factor moves its prediction ×1.95 away from average, and its zone ×1.63. Next come **occupancy** (×1.20) and the **credibility county factor** (×1.19). Floors barely matter (×1.00).

**XGBoost, explained with SHAP**, agrees on the top three drivers, the weight of geography and the two key interactions. It disagrees where features overlap (it splits year built's credit with post-FIRM), where it uses small effects the GLM's storm-aware rule rejected, and on elevation, which it pushes the wrong way in frequency for 76% of elevated exposure.

**Local explanations.** For any building profile or county, the app can now show:
- an exact **waterfall** from the Virginia average ($197 per policy-year) to the result;
- a careful **plain-English sentence**;
- a **certainty label**.

All of it is precomputed for the **133 Virginia counties and cities**, in about 0.55 MB of committed files.

## 1. Global drivers of the production GLM (`10_glm_relativities.png`, `10_glm_interactions.png`, `10_glm_importance.png`)
**Multipliers vs the base building** (A/V zone, built 1980–99, single-unit, $250K+ cover, not elevated, 2 floors, primary residence; $126.78 per policy-year before the county factor):

| Factor (pure-premium multiplier) | | Factor | |
|---|---|---|---|
| Built 1960–79 | **3.86** | Non-residential | **4.59** (severity) |
| Built before 1960 | 2.82 | Multi-unit residential | 2.84 (severity) |
| Built 2000+ | 0.54 | Contents only / under $100K | 0.36 / 0.79 |
| B/C/X zone | **0.36** | Elevated | 0.72 (A/V) |
| Not primary residence | 1.07 | 1 floor / 3+ floors | 1.01 / 1.01 |

**Interactions** (shown as two-way tables):
- **Year built × zone (frequency):** the old-building penalty is smaller outside A/V (1960–79 × B/C/X ×0.74).
- **Coverage × occupancy (severity):** low limits cap big buildings' claims. A multi-unit building under $100K is 1.11× base, not 2.84 × 0.79 = 2.25×.
- **Elevated × zone (severity):** elevation saves less outside A/V (×0.89 vs ×0.72).

**Importance across the portfolio.** A big multiplier on a rare level matters less than a moderate one on half the buildings. So importance is measured as the exposure-weighted **mean |SHAP value|**: how far each factor moves a building's prediction from the portfolio average, on the log scale. For a GLM it has an exact closed form.

| Rank | Factor | Typical multiplier | Share |
|---|---|---|---|
| 1 | Year built | ×1.95 | 38% |
| 2 | Flood zone | ×1.63 | 28% |
| 3 | Occupancy | ×1.20 | 10% |
| 4 | County factor | ×1.19 | 10% |
| 5 / 6 | Elevated / building coverage | ×1.12 each | 6% each |
| 7 / 8 | Primary residence / floors | ×1.03 / ×1.00 | 2% / 0.2% |

Occupancy has the widest multiplier range (4.6×) but ranks third, because 89% of insured buildings are single-unit homes.

## 2. XGBoost explained with SHAP (`10_shap_beeswarm.png`, `10_shap_dependence.png`)
**Method.**
- Exact **TreeSHAP** from XGBoost itself, on the three Milestone 10 models (refitted on all 16 years).
- Values are on the log scale. Rows sum exactly to the model's output (asserted).
- One-hot columns are summed into their feature. Pure premium (freq × sev) = frequency SHAP + severity SHAP.
- **Lighter approach:** beeswarms on 5,000 rows sampled in proportion to exposure, interaction values on 3,000 rows. Importances use every row.
- The whole notebook runs in under 2 minutes.

**XGBoost importance (mean |SHAP|, typical multiplier):**
- **Frequency:** year built ×1.50, zone ×1.34, post-FIRM ×1.08, coverage ×1.06.
- **Severity:** occupancy ×1.25, county group ×1.17, coverage ×1.15, zone ×1.12.
- **Pure premium (freq × sev):** zone ×1.52, year built ×1.51, occupancy ×1.27, post-FIRM ×1.22, county group ×1.17.

**Dependence plots and interactions.**
- **Frequency:** the strongest interaction is year built × zone. XGBoost's 1960–79 × B/C/X averages ×0.87, against the GLM's ×0.74.
- **Severity:** the strongest is coverage × occupancy. Low-limit multi-unit and non-residential claims are shrunk ×0.56–0.73, and $250K+ non-residential is lifted ×1.27.
- These are exactly the interactions the hybrid GLM adopted in Milestone 11.

## 3. GLM vs XGBoost: do they tell the same story? (`10_importance_shap_vs_glm.png`, `10_effect_by_level.png`, `09_importance_comparison.csv`)
Same yardstick (mean |SHAP|), same rows, same scale. Rank correlations: frequency 0.70, severity 0.67, pure premium 0.72.

| | Agree | Disagree |
|---|---|---|
| **Top drivers** | year built, zone, occupancy are the top 3 for pure premium in both; occupancy #1 for severity in both | XGBoost ranks zone just above year built |
| **Geography** | **identical weight** (0.169 vs 0.169, pure premium); #2 for severity in both. Credibility on county A/E and trees on county group agree | n/a |
| **Interactions** | year built × zone and coverage × occupancy, same direction and similar size | n/a |
| **Post-FIRM** | n/a | XGBoost 0.20 (#4); GLM 0 (failed the keep rule). It overlaps year built: XGBoost's year built + post-FIRM (0.41 + 0.20) ≈ the GLM's year built (0.67) |
| **Small frequency effects** | n/a | XGBoost uses contents-only cover (×1.35), multi-unit (×1.10) and primary residence; the GLM rejected them on unseen years |
| **Elevation** | both lower severity for elevated buildings | XGBoost frequency *raises* the prediction for 76% of elevated exposure: it learned where elevated buildings are |

**Reading.** The two models agree about what matters and in which direction. XGBoost's extra accuracy (Milestone 11: +4.3 points) comes from many small, interacting effects that are hard to defend one by one, plus finer geography. The hybrid GLM already carries the two biggest interactions. **SHAP closes part of the explanation gap from Milestone 11 but not the direction problem.** An explanation that says "being elevated raises your expected claims" would be accurate about the model and wrong about the world. That's one more reason the app keeps the GLM.

## 4. "Explains the model, not the world"
SHAP values and relativities describe how the **model's prediction** moves with a feature. They don't describe what would happen if a building changed. Features are often **proxies** for things we can't see:
- **Coverage band stands in for building value and type.**
  - The multiplier for under $100K of cover is ×0.79 on claim size. That doesn't mean "buy less cover and the flood does less damage".
  - Low limits go with smaller and cheaper buildings: 56% of contents-only exposure is multi-unit or non-residential, vs 10% for $250K+ (`09_proxy_coverage.csv`).
  - Only one part is genuinely causal: the limit caps the payout. 5.7% of under-$100K claims hit a limit, vs 0.8% for $250K+.
- **Elevated: Simpson's paradox.**
  - 67% of elevated buildings are in A/V zones, vs 40% of others.
  - So the one-way gap in actual loss cost (×0.58) understates the within-zone gap (×0.40 in A/V, ×0.57 in B/C/X).
  - The GLM gives elevation ×0.72 (A/V) and credits part of the rest to year built, because elevated buildings are newer.
- **Year built 1960–79 (×3.86)** stands in for pre-FIRM construction in older, low-lying neighbourhoods, not age itself.

The app's sentences are therefore descriptive and area-level: "mainly because 46% of insured buildings here are multi-unit or non-residential". They never give advice for an individual building.

## 5. Local explanations (`10_waterfall_profile_*.png`, `10_waterfall_county_*.png`)
**Profile waterfall:**
- It starts at the **Virginia average** ($197.21 per policy-year: the model averaged over all insured Virginia buildings 2010–2025, close to actual $198.13).
- Each step is a **multiplicative Shapley value**: the move from setting that factor to the profile's level on every Virginia building, averaged over all 2⁸ = 256 orders.
- The steps multiply exactly to the result.
- The exact rating worksheet (from the base building) is returned too.

| Profile | Expected loss / policy-year | vs Virginia | Biggest steps |
|---|---|---|---|
| Virginia Beach, 1960s 1-storey house, A/V | **$598** | 3.0× | 1960–79 ×1.75, A/V ×1.50, county ×1.21, single-unit ×0.79 |
| Norfolk, 2000+ elevated 2-storey second home, A/V | **$44** | 0.22× | 2000+ ×0.26, A/V ×1.37, elevated ×0.83, county ×0.83 |
| Roanoke city, pre-1960 house, B/C/X, under $100K | **$105** | 0.53× | B/C/X ×0.52, pre-1960 ×1.45, under $100K ×0.78 |
| Arlington, 1970s commercial, B/C/X, $100K–249K | **$269** | 1.36× | non-residential ×2.23, 1960–79 ×1.61, B/C/X ×0.46, $100K–249K ×0.69 (interaction) |

Example sentence: *"For insured buildings with this profile in Virginia Beach city, the model's expected flood loss is about $598 per policy-year in an average year, about 3.0x the Virginia average, mainly because it was built in 1960-1979 and it is in a high-risk A/V flood zone. This describes the average for a group of similar insured buildings, not a quote or a prediction for any individual property."*

**County breakdown:** ratio to Virginia = **property mix effect × county-factor effect** (exact). The mix is split into one index per factor plus an "overlap and interactions" step.

| County | Expected / policy-year | vs Virginia | Mix | County factor | Z | Certainty |
|---|---|---|---|---|---|---|
| Norfolk city | $199 | 1.01× | ×1.21 | ×0.83 | 0.98 | High |
| Newport News city | $316 | 1.60× | ×0.95 | ×1.69 | 0.91 | High |
| Roanoke city | $543 | 2.75× | ×2.81 | ×0.98 | 0.79 | Medium |
| Caroline County (lowest Z with ≥ 1,000 policy-years) | $164 | 0.83× | ×0.78 | ×1.07 | 0.03 | Low |

Example sentences:
- *"Model-based expected flood losses for insured buildings in Norfolk city are close to the Virginia average (about $199 vs $197 per policy-year in an average year): pushing it up, 68% of insured buildings here were built before 1980 (Virginia: 50%); pulling it down, its own claims history has had smaller claims than the model expects (county factor 0.83, high certainty)."*
- *"…Newport News city are about 1.6x the Virginia average…, mainly because its own claims history has had larger claims than the model expects (county factor 1.69, high certainty) and 60% of insured buildings here were built before 1980 (Virginia: 50%)."*

**Across all 133 localities:**
- **Mix explains most of the differences.** Exposure-weighted variance of the log ratio: mix 0.129 vs county factor 0.042. The mix step is the bigger one in 78% of the 32 large counties.
- **Overlap step:** within ×0.8–1.25 for 83% of counties and 98% of exposure. It mostly reflects coverage × occupancy.
- **Certainty labels:** 11 High, 24 Medium, 98 Low. The 11 High localities hold 71% of policy-years.
- **County average vs profile.** Roanoke city is 2.75× as a county but 0.53× for an older B/C/X house there. The county number reflects its commercial-heavy insured mix, not every building in it.

## 6. What the app gets (`app/data/`)
| File | What |
|---|---|
| `county_summary.csv` | every locality: expected frequency, severity, pure premium, ratio, county factor, Z, certainty, mix, why-it-differs factors, sentence, uncertainty note, actual 2010–2025 experience |
| `rating_factors.json` | the production model as lookup tables (tested equal to `production.predict`) |
| `portfolio.parquet` | the Virginia insured mix (background for the profile waterfall) |
| `county_history.csv` | county × year policy-years, claims, losses (2025 $) for the history chart |
| `virginia_averages.json` | statewide averages, mix, certainty rule, disclaimer |

`tests/test_explain.py` runs `explain_county` and `explain_profile` for **every** county plus 60 random profiles, **using only committed files**. All checks pass.

## Limitations
- **Explanations describe the model, not causes** (section 4).
- **SHAP baselines differ slightly.** XGBoost's TreeSHAP uses the trees' training distribution; the GLM's SHAP uses the exposure-weighted portfolio. Importance comparisons use mean |SHAP|, which isn't sensitive to this.
- **Interaction values come from a 3,000-row sample**, and the beeswarms from 5,000 rows (lighter approach). Importances use every row.
- **High expected loss costs in small inland cities** (Bristol, Danville, Winchester) come from their commercial-heavy insured mix. Their actual experience is thin and erratic (Bristol: 3 claims) and their certainty is Low or Medium. The app must show the certainty and the actual history next to the estimate.
- **The county mix uses all of 2010–2025** (the basis of the Virginia average). The 2025 mix gives a Virginia average of $209 rather than $197.
- **The county factor is the least stable part** of the model (Milestone 11) and is labelled less certain everywhere.

## Interview-ready sentences
1. "My production model is a GLM, so it explains itself: a base rate times a multiplier for each factor. To rank factors I used mean absolute SHAP values over the insured portfolio. Year built and flood zone dominate, each typically moving a prediction 60–95% from average."
2. "I explained the XGBoost challenger with TreeSHAP and compared it with the GLM on the same rows and scale. They agree on the top drivers, the weight of geography and the two key interactions. But XGBoost's frequency model raises predicted claims for 76% of elevated buildings, because it learned where elevated buildings are rather than what elevation does."
3. "SHAP explains the model, not the world. Coverage band looks like it lowers claim size, but it's really standing in for building value. Only the policy-limit cap is causal. So my app's explanations are descriptive and area-level and never advise an individual owner."
4. "For the app I built local explanations as exact multiplicative Shapley values from the Virginia average. Each county's difference splits exactly into property mix and the credibility county factor. That shows Newport News is high because of its own claims history, while Norfolk's risky mix is offset by its smaller-than-expected claims."
