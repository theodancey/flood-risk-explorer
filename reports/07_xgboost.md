# Report 07: XGBoost Challenger Models: Frequency, Severity, Pure Premium (Virginia, 2010–2025)

*Milestone 10, 2026-11-11.*
- *Source:* `notebooks/08_xgboost.ipynb` (code: `src/models/xgb_models.py`, with a self-test).
- *Figures:* `reports/figures/08_*.png`.
- *Tables:* `reports/07_*.csv`.
- *Settings:* `models/xgb_settings.json` (committed).
- *Saved models:* `models/xgb_{frequency,severity,tweedie}_{dev,all}.ubj` (gitignored; rebuild with `.venv\Scripts\python.exe src\models\xgb_models.py fit-final`).

*Area-level analysis of aggregated public data: every number describes the average for a group of similar insured buildings, never a prediction for an individual property. This is a preview against the GLMs. The full comparison is Milestone 11.*

## In one paragraph
I built three **gradient-boosted tree models (XGBoost)** as challengers to the GLMs, on the same rows, features and validation design:
- **frequency:** Poisson, with log(exposure) as the offset;
- **severity:** Gamma, on individual claims, with no large-loss cap;
- **pure premium:** Tweedie, p = 1.7.

**How they were tuned.** A randomized search scored every setting on the **11 development years (2010–2020) only**, by leave-one-year-out:
- 300 settings × 11 folds, about 3.5 hours on a laptop;
- tree count chosen by early stopping on the cross-validation mean;
- a separate search for each geography variant;
- pre-stated rules for geography and monotonic constraints.

**Results.** On unseen years XGBoost beats the GLMs on all three models:

| | Development years | Holdout 2021–2025 |
|---|---|---|
| Frequency | 6.75% vs 5.58% | 4.34% vs 2.21% |
| Severity | 25.43% vs 19.16% | 23.66% vs 16.65% |
| Tweedie | 17.44% vs 13.54% | 20.85% vs 10.42% |

**Question 7.** XGBoost finds **no useful geography for frequency**: county group is worse in 10 of 11 years. It does find **real county and Tidewater signal for claim size**: +2.0 points, confirmed on the holdout. That's the same split as Milestone 9's credibility result.

**Pure premium.** The XGBoost Tweedie scores best overall but **fails the double lift and segment checks**, just like the GLM Tweedie, so **frequency × severity stays the recommended structure**.

## 1. What was built
| Model | Objective | Modelling unit | Offset (`base_margin`) |
|---|---|---|---|
| Frequency | `count:poisson` | 170,806 rating cells (10 features × year), exposure > 0 | log(exposure) + log(year level) |
| Severity | `reg:gamma` | 9,553 paid claims (building + contents, 2025 $, ICC excluded, **uncapped**) | log(year level) |
| Pure premium | `reg:tweedie`, power 1.7 | same cells; label = loss cost, weight = exposure | log(year level) |

**The year is handled like the GLMs' year factor.**
- The **year level** comes from a year-only model on the training years. It sits in `base_margin` (XGBoost's offset), so a hurricane year isn't blamed on whichever building types it happened to hit.
- An unseen year is predicted at the **average-year level**, set so training predictions sum to training actuals.
- Year is never a tree feature.

**Features (all candidates, not just the GLMs' final sets):** flood zone, year built, building coverage, occupancy, elevated, post-FIRM, floors, primary residence, plus the Tidewater flag and county group as geography candidates.
- Ordered bands are integer codes, which monotone constraints need.
- Occupancy and county group are one-hot. There's no target encoding, so no leakage.

**Pure premium from the two models.** XGBoost freq × sev = frequency × severity × ICC bridge (1.0092) × an off-balance factor fitted on the training years. The Tweedie gets its own off-balance for losses in cells with no exposure, as in Milestone 9.

## 2. Tuning without leakage (`08_tuning_search.png`, `08_early_stopping.png`)
**Scoring.** Each random setting was scored by **leave-one-year-out on 2010–2020**: 11 fits, each predicting one unseen year. The score is the **mean re-based deviance improvement over the no-features model**, the same metric the GLMs were chosen on. **2021–2025 was never used for any choice.**

**Early stopping on the cross-validation mean.**
- The 11 fold models grow side by side, 10 trees at a time (20 for severity).
- After each step every held-out year is scored and the scores are averaged.
- Stop after 150 trees without improvement (caps: frequency 1,500, severity 4,000, Tweedie 1,500).
- The best tree count becomes `n_estimators`.

**Search space** (seed 2026): max_depth 2–6; learning_rate 0.01–0.1 (log scale); min_child_weight in each objective's hessian units (frequency 1–300 ≈ expected claims per leaf, severity 3–150 ≈ claims, Tweedie 100–100,000); subsample 0.6–1; colsample_bytree 0.5–1; reg_lambda 0.1–50; reg_alpha 0 or 0.1–5; gamma 0 or 0.01–1.

**Budget.** 40 settings for frequency and severity and 20 for the Tweedie, **per geography variant**.

**One search per geography variant.** This change was made after a dry run, before the real search. It showed that county group drags frequency down so much that a single search with it included would pick settings that work around it. That would make the geography test unfair.

**Tie rule:** within 0.1 points of the best, take the shallowest, then fewest trees. This is why frequency uses setting 17 (depth 5, 6.75%) over setting 5 (depth 6, 6.83%).

**What the search showed.**
- The geography variant mattered more than any single hyperparameter.
- Learning rate and depth mattered little once early stopping chose the tree count.
- min_child_weight mattered most: frequency likes small leaves, severity large ones.
- The early-stopping curves show the classic shape: rising, a peak, then a slow decline as later trees fit storm noise. For the Tweedie, hurricane years peak early and fall fast.

**Final settings** (`reports/07_xgb_final_settings.csv`):

| | Frequency | Severity | Pure premium (Tweedie) |
|---|---|---|---|
| Geography | none | Tidewater + county group | Tidewater + county group |
| Monotone constraints | no | no | **yes** |
| max_depth | 5 | 5 | 5 |
| learning_rate | 0.023 | 0.016 | 0.021 |
| trees | 170 | 940 | 120 |
| min_child_weight | 1.8 | 88 | 5,025 |
| subsample / colsample | 0.71 / 0.96 | 0.83 / 0.74 | 0.77 / 0.89 |
| reg_lambda / alpha / gamma | 0.73 / 0.25 / 0.02 | 0.99 / 0 / 0.05 | 38.6 / 1.13 / 0 |
| Dev-LOYO improvement | 6.75% | 25.43% | 17.44% |

## 3. Geography: Question 7 (`08_geography.png`, `reports/07_xgb_geography_*.csv`)
**Rule** (the same as the GLMs): keep a geography step only if it improves the mean development score **and** wins at least 6 of 11 years. Each variant has its own tuned settings.

| | + Tidewater | + county group | Chosen | Holdout (no geo → chosen) |
|---|---|---|---|---|
| Frequency | −0.31 pts, 5/11 ✗ | −1.65 pts, 1/11 ✗ | **no geography** | 4.34% (Tidewater 4.38%, county 3.08%) |
| Severity | +0.47 pts, 7/11 ✓ | +1.54 pts, 8/11 ✓ | **Tidewater + county** | 21.1% → 23.7% |
| Pure premium (Tweedie) | +0.36 pts, 6/11 ✓ | +0.35 pts, 7/11 ✓ | **Tidewater + county** | 14.0% → 20.8% |

**Answer to Q7.**
- **Frequency:** even a flexible model can't turn county or Tidewater into better predictions of *how often* claims happen in an unseen year. County frequency differences are mostly about which storms hit where.
- **Severity:** **claim size does differ by county** in a way that holds up on unseen years. The trees find it where the severity GLM didn't (the GLM's county test failed by 0.9 points), probably because the trees let a county's effect differ by occupancy and zone.
- This matches Milestone 9's Bühlmann–Straub result: zero credibility for county frequency, high for county severity. That supports keeping the credibility county factor in the app.

## 4. Monotonic constraints (`08_monotone_pd.png`, `reports/07_xgb_monotone_*.csv`)
**Constraints (actuarial directions):**
- **Frequency:** zone, elevated, year built and post-FIRM all "never up".
- **Severity:** coverage "never down"; elevated and zone "never up".
- **Tweedie:** zone, elevated and post-FIRM "never up"; coverage "never down". Year built is left free because the GLM loss cost peaks in 1960–79.

**Rule:** use the constraints unless they cost > 0.25 points or are worse in ≥ 8 of 11 years. Tested on the top 5 settings, with the tree count re-picked.

| | Constrained − unconstrained (chosen setting) | Years better | Top-5 range | Decision |
|---|---|---|---|---|
| Frequency | **+0.10** | 2 of 11 | +0.10 to +0.34 | unconstrained (worse in 9 of 11) |
| Severity | −0.54 | 3 of 11 | −0.54 to −0.66 | unconstrained (cost > 0.25) |
| Pure premium (Tweedie) | +0.05 | 5 of 11 | +0.02 to +0.09 | **constrained** |

**Frequency is the close call.** The constraints improve the average, but almost entirely through 2016 (Hurricane Matthew, about +5 points), and they're worse in 9 of the 11 years. The pre-stated rule is designed against exactly that.

**Cost.** The unconstrained model's average elevated effect is **1.01**: elevation very slightly *raises* frequency, which isn't defensible. Milestone 11 should revisit this on all 16 years.

**How the effects compare with the GLMs:**
- **Zone, year built and post-FIRM** agree closely.
- **The Tweedie's coverage effect is nearly flat** (1.0–1.2×, vs 2.5–3.6× for frequency × severity). This is the Tweedie weighting problem from Milestone 9 again.

## 5. Held-out performance and the GLM preview (`08_ae_by_year.png`, `08_lift_charts.png`, `reports/07_xgb_preview_vs_glm.csv`)
The GLMs were refitted inside the same folds with their final features, scored by the same code on the same rows.

| | Dev-LOYO XGB / GLM | Normal years | Hurricane years | Years XGB wins | Dev Gini | Holdout XGB / GLM | Holdout Gini |
|---|---|---|---|---|---|---|---|
| Frequency (Poisson) | **6.75** / 5.58 | 5.23 / 2.38 | 10.81 / **14.10** | 9 of 11 | 0.34 / 0.31 | **4.34** / 2.21 | 0.33 / 0.26 |
| Severity (Gamma) | **25.43** / 19.16 | 28.41 / 22.73 | 17.51 / 9.63 | 11 of 11 | 0.36 / 0.32 | **23.66** / 16.65 | 0.35 / 0.30 |
| Pure premium (Tweedie p = 1.5 yardstick) | **17.44** / 13.54 | 20.01 / 14.44 | 10.58 / 11.13 | 8 of 11 | 0.56 / 0.50 | **20.85** / 10.42 | 0.65 / 0.48 |

**Reading the table:**
- **XGBoost wins almost everywhere**, including the clean holdout (which also checks that the tuning wasn't over-optimistic).
- **Frequency is weaker in hurricane years.** It also has a flatter top-to-bottom lift (4.9× vs 6.7× on development years) because its top decile is over-predicted (2.4× predicted vs 1.7× actual).
- **Severity's gain partly comes from county group.** The champion pricing structure adds county severity through the credibility factor instead, so Milestone 11 must compare like with like.
- **Raw A/E by year** moves almost identically for both methods (e.g. 2016 frequency 3.74 vs 3.75). Both predict an average year, so storms show as spikes. 2021–2025 were quiet years (frequency A/E about 0.2).

## 6. Feature importance and interactions (`08_importance.png`, `08_interactions.png`)
**Gain share** (models fitted on all 16 years):
- **Frequency:** year built 45%, zone 21%.
- **Severity:** occupancy 34%, county group 20%, coverage 12%.
- **Tweedie:** county 26%, occupancy 21%, year built 17%, Tidewater and post-FIRM 11% each.

Gain measures training use, not held-out value or direction. **SHAP values in Milestone 12** will do this properly.

**Strongest interactions (interaction ratio 1.00 = effects simply multiply, as in a GLM):**
- **Frequency, year built × zone:** 19% of split gain. 1960–79 buildings in B/C/X zones have ratio 0.73: the old-building penalty is smaller outside high-risk zones.
- **Severity, coverage × occupancy:** low-coverage multi-unit and non-residential buildings have ratios 0.46–0.74. Their big claims are capped by low limits.
- **Tweedie, year built × occupancy:** old multi-unit buildings have ratio 2.7–3.0.
- **Elevated × zone (flagged in Milestone 8):** 1.08, a small effect.

## 7. Pure premium: XGBoost Tweedie vs XGBoost frequency × severity (`08_double_lift.png`, `reports/07_xgb_segment_balance.csv`)
| | Dev-LOYO | Holdout | Holdout Gini |
|---|---|---|---|
| XGBoost Tweedie | **17.44%** | **20.85%** | 0.65 |
| XGBoost freq × sev | 17.00% | 13.39% | 0.52 |
| GLM Tweedie | 13.54% | 10.42% | 0.48 |
| GLM freq × sev (no county factor) | 13.17% | 6.82% | 0.42 |

**The double lift (development years, out of fold) sides with freq × sev.**
- In the decile where the Tweedie is relatively highest, it predicts 1.60× average and actual is 0.57×. Freq × sev predicts 0.72×.
- End-decile error is **0.78 for the Tweedie vs 0.24 for freq × sev**.

**Segment A/E, out of fold** (1.00 = balanced; above 1 = under-predicted):

| Segment | XGB Tweedie | XGB freq × sev | GLM freq × sev |
|---|---|---|---|
| Multi-unit | 1.49 | 1.15 | 1.33 |
| Under $100K coverage | 0.59 | 0.81 | 0.72 |
| Contents only | 0.71 | 0.97 | 1.61 |
| Built 2000+ | 0.70 | **0.70** | 0.95 |
| Elevated | 0.85 | 0.82 | 0.83 |

**Conclusion: Milestone 9's recommendation stands.** Frequency × severity is the pricing structure, and the Tweedie remains a challenger.

**XGBoost freq × sev balances multi-unit and contents-only better than the GLM, but over-predicts 2000+ buildings.** Its frequency effect for 2000+ is 0.37× vs the GLM's 0.23×. That's likely because early stopping after 170 small steps shrinks extreme effects toward the average. It's a Milestone 11 check.

## 8. Sensitivity checks
- **Severity large losses:** capping at P99 with a flat load scores 25.16% vs 25.43% uncapped. **Uncapped stays**, matching the GLM.
- **Frequency year levels:** one round of back-fitting changes the score by +0.01 points. **The simple year-only levels stay.**

## Limitations
- **Selection optimism.** The best of 300 settings was picked on the same 11 folds, so development scores are a little flattering. The holdout (one-time, after all choices) is the cleaner number, and it agrees.
- **Storm dependence.** Sixteen years include only three hurricane years. Frequency XGBoost is *worse* than the GLM in them. A model ranked mainly on normal years may not be the better one in the year that matters most.
- **Explainability.** 170–940 trees can't be read like a relativity table. Gain importance is crude, and SHAP (Milestone 12) is needed before anything goes into the app.
- **Defensibility.** Frequency and severity are unconstrained. The frequency model's average elevated effect is slightly above 1. The Tweedie's coverage effect is nearly flat.
- **Like-for-like.** Severity XGBoost includes county effects, while the GLM pricing structure adds them separately through credibility. Milestone 11 must compare the full structures.
- **Area-level only.** Cells are groups of policies with the same characteristics in a county group and year, so predictions are averages for groups, not property quotes.

## Interview-ready sentences
1. "I tuned XGBoost challengers to my GLMs with leave-one-year-out cross-validation on 2010–2020 only, early stopping on the average over held-out years, and a one-time 2021–2025 holdout. XGBoost improved deviance on unseen years by about 1 point for frequency and 6–7 points for severity."
2. "I put log(exposure) and a calendar-year level in XGBoost's base_margin, which is its version of a GLM offset. So the trees learn rates for building types rather than which year had a hurricane."
3. "Gradient boosting confirmed my GLM finding that county doesn't predict claim frequency in unseen years: it was worse in 10 of 11. But it found real county differences in claim size, which matches my Bühlmann–Straub credibility result."
4. "The XGBoost Tweedie had the best overall deviance, but a double-lift chart showed frequency × severity was right where they disagreed. So I kept the two-model structure and didn't choose on one metric alone."
