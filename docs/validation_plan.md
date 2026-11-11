# Validation Plan (weeks 3–6)

*Written 2026-11-11 (Task 5). Code: `src/validation.py`. Evidence: `notebooks/04_one_way_analysis.ipynb`, section 7.*

## Why validate at all, and why by year?
A model always fits the data it was trained on better than new data. **Validation** means scoring the model on data it has *not* seen, to estimate how well it will do in real use and to compare models fairly (GLM vs XGBoost, feature set A vs B).

**We always hold out whole calendar years, never random rows.** Flood claims arrive in storms: Hurricane Matthew produced 2,343 paid claims across thousands of rating cells in 2016. If random 2016 rows sat in both the training and the test data, the model would already "know" about Matthew, and the test score would be far too optimistic. Holding out the whole year means the test really is weather the model never saw. It also matches how a rate is used: priced from past years, applied to a future year.

## The data we're splitting
16 years (2010–2025), 9,557 paid claims. **Three catastrophe ("cat") years** (2011 Irene, 2012 Sandy, 2016 Matthew) hold **66% of paid claims and 68% of losses**, but only 20% of exposure (`validation.CAT_YEARS`, confirmed from the data by `validation.cat_years()`).

| Test year | 2010 | **2011** | **2012** | 2013 | 2014 | 2015 | **2016** | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Paid claims | 245 | **2,795** | **1,197** | 230 | 305 | 358 | **2,343** | 82 | 615 | 281 | 429 | 151 | 110 | 56 | 185 | 175 |
| Non-tidal inland share | 27% | 3% | 2% | 75% | 10% | 26% | 4% | 28% | 53% | 10% | 38% | 16% | 30% | 20% | 75% | 51% |

The second row matters: hurricane years are almost entirely coastal, while several normal years are dominated by inland river and flash flooding (2013, 2018, 2024). Notebook 04 found that several factors (flood zone, coverage band, elevated, coastal) behave **differently in the two kinds of year**.

## Options compared

### (a) Time-based holdout: train 2010–2020, test 2021–2025
- **For:** honest about time (it never trains on the future), simple, and mirrors real pricing. One clean final number.
- **Against:** it's a **single draw of the weather**. The 2021–2025 test window has **no cat year**, only **677 paid claims** (7% of the total), and **44% of its claims came from non-tidal inland counties vs 12.5% in 2010–2020**. A model trained on hurricane-heavy years is tested on a different flood mix. One lucky or unlucky window can decide "GLM vs XGBoost." The window also straddles **Risk Rating 2.0** (Oct 2021), which changed how several fields are recorded.
- **Cat years:** all three are in training, none in test. The test says nothing about how the model handles a hurricane year, and a model that predicts the long-run average (cat years included) will look like it **over-predicts** the quiet test years. That's correct pricing behaviour, not a bug.

### (b) Leave-one-year-out cross-validation (LOYO): 16 folds
For each year Y: train on the other 15 years, predict year Y, score it. Every year is a test year exactly once.
- **For:** **16 separate tests** spanning hurricane years, inland-flood years, and quiet years. We can see whether one model wins *consistently* (e.g. "XGBoost beats the GLM in 12 of 16 years") instead of trusting one window. Every claim is used for both training and testing (in different folds), which matters with only 9,557 claims.
- **Against:** for early test years the model has trained on *later* years ("training on the future"). That's mostly harmless for **relativities** (notebook 04: the stable factors don't trend), but it can flatter the overall **level** if costs trend upward. Losses are already in 2025 dollars, which removes general inflation. Also 16 model fits instead of 1, which is cheap for GLMs and fine for light XGBoost tuning.
- **Cat years:** each cat year is one fold, trained *without* that storm. Its errors are huge in absolute terms (thousands of claims), so **a plain sum of deviance over all folds would be dominated by 2011 and 2016**. Fix: score each fold separately, give every year equal weight, and report results **with and without the cat folds**.

### (c) Rolling origin ("forward chaining"): train on years < Y, test on Y
Never uses the future, but early folds train on very little data (testing 2016 uses only 6 years, with 2 storms), and later folds overlap heavily. It's available as `validation.forward_chaining()` as a sensitivity check, but isn't the main design.

## Recommendation
1. **Main comparison of models and feature sets: LOYO over all 16 years** (`validation.leave_one_year_out(df)`). Report every metric **per fold**, plus summaries via `validation.summarize_scores()`: mean and median over all folds (each year counts once), mean over the 13 non-cat folds, mean over the 3 cat folds, and the number of folds each model wins.
2. **Model development** (choosing features, interactions, XGBoost hyperparameters) uses **LOYO on 2010–2020 only** (`leave_one_year_out(df, val.DEV_YEARS)`, 11 folds), so 2021–2025 stays untouched until the end.
3. **Final one-time check: the forward holdout**, train 2010–2020 and test 2021–2025 (`validation.time_holdout(df)`). It's scored once, after all choices are frozen, and reported next to the LOYO results, with the caveat that it's one quiet, inland-heavy window.

**Why LOYO as the headline:** with three storms holding two-thirds of the claims, *which years land in the test set* changes the answer more than most modelling choices. Notebook 04 showed the 2021–2025 relativities differ from 2010–2020 for exactly the storm-dependent factors. LOYO averages over the weather; a single holdout bets on it.

## How each fold is scored
**Frequency** (Poisson GLM / XGBoost, `claims` with `exposure` as the offset; rows with `exposure_matched = False` dropped in both train and test):
- **Poisson deviance** on the test year, raw, and also after **re-basing**: scale the predictions so they add up to the actual claim count of that year. No model can predict *whether* a hurricane comes. Re-basing removes the year's overall level, so the score measures what models *can* do: put the claims in the right cells. Re-based scores are the main comparison; raw scores show calibration.
- **Gini / lift**: sort cells from lowest to highest predicted frequency and check that actual claims pile up at the high end. A Gini of 0 means random ranking; higher is better. Lift charts are computed by decile of predicted risk.
- **Actual ÷ expected** overall and by county group, flood zone, and coastal flag (it should be about 1 in every group after re-basing).

**Severity** (Gamma GLM / XGBoost, average paid claim per cell, weighted by `claims`): Gamma deviance and actual ÷ expected average claim. Quiet years have few claims (2023: 56), so severity summaries also report claim-weighted means across folds. Unmatched claims (`exposure_matched = False`) are kept for severity.

**Pure premium**: predicted frequency × severity (and the Tweedie cross-check), scored with Tweedie deviance, Gini, and actual ÷ expected losses.

## Leakage checks
- **Splits are by year only.** `src/validation.py`'s self-test checks that no test year appears in its own training data and that every row is tested exactly once.
- **County pooling** (`county_group`) uses each county's exposure over all years. That's exposure, not claims or losses, so it isn't outcome leakage (documented in `docs/modeling_dataset.md`).
- **An honest caveat about the shortlist:** notebook 04's one-way analysis looked at all 16 years, including 2021–2025, before the feature shortlist was written. That's a mild form of peeking at the final holdout. It's limited because the shortlist is mostly standard NFIP rating factors that any actuary would include, and the stability checks that informed it looked across all years rather than tuning to 2021–2025. From week 3 on, no modelling choice uses 2021–2025.

## Usage
```python
import validation as val
for year, train, test in val.leave_one_year_out(df):          # main comparison, 16 folds
    ...fit on train, score on test, append a row with test_year + metric...
scores = pd.DataFrame(rows)                                    # columns: model, test_year, metric
val.summarize_scores(scores, "deviance_rebased")               # all / non-cat / cat / median

for year, train, test in val.leave_one_year_out(df, val.DEV_YEARS):   # development only
    ...
train, test = val.time_holdout(df)                             # final one-time check
```
