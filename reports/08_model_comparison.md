# Report 08: GLM vs XGBoost, Head to Head (Decision Memo)

*Milestone 11, 2026-11-11.*
- *Source:* `notebooks/09_model_comparison.ipynb` (code: `src/models/compare.py`).
- *Figures:* `reports/figures/09_*.png`.
- *Tables:* `reports/08_*.csv`.
- *Production model:* `models/production_model.json` (how to use it: `docs/production_model.md`).

*Area-level analysis of aggregated public data: every number describes the average for a group of similar insured buildings, never a prediction for an individual property.*

## The question
Which model should the app use for area-level expected flood loss cost?
- **(a) the GLM champion:** Poisson frequency × Gamma severity × Bühlmann–Straub county factor;
- **(b) a GLM with the interactions XGBoost found;**
- **(c) XGBoost:** frequency × severity, or a single Tweedie model.

The decision rule was written in `PROGRESS.md` **before** any result was computed.

## Fair setup (`08_fair_setup.csv`)
**Identical for every model:**
- the same data file, rows and folds, and the same test rows (asserted);
- the same loss treatment: 2025 dollars, no cap, ICC bridge, off-balance refitted in every fold;
- the same yardstick: re-based deviance improvement over a no-features model, each unseen year counting once, plus Gini.

XGBoost frequency × severity is assembled by the **same code** as the GLM version, so the off-balance and the county factor are computed identically.

**Headline design:** 16-year leave-one-year-out, plus the 11 development folds and the one-time 2021–2025 holdout. Nothing was re-tuned.

**Deliberate differences:**
- XGBoost was offered all 10 features; the GLMs kept only those that passed the storm-aware keep rule.
- County enters XGBoost severity through its trees, and the GLM through credibility. A like-for-like variant (XGBoost without geography + the same credibility factor) is also scored.

**Reproduction check:** Milestone 9's and Milestone 10's scores reproduce exactly.

## Scorecard: pure premium (`08_scorecard.csv`, `09_scorecard.png`)
| Model | 16-yr LOYO improvement | normal / cat years | Gini | Years beating champion | Holdout 2021–25 (Gini) | Double-lift end error vs champion | Segments outside 0.8–1.25 (new) |
|---|---|---|---|---|---|---|---|
| **GLM champion** (freq × sev + county) | 10.7% | 11.3 / 8.3 | 0.473 | n/a | 8.9% (0.472) | n/a | n/a |
| **Hybrid GLM + 3 interactions + county** | 11.2% | 12.1 / 7.1 | 0.481 | 9 of 16 | 10.3% (0.495) | 0.32 vs 0.24 | none |
| GLM Tweedie | 11.5% | 11.8 / 10.1 | 0.486 | 12 of 16 | 9.8% (0.479) | n/a | n/a |
| XGB freq × sev | 15.0% | 14.8 / 15.8 | 0.533 | 15 of 16 | 12.6% (0.518) | **0.08 vs 1.12** | built 2000+ (0.69), Alexandria, Chesapeake |
| XGB freq × sev (no geo) + county | 13.3% | 14.2 / 9.3 | 0.512 | 14 of 16 | 11.9% (0.511) | n/a | n/a |
| XGB Tweedie | **18.3%** | 20.1 / 10.4 | **0.596** | 15 of 16 | **19.6%** (0.645) | 0.36 vs 0.74 | built 2000+, B/C/X, Fairfax, small inland |

**Components** (16-year LOYO, GLM → XGBoost):
- **Frequency:** 4.7% → 6.5%. Catastrophe years 14.2% → **10.8%**: the GLM is better in the storms.
- **Severity:** 15.7% → 22.9%.
- **NB vs Poisson:** the NB challenger is not significantly better (+0.5, CI −0.4 to 1.3).

## Is the accuracy difference real? Yes, for XGBoost (`09_bootstrap_ci.png`, `09_per_year_wins.png`)
| Challenger − GLM | Points | Years won | Year-bootstrap 95% CI | Cluster-bootstrap CI (LOYO / holdout) | Gini difference (CI) |
|---|---|---|---|---|---|
| XGB freq × sev, pure premium | **+4.3** | 15/16 (sign p = 0.0005) | **2.9 to 5.7** | 3.0 to 5.5 / 1.9 to 6.4 | +0.060 (0.041 to 0.080) |
| XGB, severity | +7.2 | 16/16 | 5.4 to 9.1 | 3.7 to 8.7 / 4.1 to 10.5 | +0.056 |
| XGB, frequency | +1.8 | 14/16 | 0.1 to 3.2 | 1.1 to 2.7 / 1.0 to 3.4 | +0.051 |
| Hybrid GLM, pure premium | +0.4 | 9/16 (p = 0.80) | **−0.3 to 1.1** | 0.07 to 0.87 / 0.8 to 2.0 | +0.008 (0.000 to 0.017) |

**How to read the two bootstraps:**
- The **year bootstrap** asks whether a gap survives a different draw of weather years.
- The **cluster bootstrap** asks whether it survives resampling county groups within each year.

**XGBoost's gap passes both, on every component.** Its lead comes mostly from claim *size*.

**The hybrid's gain** is clear within years but **not consistent across years**.

## Beyond accuracy
**Stability** (`08_stability.csv`, `09_stability.png`). The test: change the training data and measure how much the same 2025 cells' *relative* predictions move.
- **Frequency and severity:** the GLMs move about 2–3% when one year is dropped; XGBoost moves 4.3–4.4%.
- **Random seed:** XGBoost also moves **1.7–4.5% when only the random seed changes**.
- **Biggest mover:** the champion's county factor. Dropping 2011 moves the champion's pure premium 42%, vs 17.5% without the county factor.

**Catastrophe years** (`09_cat_years.png`):
- No model forecasts a storm: raw A/E is 8 in 2016 for all of them.
- **XGBoost freq × sev places storm losses best** (15.8% vs 8.3% for the champion). In a held-out hurricane year, the credibility county factor points the wrong way.
- **For frequency alone, the GLM is better in 2011 and 2016.**
- Every model under-predicts Tidewater in storm years.

**Explainability** (`08_direction_checks.csv`, `08_explainability.csv`). We flipped one feature at a time on every 2025 cell to see which way each model moves:
- **XGBoost often moves the wrong way:**
  - elevation *raises* expected claim frequency for **78%** of exposure;
  - a higher coverage limit *lowers* expected claim size for **22%**;
  - in pure premium, elevation raises the loss cost for 30% of exposure and higher coverage lowers it for 34%.
- **The GLMs (including the hybrid's interaction cells) never move the wrong way.**
- **Size of the explanation:** the hybrid is 29 multipliers a customer or regulator can read, plus an exact rating worksheet. XGBoost is 3,000–13,600 tree leaves, and SHAP (Milestone 12) would be needed to explain any single prediction.

## Hybrid: GLM + XGBoost's interactions (`08_hybrid_tests.csv`, `09_hybrid_interactions.png`)
Tested with the existing keep rule on 2010–2020 only:

| Term | Storm-aware p | Dev-LOYO gain | What it says |
|---|---|---|---|
| Severity: coverage × occupancy | < 10⁻²⁰ | +1.16 pts, 11/11 yrs | low limits cap big buildings' claims: multi-unit under $100K = 1.11× base, not 1.45× |
| Frequency: year built × zone | 0.002 | +0.08, 8/11 | the old-building penalty is smaller outside A/V (1960–79 interaction 0.74, matching the trees' 0.73) |
| Severity: elevated × zone | 0.037 | +0.12, 6/11 | elevation saves less outside the floodplain (0.89× vs 0.72×) |

**Frequency second pass** (the Milestone 8 follow-up): no dropped feature passes.

**What the hybrid closes:**
- about **a tenth** of the XGBoost gap on 16-year LOYO and **a third** on the holdout;
- some of the segment imbalance: contents-only 1.45 → 1.35, multi-unit 1.30 → 1.19, under-$100K 0.74 → 0.85.

**Where it doesn't help:** it's slightly worse in catastrophe years and on its double lift against the champion.

## Recommendation: the app uses the hybrid GLM
The pre-stated rule decides it:

1. **XGBoost doesn't qualify, even before explainability.**
   - It passes accuracy, holdout, catastrophe years and double lift.
   - It **fails the segment check:** it over-predicts buildings from 2000 or later (A/E 0.69) and mis-sizes Alexandria (1.39) and Chesapeake (0.78), where the champion is balanced.
   - It also moves the wrong way on elevation and coverage for a large share of cells, and its rates change with the random seed.
2. **The hybrid GLM replaces the plain GLM.**
   - Its three terms passed the keep rule.
   - It's better on 16-year LOYO (11.2% vs 10.7%) and the holdout (10.3% vs 8.9%).
   - It keeps a relativity table: two-way tables for the interacting factors.
3. **Disclosed:** the hybrid's gain is small and not consistent across years (9 of 16). The frequency interaction adds almost nothing out of sample. And **XGBoost's accuracy lead is real** (about 4 points). It stays the benchmark: Milestone 12 (SHAP) will show whether its extra signal can be explained, constrained to sensible directions, or moved into further GLM terms.

**What changes for the app.** Milestone 9's worksheets change most where the interaction bites:
- Arlington 1970s commercial building ($100K–249K cover): **$516 → $269**;
- Newport News condo building: $579 → $656;
- Virginia Beach 1960s house: $585 → $598.

## Limitations
- **Only three hurricanes in 16 years.** Catastrophe-year conclusions rest on three data points.
- **Selection optimism.** XGBoost's hyperparameters and the GLMs' features were chosen on the same 2010–2020 folds that are part of the 16-year LOYO. The holdout (clean for both) agrees on the ranking.
- **County effects are unstable** in every structure. The credibility factor helps on average, hurts in storm years, and moves the most when one year is dropped.
- **Area level only:** all predictions are averages for groups of insured buildings.

## Interview-ready sentences
1. "I compared GLMs and XGBoost on identical folds with the same loss treatment and yardstick. XGBoost was about 4 points better on unseen-year deviance, in 15 of 16 years, and both a year bootstrap and a county-cluster bootstrap put the gap clearly above zero. So the accuracy difference is real."
2. "I still recommended the GLM, using a rule I wrote down before seeing results. XGBoost over-predicted newer buildings by 30% and two county groups by 20–40%, and its predictions moved the wrong way on elevation for most cells. Its rates also changed when only the random seed changed."
3. "A double-lift chart showed that where XGBoost and the GLM disagreed most, XGBoost was right. So I took the interactions XGBoost found back into the GLM. Coverage × occupancy was highly significant, and the hybrid closed about a third of the gap on the holdout while keeping a relativity table."
4. "Model stability mattered too. My GLM frequency and severity predictions moved about 2–3% when a year was dropped, versus about 4% for XGBoost. But the least stable part of my own model was the county credibility factor, which a single hurricane year can move a lot."
5. "In hurricane years no model forecasts the storm, so I judged them on whether they put the storm's losses in the right cells. XGBoost did best overall, but the GLM was better at storm frequency. That's a reminder that a model tuned mostly on normal years may not be best in the year that matters most."
