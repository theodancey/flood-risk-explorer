# Report 10: From Virginia to Virginia + North Carolina + Maryland

*Milestone 13, 2026-11-13.*
- *Source:* `notebooks/11_multistate.ipynb`.
- *Code:* `src/multistate.py` (transfer test, refit, validation, consistency, XGBoost check, production save; self-test). Also used: `src/build_dataset.py --states VA NC MD`, and `src/explain.py` (now state-aware).
- *Figures:* `reports/figures/11_*.png`.
- *Tables:* `reports/10_*.csv`.
- *App data:* `app/data/` (257 counties in three states).

*Area-level analysis of aggregated public data: every number describes the average for a group of similar insured buildings in an average year, never a prediction or quote for an individual property.*

## In one paragraph
The model built on Virginia **ranks risk well in North Carolina and Maryland without any change**, but it gets **North Carolina's level badly wrong**. NC's actual losses were **6.6x** what the Virginia model expected, almost all of it in four hurricane years. Refitting the same GLM on all three states with a **state factor** fixes the level (NC 6.2x Virginia, Maryland 1.0x) and keeps the building relativities nearly unchanged.

Most relativities point the same way in every state; their sizes differ. State-specific versions were tested and **made unseen years worse**, so they weren't adopted. The county credibility factor also **made unseen years worse**, badly in NC, where a county's history is mostly one storm's footprint, so the production model now has **no county factor**. XGBoost is more accurate on average, but it fails the pre-stated rule on catastrophe years, segment balance and (for one version) explainability, so **the GLM stays in production**.

## 1. Data: totals by state (2010-2025, losses in 2025 dollars)

| | Policy-years | Paid claims | Losses | Claims per 1,000 policy-years | Average paid claim | Loss cost per policy-year |
|---|---:|---:|---:|---:|---:|---:|
| Virginia | 1,482,413 | 9,557 | $292.7M | 6.45 | $30,625 | $197 |
| North Carolina | 1,945,568 | 37,825 | $1,923.0M | 19.44 | $50,840 | $988 |
| Maryland | 674,957 | 4,512 | $133.9M | 6.68 | $29,681 | $198 |
| **All three** | **4,102,937** | **51,894** | **$2,349.6M** | **12.65** | **$45,278** | **$573** |

- **Same recipe as Virginia:** the same 8 features and groupings, the same 1% rare-level rule (on the pooled table; no level is below 1% in any single state), CPI-U to 2025 dollars, and the same checks.
- **Checks:** all pass for each state and for the pooled table. The Virginia rows match the Virginia-only table, and Virginia's totals still equal the notebook 02 reference.
- **Tidal ("coastal_tidal") definitions** come from each state's own law. Sources are in `docs/modeling_dataset.md`.
  - Virginia: Tidewater, 46 localities.
  - North Carolina: the 20 CAMA counties, N.C.G.S. § 113A-103(2).
  - Maryland: the 16 Critical Area counties + Baltimore City, Md. Code Nat. Res. § 8-1807.
- **Cat years** (each state's 3 worst years by losses): VA 2011, 2012, 2016; NC 2016, 2018, 2024; MD 2012, 2016, 2018.
- **Data issue found:** 3 claims (2 NC, 1 MD) had negative payments (refunds). Each was floored at $0.

## 2. Transfer test: the Virginia model, unchanged (`11_transfer_ae_by_year.png`, `11_transfer_segments.png`, `11_transfer_lift.png`)

| | A/E pure premium | A/E frequency | A/E severity | Mean per-year improvement | Gini |
|---|---:|---:|---:|---:|---:|
| Virginia (in-sample) | 1.00 | 1.00 | 1.00 | 15.0% | 0.53 |
| North Carolina | **6.55** | 3.95 | 1.64 | 12.6% | 0.49 |
| Maryland | **0.99** | 1.07 | 0.88 | 13.9% | 0.51 |

*A/E = actual / expected losses. "Improvement" = the re-based Tweedie deviance improvement over a no-features model, scored within each state and year, so it measures ranking, not level.*

**What transferred well: the ranking.** Out of state, the Virginia model sorts buildings into risk order nearly as well as in Virginia: NC 12.6% and MD 13.9% improvement, vs 15.0% in-sample. In fact, on NC's unseen years the unchanged Virginia model ranks as well as a model refitted on all three states (section 4).

**What didn't: North Carolina's level.**
- **Storm years:** Florence 2018 had **46x** the Virginia model's expected losses, Matthew 2016 16x, Helene 2024 16x, Irene 2011 13x.
- **Quiet years:** NC is *below* the Virginia level (0.2x-0.8x).
- **Conclusion:** NC's gap is hurricane exposure, not different buildings.

Maryland's level is right (0.99); its worst year is Sandy 2012 (3.3x).

**Segments that didn't transfer** (A/E within each year, outside 0.8-1.25):
- Maryland:
  - contents-only policies 3.0;
  - non-residential 1.9;
  - non-tidal counties 2.1;
  - multi-unit 0.53.
- North Carolina:
  - built 2000 or later 1.7;
  - built 1960-79 0.66;
  - primary residences 1.4, second homes 0.74.

## 3. Refit with a state factor

Same structure as the Milestone 11 production model (same features and interactions), with two changes:
- **State x year effects:** each state gets its own effect for each year, because storms hit states in different years.
- **A state factor:** each state's average-year level vs Virginia, for the same building mix.

Loads: the large-loss load stays 1.000, the ICC bridge is pooled, and the off-balance is per state (each state's predictions add up to its actual losses).

| State factor vs Virginia | Frequency | Severity | Off-balance ratio | **Pure premium** |
|---|---:|---:|---:|---:|
| North Carolina | 3.51 | 1.73 | 1.03 | **6.22** |
| Maryland | 1.07 | 0.89 | 1.06 | **1.01** |

The building relativities (pooled) are close to Virginia's:
- B/C/X zone 0.34x;
- built before 1960 2.08x, 1960-79 2.00x, 2000+ 0.68x;
- non-residential 4.28x, multi-unit 2.77x;
- elevated 0.71x.

**Robustness:** a shared year effect instead of state x year effects changes no relativity by more than 3.5%.

**Check values** (1960s one-floor house in an A/V zone):
- Virginia Beach **$453.69**. The Virginia-only model gave $598.07: its county factor of 1.21 is gone, and the pooled 1960-79 frequency relativity is smaller.
- Dare County, NC $2,820.24.
- Worcester County, MD $458.67.

## 4. Validation: unseen years and an unseen state (`11_loyo_by_state.png`, `11_county_form.png`)

| Mean improvement | VA | NC | MD |
|---|---:|---:|---:|
| **3-state GLM, no county factor (production)**: 16-year LOYO | 11.0% | 12.5% | 13.3% |
| same: 2021-2025 holdout | 9.5% | 16.7% | 16.6% |
| 3-state GLM + pooled county factor: LOYO | 10.7% | **6.2%** | 14.6% |
| same: holdout | 11.3% | **0.04%** | 14.2% |
| Virginia model, unchanged: LOYO | (15.0% in-sample) | 12.6% | 13.9% |
| Same approach on that state alone (+ county factor): LOYO | 11.0% | 6.3% | 14.1% |
| **Leave-one-state-out** (fit on the other two states) | 10.9% | 13.2% | 12.9% |

- **Out of sample** (LOYO: each year predicted by a model that never saw it) and **out of state** (leave-one-state-out) tell the same story. The relativities travel; the level doesn't. Predicted from VA + MD, NC's raw A/E is 6.3, which is why each state needs its own level (the state factor).
- **The county factor:** three forms were compared under a rule written before the test.

  | County factor | LOYO mean (VA / NC / MD) | Average | Holdout (VA / NC / MD) |
  |---|---|---:|---|
  | frequency + severity | 10.71 / 6.24 / 14.57 | 10.51 | 11.28 / 0.04 / 14.22 |
  | severity only | 12.37 / 11.19 / 13.43 | 12.33 | 11.80 / 14.78 / 16.46 |
  | **none (chosen)** | 10.97 / 12.51 / 13.34 | 12.27 | 9.52 / 16.70 / 16.57 |

  - Severity-only has the best average, but only by 0.06 points, and its holdout falls below "none" in NC and MD. The rule allows at most one state, so **production has no county factor**.
  - Why it fails in NC: a county's flood history there is mostly one hurricane's footprint (Florence: the Neuse and Cape Fear river counties; Helene: the mountains), and the next storm goes elsewhere.

## 5. Consistency of relativities across states (`11_relativities_frequency.png`, `11_relativities_severity.png`)
The same model was fitted to each state alone, with storm-aware (county-group x year clustered) 95% CIs, and a cluster-robust Wald test of "same effect in all three states". Labels follow the rule pre-stated in PROGRESS.md:

| Label | Terms |
|---|---|
| **Consistent** (same direction everywhere; equality not rejected) | flood zone (frequency, p = 0.22), floors (severity, p = 0.58) |
| **Differs in size** (same direction, p < 0.05) | year built (frequency), occupancy, elevated, flood zone, elevated x zone (severity) |
| **Inconsistent** (a state's CI on the other side of 1) | floors and year built x zone (frequency); year built (severity) |
| **Unclear** (point estimates on both sides; no CI excludes 1) | coverage, coverage x occupancy, primary residence (severity) |

- **The big effects are consistent in direction:**
  - B/C/X zone lowers frequency (VA 0.51, NC 0.45, MD 0.58);
  - older buildings claim more (1960-79: VA 3.0, NC 1.5, MD 2.1);
  - non-residential claims are bigger (VA 4.6, NC 3.8, MD 6.6);
  - elevated claims are smaller (VA 0.72, NC 0.69, MD 0.93).
- **The inconsistent ones are small effects that flip sign.** For example, one-floor frequency is VA 0.89 vs NC 1.11.
- **Primary residence:** second homes have smaller claims in NC (0.76, plausibly Outer Banks beach houses) but not in VA (1.07) or MD (1.09).
- **Adopting state-specific effects:** 7 term x state interactions passed the per-term keep rule. All 7 together were **worse in every state** on both the 16-year LOYO (VA 10.71 -> 10.65, NC 6.24 -> 5.99, MD 14.57 -> 14.20) and the holdout, so none were adopted. They are flagged only.

## 6. Credibility: what more data changed (`11_credibility_z.png`, `11_county_history_vs_model.png`)

| Bühlmann-Straub | Frequency k | Severity k | Median severity Z |
|---|---:|---:|---:|
| Virginia alone (134 counties) | infinite (Z = 0) | 33 | 0.25 |
| North Carolina alone (101) | 230 | 57 | 0.55 |
| Maryland alone (25) | 84 | 51 | 0.64 |
| **Pooled (260)** | **160** | **41** | **0.41** |

*k = how many claims a county needs before its own history gets half the weight.*

- **More counties changed the estimate:** frequency differences between counties now look partly real (finite k), where Virginia alone said they were pure storm noise.
- **But the out-of-sample test** (section 4) showed they don't repeat. Credibility estimates *how much* counties differ in the data; only validation says whether that difference *predicts*.
- **The cost of no county factor:** the model's county estimates can be far from county history. Coastal resort counties ran lower (Brunswick 0.35x, New Hanover 0.28x, Worcester 0.39x), and a river county hit by one storm ran higher (Craven 3.6x).
  - Only about 16% of exposure sits in counties within 0.8-1.25x.
  - So the app's **certainty label** now also checks whether the county's history agrees with the model: High certainty needs Z ≥ 0.8 *and* agreement within 0.8-1.25x.

## 7. XGBoost check (`11_xgb_vs_glm.png`, `10_xgb_decision.csv`)
Frozen Milestone 10 settings (no re-tuning), refitted once per fold. `state` was added to the severity/Tweedie geography, and memory peaked at 1.3 GB, so no sampling was needed.

| Milestone 11 criterion | XGB Tweedie | XGB freq x sev |
|---|---|---|
| 1. LOYO better, bootstrap CI > 0 | pooled +3.8 pts (CI 1.6 to 5.5); **MD CI includes 0** | pooled +1.5 (CI 0.4 to 2.5); **NC, MD CIs include 0** |
| 2. Holdout better | yes, all states | yes, all states |
| 3. Not worse in cat years | **fails VA (13.2 vs 16.3), MD (15.4 vs 21.5)** | **fails VA (14.8 vs 16.3)** |
| 4. Double-lift end-decile error | passes | passes |
| 5. No segment newly outside 0.8-1.25 | **fails: B/C/X zone 0.67-0.79, coverage under $100K 0.61-0.71** | **fails: elevated 0.72-0.75, non-residential 0.66** |
| 6. Explanations | passes: monotone constraints, 0% wrong-way flips | **fails: elevation raises frequency for 96-98% of exposure** |

**The recommendation holds: the GLM stays in production.** XGBoost remains the accuracy benchmark. Its average edge is real, but it comes with worse catastrophe-year performance and unbalanced segments, which matter most for pricing.

## 8. What the app gets (`app/data/`)
- **257 counties/cities:** VA 133, NC 100, MD 24.
- Each is compared with **its own state's** average: VA $197, NC $988, MD $198 per policy-year (average year, 2025 $).
- `state_averages.json` replaces `virginia_averages.json`.
- `explain_county` / `explain_profile` take a state, or infer it from the county code. Names resolve per state ("Montgomery County" exists in all three, so it needs ", MD" or `state=`).
- Waterfalls start from the state average.
- **Certainty labels** (no county factor): High 12 counties (14% of exposure), Medium 46 (43%), Low 199 (44%). Each uncertainty note gives the county's history vs the model.
- **Tests:** `tests/test_explain.py` passes for every county in all three states, plus random profiles, cross-state name checks and rating-table parity.

## Limitations
- **Three neighbouring Atlantic states.** The NC state factor is dominated by four hurricanes in 16 years, so another 16 years could give a quite different level.
- **No geography inside a state** beyond the building mix: county estimates can differ a lot from county history (section 6).
- **Tidal definitions differ by state** (each state's own law), so "tidal" doesn't mean exactly the same thing in each.
- **NFIP buyers aren't random:** results describe the insured population.
- **Decisions made in this milestone** (all in PROGRESS.md):
  - dropping the county factor reverses Theo's Milestone 9 request (Questions for Theo 9);
  - the certainty label's agreement condition was added after seeing county history vs model.
- **Area level only.** Nothing here is a prediction for an individual property.

## Interview-ready sentences
1. "Before refitting, I applied my Virginia model unchanged to North Carolina and Maryland. The ranking transferred almost perfectly, but North Carolina's losses were 6.6 times the Virginia level, almost all in four hurricane years. That told me the relativities generalize but the level is state-specific, so I added a state factor."
2. "I checked whether each rating factor has the same effect in each state by fitting every state separately. The big effects, like flood zone, building age, occupancy and elevation, all point the same way, which is good evidence they're real. Where states differed, I tested state-specific effects on unseen years, and they made predictions worse, so I kept one set of relativities."
3. "With three states, the county credibility factor made North Carolina's held-out years much worse, because a county's history there is mostly one hurricane's footprint and the next storm hits somewhere else. Under a rule I wrote down before the test, I dropped the factor, and I made the app's certainty labels say where county history disagrees with the model."
4. "XGBoost was about four points more accurate on average, but under my pre-stated six-criterion rule it failed on catastrophe years and segment balance, so the explainable GLM stayed in production, and XGBoost remains the benchmark."
