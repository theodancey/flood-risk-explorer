# Report 06: Pure Premium: Frequency × Severity, County Credibility, Tweedie, Loss Ratios (Virginia, 2010–2025)

*Milestone 9, 2026-11-11.*
- *Source:* `notebooks/07_pure_premium.ipynb` (code: `src/models/pure_premium.py`).
- *Figures:* `reports/figures/07_*.png`.
- *Tables:* `reports/06_*.csv`.
- *Saved model for the app:* `models/pure_premium.json`, which points to `models/frequency_glm.json` and `models/severity_glm.json`.

*All modelled dollars are 2025 dollars at an average-year level. Loss ratios are nominal ÷ nominal. Area-level analysis of aggregated public data: nothing here is a prediction or a quote for an individual property.*

## In one paragraph
**What the model is.** The expected loss cost per policy-year (the **pure premium**) is the frequency GLM × the severity GLM, then × a large-loss load (1.000, no cap), × an ICC bridge (1.0092), × an **off-balance factor of 0.9933** (so predicted total losses equal actual), × a **credibility-weighted county factor**.

**Results.**
- **Base building** (A/V zone, single-family, built 1980–99, 2 floors, $250K+ cover, primary residence, not elevated): **$138 per policy-year**.
- **Biggest drivers:** year built 1960–79 **3.5×** and before 1960 **2.8×**; non-residential **3.1×**; multi-unit **2.1×**; B/C/X zone **0.34×**; contents-only 0.28×; built 2000+ 0.58×; elevated 0.76×.
- **County factor:** Bühlmann–Straub credibility found that county **frequency** differences are no bigger than storm noise (**Z = 0**). County **severity** differences are real (k ≈ 33 claims, Z up to 0.98). Factors run from **0.73 (York County) to 1.78 (Newport News)**.
- **County factor on unseen years:** it improves county-level fit (mean |log A/E| 0.50 → 0.34) and the overall fit (LOYO 10.2% → 10.7%; holdout 6.4% → 8.9%), but slightly worsens the development-year LOYO (12.7% → 12.2%).
- **Tweedie GLM (p = 1.7):** slightly better on overall deviance and Gini. But frequency × severity wins the **double-lift** test clearly, and the Tweedie gets some segments badly wrong. **The app should use frequency × severity + the county factor.**
- **Loss ratios:** Virginia's NFIP **loss ratio is 19% for 2010–2025** (107% in 2016, median year 10%). Relative to modelled risk, **pre-1980 buildings, A/V zones and Newport News / Portsmouth** look under-priced, and **post-1980 buildings and B/C/X zones** over-priced.

## 1. Combining the two models (`07_combined_relativities.png`)
> pure premium = exposure × frequency(avg year) × severity(avg year) × large-loss load × ICC bridge × off-balance × county factor

- **Average-year level:** both GLMs have a calendar-year factor. Predictions use the exposure-weighted average year effect, because next year's weather is unknown.
- **Large-loss load = 1.000.** Milestone 8 chose no cap. With a P99 cap ($241,051) the load would be 1.034, the base would be $139.55 instead of $137.83, and the non-residential relativity would move most (+13.5%). Every other level would move ≤ 5%.
- **Off-balance = 0.9933**, the product of three effects:

| Effect | Factor | Why |
|---|---|---|
| Mix (freq × sev cell by cell vs actual building + contents) | 0.9808 | the average of a product ≠ the product of the averages; a log-link Gamma balances ratios, not dollars |
| ICC (actual ICC share ÷ bridge factor) | 1.0005 | the bridge is an average |
| Unmatched claims (all losses ÷ matched-cell losses) | 1.0122 | 1.23% of losses sit in cells with no exposure |
| **Off-balance** | **0.9933** | predicted total = actual total = **$292,685,981** |

**Combined relativities** (`reports/06_combined_relativities.csv`; base = $137.83):

| Factor level | Frequency | Severity | **Pure premium** |
|---|---|---|---|
| Year built 1960–1979 | 2.75 | 1.28 | **3.52** |
| Year built before 1960 | 2.73 | 1.02 | **2.79** |
| Year built 2000+ | 0.64 | 0.90 | **0.58** |
| B/C/X zone | 0.46 | 0.75 | **0.34** |
| Non-residential | (1) | 3.07 | **3.07** |
| Multi-unit residential | (1) | 2.10 | **2.10** |
| Coverage $0 (contents only) | (1) | 0.28 | **0.28** |
| Coverage under $100K | (1) | 0.69 | **0.69** |
| Coverage $100K–$249K | (1) | 0.85 | **0.85** |
| Not primary residence | (1) | 1.12 | **1.12** |
| Elevated | (1) | 0.76 | **0.76** |
| 1 floor / 3+ floors | 0.89 / 1.13 | 1.12 / 0.89 | **1.00 / 1.01** |

(1) = the factor isn't in that model. **Zone and building age drive both parts and compound**: a 1960s building has 3.5× the loss cost. **Floors cancel out.**

## 2. Credibility-weighted county factor (`07_credibility_z.png`, `07_county_factors.png`, `07_county_ae_validation.png`)
Neither GLM has geography: county effects failed on unseen years in both milestones. The county factor blends each county's own experience with the statewide model:

> factor = [Z_f × A/E_freq + (1 − Z_f)] × [Z_s × A/E_sev + (1 − Z_s)], re-centred so the statewide total doesn't change

- **Frequency A/E** = actual claims ÷ GLM-expected claims. It uses the *fitted* year effects, so a storm year's overall level isn't counted as a county effect.
- **Severity A/E** = actual building + contents dollars ÷ expected.

**Two methods compared.**
- **Limited fluctuation (classical).**
  - Full credibility means ±5% with 90% probability, which takes n_full = (1.645 ÷ 0.05)² = **1,082 claims** for frequency.
  - For severity it's 1,082 × CV² = **2,713 claims** (CV = 1.583).
  - Partial credibility: Z = √(n ÷ n_full).
  - Weakness: it treats storm-clumped claims as independent.
- **Bühlmann–Straub (greatest accuracy).** It measures noise (EPV, year-to-year variation within a county) and signal (VHM, true differences between counties) from county × year data. Then k = EPV ÷ VHM and Z = w ÷ (w + k).
  - **Frequency:** EPV 17.1 and VHM **−0.049** (≤ 0), so k = ∞ and **Z = 0 for every county.** Counties' frequency differences are what storm noise alone would produce.
  - **Severity:** EPV 2.25, VHM 0.067, so **k = 33.4 claims**. Virginia Beach (2,072 claims) gets Z = 0.98; a county with 30 claims gets Z ≈ 0.47.

**Validation** (county factors estimated from the training years only, inside each fold):

| Model | LOYO 16 yrs: mean / non-cat / cat improvement | Gini | Dev LOYO 2010–20 | Holdout 2021–25 (Gini) | Held-out county mean \|log A/E\| | Counties within 0.8–1.25 |
|---|---|---|---|---|---|---|
| GLM alone (freq × sev) | 10.2% / 9.0% / 15.3% | 0.456 | 12.7% | 6.4% (0.428) | 0.497 | 26% |
| + limited-fluctuation factor | 9.8% / 10.2% / 7.9% | 0.458 | 10.7% | 6.7% (0.434) | 0.339 | 37% |
| **+ Bühlmann–Straub factor** | **10.7% / 11.3% / 8.3%** | **0.473** | 12.2% | **8.9% (0.472)** | **0.339** | 29% |
| Tweedie p = 1.7 (no county) | 11.5% / 11.8% / 10.1% | 0.486 | 13.0% | 9.8% (0.479) | 0.464 | 26% |

(Improvement = re-based Tweedie deviance, scored at p = 1.5, vs a no-features model. County A/E covers the 38 counties with ≥ 30 claims.)

**Honest reading:**
- **What it does:** both county factors cut the county-level error by about a third on unseen years.
- **Overall fit:** Bühlmann–Straub also improves the overall fit on all 16 LOYO years (better than the GLM alone in **13 of 16 years**) and on the holdout. Limited fluctuation hurts overall (10 of 16 years).
- **Where it costs:** on the development years alone, both slightly hurt, and both lose ground in catastrophe years. A storm's footprint differs from the counties' history, so there a county factor can point the wrong way.
- **Chosen: Bühlmann–Straub**, by the plan's rule: improve county A/E without worsening the overall held-out fit. It's also the method that accounts for storm clumping.
- **App label:** it's still labelled **less certain** in the app, as Theo asked in Question 7. Many counties remain outside 0.8–1.25 even with it.

**Selected factors** (`reports/06_county_factors.csv`):

| County | Claims | Freq A/E | Sev A/E | Z (sev) | **Factor** | LF factor |
|---|---|---|---|---|---|---|
| Virginia Beach city | 2,070 | 1.29 | 1.15 | 0.98 | **1.20** | 1.42 |
| Norfolk city | 1,332 | 1.14 | 0.80 | 0.98 | **0.84** | 0.95 |
| Hampton city | 834 | 0.64 | 0.76 | 0.96 | **0.81** | 0.57 |
| Chesapeake city | 619 | 0.96 | 0.78 | 0.95 | **0.83** | 0.85 |
| Fairfax County | 517 | 1.31 | 0.96 | 0.94 | **1.02** | 1.16 |
| Newport News city | 337 | 1.51 | 1.76 | 0.91 | **1.78** | 1.58 |
| Accomack County | 327 | 0.80 | 1.11 | 0.91 | **1.16** | 0.90 |
| York County | 75 | 0.27 | 0.55 | 0.69 | **0.73** | 0.73 |

- **Range:** 126 counties have claims. Factors run 0.73–1.78, and counties with no claims get the re-centring scale (1.052).
- **Gap to keep in mind:** big frequency A/Es (Danville 6.5, Pulaski 3.1) get **no** frequency credit, because Bühlmann–Straub judges them storm noise. Limited fluctuation would give them some.

## 3. Tweedie GLM for pure premium (`07_tweedie_power.png`, `07_ae_by_year.png`, `07_lift_and_double_lift.png`)
- **What it is:** a **compound Poisson–Gamma** model (Tweedie, 1 < p < 2) fitted directly to loss cost. It uses all seven features plus a year factor, log link, and exposure weights.
- **Power:**
  - *Theory:* p = (α + 2) ÷ (α + 1), with Milestone 8's Gamma shape α = 0.958, gives **1.51**.
  - *Grid on development LOYO:* 1.2 → 10.9%, 1.5 → 12.6%, **1.7 → 13.0% (best)**, 1.8 → 12.9%. **Chosen p = 1.7**, by the pre-stated rule (best on development years).

| Test | Freq × sev | Freq × sev + BS county | Tweedie p = 1.7 |
|---|---|---|---|
| LOYO 16 yrs mean improvement / Gini | 10.2% / 0.456 | 10.7% / 0.473 | **11.5% / 0.486** |
| Cat years (2011, 2012, 2016) | **15.3%** | 8.3% | 10.1% |
| Development LOYO | 12.7% | 12.2% | **13.0%** |
| Holdout 2021–25 | 6.4% | 8.9% | **9.8%** |
| Double lift: mean \|model − actual\|, end deciles | **0.17** | | 0.63 |
| Pooled lift: top ÷ bottom decile actual loss cost | 25× | | 27× |

- **A/E by year** is almost identical for the two structures: both are average-year rates, so 8.0 in 2016 and 0.06 in 2023.
- **Double lift:**
  - In the decile where the Tweedie predicts much more than frequency × severity, actual = 0.72× average, frequency × severity = 0.96×, Tweedie = **2.09×**.
  - Where frequency × severity predicts more, actual = 2.06×, frequency × severity = 2.06×, Tweedie = 1.19×.
  - **Where they disagree, frequency × severity is right.**
- **Segment balance (in-sample A/E, re-based per year, `reports/06_segment_balance.csv`):**
  - The Tweedie over-predicts **non-residential (A/E 0.50)** and **under-$100K coverage (0.44)** by about 2×.
  - Its coverage relativities run the wrong way: under $100K = 1.54× vs $250K+. Lower limits can't produce bigger losses.
  - Cause: a log-link GLM weights cells by mean^(1−p) = mean^(−0.7), so high-cost cells get little say in the fit.
  - Frequency × severity is closer on those segments but **not perfect either**: multi-unit 1.33, contents-only 1.61, elevated 0.85. The frequency model has no occupancy or elevation term, and these A/Es suggest those effects deserve another look (frequency second pass / XGBoost).

**Recommendation: the app uses frequency × severity with the Bühlmann–Straub county factor.**
1. The Tweedie's overall-score edge is small (≈ 1 point), and part of it comes from scoring on its own criterion.
2. The double lift, the segment sense, and catastrophe-year performance favour frequency × severity.
3. It shows "how often" vs "how big", which is the explanation the app promises.

The Tweedie stays as the challenger and as the baseline for an XGBoost Tweedie, which must pass the same segment-balance check.

## 4. Loss ratios (`07_loss_ratios.png`, `reports/06_loss_ratios.csv`)
- **Premium data:** the policy file has `totalInsurancePremiumOfThePolicy` (the premium) and `policyCost` (the premium plus the federal policy fee, HFIAA surcharge and reserve fund assessment). Both are earned day by day like exposure, and earned exposure reconciles exactly to the cell table.
- **2010–2025:** earned premium **$1,137M**, losses $220M (nominal), so the **loss ratio is 19.4%** (15.8% on policy cost). Average premium $767 per policy-year.
- **By year:** **2016 107%**, 2011 61%, 2012 34%, 2018 24%. Every other year is 2–14%, and the median year is 10%.
- **By factor (2010–2025):** loss ratios are higher for **pre-1980 buildings** (1960–79: 32%, before 1960: 18%) than for newer ones (1980–99: 11%, 2000+: 6%). They're also higher for multi-unit buildings (30%), under-$100K coverage (30%), A/V zones (21% vs 16%) and non-elevated buildings (23% vs 14%).

**Modelled loss vs 2023–2025 premium, relative to the statewide ratio** (> 1 = premium low relative to modelled risk):

| Segment | Relative indication | Segment | Relative indication |
|---|---|---|---|
| Built 1960–79 | **1.69** | Built 2000+ | **0.30** |
| Coverage under $100K | 1.47 | Built 1980–99 | 0.50 |
| Built before 1960 | 1.31 | Coverage $0 (contents only) | 0.60 |
| Coverage $100K–249K | 1.27 | B/C/X zone | 0.63 |
| A/V zone | 1.18 | Elevated | 0.78 |
| Newport News city | **2.40** | York County | **0.47** |
| Portsmouth city | 1.60 | Prince William County | 0.55 |
| Virginia Beach city | 1.28 | Chesapeake city | 0.75 |

- **Statewide:** 2023–25 modelled loss $48.6M vs earned premium $181.1M, a modelled loss ratio of 27%.

**Caveats:**
- **What premiums pay for:** NFIP premiums pay for expenses, catastrophe and reserve loadings, debt service, and **national** cross-subsidies (Virginia's premiums also pay for Gulf hurricanes), so a 19% loss ratio isn't "overcharging".
- **Risk Rating 2.0** (October 2021) re-priced every policy, with statutory caps on yearly increases (the "glide path"). Many older and pre-FIRM policies are still below their full-risk rate, which fits the older-building indications.
- **Only relative signals are meaningful.** Even those depend on our model, which has no geography beyond the county factor and no elevation data beyond a yes/no flag. RR2.0 itself uses much richer property data.

## 5. Rating worksheets (`07_rating_worksheets.png`, `reports/06_rating_worksheets.csv`)
Each worksheet is a typical insured building with these characteristics in that county, in an average year.

| Profile | Frequency per 1,000 | Severity | County factor | **Expected annual loss** | vs base |
|---|---|---|---|---|---|
| Virginia Beach, 1960s 1-storey house, A/V zone | 11.8 | $41,017 | 1.20 | **$585** | 4.2× |
| Norfolk, 2000+ elevated 2-storey second home, A/V | 3.1 | $21,623 | 0.84 | **$57** | 0.41× |
| Roanoke, pre-1960 1-storey house, B/C/X, under $100K cover | 5.4 | $16,866 | 0.99 | **$90** | 0.65× |
| Newport News, 1980s 3+ storey condo building, A/V | 5.5 | $59,477 | 1.78 | **$579** | 4.2× |
| Arlington, 1970s 2-storey commercial building, B/C/X, $100K–249K | 6.1 | $79,540 | 1.07 | **$516** | 3.7× |

Worked example (Virginia Beach):
1. Frequency: 4.82 per 1,000 × 2.748 (1960–79) × 0.892 (1 floor) = 11.82 per 1,000.
2. Severity: $28,498 × 1.282 (1960–79) × 1.123 (1 floor) = $41,017.
3. Loss cost: 11.82/1,000 × $41,017 = $484.81.
4. Loads: × 1.000 (load) × 1.0092 (ICC) × 0.9933 (off-balance) = $486.04.
5. County: × 1.204 (Virginia Beach, Z_sev 0.98) = **$585**.

## Limitations
- **Average-year rates.** A single year's actual losses can be 0.06× to 8× the prediction (2023 vs 2016). The pure premium is a long-run average, not a forecast for any year.
- **County factors are less certain.** They help on average but can point the wrong way in a catastrophe year, and frequency gets no county credit at all. With only 16 years and 3 big storms, the Bühlmann–Straub variance estimates are themselves noisy.
- **Segment imbalance.** The combined model under-predicts multi-unit (A/E 1.33) and contents-only (1.61), and over-predicts elevated (0.85) and non-residential (0.75) buildings. These are candidates for the frequency second pass and the XGBoost comparison.
- **Tweedie comparison scored on Tweedie deviance**, the challenger's own criterion. The double lift and segment balance were added for that reason.
- **Loss ratios** mix pre- and post-RR2.0 premiums and nominal dollars, and NFIP premiums aren't pure premiums. The relative indications are signals for discussion, not pricing recommendations.
- **Area level only.** Every number describes a group of similar insured buildings, never an individual property.

## Interview-ready sentences
1. "I combined a Poisson frequency GLM and a Gamma severity GLM into a pure premium by multiplying relativities, then applied an off-balance factor of 0.993 so total predicted losses equal actual. The factor comes mostly from the mix effect: the average of a product isn't the product of the averages."
2. "For geography I used Bühlmann–Straub credibility on county actual-to-expected ratios. It showed that county *frequency* differences in Virginia are pure storm noise (Z = 0), while county *severity* differences are credible after about 33 claims. On unseen years the factor cut county-level error by a third and improved the overall holdout fit."
3. "A Tweedie GLM with p = 1.7 scored slightly better on deviance and Gini, but the double-lift chart showed frequency × severity was right where the two disagreed. The Tweedie also over-predicted commercial buildings by 2× because of its mean^(1−p) weighting, so I recommended frequency × severity."
4. "Virginia's NFIP loss ratio was 19% over 2010–2025, but 107% in 2016. Relative to modelled risk, pre-1980 buildings look under-priced and newer buildings over-priced, which fits Risk Rating 2.0's glide path still phasing in full-risk rates."
5. "Every number is area-level: the app shows the expected loss for a typical insured building with given characteristics in a county, with the county factor labelled as less certain."
