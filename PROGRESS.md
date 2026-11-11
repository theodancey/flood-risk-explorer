# Progress Log

## Summary (latest: 2026-11-11, Milestone 8: severity GLM)
All six parts are done, committed, and pushed (no data files in Git).

- **Final severity model** (`notebooks/06_severity_glm.ipynb`, write-up **`reports/05_severity_glm.md`**): a **Gamma GLM with log link** on **9,553 individual paid claims** (building + contents, 2025 dollars), calendar year as a factor, average-year prediction. Features: **occupancy** (multi-unit **2.10×**, non-residential **3.07×**), **building coverage** (under $100K **0.69×**, contents-only 0.28×), **elevated 0.76×**, **B/C/X zone 0.75×**, **built 1960–79 1.28×**, floors (1: 1.12×, 3+: 0.89×), not primary residence 1.12×. Base claim $28,498 (average year). Storm-aware CIs are 1.7× wider than plain Gamma.
- **Modelling unit:** individual claims. The features come from the claim record itself, and the claims rebuild the cell table **cell by cell**. Claim-weighted cell averages give identical coefficients. **ICC is excluded** from the amount (0.96% of dollars) and added back as a bridge factor of 1.0092.
- **Distribution:** median $14K vs mean $30K, skewness 4.7. Top 1% of claims = 11% of dollars, top 5% = 31%. **178 claims (1.9%) hit a coverage limit**, holding 7.2% of dollars.
- **Large losses: no cap.** Uncapped is best on unseen years, and capping at P99 doesn't stabilize anything. Capping at P95 with a flat load moved cost from commercial and multi-unit buildings onto homes (multi-unit 2.1 → 1.7). NFIP limits already cap the tail. P99 (load 1.03) is kept as a documented sensitivity.
- **Selection** (same storm-aware + LOYO keep rule as frequency, 2010–2020 only): the forward pass kept year built, coverage, zone, floors. A **second pass** (added after a dry run showed order effects) added primary residence (narrowly) and elevated. **Geography failed again:** Tidewater isn't significant after clumping (p = 0.084), and county group makes unseen years worse (−0.91 points).
- **Frequency vs severity:** zone and building age drive **both**. Occupancy, coverage, elevation and second homes drive **severity only**. **Floors offset** (frequency 0.89 / severity 1.12, so pure premium 1.00).
- **Fit:** Gamma beats lognormal (AIC 197,754 vs 197,886; unseen-year improvement 19.2% vs 17.6%). The lognormal's mean is 0.54× uncorrected and 1.12× with the σ²/2 correction. The Gamma under-fits the top ~2% tail.
- **Held-out:** LOYO beats the no-features baseline in **14 of 16 years**, mean improvement **15.7%** (non-cat 17.1%, cat 9.8%), Gini 0.29, top/bottom decile 4.7×. **Holdout 2021–2025: 16.7%, A/E 1.02.** Raw A/E by year 0.49–1.77 (2016 Matthew). Leftover signal: county (Newport News 1.67, Norfolk 0.79) and a flat, under-predicted bottom of the lift chart.
- **Saved:** `models/severity_glm.json` (committed), pickle gitignored, `reports/05_*.csv`. Reusable code: **`src/models/severity_glm.py`** (with self-test). `frequency_glm.make_spec` gained a `weight` argument (backward compatible; frequency self-test still passes).
- Learning notes for Milestone 8 explain severity vs frequency, skewness and heavy tails, the Gamma distribution, policy limits and censoring, capping and large-loss loads, ICC, residuals and Q–Q plots, Gamma vs lognormal, order effects, and why drivers differ, with 4 interview Q&As.

**Judgment calls worth a look** (details under Milestone 8 decisions): (1) **no cap**, against the usual habit of capping; (2) the **second pass** was added after seeing the forward pass. It's the same rule, but it means the frequency model (forward pass only) should get the same check later; (3) **primary residence** is borderline (Wald p = 0.047, CI 1.00–1.25); (4) ICC handled as a bridge factor rather than modelled.

**What you need to do:** nothing required. Read `reports/05_severity_glm.md` (about 10 minutes) and the Milestone 8 learning notes. No new questions.

## Summary (previous: 2026-11-11, Milestone 7: frequency GLM)
All five parts are done, committed, and pushed (no data files in Git).

- **Final frequency model** (`notebooks/05_frequency_glm.ipynb`, write-up **`reports/04_frequency_glm.md`**): a Poisson GLM with log link, log(exposure) offset and calendar year as a factor, plus **flood zone, year built and floors**. B/C/X zones **0.46×** A/V (storm-aware 95% CI 0.34–0.61); built before 1960 **2.73×**, 1960–79 **2.75×**, 2000+ **0.64×** vs 1980–99; floors 0.89× (1) / 1.13× (3+). Base rate: 4.82 paid claims per 1,000 policy-years in an average year.
- **How features were chosen:** each shortlist feature was tested against the current model on 2010–2020 only. It was kept only if a **storm-aware (cluster-robust) Wald test** gave p < 0.05 **and** it improved **leave-one-year-out** deviance. AIC and plain LRTs would have kept 10 of 11 candidates. Coverage, occupancy, elevated, post-FIRM, primary residence and all three geography options (Tidewater, Tidewater + Census, county group) failed. County group cut training deviance by 1,170 but made 10 of 11 unseen years worse.
- **One-way vs GLM (confounding):** post-FIRM 0.35 → 0.79, elevated 1.08 → 0.99, second homes 1.13 → 0.83, coverage under $100K 1.89 → 1.14, non-tidal 1.37 → 1.09.
- **Over-dispersion:** clearly present (Cameron–Trivedi p < 0.0001). φ grows from about 2 to 12 when cells are combined: storm clumping. Compared quasi-Poisson, negative binomial (α = 3.6) and cluster-robust errors. **Choice:** Poisson relativities (they balance to actual claims in every level) with cluster-robust (county group × year) CIs, about 3.6× wider than plain Poisson. NB has a better AIC and is slightly better on unseen years (6.2% vs 5.6%), so it's the challenger for the model-comparison milestone.
- **Held-out:** LOYO over 16 years: beats the no-features baseline in **14 of 16 years**; mean re-based deviance improvement **4.7%** (2.5% normal years, 14.1% cat years); Gini about 0.30. Final 2021–2025 holdout: 2.2% improvement, Gini 0.26. Raw A/E is 5.6 / 1.9 / 4.8 in the Irene / Sandy / Matthew years and 0.1–0.7 in quiet years. That's what an average-year rate does (summed over all years, A/E = 1.00).
- **Saved:** `models/frequency_glm.json` (5 KB, committed; what the app will read), `models/frequency_glm.pickle` (1.4 MB, gitignored), `reports/04_frequency_relativities.csv`. Reusable code: **`src/models/frequency_glm.py`** (with a self-test).
- Learning notes for Milestone 7 explain regression, GLMs, Poisson, log link, offset, e^β, CIs, over-dispersion, deviance/AIC, LRTs and lift charts from scratch, with 4 interview Q&As.

**Judgment calls worth a look** (details under Milestone 7 decisions): (1) the keep rule is stricter than AIC/LRT, so the model is much smaller than the shortlist, and **geography is not in the frequency model**; (2) floors was kept narrowly and is doubtful under storm-aware CIs; (3) Poisson over NB despite NB's better AIC; (4) the 9-feature model is slightly better in 10 of 16 individual (mostly normal) years, but worse on average, in hurricane years and on the holdout.

**What you need to do:** nothing required. Read `reports/04_frequency_glm.md` (about 10 minutes) and the Milestone 7 learning notes. Question 7 below is optional.

## Summary (previous: 2026-11-11, Task 5: one-way analysis, coastal question, validation design)
All five parts are done, committed, and pushed (no data files in Git).

- **`notebooks/04_one_way_analysis.ipynb`** (run, 8 charts in `reports/figures/04_*.png`) covers every candidate feature: exposure, claims, frequency, severity, pure premium, relativities to the highest-exposure level, 95% CIs **widened for over-dispersion** (storm clumping makes plain Poisson intervals far too narrow), and credibility flags (1,082-claim rule). Reusable code is in **`src/one_way.py`** (with a self-test).
- **What matters most.** *Frequency:* building age (pre-1980 about 3× the 1980–99 rate) and post-FIRM (0.35×) are **stable in all 16 years**. Flood zone (B/C/X 0.41×) is real but much stronger in hurricane years. *Severity:* occupancy (multi-unit 2.3×, non-residential 3.1×), elevated (0.54×), second homes (1.7×). Ranked tables and the **week 3 GLM shortlist** are in **`reports/03_one_way_findings.md`**.
- **Confounding example:** elevated buildings look riskier one-way (1.08×) but are safer within *every* flood zone (zone-adjusted 0.86×), because 67% of them sit in A/V zones (Simpson's paradox). Post-FIRM is mostly building age (0.35× → 0.86× after adjusting for zone and year built).
- **Coastal (your Q6):** I added `coastal_tidal` = **Tidewater Virginia, Code of Virginia § 62.1-44.15:68** (46 localities, including Chesapeake). It beats the Census list on every measure: inland relativity 1.42 vs 1.04, and Poisson GLM deviance gain 39 vs 0.4. **Recommendation: use Tidewater.** Caveat: coastal effects flip between hurricane years (coast riskier) and normal years (inland riskier). Both flags are in the dataset, and small counties are now pooled by tidal/non-tidal.
- **Validation (`docs/validation_plan.md`, `src/validation.py`):** I compared a 2021–2025 holdout, leave-one-year-out (LOYO), and rolling origin. **Recommendation: LOYO as the main comparison** (scored per year, with and without the 3 cat years), development on 2010–2020 only, and the 2021–2025 holdout as a one-time final check. Reason: 2021–2025 has no hurricane, only 7% of claims, and 44% of its claims are inland (vs 12.5% before), so it's one unusual draw of the weather.
- Learning notes for Task 5 cover one-way analysis, relativities, CIs and credibility, confounding, why GLMs are needed, and holdout vs cross-validation. README and the project plan are updated.

**Judgment calls worth a look** (details under Task 5 decisions): (1) the validation headline changed from the planned 2021–2025 holdout to LOYO; (2) `county_group`'s small-county pools are now tidal/non-tidal (notebook 03 re-run); (3) notebook/report numbering shifted by one (GLMs are now notebook 05 / report 04).

**What you need to do:** nothing required. Read `reports/03_one_way_findings.md` (about 10 minutes) and the Task 5 learning notes.

## Summary (previous: 2026-11-11, Task 4: modelling dataset)
Done, committed, and pushed (no data files in Git).

- **`src/build_dataset.py`** builds **`data/processed/va_model_data.parquet`**: **252,531 rating cells** (county × year × 8 features), Virginia 2010–2025, with exposure, paid claims, all claims, and losses in nominal and **2025 dollars**. Runtime about 20 seconds.
- **The checks pass on every run** and stop the script if anything is off. Totals equal notebook 02 exactly: **1,482,412.77 policy-years, 12,545 claims, 9,557 paid, $220,183,315.29**. No negatives, no missing values, no duplicate cells, and every claim is counted once. The script also self-tests the grouping rules.
- **Features** (each complete in both files): flood zone (A/V vs B/C/X), occupancy (3 groups, consistent across Risk Rating 2.0), floors (1/2/3+), elevated, primary residence, post-FIRM, construction-year band (4), building-coverage band (4). Plus `county_group` (32 credible counties + 3 pooled groups) and `coastal`. **Rare-category rule:** every level holds ≥1% of exposure.
- **Inflation:** CPI-U from the **BLS public API** (free, no key). Losses rise 33% to $292.7M in 2025 dollars, and the loss cost is about **$197/policy-year** (vs $149 nominal).
- **Unmatched claims:** **95 paid claims (0.99%, 1.23% of losses)** have feature combinations with no in-force policy. They're kept with exposure 0 and a flag (frequency models drop them; severity uses them), and the reasons are reported.
- New docs: **`docs/modeling_dataset.md`** (every column and rule, CPI method, leakage, limitations), **`notebooks/03_dataset_summary.ipynb`** (run, 2 figures), Task 4 in `LEARNING_NOTES.md`. README and the project plan are updated.
- `src/download_data.py` has two new options, `cpi` and `coastal` (small files). `openpyxl` was added to requirements (to read the Census `.xlsx`).

**Judgment calls worth a look** (details under Task 4 decisions): (1) the full county is kept in the cell key, so there are many small cells (median 1 policy-year, 98% with zero claims, which is normal for Poisson); (2) coverage bands stay in nominal dollars on purpose; (3) October 2025 CPI is missing (shutdown) and was interpolated; (4) Chesapeake counts as *inland* on the official coastline list (Question 6).

**What you need to do:** nothing required. Read `docs/modeling_dataset.md` and the Task 4 learning notes. Question 6 is an FYI.

## Summary (previous: 2026-11-10, Task 3: actuarial basics + project plan)
All five parts are done, committed, and pushed (no data files in Git).

- **`notebooks/02_actuarial_basics.ipynb`** (run, outputs saved) computes **earned exposure**, **frequency**, **severity**, and **pure premium** for Virginia by year and by county. It has 5 new charts in `reports/figures/02_*.png`. All the built-in checks pass: the toy-policy test, exposure reconciling to total days of cover (within 0.01%), every claim landing in the tables, and frequency × severity = pure premium.
- **`src/actuarial.py`**: reusable code for these measures, so the modelling weeks compute them the same way. It also saves `data/processed/va_county_year.parquet`.
- **`reports/02_actuarial_basics.md`**: 8 interview-ready findings. Headline: Virginia's loss cost is **about $149 per policy-year** (6.5 paid claims per 1,000 policy-years × $23,039 average paid claim, 2010–2025). The **worst 3 years produced 69% of losses**, and the median year's loss cost ($79) is about half the average.
- **`docs/PROJECT_PLAN.md`** (one page, with the week 2–9 schedule), a rewritten **README.md** with "Key findings so far", and a **Task 3** section in `LEARNING_NOTES.md`.
- `src/download_data.py counties` now downloads a tiny free Census file of county names (used for chart labels). `scipy` was added to `requirements.txt` (it was already installed, and the notebook now uses it directly).

**Two judgment calls you should know about** (details under Decisions): (1) **2009 exposure is estimated**, because the policy file is missing policies that started in 2008, so headline numbers use 2010–2025; (2) exposure counts **one per policy term**, not per condo unit.

**What you need to do:** nothing required. Read `reports/02_actuarial_basics.md` and the new learning notes before interviews. Question 5 is a heads-up about week 8.

## Summary (previous: Task 2, data download and first look)
The data is downloaded and explored. Everything except the data files is committed and pushed.

| File (in `data/raw/`, not in Git) | Rows | Size |
|---|---|---|
| `NfipClaimsV3.parquet`: all US NFIP claims, 84 columns | 2,725,989 | **216.2 MB** |
| `NfipPoliciesV3_VA.parquet`: Virginia policies, 37 columns | 1,668,793 | **49.4 MB** |
| `NfipClaimsV3_fields.json`, `NfipPoliciesV3_fields.json`: FEMA's official field descriptions | n/a | < 0.1 MB each |

- **Disk check:** C: had 812.9 GB free, well over the 10 GB minimum.
- **Heads-up:** FEMA has **retired the v2 datasets** you named (`FimaNfipClaims`, `FimaNfipPolicies`, which now return 404). I used their official replacements, **`NfipClaims` v3 and `NfipPolicies` v3**. See Decisions.
- New files: `src/download_data.py`, `docs/data_dictionary.md`, `notebooks/01_first_look.ipynb` (run, outputs saved), 7 charts in `reports/figures/`, `reports/01_data_overview.md`, and a Task 2 section in `LEARNING_NOTES.md`.
- Headline findings: claims are dominated by a few catastrophe years. Virginia's claims are concentrated in Hampton Roads. Claim sizes are very skewed (median $13K vs mean $42K). Policy data only starts in 2009. Details are in `reports/01_data_overview.md`.

**What you need to do:** nothing required. To get the data on another computer, run `.venv\Scripts\python.exe src\download_data.py all` (about 35 minutes). Optionally, confirm you're happy with the v2 → v3 switch (Question 3).

## Questions for Theo
7. *(Optional, Milestone 7)* **Geography in the app.** The frequency model has no county or coastal factor, because those effects didn't hold up on unseen years. So for now, county differences in the app will come only from each county's mix of flood zones and building ages (the severity model, next milestone, may add geography back). If you'd rather the app show a county factor anyway (e.g. a credibility-weighted one, clearly labelled as less certain), tell me and I'll plan it for the app milestone. Otherwise I'll test geography again with XGBoost. ✅ Theo: Yes, plan a credibility-weighted county factor for the app milestone (blend each county's experience with the statewide average based on its exposure; explain the credibility method used and label it clearly as less certain in the app). Also still test geography with XGBoost as planned.
   *Status (Milestone 8):* added to the app row of `docs/PROJECT_PLAN.md`. The severity model also dropped geography (Tidewater and county group failed on unseen years), so the credibility-weighted county factor will cover both frequency and severity. Out-of-fold severity A/E by county (Newport News 1.67, Norfolk 0.79) shows it is needed.

5. *(Week 8, decided)* **Hosting the app.** The plan puts the Streamlit app on a free host (Streamlit Community Cloud), which needs an account linked to your GitHub. **Decision: you'll create the Streamlit Community Cloud account yourself in week 8.** I won't sign up for anything.

### Earlier (resolved)
6. *(Resolved in Task 5)* **Coastal definition.** Original note: the Census coastline-county list (25 localities) leaves out Chesapeake, which holds a third of "inland" exposure. ✅ Theo: compare it with a broader "tidal" definition, cite the source, and keep whichever separates risk better (or explain why neither does). **Outcome:** Tidewater Virginia (Code of Virginia § 62.1-44.15:68, 46 localities) separates risk clearly better and is recommended. Both flags are kept as features. See `reports/03_one_way_findings.md`.
3. *(Resolved)* **v2 → v3.** The v2 datasets are gone from OpenFEMA, so I used v3, which is the same NFIP program data under new names, with a few renamed fields.
4. *(Resolved)* **FEMA's one-file claims download refused our script (HTTP 403 "Forbidden").** The script downloaded the same data through the official API instead. If you ever want the single file, download `https://www.fema.gov/about/reports-and-data/openfema/v3/NfipClaimsV3.parquet` in your browser and save it as `data/raw/NfipClaimsV3.parquet`.
1. *(Optional)* To pick this project's Python from a list inside Jupyter (outside VS Code), run once:
   `.venv\Scripts\python.exe -m ipykernel install --user --name flood-risk-explorer`
   I skipped this because it writes to `%APPDATA%\jupyter`, which is outside the project folder. VS Code finds `.venv` without it. Resolved
2. *(FYI)* You turned off Windows Smart App Control because it was blocking SciPy's `.pyd` files. I re-ran the check afterwards and everything imports, including a direct SciPy test. Smart App Control can't be turned back on without resetting Windows, so you may want to read up on what that means for your security. That's your call; I didn't change any system settings. Resolved

## Decisions
- **Python 3.13 instead of 3.15.** On 3.15, `pip install` failed: `pyyaml` (needed by the Jupyter packages) has no prebuilt wheel for 3.15 yet. pip tried to compile it and stopped with "Microsoft Visual C++ 14.0 or greater is required." As instructed, I ran `py install 3.13` (installed 3.13.17) and rebuilt `.venv`. Everything installed from prebuilt wheels. I didn't use the Astral CPython 3.14 that's also on the machine, because you named 3.13.
- **Two requirements files.** `requirements.txt` lists the 14 libraries we chose, with pinned versions (easy to read). `requirements-lock.txt` is the full `pip freeze`, including dependencies, for exact reproducibility. Note: pip chose a pre-release of `wrapt` (2.6.0.dev1), which is a dependency of a dependency. It works fine; worth watching if anything odd shows up later.
- **`check_setup.py` recreates `data/raw` and `data/processed`.** `data/` isn't in Git, so a fresh clone won't have those folders.
- **Jupyter version shown is `jupyter-core`'s.** The `jupyter` bundle has no version of its own.

### Task 2 decisions (2026-11-10)
- **Used NFIP v3 datasets.** `api/open/v2/FimaNfipClaims` and `.../FimaNfipPolicies` return 404, and the OpenFEMA catalog now lists `NfipClaims` v3 and `NfipPolicies` v3 instead.
- **Claims came through the API, not the single-file download.** The full-file URL returned 403 to an honestly identified script. The download script tries the file first and falls back to the official API, which gives the same data. That's 273 pages of 10,000 rows, about 25 minutes.
- **Keyset pagination instead of `$skip`.** The Virginia policies download started with `$skip` paging. By page 40 or so it had slowed from about 5 s to about 60 s per page, with timeouts, because the server has to count past every skipped row. I switched to "next 10,000 rows with `id` greater than the last one," which stayed at about 5 s per page. The 44 pages already saved were kept and the download resumed.
- **Page files are saved as they arrive, then combined.** Each page is written atomically (temporary file, then rename), so a crash never leaves a half-written page. After combining, the row count is checked against the API's total before the page files are deleted. Both files matched exactly.
- **Column types come from FEMA's official field list.** Every page uses the same types (int, float, bool, timestamp, text), so pages combine cleanly. ZIP, FIPS, and census codes stay as text, which keeps leading zeros.
- **37 policy columns** (list in `src/download_data.py`: dates, exposure/coverage/premium, location, flood zone, building characteristics).
- **County names typed in by hand** for the 30 counties in the two top-15 charts (Census names, each cross-checked against FEMA's community names in the data). This avoided an extra non-FEMA download.
- **No new packages.** Everything used was already in `requirements.txt` (nbconvert/nbformat come with `jupyter`).

### Task 3 decisions (2026-11-10)
- **Earned exposure method.** Cover runs from `policyEffectiveDate` to `policyTerminationDate`, stopping at `cancellationDateOfFloodPolicy` if that's earlier. A cancellation on or before the start date means 0 exposure (1,219 terms). Each year earns (days of cover in the year) ÷ (days in the year), so a full year = 1.0. Check: total earned over all years = 1,624,225 policy-years vs total cover days ÷ 365.25 = 1,624,118.
- **One policy term = one unit (not weighted by `policyCount`).** Each claim record is one policy's claim, so this keeps the top and bottom of the frequency ratio in the same unit. Weighting by `policyCount` (condo units) would add 13%. That version is shown in the notebook for comparison.
- **2009 exposure is estimated; headline window 2010–2025.** The policy file only has terms that *start* on or after 2009-01-01, so 2009's raw earned exposure (44K) is only 46% of the truth. Estimate: assume 2008's policies looked like 2009's, which gives ≈ the total earned by 2009-starting terms (96K). 2009 is kept in charts (hatched bar) because it holds Virginia's biggest event in the data (the Nov 2009 nor'easter, $106M), but averages use 2010–2025. 2026 is excluded (incomplete).
- **Total paid = building + contents + ICC (`amountPaidOn*`), with blanks as $0.** This matches notebook 01 and is what the insurer paid. ICC is part of the policy's coverage. Cross-checked against the `net*PaymentAmount` fields: totals differ by 0.12%.
- **Claim = paid claim (> $0)** on both the frequency and severity side, so frequency × severity = pure premium exactly. 24% of 2010–2025 claims closed without payment. The all-claims frequency is shown alongside.
- **County ranking uses a credibility floor:** only counties with ≥5,000 policy-years in 2010–2025 are ranked (32 of 141, holding 90% of exposure). Smaller counties swing wildly on one or two claims.
- **County names from the Census Bureau** (`national_county2020.txt`, free, no sign-up) via a new `counties` option in `src/download_data.py`. Saved to `data/raw/` (not in Git). This replaces hand-typing names.
- **Dollars stay nominal** for now. Inflation adjustment is planned for week 2 (see `docs/PROJECT_PLAN.md`).
- **Chart palette:** kept notebook 01's blue/orange pair (a standard colorblind-safe pair). The palette-checking script I'd normally run needs Node.js, which isn't installed, and I didn't install system software to run it.

### Task 4 decisions (2026-11-11)
- **Rating cell = year × county × 8 features.** Policies and claims share no ID, so both are put into cells with the same `make_features()` rules. The full county stays in the key (it can be pooled later but never split again, and the app needs counties). The models use `county_group`.
- **Features kept:** only those complete and coded alike in both files. **Rejected:** basement type (54–58% blank), CRS class (63% blank in claims vs 18% in policies), foundation type (mostly "not reported" pre-RR2.0), deductible (code set changed with RR2.0; a candidate for later), `floodZoneCurrent` (possible post-loss map change = leakage risk), ZIP (too sparse).
- **Occupancy groups are time-consistent:** {1, 11, 14, 16} / {2, 3, 12, 13, 15} / {4, 6, 17, 18, 19}. Pre-RR2.0, condo units and mobile homes were coded 1.
- **Rare-category rule: each level ≥1% of 2010–2025 exposure**, merged into the most similar level (V → A, split-level → 2 floors, mobile → 1 floor). Blanks/implausible values go to the most common level: 114 zones, 237 floors, 728 construction years (policy terms). Counts are printed every run.
- **`county_group`:** a county keeps its own level at ≥5,000 policy-years (the same floor as notebook 02's ranking; 32 counties). The rest are pooled by coastal/inland. Blank and **out-of-state census codes** (8 codes, 52 policy-years, 0 claims, almost certainly geocoding errors) go to `Other/unknown county`. They're kept, so totals still reconcile with notebook 02.
- **Coverage bands are nominal.** The $250K NFIP limit is fixed in nominal dollars, so restating limits would move buildings between bands just because time passed.
- **CPI source: BLS API, not FRED.** FRED's CSV link answered `curl` but timed out for every Python request (likely bot filtering). I didn't try to get around it; I used the BLS Public Data API v1 (the original publisher, no key). Same series (CUUR0000SA0 = CPIAUCNS). Annual averages; factor = CPI(2025)/CPI(year of loss). **October 2025 wasn't published** (government shutdown) and was linearly interpolated (2025 average 322.153 vs 321.943 from 11 months, 0.07%). Check: 2010 = 218.056, matching BLS.
- **Coastal flag** from the Census Bureau's 2016 coastline-county list (`coastline-counties-list.xlsx`, Aug 2018 release). Needed `openpyxl` (installed in `.venv`, added to requirements).
- **Unmatched claims kept, not forced into a nearby cell:** 111 cells / 122 claims (95 paid) with exposure 0 and `exposure_matched=False`. None are in a county-year without exposure. Ignoring county alone would re-match 118, and ignoring coverage band 105, so they're rare combinations (probably small differences between claim and policy records, or missing policy rows).
- **Checks are assertions inside the script** (rather than a pytest suite), so they run on every build without a new package.
- Project-plan week-2 file names updated to the real ones (`build_dataset.py`, `va_model_data.parquet`).

### Milestone 8 decisions (2026-11-11)
- **Individual claims as the modelling unit.** Features are on the claim record (same `make_features()`), so attaching them needs no claim-to-policy link. `severity_glm.build_claims_table()` asserts that the claims rebuild every cell's paid-claim count and 2025-dollar losses. The claim table is cached in `data/processed/va_claims_severity.parquet` (not in Git). The 95 unmatched claims are kept.
- **Amount = building + contents, 2025 $; ICC excluded and bridged.** ICC is 0.96% of dollars, 78 claims, 4 ICC-only (b+c = $0, which a Gamma can't take). Bridge factor = (total paid per paid claim) ÷ (b+c per claim with b+c > 0) = 1.0092, so pure premium = frequency × severity × bridge reproduces total paid. Including ICC changes relativities by at most 8% (built 2000+).
- **Base levels = the frequency model's**, not the most-claims level, so the two relativity tables line up level for level.
- **No cap.** Decided from the development-years comparison (none / P99 / P95, cap set per training fold). See report 05 §3. The cap and load machinery stays in the module for XGBoost.
- **Keep rule unchanged from frequency** (cluster Wald p < 0.05 by county group × year AND better dev-LOYO re-based deviance). In-sample tests are reported but not used. AIC and the F-test would have kept 9 of 10.
- **Second pass added after a dry run** showed elevated failing only because it was tested before zone and year built. It re-tests every dropped feature against the current model with the same rule and adds the best passer each round. Disclosed in the notebook and report. The frequency model had no second pass, so that's a follow-up for the model-comparison milestone (added to the plan).
- **Gamma log link doesn't balance dollars exactly** (it balances the average ratio actual/predicted per level). The average-year level is chosen so total predicted = total actual training dollars, so overall calibration holds by construction.
- **Cluster-robust errors require unweighted rows** (statsmodels warns otherwise), so weights are only passed for cell averages.
- **Lognormal comparison** uses AIC on the dollar scale (with the Jacobian, Σ log y) and an ML-shape Gamma likelihood, so both have their own fitted spread. Q–Q plots use quantile residuals for the Gamma.
- **Numbering:** pure premium becomes report 06 / notebook 07, XGBoost notebook 08, the model comparison report 07 / notebook 09 (`docs/PROJECT_PLAN.md` updated, week 4 split into 4a done / 4b next).
- The notebook generator lives outside the repo (as for notebooks 04–05). No new packages.

### Milestone 7 decisions (2026-11-11)
- **Year as a factor in every model; average-year prediction.** Storm years would otherwise distort the building relativities. An unseen year is predicted at the level where training predictions sum to training claims (`FreqGLM.predict(..., "average")`).
- **Keep rule = cluster-robust Wald p < 0.05 AND better dev-LOYO** (mean re-based % improvement over the no-features model, each year weighted once). Decided before running the selection. Plain LRT/AIC assume independent claims and would have kept 10 of 11 candidates. The quasi-F (cell-level φ of about 2) barely changes that, because cell-level φ understates storm clumping.
- **Forward selection in shortlist order**, starting from flood zone (as asked). Geography: (A) Tidewater, the Milestone 6 recommendation; (B) + Census coastal; (C) county group replacing the flags (it's collinear with them). All failed on unseen years. The result is order-dependent in principle, but the rejected features fail by clear margins, except post-FIRM (−0.02 points, a tie) and floors (+0.08, kept).
- **Relativity table from a refit on all 16 years.** Selection used 2010–2020 only; validation refits inside each fold.
- **Intervals: cluster-robust by county group × year** (up to 560 clusters on all years). The year-cluster version (16 clusters) is shown as a pessimistic sensitivity check. **Poisson, not NB, for the relativities:** balance by level, invariance to how cells are aggregated, standard practice; NB's gain on unseen years is 0.6 points vs 5.6 for the features themselves. NB is re-tested in the model-comparison milestone.
- **Fitting speed:** hand-built 0/1 design matrices instead of statsmodels formulas (18 s → about 0.2–1 s per fit). Cells with identical features + county group + year are summed first (173,363 rows); this leaves coefficients and deviance differences unchanged (checked in the self-test).
- **Lift chart ranking:** cells are ranked by the product of their feature relativities. My first two versions ranked by re-based, then fold-normalized predictions, which partly ranked *years* (storm-year cells sorted high). The first version showed a misleading 56× top/bottom ratio. Per-fold Gini is computed within a single year and wasn't affected.
- **Comparison model:** a "full" GLM with all nine shortlist features is reported alongside the final model, to show what the dropped features do: slightly better in most normal years, much worse in hurricane years and on the holdout.
- **`models/*.pickle` gitignored** (1.4 MB, version-fragile, rebuildable). `models/frequency_glm.json` (5 KB) is committed and carries everything needed to predict, plus the relativity table.
- **Report numbering:** frequency got its own report 04, so severity/pure premium becomes report 05 and the model comparison report 06 (`docs/PROJECT_PLAN.md` updated).
- The notebook generator script lives outside the repo (as for notebook 04). The notebook itself is the reproducible artifact. No new packages.

### Task 5 decisions (2026-11-11)
- **Tidal source: the statute itself.** "Tidewater Virginia" in Code of Virginia § 62.1-44.15:68 (law.lis.virginia.gov, fetched 2026-11-11). The same localities define Virginia's Coastal Zone Management area, but DEQ's website refused automated access (403), so I cite only the statute. The 46 names were matched to FIPS codes via the Census names file, and the build checks that each name matches. All 25 Census coastline localities are inside Tidewater.
- **Small-county pooling switched to Tidewater** (`Other tidal (small)` / `Other non-tidal (small)`). The old `Other inland (small)` mixed 5.5 and 9.3 claims per 1,000. Notebook 03 was re-run (only its county-group table labels changed). Totals are unchanged.
- **One-way tables use every row**, including the 1% of unmatched claims, so frequency × severity = pure premium exactly. The GLMs will drop them for frequency, as documented.
- **Over-dispersion-widened CIs** are shown in the charts (√φ, with φ from the year × level table; a level-only model for a level's frequency, level + year for relativities, because common storm swings cancel in a ratio). Plain exact Poisson CIs are in the tables too. Severity CIs use the Gamma dispersion.
- **Importance score** = one-way deviance drop ÷ parameters ÷ φ (a quasi-likelihood F). It ranks features without being fooled by over-dispersion.
- **Stability verdict rules** are stated in the notebook: "flips" if the cat-year and normal-year relativities sit on opposite sides of 1; "storm-dependent" if they differ by more than 2×; otherwise stable / mostly stable / mixed by share of years pointing the same way. My first rule compared all years with excluding-cat years and mislabelled the coverage band as stable, so I changed it.
- **No dual-axis charts.** The classic exhibit (exposure bars + relativity line on two y-axes) was drawn as stacked panels sharing the x-axis, so no number is read off the wrong scale. Same blue/orange palette as notebooks 01–03, plus aqua for pure premium.
- **Validation: LOYO headline, holdout as final check** (reasons in `docs/validation_plan.md`). Development uses only 2010–2020. Caveat noted: notebook 04 itself looked at all years before the shortlist was written.
- **Renumbering:** this task took notebook 04 and report 03, so the GLMs move to notebook 05 / report 04, etc. (`docs/PROJECT_PLAN.md` updated, with a "2b" row for this task).
- **2024 attribution:** 75% of 2024's claims were in non-tidal localities. I attributed this to Hurricane Helene's remnants flooding southwest Virginia (Sept 2024), and checked it against the claims' `floodEvent` field: 109 of the 139 non-tidal paid claims in 2024 are labelled "Hurricane Helene."
- No new packages.

## Log
### 2026-11-11: Milestone 8, severity GLM
1. Read the plan docs. Committed your Q7 answer. Probed the claims file: ICC is 0.9% of dollars, 85/104 building/contents limit hits. (My first probe command hung on a stray `cat`, so I stopped it and re-ran it as a script.)
2. Wrote `src/models/severity_glm.py` with a simulated-data self-test. Two first-run fixes: the dollar-balance check (a log-link Gamma only balances ratios) and the noise-feature check (noise passed the rule by chance; the test now checks effect sizes). Built the claim table: it reconciles cell by cell. Committed.
3. Dry-ran the cap comparison and selection, which surfaced the order effect behind the second pass. Built notebook 06 from a generator and ran it with nbconvert (no errors first time). Checked all 9 charts. Fixed the broken log-axis labels, overlapping percentile labels and raw feature names. Rewrote the residual narrative (my pre-written text had the Gamma's tail misfit backwards). Committed.
4. Saved the model JSON and tables. Committed. Wrote report 05, checking numbers against the notebook. A spot check corrected my explanation of the under-predicted bottom decile (it's elevated buildings with $100K+ cover, not contents-only claims). Committed.
5. Updated LEARNING_NOTES, README, PROJECT_PLAN, and this file. Committed and pushed.

### 2026-11-11: Milestone 7, frequency GLM
1. Read the plan docs. Confirmed `main` and that the existing self-tests pass. Profiled fit speed: formulas took 18 s per fit, so I built the design matrices by hand.
2. Wrote `src/models/frequency_glm.py` with a simulated-data self-test (recovers known relativities, balance, offset, aggregation invariance, JSON round trip). Raised the simulated claim rate after one noise-driven failure. Committed.
3. A dry run showed in-sample tests keeping everything, while LOYO rejected most features. I replaced the step-comparison helper with `forward_selection` and a pre-stated keep rule.
4. Built notebook 05 from a generator script and ran it with nbconvert. Fixed a bool-index bug in the A/E chart. Corrected four write-up claims that the results contradicted (NB *is* slightly better out of sample; full vs final is mixed; φ is not monotone in aggregation; one-way values without unmatched claims). Fixed the lift-chart ranking twice (see decisions) and chart layout issues (overlapping titles, legend, footnote, CI order, tick labels). Committed notebook + 7 figures.
5. Saved the model JSON (pickle gitignored) and tables. Committed. Wrote `reports/04_frequency_glm.md`, checking numbers against the notebook output. Committed.
6. Updated LEARNING_NOTES, README, PROJECT_PLAN, and this file. Committed and pushed.

### 2026-11-11: Task 5, one-way analysis + coastal + validation
1. Committed your Q6 answer. Fetched the Tidewater statute, mapped 46 localities to FIPS, and added `coastal_tidal` with checks to `build_dataset.py`. Rebuilt (totals unchanged). Committed.
2. Wrote `src/one_way.py` and `src/validation.py` with self-tests. Fixed two edge cases the first runs hit (Cramér's V of a column with itself; stratifying a feature by itself). Committed.
3. Built notebook 04 from a generator script and ran it with nbconvert. Found the Census-vs-Tidewater result and the mixed small-county pool, so I switched pooling to Tidewater, rebuilt, and re-ran notebook 03. Committed.
4. Checked every chart visually and fixed overlapping titles, `$` signs rendering as math, clipped points, colliding labels, and an invisible low-credibility hatch (now labelled). Fixed the stability verdict rule. Committed notebook 04 + 8 figures.
5. Wrote `docs/validation_plan.md` (numbers re-checked against the data). Committed.
6. Wrote `reports/03_one_way_findings.md`, checking each number against the notebook output. Committed.
7. Updated LEARNING_NOTES, README, PROJECT_PLAN, and this file. Committed and pushed.

### 2026-11-11: Task 4, modelling dataset
1. Planning: read the docs, profiled every candidate feature in both files, and prototyped the cell grouping (cell counts, unmatched claims under different groupings).
2. Committed your README pitch and the PROGRESS question updates.
3. Added `cpi` and `coastal` to `download_data.py`. FRED timed out from Python, so I switched to the BLS API and downloaded both. Let the `actuarial.py` loaders take extra columns. Committed.
4. Wrote `src/build_dataset.py`. All checks passed on the first run. Made the unmatched-claim reasons readable. Found and pooled the out-of-state census codes. Confirmed two runs give identical output (same hash). Committed.
5. Wrote `docs/modeling_dataset.md`. Committed.
6. Built and ran notebook 03 with nbconvert (no errors). Checked both charts and moved a legend that overlapped the bars. Committed.
7. Updated LEARNING_NOTES, README, PROJECT_PLAN, and this file. Committed and pushed.

### 2026-11-10: Task 3, actuarial basics + project plan
1. Checked the policy data during planning: term lengths, cancellations, `policyCount`, and the 2009 start-date gap. Also checked which claim payment fields are complete.
2. Wrote `src/actuarial.py` (with a self-test on made-up policies) and added the Census county-names download. Ran it: the self-test passed and the claim totals match the raw data. Committed.
3. Built `notebooks/02_actuarial_basics.ipynb` and ran it with nbconvert: no errors, all asserts pass. Checked every chart visually. Fixed a duplicated "$1K" axis label, overlapping labels on the loss-concentration chart, and an over-strong chart title. Re-ran and committed.
4. Wrote `reports/02_actuarial_basics.md`. Re-checked every number against the data before committing. That caught and fixed three wrong totals in my first draft. Committed.
5. Wrote `docs/PROJECT_PLAN.md`. Committed. Rewrote `README.md` and added `scipy` to `requirements.txt`. Committed.
6. Updated this file and `LEARNING_NOTES.md`. Committed and pushed.

### 2026-11-10: Task 2, data download and first look
1. Checked disk (812.9 GB free). Committed your pending CLAUDE.md/PROGRESS.md edits.
2. Wrote and tested `src/download_data.py` on a single ZIP code (24060, 1,037 rows, 3 pages): the second run skipped correctly. Committed.
3. Downloaded claims (full file refused with 403, so the API fallback ran): 2,725,989 rows, 216.2 MB.
4. Started VA policies. Paging slowed and timed out, so I stopped it, switched to keyset pagination, re-tested, and resumed from page 45: 1,668,793 rows, 49.4 MB. Re-running `download_data.py all` skips both files. Committed.
5. Wrote `docs/data_dictionary.md` from FEMA's official field descriptions. Committed.
6. Built and ran `notebooks/01_first_look.ipynb` with `jupyter nbconvert --execute`: no errors. Checked every chart visually and fixed overlapping labels, the partial-year 2026 bar, and decimal year ticks. Committed.
7. Wrote `reports/01_data_overview.md`, this log, and the Task 2 learning notes. Committed and pushed.

### 2026-11-10: Project setup
1. `git init -b main`, `.gitignore`, and the folder structure with `.gitkeep` files. Committed.
2. Tried a Python 3.15 venv: failed on `pyyaml`. Installed Python 3.13 and rebuilt the venv. All 14 packages installed.
3. Wrote `requirements.txt`, `requirements-lock.txt`, and `src/check_setup.py`. Ran it: 14/14 OK. Committed.
4. Added README.md and LEARNING_NOTES.md. Committed. Added remote `origin` and pushed to `main` (succeeded using your saved GitHub login).
5. Re-ran `check_setup.py` after you disabled Smart App Control: still 14/14 OK, and SciPy works.
6. Added this PROGRESS.md. Committed and pushed.

Output of `src/check_setup.py`:
```
Python 3.13.17
  OK    pandas         3.1.1
  OK    numpy          2.5.4
  OK    pyarrow        26.0.1
  OK    requests       2.34.2
  OK    matplotlib     3.11.2
  OK    seaborn        0.13.2
  OK    plotly         7.1.0
  OK    scikit-learn   1.9.1
  OK    statsmodels    0.15.0
  OK    xgboost        3.4.1
  OK    shap           0.54.1
  OK    jupyter-core   5.9.1
  OK    ipykernel      7.4.0
  OK    streamlit      1.67.0
All 14 libraries imported successfully.
```
