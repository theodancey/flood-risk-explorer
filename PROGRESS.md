# Progress Log

## Summary (latest: 2026-11-11, Task 4: modelling dataset)
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
6. *(FYI, no action needed)* **"Coastal" uses the Census Bureau's official coastline-county list**, which has 25 Virginia counties and cities. **Chesapeake city isn't on it** (it has tidal rivers but doesn't touch the Bay or ocean directly), and it holds a third of the "inland" exposure. That's partly why coastal vs inland looks flat in the one-way view ($192 vs $203 per policy-year). I kept the official definition because it's citable. In week 3 I can test a broader "tidal" definition if you'd like. ✅ Theo: Yes. In Milestone 6, compare the official coastline-county definition with a broader "tidal" definition (e.g., including Chesapeake and other Hampton Roads localities with tidal flooding), cite the source for the broader list, and keep whichever separates risk better (or explain why neither does).
5. *(Week 8, decided)* **Hosting the app.** The plan puts the Streamlit app on a free host (Streamlit Community Cloud), which needs an account linked to your GitHub. **Decision: you'll create the Streamlit Community Cloud account yourself in week 8.** I won't sign up for anything.

### Earlier (resolved)
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

## Log
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
