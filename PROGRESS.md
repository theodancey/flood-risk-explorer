# Progress Log

## Summary (latest: 2026-11-10, Task 2: data download and first look)
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
3. *(FYI, no action needed)* **v2 → v3.** The v2 datasets are gone from OpenFEMA, so I used v3, which is the same NFIP program data under new names, with a few renamed fields. If you saw "v2" in a tutorial or job posting, that's the reason for the difference.
4. *(FYI)* **FEMA's one-file claims download refused our script (HTTP 403 "Forbidden").** fema.gov's web pages block automated requests, and only the API answers. I didn't try to get around the block (e.g. by pretending to be a browser), so the script downloaded the same data through the official API instead. If you ever want the single file, download `https://www.fema.gov/about/reports-and-data/openfema/v3/NfipClaimsV3.parquet` in your browser and save it as `data/raw/NfipClaimsV3.parquet`. The script will then skip it.

### Earlier (resolved)
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

## Log
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
