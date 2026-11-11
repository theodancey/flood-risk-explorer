# Learning Notes

Plain-English explanations of each step, plus interview prep.

---

## Task 1: Project setup (2026-11-10)

### What I did
I made the project folder a Git repository, which means every change gets saved as a "commit" you can go back to. I created the folders for raw data, cleaned data, notebooks, reusable code, the app, and charts, and I set up a `.gitignore` so data files and the virtual environment never get uploaded. I made a virtual environment (`.venv`): a private copy of Python just for this project. Then I installed the 14 libraries we need into it and wrote `src/check_setup.py`, which proves each one imports correctly.

### Why I did it that way
- **Python 3.13 instead of 3.15.** Python 3.15 came out only weeks ago, and some libraries haven't published prebuilt installers ("wheels") for it yet. `pyyaml`, which Jupyter depends on, would have had to be compiled from C source code, and that needs Microsoft's C++ compiler. Python 3.13 is about a year old, so everything is supported. In data work, a stable, well-supported tool beats the newest one.
- **Virtual environment.** Each project gets its own library versions, so upgrading something for another project can't break this one.
- **Pinned versions** (`pandas==3.1.1`). Pinning means anyone (including future you) can rebuild exactly the same environment. That matters for **reproducibility**: if you rerun a model a year from now, you want the same numbers. `requirements.txt` lists the 14 libraries we chose. `requirements-lock.txt` lists every package, including the ones they depend on.
- **Data stays out of Git.** The FEMA files are large and can be re-downloaded at any time, and Git is built for code, not big datasets. Keeping data out also prevents accidentally publishing anything sensitive later.
- **Separate frequency and severity.** This is a plan for later, not something built yet. The libraries chosen (statsmodels for GLMs, xgboost for boosting, shap for explanations) match the classic actuarial approach: expected loss = how often claims happen × how big they are.

### Interview questions
1. **Why use a virtual environment?**
   It isolates this project's library versions from the rest of the computer, so the project works the same way every time and other projects can't break it.
2. **Why didn't you put the data in your GitHub repo?**
   The NFIP data is large and publicly re-downloadable, and Git isn't built for big data files. I commit the *code* that downloads and cleans the data instead, so anyone can reproduce it.
3. **You had Python 3.15 installed. Why use 3.13?**
   Some dependencies didn't have prebuilt packages for 3.15 yet. A one-year-old Python version has full library support, and reliability matters more than having the newest features for an analysis project.

---

## Task 2: Getting the data and a first look (2026-11-10)

### What I did
I wrote `src/download_data.py`, a reusable script that downloads FEMA's flood insurance data straight from FEMA's official OpenFEMA service. It retries when the network hiccups, skips files it already has, and can resume a half-finished download. I downloaded every NFIP claim in the US (about 2.7 million) and every Virginia policy record (about 1.7 million, keeping only the 37 columns we'll need). Then I wrote a plain-English data dictionary (`docs/data_dictionary.md`) and a notebook (`notebooks/01_first_look.ipynb`) that counts claims by year, state, and county, checks for missing and odd values, and plots the size of claim payments. The main findings are in `reports/01_data_overview.md`.

### Why I did it that way

**What Parquet is.** A spreadsheet or CSV file stores data **row by row**, as plain text. Parquet stores it **column by column**, in a compressed binary format.
- *Intuition:* imagine a filing cabinet with one drawer per column ("all the dates", "all the payment amounts") instead of one folder per claim. If you only need 5 of 84 columns, you open 5 drawers and ignore the rest.
- *Example:* our claims file has 84 columns, but the notebook reads only 24 of them. That's faster and uses much less memory. Parquet also remembers each column's **type** (number, date, text). A ZIP code like `02134` stays text and keeps its leading zero, which a CSV opened in Excel would often lose.
- *Why it's smaller:* values in one column are similar (lots of `VA`, `VA`, `VA`…), so they compress very well.

**What an API and pagination are.**
- An **API** (Application Programming Interface) is a website made for programs instead of people. Instead of clicking around, our script sends a request like "give me NFIP policies where state = VA, only these columns," and FEMA's server sends back the data as text (JSON).
- **Pagination**: FEMA returns at most **10,000 rows per request**, so we ask for the data in pages. Page 1 is `$skip=0, $top=10000`, page 2 is `$skip=10000`, and so on. Virginia's 1.67 million policy rows took about 167 requests. It's like reading a long book one chapter at a time.
- *Being polite:* the script pauses half a second between requests and backs off (waits 2, 4, 8, 16 seconds) when the server has trouble. That's good etiquette for a free public service.
- *Resuming:* each page is saved to its own small file as soon as it arrives. If the internet drops at page 150, re-running the script starts at page 151.
- *Sorting:* we sort by `id` (`$orderby=id`) so the rows come back in the same order every time. Without that, page 2 might overlap with page 1 or skip rows.
- *A lesson learned (keyset pagination):* I first used "skip the first N rows" (`$skip`). It got slower and slower: page 40 took about 60 seconds instead of 5, because the server had to count past 400,000 rows just to find where page 41 starts. I switched to asking for "the next 10,000 rows whose `id` is bigger than the last `id` I received." The server can jump straight there, so every page took about 5 seconds again. This technique is called **keyset pagination**.

**Why the claim-payment chart uses a log scale.**
- *Intuition:* most flood claims are a few thousand to a few tens of thousands of dollars, but some are hundreds of thousands. Data like this is called **right-skewed**: a long tail stretching to the right. On a normal axis, the big claims force the axis out so far that all the ordinary claims get crammed into one thin bar at the left edge.
- *Example:* on a normal axis from $0 to $1,000,000, the gap between $1,000 and $10,000 is invisible (less than 1% of the width). On a **log scale**, $1,000 → $10,000 gets the same width as $10,000 → $100,000, because both are "×10."
- *Formula (only now):* a log scale plots log₁₀(amount) instead of the amount itself. log₁₀(1,000) = 3, log₁₀(10,000) = 4, log₁₀(100,000) = 5, so equal steps mean equal *ratios*.
- *Why actuaries care:* skewed claim sizes are exactly why the plan uses a **Gamma** model for severity later. A Gamma distribution is a probability shape that only allows positive values and has a long right tail, which fits claim sizes much better than the bell curve (normal distribution).

**What "exposure" means (high level).**
- *Intuition:* **exposure** is "how much risk was there to begin with." If County A had 100 claims and County B had 50, County A isn't necessarily riskier: maybe it had 10,000 insured homes and County B had 1,000.
- *Example:* A: 100 claims ÷ 10,000 policy-years = 1 claim per 100 policies per year. B: 50 ÷ 1,000 = 5 per 100. County B is **5× riskier per home**, even though it had fewer claims.
- The usual unit is the **policy-year** (or "earned exposure"): one home insured for one full year. A home insured for 6 months counts as 0.5.
- *Formula:* **claim frequency** = number of claims ÷ exposure. Later, the frequency model (a **Poisson GLM**) will use exposure as an "offset," which means the model predicts claims *per unit of exposure* rather than raw counts. That's why we downloaded the policies data at all: it's the denominator.

**Other decisions:**
- *v3 instead of v2:* FEMA retired the v2 dataset names we planned to use. The v3 datasets are their official replacements.
- *API instead of FEMA's single big file:* FEMA's website refused our script's request for the one-click download file (HTTP 403 means "forbidden"). We didn't try to trick it. The API is also an official FEMA source; it's just slower.
- *Only 37 policy columns:* we kept the columns tied to exposure (dates, coverage amounts, policy counts), location, flood zone, and building features, to keep the file small.

### Interview questions
1. **Why did you download policies as well as claims?**
   Claims alone tell you where losses happened, not how risky an area is. To get a rate, I need exposure (how many policies were in force), which comes from the policy data. Frequency = claims ÷ exposure.
2. **Why Parquet instead of CSV?**
   Parquet stores data by column, compresses well, and keeps data types. It's smaller on disk, much faster to load, and I can read just the columns I need from a multi-million-row file. It also keeps ZIP codes and FIPS codes as text, so leading zeros aren't lost.
3. **Why did you plot claim payments on a log scale?**
   Claim sizes are right-skewed: most are small, a few are huge. A linear axis hides almost all of the data in one bar. A log scale spaces values by ratio, so you can see the whole range. In this data the median paid claim is about $13,000 but the mean is about $42,000, and the top 10% of claims account for half of all dollars paid. The same skewness is why I'll use a Gamma model, not a normal model, for severity.
4. **Did anything go wrong while downloading?**
   Yes, twice. FEMA retired the dataset version I planned to use, so I checked the official catalog and switched to the replacement (v3). Then the paginated download slowed to a crawl because of `$skip` paging. I diagnosed it and switched to keyset pagination, which kept the 44 pages already downloaded. Both files' row counts matched FEMA's totals exactly.
