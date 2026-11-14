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

---

## Task 3: Actuarial basics for Virginia, and the project plan (2026-11-10)

### What I did
I computed the four building blocks of insurance pricing for Virginia: **exposure**, **frequency**, **severity**, and **pure premium**, by year (2010–2025) and by county. I measured exposure properly as **earned policy-years**, splitting each policy across the calendar years it actually covered, instead of just counting policies. I matched claims to exposure by year and county, made charts (measures by year with storms labelled, a top-15 county ranking, and how much of the losses came from the worst few years), and compared the data's shape with the Poisson and Gamma distributions that the models will use. The reusable code is in `src/actuarial.py`, the analysis is in `notebooks/02_actuarial_basics.ipynb`, the findings are in `reports/02_actuarial_basics.md`, and the plan for weeks 2–9 is in `docs/PROJECT_PLAN.md`.

### Why I did it that way

**Earned exposure vs. counting policies.**
- *Intuition:* exposure should measure "how much time was insured," not "how many contracts were signed." A policy that starts on 1 December covers one month of that year and eleven months of the next. Counting it as a whole policy in the first year puts the exposure in the wrong year.
- *Example:* a policy from 2 July 2015 to 2 July 2016 earns about 0.5 policy-years in 2015 and 0.5 in 2016. A policy cancelled after three months earns only 0.25. A policy cancelled before it started earns 0.
- *Formula:* for each policy and year, earned = (days of cover inside that year) ÷ (days in the year). Then add up over all policies.
- *What we found:* from 2010 on, the simple count is only about 2% off, because most policies renew every year (one term ends and the next begins). But in **2009** the true exposure is far lower than the count suggests. The file only includes terms that *started* in 2009 or later, so policies already running when 2009 began are missing, and the raw 2009 exposure is only 46% of the truth. Without fixing this, 2009's claim frequency would look twice as high as it really was. I estimated 2009 by assuming the missing 2008 policies looked like the 2009 ones, marked it clearly as an estimate, and kept the headline numbers to 2010–2025.
- *One policy = one unit:* condo-association policies can cover dozens of units (`policyCount`). Each claim record is one policy's claim, so I count each policy term once on the exposure side too. A ratio only makes sense if the top and bottom count the same thing.

**Why "total paid" (building + contents + ICC).**
- *Intuition:* the premium has to pay for what the insurer actually pays out, so that's what we measure.
- *Paid, not damage:* the adjuster's damage estimate is higher than the payment, because the policyholder pays the **deductible** first and payments are capped by the **coverage limit**. Pricing is based on payments.
- *ICC (Increased Cost of Compliance):* up to $30,000 to raise or flood-proof a badly damaged building so it meets local building rules. It's part of the policy's coverage, so it's a real cost (1.3% of Virginia's dollars).
- *Blanks = $0:* a claim with no payment amount was closed without payment (for example, damage below the deductible). FEMA's "net" payment fields (after money recovered later) give the same total within 0.12%, so the choice barely matters here.
- *A "claim" = a paid claim:* using paid claims for both frequency and severity makes pure premium = frequency × severity exactly.

**Frequency, severity, and pure premium (with Virginia's numbers).**
- Frequency = 9,557 paid claims ÷ 1.48 million policy-years = **6.5 per 1,000 policy-years**.
- Severity = $220 million ÷ 9,557 claims = **$23,039 per paid claim**.
- Pure premium = $220 million ÷ 1.48 million policy-years = **$149 per policy-year** = 0.0065 × $23,039. That's the part of the price that pays expected claims, before expenses, profit, and catastrophe margins are added.

**What "catastrophe-driven" losses mean for pricing.**
- *Intuition:* most of the money is lost in a few rare, extreme years, not spread evenly. Virginia's worst 3 of 17 years produced **69%** of losses.
- *Mean vs. median:* the average loss cost is $149, but the median year's is only $79. In most years the insurer collects more than it pays; in a storm year it pays out many years' worth at once.
- *Pricing consequences:* (1) a price based on recent "typical" years would be far too low, so pricing needs long histories or catastrophe models; (2) the insurer needs **capital and reinsurance** (insurance for insurers) to survive a bad year; (3) when testing models, a test period with or without one big storm can make a model look great or terrible, so we test on several held-out years; (4) prices often include an explicit **catastrophe load**, an extra charge for rare big events.

**Why Poisson and Gamma are the starting points, and what the data says.**
- **Poisson** counts rare, independent events, and its variance equals its mean. It's simple, and with exposure as an offset it models "claims per policy-year." *But:* Virginia's county-year counts vary about 28× more than Poisson allows (**overdispersion**), because storms make claims arrive together. The fix is the **negative binomial**, which adds a parameter for the extra spread.
- **Gamma** describes positive, right-skewed amounts whose spread grows with the mean, like claim dollars. *But:* it under-states the very largest claims. A **lognormal** distribution or capping large claims are common refinements.

**Credibility.** A county with 300 policy-years can have a huge loss cost from one unlucky claim. **Credibility** means how much you trust a number given the amount of data behind it. I only ranked counties with at least 5,000 policy-years. Later the models will blend each area's own experience with the statewide average more formally.

### Interview questions
1. **How did you measure exposure, and why not just count policies?**
   I used earned policy-years: each policy contributes the fraction of each calendar year it was in force, stopping at cancellation. Counting policies by start year puts exposure in the wrong year and ignores cancellations. It was close in normal years (about 2% off), but it would have halved the true exposure for 2009 because of how the data was cut, which would have doubled that year's apparent claim frequency.
2. **What's the pure premium for Virginia, and what does it mean?**
   About $149 per policy-year for 2010–2025: 6.5 paid claims per 1,000 policy-years times a $23,000 average paid claim. It's the expected claim cost per policy, the base before adding expenses, profit, and a catastrophe margin. But the median year is only $79, because a few storm years drive the average.
3. **What makes flood risk hard to price?**
   It's catastrophe-driven: three of seventeen years produced 69% of Virginia's losses, and claim counts are about 28 times more variable than a Poisson model assumes. You need long histories, overdispersed models like the negative binomial, careful testing on held-out years, and capital or reinsurance for the bad years.

---

## Task 4: Building the modelling dataset (2026-11-11)

### What I did
I built the table the models will learn from: `data/processed/va_model_data.parquet`, made by `src/build_dataset.py`. Each row is a **rating cell**: a group of Virginia policies that share the same county, year, flood-zone group, occupancy, floors, elevated / primary-residence / post-FIRM flags, construction-year band, and coverage band. For each cell I added up the exposure (policy-years), paid claims, and paid losses, and converted the losses to **2025 dollars** using the government's inflation index (CPI-U). The script checks itself every time it runs: the totals must match notebook 02 exactly, nothing may be negative or missing, and every claim must land in a cell. The rules are written up in `docs/modeling_dataset.md`, and `notebooks/03_dataset_summary.ipynb` describes the result.

### Why I did it that way

**Rating cells.**
- *Intuition:* an insurer can't know the risk of one particular house, but it can learn the risk of "houses like this one." A rating cell is a bucket of similar policies. We count how much insurance was in the bucket and how many claims came out of it.
- *Example:* "Norfolk, 2016, A/V zone, single-unit home, 1 floor, not elevated, primary residence, pre-FIRM, built 1960–79, $100K–$249K coverage" is one cell. Say it had 40 policy-years and 3 paid claims.
- *Why cells here:* the claims file and the policy file have **no shared ID**, so a claim can't be linked to its exact policy. But both files record the same building facts. So I put policies and claims into cells using the *same* rules on both sides, then compare claims with exposure inside each cell.
- *No information lost:* for a Poisson GLM with categorical features, adding up rows that share features gives exactly the same fitted model as using each policy separately. The model only "sees" the sums.

**Choosing and grouping features.**
- I used only features that are filled in, and coded the same way, in **both** files. Basement type (more than half blank), CRS class (blank in 63% of claims but only 18% of policies, so the cells wouldn't line up), and foundation type (mostly "not reported") were left out.
- FEMA changed its occupancy codes in October 2021 (Risk Rating 2.0). Before then a condo unit or a mobile home was coded "single family," so I grouped them together. The groups then mean the same thing in 2012 and in 2024.

**Sparse cells and the rare-category rule.**
- *Intuition:* with many features, some combinations are rare. A level that holds very little exposure (say, V-zone homes, 0.7% of exposure) gives the model too little data to learn from, so its estimate would be noisy.
- *Rule:* every level of every feature must hold at least **1% of exposure**. Smaller levels are merged into the most similar one: V zones join A zones (both high risk), split-level joins 2 floors, mobile homes join 1 floor.
- *Counties:* 32 counties have at least 5,000 policy-years and get their own level. The rest (about 100 smaller Virginia counties and cities) are pooled into "other coastal" / "other inland," and blank or out-of-state codes go into "other/unknown." This is a simple version of **credibility**: trust an area's own experience only when there's enough of it. The full county code stays in the table, so the pooling can be changed later.
- *Cells are still small* (median 1 policy-year). That's fine: a sparse cell isn't a problem for a Poisson model; a rare *level* is.

**Targets vs. features.**
- A **feature** is something known *before* the loss that the model uses to predict, e.g. flood zone or year built.
- A **target** is what the model predicts: `claims` (frequency model) and `losses_2025` (severity model).
- **Exposure** is neither. It's the amount of insurance at risk, given to the model as an **offset**, so the model predicts claims *per policy-year* and then scales up by each cell's size.

**Trending and inflation adjustment.**
- *Intuition:* a $20,000 repair in 2011 would cost about $29,000 today. If we mixed old and new dollars, the model would think claims are getting worse when partly it's just prices rising.
- *Example:* CPI-U averaged 218.06 in 2010 and 322.15 in 2025, so factor = 322.15 ÷ 218.06 = 1.477, and a $10,000 claim in 2010 becomes $14,774 in 2025 dollars.
- *Formula:* losses_2025 = losses_nominal × CPI(2025) ÷ CPI(year of loss).
- *Real-world wrinkle:* the BLS didn't publish October 2025 CPI (government shutdown), so I filled it with a straight line between September and November. That changes the 2025 average by 0.07%.
- *Actuarial term:* restating past losses to today's cost level is part of **trending**. A full trend study would use construction-cost indices too (repair costs rose faster than CPI in 2021–23). CPI is the standard, transparent first step.
- Coverage bands stay in **nominal** dollars on purpose: the NFIP's $250,000 limit hasn't changed since 1994, so "insured to the limit" means the same thing every year.

**Data leakage.**
- *Intuition:* leakage is like a student who has seen the answer key: amazing on the practice test, useless on the real one. It happens when a feature secretly contains information from the outcome.
- *Example here:* the claims file has water depth, damage amount, and cause of damage. These would "predict" claims perfectly, but they only exist *because* a claim happened, and you'd never know them when pricing a policy. Also `floodZoneCurrent` (today's map) could have been redrawn after the flood, so I used `ratedFloodZone` (the zone the policy was priced on).
- *Rule I followed:* every feature describes the policy or building before the loss. County-level features use only exposure and an external list, never claim counts.

**Why so many cells have zero claims.**
- *Intuition:* flood claims are rare: about 6 per 1,000 policy-years. A cell with 1 policy-year has a tiny expected count, so zero is the normal result.
- *Poisson math:* if a cell's expected number of claims is λ, P(zero claims) = e^(−λ). With λ = 1 × 0.0065, that's about 99.4%. Even 100 policy-years (λ = 0.65) gives about 52% zeros.
- *Result:* 98% of cells have zero paid claims. Poisson with each year's statewide rate predicts 97%, very close.
- *Why the GLM is fine with this:* maximum likelihood counts a zero in a tiny cell as very probable, so it barely moves the estimates, while a zero in a big cell pushes that cell's rate down. **Dropping the zero rows would be a big mistake:** you'd keep only the cells that had claims, and frequency would look far too high.
- *Bonus finding:* big cells have *more* zeros than a flat rate predicts. Risk isn't spread evenly; claims cluster in certain kinds of cells. That's what the features need to explain.

**Unmatched claims.** 95 paid claims (1% of paid claims) have a feature combination that no in-force policy had, so their cell has zero exposure. I kept them, flagged, so totals still reconcile. They can't be used for frequency (no exposure to divide by), but they're fine for severity. I documented this instead of quietly forcing them into the nearest cell.

### Interview questions
1. **How did you combine policy and claim data when they don't share an ID?**
   I built rating cells. I defined each feature the same way on both files and grouped policies by county, year, and eight building and policy features to get exposure. Then I put each claim into the cell matching its own features. The totals reconcile exactly with the raw data. About 1% of paid claims fell into combinations with no exposure, and I kept them, flagged, for the severity model.
2. **Why adjust for inflation, and how?**
   Claims from 2011 and 2024 are in different dollars, so without adjusting, the model would confuse price inflation with rising flood risk. I multiplied each year's losses by CPI-U for 2025 divided by CPI-U for that year, using BLS data. It raised total losses by 33% and severity from about $23K to $31K. I'd refine it with a construction-cost index, since repair costs outpaced CPI recently.
3. **98% of your rows have zero claims. Isn't that a problem?**
   No, that's what rare events look like. A cell with one policy-year expects about 0.006 claims, so zero is almost certain, and a Poisson model predicts nearly the same 97% zero share. The Poisson GLM with a log-exposure offset uses those zeros correctly. Dropping them would bias frequency upward. Sparse cells aren't the issue; rare feature levels are, so I merged any level with under 1% of exposure.

## Task 5: One-way analysis, the coastal question, and validation design (2026-11-11)

### What I did
I looked at every candidate rating factor on its own: for each level (e.g. "A/V zone" vs "B/C/X zone") I computed exposure, claims, frequency, severity, and pure premium, plus **relativities** to a base level, with 95% confidence intervals and credibility flags. I checked whether each effect holds year after year and without the three hurricane years, and measured how much the factors overlap. I built a second, broader coastal flag from Virginia law ("Tidewater Virginia," 46 localities) and showed it separates risk much better than the Census coastline list. Then I chose how the models will be tested from week 3 on. Files: `notebooks/04_one_way_analysis.ipynb`, `src/one_way.py`, `src/validation.py`, `docs/validation_plan.md`, `reports/03_one_way_findings.md`.

### Why I did it that way

**One-way analysis.**
- *Intuition:* before building a model that looks at everything at once, look at each factor alone: "how do old buildings compare with new ones?" It's the first exhibit in every actuarial rate review. It catches data problems, shows which factors are worth modelling, and gives a baseline to compare the model against.
- *Example:* pre-1960 buildings had 10.21 paid claims per 1,000 policy-years; 1980–99 buildings had 3.43.
- *Formula:* for each level, frequency = claims ÷ exposure, severity = losses ÷ claims, pure premium = losses ÷ exposure = frequency × severity.

**Relativities and the base level.**
- *Intuition:* a relativity says "how many times riskier than the reference group." Rating plans are built by multiplying relativities together, so it's the natural language for rating factors.
- *Example:* pre-1960 frequency ÷ 1980–99 frequency = 10.21 ÷ 3.43 = **2.98**: about 3 times as many claims per policy-year.
- *Base level:* by convention, the level with the **most exposure**, because its estimate is the most reliable, and every other level is measured against it. The base always has relativity 1.
- *Charts* use a log scale, so 2× and 0.5× sit the same distance from 1.

**Confidence intervals and over-dispersion.**
- *Intuition:* 3,196 claims is a lot of data, but it's still a sample of the weather. A **95% confidence interval** is a range that, if we could replay these 16 years many times, would contain the true long-run frequency 95% of the time.
- *Poisson interval:* if claims arrived independently (like raindrops), the standard error of a claim count n is about √n. For 3,196 claims that's √3,196 ≈ 57, or 1.8% of the count, so the 95% interval is about ±3.5% (1.96 × 1.8%): very narrow.
- *But flood claims come in clumps:* one hurricane floods thousands of homes at once. So the counts vary much more than Poisson predicts. This is **over-dispersion**, measured by φ ("phi"): φ = 50 means 50× the Poisson variance. I estimated φ from the year-by-level table and widened the intervals by √φ. Example: the B/C/X frequency relativity is 0.41, with a widened 95% CI of 0.30–0.56.
- *Formula (on the log scale):* relativity × exp(±1.96 × √(φ × (1/n_level + 1/n_base))).

**Credibility.**
- *Intuition:* how much should we trust a group's own numbers? A town with 3 claims could look terrible or perfect just by luck.
- *Classic rule ("limited fluctuation" credibility):* **1,082 claims** gives 90% confidence that the observed frequency is within ±5% of the truth. That number comes from (1.645 ÷ 0.05)², using the normal approximation to the Poisson.
- *Partial credibility:* with n claims, Z = √(n ÷ 1,082). A blended estimate would be Z × (own experience) + (1 − Z) × (a broader average). Example: 270 claims → Z = 0.5, so trust it halfway.
- *In this project:* levels under 2% of exposure or under 30 claims are flagged (only "$0 contents-only" coverage). Small counties are pooled before modelling, which is credibility in its simplest form.

**Stability across years.**
- *Intuition:* a real risk factor should show up most years, not just in one storm. If a factor only mattered in 2016, next year's storm may hit somewhere else.
- *What I found:* building age and post-FIRM point the same way in **all 16 years**. Flood zone is real in both kinds of year but 2.6× stronger in hurricanes (B/C/X 0.29× vs 0.75×). "Elevated" and both coastal flags **flip direction** between hurricane years and normal years. Virginia has two flood regimes: coastal storm surge (rare, huge) and inland river and flash flooding (frequent, steadier).

**Confounding.**
- *Intuition:* confounding means a factor looks important (or unimportant) because it travels with another factor that's the real cause. Ice-cream sales and drownings both rise in summer; ice cream isn't the cause.
- *Our example, elevated buildings:* on their own they look slightly *riskier* (6.72 vs 6.25 claims per 1,000). But within the high-risk zone they claim *less* (8.46 vs 9.78), and within the low-risk zone they also claim *less* (3.23 vs 3.91). How can they be lower in every zone but higher overall? Because 67% of elevated buildings are in high-risk zones (floodplain rules require elevation there) vs 40% of non-elevated ones. This reversal is **Simpson's paradox**. Comparing like with like (within zone) gives **0.86**: elevation is linked to about 14% fewer claims.
- *A second example:* post-FIRM buildings have 0.35× the claims, but most of that is building **age**. After also adjusting for year built, it's 0.86.
- *Tool:* a **Mantel–Haenszel** rate ratio: compute the comparison inside each zone, then take a weighted average across zones.

**Why we need GLMs.**
- Comparing within zones works for one confounder. We have 8 building features plus geography, and slicing by all of them at once leaves groups too small to compare.
- A **GLM** (generalized linear model) is regression adapted to counts and dollar amounts. A Poisson GLM with a log link says log(expected frequency) = base + effect of zone + effect of age + ..., so on the original scale the relativities **multiply**, exactly like a rating plan. It estimates every factor's relativity *at the same time*, each one "holding the others fixed." It's the automatic, multi-factor version of what I did by hand for elevated buildings.

**The coastal question.**
- The Census list's "inland" group mixed tidal Hampton Roads cities with *low* frequency (Chesapeake 4.7 per 1,000) with mountain towns with *high* frequency (Roanoke city 14.5), so coastal vs inland looked flat (1.04).
- **Tidewater Virginia**, defined in state law (Code of Virginia § 62.1-44.15:68) and used for Virginia's coastal zone program, draws the line at tidal water. Non-tidal Virginia then shows 1.42× the frequency and 1.81× the pure premium. A small Poisson GLM confirmed it fits better: deviance improvement 39 vs 0.4.
- I also switched small-county pooling to tidal/non-tidal, because the old "other inland" pool mixed two very different kinds of place. (This updates the Task 4 pooling.)

**Holdout and cross-validation.**
- *Intuition:* a model always looks good on the data it learned from, like a student grading their own homework. To know how it will do in real use, test it on data it never saw.
- *Holdout:* keep some data aside, train on the rest, test once. Ours: train on 2010–2020, test on 2021–2025.
- *Cross-validation:* do that several times with different pieces held out, then average. **Leave-one-year-out (LOYO):** 16 rounds, each holding out one year.
- *Why years, not random rows:* a hurricane hits thousands of rows in the same year. Random rows would let the model "see" the storm in training, and the test would be far too easy.
- *Catastrophe years:* the 2021–2025 holdout has **no** hurricane year and only 677 claims (44% from inland non-tidal areas vs 12.5% before), so it's a single, unusual draw of the weather. LOYO tests across 16 different kinds of year, including each hurricane year. To stop 2011 and 2016 dominating the average, every year is scored separately and results are shown with and without the cat years.
- *Decision:* LOYO is the main comparison (development uses only 2010–2020), and the 2021–2025 holdout is a one-time final check.

### Interview questions
1. **Your one-way analysis says elevated buildings have more claims. Does elevation not work?**
   It's confounding. Elevated buildings are concentrated in high-risk flood zones, because floodplain rules require elevation there: 67% of elevated exposure is in A/V zones vs 40% for non-elevated. Within each zone, elevated buildings have fewer claims (8.5 vs 9.8 per 1,000 in A/V). Adjusting for zone gives a relativity of about 0.86. That's exactly why we fit a GLM instead of trusting one-way relativities.
2. **How did you decide which features matter?**
   Three tests. Size and significance: relativities with confidence intervals widened for over-dispersion, plus a deviance-based importance score. Stability: does the effect hold year by year and without the three hurricane years? Overlap: does it survive adjusting for flood zone and building age? Building age and post-FIRM pass all three for frequency. Occupancy, elevation, and second homes lead for severity. The Census coastal flag fails all three.
3. **Why leave-one-year-out instead of a simple train/test split?**
   Three storm years hold two-thirds of our claims, so which years land in the test set can change the answer more than the model does. Our 2021–2025 holdout has no hurricane, few claims, and an unusually inland mix of floods. LOYO gives 16 tests across hurricane, inland-flood, and quiet years, so I can see whether a model wins consistently. I score each year separately so the storm years don't dominate, and I keep the 2021–2025 holdout as a final check.

## Milestone 7: The claim-frequency GLM (2026-11-11)

### What I did
I built the project's first real model: a **Poisson GLM** that predicts how many paid flood claims each rating cell will have, using **log(exposure) as an offset** so it really models the claim *rate* per policy-year. I started from flood zone alone and tested the shortlist features one at a time. A feature stayed only if it was still significant after allowing for storms *and* it helped predict years the model hadn't seen. The final model is **flood zone + year built + floors** (plus calendar year). I compared its relativities with the one-way ones, tested and handled **over-dispersion**, and scored it on held-out years against a "no features" baseline. Files: `src/models/frequency_glm.py`, `notebooks/05_frequency_glm.ipynb`, `reports/04_frequency_glm.md`, `models/frequency_glm.json`.

### Why I did it that way

**Regression.**
- *Intuition:* regression finds the best rule for predicting an outcome from some inputs ("features"). It's "draw the line of best fit," generalized to many inputs.
- *Example:* predicting a house price from its size: price ≈ $50,000 + $150 × square feet. The $150 is a **coefficient** (written β, "beta"): how much the prediction changes per unit of the feature.
- *Categorical features* (like flood zone) get one 0/1 column per level, except a **base level**, which is the reference that the other levels' coefficients are measured against.
- *Formula (ordinary linear regression):* y = β₀ + β₁x₁ + β₂x₂ + … + noise, where the noise is assumed normal (bell-shaped) with the same spread everywhere.

**GLM (generalized linear model).**
- *Intuition:* ordinary regression assumes the outcome can be any number, with a bell-shaped spread. Claim counts are whole numbers ≥ 0, and claim sizes are positive and very skewed. A GLM keeps the "β × features" idea but lets you choose (1) a **distribution** that suits the outcome and (2) a **link function** connecting the features to the mean.
- *Formula:* g(expected y) = β₀ + β₁x₁ + …, where g is the link. Fitted by **maximum likelihood**: pick the β's that make the observed data most probable.

**Poisson distribution.**
- *Intuition:* the classic model for counts of independent events in a fixed amount of time or exposure (calls per hour, typos per page, claims per policy-year). It has one parameter, the mean μ, and its **variance equals its mean**.
- *Example:* if a group of policies expects 4 claims a year, Poisson says 0 claims has probability e^−4 ≈ 2%, exactly 4 about 20%, and the typical spread is ±2 (√4).
- *Formula:* P(N = k) = e^(−μ) × μ^k ÷ k!.

**Log link.**
- *Intuition:* we model log(expected claims) as β₀ + β₁x₁ + …, so on the normal scale the effects **multiply**: expected = e^β₀ × e^β₁ × …. That (1) keeps predictions positive and (2) matches how rating plans work: a base rate times a factor for zone, times a factor for age, and so on.
- *Example:* base rate 4.82 per 1,000 policy-years (A/V zone, built 1980–99, 2 floors, average year); B/C/X zone × 0.46; built before 1960 × 2.73 → 4.82 × 0.46 × 2.73 ≈ 6.0 per 1,000.

**Offset.**
- *Intuition:* a cell with 100 policy-years should expect 100× the claims of a cell with 1 policy-year, all else equal. The offset builds that in: log(exposure) goes into the equation with its coefficient **fixed at 1** (not estimated).
- *Formula:* log(expected claims) = log(exposure) + β₀ + β₁x₁ + …, which is the same as expected claims ÷ exposure = e^(β₀ + β₁x₁ + …). So the β's describe the **rate** (frequency), and exposure just scales it.

**Reading e^β as relativities.**
- *Intuition:* because of the log link, e^β is the factor by which a level multiplies the claim rate compared with the base level, **holding the other features fixed**. That's exactly an actuarial relativity.
- *Example:* β for B/C/X = −0.78 → e^−0.78 = 0.46: moderate/low-risk zones have 54% fewer claims per policy-year than A/V zones with the same building age and floors.
- *Unlike one-way relativities,* GLM relativities are adjusted for the other features. Post-FIRM is 0.35 one-way but about 0.79 once year built is in the model, because "post-FIRM" mostly means "newer building."

**Calendar year as a feature, and average-year prediction.**
- Year is in the model as a factor, so hurricane years (Irene 4.2× the average-year level, Matthew 3.9×) don't get blamed on building features. To predict a *future* year, whose weather nobody knows, the model uses the **average-year level**: the level at which the training years' predicted claims add up to their actual claims.

**Confidence intervals (CIs) for relativities.**
- *Intuition:* the estimate 0.46 comes from one 16-year sample of the weather. A **95% CI** is the range that would contain the true relativity in 95% of replays. If it excludes 1, the effect is "statistically significant."
- *Formula:* on the log scale, β ± 1.96 × SE (SE = **standard error**, the estimate's typical sampling error); then take e^ of both ends. B/C/X: plain Poisson CI 0.44–0.48; storm-aware CI 0.34–0.61.

**Over-dispersion (and the fixes).**
- *Intuition:* Poisson assumes variance = mean, i.e. claims happen independently. Floods break this: one storm floods thousands of homes at once, so counts swing far more than Poisson expects. The coefficients are still fine, but plain-Poisson CIs are **too narrow** and its tests too eager.
- *Measure:* **φ (phi)** = Σ (actual − expected)² ÷ expected, divided by the degrees of freedom. About 1 under Poisson. Ours is about 2 on small cells and about 12 when cells are combined by tidal/non-tidal area. A φ that grows when cells are combined is the signature of cells **moving together** (storms). The **Cameron–Trivedi test** confirmed over-dispersion (p < 0.0001).
- *Fix 1, quasi-Poisson:* same β's; multiply every SE by √φ. Simple, but φ from small cells understates storm clumping.
- *Fix 2, negative binomial (NB):* a count distribution with variance = μ + α·μ² (α ≈ 3.6 here). Better AIC, and slightly better on unseen years. But it changes the β's (big cells get less weight, so zone becomes 0.61 instead of 0.46), it depends on how cells are cut, and its CIs still treat cells as independent.
- *Fix 3, cluster-robust ("sandwich") errors:* same β's as Poisson; the SEs are measured from how much whole **clusters** (county group × year) actually vary. That lets claims from the same storm in the same area move together. CIs come out about 3.6× wider than plain Poisson's.
- *My choice:* Poisson relativities (they **balance**: fitted claims equal actual claims in every level of every feature, and they're standard practice) with cluster-robust CIs. NB stays as the challenger for the model-comparison milestone.

**Deviance and AIC.**
- *Deviance intuition:* how far the model's predictions are from a "perfect" model that fits every cell exactly. It's the GLM version of the sum of squared errors. Lower is better.
- *Poisson deviance formula:* 2 × Σ [ y × log(y ÷ μ) − (y − μ) ], where y = actual and μ = predicted claims.
- *Catch:* adding any feature always lowers training deviance, even pure noise.
- *AIC (Akaike information criterion)* = −2 × log-likelihood + 2 × (number of parameters). It is fit plus a penalty for complexity; lower is better. Adding a feature "pays" if it cuts deviance by more than 2 per parameter. Here AIC would have added 10 of 11 candidates, because its penalty assumes independent claims.

**Likelihood-ratio test (LRT), and its storm-aware versions.**
- *Intuition:* compare a model with and without a feature. If the feature does nothing, the deviance drop behaves like a χ² ("chi-square") random variable with degrees of freedom = parameters added. A drop far bigger than that is "significant."
- *Example:* adding year built cut deviance by 2,397 for 3 parameters. Under "no effect" we'd expect about 3. p < 0.0001.
- *Over-dispersion breaks it:* the χ² rule assumes independent claims. The **quasi-F** test divides the drop by φ. The **cluster-robust Wald test** asks whether the new β's are zero using storm-aware SEs. Building coverage, for example, passes the LRT (p < 0.0001) but not the cluster Wald (p = 0.08).

**Validation by held-out years, and the keep rule.**
- Each candidate was also scored by **leave-one-year-out** on 2010–2020: fit on 10 years, predict the 11th, repeat. A feature stayed only if it was storm-aware significant *and* improved the held-out deviance. County group cut training deviance by 1,170 but made 10 of 11 held-out years worse, because *which* counties flood depends on where the storm goes. This is **overfitting**: learning the noise (past storm tracks) instead of the signal.

**Re-basing, the baseline, and actual vs expected (A/E).**
- *Baseline:* "every policy-year gets the average frequency." Any useful model must beat it.
- *Re-basing:* scale a year's predictions so they add up to that year's actual claims. That removes the "was there a hurricane?" part, which nobody can predict, and scores only *where* the claims landed.
- *A/E* = actual claims ÷ expected claims. 1.0 is perfect. In hurricane years raw A/E was 1.9–5.6, and in quiet years mostly 0.1–0.7. That's correct behaviour for an average-year rate, which is meant to be right *on average* (summed over all 16 years, A/E ≈ 1).

**Lift charts and the Gini index.**
- *Intuition:* does the model put the risky cells at the top? Sort cells from lowest to highest predicted frequency, cut them into 10 groups of equal exposure (**deciles**), and plot the actual frequency in each group. A good model's bars climb steeply; a useless one's are flat.
- *Gini index:* plot cumulative share of exposure (x) vs cumulative share of actual claims (y), sorted from safest to riskiest. Gini = 1 − 2 × (area under that curve). 0 = random ranking; higher = better. Ours is about 0.3 per held-out year.
- *A trap I caught:* when pooling several years into one lift chart, ranking by the re-based prediction sorts storm-year cells to the top and makes the chart look far better than it is. My first draft did this (it showed a 56× top-to-bottom ratio). I now rank by the year-free predicted rate.

### Interview questions
1. **Why a Poisson GLM with an offset, rather than regressing claim frequency directly?**
   Claim counts are small non-negative whole numbers, and 97% of our cells have zero claims, so a normal-errors regression on frequency would be badly wrong: it can predict negative rates, and it gives a cell with 0.1 policy-years the same weight as one with 1,000. Poisson is the natural count distribution. The log link makes the effects multiply like a rating plan. The log(exposure) offset makes the model predict a rate per policy-year while still weighting each cell by its exposure. With categorical features it also balances: fitted claims equal actual claims in every level.
2. **Your one-way analysis had nine useful-looking factors. Why does the final model only have three?**
   Two reasons. First, confounding: post-FIRM, elevation and second-home status mostly reflected building age and flood zone, so once those are in the model, they add little. Second, storm dependence: coverage band and geography fit the training years well, but their effects change from storm to storm, so they made held-out years worse. AIC and plain likelihood-ratio tests would have kept almost everything, because they assume claims are independent. I used cluster-robust tests and leave-one-year-out validation instead. The trade-off: the nine-feature model is slightly better in ordinary years, but much worse in hurricane years.
3. **Your data is over-dispersed. What did you do about it?**
   I tested it (Pearson φ and the Cameron–Trivedi test), then compared quasi-Poisson, negative binomial and cluster-robust standard errors. The key finding was that φ grows when you combine cells, which means the extra variance comes from storms making whole areas move together, not from cell-level noise. Quasi-Poisson and NB both still treat cells as independent, so their intervals are too narrow. I kept the Poisson relativities, which balance to the actual claims, and reported cluster-robust confidence intervals by county group × year, about 3.6× wider than plain Poisson's. NB fits better on AIC and is slightly better out of sample, so it's the challenger for the next milestone.
4. **How good is the model?**
   It beats a no-features baseline in 14 of 16 held-out years: typically a few percent of deviance, more in hurricane years. Its Gini is about 0.3, and the lift chart rises clearly in held-out years and in the 2021–2025 holdout. The weak spot is the very top decile, where the small floors effect decides the order. In raw terms it under-predicts hurricane years by up to 5.6× and over-predicts quiet years. That's by design: it's an average-year rate, and no frequency model can predict when a hurricane will come.


## Milestone 8: The claim-severity GLM (2026-11-11)

### What I did
I built the second half of the loss model: a **Gamma GLM with a log link** that predicts **how big** a paid flood claim is, in 2025 dollars. I fitted it to the 9,553 individual paid claims, each carrying the same features as the rating cells, and checked that they add back up to the cell table exactly. Before modelling, I looked at how skewed claim sizes are, how many claims hit their coverage limit, and whether capping the largest claims helps. It doesn't, so the model uses uncapped amounts. I built the model step by step with the same storm-aware rule as the frequency model, plus a second pass to fix order effects. The final model has **occupancy, year built, building coverage, flood zone, floors, primary residence and elevation**; geography was tested and dropped. Files: `src/models/severity_glm.py`, `notebooks/06_severity_glm.ipynb`, `reports/05_severity_glm.md`, `models/severity_glm.json`.

### Why I did it that way

**Severity vs frequency.**
- *Intuition:* expected loss cost has two parts. **Frequency** is how *often* claims happen (claims per policy-year). **Severity** is how *big* a claim is when it happens (dollars per paid claim). Multiply them and you get the **pure premium**, the expected loss per policy-year.
- *Example:* 6.45 paid claims per 1,000 policy-years × $30,625 per claim ≈ $197 per policy-year (Virginia, 2025 dollars).
- *Why model them separately:* they have different drivers (below) and different statistical shapes. Counts are whole numbers, mostly 0; claim sizes are positive dollar amounts with a long right tail. Two models, each with a suitable distribution, are easier to understand and check than one.
- *Formula:* pure premium = frequency × severity = (claims ÷ exposure) × (losses ÷ claims) = losses ÷ exposure.

**Skewness and heavy tails.**
- *Intuition:* a distribution is **right-skewed** when most values are small but a few are very large, so the long "tail" sits on the right. Then the **mean** (the average) is pulled far above the **median** (the middle value). A **heavy tail** means extreme values are more common than a bell curve would suggest.
- *Example:* our median claim is $14,003, but the mean is $30,345 (2.2×). The largest 1% of claims (96 claims) hold 11% of all dollars, and the largest 5% hold 31%.
- *Why it matters:* the average is what pricing needs, but a few big claims can move it, so models must be robust to them. Ordinary least squares (which squares the errors) gets dragged around by big claims. That's one reason to use a Gamma GLM.
- *Formula (skewness):* the average of ((x − mean) ÷ sd)³. It's 0 for a symmetric distribution; ours is 4.7.

**The Gamma distribution.**
- *Intuition:* a flexible distribution for amounts that are always positive and skewed to the right. Its key property: its **spread grows in proportion to its mean**. A $100K claim can miss by $100K, while a $5K claim misses by about $5K. Errors are about *percentages*, not fixed dollar amounts, which is how claim sizes behave.
- *Example:* if expected claims are $10,000 and the coefficient of variation (sd ÷ mean) is 1.2, a typical claim is $10,000 ± $12,000. That's very uncertain one claim at a time, but the *average* over many claims is well determined.
- *Formula:* mean μ, variance φ × μ² (φ is the **dispersion**; ours is 1.53, so sd ≈ 1.24 × μ). With a log link, log(μ) = β₀ + β₁x₁ + …, so each feature **multiplies** the expected claim by e^β, exactly like the frequency model.
- *Gamma deviance:* 2 × Σ[(y − μ)/μ − log(y/μ)]. It depends only on the **ratio** y/μ, so a claim that is 2× its prediction counts the same whether it's $10K or $200K. That's why one huge claim doesn't dominate the fit.
- *A surprise I hit:* unlike Poisson, a log-link Gamma does **not** make predicted dollars add up exactly to actual dollars. It balances the *average ratio* actual ÷ predicted within each level. So the average-year level is set so total predicted dollars equal total actual dollars.

**Modelling unit: individual claims vs cell averages.**
- *Intuition:* a Gamma GLM on cell averages, weighted by how many claims each cell has, gives exactly the same coefficients as fitting every claim separately (I checked: they differ by 7×10⁻¹²). But only individual claims let you see percentiles, policy-limit hits, residuals and capping effects.
- *Why it was possible here:* every feature is recorded on the claim itself, so no claim-to-policy link is needed.

**Policy limits.**
- *Intuition:* an insurance policy pays at most its **limit**, the amount of coverage bought. NFIP homes can buy up to $250,000 of building and $100,000 of contents cover. If the damage is bigger, the payment stops at the limit. A claim at the limit is **censored**: the damage was *at least* that much, but we can't see how much more.
- *Example:* 178 claims (1.9%) paid 99% or more of a limit, holding 7.2% of dollars. 85 building limits: 16 at the $250K maximum, 69 at a lower limit the owner chose.
- *Why it matters:* (1) limits cap the tail, so NFIP claims can't be unboundedly large; (2) the coverage-band feature is partly **mechanical**: low limits mean smaller claims (under $100K of cover: 0.69×; contents-only: 0.28×).

**Capping and large-loss loads.**
- *Intuition:* to stop a few huge claims from swinging the factors, actuaries often **cap** ("limit") claims at a threshold for modelling, then add the cut-off dollars back as a flat **large-loss load** spread over everyone.
- *Example:* capping at the 99th percentile ($241,051) removes about 3.3% of dollars, so the load is 1 ÷ (1 − 0.033) ≈ 1.034: every prediction is multiplied by 1.034.
- *Formula:* load = total losses ÷ total capped losses; predicted severity = capped-model prediction × load.
- *What I found:* capping didn't help on unseen years, and at the 95th percentile it was **harmful**. The load is flat, but the capped-off dollars come mostly from big buildings (39% of non-residential dollars sit above P95, vs 5% for single-family homes), so capping shifted cost from commercial buildings onto homes. Combined with policy limits that already cap the tail, the decision was **no cap**. The load also swings a lot by year (1.00 to 1.21 at P99), which is why loads are averaged over many years.

**ICC (Increased Cost of Compliance).**
- *Intuition:* a separate NFIP coverage (up to $30,000) that pays to elevate, move or demolish a building after the local government declares it "substantially damaged." It's triggered by a local decision, not directly by the damage, so I left it out of the modelled amount (it's under 1% of dollars) and added it back as a small multiplier (1.0092).

**Residuals.**
- *Intuition:* a **residual** is what the model got wrong for one data point: actual vs predicted. If the model is right, residuals look like random noise. They show no pattern against the predictions or the features. A pattern means the model is missing something.
- *Deviance residual:* each claim's signed share of the deviance. Plot it against the fitted value: a flat average line means no systematic bias for small vs large claims.
- *Quantile residual:* ask "what percentile of its own predicted distribution did this claim land on?" and convert that to a normal z-score. If the distribution is right, these z-scores follow a standard bell curve, which you check with a **Q–Q plot** (sorted residuals against the quantiles a bell curve expects; a straight diagonal line means a good fit).
- *What ours showed:* the middle 95% of claims fit well. The top ~2% are bigger than a Gamma expects. The lognormal had the opposite problem (its tail is too heavy).

**Gamma vs lognormal.**
- *Intuition:* the alternative is to take log(claim size) and fit an ordinary regression. It handles skew too, but it predicts the **median** on the original scale, not the mean. To get the mean you have to multiply by a correction (e^(σ²/2)), and if the spread differs between groups, that correction is wrong.
- *Example:* uncorrected, the lognormal predicted only 54% of actual dollars. With the textbook correction it predicted 112%. The Gamma GLM predicts the mean directly (99%), had a better AIC, and did better on unseen years, so it's the choice.

**Order effects in step-by-step selection.**
- *Intuition:* forward selection tests each feature once, against whatever is in the model at that moment. A feature tested *before* its overlapping partners can look useless and never get another chance.
- *Example:* elevation was tested second, when the model had only occupancy, and it failed. Re-tested at the end (with zone and building age in), it passed clearly. So I added a **second pass** that re-tests every dropped feature with the same rule. I was open that I added it after a dry run showed the problem.

**Why drivers differ between frequency and severity.**
- *Intuition:* frequency asks *"does water get into the building?"* That's about **hazard and vintage**: is the building in the floodplain (flood zone), and was it built before modern floodplain rules (building age)? Severity asks *"once water is in, how much does it damage?"* That's about **what's in harm's way**: how big and valuable the building is (occupancy), how much cover was bought (limit), and whether the living space sits above the water (elevation, floors).
- *Our results:*
  - **Flood zone and building age drive both.** B/C/X zones claim 0.46× as often *and* claims are 0.75× as big, so the loss cost is 0.34×.
  - **Occupancy, coverage, elevation and second homes drive only severity.** Multi-unit claims are 2.1× bigger but not much more frequent.
  - **Floors work in opposite directions:** single-storey homes claim slightly *less* often (0.89×) but *bigger* (1.12×), because all their value is at ground level. The product is about 1.00.
- *Why it matters for pricing:* a factor can look unimportant in one model and matter a lot in the other. Rating only on frequency would miss most of what makes commercial and multi-unit flood losses expensive.

### Interview questions
1. **Why a Gamma GLM for claim severity?**
   Claim sizes are positive and right-skewed, and their spread grows with their size. A $200K claim varies far more in dollars than a $5K one. The Gamma has exactly that property (variance proportional to the mean squared). Its deviance judges errors as ratios, so a few huge claims don't dominate the fit. With a log link, the factors multiply like a rating plan. I compared it with a lognormal: the Gamma had a better AIC and better held-out performance, and it predicts the mean directly, while the lognormal's mean needed a back-transform correction that over-shot by 12%.
2. **Did you cap large losses?**
   I tested caps at the 95th and 99th percentile with a flat large-loss load. Neither helped on unseen years, and the 95th-percentile cap distorted the factors: the load spreads capped dollars evenly, but large claims come mostly from multi-unit and commercial buildings, so it underpriced them and overpriced single-family homes. NFIP policy limits already cap the tail, and the Gamma is fairly robust to large claims, so I modelled uncapped amounts and kept the 99th-percentile cap (load 1.03) as a documented sensitivity check.
3. **What drives frequency vs severity in your models?**
   Flood zone and building age drive both: they decide whether water gets in, and somewhat how deep. Occupancy, coverage limit, elevation and second-home status drive only severity: they decide how much is damaged once water gets in. A multi-unit building's claims are about 2.1× a home's, and an elevated building's about 0.76×. Floors are the interesting case: single-storey buildings claim a bit less often but bigger, and the two effects cancel in the loss cost.
4. **How do you know the severity model works?**
   On leave-one-year-out validation it beats a no-features baseline in 14 of 16 years, with a mean improvement of about 16% in Gamma deviance, and 16.7% on the untouched 2021–2025 holdout, where A/E is 1.02. The top decile's actual average claim is 4.7× the bottom decile's. Out-of-fold A/E is close to 1 for almost every feature level. The weaknesses: the bottom deciles are flat (and under-predicted), county-level differences remain (Newport News 1.67, Norfolk 0.79), and storm years still swing the overall level, from 0.49 to 1.77 of the average.

## Milestone 9: Pure premium, county credibility, Tweedie and loss ratios (2026-11-11)

### What I did
I combined the frequency GLM and the severity GLM into one **pure premium**: the expected loss per policy-year for every rating cell. I applied the (unit) large-loss load, the ICC bridge, and an **off-balance factor** so the model's total equals actual losses. Then I built the **credibility-weighted county factor** you asked for in Question 7. I compared two classic methods (limited fluctuation and Bühlmann–Straub) and tested them on years the model hadn't seen. I also fitted a one-step **Tweedie GLM** as a challenger, compared it with frequency × severity using lift, double-lift and Gini, and computed actual NFIP **loss ratios** from the premium fields. Five **rating worksheets** show the multiplication step by step. Files: `src/models/pure_premium.py`, `notebooks/07_pure_premium.ipynb`, `reports/06_pure_premium.md`, `models/pure_premium.json`.

### Why I did it that way

**Multiplying relativities.**
- *Intuition:* a **relativity** says how much riskier one level of a factor is than the **base level** (the reference group, relativity 1.00). Because both GLMs use a log link, effects **multiply**: start from the base building's expected loss and multiply by one relativity per factor.
- *Example:* the base building costs about $138 per policy-year. A building like it but built 1960–79 has a frequency relativity of 2.75 and a severity relativity of 1.28, so its pure-premium relativity is 2.75 × 1.28 = 3.52, and its expected loss is about $138 × 3.52 ≈ $485.
- *Formula:* pure premium = base frequency × base severity × Π (frequency relativity) × Π (severity relativity) × loads. Π means "multiply them all together". A factor that's only in one model has relativity 1 in the other.
- *Why it matters:* the combined table shows *why* a segment is expensive. Floors are the neat example: one-floor buildings claim less often (0.89×) but bigger (1.12×), so their pure premium barely changes (≈1.00).

**Large-loss loads.**
- *Intuition:* if you cap claims at a threshold when fitting the model, the dollars above the cap must be put back somewhere. The **large-loss load** spreads them evenly over everyone as a flat multiplier.
- *Example:* capping at the 99th percentile removes 3.3% of dollars, so load = 1 ÷ (1 − 0.033) ≈ 1.034.
- *Here:* Milestone 8 chose **no cap**, so the load is 1.000. The code still multiplies by it (it's stored in the severity model), and the notebook shows the P99 version as a sensitivity: it barely changes the relativities.

**The off-balance correction.**
- *Intuition:* after you multiply everything together, the predicted total for the whole state should equal what actually happened. If it doesn't, multiply everything by one number that fixes it. That number is the **off-balance factor**.
- *Why it's needed:* each model is balanced on its own, but (1) the average of a product isn't the product of the averages (cells with high frequency can have lower severity), (2) the ICC bridge is an average, and (3) about 1.2% of losses are in cells with no exposure, which a frequency model can't predict.
- *Example:* predicted total before correction ≈ $294.7M, actual $292.7M, so off-balance = 292.7 ÷ 294.7 ≈ 0.993.
- *Formula:* off-balance = Σ actual losses ÷ Σ predicted losses (before the correction). The same idea is used again to **re-centre** the county factors so they move money between counties without changing the state total.

**Credibility and the Z formula.**
- *Intuition:* how much should you trust a county's own experience versus the statewide model? A big county with thousands of claims mostly speaks for itself. A small county with five claims is mostly luck. **Credibility** Z (between 0 and 1) is the weight on the county's own experience.
- *Formula:* blended factor = Z × (county's actual ÷ expected) + (1 − Z) × 1.
- **Limited fluctuation ("classical"):** pick a tolerance. "I want the county's A/E to be within ±5% of the truth 90% of the time." The number of claims that guarantees this is n_full = (1.645 ÷ 0.05)² ≈ 1,082 (for severity, multiply by CV², the squared coefficient of variation of a claim, ≈ 2,713). Below that, Z = √(n ÷ n_full). *Example:* 270 claims → Z = √(270 ÷ 1,082) = 0.50.
- **Bühlmann–Straub ("greatest accuracy"):** measures noise and signal from the data. **EPV** (expected process variance) is how much a county bounces around year to year. **VHM** (variance of hypothetical means) is how much counties truly differ. k = EPV ÷ VHM, and Z = n ÷ (n + k). *Example:* k = 33 claims, so a county with 300 claims has Z = 300 ÷ 333 = 0.90.
- *What I found:* for **frequency**, Bühlmann–Straub found the counties' differences are no bigger than storm noise (VHM < 0, so Z = 0). For **severity**, k is only about 33 claims, so big counties get high credibility. Limited fluctuation can't see storm clumping (it treats 2,000 claims from one hurricane as 2,000 independent pieces of evidence), so it gives frequency more weight than the data supports.

**Tweedie distributions.**
- *Intuition:* a loss cost per cell is usually exactly $0 (no claims), and otherwise a skewed positive amount. A **Tweedie** distribution with power p between 1 and 2 has exactly that shape: it's a *compound Poisson–Gamma*, meaning a Poisson number of claims, each with a Gamma-distributed size. So one GLM can model losses ÷ exposure directly, instead of two models.
- *Example:* a cell with 10 policy-years and a 0.5% claim rate has about a 95% chance of $0, and otherwise one or two claims of tens of thousands of dollars.
- *Formula:* variance = φ × mean^p. p = 1 behaves like Poisson, p = 2 like Gamma. The theory link with claim size is p = (α + 2) ÷ (α + 1), where α is the Gamma shape. Our α ≈ 0.96 gives p ≈ 1.51. I also tried a grid of p from 1.2 to 1.8 on the development years; 1.7 scored best, so I used it.
- *Trade-off:* one model is simpler, but it can't tell you whether a segment is costly because claims are frequent or because they're big. It also forces each factor to have one combined effect.

**Lift and double-lift charts.**
- *Intuition:* a **lift chart** sorts the cells by predicted cost, cuts them into 10 equal-exposure groups (deciles), and shows actual cost per group. A good model's actual bars rise steeply from left to right.
- A **double-lift chart** compares two models head to head. Sort the cells by the **ratio** model A ÷ model B. Decile 1 holds the cells where A is much lower than B; decile 10, where A is much higher. In each decile, plot actual and both predictions. Whichever model tracks the actual line is right where they disagree, and that's the part that matters for pricing.
- *Example (ours):* in decile 10, frequency × severity predicts 2.06× average, the Tweedie 1.19×, and actual is 2.06×. In decile 1 the Tweedie predicts 2.09× and actual is 0.72×. Frequency × severity was right at both ends.

**The Gini coefficient.**
- *Intuition:* how well does a model put the risky groups at the top? Sort cells from lowest to highest predicted cost and plot cumulative share of exposure (x) against cumulative share of actual losses (y). A random ranking gives the diagonal. A good ranking hugs the bottom (the cheap cells hold almost no losses) and shoots up at the end.
- *Formula:* Gini = 1 − 2 × (area under that curve). 0 = no better than random; closer to 1 = losses concentrated in what the model calls riskiest.
- *Example:* ours is about 0.46–0.49 on unseen years: the riskiest-looking 10% of exposure holds far more than 10% of losses.
- *Caveat:* Gini only measures **ranking**, not the level. A model that is wrong by 2× everywhere has the same Gini. That's why it's paired with A/E and deviance.

**Loss ratios.**
- *Intuition:* a **loss ratio** = losses ÷ premiums: how many cents of each premium dollar went to paying claims. **Earned premium** is the part of premium that belongs to a year, spread day by day like earned exposure.
- *Example:* Virginia NFIP 2010–2025 earned about $1.14 billion of premium and paid about $220 million of losses (nominal), a loss ratio of 19%. In 2016 (Hurricane Matthew) it was 107%.
- *Why it's not "overcharged":* NFIP premiums must also pay for expenses (agents, insurers that service policies), a catastrophe and reserve loading, debt, and **national** catastrophes like Katrina, Harvey and Ian. Virginia's quiet years help pay for Louisiana's bad ones. Risk Rating 2.0 (October 2021) also re-priced every policy, with yearly increase limits ("glide path"), so premiums are still moving.
- *How I used it:* the useful comparison is **relative**. Divide each segment's modelled-loss-to-premium ratio by the statewide one, which removes the common loadings. Segments above 1 pay less than their modelled share of risk; below 1, more.

**Why Bühlmann–Straub was chosen and the app uses frequency × severity.**
- *County method:* both credibility methods cut county-level error on unseen years by about a third (mean |log A/E| 0.50 → 0.34). Only Bühlmann–Straub *also* improved the overall fit: LOYO 10.2% → 10.7%, better in 13 of 16 years, and holdout 6.4% → 8.9%. On the development years alone both slightly hurt, which I report honestly.
- *Structure:* the Tweedie (p = 1.7) scored about 1 point better on deviance and Gini. But the double lift showed frequency × severity was right where the two disagreed. The Tweedie predicted about 2× too much for commercial buildings and low-coverage policies, because at p = 1.7 the fitting gives high-cost cells little weight (each cell is weighted by mean^(1−p)).

### Interview questions

**Q: What is an off-balance factor and why did you need one?**
A: It's a single multiplier that makes the model's predicted total equal the actual total. Each GLM balances on its own, but multiplying them cell by cell doesn't preserve totals, because the average of a product isn't the product of the averages. Mine was 0.993: a 1.9% mix effect, partly offset by 1.2% of losses in cells with no exposure.

**Q: How did you handle geography when county effects didn't validate in your GLMs?**
A: With credibility. Each county's actual ÷ expected ratio is blended with 1 using a weight Z. Bühlmann–Straub estimates Z from the data: county frequency differences were no bigger than storm noise, so Z = 0, while severity differences became credible after about 33 claims (k = 33.4). Out of sample, it cut county-level error by a third, and I still label it as less certain in the app.

**Q: Why not just use a Tweedie GLM for pure premium? It's one model.**
A: I tried it, and it scored slightly better on overall deviance. But a double-lift chart showed that where the two models disagreed, frequency × severity matched actual losses and the Tweedie didn't. The Tweedie also produced a backwards coverage relativity (lower limits → higher losses). Two models also explain *why* an area is expensive: more claims, or bigger ones.

**Q: Virginia's NFIP loss ratio is 19%. Is flood insurance overpriced there?**
A: Not necessarily. NFIP premiums also pay for expenses, catastrophe and reserve loadings, debt, and national cross-subsidies, and one bad year (107% in 2016) can wipe out many quiet ones. The useful part is relative. After dividing out the statewide ratio, pre-1980 buildings pay less than their modelled share and newer buildings more. That fits Risk Rating 2.0's glide path still phasing in full-risk rates.

## Milestone 10: XGBoost challenger models (2026-11-11)

### What I did
I built three **gradient-boosted tree models (XGBoost)** to challenge the GLMs: claim frequency (Poisson), claim severity (Gamma) and pure premium (Tweedie). They use the same data, all the candidate features and the same validation design. I tuned their settings with a randomized search scored only on the 11 development years (2010–2020), one held-out year at a time, with early stopping. Then I applied rules written in advance to decide whether geography helps and whether to force sensible directions (monotonic constraints). Finally, I checked them once on the 2021–2025 holdout and against the GLMs. XGBoost was better on unseen years for all three, but its Tweedie failed the same double-lift check as the GLM Tweedie, so frequency × severity stays the pricing structure. Files: `src/models/xgb_models.py`, `notebooks/08_xgboost.ipynb`, `reports/07_xgboost.md`, `models/xgb_settings.json`.

### Why I did it that way

**Decision trees.**
- *Intuition:* a decision tree is a flowchart of yes/no questions. "Is the building in an A/V zone? If yes, was it built before 1980?" Each final box (a **leaf**) holds a prediction: the average outcome of the training rows that ended up there.
- *Example:* a depth-2 tree splits on zone first, then on year built inside each zone. That gives 4 leaves, e.g. A/V & old → 9 claims per 1,000 policy-years, B/C/X & new → 1.5.
- *How it chooses a question:* it tries every feature and every cut point, and picks the split that improves the fit the most (for us, the drop in Poisson, Gamma or Tweedie deviance). **Depth** = how many questions in a row. Deeper trees can describe very specific groups, but each leaf then holds fewer rows, so its average is noisier.

**Gradient boosting.**
- *Intuition:* one tree is crude. **Boosting** builds trees one after another, and each new tree tries to fix the mistakes left by all the earlier ones. It's like a team of weak forecasters where each one only studies what the previous ones got wrong.
- *Example:* tree 1 learns "A/V zones claim more". Tree 2 looks at what's still off and learns "and old buildings in A/V zones even more". Tree 3 nudges multi-unit buildings, and so on for hundreds of trees.
- *Formula (log scale):* prediction = exp(base_margin + η × tree₁ + η × tree₂ + … + η × tree_M). η is the **learning rate**. "Gradient" means each tree is fitted to the slope of the loss: the direction that most reduces the deviance.
- *Link to GLMs:* a GLM adds one fixed number per factor level on the log scale. Boosting adds many small tree outputs on the same log scale. With depth-1 trees (one question each) it's very close to a GLM. Deeper trees allow **interactions** (an effect that depends on another feature).

**XGBoost objectives.**
- *Intuition:* the **objective** tells XGBoost what kind of outcome it's predicting, so it uses the right deviance. It's the same choice as a GLM's distribution.
- `count:poisson` for claim counts (frequency), like the Poisson GLM. `reg:gamma` for positive, right-skewed amounts (severity), like the Gamma GLM. `reg:tweedie` (with `tweedie_variance_power` = 1.7) for loss cost that's often exactly $0, like the Tweedie GLM.
- All three use a **log link**: trees are added on the log scale, so effects multiply, just as in the GLMs.

**base_margin = the offset.**
- *Intuition:* in a GLM, the **offset** is a known piece of the log prediction that isn't estimated, e.g. log(exposure): twice the policy-years → twice the expected claims. XGBoost's version is `base_margin`: a starting value per row that the trees add to.
- *Example:* a cell with 200 policy-years in 2018 gets base_margin = log(200) + log(2018's year level). If the trees add log(0.004), the prediction is 200 × 2018 level × 0.004 claims.
- *What I put in it:* log(earned exposure) for frequency (so trees learn a *rate*), plus the **log of the calendar year's level** from a year-only model on the training years (so a hurricane year isn't blamed on whichever building types happened to be flooded). An unseen year gets the **average-year level**, the same idea as the GLMs' average-year prediction. Year is never a tree feature, because a tree can't learn anything useful about a year it has never seen.

**Hyperparameters.**
- *Intuition:* **parameters** are learned from the data (the leaf values). **Hyperparameters** are settings we choose that control *how* the model learns.
- `max_depth` (how specific the groups can get); `learning_rate` (how much of each tree to add: small is cautious and needs more trees); `n_estimators` (number of trees); `min_child_weight` (a leaf needs at least this much evidence; for Poisson that's roughly "expected claims in the leaf"); `subsample` and `colsample_bytree` (each tree sees a random share of rows and features, which adds healthy randomness); `reg_lambda`, `reg_alpha` (shrink leaf values toward zero); `gamma` (a split must improve the fit by at least this much).
- *Analogy:* they're like a GLM analyst deciding which features and how many levels to allow, but turned into numbers you can search over.

**Early stopping and over-fitting.**
- *Intuition:* **over-fitting** means memorizing the training data's noise instead of the real pattern. Here, that means learning *which cells happened to be flooded in 2016* instead of *which kinds of buildings flood more*. More trees always fit the training years better, but past some point they do worse on unseen years.
- *Early stopping:* keep adding trees, score the held-out years after every few, and stop when the held-out score hasn't improved for a while (150 trees here). Use the tree count at the best score.
- *Example:* with the chosen frequency settings, the held-out score rises for the first few dozen trees, peaks, then slowly falls as later trees chase storm noise (see `reports/figures/08_early_stopping.png`).

**Tuning leakage.**
- *Intuition:* **leakage** means information from the test data sneaks into the choices you made, so the test score looks better than the model really is. If you try 40 settings and pick the one that does best on 2021–2025, then 2021–2025 is no longer an honest test. You've tuned to it.
- *What I did:* every tuning choice (settings, tree count, geography, constraints) used leave-one-year-out on **2010–2020 only**. The 2021–2025 holdout was scored once, after everything was frozen. A milder form remains: picking the best of 40 settings on the same 11 folds is slightly optimistic (**selection optimism**), which is why the holdout matters.

**Monotonic constraints.**
- *Intuition:* a **monotonic constraint** forces the prediction to move in only one direction as a feature increases, holding everything else fixed. Without it, trees can produce patterns no one can defend, such as a higher coverage limit giving a *smaller* expected claim in some small corner of the data.
- *Example:* "elevated buildings never claim *more often* than the same building not elevated." XGBoost enforces this inside every split, so it holds for every combination of the other features.
- *How:* in XGBoost, `monotone_constraints` = −1 (non-increasing), +1 (non-decreasing) or 0 (free) per feature. This needs ordered codes (year built 0 = before 1960 … 3 = 2000+), which is why bands were coded as numbers rather than one-hot.
- *Trade-off:* constraints can cost a little accuracy if the true pattern isn't monotone, so only directions with a clear physical or actuarial reason were constrained. Year built stays free for severity and pure premium because the GLMs show a real 1960–79 peak.

**Feature importance (gain).**
- *Intuition:* **gain** adds up how much each feature's splits improved the training fit, across all trees. A feature that gets used for many big improvements has high gain.
- *Caveats:* it measures *training* fit, not value on unseen years. It favours features with many possible split points or many one-hot columns. And it says nothing about direction (does the feature raise or lower risk?). A feature can have high gain and still fail the held-out test, which is exactly what county group did for frequency.
- *Better tool (Milestone 12):* **SHAP values** split each individual prediction into per-feature contributions with a sign, and add up exactly to the prediction.

**What the comparison found, and why it isn't the final word.**
- *Results on unseen years:*
  - **Frequency:** XGBoost 6.75% vs GLM 5.58% (holdout 4.3% vs 2.2%).
  - **Severity:** 25.4% vs 19.2% (holdout 23.7% vs 16.7%).
  - **Pure premium (Tweedie):** 17.4% vs 13.5%.
- *Why the trees win:* they find interactions (e.g. old buildings are much worse in A/V zones than in B/C/X), and they use county for claim size, which the GLM rejected.
- *Why it isn't decisive yet:*
  - Frequency XGBoost is worse in the three hurricane years.
  - The severity comparison isn't like for like (the GLM pricing structure adds county severity through credibility).
  - A model with 940 trees is much harder to explain and defend than a relativity table.
- Milestone 11 does the full comparison; Milestone 12 explains the trees with SHAP.

**Geography (your Question 7).**
- *Frequency:* even a flexible model couldn't use county or Tidewater to predict *how often* claims happen in an unseen year. County group was worse in 10 of 11 years.
- *Severity:* county and Tidewater help claim *size*, by +2 points, confirmed on the holdout.
- This matches the Bühlmann–Straub result from Milestone 9: county frequency differences are storm noise, but county severity differences are real.

### Interview questions

**Q: How did you tune XGBoost without overfitting to your test data?**
A: Every choice (hyperparameters, number of trees, geography, constraints) was made with leave-one-year-out cross-validation on 2010–2020 only. Each setting was scored by its average improvement over 11 unseen years, and early stopping watched that same average. The 2021–2025 holdout was scored once, after everything was frozen. It confirmed the gains, which suggests the selection optimism was small.

**Q: How do you handle exposure in XGBoost?**
A: The same way a GLM does, with an offset. XGBoost calls it `base_margin`: a starting value on the log scale that the trees add to. I put log(earned exposure) there, so the trees model claims per policy-year, plus the log of the calendar year's level from the training years, so a hurricane year isn't blamed on building features. Prediction uses an average-year level.

**Q: Your XGBoost Tweedie had the best deviance. Why didn't you pick it?**
A: Because a double-lift chart showed that where it and frequency × severity disagreed, frequency × severity matched actual losses: end-decile error 0.24 vs 0.78. The Tweedie also under-predicted multi-unit buildings by about a third and over-predicted low-coverage policies. That's the same pattern I'd seen in the GLM Tweedie, so it's the structure, not the algorithm. One overall metric can hide segment errors that matter for pricing.

**Q: What are monotonic constraints, and did they cost accuracy?**
A: They force the model's prediction to move only one way as a feature increases, for example that elevation never increases claim frequency. For the Tweedie they cost nothing, so I used them. For severity they cost about half a point, more than my pre-set allowance. For frequency they helped on average, but only because of Hurricane Matthew; they were worse in 9 of 11 years. So my rule left it unconstrained, and I flagged the slightly odd elevation effect for the comparison milestone.


## Milestone 11: GLM vs XGBoost head to head, and the production model (2026-11-11)

### What I did
I compared the GLM champion (frequency × severity + county credibility factor) with the XGBoost models from Milestone 10. Every model was refitted on exactly the same training years and scored on exactly the same unseen rows, with the same loss treatment and the same yardstick. XGBoost turned out to be genuinely more accurate: about 4 points better on pure premium, in 15 of 16 unseen years, and two kinds of bootstrap put the gap clearly above zero. I then checked things accuracy doesn't show: stability, hurricane years, and whether each model's effects point in sensible directions. Finally, I took the interactions XGBoost found and added them to the GLM, creating a "hybrid" that became the production model the app will load.

### Why I did it that way

**Fair comparisons.**
- *Intuition:* comparing two models is like comparing two students. It's only fair if they sit the same exam, under the same rules, marked with the same answer key. If one model saw an easier test year, or had its losses capped differently, the "winner" could just be the luckier setup.
- *Example:* in Milestone 10, the GLM pure premium was scored *without* its county factor, while XGBoost had county inside its severity trees. That's not like for like. This time, both frequency × severity structures were built by literally the same code, and I added a variant (XGBoost without geography + the same credibility county factor) so the county handling matches too.
- *Checklist I used:* same data file, same rows, same folds, same test rows (asserted in code), same dollars and loads, same metric, and the same tuning budget (both chosen on 2010–2020 only). Any differences are written down, not hidden.

**The Gini coefficient.**
- *Intuition:* it measures how well a model **ranks** risks, ignoring whether its overall level is right. Sort all cells from lowest to highest predicted risk, then ask: do the actual losses pile up at the high end?
- *Example:* if the safest 50% of exposure (by the model's ranking) produced only 10% of losses, the model ranks well. If it produced 50%, it ranks no better than random.
- *Formula:* plot the cumulative share of exposure (x) against the cumulative share of actual losses (y), sorted by predicted risk. A random model gives the diagonal. Gini = 1 − 2 × (area under that curve), so 0 = random and higher = better ranking.
- *Our numbers:* pure premium 0.47 (GLM champion) vs 0.53 (XGBoost frequency × severity).

**Double-lift charts.**
- *Intuition:* when two models mostly agree, comparing them on average hides where they differ. A double lift zooms in on the **disagreements** and asks who's right there.
- *How:* for each cell, compute the ratio model A ÷ model B. Sort cells by that ratio and cut them into 10 equal-exposure groups. In the left group, A is low relative to B; in the right group, A is high relative to B. Plot actual losses and both models' predictions in each group. Whichever line tracks the actual bars at the ends is right where they disagree.
- *Our result:* in the group where the GLM was relatively highest, it predicted 2.18× average, XGBoost 0.93×, and actual was 0.83×. End-decile error: GLM 1.12, XGBoost 0.08. That's why I went looking for what XGBoost knew that the GLM didn't.

**Bootstrap confidence intervals.**
- *Intuition:* we only have one history (16 years). The **bootstrap** asks: "if history had come out slightly differently, how much would my answer change?" It builds many alternative datasets by **resampling with replacement** from what we have, and recomputes the answer on each.
- *Example (year bootstrap):* XGBoost beat the GLM by different amounts each year (+9.5 in 2011, −0.5 in 2013, and so on). Draw 16 years at random *with replacement* (some years twice, some not at all), average the differences, and repeat 10,000 times. The middle 95% of those 10,000 averages is the **95% confidence interval**: 2.9 to 5.7 points. It doesn't include 0, so the gap isn't luck.
- *Cluster bootstrap:* within each year, resample whole **county groups** (not single cells), because cells in the same storm and place move together. Resampling single cells would pretend we have far more independent evidence than we do (the same over-dispersion lesson as Milestone 7).
- *Sign test:* the simplest check. If two models were equal, winning 15 of 16 years would be like getting 15 heads in 16 coin flips, which has probability about 0.0005.

**Model stability.**
- *Intuition:* a rate that jumps every time you add a year of data is hard to trust and hard to file with a regulator. A good model's predictions for the same kind of building should move a little, not a lot, when the training data changes slightly.
- *How I measured it:* fix a grid (all 2025 rating cells). Predict it with the all-years model and with each model that left out one year. Divide each set of predictions by its own average, so a change in overall *level* doesn't count. The measure is the average % change in a cell's *relative* prediction.
- *Results:*
  - For frequency and severity, the GLMs move about 2–3% and XGBoost about 4%.
  - XGBoost also moves 2–4% when **only the random seed** changes. Its row and column sampling makes each run slightly different.
  - Surprise: the least stable piece of my own model is the **county credibility factor**. Dropping 2011 moves the champion's pure premium by 42% on average, vs 17.5% without the county factor.

**Why insurers value explainability.**
- *Regulators:* in the US, rates are filed with state regulators, who expect a rating plan: a table of factors, each with a sensible direction and supporting data. "The model says so" isn't an acceptable reason.
- *Customers:* a policyholder can ask why their price is what it is. With a GLM the answer is a worksheet: base × 3.0 for a 1960s building × 0.89 for one floor × 1.21 for Virginia Beach.
- *Checking for mistakes:* a GLM's few numbers can be checked against common sense. I flipped one feature at a time on every cell. XGBoost made elevation *raise* expected claim frequency for 78% of exposure, and made higher coverage *lower* expected claim size for 22%. Neither can be defended. The GLMs can't do that, because each factor has one multiplier.
- *Trade-off:* explainability can cost accuracy, which is why the honest answer here is "XGBoost is more accurate, and here's why we still don't use it yet."

**Hybrid GLM + interactions.**
- *Intuition:* an **interaction** means one factor's effect depends on another. A plain GLM multiplies effects, assuming each works the same everywhere. Trees find interactions automatically, so you can use XGBoost as a *detective* and the GLM as the *courtroom*: add only the interactions XGBoost points to, and keep them only if they pass the same statistical tests as any other term.
- *Example:* low coverage limits mean smaller claims (0.69×), and multi-unit buildings mean bigger ones (2.10×). The plain GLM multiplies these: 1.45×. But a multi-unit building with under $100K of cover hits its limit, so its claims can't be that big. The hybrid's two-way table says 1.11×.
- *Formula:* log(severity) = base + coverage effect + occupancy effect + **(coverage × occupancy) effect** + … The extra term is one coefficient per combination of non-base levels (6 here), so the result is still a small table of multipliers.
- *Result:* coverage × occupancy was overwhelmingly significant (better in all 11 development years). The hybrid closed about a third of the XGBoost gap on the 2021–2025 holdout. But its 16-year gain is small (+0.4 points) and not consistent across years (9 of 16), which I disclosed.

**Why the GLM still won the decision.** I wrote the decision rule in PROGRESS.md *before* computing anything, so I couldn't bend it to a result I liked.
- XGBoost passed the accuracy, holdout, catastrophe-year and double-lift tests.
- It failed the segment test: it over-predicted buildings from 2000 or later by about 30%, and mis-sized Alexandria and Chesapeake, where the GLM was balanced.
- It also can't produce a rating worksheet yet.

So the app uses the hybrid GLM, and XGBoost stays as the benchmark that tells us how much signal is still on the table.

### Interview questions

**Q: XGBoost was more accurate. Why didn't you use it?**
A: The accuracy gap was real (about 4 points, 15 of 16 years, bootstrap CI 2.9 to 5.7), but accuracy wasn't my only pre-stated test. XGBoost over-predicted newer buildings by about 30% and mis-sized two county groups. It moved the wrong way on elevation for 78% of cells, and its rates changed when only the random seed changed. For a rating plan that has to be explained to customers and regulators, that's disqualifying for now. Instead, I moved its best interactions into the GLM.

**Q: How did you decide whether the difference between the models was real?**
A: Three ways. Per-year wins with a sign test: 15 of 16 years, p ≈ 0.0005. A year bootstrap, resampling the 16 per-year differences, which asks whether the gap survives a different draw of weather. And a cluster bootstrap, resampling county groups within each year, which asks whether it survives within-year noise. XGBoost's gap was clear of zero in all three. My hybrid GLM's gain was clear within years but not across years, so I described it as small.

**Q: What's a double-lift chart, and what did it show?**
A: Sort cells by the ratio of two models' predictions and compare actual losses with each model in the extreme groups, where they disagree most. Where my GLM thought cells were much riskier than XGBoost did, actual losses were close to XGBoost's prediction (0.83× average vs the GLM's 2.18×). That told me XGBoost knew something the GLM didn't. I traced it to interactions like coverage × occupancy and added those to the GLM.

## Milestone 12: Explainability: global and local explanations, SHAP, and the app data (2026-11-12)

### What I did
I explained what drives the production GLM and the XGBoost challenger, both for Virginia as a whole and for individual building profiles and counties. For the GLM I showed its multiplier (relativity) table and ranked factors by how much they move predictions across all insured buildings. Year built and flood zone dominate. For XGBoost I computed SHAP values and drew beeswarm and dependence plots, then compared the two models on the same yardstick: they agree on the main story but disagree on a few telling details. Finally I built reusable code (`src/explain.py`) that turns any profile or county into a step-by-step "waterfall" from the Virginia average plus a careful plain-English sentence, and precomputed everything the app needs into small files in `app/data/`.

### Why I did it that way

**Global vs local explanations.**
- *Intuition:* a **global** explanation answers "what matters to this model overall?" A **local** explanation answers "why did *this* case get *this* number?" Think of a teacher's grading rubric (global: essays are 40% of the grade) vs one student's report card (local: you got a B because your essay was weak).
- *Example:* globally, year built is the biggest driver across Virginia. Locally, the Norfolk elevated second home is cheap mainly because it was built after 2000 (×0.26 vs the Virginia average), even though it sits in a high-risk zone (×1.37).
- The app needs both: global for "how the model works", local for "why this county".

**SHAP and the game-theory idea behind it (Shapley values).**
- *Intuition:* imagine three friends share a taxi and the fare depends on who rides. How do you split the cost fairly? Lloyd Shapley's answer (Nobel Prize, 2012): list every order in which the friends could have joined, record how much each person *added* to the fare at the moment they joined, and average that over all orders. A model's features are the "players" and the prediction is the "fare".
- *Toy example:* a model predicts 2 × 3 = 6 (on the log scale, log 2 + log 3). Add feature A first: it moves the prediction from 1 to 2 (×2), then B moves it from 2 to 6 (×3). Add B first: B gives ×3 and A gives ×2. Averaged over both orders, A gets ×2 and B gets ×3, and together they explain the whole move from 1 to 6. With interactions the order matters, and averaging over orders shares the interaction fairly. That's why it's fair.
- *Formula (for reference):* φ_j = Σ over sets S without j of [|S|! (n − |S| − 1)! / n!] × [v(S ∪ {j}) − v(S)], where v(S) is the prediction when only the features in S are "known". It's just the weighted average of j's added value over all orders.
- **SHAP** (SHapley Additive exPlanations) applies this to each prediction. The values add up exactly: average prediction + sum of SHAP values = this prediction. For trees there's a fast exact algorithm (**TreeSHAP**), built into XGBoost.
- *How I used it:* on the log scale, so each SHAP value is a multiplier (exp(φ)). Pure premium SHAP = frequency SHAP + severity SHAP, because logs add.
- *Lighter approach:* I computed SHAP for every row (fast). Plots use 5,000 rows sampled in proportion to exposure, and interaction values use a 3,000-row sample.

**Beeswarm, dependence and waterfall plots.**
- **Beeswarm (SHAP summary plot):** one row per feature, one dot per building group. A dot's left-right position is that feature's SHAP value for that group (left = lowers the prediction, right = raises it), and its colour is the feature's value. Example: in the frequency beeswarm, "year built" splits into a right-hand cluster (old buildings, light colour) and a left-hand one (newer, dark). Wide rows matter a lot; narrow rows barely matter.
- **Dependence plot:** zooms in on one feature. x = the feature's level, y = its SHAP value, colour = a second feature. If the colours separate vertically, the two features **interact**. Example: in severity, the coverage effect differs by occupancy (low-limit apartment buildings get pushed down hardest). That's the same coverage × occupancy interaction the hybrid GLM uses.
- **Waterfall:** a local explanation drawn as a staircase. Start at the average (Virginia, $197), and each bar is one factor moving the number up (orange) or down (blue) until you reach the result. Example: Virginia Beach 1960s house: $197 × 1.75 (1960–79) × 1.50 (A/V) × 1.21 (county) × … = $598.
- *My waterfalls:* each step is a multiplicative Shapley value. "Setting" a factor means giving every insured Virginia building the profile's value and re-averaging, averaged over all 256 orders. So the steps multiply **exactly** to the result, and no order was chosen by me.

**Why GLMs are explainable by design.**
- A log-link GLM predicts by multiplying: base × one multiplier per factor level. Each multiplier is the same for every building, so the whole model fits in a table a regulator or customer can read (29 numbers plus three small two-way tables).
- *Bonus fact:* for a GLM, SHAP values have a closed form: each factor's term minus the average term. So the GLM's SHAP values and its relativity table tell the same story, just measured from a different starting point (the average vs the base building). That let me compare GLM and XGBoost explanations **like for like**.
- *Importance vs multiplier size:* non-residential has the biggest multiplier (4.6×) but occupancy ranks only third in importance, because 89% of insured buildings are single-unit homes. Importance = big effect × how many buildings it touches.

**Correlation vs causation in model explanations.**
- *Intuition:* a model learns **patterns in data**, not cause and effect. A SHAP value says "the model's prediction is higher for groups with this feature", not "this feature causes floods". Features often stand in (are **proxies**) for things the data doesn't record.
- *Example 1, coverage:* the GLM says claims are smaller for buildings with under $100K of cover (×0.79). It would be silly to tell someone "buy less insurance and the flood will do less damage". Low cover goes with smaller, cheaper buildings and with units inside bigger insured buildings. Only one piece is truly causal: the **limit caps what the insurer pays** (5.7% of under-$100K claims hit the limit, vs 0.8% for $250K+).
- *Example 2, elevation:* XGBoost's frequency model *raises* predicted claims for 76% of elevated buildings. Elevation can't cause claims. Elevated buildings are concentrated in coastal high-risk areas (67% are in A/V zones), and the trees picked up that location pattern. This is **Simpson's paradox** again: within each zone, elevated buildings have much lower losses (×0.40 in A/V).
- *Example 3, correlated features share credit:* XGBoost gives "post-FIRM" a lot of weight and the GLM gives it none, but XGBoost's year built + post-FIRM ≈ the GLM's year built. They carry the same information. SHAP splits credit among correlated features however the model happened to split it, so a feature's SHAP value isn't its "true" effect.
- *What I did about it:* the app's sentences are descriptive and area-level ("mainly because 46% of insured buildings here are multi-unit or non-residential"). They never tell an owner what to change.

**County explanations: mix vs credibility factor.**
- A county's expected loss differs from Virginia's for two reasons, and they multiply exactly:
  1. **Property mix:** what its insured buildings are like (zones, ages, building types).
  2. **County factor:** its own credibility-weighted claims history.
- *Example:* Norfolk is right at the Virginia average, but for opposite reasons that cancel. Its mix is riskier (68% of buildings built before 1980), and its claims have been smaller than expected (factor 0.83). Newport News is 1.6× almost entirely because of its factor (1.69).
- **Certainty label:** from the credibility Z (High ≥ 0.8, Medium 0.5–0.8, Low below that or no claims). It tells the app user how much of the county factor rests on local evidence. Only 11 of 133 localities are High, but they hold 71% of insured policy-years.

### Interview questions

**Q: What is a SHAP value, in plain English?**
A: It's a fair way to split one prediction into contributions from each feature, borrowed from game theory. You consider every order in which the features could be added to the model, measure how much each one changes the prediction when it's added, and average. The contributions add up exactly to the difference between this prediction and the average. I used them on the log scale, so each one reads as a multiplier, like "this building's age multiplies expected claims by 1.5".

**Q: Your GLM is already interpretable. Why compute SHAP values for it?**
A: To compare it fairly with XGBoost. A GLM's SHAP values have a closed form (each factor's term minus its average), so I could measure both models with the same yardstick, mean absolute SHAP, on the same rows. They agreed on the top three drivers and on the weight of geography. They disagreed where features overlap (post-FIRM vs year built) and on elevation, where XGBoost's frequency model goes the wrong way for 76% of elevated buildings.

**Q: Can you use a model explanation to tell someone how to reduce their flood risk?**
A: Not directly. Explanations describe the model, not causes. For example, my model gives lower claim sizes to buildings with low coverage, but that's because low coverage stands in for cheaper buildings; only the policy-limit cap is causal. Elevation is a good counter-example: the evidence within flood zones does support it, but XGBoost still got it backwards because elevated buildings cluster in risky places. So my app sticks to area-level, descriptive statements and says clearly that it isn't advice or a prediction for any property.

## Milestone 13: Three states (Virginia + North Carolina + Maryland), out-of-state validation, state factors and credibility (2026-11-13)

### What I did
I added North Carolina and Maryland to the project. I downloaded their policies, built one rating-cell table for all three states, and used each state's own law to define which counties are "tidal". First, I tested the Virginia model on the other two states without changing it. It ranked risk almost as well as it does in Virginia, but it underestimated North Carolina's overall level 6.6-fold.

Then I refit the same model on all three states, adding a **state factor**, and validated it three ways: on unseen years, on a five-year holdout, and on a whole state left out. I checked whether each relativity points the same way in every state. I also tested state-specific effects and the county credibility factor. Both made predictions on unseen years worse, so neither went into production. XGBoost, refit with its frozen settings, again failed the pre-stated decision rule, so the GLM stays.

### Why I did it that way

**Out-of-sample vs out-of-state validation.**
- *Intuition:* a model is only useful if it works on data it hasn't seen. Testing it on the data it learned from is like marking your own homework with the answer key open.
- **Out-of-sample** means testing on rows the model wasn't fitted on. I had used two kinds of these since Milestone 7:
  - **leave-one-year-out (LOYO):** fit on 15 years, predict the 16th, and repeat for every year;
  - **holdout:** fit on 2010–2020, predict 2021–2025 once.
- **Out-of-state** is a stricter version: fit on two states and predict the third, a state the model has never seen. It answers a different question: does what the model learned about buildings (zones, ages, elevation) apply in a new place?
- *Example:* the Virginia model, unchanged, ranked North Carolina's buildings well. Its Gini was 0.49 vs 0.53 at home. (Gini measures how well a model sorts risky from safe: 0 is random, 1 is perfect.) But it got the **level** badly wrong (actual ÷ expected, A/E = 6.55). So *ranking* transferred and *level* didn't. That's exactly why insurers price each state with its own base rate.
- *Lesson:* every validation answers a specific question. LOYO asks "will next year look like this?" Leave-one-state-out asks "will a new place look like this?" You need the one that matches how the model will be used.

**State factors.**
- *Intuition:* some differences between states have nothing to do with the buildings. North Carolina sticks out into the Atlantic and gets hit by hurricanes far more often. A state factor is one multiplier per state that carries this "where you are" level, so the building relativities only have to explain differences between buildings.
- *Example:* pure premium = Virginia base × building relativities × state factor. NC's factor is **6.22** and MD's is **1.01**. An identical building costs about 6.2 times as much to insure in NC as in VA, on these data.
- *Why state × year, not one shared year factor:* storms hit different states in different years (Florence in 2018 hit NC, not VA). A single year factor would average NC's terrible 2018 with Virginia's quiet 2018 and leave the rest for the building relativities to absorb. So each state gets its own year effects, and the state factor is the gap between its average-year level and Virginia's.
- *Off-balance per state:* after multiplying everything out, I rescale each state slightly (NC by 1.026, MD by 1.060) so the model's total expected losses equal that state's actual total. Then no state subsidizes another.

**Why relativities that agree across datasets signal a real effect.**
- *Intuition:* if a coin lands heads 6 times out of 10, that could be luck. If three different coins each land heads about 60% of the time, it's much harder to blame luck.
- In the same way, a relativity fitted on one state could be noise, or a quirk of that state's storms. If Virginia, North Carolina and Maryland, with different storms, coastlines and building stock, each independently show the same direction, the effect is very unlikely to be a fluke. That's **replication**, one of the strongest forms of evidence in statistics.
- *What I found:*
  - **Consistent:** flood zone (frequency) and number of floors (severity) point the same way, with sizes not significantly different.
  - **Differs in size:** year built (frequency) and elevation, occupancy and zone (severity) point the same way, but the size varies by state.
  - **Inconsistent:** a few smaller terms, e.g. floors for frequency, flip direction between states. Those are the ones to trust least.
- *How I tested it:* a **Wald test** checks whether a set of coefficients is the same across groups. I used a cluster-robust version, which treats claims from the same storm and area as related rather than independent, so one storm doesn't count as thousands of separate pieces of evidence.
- *Agreeing in direction but not in size is fine:* the model uses one pooled relativity, which is a weighted average. When I tried giving each state its own relativities, predictions on unseen years got **worse in every state**. The extra precision was mostly fitting noise.

**How more data changes credibility.**
- *Intuition (recap from Milestone 9):* credibility is how much to trust a small group's own experience vs the bigger average. **Z** is between 0 and 1: estimate = Z × own experience + (1 − Z) × average.
- *Bühlmann–Straub* sets Z = n / (n + k), where n is the group's exposure (or claims) and k = EPV / VHM:
  - **EPV** (expected process variance): how much a county's results bounce around from luck alone;
  - **VHM** (variance of hypothetical means): how much counties *really* differ.

  A small k means real differences are large compared with luck, so you trust local experience sooner.
- *What happened:*
  - In Virginia alone, the frequency VHM came out at or below zero (k = ∞, so Z = 0): differences between counties' claim rates looked like pure storm luck.
  - With three states, VHM turned positive. Pooled k for frequency was 160; per state, NC 230 and MD 84. Severity k barely moved (VA 33, pooled 41).
  - More counties gave the method enough spread to detect real differences.
- *The twist:* "detectable" isn't the same as "useful for prediction". In North Carolina, a county's history is dominated by one or two hurricanes' footprints, and the next hurricane goes somewhere else. The county factor made unseen years worse (NC's LOYO score halved, from 12.5% to 6.2%), so the pre-stated rule removed it.
- *Lesson:* more data makes credibility estimates more stable, but Bühlmann–Straub assumes each county's true level is stable over time. Catastrophe-driven flood history breaks that assumption. Always test the credibility-weighted result on unseen years.

**A practical lesson: memory.**
- The first 3-state refit was stopped by the computer at about 10 GB of memory. The data was only 0.17 GB.
- The cause was that statsmodels' result objects point at each other (a "reference cycle"). Python frees those only when its cycle collector runs, so hundreds of old fits piled up.
- Forcing a collection before each fit (`gc.collect()`) kept memory at 0.5–1.3 GB with identical results.
- *Lesson:* measure before optimizing. The suggested fixes (float32, categorical types, chunking) would have saved almost nothing, because the data was never the problem.

### Interview questions

**Q: Your Virginia model was off by 6.6× in North Carolina. Does that mean it was a bad model?**
A: No. It ranked North Carolina's risks almost as well as Virginia's (Gini 0.49 vs 0.53), so what it learned about buildings carried over. What didn't carry over was the overall level: North Carolina gets far more hurricane damage. That's a state-level effect, so I added a state factor (6.2× for NC) and kept the building relativities shared. Out-of-state testing is useful precisely because it separates those two things.

**Q: How did you decide whether to give each state its own relativities?**
A: I fitted each state separately and compared the relativities with confidence intervals and cluster-robust Wald tests. The main ones (flood zone, year built, elevation, occupancy) point the same way in all three states, which is strong evidence they're real. Some differed in size, so I tested a model with state-specific effects on every left-out year and the 2021–25 holdout. It was worse in every state, so I kept one shared set of relativities plus a state factor.

**Q: Why did you remove the county credibility factor when you had more data?**
A: With three states, Bühlmann–Straub found real-looking differences between counties, where Virginia alone had found none for frequency. But when I tested it on unseen years, it made predictions worse, especially in North Carolina, where a county's history is mostly the footprint of one or two hurricanes. I'd written the rule down before testing, so I followed it and dropped the factor. The app still shows each county's own history and a certainty label, so the information isn't hidden; it just doesn't change the price.


## Milestone 14: The Streamlit app: from Python scripts to a web app (2026-11-13)

### What I did
I built a three-page web app with Streamlit. **County Explorer** shows a county's expected annual flood loss per policy-year next to its state's average, with a plain-English summary, a certainty label, the county's mix of insured buildings, its own claims history (major storms labelled) and a chart of why it differs from the state. **Property Profile Calculator** lets you describe a type of building and shows its expected claim frequency, claim size and loss cost, with a waterfall chart of each characteristic's effect. **How It Works** explains the data, methods, findings and limitations. The app reads only small precomputed files (about 1.3 MB) and every page carries a disclaimer. Automated tests click through all 257 counties and a dozen building profiles, and check that the app never touches the raw data.

### Why I did it that way

**What Streamlit is.**
- *Intuition:* building a website usually means learning HTML (page structure), CSS (styling) and JavaScript (interactivity), plus a web server. Streamlit lets a data person skip all that: you write an ordinary Python script, and Streamlit turns it into a web page.
- *Example:* the line `st.metric("Expected annual loss", "$453")` puts a big number on the page; `st.selectbox("County", counties)` puts a searchable drop-down list there and gives the script back whatever the visitor picked.
- It's free and open source, and Streamlit Community Cloud can host apps for free (planned for Milestone 16).

**How a script becomes a web app: the rerun model.**
- *Intuition:* every time the visitor changes anything (picks a county, clicks a button), Streamlit runs the **whole script again from top to bottom**, with the widgets now returning the new choices, and redraws the page. There are no "event handlers" to wire up: the script simply describes what the page should look like for the current choices.
- *Example:* the County Explorer script reads the chosen county near the top, then looks up its numbers and draws its charts. Pick another county and the script reruns, so everything below updates automatically.
- **Widgets** (drop-downs, radio buttons) remember their value between reruns via a `key`, e.g. `key="ce_county"`. Each visitor gets their own **session**, so two people using the app at once don't see each other's choices.
- **Multi-page:** `app/app.py` lists the three page scripts with `st.navigation`, which builds the menu and runs whichever page is selected.

**Caching: why the app is fast even though it reruns constantly.**
- *Intuition:* if every click re-read every file and recomputed everything, the app would be slow. Caching means "remember the answer the first time, and reuse it whenever the same question comes back".
- **`st.cache_data`** is for data: file loads (the county table, the history table) and computed results (one county's explanation, one building profile's waterfall). The function runs once per distinct input; afterwards Streamlit hands back a copy of the saved result. *Example:* the first time anyone rates "1960s house in an A/V zone, Virginia Beach", the exact Shapley waterfall takes about 1.5 seconds; after that it's instant.
- **`st.cache_resource`** is for one shared object that shouldn't be copied, like a loaded model. The rating tables + insured portfolio ("explainer") is built once per server and shared by all visitors.
- The difference: `cache_data` returns a fresh copy each time (safe if the page changes it); `cache_resource` returns the same object (fast, but it must never be modified).

**Why the app uses precomputed data, not the raw data or the full model.**
- The raw FEMA files are about 370 MB and can't go in Git (a project rule); a free host won't have them. Building the modelling table takes minutes, far too slow for a web page.
- So the heavy work was done once, offline, in earlier milestones, and the results were saved as small summary files in `app/data/`: one row per county, one row per county-year, the insured mix, and the model as **lookup tables** (base rate and multipliers).
- *Example:* rating a building only needs base rate × the multiplier for each characteristic × the state factor, which is a few lookups, not a model fit.
- This also makes the app **lightweight to deploy**: it needs only five libraries (streamlit, pandas, numpy, pyarrow, plotly), not statsmodels, XGBoost or SHAP. To get there I changed one line of design in `src/explain.py`: the full model code (which needs statsmodels) is now imported only when rebuilding the app data, never when the app runs. A test checks that none of the heavy libraries get loaded.
- Privacy and honesty bonus: every file is an aggregate, with no policy-level or claim-level rows.

**Responsible design: disclaimers and not over-claiming.**
- *Intuition:* a number on a polished web page looks authoritative. Someone could read "$4,783" as "my building's flood insurance price", or think the app is an official FEMA tool. Both would be wrong, so the app has to say so clearly, where people will see it.
- Every page starts with a short disclaimer: these are **model-based, area-level averages** from public FEMA NFIP data; **not a prediction for any specific property**, **not an insurance quote** and **not financial advice**; and **not affiliated with or endorsed by FEMA**. No FEMA logo or branding is used.
- *Why "area-level"?* The data is aggregated and anonymized, and the model predicts the **average** insured building of a type in an area in an **average** year. A real building's risk depends on things the data can't see (its exact ground height, its distance to a creek).
- *Why "not a quote"?* The app shows the **pure premium**, the expected claim cost only. A real price adds expenses, a catastrophe load and the NFIP's own rules.
- **Certainty labels** are part of the same idea: where a county's own history disagrees with the model, or there's little history, the app says so instead of showing one confident number.
- Input checks are honest too: an unusual combination (built before 1960 but "post-FIRM") gets a warning, a rare building type gets a note that there's little local data, and an impossible one (a business as a primary residence) is prevented.

**Design choices.**
- **Colour-blind-friendly colours:** orange = raises the expected loss, blue = lowers it, grey = totals and the state comparison. Blue/orange is a standard pair that people with the common forms of colour blindness can tell apart, and the bars also have labels (×1.63), so meaning never depends on colour alone.
- **Waterfall charts** start at the state average and show each factor's effect as a step, so a visitor can see *why* a number is high or low.
- **Mobile-friendly:** a centred layout, and columns that stack on narrow screens.

**Testing a web app automatically.**
- Streamlit's `AppTest` runs the app's scripts without a browser, lets a test "click" widgets (pick a state, pick a county) and read what's on the page.
- The tests visit **all 257 counties** and check the big number matches the data file; rate 12 profiles across the three states and check the numbers equal `src/explain.py`'s; try the edge cases; and **log every file the app opens** to prove it never reads `data/raw` or `data/processed`.

### Interview questions

**Q: How does your app compute a prediction so fast without loading the model?**
A: The production model is a GLM, so a prediction is a base rate times one multiplier per characteristic, times a state factor. I exported those multipliers as small lookup tables, so rating a building is a handful of lookups. The heavier part, the exact Shapley waterfall that explains a prediction, takes about a second, and Streamlit's cache stores each result so a repeat request is instant. The app doesn't need statsmodels or XGBoost at all, which keeps it lightweight to host.

**Q: How did you make sure people don't misuse the numbers?**
A: Three ways. Every page has a visible disclaimer: these are area-level model averages from public NFIP data, not a prediction for any property, not a quote and not financial advice, and not affiliated with FEMA. The app calls the number an "expected annual loss per policy-year" and explains that a real premium adds expenses and catastrophe loads. And every county shows a certainty label and its own claims history, so where the model and the history disagree, the user can see it.

**Q: What's the difference between st.cache_data and st.cache_resource, and where did you use each?**
A: Both remember a function's result so it isn't recomputed on every rerun. cache_data stores data and hands back a copy each time, so I used it for loading the CSV and JSON files and for each county or profile result. cache_resource stores one shared object without copying, so I used it for the explainer object (the rating tables plus the insured portfolio), which is built once and shared by every visitor.


## Milestone 15: Address lookup, county map, "what if?" and plain language (2026-11-13)

### What I did
I added four things to the live app.
1. **County map:** a clickable map of every county in the three states, shaded by expected yearly flood loss (or by how each county compares with its state).
2. **Address lookup:** you type a street address, and the app finds its county, flood zone and whether the building is older or newer than the area's first flood map, using free government map services.
3. **"What if?":** shows how the model's estimate changes if a building is raised off the ground.
4. **Plain-language pass:** I rewrote the main pages for a homeowner with no insurance background, moved technical terms into "Learn more" sections and the How It Works page, and fixed several confusing spots and two bugs.

Every new feature has automated tests, and the address tests use fake web answers so they never need the internet.

### Why I did it that way

**Geocoding: turning an address into a place on the map.**
- *Intuition:* a computer can't look up a flood map with "1000 Bank St". It needs a point: latitude and longitude. **Geocoding** is the translation from address to point. The geocoder also says which state and county that point is in.
- *Example:* the U.S. Census Geocoder turns "1000 Bank St, Richmond, VA 23219" into about (37.5385° N, 77.4353° W), in Richmond city (county code 51760), Virginia. It's free and needs no account. The Census Bureau keeps it because it already maintains every US street's address ranges for the census.
- How it works inside: the Census knows that, say, 1000-1098 is on the left side of one block of Bank St, so it places number 1000 at the start of that block. That's why a match is approximate (within a lot or two), which is fine for a county and close to fine for a flood zone.

**Map services and point queries: "what flood zone is this point in?"**
- *Intuition:* FEMA's flood maps are stored as **shapes** (polygons) on a server: each shape is one flood zone area. A **point query** asks the server: "which shapes contain this point?", and the server answers with that shape's details, without sending the whole map.
- *Example:* asking FEMA's National Flood Hazard Layer about a point on the Virginia Beach oceanfront returns `FLD_ZONE = "VE"`: a high-risk coastal zone with waves. The same question for the Virginia State Capitol returns `"X"`, "area of minimal flood hazard".
- This service is an **ArcGIS REST service**: a standard way to publish map layers on the web. Each layer has a number (layer 28 is "Flood Hazard Zones", layer 22 is "Political Jurisdictions", i.e. which town or county the point is in). A query is just a web address with the point and the fields wanted, and the answer comes back as JSON (structured text).
- The app turns FEMA's zone codes into the model's two groups: zones starting with A or V are high risk ("Special Flood Hazard Areas", at least a 1% chance of flooding each year); X, B and C are lower risk. Zone D ("possible but not studied") and unmapped areas aren't guessed: the user chooses.

**How pre-FIRM and post-FIRM is decided.**
- *Intuition:* a **FIRM** (Flood Insurance Rate Map) is the official flood map for a community. Once a community gets its first FIRM, new buildings must follow flood-safe rules (for example, the lowest floor above the expected flood height). So a building is **post-FIRM** if it was built after the community's first map, and **pre-FIRM** if before. Pre-FIRM buildings usually flood more.
- **FEMA's rule:** post-FIRM means construction started after December 31, 1974, **or** on or after the date of the community's first FIRM, **whichever is later**.
- *Example:* Richmond's first FIRM is June 15, 1979, which is later than the end of 1974, so the cutoff is June 15, 1979. A house built in 1985 is post-FIRM; one built in 1970 is pre-FIRM. One built in 1979 could be either, and the app only has the year, so it says "not sure" and asks.
- *Example 2:* Virginia Beach's first FIRM is 1970, earlier than 1975, so the cutoff is January 1, 1975. A 1974 house is pre-FIRM and a 1975 house is post-FIRM.
- The FIRM dates come from FEMA's **Community Status Book** (the official list of communities in the flood insurance program). I saved a small copy in the app (1,084 communities, 76 KB), because the dates almost never change and a saved copy can't be "down".

**Choropleth maps.**
- *Intuition:* a **choropleth** is a map where each area is filled with a colour that shows a number, like an election map shaded by vote share. Here each county is shaded by its expected yearly flood loss per insured building.
- **Colour choices:**
  - For "how much", I used **one colour from light to dark** (light orange = low, dark = high). One hue reads naturally as "more", and it works for colour-blind readers.
  - For "compared with the state", I used **two colours with grey in the middle** (blue = below average, grey = about the same, orange = above). That's called a diverging scale, and it matches the app's other charts (orange raises, blue lowers).
- **Log scale:** North Carolina counties run up to about 10 times Virginia's. On a normal scale every Virginia county would be the same pale colour. On a **log scale**, each step in colour means "times 2" instead of "+$200", so you can still see differences in Virginia while NC stays darker. *Example:* $100 → $200 is the same colour step as $1,000 → $2,000.
- The county outlines come from the Census Bureau's simplified "cartographic boundary" files. I cut them to the three states and rounded the coordinates to about 100 m, which is invisible at map scale. The result is 0.34 MB, small enough to ship with the app so it never downloads a map at runtime.

**Caching, again: the same idea with new rules for live services.**
- *Intuition:* asking a government server takes about a second. If a visitor clicks around after a lookup, the app reruns, and it shouldn't ask again each time. So results are remembered (cached).
- **Failures aren't cached.** If FEMA's server is down and we cached "down", the app would keep saying so for a day after it came back. The cached function raises an error on failure (so nothing is stored), and the page catches it and shows a friendly message.
- **Privacy and caching:** addresses are personal. Geocoding answers are kept only in memory, for at most 10 minutes. Flood-zone answers are stored by map point, not by address.

**Graceful fallbacks.**
- *Intuition:* a "graceful" app degrades step by step instead of crashing. Every outside service can be slow, down, or unable to find the address, so each step has a plan B.
- **Short timeouts** (6 seconds) and **two retries** for temporary problems. No retry for "bad request" errors, which won't fix themselves.
- If the geocoder fails, a plain message appears and you choose the inputs yourself. If only the flood map fails, the state and county are still filled in. If the community can't be matched, pre/post-FIRM says "not sure". Nothing ever shows a technical error, and no message repeats the address.
- The tests fake each of these situations (timeouts, server errors, no match, an address in DC) to prove the page still works.

**Why the "what if?" number is an average effect, not a guarantee.**
- *Intuition:* the model learned from thousands of insured buildings that raised buildings, *on average*, have smaller flood claims than similar buildings that aren't raised. The "what if?" number just reruns the model with "raised" switched on.
- *Example:* a 1960s single-family home in a high-risk zone in Virginia Beach: about $454 a year → about $321 if raised (about −29%).
- **What it isn't:**
  - An engineering estimate. It knows nothing about this building's foundation, ground height or how high it would be raised.
  - A promise of savings. It's an expected claim cost, not a premium.
- **Correlation, not pure cause:** raised buildings also differ in other ways, for example many are newer or on the coast. In the data this showed up as **Simpson's paradox**: within high-risk zones, raised buildings had much smaller claims (×0.40), but across all zones the gap looked smaller (×0.58), because raised buildings are concentrated in the riskiest places. The model's figure is its best average, so the app labels it that way.
- Only "raised" is offered. Zone, age and use describe the building and can't be changed, and coverage changes how much a policy pays, not how likely a flood is.

**Plain language without losing the detail.**
- The main pages are for homeowners; the How It Works page is for reviewers and interviewers. So "pure premium" became "expected yearly loss", "frequency" became "chance of a claim each year", and "Shapley values" became "a fair-share split". Each technical term still appears, defined, on How It Works or in a "Learn more" section.
- Certainty became two signals that answer real questions: "How much local data is there?" and "Do past claims agree with the model?". The old single "Low" label was red for 199 of 257 counties, which made everything look broken.

### Interview questions

**Q: How does your app find the flood zone for an address?**
A: Two free government services. First, the U.S. Census Geocoder turns the address into latitude and longitude, plus the state and county. Then I send that point to FEMA's National Flood Hazard Layer, an ArcGIS map service, as a point query: "which flood-zone polygon contains this point?" It returns the zone code, such as AE or X, which I map to the model's high-risk or lower-risk group. I tell users it's not an official determination and link to FEMA's Map Service Center to check.

**Q: What happens if FEMA's server is down?**
A: The app degrades gracefully. Each call has a 6-second timeout and two retries. If it still fails, the page says the flood map service didn't answer and asks the user to choose the zone, while keeping whatever did work, such as the county. Failures are never cached, so the next try asks again. My tests simulate timeouts, server errors and unmatched addresses with fake responses, and they block real network access, so they run offline.

**Q: Your app says raising a building lowers expected losses by about 29%. Would you tell a homeowner they'll save 29%?**
A: No. That's the model's average difference between similar insured buildings that are and aren't raised: an association learned from data, not an engineering estimate for one building. Raised buildings differ in other ways too; I even found a Simpson's paradox, where the effect looked different within high-risk zones than overall. And it's an expected claim cost, not a premium. The app says this right next to the number.

## Milestone 16: Fixes from testing the live app, and a regional test that failed (2026-11-13)

### What I did
I tested the deployed app like a real user and fixed what I found. The arrows next to "compared with the state average" pointed the wrong way, the map grabbed the scroll wheel, and an address found on the first page didn't carry over to the calculator. I made the map open on "compared with the state average", added thick state borders and labels, and added a small card after a county is clicked. I also explained more honestly that North Carolina's level comes from a few hurricanes. Then I tested a modelling idea, one level per North Carolina region instead of one per state, using a rule I wrote down before running it. It failed, so the model didn't change.

### Why I did it that way

**The arrow bug: read the library, then test the fix.**
- *Intuition:* Streamlit's `st.metric` draws an up or down arrow next to a comparison. It decides which by looking at whether the text starts with a minus sign. Our comparisons are words ("15% below the Virginia average"), so every one counted as "up". Combined with the "inverse" colour setting (up = bad = red), good news was shown as a red up arrow.
- *Fix:* one small function decides the arrow and colour from the number itself: below → down and green, above → up and orange, about the same → none.
- *Testing the test:* I temporarily put the old behaviour back and checked that the new test fails. A test that can't fail proves nothing.

**Why the map now starts with "compared with the state average".**
- *Intuition:* comparing two counties in the same state is like comparing two houses on the same street. They share the same weather history, so differences mostly come from the buildings. Comparing North Carolina with Virginia is more like comparing two streets after different storms. North Carolina's level is set mostly by Matthew (2016), Florence (2018) and Helene (2024). Another 16 years of hurricanes could move it a lot.
- *So:* the default view shows the more certain comparison, and the app says plainly that state-to-state differences are less certain.

**One storm level per state, and what that means for inland counties.**
- *Example:* Lee County, NC (inland) has a model estimate of about $1,218 a year, but its 2010-2025 average was $551.
- *Why:* the model gives every NC county the same statewide "storm level", and that level is driven by coastal and mountain hurricanes. For counties that storms mostly missed, the estimate may be too high. The app now says so instead of implying the estimate is surely right.

**The regional test: a pre-stated rule, and accepting a "no".**
- *Idea:* give NC's Mountains, Piedmont and Coastal Plain their own levels. Regions came from an official county-level list: NOAA's climate divisions, which group NC's counties into Mountain, Piedmont and Coastal Plain divisions.
- *Pre-stated rule:* the same rule as the county factor, written and committed *before* running. Adopt only if the average leave-one-year-out (LOYO) score over three states improves, the 2021-2025 holdout isn't worse in more than one state, and NC's own score doesn't fall. Writing the rule first stops me from moving the goalposts after seeing the numbers.
- *Reminder, LOYO:* fit on 15 years, predict the 16th, repeat for every year, and average how much better than a "no information" model each prediction is.
- *What looked promising:* over all 16 years, one statewide level was too high in the Piedmont (actual about half of expected) and too low in the Mountains (about 2×, mostly Helene).
- *What happened on unseen years:* NC's LOYO score fell from 12.5% to 8.9% (better in only 3 of 16 years), and its holdout from +16.7% to −18.2% (worse than predicting the same rate everywhere).
- *Why:* a region's history is mostly where past storms happened to go. Helene made the Mountains look risky, but a model that learned that would have over-charged the Mountains in other years. This is **overfitting to storm tracks**, the same lesson as the county factor in Milestone 13.

**Small UX fixes that matter.**
- The calculator now receives the address itself, so "add a year and look up again" works. Changing only the year re-applies the age group and the pre/post-FIRM answer (pre-FIRM = built before the area's first flood insurance rate map).
- Unfound addresses get one automatic retry without "Apt 4B"-style parts, and advice to add the ZIP code or try a nearby house number.
- "Main home: Yes" raises the estimate. I checked the data before explaining it: main homes' claims averaged about $39,000 vs $29,000 for second homes, with twice as much paid for belongings. The effect comes mostly from North Carolina.

### Interview questions

**Q: You found that your model was too high for inland North Carolina counties. Why didn't you fix it?**
A: I tested the fix, one level per region, with a rule I wrote down before running it. Over the past 16 years, the Piedmont's actual losses were about half the statewide estimate, so a regional level looked obviously right. But when I predicted each year from the other 15, the regional model was worse in 13 of 16 years, and on the 2021-2025 holdout it was worse than a model with no information at all. Regional history mostly reflects where past hurricanes went, and the next one goes somewhere else. So I kept one level per state and changed the app's wording instead: it says inland estimates may be high.

**Q: Why write the decision rule down before running the test?**
A: Because once you see the numbers, it's tempting to pick whichever measure makes your idea look good. Committing the rule to Git first, with a timestamp, makes the test honest and auditable. It's the same idea as pre-registering a study. Here the rule was the one already used for the county factor, so the two geography decisions are judged the same way.

**Q: Tell me about a bug you found by using your own app.**
A: The metric arrows. "15% below the Virginia average" showed a red up arrow, which says the opposite of the words. I read Streamlit's source and found it decides the arrow by whether the text starts with a minus sign. Since my comparisons are words, everything counted as "up". I set the arrow and colour explicitly from the number, wrote a test across every county and several calculator profiles, and checked the test fails on the old code.

---

## Milestone 16 follow-up: the "stale module" error after deploys (2026-11-13)

### What I did
After a push, the live app sometimes broke with `module 'common' has no attribute ...` until it was rebooted. The cause: Streamlit re-runs each page's script on every click, but a *module* you `import` (like `app/common.py`) is loaded only once per server and then kept in memory. So the new pages were running against the old shared code. I added a small check at the top of `app/app.py`: on each run it looks at the file dates of `common.py`, `lookup.py` and `src/explain.py`, and reloads any that changed. I also added a test that fakes an old copy and checks it gets reloaded, plus a note in `app/README.md` to use "Reboot app" if it ever happens again.

### Why this way
- **Scripts vs modules.** Python remembers every imported module in a dictionary called `sys.modules`. A second `import common` just hands back the copy already in memory and doesn't read the file again. `importlib.reload` makes Python read the file again and update that same module in place.
- **Reload only when needed.** Reloading on every click would re-run `common.py`'s setup each time. Comparing file dates costs almost nothing.
- **One place.** All three pages run *inside* `app.py` (the menu, `st.navigation`), so one check covers every page, and the pages' own `import common` gets the fresh copy.
- **Order matters.** `common` imports `explain` and `lookup`, so those are reloaded first.
- No statistics here; this is software reliability, which matters just as much in a deployed model.

### Interview questions
**Q: Your app broke after a deploy even though the code was correct. Why?**
A: The server kept the old version of a shared module in memory. Streamlit re-runs page scripts but doesn't re-import modules, so new pages called functions the old module didn't have. I made the app reload a module when its file changes and added a test that simulates the stale state.

**Q: Why not just reload the module on every request?**
A: It works, but it re-runs the module's setup code on every click for no benefit. Checking the file's modified time is nearly free and reloads only after a real change. I also kept the manual fallback, rebooting the app, documented.

**Q: How did you test something that only happens on the cloud server?**
A: I reproduced the condition locally. In a test, I deleted an attribute from the in-memory module and marked it as loaded from an older file, then re-ran the app. The test checks that the attribute is back and that an unchanged module is *not* reloaded.
