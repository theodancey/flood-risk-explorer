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
