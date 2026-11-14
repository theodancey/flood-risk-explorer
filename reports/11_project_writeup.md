# Flood Risk Explorer: Project Write-Up

*Theo Dancey · Statistics & Data Science, Virginia Tech · November 2026*
*Live app: [flood-risk-explorer.streamlit.app](https://flood-risk-explorer.streamlit.app) · Code: this repository*

> All results are **area-level** estimates from aggregated public data. Nothing here is a prediction, quote or risk rating for an individual property.

## The problem
Flood is one of the hardest risks to insure. Most years are quiet, then one hurricane produces more losses than the previous decade combined. I wanted to answer three questions the way an actuary would, using only public data:

1. How often do flood insurance claims happen in an area (**frequency**)?
2. How big are they when they happen (**severity**)?
3. Putting those together, what is the expected yearly loss per insured building (**pure premium**), and what drives it?

Then I wanted to make the answers usable: a web app where anyone can pick a county and see its expected loss and the reasons behind it.

## The data
FEMA publishes the National Flood Insurance Program (NFIP) records through its free OpenFEMA API. I downloaded **2.7 million claim records** (nationwide, since 1978) and **4.6 million policy records** for Virginia, North Carolina and Maryland (since 2009). The policy records give **exposure**: one building insured for one year counts as one *policy-year*, so a 6-month policy counts as 0.5. After cleaning, the modelling period 2010–2025 covers **4.1 million policy-years and 51,894 paid claims** in **257 counties**. I converted all losses to 2025 dollars with the CPI so that a 2011 claim and a 2024 claim are comparable.

The data is grouped: FEMA removes exact addresses, so I can describe *groups* of similar buildings in an area, never an individual home. That is why the app only reports area-level results.

## My approach
- **Actuarial basics first.** Before any model, I computed frequency, severity and pure premium by year and county. This showed the most important fact about flood: **3 storm years caused 69% of Virginia's 2009–2025 losses**, and the median year's loss cost was about half the average.
- **One factor at a time.** I looked at each rating factor (flood zone, building age, occupancy, elevation, floors, coverage and others) on its own. Some one-way results were misleading. For example, elevated buildings looked riskier overall but were safer within every flood zone (Simpson's paradox), which is why multi-factor models are needed.
- **GLMs (generalized linear models).** A GLM predicts a quantity as a base rate times one multiplier per factor, which makes it easy to read. I used a **Poisson GLM** for claim counts (with exposure as an offset, so more policy-years means proportionally more expected claims) and a **Gamma GLM** for claim size. Multiplying the two gives the pure premium.
- **Validation on unseen years.** Each year is predicted by a model that never saw it (*leave-one-year-out*), and 2021–2025 was also held out once. I scored models by **deviance improvement**: how much better they predict than a no-information model that gives everyone the average.
- **A challenger.** I tuned **XGBoost** (gradient-boosted trees) without leaking test years into the tuning, then compared it with the GLM on identical folds, with bootstrap confidence intervals.
- **Explanations.** I used relativity tables and **SHAP** values (a way of splitting a prediction into each factor's contribution) to show what drives risk, and built exact waterfall charts for the app.

## Key decisions and why
- **I wrote the decision rules down before seeing results.** For example: "XGBoost replaces the GLM only if it is better on unseen years *and* not worse in catastrophe years *and* doesn't push any customer segment out of balance." This stopped me from choosing whichever model happened to look best.
- **Storm-aware tests instead of plain p-values.** Claims cluster in storms, so a plain Poisson test treats one hurricane's thousands of claims as thousands of independent facts. With cluster-robust confidence intervals (about 3.6× wider) and the unseen-year test, most features that looked "significant" dropped out, including geography.
- **No county factor in the final price.** I tested a credibility-weighted county factor (Bühlmann–Straub credibility blends a county's own history with the state average, depending on how much data it has). It helped in Virginia alone, but with three states it made unseen years worse, badly in North Carolina, where a county's history is mostly one hurricane's footprint and the next storm goes elsewhere. My pre-stated rule dropped it. The app still shows each county's own history and says how much data backs the estimate.
- **A state factor.** I first applied the Virginia model unchanged to North Carolina and Maryland. It **ranked** risk almost as well out of state, but North Carolina's losses were **6.6×** what it expected, nearly all in four hurricane years. So the relativities travel between states but the level doesn't, and each state gets its own level.
- **GLM over XGBoost.** XGBoost was genuinely more accurate on average, but it failed my pre-stated checks (details below). I kept a **hybrid GLM** that adds the interactions XGBoost found, such as coverage × occupancy, so I kept some of its insight in a model that can be written as a rating table.

## Results
- **The production GLM beats the no-information baseline by 11–13% deviance on unseen years** (Virginia 11.0%, North Carolina 12.5%, Maryland 13.3%, 16-year leave-one-year-out).
- **XGBoost was about 4 points more accurate on average** (pooled +3.8 points, 95% CI 1.6 to 5.5), but it was **worse in catastrophe years** in Virginia and Maryland and **pushed segments out of balance** (for example, some segments' actual losses were only 0.61–0.79 of what it predicted). The GLM stays in production; XGBoost is the accuracy benchmark.
- **What drives risk:** building age and flood zone matter most. In the Virginia model, buildings from 1960–79 cost about 3.5× the base building, and moderate/low-risk zones (B/C/X) about 0.34×. Occupancy, coverage and elevation affect mainly claim *size*. These big effects point the same way in all three states, which is good evidence they're real.
- **North Carolina is a different kind of flood state:** a loss cost of **$988** per policy-year vs about **$198** in Virginia and Maryland (2025 dollars), driven by Irene, Matthew, Florence and Helene.
- **The app** covers all 257 counties with a clickable map, an address lookup (Census Geocoder and FEMA's flood map), county pages and a building-type calculator, backed by automated tests.

## Limitations
- **Catastrophes dominate.** Sixteen years contain only a handful of major storms, so the North Carolina level in particular could look quite different with another 16 years of weather.
- **No geography inside a state** beyond each county's mix of buildings. County estimates can differ a lot from county history; the app says so where they do.
- **Insured buildings only.** People choose whether to buy NFIP cover, so results describe the insured population, not every building.
- **Associations, not causes.** For example, coverage band partly stands in for building value.
- **Three neighbouring Atlantic states,** with slightly different legal definitions of "tidal" areas.
- **Area level only.** Nothing here prices an individual property.

## What I'd do next
1. **Bring geography back through storm-independent features**, such as ground elevation, distance to the coast and each area's share of buildings in high-risk zones, instead of county claims history.
2. **Catastrophe-adjusted pricing:** a separate catastrophe load built from storm models, rather than whichever storms happened to hit in 2010–2025.
3. **Finer detail and more states:** ZIP-code or census-tract level, and Gulf Coast states with a different flood mix.

## How I built it
I built this project with Claude Code (an AI coding assistant), which wrote much of the code and documentation under my direction. I made the modelling decisions: the questions, the data and validation design, the rules for keeping or dropping features and models, and what the app shows. Every number here comes from the reports in this repository (`reports/01`–`10`).
