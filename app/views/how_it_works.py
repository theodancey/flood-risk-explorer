"""
Page 3: How It Works. A short summary, data sources, methods, key findings (with the North Carolina
numbers each clearly labelled), how the certainty signals work, limitations, future work and an
About section. This page stays technical, but every term is defined the first time it is used.
The few numbers that come from the app data are read from app/data/state_averages.json so they
always match the other pages.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

st.title("How It Works")
st.caption("Where the numbers come from, how the model was built and tested, and what it can't tell you.")
C.disclaimer()

av = C.state_averages()["states"]
pp = {s: av[s]["expected_pure_premium_per_policy_year"] for s in C.STATES}
act = {s: av[s]["actual_loss_cost_2010_2025"] for s in C.STATES}

st.header("In short")
st.markdown(f"""
- **What it shows:** the average yearly flood insurance loss for insured buildings in each county of
  Virginia, North Carolina and Maryland, and for kinds of buildings you describe. It is built from
  16 years (2010-2025) of public FEMA flood insurance data.
- **How:** a classic insurance pricing model (two GLMs: how often claims happen × how big they are),
  tested on years and states it had never seen. Building age and flood zone matter most.
- **What it isn't:** a prediction, rating or quote for any one property. North Carolina's level
  ({C.money(pp['NC'])} a year) is much higher than Virginia's ({C.money(pp['VA'])}) or Maryland's
  ({C.money(pp['MD'])}), mostly because of four big hurricanes.
""")

st.header("Data sources")
st.markdown(f"""
All data is free and public. Downloaded through official APIs; nothing here is affiliated with the
agencies below.

| Data | Source | Downloaded |
|---|---|---|
| NFIP **claims** (v3): 2.7 million US flood insurance claims since 1978 | OpenFEMA API, [`NfipClaims` v3](https://www.fema.gov/api/open/v3/NfipClaims) ([OpenFEMA data sets](https://www.fema.gov/about/openfema/data-sets)) | 2026-11-10 |
| NFIP **policies** (v3): 4.6 million policy terms since 2009 in VA, NC and MD | OpenFEMA API, [`NfipPolicies` v3](https://www.fema.gov/api/open/v3/NfipPolicies) | VA 2026-11-10; NC, MD 2026-11-12 |
| Inflation (CPI-U), to put all losses in 2025 dollars | [U.S. Bureau of Labor Statistics public API](https://www.bls.gov/developers/), series CUUR0000SA0 | 2026-11-10 |
| County names | [U.S. Census Bureau 2020 county codes](https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt) | 2026-11-10 |
| Tidal-area definitions (one per state) | [Code of Virginia § 62.1-44.15:68](https://law.lis.virginia.gov/vacode/title62.1/chapter3.1/section62.1-44.15:68/) (Tidewater); [N.C. Gen. Stat. § 113A-103](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/BySection/Chapter_113A/GS_113A-103.html) (CAMA counties); [Md. Code, Nat. Res. § 8-1807](https://mgaleg.maryland.gov/mgawebsite/Laws/StatuteText?article=gnr&section=8-1807) (Critical Area) | 2026-11-11 to 11-12 |

FEMA publishes the NFIP (National Flood Insurance Program) data with personal details removed and
locations coarsened, so the finest reliable level is the **county**. This product uses FEMA's OpenFEMA
API but is not endorsed by FEMA, and FEMA may update the data at any time.
""")

st.header("Methods in plain English")
st.markdown("""
**The four basic measures** (the actuary's toolkit):
- **Exposure**: how much insurance was in force, in *policy-years*. One building insured for a full
  year = 1 policy-year; for six months = 0.5. The other pages call this "insured buildings per year".
  Comparing claims to exposure (not just counting claims) is what makes areas of different sizes comparable.
- **Frequency**: paid claims per policy-year (the other pages: "chance of a claim each year").
  *Example:* 65 claims from 10,000 policy-years = 6.5 per 1,000.
- **Severity**: the average size of a paid claim ("typical claim size"). *Example:* $2M paid over 65
  claims ≈ $30,800 per claim.
- **Pure premium** (expected loss cost): frequency × severity = the expected claim cost per policy-year.
  The other pages call it the **"expected yearly loss"**. *Example:* 0.0065 × $30,800 ≈ $200. This is
  the part of an insurance price that pays for claims, before expenses and profit.

**The models.** A **GLM** (generalized linear model) is a regression built for outcomes like claim
counts and claim sizes. Each characteristic gets a **multiplier** (a *relativity*): for example,
"buildings in B/C/X zones have about 0.34× the expected loss of A/V-zone buildings". The price is a
base rate times one multiplier per characteristic, which is why insurers have used GLMs for decades:
every number can be read and checked.
- **Frequency:** a *Poisson* GLM (a probability distribution for counts) with exposure as an *offset*
  (a term that scales the prediction to the time insured), so it predicts a claim rate per policy-year.
- **Severity:** a *Gamma* GLM (a distribution for positive, right-skewed amounts: many small claims and a
  few huge ones) fitted to individual claims.
- Characteristics used: flood zone, year built, occupancy (what the building is used for), building
  coverage, elevated, floors, primary residence, plus three **interactions** (pairs of characteristics
  whose combined effect differs from the product of their separate effects; the charts call this
  "Combined effects") and a **state factor** (one multiplier per state). All losses are in 2025 dollars,
  for an *average year* of storms.

**Explaining one result: Shapley values.** The "what raises or lowers it" charts split the gap
between the state average and the result into one bar per characteristic. Because characteristics
overlap, the split uses **Shapley values** (from game theory: each characteristic's effect averaged over
every possible order of adding them), so the bars don't depend on order and multiply exactly to the
result. The app calls this a "fair-share split". Effects under 2% are grouped as "Other small effects".

**GLM vs XGBoost.** XGBoost (gradient-boosted decision trees) is a flexible machine-learning model. It
was about 4 points more accurate on average, but under a decision rule **written down before the
results**, it failed on catastrophe years and on balance across segments (it over- or under-predicted
some groups of buildings, such as newer buildings or low-risk zones), and one version moved predictions
the wrong way for elevated buildings. For pricing, getting the storm years and every segment right
matters more than a small average gain, so **the GLM is in production** and XGBoost stays as the benchmark.

**Validation: testing on data the model hasn't seen.**
- **Leave-one-year-out (LOYO):** fit on 15 of the 16 years (2010-2025), predict the year left out, and
  repeat for every year. Floods come in storm years, so holding out whole years is the honest test:
  a model that has "seen" Hurricane Florence can't be tested on Florence.
- **Holdout:** fit on 2010-2020 and predict 2021-2025, once, at the end.
- **Multistate transfer test:** the model was first built on Virginia alone, then applied **unchanged** to
  North Carolina and Maryland. It ranked their buildings almost as well as Virginia's (the risk factors
  travel), but badly under-estimated North Carolina's overall level (see the numbers below), which is why
  each state now has its own level. A **leave-one-state-out** test (fit on two states, predict the third)
  confirmed it.
""")

st.header("Key findings")
st.markdown(f"""
1. **North Carolina is a different kind of flood state.** Three numbers describe it, and each answers a
   different question:

   | Number | What it compares | Question it answers |
   |---|---|---|
   | **{C.money(act['NC'])} vs {C.money(act['VA'])}** a year (about {act['NC'] / act['VA']:.0f}×) | Actual 2010-2025 loss per policy-year, NC vs VA, raw | What really happened, with each state's own mix of buildings |
   | **6.2×** | The fitted NC state factor vs Virginia | How much riskier NC is *for the same kind of building* (NC's insured buildings are newer on average, so this is higher than the raw ratio) |
   | **6.6×** | NC's actual claims vs what the Virginia-only model expected for NC's buildings | How wrong a Virginia-only model was for NC, before the state factor was added |

   NC claims happen about 3× as often and are about 1.7× as large, driven by Irene, Matthew, Florence
   and Helene. The model's NC average for a typical year is {C.money(pp['NC'])} (VA {C.money(pp['VA'])},
   MD {C.money(pp['MD'])}).
2. **A few storms drive most losses.** In the early Virginia analysis, which used 2009-2025 (it included
   the 2009 nor'easter, before the 2010-2025 window used everywhere else), the worst 3 years produced
   **69%** of all losses. In North Carolina in 2018, Hurricane Florence's claims were about **46 times**
   what the Virginia-only model expected for that year.
3. **Building age and flood zone drive risk most.** Buildings from 1960-79 have about **2×** the expected
   loss of 1980-99 ones (pooled over three states); B/C/X zones about **0.34×** A/V zones;
   non-residential buildings about **4.3×** single homes; elevated buildings about **0.71×**.
4. **The risk factors transfer between states; the level doesn't.** The Virginia model ranked NC and MD
   buildings nearly as well as Virginia's (Gini 0.49-0.51 vs 0.53, where the Gini coefficient measures how
   well a model sorts risky from safe buildings, 0 = no better than random). The fitted state factor puts
   NC at **6.2×** Virginia and MD at **1.0×**.
5. **The validated model beats a no-information baseline on unseen data in every state**: 11-13% deviance
   improvement on leave-one-year-out and 10-17% on the 2021-2025 holdout (deviance = a goodness-of-fit
   score for these distributions; the baseline predicts the same rate for every building).
6. **County claims history didn't predict the future.** Adjusting each county by its own history made
   unseen years worse, badly in NC (holdout improvement 16.7% → 0.04%), so the model doesn't use it.
""")

st.header("How the certainty signals work")
st.markdown("""
The County Explorer shows two plain signals instead of one label:

- **How much local claims data** there is to check the estimate against: *plenty*, *some* or *limited*.
  This is the county's **credibility Z** (a number from 0 to 1): how much weight the county's own claim
  history would deserve next to the statewide model, given how many claims it has. It is worked out with
  the **Bühlmann-Straub** method, the standard actuarial credibility formula: Z = n / (n + k), where n is
  the county's number of claims and k measures how noisy claims are compared with real differences
  between counties. *Plenty* = Z ≥ 0.8; *some* = 0.5 ≤ Z < 0.8; *limited* = Z < 0.5 or no paid claims.
  (Z is for claim size; claim frequency earned Z = 0, because county differences in how often claims
  happen were no bigger than storm-to-storm noise.)
- **Whether past claims agree with the model**, shown only when they differ a lot. This is the county's
  2010-2025 claims divided by what the model expected *for each year's actual storms* (A/E, actual vs
  expected; 1.0 = perfect agreement). It is shown when A/E is below 0.5 or above 2.0.
- **Red** is used only when the data is limited *and* A/E is below 1/3 or above 3.

These replace the older High / Medium / Low label (High = Z ≥ 0.8 and A/E within 0.8-1.25; Low = no
claims, Z < 0.5, or A/E outside 0.5-2.0; Medium = everything else). The estimate itself never uses the
county's history (see finding 6); the signals only say how far to trust it.
""")

st.header("Limitations")
st.markdown("""
- **Area-level only.** The data is aggregated and anonymized. Every number is the average for a group of
  insured buildings in an average year, never a prediction, rating or quote for a specific property.
- **Catastrophe-driven losses.** Sixteen years is a small sample of hurricanes; another 16 years could
  give quite different levels, especially for North Carolina. This is not a catastrophe (hurricane
  simulation) model.
- **Data quality.** Policy data starts in 2009; some building fields are blank or coded differently
  before and after FEMA's Risk Rating 2.0 (October 2021); recent years can still gain late claims; a few
  county codes are wrong.
- **Who buys NFIP insurance isn't random.** Results describe insured buildings, which are concentrated
  in high-risk zones.
- **No county factor.** A county's estimate comes from its mix of insured buildings and its state's
  level, not its own claims history: in testing, county adjustments made predictions for unseen years
  worse, because a county's history is often one storm's footprint and the next storm goes elsewhere.
  So county estimates can differ a lot from county history; the certainty signals say where.
- **Coverage band and year built are partly stand-ins** (for building value and pre-flood-map
  construction), so the multipliers describe the model, not necessarily cause and effect.
""")

st.header("Future work")
st.markdown("""
- **Bring geography back through storm-independent features** rather than county claims history: for
  example ground elevation, distance to the coast or a river, and each area's share of exposure in FEMA
  high-risk zones. These describe *where water can go*, so they should hold up in years with different storms.
- ZIP-code or census-tract detail, more states, and catastrophe-adjusted pricing (spreading rare storm
  losses over many years with a catastrophe load).
""")

st.header("About")
st.markdown(f"""
Built by **Theo Dancey**, a Statistics & Data Science student at Virginia Tech and aspiring actuary, as
a portfolio project in actuarial modelling: exposure, frequency and severity GLMs, credibility,
validation on unseen years and states, and model explainability.

Code, reports and the full write-up: [{C.GITHUB_URL.removeprefix('https://')}]({C.GITHUB_URL})
""")

C.footer()
