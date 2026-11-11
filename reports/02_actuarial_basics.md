# 02 · Actuarial Basics: Virginia Frequency, Severity, and Pure Premium

*Analysis: `notebooks/02_actuarial_basics.ipynb`. Code: `src/actuarial.py`. Data: FEMA OpenFEMA NFIP claims and policies (v3), Virginia, downloaded 2026-11-10. All dollars are nominal (not adjusted for inflation).*

## Method in one box

| | How it's measured |
|---|---|
| **Exposure** | **Earned policy-years.** Each policy term is split across the calendar years it actually covered (days in the year ÷ days in that year), stopping early if the policy was cancelled. One policy term counts as 1, even a condo policy covering many units. |
| **Claims** | **Paid claims** (total paid > $0), matched to exposure by year of loss and county. |
| **Dollars** | **Total paid** = building + contents + Increased Cost of Compliance (ICC) payments, with blanks counted as $0. FEMA's "net" payment fields give the same total within 0.12%. |
| **Measures** | Frequency = paid claims ÷ exposure. Severity = dollars ÷ paid claims. Pure premium = dollars ÷ exposure = frequency × severity. |
| **Years** | Headline numbers use **2010–2025**. 2009 exposure is *estimated*, because the policy file is missing policies that started in 2008. 2026 is incomplete and excluded. |

## Key findings (interview-ready)

1. **Virginia's average flood loss cost is about $149 per policy per year.** Over 2010–2025 there were **1.48 million earned policy-years**, **9,557 paid claims**, and **$220.2 million** paid. That works out to a frequency of **6.5 paid claims per 1,000 policy-years** and an average paid claim (severity) of **$23,039**: 0.0065 × $23,039 ≈ **$149**. This is the pure premium, the part of the price that pays claims before expenses and profit.

2. **The "average" year is not a typical year: losses are catastrophe-driven.** The *median* year's pure premium is only **$79**, about half the $149 average. The three worst years (**2009 nor'easter, 2016 Hurricane Matthew, 2011 Hurricane Irene**) produced **69%** of all 2009–2025 losses, and the worst five produced **82%**. Only 4 of the 16 years from 2010 to 2025 had a loss cost above the average. A price set on a typical year would badly under-charge, so the premium has to build a cushion in quiet years to pay for storm years.

3. **Storm years are frequency events, not severity events.** Claim frequency swings about **40-fold**, from 0.7 per 1,000 policy-years (2023) to 28 per 1,000 (2011, Irene). The swing is even bigger if you count the estimated 2009 (about 62). Severity stays within about **$10K–$44K**. When a big storm hits, many more homes flood at once, but the average claim doesn't change dramatically. So pure premium follows frequency, and the frequency model will do most of the work.

4. **Earned exposure matters most where the data has gaps.** For 2010–2025, counting policies by start year differs from true earned exposure by only about **2%**, because most policies renew every year. But for **2009**, the raw earned exposure is only **46%** of the estimate, because policies already in force at the start of 2009 aren't in the file. Without that correction, 2009's claim frequency would look about **twice as high** as it was. Separately, notebook 01's method (weighting by `policyCount`) overstates exposure by about **13%** compared with one-policy-one-count.

5. **The riskiest places per policy aren't always the places with the most losses.** Ranked by pure premium (counties/cities with ≥5,000 policy-years), **Salem city ($455)**, **Newport News ($430)**, **Roanoke County ($305)**, and **Roanoke city ($301)** top the list, about 2–3× the state average. Three of the four are inland, in the Roanoke Valley, where **inland flooding** (about half river overflow and a third heavy rainfall, concentrated in 2013, 2018, and 2020) drives high frequency (13–15 paid claims per 1,000 policy-years). Meanwhile **Virginia Beach** has the most dollars by far: **30% of all losses** from **20% of the exposure**, at $222 per policy-year.

6. **Big totals can hide moderate risk.** **Hampton** and **Chesapeake** each have more than 130,000 policy-years (about 9–11% of exposure each), yet their pure premiums ($78 and $68) are about half the state average. **Poquoson**, which was 4th in total claims since 1978 (about three-quarters of them from Hurricane Isabel in 2003 and the 2009 nor'easter, both outside this window), has a pure premium of just $96 over 2010–2025. Rates, not totals, are what pricing needs, and the time window matters.

7. **Claim severity is skewed and has been rising.** The median paid claim is about **$10,200**, but the mean is about **$21,000**. A Gamma distribution captures the general shape but under-states the biggest claims (its 99th percentile is $109K vs $155K actual). The average paid claim rose from about **$16,500 (2010–2015)** to about **$27,200 (2020–2025)**. Some of that is inflation in building costs, which we'll adjust for before modelling.

8. **Claim counts are far more variable than a simple Poisson model assumes.** Across county-years, the claim counts vary about **28 times** more than a Poisson distribution with the right exposure would allow, and **66%** of county-years have zero paid claims versus 49% expected. Storms make claims arrive in clusters, so the frequency model will need a **negative binomial** (or similar) to handle this extra spread, and the evaluation must be careful with storm years.

## Caveats
- **Area-level only.** These are averages over many properties in a county or year, never a prediction for any single home.
- **Small areas are noisy.** For example, Rockingham County's $66K severity comes from just 13 paid claims, with about 70% of its losses in a single year (2018). That's why the ranking uses a minimum-exposure rule, and why the models will need credibility weighting.
- **Nominal dollars**, no inflation adjustment yet (planned for week 2).
- **About 24% of claims close without payment.** Counting all claims, frequency would be 8.5 rather than 6.5 per 1,000. Pure premium is unaffected.
- **County coding:** FEMA notes a claim's `countyCode` may not always be the property's exact county. Only 0.45% of exposure and 12 claims lack a county.
