# 01 · Data Overview: NFIP Claims and Policies

*Downloaded 2026-11-10 from FEMA's OpenFEMA API. Analysis: `notebooks/01_first_look.ipynb`. Field meanings: `docs/data_dictionary.md`.*

## What the data contains

| | Claims | Policies |
|---|---|---|
| FEMA dataset | NFIP Redacted Claims **v3** (`NfipClaims`) | NFIP Redacted Policies **v3** (`NfipPolicies`) |
| Coverage | All of the US and territories (58 states/territories) | **Virginia only** (`propertyState = 'VA'`) |
| Rows | **2,725,989** claims | **1,668,793** policy records |
| Columns | All 84 | 37 of 87 (the ones needed for exposure) |
| Time span | Losses from 1978-01-01 to 2026-09-07 | Policy terms starting 2009-01-01 to 2026-11-29 |
| File | `data/raw/NfipClaimsV3.parquet`, **216.2 MB** | `data/raw/NfipPoliciesV3_VA.parquet`, **49.4 MB** |
| FEMA's last refresh | 2026-08-04 | 2026-10-14 |

- **Claims:** one row is a single flood insurance claim. It records when and where the flood happened, the flood zone, building details, and how much was paid for the building, contents, and Increased Cost of Compliance (ICC).
- **Policies:** one row is one policy term (usually a year of coverage) for one building. It records coverage amounts, premium, flood zone, location, and building details. Policies give us **exposure**, the "how many homes were insured" denominator we need to turn claim counts into claim *rates*.
- **Privacy:** both datasets are "redacted." There are no names or addresses, and locations are blurred (latitude/longitude rounded to 1 decimal). So all of our results will be **area-level**, never per property.

## Key findings

1. **A few catastrophe years dominate.** 2005 (Katrina, Rita, Wilma) alone has **277,027 claims**, about 10% of all claims since 1978. The top three years (2005, 2012 Sandy, 2017 Harvey/Irma/Maria) add up to 22%. A typical year has about 43,000 claims (median of yearly counts). For an insurer, this "lumpiness" is the core challenge of catastrophe risk.

2. **Three states account for almost half of all claims.** Louisiana (485,743), Florida (448,618), and Texas (394,204) together make up **49%** of claims, and the top 5 states (adding NJ and NY) make up **63%**.

3. **The top counties are coastal metro areas.** Harris County, TX (Houston) leads with **170,854** claims, followed by Jefferson Parish and Orleans Parish, LA (New Orleans).

4. **Virginia ranks 12th, and its risk is concentrated in Hampton Roads.** Virginia has **50,609** claims. The six Virginia areas with the most claims are all Hampton Roads cities: Virginia Beach (6,260), Norfolk (6,019), Hampton (5,774), Poquoson (4,290), Chesapeake (2,634), and Portsmouth (1,748). Together they hold **53%** of all Virginia claims. Inland, the Roanoke area (Roanoke city, Salem, Roanoke County) also makes the top 15, which is a reminder that river flooding matters too.

5. **Claim sizes are highly skewed.** Among the 2.1 million claims with a payment, the **median is $13,042 but the mean is $42,443**, more than 3× higher. The largest 10% of claims account for **51% of all dollars paid**, and the biggest single claim is about $10.8 million. Shapes like this are why we plot payments on a log scale and why a **Gamma** model (designed for positive, right-skewed amounts) is the standard choice for severity.

6. **About 22% of claims were closed with no payment** (608,577 claims), for example because the damage was below the deductible or wasn't caused by a flood. Frequency models need a clear rule for whether these count as "claims."

7. **Virginia's claim rate swings wildly from year to year.** Claims per 1,000 policies ranged from **1.1 (2023) to 60.8 (2009)**. Each spike lines up with one named storm: the November 2009 nor'easter (6,286 claims), Hurricane Irene in 2011, and Hurricane Matthew in 2016. Virginia's single largest event, **Hurricane Isabel (2003, 11,108 claims)**, happened before the policy data starts.

8. **Flood zones matter, but they don't tell the whole story.** In Virginia, high-risk "A" zones hold **52%** of policies but **74%** of claims. Moderate/low-risk zones (B, C, X) hold 47% of policies yet still produce **21%** of claims, so the cheaper zones aren't risk-free.

9. **Fewer Virginians are insured than a decade ago.** Policies by start year fell from a peak of **117,540 (2012)** to **92,259 (2025)**, a 21% drop. (This report doesn't test *why*. Price changes, such as FEMA's 2021 "Risk Rating 2.0" pricing overhaul, are one hypothesis worth checking.)

## Data quality issues to handle in cleaning

| Issue | Details | Plan |
|---|---|---|
| **v2 datasets retired** | FEMA retired `FimaNfipClaims`/`FimaNfipPolicies` v2 (404 Not Found). We use v3, their official replacement. | Note it in docs. Field names mostly match v2, with a few renamed (e.g. `floodZone` → `ratedFloodZone`). |
| **Exposure only from 2009** | Policy data starts in 2009, but claims go back to 1978. | Model frequency on 2009+ only. Earlier claims can still inform severity. |
| **Payment fields blank** | 20.8% of claims have *all three* payment fields blank (same rows). 86 claims have a negative total (re-issued cheques). | Treat blank as $0 / closed without payment. Investigate or drop negatives. |
| **Nominal dollars** | Amounts aren't inflation-adjusted. A 1985 dollar ≠ a 2025 dollar. | Adjust with a price index (e.g. CPI) or restrict to recent years before comparing severity. |
| **No county in policies** | Policies have no county field. | County = first 5 digits of `censusGeoid` (12 digits in all rows where present; 0.45% missing). We confirmed this rule: in claims, `countyCode` equals the first 5 digits of `censusGeoid` in 100% of rows that have both. |
| **Missing location** | `countyCode` is blank for 2.3% of claims and `censusGeoid` for 5.0%. | Fall back to ZIP or community number. Report how many are lost. |
| **County ≠ community** | FEMA warns that `countyCode` may not match the property's exact county. Some claims coded to Virginia Beach list Norfolk as their community. | Accept for area-level work. Mention it as a limitation. |
| **Policy rows ≠ policies in force** | Each row is a policy *term*. Counting by start year is a rough approximation. Condo master policies cover many units (`policyCount`). | Compute **earned exposure** (fraction of each term falling in each year, × `policyCount`). |
| **Incomplete recent years** | 2026 is partial (claims run to Sept 2026). 79 policy terms start after the download date. 2025 has only 13,143 claims nationally, which may be a quiet year or late-reported claims. | Exclude 2026. Treat 2025 with caution. |
| **Sparse detail fields** | Mostly blank: `elevationDifference` (67% of policies), `basementEnclosureCrawlspaceType` (59%), `contentsDamageAmount` (58% of claims), `nonPaymentReasonBuilding` (78%). | Don't rely on them as model features without a "missing" category. |
| **Coding changes** | Risk Rating 2.0 (Oct 2021) added new occupancy codes and changed rating methods. `waterDepth` is sometimes in feet instead of inches. | Map old and new codes to common groups. Don't use `waterDepth` without checks. |
| **Blurred coordinates** | Lat/long are rounded to 1 decimal (about 11 km). | Group by county/ZIP/tract, never by coordinates. |
| **Good news** | No duplicate IDs in either file. `yearOfLoss` always matches `dateOfLoss`. All county codes are 5 digits. | n/a |

## Official sources
- Claims API (v3): https://www.fema.gov/api/open/v3/NfipClaims
- Policies API (v3): https://www.fema.gov/api/open/v3/NfipPolicies
- Official field definitions: https://www.fema.gov/api/open/v1/DataSetFields?$filter=openFemaDataSet%20eq%20%27NfipClaims%27%20and%20datasetVersion%20eq%203 (and the same with `NfipPolicies`)
- Dataset catalog (record counts, refresh dates): https://www.fema.gov/api/open/v1/DataSets
- OpenFEMA data sets list (human-readable): https://www.fema.gov/about/openfema/data-sets (fema.gov web pages block scripted access, so I couldn't check this link automatically)

*Data note: this product uses FEMA's OpenFEMA API but isn't endorsed by FEMA. FEMA may update the data at any time, so re-running `src/download_data.py` later can give slightly different counts.*
