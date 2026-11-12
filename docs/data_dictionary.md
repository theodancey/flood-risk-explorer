# Data Dictionary

Plain-English guide to the most important fields in the two FEMA datasets this project uses. Meanings are based on FEMA's official field descriptions (OpenFEMA `DataSetFields` endpoint, saved locally as `data/raw/*_fields.json` when you run `src/download_data.py fields`).

| Dataset | What one row is | Our file |
|---|---|---|
| **NFIP Redacted Claims v3** (`NfipClaims`) | One flood insurance **claim** (a request for payment after a flood) | `data/raw/NfipClaimsV3.parquet`, all of the US, all 84 columns |
| **NFIP Redacted Policies v3** (`NfipPolicies`) | One **policy transaction**: a flood insurance policy for one policy term (usually one year) | `data/raw/NfipPoliciesV3_VA.parquet`, Virginia only, 37 of 87 columns |

> **Why "Redacted"?** FEMA removes names and street addresses and blurs locations (for example, latitude/longitude rounded to 1 decimal ≈ 11 km) to protect policyholders' privacy. That's why this project works at the **area level** (ZIP, county, census tract) and never at the level of an individual property.

> **Note on versions.** The task asked for the "FIMA NFIP Redacted ... v2" datasets (`FimaNfipClaims`, `FimaNfipPolicies`). FEMA has retired those (their API URLs now return 404). `NfipClaims` / `NfipPolicies` **v3** are their official replacements with the same kind of data.

## Gotchas that apply to both datasets
- **Money is in nominal dollars.** A $50,000 claim in 1985 isn't comparable to a $50,000 claim in 2024. We'll need to adjust for inflation (or at least compare within similar years) before modelling.
- **Many fields are codes, not words.** For example, `occupancyType = 1` means "single family residence." The code lists are in FEMA's descriptions, and the key ones are copied below.
- **Empty doesn't always mean "no."** FEMA notes that a blank often means "not reported." Older records are much more likely to be blank.
- **Two flood-zone fields.** `ratedFloodZone` is the zone used to price the policy. `floodZoneCurrent` is the zone on today's map. They can differ because maps get redrawn.
- **Rating changed in October 2021 ("Risk Rating 2.0").** FEMA's new pricing method introduced new 2-digit occupancy codes and made some older rating fields less meaningful after 2021.

### Flood zones (from FEMA's Flood Insurance Rate Maps, FIRMs)
| Zone | Plain English |
|---|---|
| **A, AE, A1–A30, AH, AO, A99** | **High risk:** the "Special Flood Hazard Area" (SFHA), at least a 1% chance of flooding each year (the "100-year flood"). Most mortgages here require flood insurance. |
| **V, VE, V1–V30** | **High risk, coastal**, with storm waves ("velocity"). Usually the most expensive. |
| **B, X (shaded)** | Moderate risk (between the 100-year and 500-year flood areas). |
| **C, X (unshaded)** | Minimal risk. |
| **D** | Possible but undetermined risk. |

---

## Claims dataset (`NfipClaims` v3)

### When and where
| Field | Meaning | Units / format | Gotchas |
|---|---|---|---|
| `dateOfLoss` | Date water first entered the building | date | Many claims share one date (a single storm). |
| `yearOfLoss` | Year of the flood | year | Easiest field for "claims per year." |
| `state` | State abbreviation | e.g. `VA` | Includes territories (PR, VI, GU…). |
| `countyCode` | County FIPS code (a standard government code for each county) | text (format checked in notebook 01), e.g. `51810` = Virginia Beach | FEMA warns it may not always be the exact county of the property. Can be missing. |
| `censusGeoid` | Census **block group**: a small Census area of about 600–3,000 people | 12-digit text | First 5 digits = county, first 11 = census tract. |
| `reportedZipCode` | 5-digit ZIP code | text | Keep as text so leading zeros (e.g. `02134`) aren't lost. |
| `latitude`, `longitude` | Approximate location | decimal degrees, **1 decimal place** | Blurred for privacy (≈ 11 km). Can land in the wrong county. FEMA says to use `state`/`countyCode` for grouping. |
| `nfipCommunityName`, `nfipRatedCommunityNumber` | The local government (town/county) that runs floodplain rules | text; 6-digit community number (first 2 = state) | Useful for linking to community-level datasets later. |
| `eventDesignationNumber`, `floodEvent`, `ficoNumber` | Which named catastrophe the claim belongs to (e.g. "Hurricane Harvey") | text / number | EDN replaced FICO numbers in June 2020. Most small floods have no event name. |

### How much was paid (severity)
**Severity** means how big a claim is in dollars. (**Frequency** means how often claims happen.)

| Field | Meaning | Units | Gotchas |
|---|---|---|---|
| `amountPaidOnBuildingClaim` | Paid for damage to the building (walls, floors, wiring…) | dollars | Can be **negative**: a cheque that wasn't cashed and was re-issued. Often 0 or blank. |
| `amountPaidOnContentsClaim` | Paid for belongings (furniture, appliances, clothes) | dollars | Same issues. Basement contents aren't covered. |
| `amountPaidOnIncreasedCostOfComplianceClaim` | **ICC**: extra money (up to $30,000) to elevate/floodproof/demolish to meet local rules after a flood | dollars | Usually blank. |
| `netBuildingPaymentAmount`, `netContentsPaymentAmount`, `netIccPaymentAmount` | Payment minus any money recovered later | dollars | Closer to the true cost. Available mostly for newer claims. |
| `buildingDamageAmount`, `contentsDamageAmount` | Adjuster's estimate of the damage | whole dollars | Damage ≠ payment. Payment is capped by coverage and reduced by the deductible. |
| `totalBuildingInsuranceCoverage`, `totalContentsInsuranceCoverage` | The policy's coverage limit (the maximum it would pay) | whole dollars | NFIP caps: $250,000 building / $100,000 contents for homes. |
| `buildingDeductibleCode`, `contentsDeductibleCode` | **Deductible**: the amount the policyholder pays first, before insurance pays | code: `0`=$500, `1`=$1,000, `2`=$2,000, `5`=$5,000, `A`=$10,000, … | Code letters, not dollars. Translate before using. |
| `nonPaymentReasonBuilding`, `nonPaymentReasonContents` | Why a claim was closed without payment | code, e.g. `01` = damage less than deductible, `06` = not actually a flood | Explains many of the $0 claims. |
| `buildingPropertyValue`, `buildingReplacementCost` | Building value before the flood / cost to rebuild | whole dollars | Replacement cost is **not** market value (no land). |
| `waterDepth` | Depth of flood water | inches (sometimes feet!) | FEMA admits mixed units, so treat it carefully. |
| `floodWaterDuration` | Hours water stayed in the building | hours | Mostly blank. |
| `causeOfDamage` | What caused the flooding | code: `1` = tidal (storm surge), `2` = river/stream overflow, `4` = rainfall accumulation, … | Useful for coastal vs. inland comparisons. |

### Building characteristics
| Field | Meaning | Codes / gotchas |
|---|---|---|
| `occupancyType` | What the building is used for | `1` single family, `2` 2–4 units, `3` 5+ units, `4` non-residential; 2-digit codes (`11`, `12`, …) are Risk Rating 2.0 versions. |
| `primaryResidenceIndicator` | Owner lives there most of the year | true/false |
| `numberOfFloorsInTheInsuredBuilding` | Floors | `1`, `2`, `3` = 3+, `4` split-level, `5` mobile home, `6` townhouse 3+ floors |
| `basementEnclosureCrawlspaceType` | Basement type | `0` none, `1` finished basement, `2` unfinished, `3` crawlspace, `4` subgrade crawlspace |
| `elevatedBuildingIndicator` | Built up on posts/piles/walls | true/false |
| `postFIRMConstructionIndicator` | Built **after** the community's first flood map (so built to flood-safe rules) | true/false. "Pre-FIRM" (older) buildings tend to flood more. |
| `originalConstructionDate` | When it was built | date; often missing or a placeholder |
| `elevationDifference` | Height of the lowest floor above (+) or below (−) the **Base Flood Elevation** (the 100-year flood water level) | whole feet; mostly blank (needs an Elevation Certificate) |
| `ratedFloodZone`, `floodZoneCurrent` | Flood zone | see the table above |
| `crsClassCode` | Community Rating System class: 1 (best) to 10 (no discount). Communities earn premium discounts for good floodplain management. | blank or 10 = no discount |
| `condominiumCoverageTypeCode` | Condo coverage type | `N` not condo, `U` unit, `H`/`L` whole-building condo master policy (RCBAP) |
| `policyCount` | Number of insured units on the policy | Usually 1. A condo master policy (**RCBAP**) can cover many units. |

---

## Policies dataset (`NfipPolicies` v3), Virginia, North Carolina and Maryland, 37 columns

Policies are how we measure **exposure**: how much insurance is in force, in how many places, for how long. Without exposure, a county with 10,000 claims might look risky only because it has 500,000 policies.

| Field | Meaning | Units / format | Gotchas |
|---|---|---|---|
| `id` | Record ID | number | One row = one policy **term**, so the same building appears once per year it was insured. |
| `policyEffectiveDate` | Start of coverage | date | Coverage usually begins 30 days after you buy it (the standard waiting period). |
| `policyTerminationDate` | End of coverage | date | Usually 1 year after start. Can be earlier if cancelled. |
| `policyTermIndicator` | Length of term | `1` = 1 year, `3` = 3 years | Multi-year terms count as more exposure. |
| `cancellationDateOfFloodPolicy` | Date the policy was cancelled, if ever | date | Mostly blank (not cancelled). |
| `originalNBDate` | Date the building was **first** insured (NB = "new business") | date | Shows how long a property has stayed insured. |
| `policyCount` | Number of insured units | count | **Use this, not row counts,** when counting policies. Condo master policies count as many units. |
| `totalBuildingInsuranceCoverage`, `totalContentsInsuranceCoverage` | Coverage limits | whole dollars | Building 0 = contents-only policy (common for renters). |
| `buildingDeductibleCode`, `contentsDeductibleCode` | Deductibles | codes as in claims | |
| `totalInsurancePremiumOfThePolicy` | **Premium**: price paid for the insurance | whole dollars | Negative = a refund. Excludes fees. |
| `policyCost` | Premium plus fees and surcharges | whole dollars | What the customer actually paid. |
| `buildingReplacementCost` | Estimated cost to rebuild | whole dollars | Mostly reported after Risk Rating 2.0 (Oct 2021). |
| `propertyState` | State | `VA`, `NC`, `MD` | One file per state: `NfipPoliciesV3_VA/NC/MD.parquet`. |
| `reportedZipCode` | ZIP code | 5-digit text | |
| `reportedCity` | City | text | Often `"NA"`, which FEMA uses for "unavailable." |
| `censusGeoid` | Census block group | 12-digit text | **Policies have no county field.** Use the first 5 digits as the county FIPS code. |
| `latitude`, `longitude` | Blurred location | 1 decimal place | Same privacy rounding as claims. |
| `nfipRatedCommunityNumber`, `nfipCommunityNumberCurrent`, `nfipCommunityName` | Local community | text | Names sometimes have a `*` suffix (e.g. `MONTGOMERY COUNTY *`), which marks unincorporated areas. |
| `crsClassCode` | Community discount class | 1–10 | Blank is common. |
| `ratedFloodZone`, `floodZoneCurrent` | Flood zone | see table above | `floodZoneCurrent` is often blank. |
| `rateMethod` | How the policy was priced | code, e.g. `7` = Preferred Risk Policy (cheap policy for low-risk zones), `1` = Manual | Many codes. Changed with Risk Rating 2.0. |
| `elevationDifference` | Lowest floor vs. 100-year flood level | feet | Mostly blank. |
| `elevatedBuildingIndicator`, `postFIRMConstructionIndicator`, `primaryResidenceIndicator` | Yes/no building facts | true/false | |
| `occupancyType`, `numberOfFloorsInInsuredBuilding`, `basementEnclosureCrawlspaceType`, `condominiumCoverageTypeCode` | Building use and shape | codes as in claims | |
| `foundationType` | Foundation | `1` slab, `2` basement, `3` crawlspace, `4`–`6` elevated; `0` / blank = not reported | Mostly reported only after Risk Rating 2.0. |
| `originalConstructionDate` | Year built | date | |

## Differences by state (Milestone 13)

The same 37 policy columns were downloaded for each state with the same API filter
(`propertyState eq 'XX'`). Every field has the same meaning and coding in all three states. The
differences are in how often fields are filled in, and in the building mix:

| | Virginia | North Carolina | Maryland |
|---|---|---|---|
| Policy rows (all years downloaded) | 1,668,793 | 2,207,699 | 764,268 |
| Paid claims 2010-2025 | 9,557 | 37,825 | 4,512 |
| Rows with a **V zone** (coastal wave zone) | 0.8% | **4.9%** | 1.1% |
| `crsClassCode` blank | 17.5% | 23.4% | **54.4%** |
| `elevationDifference` blank | 66.8% | 53.4% | 64.5% |
| `basementEnclosureCrawlspaceType` blank | 58.7% | 66.2% | 48.5% |
| Claims with a negative payment (refund/correction) | 0 | 2 | 1 |

- **Fields the model uses** (flood zone, year built, floors, occupancy, elevated, primary
  residence, coverage, post-FIRM, county) are more than 99.9% complete in every state. The
  differences above are all in fields we don't model, so no state-specific cleaning was needed.
- **V zones** are about six times more common in NC (the Outer Banks). We still merge V into the
  "A/V high risk" group in every state, so all three states use the same features. The 1% rule can
  only force merges, never splits.
- **Negative payments:** three NC/MD claims have a negative total paid. Each total is floored at
  $0, so the claim counts as "not paid" (see `actuarial.total_paid`).
- **Census geocoding errors:** a few policies in one state's file carry a county code from
  another state (e.g. an NC code in the Virginia file). They are pooled into that state's
  "Other/unknown county" group, so a county code never appears under two states.
- **Tidal ("coastal_tidal") definitions differ by state** because each state defines its own
  tidal zone in law: Virginia uses Tidewater (46 localities), North Carolina the 20 CAMA counties,
  and Maryland the 16 counties + Baltimore City in the Critical Area. Sources are in
  `docs/modeling_dataset.md`.

## Official sources
These URLs were used (and worked) for this project's downloads:
- Claims API (v3): https://www.fema.gov/api/open/v3/NfipClaims
- Policies API (v3): https://www.fema.gov/api/open/v3/NfipPolicies
- Official field definitions: https://www.fema.gov/api/open/v1/DataSetFields?$filter=openFemaDataSet%20eq%20%27NfipClaims%27%20and%20datasetVersion%20eq%203 (swap in `NfipPolicies` for the policy fields)
- OpenFEMA dataset catalog entry (record counts, last refresh, download links): https://www.fema.gov/api/open/v1/DataSets?$filter=name%20eq%20%27NfipClaims%27

Human-readable pages: start at the OpenFEMA data sets list, https://www.fema.gov/about/openfema/data-sets. fema.gov's web pages refuse scripted requests, so I couldn't confirm the exact page links. Open them in a browser to check.
