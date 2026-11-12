"""
Core actuarial measures for the Virginia NFIP data: earned exposure, frequency,
severity, and pure premium.

Plain-English definitions (see LEARNING_NOTES.md for more):
  * Exposure (earned policy-years): how much insurance was "in force". One policy
    covering a full calendar year earns 1.0. Covering half the year earns 0.5.
  * Frequency     = paid claims / exposure          (how OFTEN claims happen)
  * Severity      = dollars paid / paid claims      (how BIG a typical claim is)
  * Pure premium  = dollars paid / exposure         (expected loss cost per policy-year)
                  = frequency x severity

Used by notebooks/02_actuarial_basics.ipynb, and later by the modelling notebooks.

Example (from the project folder):
    .venv\\Scripts\\python.exe src\\actuarial.py      # runs the self-test and builds
                                                    # data/processed/va_county_year.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

CLAIMS_FILE = RAW / "NfipClaimsV3.parquet"
POLICIES_FILE = RAW / "NfipPoliciesV3_VA.parquet"
COUNTY_NAMES_FILE = RAW / "census_counties_2020.txt"

# The three "amount paid" fields. Their sum is what the NFIP actually paid on a claim.
PAID_COLS = [
    "amountPaidOnBuildingClaim",                    # damage to the building itself
    "amountPaidOnContentsClaim",                    # belongings inside
    "amountPaidOnIncreasedCostOfComplianceClaim",   # ICC: rebuilding to meet local code
]
NET_COLS = ["netBuildingPaymentAmount", "netContentsPaymentAmount", "netIccPaymentAmount"]

# Years with complete data on BOTH sides (claims and exposure). See notebook 02:
#   2009 is missing the policies that started in 2008, and 2026 is still in progress.
FIRST_FULL_YEAR = 2010
LAST_FULL_YEAR = 2025


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def policies_file(state="VA"):
    """Path of one state's policy download (data/raw/NfipPoliciesV3_<ST>.parquet)."""
    return RAW / f"NfipPoliciesV3_{state.upper()}.parquet"


def load_policies(state="VA", path=None, extra_cols=()):
    """One state's policy terms, with the columns needed for exposure, plus a county code
    and a 'state' column (2-letter postal code).

    extra_cols: any other policy columns to load too (e.g. building features)."""
    cols = ["id", "policyEffectiveDate", "policyTerminationDate",
            "cancellationDateOfFloodPolicy", "policyCount", "censusGeoid", "ratedFloodZone"]
    cols += [c for c in extra_cols if c not in cols]
    pol = pd.read_parquet(path or policies_file(state), columns=cols)
    # Policies have no county field. The first 5 digits of the census block group ID
    # are the county FIPS code (checked in notebook 01: true for 100% of claims).
    pol["county"] = pol["censusGeoid"].str[:5]
    pol["state"] = state.upper()
    return pol


def load_claims(state="VA", path=CLAIMS_FILE, first_year=2009, extra_cols=()):
    """One state's claims (2-letter postal code) from first_year onward, with total paid.

    extra_cols: any other claim columns to load too (e.g. building features)."""
    cols = ["id", "dateOfLoss", "yearOfLoss", "state", "countyCode", "censusGeoid",
            "floodEvent", "policyCount", *PAID_COLS, *NET_COLS]
    cols += [c for c in extra_cols if c not in cols]
    clm = pd.read_parquet(path, columns=cols, filters=[("state", "==", state.upper())])
    clm = clm[clm["yearOfLoss"] >= first_year].copy()
    clm["county"] = clm["countyCode"]
    clm["total_paid"] = total_paid(clm)
    clm["is_paid"] = clm["total_paid"] > 0          # a "paid claim" = paid more than $0
    return clm


def load_va_policies(path=POLICIES_FILE, extra_cols=()):
    """Virginia policy terms (the original Virginia-only loader, kept for older code)."""
    return load_policies("VA", path, extra_cols)


def load_va_claims(path=CLAIMS_FILE, first_year=2009, extra_cols=()):
    """Virginia claims (the original Virginia-only loader, kept for older code)."""
    return load_claims("VA", path, first_year, extra_cols)


def total_paid(claims):
    """Building + contents + ICC payments. A blank means nothing was paid, so it counts as $0.

    A few claims (none in Virginia; 2 in NC, 1 in MD) have a NEGATIVE total: a refund or
    correction recorded against the claim. We floor the total at $0, so such a claim counts as
    "not paid" rather than subtracting money from its rating cell (Milestone 13 decision)."""
    return claims[PAID_COLS].fillna(0).sum(axis=1).clip(lower=0)


def load_county_names(path=COUNTY_NAMES_FILE):
    """Dict of 5-digit county FIPS -> 'Name' (e.g. '51810' -> 'Virginia Beach city').

    Returns an empty dict if the Census file hasn't been downloaded
    (run: src\\download_data.py counties)."""
    if not Path(path).exists():
        return {}
    tbl = pd.read_csv(path, sep="|", dtype=str)
    fips = tbl["STATEFP"] + tbl["COUNTYFP"]
    return dict(zip(fips, tbl["COUNTYNAME"]))


# ---------------------------------------------------------------------------
# Earned exposure
# ---------------------------------------------------------------------------
def coverage_period(policies):
    """Start and end of the time each policy term actually provided cover.

    * Start = policyEffectiveDate.
    * End   = policyTerminationDate, or the cancellation date if the policy was
              cancelled earlier.
    * A cancellation dated on or before the start ("cancelled flat") means the policy
      never provided cover, so end = start and it earns zero exposure.
    """
    start = policies["policyEffectiveDate"]
    end = policies["policyTerminationDate"]
    cancel = policies["cancellationDateOfFloodPolicy"]
    end = end.where(cancel.isna() | (cancel >= end), cancel)   # cancelled early -> stop there
    end = end.where(end > start, start)                        # never negative
    return start, end


def earned_exposure(policies, years, by=None, weight=None):
    """Earned policy-years in each calendar year (optionally per group, e.g. county).

    For each policy and each year Y:
        earned = (days of cover that fall inside year Y) / (days in year Y)
    so a policy covering all of Y earns exactly 1.0, and one covering Jul 1 to Dec 31
    earns about 0.5. Then we add up across policies.

    weight: optional column name to multiply by (e.g. "policyCount" to count every
            unit in a condo building separately). Default: each policy term counts once.

    Returns a DataFrame with columns [*by, year, exposure].
    """
    start, end = coverage_period(policies)
    w = policies[weight].to_numpy(float) if weight else np.ones(len(policies))
    keys = [policies[b] for b in (by or [])]
    pieces = []
    for y in years:
        y0 = pd.Timestamp(year=y, month=1, day=1, tz=start.dt.tz)
        y1 = pd.Timestamp(year=y + 1, month=1, day=1, tz=start.dt.tz)
        # Overlap of [start, end) with [y0, y1). Clip at 0 when there's no overlap.
        days = (end.clip(upper=y1) - start.clip(lower=y0)).dt.days.clip(lower=0)
        earned = days.to_numpy(float) / (y1 - y0).days * w
        if by:
            part = pd.Series(earned, index=policies.index).groupby(keys, dropna=False).sum()
            part = part.rename("exposure").reset_index()
            part.columns = [*by, "exposure"]
        else:
            part = pd.DataFrame({"exposure": [earned.sum()]})
        part.insert(len(by or []), "year", y)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def estimate_2009_exposure(policies, by=None, weight=None):
    """Approximate 2009 exposure, which the data under-counts.

    The policy file only has terms that START in 2009 or later, so policies that
    started in 2008 and were still running in 2009 are missing. Assumption: the
    policies that started in 2008 looked like the ones that started in 2009 (the
    policy base changes only slowly). Under that "steady state" assumption, 2009
    exposure = all the exposure earned by terms that started in 2009, wherever it
    fell in time (roughly one policy-year per annual term)."""
    start, end = coverage_period(policies)
    in_2009 = start.dt.year == 2009
    days = (end - start).dt.days[in_2009]
    w = policies.loc[in_2009, weight].to_numpy(float) if weight else 1.0
    earned = pd.Series(days.to_numpy(float) / 365.0 * w, index=days.index)
    if by:
        out = earned.groupby([policies.loc[in_2009, b] for b in by], dropna=False).sum()
        out = out.rename("exposure").reset_index()
        out.columns = [*by, "exposure"]
    else:
        out = pd.DataFrame({"exposure": [earned.sum()]})
    out.insert(len(by or []), "year", 2009)
    return out


# ---------------------------------------------------------------------------
# Frequency, severity, pure premium
# ---------------------------------------------------------------------------
def claim_totals(claims, by=None):
    """Claim counts and dollars per year (and group). Columns: claims_all, claims, losses."""
    keys = [*(by or []), "yearOfLoss"]
    out = (claims.groupby(keys, dropna=False)
                 .agg(claims_all=("id", "size"), claims=("is_paid", "sum"),
                      losses=("total_paid", "sum"))
                 .reset_index()
                 .rename(columns={"yearOfLoss": "year"}))
    return out


def add_measures(df):
    """Add frequency, severity, and pure premium columns to a table that has
    exposure, claims (paid claims), claims_all, and losses."""
    df = df.copy()
    df["frequency"] = df["claims"] / df["exposure"]               # paid claims per policy-year
    df["frequency_all"] = df["claims_all"] / df["exposure"]       # including $0 claims
    df["severity"] = df["losses"] / df["claims"].where(df["claims"] > 0)   # $ per paid claim
    df["pure_premium"] = df["losses"] / df["exposure"]            # $ per policy-year
    return df


def summarize(exposure, claims_tbl, by=None):
    """Join exposure to claim totals (on group + year) and compute the measures.

    Rows with exposure but no claims get 0 claims and $0 losses (a quiet year is real
    information, not missing data)."""
    keys = [*(by or []), "year"]
    df = exposure.merge(claims_tbl, on=keys, how="left")
    df[["claims_all", "claims", "losses"]] = df[["claims_all", "claims", "losses"]].fillna(0)
    return add_measures(df)


def collapse(df, by=None):
    """Add up a summarize() table over years (e.g. to get one row per county)."""
    sums = ["exposure", "claims_all", "claims", "losses"]
    out = df.groupby(by)[sums].sum().reset_index() if by else df[sums].sum().to_frame().T
    return add_measures(out)


# ---------------------------------------------------------------------------
# Self-test + build the processed county-year table
# ---------------------------------------------------------------------------
def _self_test():
    """Tiny made-up policies with known answers."""
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    toy = pd.DataFrame({
        "policyEffectiveDate":   [ts("2015-07-02"), ts("2015-01-01"), ts("2015-01-01"), ts("2015-03-01")],
        "policyTerminationDate": [ts("2016-07-02"), ts("2016-01-01"), ts("2016-01-01"), ts("2016-03-01")],
        # policy 3 cancelled halfway through 2015; policy 4 cancelled before it started
        "cancellationDateOfFloodPolicy": [pd.NaT, pd.NaT, ts("2015-07-02"), ts("2015-02-01")],
        "policyCount": [1, 1, 1, 1],
    })
    e = earned_exposure(toy, [2015, 2016]).set_index("year")["exposure"]
    # policy 1: 183/365 in 2015 + 183/366 in 2016; policy 2: 1.0; policy 3: 182/365; policy 4: 0
    exp15 = 183 / 365 + 1 + 182 / 365
    exp16 = 183 / 366
    assert abs(e[2015] - exp15) < 1e-9 and abs(e[2016] - exp16) < 1e-9, e
    return True


def build_county_year(save=True):
    """County x year table (2009-2025) of exposure, claims, losses and the measures.
    2009 uses the estimate from estimate_2009_exposure(). Saved to data/processed/."""
    pol = load_va_policies()
    clm = load_va_claims()
    exp = earned_exposure(pol, range(FIRST_FULL_YEAR, LAST_FULL_YEAR + 1), by=["county"])
    exp = pd.concat([estimate_2009_exposure(pol, by=["county"]), exp], ignore_index=True)
    exp["exposure_is_estimate"] = exp["year"] == 2009
    tbl = summarize(exp, claim_totals(clm, by=["county"]), by=["county"])
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tbl.to_parquet(PROCESSED / "va_county_year.parquet", index=False)
    return tbl


if __name__ == "__main__":
    print("self-test passed:", _self_test())
    t = build_county_year()
    print(f"county-year table: {len(t):,} rows -> data/processed/va_county_year.parquet")
    print(collapse(t, ["year"]).round(4).to_string())
