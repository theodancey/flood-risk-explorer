"""
Build the modelling dataset: one row per RATING CELL, Virginia, 2010-2025.

A rating cell is a group of policies that look the same on every rating feature
(same county, same flood-zone group, same occupancy group, ...) in the same calendar
year. For each cell we add up:
    exposure        earned policy-years (how much insurance was in force)
    claims          paid claims (total paid > $0)        <- frequency target
    claims_all      all claims, including $0 ones
    losses_nominal  total paid, in the dollars of the year of loss
    losses_2025     the same, converted to 2025 dollars with CPI-U   <- severity target

Every rule used here is explained in docs/modeling_dataset.md.

Run from the project folder:
    .venv\\Scripts\\python.exe src\\build_dataset.py

Output: data/processed/va_model_data.parquet (+ va_model_data_checks.json).
Every run also runs the data checks in run_checks(). If any check fails the script stops
with an AssertionError and does NOT save the file.

Needs these downloads (src\\download_data.py): claims, policies-va, counties, cpi, coastal.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import actuarial as A   # noqa: E402  (our own module, in the same folder)

OUT_FILE = A.PROCESSED / "va_model_data.parquet"
CHECKS_FILE = A.PROCESSED / "va_model_data_checks.json"
CPI_FILE = A.RAW / "cpi_u_bls.csv"
COASTLINE_FILE = A.RAW / "census_coastline_counties.xlsx"

YEARS = range(A.FIRST_FULL_YEAR, A.LAST_FULL_YEAR + 1)    # 2010..2025
BASE_YEAR = 2025                                         # dollars are restated to 2025

# Totals from notebook 02 / reports/02_actuarial_basics.md for 2010-2025 (nominal dollars).
# The new table must add back up to exactly these numbers.
REFERENCE_TOTALS = {
    "exposure": 1_482_412.7676,
    "claims_all": 12_545,
    "claims": 9_557,
    "losses_nominal": 220_183_315.29,
}

# Rare-category rule: every level of every feature must hold at least this share of the
# total 2010-2025 exposure. Smaller levels are merged into a named, similar level (below).
MIN_LEVEL_SHARE = 0.01

# Counties with at least this many policy-years (2010-2025) keep their own county_group level.
# The same credibility floor that notebook 02 used for the county ranking.
COUNTY_CREDIBILITY_FLOOR = 5_000
UNKNOWN_COUNTY = "UNKNOWN"

# ---------------------------------------------------------------------------
# Features: the same code is applied to policies AND claims, so a claim lands in the
# cell of the policies that look like it.
# ---------------------------------------------------------------------------
# Raw FEMA columns we need. The floors column has a slightly different name in each file.
FEATURE_SOURCE_COLS = ["ratedFloodZone", "occupancyType", "primaryResidenceIndicator",
                       "elevatedBuildingIndicator", "postFIRMConstructionIndicator",
                       "originalConstructionDate", "totalBuildingInsuranceCoverage"]
POLICY_FLOORS_COL = "numberOfFloorsInInsuredBuilding"
CLAIM_FLOORS_COL = "numberOfFloorsInTheInsuredBuilding"

CELL_FEATURES = ["flood_zone_group", "occupancy_group", "floors_group", "elevated",
                 "primary_residence", "post_firm", "construction_band", "building_coverage_band"]
CELL_KEYS = ["year", "county", *CELL_FEATURES]        # one row per unique combination

# Occupancy codes (FEMA). 1-digit codes are pre-Risk Rating 2.0; 2-digit codes are the
# RR2.0 (Oct 2021+) versions. Before RR2.0, condo units (now 16) and mobile homes (now 14)
# were coded 1 = single family, so they go in the same group to stay consistent over time.
OCCUPANCY_GROUPS = {
    "single-unit residential": [1, 11, 14, 16],
    "multi-unit residential": [2, 3, 12, 13, 15],
    "non-residential": [4, 6, 17, 18, 19],
}
# Floors codes: 1, 2, 3 = 3+, 4 = split-level, 5 = mobile home, 6 = townhouse 3+ floors.
# Split-level (0.6% of exposure) and mobile home (0.5%) are too rare for their own level.
FLOORS_GROUPS = {"1": [1, 5], "2": [2, 4], "3+": [3, 6]}
FLOORS_FALLBACK = "2"                        # most common level, for the few blank codes

CONSTRUCTION_BINS = [1700, 1960, 1980, 2000, 2100]      # [1700,1960) [1960,1980) ...
CONSTRUCTION_LABELS = ["before 1960", "1960-1979", "1980-1999", "2000 or later"]
CONSTRUCTION_FALLBACK = "1980-1999"          # most common band, for blank / implausible years

# Building coverage limit, in NOMINAL dollars on purpose: the NFIP's $250,000 residential
# limit has not changed since 1994, so "at the limit" means the same thing every year.
COVERAGE_BINS = [-np.inf, 0, 100_000, 250_000, np.inf]   # right edge excluded, see below
COVERAGE_LABELS = ["$0 (contents only)", "under $100K", "$100K-$249K", "$250K or more"]

# Broader "tidal" coastal definition: the 46 localities of "Tidewater Virginia" as defined in
# the Code of Virginia, section 62.1-44.15:68 (Chesapeake Bay Preservation Act, definitions):
# https://law.lis.virginia.gov/vacode/title62.1/chapter3.1/section62.1-44.15:68/
# (checked 2026-11-11). These are the localities with tidal waters draining to the Chesapeake
# Bay or the ocean. The same list defines Virginia's Coastal Zone Management Program area.
# It contains all 25 Census coastline counties, plus 21 more (e.g. Chesapeake, Alexandria,
# Richmond city, Henrico). 29 counties + 17 independent cities; FIPS codes from the Census
# county-names file (names checked in the build).
TIDEWATER_LOCALITIES = {
    "51001": "Accomack County", "51013": "Arlington County", "51033": "Caroline County",
    "51036": "Charles City County", "51041": "Chesterfield County", "51057": "Essex County",
    "51059": "Fairfax County", "51073": "Gloucester County", "51085": "Hanover County",
    "51087": "Henrico County", "51093": "Isle of Wight County", "51095": "James City County",
    "51097": "King and Queen County", "51099": "King George County",
    "51101": "King William County", "51103": "Lancaster County", "51115": "Mathews County",
    "51119": "Middlesex County", "51127": "New Kent County", "51131": "Northampton County",
    "51133": "Northumberland County", "51149": "Prince George County",
    "51153": "Prince William County", "51159": "Richmond County",
    "51177": "Spotsylvania County", "51179": "Stafford County", "51181": "Surry County",
    "51193": "Westmoreland County", "51199": "York County",
    # independent cities
    "51510": "Alexandria city", "51550": "Chesapeake city", "51570": "Colonial Heights city",
    "51600": "Fairfax city", "51610": "Falls Church city", "51630": "Fredericksburg city",
    "51650": "Hampton city", "51670": "Hopewell city", "51700": "Newport News city",
    "51710": "Norfolk city", "51730": "Petersburg city", "51735": "Poquoson city",
    "51740": "Portsmouth city", "51760": "Richmond city", "51800": "Suffolk city",
    "51810": "Virginia Beach city", "51830": "Williamsburg city",
}


def _group_codes(codes, groups, fallback):
    """Map numeric codes to group names using a {group: [codes]} dict.
    Codes not in any group (incl. blanks) get `fallback`. Returns (labels, n_fallback)."""
    lookup = {code: name for name, code_list in groups.items() for code in code_list}
    labels = codes.map(lookup)
    n_fallback = int(labels.isna().sum())
    return labels.fillna(fallback), n_fallback


def make_features(df, floors_col):
    """Turn raw FEMA columns into the 8 rating-cell features.

    Works on policies or claims (pass the right floors column name).
    Returns (features DataFrame, dict of how many rows needed a fallback value)."""
    out = pd.DataFrame(index=df.index)
    fallbacks = {}

    # 1. Flood zone: zones starting with A or V are the high-risk Special Flood Hazard Area
    #    (SFHA). B, C, X are moderate/low risk. D (undetermined) and blanks are a handful of
    #    rows; they are not in the SFHA, so they join the moderate/low group.
    #    V zones alone are only 0.7% of exposure, below the 1% rule, so they join A.
    zone = df["ratedFloodZone"].fillna("").str.strip().str.upper()
    high = zone.str.startswith(("A", "V"))
    out["flood_zone_group"] = np.where(high, "A/V high risk", "B/C/X moderate-low risk")
    fallbacks["flood_zone_not_A_V_B_C_X"] = int((~high & ~zone.str.startswith(("B", "C", "X"))).sum())

    # 2. Occupancy and 3. floors
    out["occupancy_group"], fallbacks["occupancy_unmapped"] = _group_codes(
        df["occupancyType"], OCCUPANCY_GROUPS, "single-unit residential")
    out["floors_group"], fallbacks["floors_unmapped"] = _group_codes(
        df[floors_col], FLOORS_GROUPS, FLOORS_FALLBACK)

    # 4-6. Yes/no building facts, recorded on the policy. Blanks (none seen so far) -> False.
    for new, raw in [("elevated", "elevatedBuildingIndicator"),
                     ("primary_residence", "primaryResidenceIndicator"),
                     ("post_firm", "postFIRMConstructionIndicator")]:
        fallbacks[f"{new}_blank"] = int(df[raw].isna().sum())
        out[new] = df[raw].fillna(False).astype(bool)

    # 7. Construction year band (the building's vintage, which doesn't change over time).
    year_built = df["originalConstructionDate"].dt.year
    band = pd.cut(year_built, CONSTRUCTION_BINS, labels=CONSTRUCTION_LABELS, right=False)
    fallbacks["construction_blank_or_implausible"] = int(band.isna().sum())
    out["construction_band"] = band.astype(object).fillna(CONSTRUCTION_FALLBACK).astype(str)

    # 8. Building coverage band: exactly $0, under $100K, $100K up to $249,999, $250K+.
    cov = df["totalBuildingInsuranceCoverage"]
    band = pd.Series(pd.NA, index=df.index, dtype=object)
    band[cov <= 0] = COVERAGE_LABELS[0]
    band[(cov > 0) & (cov < 100_000)] = COVERAGE_LABELS[1]
    band[(cov >= 100_000) & (cov < 250_000)] = COVERAGE_LABELS[2]
    band[cov >= 250_000] = COVERAGE_LABELS[3]
    fallbacks["coverage_blank"] = int(band.isna().sum())
    out["building_coverage_band"] = band.fillna(COVERAGE_LABELS[0]).astype(str)
    return out, fallbacks


# ---------------------------------------------------------------------------
# Reference tables: CPI-U and coastline counties
# ---------------------------------------------------------------------------
def load_cpi_factors(path=CPI_FILE, base_year=BASE_YEAR):
    """Annual-average CPI-U and the factor that turns year-Y dollars into base-year dollars.

    factor(Y) = CPI(base year) / CPI(Y). Example: if prices rose 20% since Y, factor = 1.2.
    A missing month (BLS could not collect October 2025 during the government shutdown) is
    filled by a straight line between the months on either side, then each year is averaged.
    Returns (DataFrame [year, cpi_annual_avg, months_published, cpi_factor], notes)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing. Run: src\\download_data.py cpi")
    cpi = pd.read_csv(path).sort_values(["year", "month"]).reset_index(drop=True)
    missing = cpi[cpi["value"].isna()]
    notes = [f"{int(r.year)}-{int(r.month):02d} not published by BLS ({r.footnote}); "
             f"filled by linear interpolation" for r in missing.itertuples()]
    cpi["value_filled"] = cpi["value"].interpolate(limit_area="inside")
    annual = (cpi.groupby("year")
                 .agg(cpi_annual_avg=("value_filled", "mean"),
                      months_published=("value", "count"),
                      months=("month", "size"))
                 .reset_index())
    assert (annual["months"] == 12).all(), "every CPI year needs 12 months (even if one is filled)"
    base = annual.loc[annual["year"] == base_year, "cpi_annual_avg"].item()
    annual["cpi_factor"] = base / annual["cpi_annual_avg"]
    return annual.drop(columns="months"), notes


def load_coastline_counties(path=COASTLINE_FILE, state_fips="51"):
    """Set of 5-digit county FIPS codes on the Census Bureau's coastline-county list
    (2016 list, released Aug 2018). Virginia has 25 (counties and independent cities)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing. Run: src\\download_data.py coastal")
    tbl = pd.read_excel(path, header=3, dtype=str)
    fips = tbl.iloc[:, 0].dropna().str.strip()
    fips = fips[fips.str.fullmatch(r"\d{5}")]          # drops the footnote rows at the bottom
    return set(fips[fips.str.startswith(state_fips)])


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def load_inputs():
    """Policies and 2010-2025 Virginia claims, each with the 8 features + county."""
    pol = A.load_va_policies(extra_cols=[*FEATURE_SOURCE_COLS, POLICY_FLOORS_COL])
    clm = A.load_va_claims(first_year=A.FIRST_FULL_YEAR,
                           extra_cols=[*FEATURE_SOURCE_COLS, CLAIM_FLOORS_COL])
    clm = clm[clm["yearOfLoss"] <= A.LAST_FULL_YEAR].copy()

    pol_feat, pol_fb = make_features(pol, POLICY_FLOORS_COL)
    clm_feat, clm_fb = make_features(clm, CLAIM_FLOORS_COL)
    pol = pd.concat([pol[["id", "policyEffectiveDate", "policyTerminationDate",
                          "cancellationDateOfFloodPolicy", "policyCount", "county"]], pol_feat], axis=1)
    clm = pd.concat([clm[["id", "yearOfLoss", "county", "total_paid", "is_paid"]], clm_feat], axis=1)
    # A blank county is kept as its own "UNKNOWN" level on both sides, so no exposure or
    # claim is dropped (0.45% of exposure, a few claims).
    for d in (pol, clm):
        d["county"] = d["county"].fillna(UNKNOWN_COUNTY)
    return pol, clm, {"policies": pol_fb, "claims": clm_fb}


def explain_unmatched(unmatched, exposure):
    """Why does a claim's cell have no exposure? For each unmatched claim cell we test, one
    feature at a time, whether ignoring that feature would let it match some exposure.

    Returns two tables:
      reasons:     each unmatched cell in exactly one category (claim counts add up)
      by_feature:  for each feature, how many unmatched claims would match if that one
                   feature were ignored (a claim can count under several features)"""
    candidates = ["county", *CELL_FEATURES]
    keyed = {drop: set(map(tuple, exposure[[k for k in CELL_KEYS if k != drop]]
                                  .drop_duplicates().to_numpy()))
             for drop in candidates}
    county_years = set(map(tuple, exposure[["county", "year"]].drop_duplicates().to_numpy()))
    rows = []
    for _, r in unmatched.iterrows():
        fixes = [d for d in candidates if tuple(r[[k for k in CELL_KEYS if k != d]]) in keyed[d]]
        if (r["county"], r["year"]) not in county_years:
            reason = "a. no exposure at all in that county and year"
        elif len(fixes) == 1:
            reason = "b. differs from all in-force policies on exactly one feature"
        elif fixes:
            reason = "c. would match if any one of several features were ignored"
        else:
            reason = "d. differs on two or more features (rare combination)"
        rows.append({"reason": reason, "fixes": fixes, "claims_all": r["claims_all"],
                     "claims": r["claims"], "losses_nominal": r["losses_nominal"]})
    out = pd.DataFrame(rows, columns=["reason", "fixes", "claims_all", "claims", "losses_nominal"])
    reasons = (out.groupby("reason")
                  .agg(cells=("claims", "size"), claims_all=("claims_all", "sum"),
                       claims=("claims", "sum"), losses_nominal=("losses_nominal", "sum"))
                  .reset_index())
    by_feature = pd.DataFrame(
        [{"feature_ignored": d,
          "claims_all_that_would_match": int(out.loc[out["fixes"].map(lambda f: d in f), "claims_all"].sum()),
          "only_fix_for_claims_all": int(out.loc[out["fixes"].map(lambda f: f == [d]), "claims_all"].sum())}
         for d in candidates]).sort_values("claims_all_that_would_match", ascending=False)
    return reasons, by_feature


def build():
    pol, clm, fallbacks = load_inputs()

    # Exposure per cell. A policy term that spans two calendar years earns part of its
    # exposure in each year's cell (same features, different year).
    exposure = A.earned_exposure(pol, YEARS, by=["county", *CELL_FEATURES])
    exposure = exposure[exposure["exposure"] > 0]           # drop combinations never in force
    # Claims per cell (the same helper as notebook 02), keyed by year of loss.
    claims = A.claim_totals(clm, by=["county", *CELL_FEATURES]).rename(columns={"losses": "losses_nominal"})

    # Outer join: cells with exposure but no claims (most of them) get 0 claims and $0;
    # cells with claims but no exposure ("unmatched") are kept with exposure 0.
    cells = exposure.merge(claims, on=CELL_KEYS, how="outer", indicator=True)
    cells["exposure_matched"] = cells["_merge"] != "right_only"
    cells = cells.drop(columns="_merge")
    cells[["exposure", "claims_all", "claims", "losses_nominal"]] = (
        cells[["exposure", "claims_all", "claims", "losses_nominal"]].fillna(0))
    cells[["claims_all", "claims"]] = cells[["claims_all", "claims"]].astype(int)

    # Inflation: restate each year's losses in 2025 dollars.
    cpi, cpi_notes = load_cpi_factors()
    cells = cells.merge(cpi[["year", "cpi_factor"]], on="year", how="left")
    cells["losses_2025"] = cells["losses_nominal"] * cells["cpi_factor"]

    # County-level columns: name, coastal flag, and a pooled county_group for modelling.
    names = A.load_county_names()
    coastal = load_coastline_counties()
    cells["county_name"] = cells["county"].map(names).fillna("Unknown county")
    cells["coastal"] = cells["county"].isin(coastal)                       # Census coastline list
    cells["coastal_tidal"] = cells["county"].isin(TIDEWATER_LOCALITIES)    # Tidewater Virginia
    county_exposure = cells.groupby("county")["exposure"].sum()
    credible = set(county_exposure[county_exposure >= COUNTY_CREDIBILITY_FLOOR].index) - {UNKNOWN_COUNTY}
    # A few Virginia policies (52 policy-years, no claims) have a census code from another
    # state, which is almost certainly a geocoding error. They're pooled with the blanks.
    unknown = (cells["county"] == UNKNOWN_COUNTY) | ~cells["county"].str.startswith("51")
    cells["county_group"] = np.select(
        [cells["county"].isin(credible), unknown, cells["coastal_tidal"]],
        [cells["county"], "Other/unknown county", "Other tidal (small)"],
        "Other non-tidal (small)")
    # Small counties are pooled by the Tidewater flag, not the Census one. Notebook 04 found
    # that the old "Other inland (small)" pool mixed tidal localities (5.5 paid claims per
    # 1,000 policy-years) with non-tidal Piedmont/mountain ones (9.3 per 1,000).

    cols = ["year", "county", "county_name", "county_group", "coastal", "coastal_tidal", *CELL_FEATURES,
            "exposure", "claims", "claims_all", "losses_nominal", "cpi_factor", "losses_2025",
            "exposure_matched"]
    cells = cells[cols].sort_values(CELL_KEYS).reset_index(drop=True)

    reasons, by_feature = explain_unmatched(cells[~cells["exposure_matched"]], exposure)
    info = {"fallbacks": fallbacks, "cpi": cpi, "cpi_notes": cpi_notes,
            "unmatched_reasons": reasons, "unmatched_by_feature": by_feature, "n_claims_input": len(clm),
            "n_paid_input": int(clm["is_paid"].sum()), "losses_input": float(clm["total_paid"].sum()),
            "exposure_input": float(A.earned_exposure(pol, YEARS)["exposure"].sum()),
            "n_coastal_counties": len(coastal), "n_credible_counties": len(credible),
            "coastal_set": coastal, "county_names": names}
    return cells, info


# ---------------------------------------------------------------------------
# Data checks (run every time)
# ---------------------------------------------------------------------------
def run_checks(cells, info):
    """Assert the table is complete and consistent. Returns a dict summary for the report."""
    tot = cells[["exposure", "claims_all", "claims", "losses_nominal"]].sum()

    # 1. Totals add back up to the raw inputs AND to notebook 02's published totals.
    assert abs(tot["exposure"] - info["exposure_input"]) < 1e-6, "exposure lost in grouping"
    assert tot["claims_all"] == info["n_claims_input"], "claims lost in grouping"
    assert tot["claims"] == info["n_paid_input"], "paid claims lost in grouping"
    assert abs(tot["losses_nominal"] - info["losses_input"]) < 0.01, "losses lost in grouping"
    assert abs(tot["exposure"] - REFERENCE_TOTALS["exposure"]) < 0.01, tot["exposure"]
    assert tot["claims_all"] == REFERENCE_TOTALS["claims_all"], tot["claims_all"]
    assert tot["claims"] == REFERENCE_TOTALS["claims"], tot["claims"]
    assert abs(tot["losses_nominal"] - REFERENCE_TOTALS["losses_nominal"]) < 1, tot["losses_nominal"]
    ref = A.PROCESSED / "va_county_year.parquet"           # notebook 02's saved table, if present
    if ref.exists():
        nb2 = pd.read_parquet(ref).query("@A.FIRST_FULL_YEAR <= year <= @A.LAST_FULL_YEAR")
        assert abs(nb2["exposure"].sum() - tot["exposure"]) < 0.01
        assert nb2["claims"].sum() == tot["claims"] and nb2["claims_all"].sum() == tot["claims_all"]
        assert abs(nb2["losses"].sum() - tot["losses_nominal"]) < 1

    # 2. No negative values, no missing values, one row per cell.
    numeric = ["exposure", "claims", "claims_all", "losses_nominal", "cpi_factor", "losses_2025"]
    assert (cells[numeric] >= 0).all().all(), "negative values found"
    assert not cells.isna().any().any(), cells.columns[cells.isna().any()].tolist()
    assert not cells.duplicated(CELL_KEYS).any(), "duplicate cells"
    assert (cells["claims"] <= cells["claims_all"]).all()
    assert ((cells["claims"] > 0) | (cells["losses_nominal"] == 0)).all(), "losses with no paid claim"
    assert cells["year"].between(A.FIRST_FULL_YEAR, A.LAST_FULL_YEAR).all()
    assert (cells["exposure_matched"] == (cells["exposure"] > 0)).all()

    # 3. Every claim is in exactly one row (claims are summed, so check counts match input);
    #    unmatched claims (no exposure in their cell) are counted and explained.
    unmatched = cells[~cells["exposure_matched"]]

    # 4. CPI: 2025 factor = 1, older years > 1, and the factor shrinks as years get later.
    f = cells.groupby("year")["cpi_factor"].first()
    assert abs(f[BASE_YEAR] - 1) < 1e-12 and (f >= 1).all() and f.is_monotonic_decreasing
    cpi = info["cpi"].set_index("year")["cpi_annual_avg"]
    assert abs(cpi[2010] - 218.056) < 0.001, "2010 CPI-U should match BLS's published 218.056"

    # 5. Rare-category rule: every level of every feature holds >= 1% of exposure.
    level_shares = {}
    for feat in CELL_FEATURES:
        share = cells.groupby(feat)["exposure"].sum() / tot["exposure"]
        level_shares[feat] = share.round(4).to_dict()
        assert (share >= MIN_LEVEL_SHARE).all(), f"rare level in {feat}: {share.to_dict()}"
    assert cells["coastal"].sum() > 0 and info["n_coastal_counties"] == 25

    # 6. Tidewater list: 46 Virginia localities, each name matching the Census names file,
    #    and every Census coastline county is also Tidewater (so "tidal" is strictly broader).
    tide = TIDEWATER_LOCALITIES
    assert len(tide) == 46 and all(f.startswith("51") for f in tide)
    if info["county_names"]:
        wrong = {f: n for f, n in tide.items() if info["county_names"].get(f) != n}
        assert not wrong, f"Tidewater FIPS/name mismatch: {wrong}"
    assert info["coastal_set"] <= set(tide), info["coastal_set"] - set(tide)
    assert (cells["coastal_tidal"] | ~cells["coastal"]).all()   # coastal implies tidal
    added = sorted(set(tide) - info["coastal_set"])
    tidal_exposure = {
        "census_coastal": float(cells.loc[cells["coastal"], "exposure"].sum() / tot["exposure"]),
        "tidewater": float(cells.loc[cells["coastal_tidal"], "exposure"].sum() / tot["exposure"]),
    }

    return {
        "rows (cells)": len(cells),
        "cells with exposure": int(cells["exposure_matched"].sum()),
        "totals": {k: float(v) for k, v in tot.items()},
        "losses_2025_total": float(cells["losses_2025"].sum()),
        "unmatched": {"cells": len(unmatched), "claims_all": int(unmatched["claims_all"].sum()),
                      "claims": int(unmatched["claims"].sum()),
                      "losses_nominal": float(unmatched["losses_nominal"].sum()),
                      "share_of_paid_claims": float(unmatched["claims"].sum() / tot["claims"]),
                      "share_of_losses": float(unmatched["losses_nominal"].sum() / tot["losses_nominal"]),
                      "by_reason": info["unmatched_reasons"].to_dict(orient="records"),
                      "by_feature": info["unmatched_by_feature"].to_dict(orient="records")},
        "fallback_counts": info["fallbacks"],
        "level_exposure_shares": {k: {str(kk): vv for kk, vv in v.items()} for k, v in level_shares.items()},
        "cpi": info["cpi"].round(4).to_dict(orient="records"),
        "cpi_notes": info["cpi_notes"],
        "tidewater": {"localities": len(tide),
                      "added_beyond_census": [tide[f] for f in added],
                      "exposure_share": tidal_exposure},
        "counties": {"coastal_on_census_list": info["n_coastal_counties"],
                     "own_county_group_level": info["n_credible_counties"],
                     "county_group_levels": int(cells["county_group"].nunique())},
    }


def _self_test():
    """Feature rules on made-up rows with known answers."""
    ts = lambda y: pd.Timestamp(year=y, month=6, day=1, tz="UTC") if y else pd.NaT
    toy = pd.DataFrame({
        "ratedFloodZone": ["AE", "VE", "X", "B", "D", None],
        "occupancyType": [1, 16, 13, 18, 14, 99],
        "floors": [1, 5, 4, 3, 6, None],
        "primaryResidenceIndicator": [True, False, True, False, True, True],
        "elevatedBuildingIndicator": [False, True, False, True, False, False],
        "postFIRMConstructionIndicator": [False, True, True, False, True, False],
        "originalConstructionDate": [ts(1955), ts(1960), ts(1999), ts(2000), ts(202), ts(None)],
        "totalBuildingInsuranceCoverage": [0, 99_999, 100_000, 250_000, 249_999, 1_000_000],
    })
    f, fb = make_features(toy, "floors")
    assert f["flood_zone_group"].tolist() == ["A/V high risk"] * 2 + ["B/C/X moderate-low risk"] * 4
    assert f["occupancy_group"].tolist() == ["single-unit residential", "single-unit residential",
                                             "multi-unit residential", "non-residential",
                                             "single-unit residential", "single-unit residential"]
    assert f["floors_group"].tolist() == ["1", "1", "2", "3+", "3+", "2"]
    assert f["construction_band"].tolist() == ["before 1960", "1960-1979", "1980-1999",
                                               "2000 or later", "1980-1999", "1980-1999"]
    assert f["building_coverage_band"].tolist() == COVERAGE_LABELS[:2] + [COVERAGE_LABELS[2],
                                                    COVERAGE_LABELS[3], COVERAGE_LABELS[2], COVERAGE_LABELS[3]]
    assert fb == {"flood_zone_not_A_V_B_C_X": 2, "occupancy_unmapped": 1, "floors_unmapped": 1,
                  "elevated_blank": 0, "primary_residence_blank": 0, "post_firm_blank": 0,
                  "construction_blank_or_implausible": 2, "coverage_blank": 0}, fb
    return True


def main():
    print("self-test passed:", _self_test(), "| actuarial self-test passed:", A._self_test())
    cells, info = build()
    report = run_checks(cells, info)                  # raises if anything is wrong
    A.PROCESSED.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(OUT_FILE, index=False)
    CHECKS_FILE.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    t = report["totals"]
    u = report["unmatched"]
    print("\nAll data checks passed.")
    print(f"Saved {OUT_FILE.relative_to(A.PROJECT_ROOT)}: {len(cells):,} rating cells "
          f"({report['cells with exposure']:,} with exposure)")
    print(f"  exposure        {t['exposure']:>16,.2f} policy-years (notebook 02: {REFERENCE_TOTALS['exposure']:,.2f})")
    print(f"  claims (all)    {t['claims_all']:>16,.0f} (notebook 02: {REFERENCE_TOTALS['claims_all']:,})")
    print(f"  paid claims     {t['claims']:>16,.0f} (notebook 02: {REFERENCE_TOTALS['claims']:,})")
    print(f"  losses nominal ${t['losses_nominal']:>16,.2f} (notebook 02: ${REFERENCE_TOTALS['losses_nominal']:,.2f})")
    print(f"  losses 2025 $  ${report['losses_2025_total']:>16,.2f}")
    print(f"\nUnmatched claims (their cell has no exposure; kept with exposure = 0): "
          f"{u['claims_all']} claims, {u['claims']} paid ({u['share_of_paid_claims']:.2%} of paid claims), "
          f"${u['losses_nominal']:,.0f} ({u['share_of_losses']:.2%} of losses)")
    print(info["unmatched_reasons"].to_string(index=False))
    print("Would match if this one feature were ignored:")
    print(info["unmatched_by_feature"].to_string(index=False))
    print("\nRows that needed a fallback value:", json.dumps(info["fallbacks"], indent=1))
    print("CPI notes:", *info["cpi_notes"], sep="\n  ")
    print("\nExposure share of each feature level:")
    for feat, shares in report["level_exposure_shares"].items():
        print(f"  {feat}: " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
    print(f"\nCounties: {report['counties']}")
    tw = report["tidewater"]
    print(f"Tidewater (tidal) localities: {tw['localities']}; exposure share "
          f"{tw['exposure_share']['tidewater']:.1%} vs Census coastal {tw['exposure_share']['census_coastal']:.1%}")
    print("  added beyond the Census list:", ", ".join(tw["added_beyond_census"]))


if __name__ == "__main__":
    main()
