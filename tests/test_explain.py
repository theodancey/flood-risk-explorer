"""
Quick tests for src/explain.py (Milestone 12; three states since Milestone 13). Plain Python,
no pytest needed:

    .venv\\Scripts\\python.exe tests\\test_explain.py

They use ONLY committed files (app/data/*, models/production_model.json), exactly what the
Streamlit app will have on a host without data/processed/.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import explain  # noqa: E402

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("  FAIL:", msg)


def waterfall_ok(res):
    wf = res["waterfall"]
    prod = wf["after"].iloc[0] * wf["multiplier"].iloc[1:-1].prod()
    return math.isclose(prod, res["pure_premium"], rel_tol=1e-9) and \
        math.isclose(wf["after"].iloc[-1], res["pure_premium"], rel_tol=1e-12)


def test_every_county():
    """explain_county + a default explain_profile for EVERY county in all three states."""
    ex = explain.load()
    cs = ex.county_summary.set_index("county")
    for fips in cs.index:
        e = ex.explain_county(fips)
        st = cs.loc[fips, "state"]
        check(e["state"] == st == explain.state_of(fips), f"state {fips}")
        check(waterfall_ok(e), f"county waterfall {fips}")
        check(math.isclose(e["pure_premium"], cs.loc[fips, "expected_pure_premium"], rel_tol=1e-5),
              f"county summary vs live {fips}")
        check(math.isclose(e["ratio_to_state"], e["mix_effect"] * e["county_factor_effect"], rel_tol=1e-9),
              f"mix x factor {fips}")
        check(math.isclose(e["state_average_pure_premium"],
                           ex.averages["states"][st]["expected_pure_premium_per_policy_year"], rel_tol=1e-9),
              f"state average {fips}")
        check(e["certainty"] in ("High", "Medium", "Low"), f"certainty {fips}")
        check("not a prediction for any individual property" in e["sentence"], f"disclaimer in sentence {fips}")
        check(explain.STATE_NAMES[st] in e["sentence"], f"state named in sentence {fips}")
        # default profile in every county
        p = ex.explain_profile({"county": fips})
        check(waterfall_ok(p), f"profile waterfall {fips}")
        pred = ex.model.predict(pd.DataFrame([{**{f: p["profile"][f] for f in explain.FEATURES}, "county": fips}]))
        check(math.isclose(p["pure_premium"], pred["pure_premium"].iloc[0], rel_tol=1e-9), f"profile = predict {fips}")
    by_state = cs["state"].value_counts().to_dict()
    check(set(by_state) == {"VA", "NC", "MD"}, f"all three states present: {by_state}")
    print(f"  explain_county + default explain_profile ran for all {len(cs)} counties {by_state}")


def test_state_averages():
    """Each state's average = the exposure-weighted mean of its counties (no county = the state)."""
    ex = explain.load()
    cs = ex.county_summary
    for st in ("VA", "NC", "MD"):
        c = cs[cs["state"] == st]
        avg = float((c["expected_pure_premium"] * c["policy_years_2010_2025"]).sum() / c["policy_years_2010_2025"].sum())
        ref = ex.averages["states"][st]["expected_pure_premium_per_policy_year"]
        check(math.isclose(avg, ref, rel_tol=1e-6), f"{st} average {avg} vs {ref}")
        r = ex.explain_profile({}, state=st)            # any county, base levels
        check(r["state"] == st and r["county_name"].startswith(explain.STATE_NAMES[st]), f"profile anywhere in {st}")
        check(waterfall_ok(r), f"anywhere-in-{st} waterfall")
    print("  state averages = exposure-weighted county averages; 'anywhere in the state' profiles OK")


def test_random_profiles(n=90, seed=12):
    ex = explain.load()
    rng = np.random.default_rng(seed)
    counties = list(ex.county_summary["county"]) + [None, None, None]
    for _ in range(n):
        prof = {f: rng.choice(ex.t.levels[f]) for f in explain.FEATURES}
        prof["county"] = counties[rng.integers(len(counties))]
        prof["post_firm"] = rng.choice(["True", "False"])
        state = None if prof["county"] else rng.choice(["VA", "NC", "MD"])
        r = ex.explain_profile(prof, state=state)
        check(waterfall_ok(r), f"random profile waterfall {prof}")
        check(r["pure_premium"] > 0 and r["frequency"] > 0 and r["severity"] > 0, f"positive {prof}")
    print(f"  {n} random profiles OK (including 'anywhere in the state' for each state)")


def test_inputs():
    ex = explain.load()
    check(ex.resolve_county("Norfolk") == "51710", "name lookup")
    check(ex.resolve_county("norfolk city") == "51710", "case-insensitive lookup")
    check(ex.resolve_county("Dare County, NC") == "37055", "'Name, ST' lookup")
    check(ex.resolve_county("Montgomery County", state="MD") == "24031", "state= picks the state")
    check(ex.resolve_county("Montgomery", state="NC") == "37123", "short name within a state")
    for bad, st in (("Richmond", None), ("Atlantis", None), ("Montgomery County", None), ("51710", "NC"),
                    ("Dare County", "VA")):
        try:
            ex.resolve_county(bad, state=st)
            check(False, f"{bad} ({st}) should raise")
        except ValueError:
            pass
    for bad in ({"county": "51710", "flood_zone_group": "Z"}, {}):     # bad level; no county and no state
        try:
            ex.explain_profile(bad)
            check(False, f"{bad} should raise")
        except ValueError:
            pass
    r = ex.explain_profile({"county": "51710", "elevated": True})       # booleans accepted
    check(r["profile"]["elevated"] == "True", "boolean input")
    print("  input validation OK")


def test_rating_table_parity():
    ex = explain.load()
    rng = np.random.default_rng(3)
    rows = pd.DataFrame({f: rng.choice(ex.t.levels[f], 600) for f in explain.FEATURES})
    rows["county"] = rng.choice(list(ex.county_summary["county"]), 600)
    a, b = ex.t.rate(rows), ex.model.predict(rows)
    for col in ("frequency", "severity", "county_factor", "pure_premium"):
        check(np.allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=1e-9), f"rating parity {col}")
    print("  rating_factors.json = production.predict on 600 random rows across the three states")


if __name__ == "__main__":
    for t in (test_every_county, test_state_averages, test_random_profiles, test_inputs, test_rating_table_parity):
        print(t.__name__)
        t()
    print("ALL PASSED" if not FAILURES else f"{len(FAILURES)} FAILURES")
    sys.exit(1 if FAILURES else 0)
