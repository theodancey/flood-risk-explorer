"""
Quick tests for src/explain.py (Milestone 12). Plain Python, no pytest needed:

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
    ex = explain.load()
    cs = ex.county_summary.set_index("county")
    for fips in cs.index:
        e = ex.explain_county(fips)
        check(waterfall_ok(e), f"county waterfall {fips}")
        check(math.isclose(e["pure_premium"], cs.loc[fips, "expected_pure_premium"], rel_tol=1e-5),
              f"county summary vs live {fips}")
        check(math.isclose(e["ratio_to_virginia"], e["mix_effect"] * e["county_factor_effect"], rel_tol=1e-9),
              f"mix x factor {fips}")
        check(e["certainty"] in ("High", "Medium", "Low"), f"certainty {fips}")
        check("not a prediction for any individual property" in e["sentence"], f"disclaimer in sentence {fips}")
        # default profile in every county
        p = ex.explain_profile({"county": fips})
        check(waterfall_ok(p), f"profile waterfall {fips}")
        pred = ex.model.predict(pd.DataFrame([{**{f: p["profile"][f] for f in explain.FEATURES}, "county": fips}]))
        check(math.isclose(p["pure_premium"], pred["pure_premium"].iloc[0], rel_tol=1e-9), f"profile = predict {fips}")
    print(f"  explain_county + default explain_profile ran for all {len(cs)} counties")


def test_random_profiles(n=60, seed=12):
    ex = explain.load()
    rng = np.random.default_rng(seed)
    counties = list(ex.county_summary["county"]) + [None]
    for _ in range(n):
        prof = {f: rng.choice(ex.t.levels[f]) for f in explain.FEATURES}
        prof["county"] = counties[rng.integers(len(counties))]
        prof["post_firm"] = rng.choice(["True", "False"])
        r = ex.explain_profile(prof)
        check(waterfall_ok(r), f"random profile waterfall {prof}")
        check(r["pure_premium"] > 0 and r["frequency"] > 0 and r["severity"] > 0, f"positive {prof}")
    print(f"  {n} random profiles OK (including 'anywhere in Virginia')")


def test_inputs():
    ex = explain.load()
    check(ex.resolve_county("Norfolk") == "51710", "name lookup")
    check(ex.resolve_county("norfolk city") == "51710", "case-insensitive lookup")
    for bad in ("Richmond", "Atlantis"):
        try:
            ex.resolve_county(bad)
            check(False, f"{bad} should raise")
        except ValueError:
            pass
    try:
        ex.explain_profile({"county": "51710", "flood_zone_group": "Z"})
        check(False, "bad level should raise")
    except ValueError:
        pass
    r = ex.explain_profile({"county": "51710", "elevated": True})       # booleans accepted
    check(r["profile"]["elevated"] == "True", "boolean input")
    print("  input validation OK")


def test_rating_table_parity():
    ex = explain.load()
    rng = np.random.default_rng(3)
    rows = pd.DataFrame({f: rng.choice(ex.t.levels[f], 500) for f in explain.FEATURES})
    rows["county"] = rng.choice(list(ex.county_summary["county"]), 500)
    a, b = ex.t.rate(rows), ex.model.predict(rows)
    for col in ("frequency", "severity", "county_factor", "pure_premium"):
        check(np.allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=1e-9), f"rating parity {col}")
    print("  rating_factors.json = production.predict on 500 random rows")


if __name__ == "__main__":
    for t in (test_every_county, test_random_profiles, test_inputs, test_rating_table_parity):
        print(t.__name__)
        t()
    print("ALL PASSED" if not FAILURES else f"{len(FAILURES)} FAILURES")
    sys.exit(1 if FAILURES else 0)
