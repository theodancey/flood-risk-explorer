r"""
Automated tests for the Streamlit app (Milestones 14-15). Plain Python, no pytest needed:

    .venv\Scripts\python.exe tests\test_app.py

Uses Streamlit's built-in testing tool (streamlit.testing.v1.AppTest), which runs the app's
scripts without a browser and lets the test click widgets and read what's on the page.

What it checks:
  1. Every page loads with no error and shows the short disclaimer at the top and the full one
     (plus name, GitHub link, data source, date) in the footer.
  2. County Explorer: EVERY county in all three states (257) renders, and the big number matches
     app/data/county_summary.csv; tricky names (Richmond, Fairfax, Montgomery) resolve correctly.
  3. Profile Calculator: several profiles in each state, "anywhere in the state", edge cases
     (non-residential, unusual year-built / flood-map combinations, rare building types), and the
     shown numbers equal src/explain.py's results.
  4. The app never reads data/raw or data/processed: every file read is logged and checked.
  5. The app is lightweight: it doesn't import statsmodels, scipy, scikit-learn, XGBoost, SHAP or
     matplotlib.
  6. Milestone 15: /county opens the County Explorer; waterfall labels say Yes/No (never True/False)
     and group small effects; the main pages avoid jargon outside "Learn more" expanders.
"""

import builtins
import io
import pathlib
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app.py"
APP_DATA = (ROOT / "app" / "data").resolve()
FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("  FAIL:", msg)


# ---------------------------------------------------------------------------
# File-read logger: installed BEFORE the app runs, so every read is recorded
# ---------------------------------------------------------------------------
READS = []


def _log(path):
    if isinstance(path, (str, pathlib.PurePath)):
        READS.append(str(path))


def _wrap(fn, pos=0, method=False):
    def inner(*args, **kwargs):
        target = args[pos] if len(args) > pos else kwargs.get("path", kwargs.get("file", kwargs.get("filepath_or_buffer")))
        _log(target)
        return fn(*args, **kwargs)
    return inner


builtins.open = _wrap(builtins.open)
io.open = builtins.open
pd.read_csv = _wrap(pd.read_csv)
pd.read_parquet = _wrap(pd.read_parquet)
pd.read_json = _wrap(pd.read_json)
pathlib.Path.open = _wrap(pathlib.Path.open)
pathlib.Path.read_text = _wrap(pathlib.Path.read_text)
pathlib.Path.read_bytes = _wrap(pathlib.Path.read_bytes)

from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
import explain  # noqa: E402

CS = pd.read_csv(APP_DATA / "county_summary.csv", dtype={"county": str})
EX = explain.load(APP_DATA, with_model=False)
explain._CACHE.clear()             # so the app loads its own copy (and its reads get logged)
READS.clear()                      # only reads made by the app itself count from here on


def new_app():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    check(not at.exception, f"entry point raised: {[e.value for e in at.exception]}")
    return at


def no_error(at, where):
    check(not at.exception, f"{where}: exception {[e.value for e in at.exception][:1]}")
    check(not at.error, f"{where}: error message {[e.value for e in at.error][:1]}")


def has_disclaimer(at, where):
    """Short disclaimer at the top (an info box) + full disclaimer and credits in the footer (captions)."""
    top = " ".join(i.value for i in at.info).lower()
    for phrase in ("area-level", "not a prediction", "quote", "specific property"):
        check(phrase in top, f"{where}: short disclaimer phrase missing: '{phrase}'")
    foot = " ".join(c.value for c in at.caption).lower()
    for phrase in ("area-level", "Not a prediction for any specific property", "not an insurance quote",
                   "not financial advice", "Not affiliated with or endorsed by FEMA", "Theo Dancey",
                   "github.com/theodancey/flood-risk-explorer", "OpenFEMA", "Last updated"):
        check(phrase.lower() in foot, f"{where}: footer phrase missing: '{phrase}'")
    check(not any("disclaimer" in str(c.value).lower() for c in at.sidebar.caption) and
          not any("not financial advice" in str(c.value).lower() for c in at.sidebar.caption),
          f"{where}: the sidebar still repeats the disclaimer")


JARGON = ("Shapley", "credibility Z", "policy-year", "pure premium", "Mix: overlap", "Bühlmann")


def visible_text(at):
    """Text on the page OUTSIDE expanders (the 'Learn more' sections may be technical)."""
    inside = set()
    for e in at.expander:
        for kind in ("markdown", "caption"):
            inside |= {id(x) for x in getattr(e, kind)}
    out = []
    for kind in ("markdown", "caption", "info", "warning", "subheader", "title", "metric"):
        for x in getattr(at, kind):
            if id(x) in inside:
                continue
            out.append(str(getattr(x, "label", "")) + " " + str(x.value))
    return " ".join(out)


def no_jargon(at, where):
    text = visible_text(at)
    for w in JARGON:
        check(w.lower() not in text.lower(), f"{where}: jargon '{w}' shown outside a Learn more section")


def waterfall_labels(at, key):
    """y labels of the plotly waterfall with this key."""
    import json
    for ch in at.get("plotly_chart"):
        spec = json.loads(ch.proto.spec)
        for tr in spec.get("data", []):
            if tr.get("type") == "waterfall":
                if key is None or key in str(ch.proto.id):
                    return list(tr["y"])
    return []


def check_labels(at, where, key=None):
    ys = waterfall_labels(at, key)
    check(len(ys) >= 3, f"{where}: no waterfall found")
    check(not any("True" in y or "False" in y for y in ys), f"{where}: True/False in waterfall labels {ys}")
    check(not any(y.startswith("Mix:") for y in ys), f"{where}: technical 'Mix:' label {ys}")
    return ys


def money(x):
    return f"${x:,.0f}"


# ---------------------------------------------------------------------------
# 1-2. County Explorer
# ---------------------------------------------------------------------------
def test_county_explorer():
    at = new_app()
    has_disclaimer(at, "county page (default)")
    no_error(at, "county page (default)")
    no_jargon(at, "county page (default)")
    ys = check_labels(at, "county page (default)", key="why")
    check("Other small effects" in ys, f"county page: small effects not grouped: {ys}")
    n = 0
    for state in ("VA", "NC", "MD"):
        at.segmented_control(key="ce_state").set_value(state).run()
        no_error(at, f"county page, state {state}")
        opts = at.selectbox(key="ce_county").options
        expected_n = int((CS["state"] == state).sum())
        check(len(opts) == expected_n, f"{state}: {len(opts)} counties in the list, expected {expected_n}")
        for fips in CS.loc[CS["state"] == state, "county"]:
            at.selectbox(key="ce_county").set_value(fips).run()
            no_error(at, f"county {fips}")
            row = CS[CS["county"] == fips].iloc[0]
            check(at.metric[0].value == money(row["expected_pure_premium"]),
                  f"county {fips}: shows {at.metric[0].value}, file says {money(row['expected_pure_premium'])}")
            check(any(row["name"] in s.value for s in at.subheader), f"county {fips}: name not shown")
            check(at.metric[1].value == money(row["actual_loss_cost_2010_2025"]),
                  f"county {fips}: actual average shows {at.metric[1].value}")
            if n % 25 == 0:
                check_labels(at, f"county {fips}", key="why")
                no_jargon(at, f"county {fips}")
            n += 1
    print(f"  county explorer: {n} counties rendered")
    # Names shared by several places resolve to the right FIPS (city vs county, three Montgomerys)
    for fips, name, st in [("51760", "Richmond city", "VA"), ("51159", "Richmond County", "VA"),
                           ("37153", "Richmond County", "NC"), ("51600", "Fairfax city", "VA"),
                           ("51059", "Fairfax County", "VA"), ("24031", "Montgomery County", "MD"),
                           ("37123", "Montgomery County", "NC"), ("51121", "Montgomery County", "VA")]:
        at.segmented_control(key="ce_state").set_value(st).run()
        at.selectbox(key="ce_county").set_value(fips).run()
        check(at.subheader[0].value == f"{name}, {st}", f"{fips}: header '{at.subheader[0].value}'")
    # A county with no paid claims explains why its certainty is Low
    at.segmented_control(key="ce_state").set_value("NC").run()
    at.selectbox(key="ce_county").set_value("37033").run()          # Caswell County: no paid claims
    check(any("no paid NFIP flood claims" in i.value for i in at.info), "zero-claim county message missing")


# ---------------------------------------------------------------------------
# 3. Profile Calculator
# ---------------------------------------------------------------------------
PROFILES = [   # (state, county or None, settings)
    ("VA", None, {}),
    ("VA", "51810", {"pc_built": "1960-1979", "pc_floors": "1"}),            # Virginia Beach check value
    ("VA", "51710", {"pc_elev": "True", "pc_built": "2000 or later", "pc_primary": "False"}),
    ("VA", "51037", {"pc_occ": "non-residential", "pc_cover": "$0 (contents only)"}),
    ("NC", None, {"pc_zone": "B/C/X moderate-low risk"}),
    ("NC", "37055", {"pc_built": "1960-1979", "pc_floors": "1"}),            # Dare County check value
    ("NC", "37129", {"pc_occ": "multi-unit residential", "pc_cover": "under $100K", "pc_floors": "3+"}),
    ("NC", "37033", {"pc_built": "before 1960", "pc_firm": "True"}),         # unusual combination
    ("MD", None, {"pc_cover": "$100K-$249K", "pc_primary": "False"}),
    ("MD", "24047", {"pc_built": "1960-1979", "pc_floors": "1"}),            # Worcester County check value
    ("MD", "24510", {"pc_occ": "non-residential", "pc_built": "before 1960", "pc_zone": "B/C/X moderate-low risk"}),
    ("MD", "24027", {"pc_built": "2000 or later", "pc_firm": "False", "pc_elev": "True"}),
]
CHECK_VALUES = {"51810": 453.69, "37055": 2820.24, "24047": 458.67}      # reports/10_multistate.md


def set_widget(at, key, value):
    for kind in ("selectbox", "radio", "segmented_control"):
        try:
            getattr(at, kind)(key=key).set_value(value)
            return
        except (KeyError, IndexError):
            continue
    raise KeyError(key)


def test_profile_calculator():
    at = new_app()
    at.switch_page("views/profile_calculator.py").run()
    no_error(at, "calculator (default)")
    has_disclaimer(at, "calculator page")
    for state, county, settings in PROFILES:
        at = new_app()
        at.switch_page("views/profile_calculator.py").run()
        at.segmented_control(key="pc_state").set_value(state).run()
        at.selectbox(key="pc_county").set_value(county or "__any__").run()
        for k, v in settings.items():
            set_widget(at, k, v)
        at.run()
        where = f"profile {state} {county} {settings}"
        no_error(at, where)
        check_labels(at, where, key="pc_wf")
        no_jargon(at, where)
        # What the page should show = src/explain.py on the same profile
        feats = dict(EX.t.d["bases"])
        keymap = {"pc_zone": "flood_zone_group", "pc_built": "construction_band", "pc_occ": "occupancy_group",
                  "pc_cover": "building_coverage_band", "pc_elev": "elevated", "pc_floors": "floors_group",
                  "pc_primary": "primary_residence"}
        for k, v in settings.items():
            if k in keymap:
                feats[keymap[k]] = v
        if feats["occupancy_group"] == "non-residential":
            feats["primary_residence"] = "False"
            check(at.radio(key="pc_primary").disabled, f"{where}: primary residence not locked")
        r = EX.explain_profile({**feats, "county": county}, state=state)
        check(at.metric[0].value == money(r["pure_premium"]),
              f"{where}: shows {at.metric[0].value}, explain gives {money(r['pure_premium'])}")
        if county in CHECK_VALUES:
            check(abs(r["pure_premium"] - CHECK_VALUES[county]) < 0.01,
                  f"{county}: check value {r['pure_premium']:.2f} != {CHECK_VALUES[county]}")
        if settings.get("pc_firm") == "True" and settings.get("pc_built") == "before 1960":
            check(any("Unusual combination" in w.value for w in at.warning), f"{where}: no FIRM warning")
        if settings.get("pc_firm") == "False" and settings.get("pc_built") == "2000 or later":
            # about 6% of 2000+ buildings are pre-FIRM: uncommon but real, so no warning
            check(not at.warning, f"{where}: unexpected warning")
    # Rare building type in a tiny county: friendly note, still a result
    at = new_app()
    at.switch_page("views/profile_calculator.py").run()
    at.selectbox(key="pc_county").set_value("51037").run()        # Charlotte County, VA (25 policy-years)
    at.selectbox(key="pc_occ").set_value("multi-unit residential").run()
    no_error(at, "rare profile")
    check(any("Few insured buildings exactly like this" in i.value for i in at.info), "rare-profile note missing")
    check(at.metric[0].value.startswith("$"), "rare profile: no result shown")
    # Switching state after choosing a county keeps the page valid
    at.segmented_control(key="pc_state").set_value("MD").run()
    no_error(at, "state switch with a county selected")
    check(at.selectbox(key="pc_county").value in ["__any__"] + list(CS.loc[CS["state"] == "MD", "county"]),
          f"after switching to MD the county is {at.selectbox(key='pc_county').value}")
    print(f"  profile calculator: {len(PROFILES)} profiles + edge cases")


# ---------------------------------------------------------------------------
# /county route (Milestone 15): the same County Explorer script, registered as a hidden page
# ---------------------------------------------------------------------------
def test_county_route():
    at = new_app()
    pages = at._registered_pages
    by_url = {info.get("url_pathname"): h for h, info in pages.items()}
    check("county" in by_url, f"no page registered at /county: {sorted(map(str, by_url))}")
    check("" in by_url or any(info.get("url_pathname") == "" for info in pages.values()) or "explorer" in by_url,
          "no default page")
    if "county" in by_url:
        at._page_hash = by_url["county"]
        at.run()
        no_error(at, "/county")
        check(any("County Explorer" == t.value for t in at.title), "/county did not open the County Explorer")
        check(at.metric[0].value.startswith("$"), "/county: no estimate shown")
    print("  /county route: opens the County Explorer")


# ---------------------------------------------------------------------------
# How It Works
# ---------------------------------------------------------------------------
def test_how_it_works():
    at = new_app()
    at.switch_page("views/how_it_works.py").run()
    no_error(at, "how it works")
    has_disclaimer(at, "how it works page")
    heads = [h.value for h in at.header]
    for h in ("Data sources", "Methods in plain English", "Key findings", "Limitations", "Future work", "About"):
        check(h in heads, f"how it works: section '{h}' missing")
    text = " ".join(m.value for m in at.markdown)
    check("github.com/theodancey/flood-risk-explorer" in text, "how it works: GitHub link missing")
    check("Theo Dancey" in text, "how it works: About blurb missing")
    for h in ("In short", "How the certainty signals work"):
        check(h in heads, f"how it works: section '{h}' missing")
    for term in ("Shapley", "Bühlmann-Straub", "credibility Z", "policy-year", "Pure premium", "6.2×", "6.6×"):
        check(term in text, f"how it works: '{term}' not explained")
    check("2009-2025" not in text or "included" in text, "how it works: 2009-2025 not explained")
    print("  how it works: all sections present")


# ---------------------------------------------------------------------------
# 4. Never reads data/raw or data/processed
# ---------------------------------------------------------------------------
def test_reads():
    norm = [p.replace("\\", "/") for p in READS]
    bad = [p for p in norm if "data/raw" in p or "data/processed" in p]
    check(not bad, f"the app read raw/processed data: {bad[:3]}")
    data_reads = {Path(p).resolve() for p in READS if Path(p).suffix in (".csv", ".parquet", ".json")}
    outside = [str(p) for p in data_reads if APP_DATA not in p.parents]
    check(not outside, f"data files read from outside app/data: {outside}")
    check(len(data_reads) >= 4, f"expected the app to read its data files; logged {sorted(map(str, data_reads))}")
    # Static check of the app's source code too
    for f in list((ROOT / "app").rglob("*.py")):
        src = f.read_text(encoding="utf-8").replace("\\", "/")
        check("data/raw" not in src and "data/processed" not in src, f"{f.name} mentions data/raw or data/processed")
    print(f"  file reads: {len(data_reads)} data files, all in app/data: {sorted(p.name for p in data_reads)}")


# ---------------------------------------------------------------------------
# 5. Lightweight: no heavy modelling libraries
# ---------------------------------------------------------------------------
def test_lightweight():
    code = ("import sys; sys.path.insert(0, r'%s'); import common; common.explain.load(common.DATA, with_model=False); "
            "heavy = [m for m in ('statsmodels', 'scipy', 'sklearn', 'xgboost', 'shap', 'matplotlib') if m in sys.modules]; "
            "print(','.join(heavy) or 'none')" % (ROOT / "app"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    heavy = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else f"ERROR {out.stderr[-300:]}"
    check(heavy == "none", f"the app imports heavy libraries: {heavy}")
    reqs = [l.split("==")[0] for l in (ROOT / "app" / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    check(set(reqs) == {"streamlit", "pandas", "numpy", "pyarrow", "plotly"}, f"app/requirements.txt: {reqs}")
    print(f"  heavy libraries imported by the app: {heavy}; app requirements: {reqs}")


if __name__ == "__main__":
    t0 = time.time()
    for test in (test_county_explorer, test_profile_calculator, test_county_route, test_how_it_works,
                 test_reads, test_lightweight):
        t = time.time()
        print(test.__name__)
        test()
        print(f"  ({time.time() - t:.0f} s)")
    print(f"\n{'ALL TESTS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S)'} in {time.time() - t0:.0f} s")
    sys.exit(1 if FAILURES else 0)
