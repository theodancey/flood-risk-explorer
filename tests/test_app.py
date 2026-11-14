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
  7. Address lookup with FAKE web answers (no internet: real connections are blocked).
  8. The county map (all counties, legend, hover, both colourings, a click selects the county) and
     the "what if?" elevation estimate (equal to src/explain.py).
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

# No internet during the tests: any connection to a non-local address fails (the address lookup is
# tested with fake answers instead). Local connections are allowed: Python uses them internally.
import socket  # noqa: E402
_real_connect = socket.socket.connect


def _local_only(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"network access is blocked in tests ({host})")
    return _real_connect(self, address)


socket.socket.connect = _local_only

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
        check(any(m.label == "Model estimate (typical year)" for m in at.metric), "/county did not open the County Explorer")
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
    for term in ("Shapley", "Bühlmann-Straub", "credibility Z", "policy-year", "Pure premium", "6.2×", "6.6×",
                 "Coastal Plain", "8.9%", "less certain", "One storm level per state", "Lee County"):
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
    data_reads = {Path(p).resolve() for p in READS if Path(p).suffix in (".csv", ".parquet", ".json", ".geojson")}
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
    check(set(reqs) == {"streamlit", "pandas", "numpy", "pyarrow", "plotly", "requests"}, f"app/requirements.txt: {reqs}")
    print(f"  heavy libraries imported by the app: {heavy}; app requirements: {reqs}")


# ---------------------------------------------------------------------------
# 7. Address lookup (Milestone 15), with FAKE web answers: the tests never use the internet
# ---------------------------------------------------------------------------
import requests  # noqa: E402

REAL_GET = requests.get


class FakeResponse:
    def __init__(self, data=None, status=200):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


def fake_web(state_fips="37", county="37055", county_name="Dare County", place=None, x=-75.62, y=35.91,
             zone="AE", cid="375348", geocoder_down=False, zone_down=False, state_name="North Carolina",
             matched="100 TEST ST, SOMEWHERE, NC, 27948"):
    """A stand-in for requests.get that answers like the Census Geocoder and FEMA's flood map."""
    def get(url, params=None, timeout=None):
        if "geocoder" in url:
            if geocoder_down:
                raise requests.ConnectionError("down")
            geos = {"States": [{"STATE": state_fips, "NAME": state_name}],
                    "Counties": [{"GEOID": county, "NAME": county_name}],
                    "Incorporated Places": [{"BASENAME": place}] if place else []}
            return FakeResponse({"result": {"addressMatches": [{"matchedAddress": matched,
                                                                "coordinates": {"x": x, "y": y},
                                                                "geographies": geos}]}})
        if "MapServer/28" in url:
            if zone_down:
                raise requests.Timeout("slow")
            return FakeResponse({"features": [{"attributes": {"FLD_ZONE": zone, "ZONE_SUBTY": None}}]})
        if "MapServer/22" in url:
            return FakeResponse({"features": [{"attributes": {"CID": cid}}]})
        raise AssertionError(f"unexpected URL: {url}")
    return get


def click(at, label):
    btns = [b for b in at.button if b.label == label]
    check(len(btns) == 1, f"button '{label}' not found: {[b.label for b in at.button]}")
    btns[0].click().run()


def page_text(at):
    return " ".join(str(x.value) for kind in ("markdown", "caption", "warning", "success", "info")
                    for x in getattr(at, kind))


def test_address_lookup():
    sys.path.insert(0, str(ROOT / "app"))
    import lookup
    lookup._pause = lambda s: None
    try:
        # 1. Calculator: a Dare County, NC address in zone AE, built 1985 (first map 1978) -> everything filled in
        requests.get = fake_web()
        at = new_app()
        at.switch_page("views/profile_calculator.py").run()
        text = page_text(at)
        check("Your address is only used to look up public map data and is not saved." in text,
              "calculator: privacy sentence missing")
        at.text_input(key="pc_addr").set_value("100 Test St, Kill Devil Hills, NC")
        at.number_input(key="pc_year").set_value(1985)
        click(at, "Look up")
        no_error(at, "address lookup (Dare)")
        ss = at.session_state
        check(ss["pc_state"] == "NC" and ss["pc_county"] == "37055", f"state/county not set: {ss['pc_state']} {ss['pc_county']}")
        check(ss["pc_zone"] == "A/V high risk", f"zone not set: {ss['pc_zone']}")
        check(ss["pc_built"] == "1980-1999", f"year built not set: {ss['pc_built']}")
        check(ss["pc_firm"] == "True", f"post-FIRM not set: {ss['pc_firm']}")
        text = page_text(at)
        for phrase in ("U.S. Census Geocoder", "FEMA National Flood Hazard Layer", "OpenFEMA NFIP Community Status Book",
                       "October 6, 1978", "not an official flood zone determination", "msc.fema.gov",
                       "average for similar insured buildings", "December 31, 1974"):
            check(phrase in text, f"address lookup: '{phrase}' not shown")
        feats = {**EX.t.d["bases"], "flood_zone_group": "A/V high risk", "construction_band": "1980-1999"}
        r = EX.explain_profile({**feats, "county": "37055"}, state="NC")
        check(at.metric[0].value == money(r["pure_premium"]),
              f"after lookup shows {at.metric[0].value}, expected {money(r['pure_premium'])}")
        # the user can still override what was filled in
        at.radio(key="pc_zone").set_value("B/C/X moderate-low risk").run()
        no_error(at, "override after lookup")
        check(at.session_state["pc_zone"] == "B/C/X moderate-low risk", "could not override the zone")

        # 2. An address outside VA/NC/MD: a clear message, inputs unchanged
        requests.get = fake_web(state_fips="11", county="11001", county_name="District of Columbia",
                                state_name="District of Columbia")
        at = new_app()
        at.switch_page("views/profile_calculator.py").run()
        at.text_input(key="pc_addr").set_value("1 Outside Ave, Washington, DC")
        click(at, "Look up")
        no_error(at, "outside address")
        check(any("only covers Virginia, North Carolina and Maryland" in w.value for w in at.warning),
              "outside address: message missing")
        check(at.session_state["pc_state"] == "VA", "outside address changed the state")

        # 3. The address service is down: friendly message, manual inputs still work
        requests.get = fake_web(geocoder_down=True)
        at = new_app()
        at.switch_page("views/profile_calculator.py").run()
        at.text_input(key="pc_addr").set_value("2 Down St, Norfolk, VA")
        click(at, "Look up")
        no_error(at, "geocoder down")
        check(any("didn't answer" in w.value for w in at.warning), "geocoder down: message missing")
        check(at.metric[0].value.startswith("$"), "geocoder down: no result shown")

        # 4. The flood map service is down: county filled in, zone left for the user
        requests.get = fake_web(zone_down=True, x=-75.70, y=35.95)
        at = new_app()
        at.switch_page("views/profile_calculator.py").run()
        at.text_input(key="pc_addr").set_value("3 Slow Rd, Manteo, NC")
        click(at, "Look up")
        no_error(at, "zone service down")
        check(at.session_state["pc_county"] == "37055", "zone down: county not set")
        check("Please choose the flood zone yourself" in page_text(at), "zone down: message missing")

        # 5. County Explorer: "Find my county", then "Use this address in the calculator"
        requests.get = fake_web(state_fips="51", county="51810", county_name="Virginia Beach city",
                                place="Virginia Beach", x=-75.98, y=36.85, zone="VE", cid="515531",
                                state_name="Virginia", matched="4 BEACH RD, SOMEWHERE, VA, 23451")
        at = new_app()
        check("not saved" in page_text(at), "county page: privacy sentence missing")
        at.text_input(key="ce_addr").set_value("4 Beach Rd, Virginia Beach, VA")
        click(at, "Find")
        no_error(at, "find my county")
        check(at.session_state["ce_county"] == "51810", f"find my county: {at.session_state['ce_county']}")
        check(at.subheader[0].value == "Virginia Beach city, VA", f"find my county header: {at.subheader[0].value}")
        check(any("4 Beach Rd, Somewhere, VA, 23451" in s.value for s in at.success),
              f"found-it message: {[s.value for s in at.success]}")
        click(at, "Use this address in the calculator")
        no_error(at, "explorer -> calculator")
        check(any(t.value == "Property Profile Calculator" for t in at.title), "did not open the calculator")
        check(at.session_state["pc_county"] == "51810" and at.session_state["pc_zone"] == "A/V high risk",
              "calculator not pre-filled from the County Explorer")
        # Milestone 16: the address itself comes along, so adding a year built and pressing Look up works
        check(at.session_state["pc_addr"] == "4 Beach Rd, Virginia Beach, VA",
              f"address not carried to the calculator: {at.session_state['pc_addr']!r}")
        check(at.text_input(key="pc_addr").value == "4 Beach Rd, Virginia Beach, VA", "calculator address box empty")
        at.number_input(key="pc_year").set_value(1985)
        click(at, "Look up")
        no_error(at, "carried address + year")
        check(at.session_state["pc_built"] == "1980-1999" and at.session_state["pc_firm"] == "True",
              f"year 1985: {at.session_state['pc_built']}, post-FIRM {at.session_state['pc_firm']}")
        # change ONLY the year: age group and pre/post-FIRM are re-applied
        at.number_input(key="pc_year").set_value(1962)
        click(at, "Look up")
        no_error(at, "year-only change")
        check(at.session_state["pc_built"] == "1960-1979" and at.session_state["pc_firm"] == "False",
              f"year 1962: {at.session_state['pc_built']}, post-FIRM {at.session_state['pc_firm']}")
        # an emptied address box falls back to the last address found
        at.text_input(key="pc_addr").set_value("")
        at.number_input(key="pc_year").set_value(2005)
        click(at, "Look up")
        no_error(at, "empty box, year changed")
        check(at.session_state["pc_built"] == "2000 or later", f"year 2005: {at.session_state['pc_built']}")
        # state codes stay in capitals ("VA", not "Va")
        text = page_text(at)
        check("Somewhere, VA, 23451" in text and ", Va," not in text, "state code not in capitals")
        print("  address lookup: fill-in, sources, override, outside VA/NC/MD, services down, find my county")
    finally:
        requests.get = REAL_GET


# ---------------------------------------------------------------------------
# 8. County map and "what if?" (Milestone 15)
# ---------------------------------------------------------------------------
def chart_specs(at):
    import json
    return [(str(ch.proto.id), json.loads(ch.proto.spec)) for ch in at.get("plotly_chart")]


def test_map():
    import json
    at = new_app()
    no_error(at, "map")
    maps = [(i, s) for i, s in chart_specs(at) if any(t.get("type") == "choropleth" for t in s.get("data", []))]
    check(len(maps) == 1, f"expected one county map, found {len(maps)}")
    if not maps:
        return
    trace = maps[0][1]["data"][0]
    geo = json.loads((APP_DATA / "counties_va_nc_md.geojson").read_text(encoding="utf-8"))
    with_outline = {f["id"] for f in geo["features"]}
    expected = set(CS["county"]) & with_outline
    check(set(trace["locations"]) == expected, f"map shows {len(trace['locations'])} counties, expected {len(expected)}")
    check(len(expected) >= 256, f"only {len(expected)} counties have an outline")
    check((APP_DATA / "counties_va_nc_md.geojson").stat().st_size < 2e6, "boundary file is over 2 MB")
    check("colorbar" in trace and trace["colorbar"].get("ticktext"), "map legend (colour bar) missing")
    check("Expected yearly loss" in trace["hovertemplate"], "map hover text missing")
    # Milestone 16: opens on "compared with the state average"
    check(at.segmented_control(key="ce_map_mode").value == "ratio", "map should open in 'compared with state' mode")
    check("same" in trace["colorbar"]["ticktext"], f"ratio legend: {trace['colorbar'].get('ticktext')}")
    data = maps[0][1]["data"]
    check(len(data) == 4, f"expected county layer, state borders, state labels, selected outline: {len(data)} traces")
    borders = [d for d in data if d.get("name") == "state borders"]
    check(len(borders) == 1 and sorted(borders[0]["locations"]) == ["MD", "NC", "VA"]
          and borders[0]["marker"]["line"]["width"] >= 2, "thick state borders missing")
    labels = [d for d in data if d.get("type") == "scattergeo"]
    check(len(labels) == 1 and sorted(labels[0]["text"]) == ["MD", "NC", "VA"], "state labels missing")
    check(all(d.get("hoverinfo") == "skip" for d in borders + labels), "borders/labels must not take hover or clicks")
    check(maps[0][1]["layout"].get("dragmode") is False, "map must not pan when dragged")
    # the map's config: the scroll wheel scrolls the page, not the map
    cfg = [json.loads(ch.proto.config) for ch in at.get("plotly_chart") if str(ch.proto.id) == maps[0][0]][0]
    check(cfg.get("scrollZoom") is False and cfg.get("doubleClick") is False, f"map config: {cfg}")
    check(common_config_ok(), "MAP_CONFIG / MAP_DEFAULT in common.py changed")
    caps = " ".join(c.value for c in at.caption)
    check("Florence" in caps and "less certain" in caps, "cross-state note under the map missing")
    # switch to "expected yearly loss"
    at.segmented_control(key="ce_map_mode").set_value("loss").run()
    no_error(at, "map loss mode")
    trace = [s for _, s in chart_specs(at) if any(t.get("type") == "choropleth" for t in s.get("data", []))][0]["data"][0]
    check(any(str(x).startswith("$") for x in trace["colorbar"]["ticktext"]),
          f"loss legend: {trace['colorbar'].get('ticktext')}")
    # a map click: Streamlit stores the selection under the chart's key; the callback picks the county
    sys.path.insert(0, str(ROOT / "app"))
    import common
    for ev, want in [({"selection": {"points": [{"curve_number": 0, "location": "37055"}]}}, "37055"),
                     ({"selection": {"points": [{"curve_number": 0, "customdata": ["x", "$1", "y", "z", "24047"]}]}}, "24047"),
                     ({"selection": {"points": [{"curve_number": 1, "location": "51810"}]}}, None),
                     ({"selection": {"points": []}}, None), (None, None)]:
        check(common.selected_county(ev) == want, f"map click {ev} -> {common.selected_county(ev)}, expected {want}")
    at.session_state["ce_select"] = "37055"
    at.run()
    no_error(at, "county picked on the map")
    check(at.subheader[0].value == "Dare County, NC", f"map pick: header {at.subheader[0].value}")
    check(at.segmented_control(key="ce_state").value == "NC", "map pick: state not switched")
    # a real map click also sets ce_map_pick: then a small card with a link down to the details appears
    at.session_state["ce_map_pick"] = "37105"
    at.session_state["ce_select"] = "37105"
    at.run()
    no_error(at, "map card")
    text = " ".join(m.value for m in at.markdown)
    check("Lee County, NC" in text and "(#county-details)" in text and "See details below" in text,
          "summary card under the map missing")
    check(at.subheader[0].proto.anchor == "county-details", "details heading has no #county-details anchor")
    at.selectbox(key="ce_county").set_value("37085").run()           # another county from the list: no card
    check("See details below" not in " ".join(m.value for m in at.markdown), "card should go when the county changes")
    print(f"  map: {len(expected)} counties, legend, hover, both modes, borders, labels, no scroll zoom, "
          "click -> county, summary card")


def common_config_ok():
    sys.path.insert(0, str(ROOT / "app"))
    import common
    return common.MAP_CONFIG.get("scrollZoom") is False and common.MAP_DEFAULT == "ratio"


# ---------------------------------------------------------------------------
# Milestone 16: delta arrows and colours (below average = down + green, above = up + orange)
# ---------------------------------------------------------------------------
def delta_of(m):
    from streamlit.proto.Metric_pb2 import Metric
    return Metric.MetricDirection.Name(m.proto.direction), Metric.MetricColor.Name(m.proto.color)


def test_deltas():
    sys.path.insert(0, str(ROOT / "app"))
    import common
    for r, want in [(0.5, ("green", "down")), (0.85, ("green", "down")), (1.0, ("gray", "off")),
                    (1.05, ("gray", "off")), (0.95, ("gray", "off")), (1.23, ("orange", "up")), (6.0, ("orange", "up"))]:
        got = common.delta_style(r)
        check((got["delta_color"], got["delta_arrow"]) == want, f"delta_style({r}) = {got}, expected {want}")
    want = {"down": ("DOWN", "GREEN"), "up": ("UP", "ORANGE"), "same": ("NONE", "GRAY")}
    # County Explorer: below (Virginia Beach 0.85), above (Lee County NC 1.23), same (Worcester MD 1.00)
    at = new_app()
    for fips, kind in [("51810", "down"), ("37105", "up"), ("24047", "same")]:
        at.session_state["ce_select"] = fips
        at.run()
        no_error(at, f"delta {fips}")
        got = delta_of(at.metric[0])
        check(got == want[kind], f"county {fips}: delta {got}, expected {want[kind]} ({at.metric[0].delta})")
        if kind == "down":
            check("below" in at.metric[0].delta, f"county {fips}: delta text {at.metric[0].delta}")
        for m in at.metric[1:]:                                   # no other metric has an arrow
            check(not m.delta, f"county {fips}: unexpected delta on '{m.label}'")
    # every county's arrow agrees with its ratio
    for fips, r in zip(CS["county"], CS["ratio_to_state"]):
        s = common.delta_style(r)
        check((s["delta_arrow"] == "down") == (r <= 1 / 1.1) and (s["delta_arrow"] == "up") == (r >= 1.1),
              f"county {fips}: ratio {r} gives {s}")
    # Calculator: the default building is about the state average; a 1960s high-risk building is above it;
    # a raised, newer lower-risk building is below it
    for settings, kind in [({}, "same"), ({"pc_built": "1960-1979", "pc_floors": "1"}, "up"),
                           ({"pc_zone": "B/C/X moderate-low risk", "pc_built": "2000 or later", "pc_elev": "True"}, "down")]:
        at = new_app()
        at.switch_page("views/profile_calculator.py").run()
        for k, v in settings.items():
            set_widget(at, k, v)
        at.run()
        no_error(at, f"calculator delta {settings}")
        got = delta_of(at.metric[0])
        check(got == want[kind], f"calculator {settings}: delta {got}, expected {want[kind]} ({at.metric[0].delta})")
        # "Typical lower-risk building": "Yours is N times this" points up; "Yours is N% lower" points down
        d2, got2 = at.metric[1].delta, delta_of(at.metric[1])
        exp2 = want["up"] if "times" in d2 else want["down"] if "lower" in d2 else want["same"]
        check(got2 == exp2, f"calculator {settings}: lower-risk delta {got2} for '{d2}'")
    print("  deltas: helper, 3 counties, all 257 ratios, 3 calculator profiles x 2 metrics")


def test_what_if():
    for state, county, settings in [("VA", "51810", {"pc_built": "1960-1979"}), ("NC", None, {"pc_elev": "True"}),
                                    ("MD", "24047", {"pc_occ": "non-residential"})]:
        at = new_app()
        at.switch_page("views/profile_calculator.py").run()
        at.segmented_control(key="pc_state").set_value(state).run()
        at.selectbox(key="pc_county").set_value(county or "__any__").run()
        for k, v in settings.items():
            set_widget(at, k, v)
        at.run()
        where = f"what-if {state} {county} {settings}"
        no_error(at, where)
        feats = dict(EX.t.d["bases"])
        keymap = {"pc_built": "construction_band", "pc_elev": "elevated", "pc_occ": "occupancy_group"}
        for k, v in settings.items():
            feats[keymap[k]] = v
        if feats["occupancy_group"] == "non-residential":
            feats["primary_residence"] = "False"
        no = EX.explain_profile({**feats, "elevated": "False", "county": county}, state=state)["pure_premium"]
        yes = EX.explain_profile({**feats, "elevated": "True", "county": county}, state=state)["pure_premium"]
        text = " ".join(m.value for m in at.markdown)
        if feats["elevated"] == "False":
            want = f"{money(no)} → {money(yes)}"
            check(want in text and "Raising this building" in text, f"{where}: '{want}' not shown")
        else:
            check("already raised" in text and money(no) in text, f"{where}: already-raised message missing")
        check(f"{100 * (yes / no - 1):+.0f}%" in text or "already raised" in text, f"{where}: % change missing")
        caps = " ".join(c.value for c in at.caption)
        check("not an engineering estimate" in caps and "not a promise of savings" in caps,
              f"{where}: what-if caveat missing")
    print("  what if: elevation effect matches explain.py (3 profiles), caveat shown")


if __name__ == "__main__":
    t0 = time.time()
    import sys as _sys
    only = [a for a in _sys.argv[1:]]
    tests = (test_county_explorer, test_profile_calculator, test_county_route, test_address_lookup,
             test_map, test_deltas, test_what_if, test_how_it_works, test_reads, test_lightweight)
    for test in [t for t in tests if not only or t.__name__ in only]:
        t = time.time()
        print(test.__name__)
        test()
        print(f"  ({time.time() - t:.0f} s)")
    print(f"\n{'ALL TESTS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S)'} in {time.time() - t0:.0f} s")
    sys.exit(1 if FAILURES else 0)
