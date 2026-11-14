r"""
Tests for the address lookup (app/lookup.py), Milestone 15. Plain Python, no pytest needed:

    .venv\Scripts\python.exe tests\test_lookup.py

They NEVER use the internet: every web request is answered by a fake ("mocked") response that
imitates the real Census Geocoder and FEMA flood map answers, and any real network connection is
blocked so a test can't reach the internet by accident.

What it checks:
  1. Geocoding: a good match, no match, an address outside VA/NC/MD, a service that's down, a
     service that fails once and then answers (retry), and an unreadable answer.
  2. Flood zones: A/AE/AO/VE -> high risk, X (shaded or not)/B/C -> lower risk, D / not mapped /
     empty -> left for the user, and several zones at one point -> the riskiest.
  3. Community and first-map date: by FEMA's community ID, and by name when that fails.
  4. FEMA's pre/post-FIRM rule at its edges (1974/1975, the year of the first map, a missing date).
  5. Year built -> age group.
"""

import socket
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import lookup  # noqa: E402

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("  FAIL:", msg)


# ---------------------------------------------------------------------------
# No internet in tests: block real connections, and answer requests.get with fakes
# ---------------------------------------------------------------------------
def _blocked(*args, **kwargs):
    raise RuntimeError("network access is blocked in tests")


socket.socket.connect = _blocked
socket.create_connection = _blocked
lookup._pause = lambda s: None          # no real waiting between retries


class FakeResponse:
    def __init__(self, data=None, status=200, bad_json=False):
        self.status_code, self._data, self._bad = status, data, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not JSON")
        return self._data


def geocode_answer(state_fips="51", state_name="Virginia", county="51760", county_name="Richmond city",
                   place="Richmond", x=-77.4353, y=37.5385):
    geos = {"States": [{"STATE": state_fips, "NAME": state_name}],
            "Counties": [{"GEOID": county, "NAME": county_name}],
            "Incorporated Places": [{"BASENAME": place}] if place else []}
    return {"result": {"addressMatches": [{"matchedAddress": "1000 BANK ST, RICHMOND, VA, 23219",
                                           "coordinates": {"x": x, "y": y}, "geographies": geos}]}}


def zone_answer(*zones):
    return {"features": [{"attributes": {"FLD_ZONE": z, "ZONE_SUBTY": sub}} for z, sub in zones]}


def juris_answer(cid):
    return {"features": [{"attributes": {"CID": cid, "POL_NAME1": "X"}}]} if cid else {"features": []}


class Fake:
    """Replaces requests.get. `routes` maps a URL piece to a list of answers given in turn (the last
    one repeats); an answer can be a FakeResponse or an exception to raise. Records every call."""
    def __init__(self, routes):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), timeout))
        for piece, answers in self.routes.items():
            if piece in url:
                a = answers.pop(0) if len(answers) > 1 else answers[0]
                if isinstance(a, Exception):
                    raise a
                return a
        raise AssertionError(f"unexpected URL in test: {url}")


def use(fake):
    lookup.requests.get = fake
    return fake


FIRM = pd.DataFrame([
    {"cid": "510129", "state": "VA", "community": "RICHMOND, CITY OF", "county": "RICHMOND CITY", "initial_firm": "1979-06-15"},
    {"cid": "375348", "state": "NC", "community": "DARE COUNTY", "county": "DARE COUNTY", "initial_firm": "1978-10-06"},
    {"cid": "375347", "state": "NC", "community": "KILL DEVIL HILLS, TOWN OF", "county": "DARE COUNTY", "initial_firm": "1974-06-01"},
    {"cid": "510000", "state": "VA", "community": "NOMAP COUNTY", "county": "NOMAP COUNTY", "initial_firm": ""},
])


# ---------------------------------------------------------------------------
# 1. Geocoding
# ---------------------------------------------------------------------------
def test_geocode():
    f = use(Fake({"geocoder": [FakeResponse(geocode_answer())]}))
    g = lookup.geocode("  1000 Bank St,   Richmond, VA 23219 ")
    check(g["status"] == "ok" and g["state"] == "VA" and g["county_fips"] == "51760" and g["place"] == "Richmond",
          f"geocode ok: {g}")
    check(abs(g["lat"] - 37.5385) < 1e-9 and abs(g["lon"] + 77.4353) < 1e-9, "geocode: coordinates")
    url, params, timeout = f.calls[0]
    check(params["address"] == "1000 Bank St, Richmond, VA 23219", f"address not cleaned: {params['address']!r}")
    check(timeout is not None and timeout <= 10, f"no short timeout: {timeout}")
    check(params.get("benchmark") == "Public_AR_Current" and params.get("format") == "json", "geocoder params")

    use(Fake({"geocoder": [FakeResponse({"result": {"addressMatches": []}})]}))
    check(lookup.geocode("1 Nowhere Rd, Atlantis")["status"] == "no_match", "no match")

    use(Fake({"geocoder": [FakeResponse(geocode_answer("11", "District of Columbia", "11001", "District of Columbia"))]}))
    g = lookup.geocode("1600 Pennsylvania Ave, Washington, DC")
    check(g["status"] == "outside" and "District of Columbia" in g["message"] and "Virginia" in g["message"],
          f"outside: {g}")

    use(Fake({"geocoder": [requests.Timeout("slow")]}))
    g = lookup.geocode("1000 Bank St, Richmond, VA")
    check(g["status"] == "error", f"timeout should give a friendly error: {g}")
    check("1000 Bank" not in g["message"], "the error message must not repeat the address")

    f = use(Fake({"geocoder": [requests.ConnectionError("down"), FakeResponse(status=503),
                               FakeResponse(geocode_answer())]}))
    g = lookup.geocode("1000 Bank St, Richmond, VA")
    check(g["status"] == "ok" and len(f.calls) == 3, f"retry: {g['status']} after {len(f.calls)} calls")

    use(Fake({"geocoder": [FakeResponse(bad_json=True)]}))
    check(lookup.geocode("1000 Bank St, Richmond, VA")["status"] == "error", "unreadable answer")
    use(Fake({"geocoder": [FakeResponse({"result": {"addressMatches": [{"geographies": {}}]}})]}))
    check(lookup.geocode("1000 Bank St, Richmond, VA")["status"] == "error", "answer missing fields")
    check(lookup.geocode("   ")["status"] == "empty", "empty address")
    print("  geocode: ok / no match / outside / timeout / retry / bad answers")


# ---------------------------------------------------------------------------
# 2. Flood zones
# ---------------------------------------------------------------------------
def test_zones():
    cases = {"A": "A/V high risk", "AE": "A/V high risk", "AO": "A/V high risk", "AH": "A/V high risk",
             "A99": "A/V high risk", "VE": "A/V high risk", "V": "A/V high risk",
             "X": "B/C/X moderate-low risk", "B": "B/C/X moderate-low risk", "C": "B/C/X moderate-low risk",
             "D": None, "AREA NOT INCLUDED": None, "OPEN WATER": None, "": None, None: None}
    for z, want in cases.items():
        check(lookup.zone_group(z) == want, f"zone {z!r} -> {lookup.zone_group(z)}, expected {want}")
    check(lookup.zone_group("X", "0.2 PCT ANNUAL CHANCE FLOOD HAZARD") == "B/C/X moderate-low risk", "shaded X")
    check("shaded" in lookup.describe_zone("X", "0.2 PCT ANNUAL CHANCE FLOOD HAZARD"), "shaded X label")

    use(Fake({"MapServer/28": [FakeResponse(zone_answer(("AE", None)))]}))
    z = lookup.flood_zone(36.85, -75.98)
    check(z["status"] == "ok" and z["zone_group"] == "A/V high risk", f"AE point: {z}")
    use(Fake({"MapServer/28": [FakeResponse(zone_answer(("X", "AREA OF MINIMAL FLOOD HAZARD"), ("VE", None)))]}))
    z = lookup.flood_zone(36.85, -75.98)
    check(z["zone_group"] == "A/V high risk" and z["fld_zone"] == "VE", f"two zones at a point -> riskiest: {z}")
    use(Fake({"MapServer/28": [FakeResponse(zone_answer(("D", None)))]}))
    check(lookup.flood_zone(36.85, -75.98)["status"] == "unmapped", "zone D left to the user")
    use(Fake({"MapServer/28": [FakeResponse({"features": []})]}))
    check(lookup.flood_zone(36.85, -75.98)["status"] == "unmapped", "no zone at point")
    use(Fake({"MapServer/28": [FakeResponse({"error": {"code": 500}})]}))
    check(lookup.flood_zone(36.85, -75.98)["status"] == "error", "service error answer")
    use(Fake({"MapServer/28": [requests.Timeout("slow")]}))
    check(lookup.flood_zone(36.85, -75.98)["status"] == "error", "zone timeout")
    print("  flood zones: mapping, shaded X, riskiest of several, D / none / errors")


# ---------------------------------------------------------------------------
# 3. Community + first map date
# ---------------------------------------------------------------------------
def test_community():
    use(Fake({"MapServer/22": [FakeResponse(juris_answer("510129"))]}))
    c = lookup.community(37.5, -77.4, "VA", "Richmond city", "Richmond", FIRM)
    check(c["status"] == "ok" and c["initial_firm"] == date(1979, 6, 15) and c["name"] == "City of Richmond",
          f"by CID: {c}")
    use(Fake({"MapServer/22": [requests.ConnectionError("down")]}))
    c = lookup.community(37.5, -77.4, "VA", "Richmond city", "Richmond", FIRM)
    check(c["status"] == "ok" and c["cid"] == "510129" and "name" in c["how"], f"fallback by city name: {c}")
    c = lookup.community(35.9, -75.6, "NC", "Dare County", None, FIRM)
    check(c["status"] == "ok" and c["cid"] == "375348", f"fallback by county name (unincorporated): {c}")
    c = lookup.community(35.9, -75.6, "NC", "Dare County", "Kill Devil Hills", FIRM)
    check(c["status"] == "ok" and c["cid"] == "375347", f"fallback by town name: {c}")
    c = lookup.community(35.9, -75.6, "NC", "Dare County", "Unknown Town", FIRM)
    check(c["status"] == "not_found", f"unknown town -> not found: {c}")
    use(Fake({"MapServer/22": [FakeResponse(juris_answer("510000"))]}))
    c = lookup.community(37.0, -78.0, "VA", "Nomap County", None, FIRM)
    check(c["status"] == "ok" and c["initial_firm"] is None, f"community with no FIRM date: {c}")
    print("  community: by FEMA community ID, by town / county name, not found, no date")


# ---------------------------------------------------------------------------
# 4. Pre/post-FIRM rule
# ---------------------------------------------------------------------------
def test_firm_rule():
    early = date(1970, 10, 3)          # first map BEFORE 1975: the Dec 31, 1974 date decides
    check(lookup.firm_status(1974, early)["status"] == "pre", "1974 with a 1970 map -> pre")
    check(lookup.firm_status(1975, early)["status"] == "post", "1975 with a 1970 map -> post (cutoff Jan 1, 1975)")
    late = date(1979, 6, 15)           # first map AFTER 1975: the map date decides
    check(lookup.firm_status(1978, late)["status"] == "pre", "1978 with a mid-1979 map -> pre")
    check(lookup.firm_status(1979, late)["status"] == "unsure", "1979 with a mid-1979 map -> not sure")
    check(lookup.firm_status(1980, late)["status"] == "post", "1980 with a mid-1979 map -> post")
    check(lookup.firm_status(1977, date(1977, 1, 1))["status"] == "post", "map on Jan 1 of the year -> post")
    check(lookup.firm_status(1990, None)["status"] == "unsure", "no map date -> not sure")
    check(lookup.firm_status(None, late)["status"] == "unsure", "no year -> not sure")
    r = lookup.firm_status(1980, late)["rule"]
    check("December 31, 1974" in r and "June 15, 1979" in r, f"rule text: {r}")
    print("  pre/post-FIRM rule: edges around 1975 and the first-map year")


def test_bands():
    for y, b in [(1900, "before 1960"), (1959, "before 1960"), (1960, "1960-1979"), (1979, "1960-1979"),
                 (1980, "1980-1999"), (1999, "1980-1999"), (2000, "2000 or later"), (2025, "2000 or later"),
                 (None, None), (1500, None), (3000, None)]:
        check(lookup.year_to_band(y) == b, f"year {y} -> {lookup.year_to_band(y)}, expected {b}")
    print("  year built -> age group")


def test_no_logging():
    src = (ROOT / "app" / "lookup.py").read_text(encoding="utf-8")
    check("print(" not in src and "logging" not in src and "open(" not in src,
          "lookup.py must not print, log or write files")
    for f in (ROOT / "app").rglob("*.py"):
        check("query_params" not in f.read_text(encoding="utf-8"), f"{f.name} uses st.query_params")
    print("  privacy: no printing, logging, files or URL parameters")


# ---------------------------------------------------------------------------
# Milestone 16: readable addresses, the automatic retry, the "not found" advice
# ---------------------------------------------------------------------------
def test_address_polish():
    for raw, want in [("1000 BANK ST, RICHMOND, VA, 23219", "1000 Bank St, Richmond, VA, 23219"),
                      ("100 N MAIN ST, MOUNT AIRY, NC, 27030", "100 N Main St, Mount Airy, NC, 27030"),
                      ("1 MARKET PL, BALTIMORE, MD, 21202", "1 Market Pl, Baltimore, MD, 21202")]:
        check(lookup.pretty_address(raw) == want, f"pretty_address({raw!r}) = {lookup.pretty_address(raw)!r}")
    for raw, want in [("1000 Bank St Apt 4B, Richmond, VA 23219", "1000 Bank St, Richmond, VA 23219"),
                      ("12 Main St., #5, Duck, NC", "12 Main St, Duck, NC"),
                      ("100 Oak Ave, Suite 300, Baltimore, MD 21201", "100 Oak Ave, Baltimore, MD 21201"),
                      ("1000 Bank St, Richmond, VA 23219", None)]:
        check(lookup.simpler_address(raw) == want, f"simpler_address({raw!r}) = {lookup.simpler_address(raw)!r}")
    # no match, then the simpler address matches: one automatic retry, without the unit
    none = FakeResponse({"result": {"addressMatches": []}})
    f = use(Fake({"geocoder": [none, FakeResponse(geocode_answer())]}))
    g = lookup.geocode("1000 Bank St Apt 4B, Richmond, VA 23219")
    check(g["status"] == "ok" and len(f.calls) == 2, f"retry variant: {g['status']} after {len(f.calls)} calls")
    check(f.calls[1][1]["address"] == "1000 Bank St, Richmond, VA 23219", f"retry sent {f.calls[1][1]['address']!r}")
    # nothing simpler to try: only one call, and the advice mentions the ZIP code and a nearby number
    f = use(Fake({"geocoder": [none]}))
    g = lookup.geocode("9999 Nowhere Rd, Atlantis, VA 23219")
    check(g["status"] == "no_match" and len(f.calls) == 1, f"no retry expected: {len(f.calls)} calls")
    check("ZIP code" in g["message"] and "nearby house number" in g["message"], f"advice: {g['message']}")
    # both tries fail: still the advice
    f = use(Fake({"geocoder": [none]}))
    g = lookup.geocode("5 Gone Ln Unit 2, Atlantis, VA")
    check(g["status"] == "no_match" and len(f.calls) == 2, f"two tries expected: {len(f.calls)} calls")
    print("  address polish: state codes in capitals, unit-free retry, not-found advice")


if __name__ == "__main__":
    real_get = requests.get
    for t in (test_geocode, test_zones, test_community, test_firm_rule, test_bands, test_no_logging,
              test_address_polish):
        print(t.__name__)
        t()
    requests.get = real_get
    print(f"\n{'ALL TESTS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S)'}")
    sys.exit(1 if FAILURES else 0)
