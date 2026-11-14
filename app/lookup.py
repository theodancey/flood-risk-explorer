"""
Address lookup for the Flood Risk Explorer (Milestone 15).

Turns a typed US street address into the inputs the calculator needs, using only free public
services that need no sign-up or key:

  1. U.S. Census Geocoder (geocoding.geo.census.gov): address -> map coordinates, state, county.
  2. FEMA National Flood Hazard Layer (NFHL) map service (hazards.fema.gov): coordinates -> the
     flood zone at that point, and the NFIP community (town, city or county) it belongs to.
  3. OpenFEMA NFIP Community Status Book (bundled snapshot, app/data/firm_dates.csv): community ->
     the date of its first Flood Insurance Rate Map (FIRM), to tell pre-FIRM from post-FIRM.

Design rules:
  * Short timeouts and a couple of retries, and NEVER an exception out of this module: every
    function returns a dict with a "status" so the page can fall back to manual inputs.
  * Privacy: the address is sent only to the Census Geocoder (it has to be, to find it). It is
    never printed, logged, written to disk or put in the app's own URL. Error messages are generic
    because a `requests` error message would contain the request URL, which includes the address.
  * No Streamlit in here, so the functions can be tested on their own with fake (mocked) responses.
"""

import re
import time
from datetime import date

import requests

GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
NFHL_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
ZONE_LAYER = 28           # "Flood Hazard Zones" (checked against MapServer?f=json, 2026-11-13)
JURISDICTION_LAYER = 22   # "Political Jurisdictions": the NFIP community ID (CID) at a point
TIMEOUT = 6               # seconds per try
RETRIES = 2               # extra tries after the first one
STATE_FIPS = {"51": "VA", "37": "NC", "24": "MD"}
STATE_NAMES = {"VA": "Virginia", "NC": "North Carolina", "MD": "Maryland"}
FIRST_FIRM_RULE_DATE = date(1975, 1, 1)   # post-FIRM: built after Dec 31, 1974 ... (see firm_status)
_pause = time.sleep       # the wait between retries (tests replace this, never time.sleep itself)


def _get_json(url, params, timeout=TIMEOUT, retries=RETRIES):
    """GET a JSON answer, retrying on timeouts, connection errors and server errors (5xx).
    Returns the parsed JSON, or None if the service didn't answer properly. Never raises."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                raise requests.HTTPError("server error")
            if r.status_code != 200:
                return None                       # 4xx: asking again won't help
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt < retries:
                _pause(0.4 * (attempt + 1))      # a short pause before trying again
    return None


# ---------------------------------------------------------------------------
# 1. Geocoding: address -> coordinates, state, county
# ---------------------------------------------------------------------------
NOT_FOUND = ("We couldn't find that address. Try adding the ZIP code, or a nearby house number on the same "
             "street (the Census list doesn't have every house number). You can also pick the county from the "
             "list, or choose the inputs yourself.")
# Apartment / unit parts the Census Geocoder often can't match: "Apt 4B", "Unit 2", "Suite 300", "#12"
_UNIT = re.compile(r"(?:,\s*|\s+)(?:apt|apartment|unit|suite|ste|bldg|building|fl|floor|rm|room|lot|trlr)\.?\s*#?\s*[\w-]+"
                   r"|\s*#\s*[\w-]+", re.IGNORECASE)


def simpler_address(address):
    """One automatic retry variant (Milestone 16): the address without any apartment/unit part and
    without stray punctuation. Returns None if that changes nothing (so there's nothing new to try)."""
    s = _UNIT.sub("", address)
    s = re.sub(r"[^\w\s,-]", " ", s)                  # periods, quotes, etc.
    s = re.sub(r"\s*,\s*", ", ", " ".join(s.split())).strip(" ,")
    s = re.sub(r"(, )+", ", ", s)
    return s if s and s != address else None



def pretty_address(matched):
    """The Census matched address in readable case, keeping the state code in capitals:
    '1000 BANK ST, RICHMOND, VA, 23219' -> '1000 Bank St, Richmond, VA, 23219' (Milestone 16)."""
    parts = [p.strip() for p in str(matched or "").split(",")]
    out = []
    for part in parts:
        if len(part) == 2 and part.isalpha():
            out.append(part.upper())                      # a state code: VA, NC, MD, ...
        else:
            out.append(part.title())
    return ", ".join(out)


def geocode(address):
    """Find a US street address with the Census Geocoder. If nothing matches, it tries once more
    with a simpler version of the address (no apartment/unit part, no stray punctuation).

    Returns {"status": "ok", "lat", "lon", "state" ("VA"/"NC"/"MD"), "county_fips", "county_name",
    "place" (incorporated town/city or None), "matched" (the address as the Census matched it)},
    or {"status": "empty" | "no_match" | "outside" | "error", "message": ...}."""
    address = " ".join(str(address or "").split())[:200]
    if len(address) < 5:
        return {"status": "empty", "message": "Please type a street address, city and state."}
    res = _geocode_once(address)
    if res["status"] == "no_match":
        simpler = simpler_address(address)
        if simpler and len(simpler) >= 5:
            again = _geocode_once(simpler)
            if again["status"] != "no_match":
                return again
    return res


def _geocode_once(address):
    """One question to the Census Geocoder (see geocode)."""
    data = _get_json(GEOCODER_URL, {"address": address, "benchmark": "Public_AR_Current",
                                    "vintage": "Current_Current", "format": "json",
                                    "layers": "States,Counties,Incorporated Places"})
    if data is None:
        return {"status": "error", "message": "The Census address service didn't answer. Please try "
                                              "again later, or choose the inputs yourself."}
    try:
        matches = data["result"]["addressMatches"]
    except (KeyError, TypeError):
        return {"status": "error", "message": "The Census address service sent an answer we couldn't read."}
    if not matches:
        return {"status": "no_match", "message": NOT_FOUND}
    m = matches[0]
    try:
        geo = m["geographies"]
        st_fips = geo["States"][0]["STATE"]
        county = geo["Counties"][0]
        lon, lat = float(m["coordinates"]["x"]), float(m["coordinates"]["y"])
    except (KeyError, IndexError, TypeError, ValueError):
        return {"status": "error", "message": "The Census address service sent an answer we couldn't read."}
    if st_fips not in STATE_FIPS:
        name = geo["States"][0].get("NAME", "another state")
        return {"status": "outside", "message": f"That address is in {name}. This tool only covers "
                                                "Virginia, North Carolina and Maryland."}
    places = geo.get("Incorporated Places") or []
    return {"status": "ok", "lat": lat, "lon": lon, "state": STATE_FIPS[st_fips],
            "county_fips": county["GEOID"], "county_name": county.get("NAME", ""),
            "place": places[0].get("BASENAME") if places else None,
            "matched": m.get("matchedAddress", "")}


# ---------------------------------------------------------------------------
# 2. Flood zone at a point (FEMA NFHL)
# ---------------------------------------------------------------------------
def _point_query(layer, lat, lon, fields):
    return _get_json(f"{NFHL_URL}/{layer}/query", {
        "geometry": f"{lon:.6f},{lat:.6f}", "geometryType": "esriGeometryPoint", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects", "outFields": fields, "returnGeometry": "false",
        "f": "json"})


def zone_group(fld_zone, subtype=None):
    """Map an NFHL zone code to the model's two groups.
    A, AE, AH, AO, AR, A99, V, VE ... -> 'A/V high risk'; X (shaded or not), B, C -> 'B/C/X
    moderate-low risk'; D (possible but undetermined hazard), 'AREA NOT INCLUDED', 'OPEN WATER' or
    nothing -> None (the user chooses)."""
    z = (fld_zone or "").strip().upper()
    if not z or z in ("D", "AREA NOT INCLUDED", "OPEN WATER", "NP"):
        return None
    if z in ("X", "B", "C") or z.startswith("X"):
        return "B/C/X moderate-low risk"
    if re.fullmatch(r"(A|V)[A-Z0-9]*", z):
        return "A/V high risk"
    return None


def describe_zone(fld_zone, subtype=None):
    """Plain words for a zone code, e.g. 'AE (high risk)', 'X, shaded (moderate risk)'."""
    z = (fld_zone or "").strip().upper()
    sub = (subtype or "").upper()
    if z.startswith("V"):
        return f"{z} (high risk, coastal with waves)"
    if z.startswith("A") and z != "AREA NOT INCLUDED":
        return f"{z} (high risk)"
    if z == "X" and "0.2 PCT" in sub:
        return "X, shaded (moderate risk)"
    if z in ("X", "C"):
        return f"{z} (low risk)"
    if z == "B":
        return "B (moderate risk)"
    if z == "D":
        return "D (possible flood risk that hasn't been studied)"
    return z.title() if z else "none"


def flood_zone(lat, lon):
    """Flood zone at a point. Returns {"status": "ok", "zone_group", "fld_zone", "subtype", "label"}
    or {"status": "unmapped" | "error", "message"}. Where several zones touch the point, the riskiest
    (A/V) one is used."""
    data = _point_query(ZONE_LAYER, lat, lon, "FLD_ZONE,ZONE_SUBTY")
    if data is None or "error" in data:
        return {"status": "error", "message": "FEMA's flood map service didn't answer. Please choose the "
                                              "flood zone yourself."}
    feats = [f.get("attributes", {}) for f in data.get("features", [])]
    if not feats:
        return {"status": "unmapped", "message": "FEMA's flood map has no zone at this spot (it may not be "
                                                 "mapped yet). Please choose the flood zone yourself."}
    feats.sort(key=lambda a: zone_group(a.get("FLD_ZONE")) != "A/V high risk")   # riskiest first
    a = feats[0]
    grp = zone_group(a.get("FLD_ZONE"), a.get("ZONE_SUBTY"))
    label = describe_zone(a.get("FLD_ZONE"), a.get("ZONE_SUBTY"))
    if grp is None:
        return {"status": "unmapped", "fld_zone": a.get("FLD_ZONE"), "label": label,
                "message": f"FEMA's map shows zone {label} here, which the model doesn't use. Please "
                           "choose the flood zone yourself."}
    return {"status": "ok", "zone_group": grp, "fld_zone": a.get("FLD_ZONE"), "subtype": a.get("ZONE_SUBTY"),
            "label": label}


# ---------------------------------------------------------------------------
# 3. Community and its first flood insurance rate map (FIRM) date
# ---------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^A-Z0-9 ]", "", str(s or "").upper()).strip()


def nice_name(name):
    """'RICHMOND, CITY OF' -> 'City of Richmond'; 'DARE COUNTY' -> 'Dare County'."""
    name = " ".join(str(name).split())
    if ", " in name:
        main, kind = name.split(", ", 1)
        name = f"{kind} {main}"
    return name.title().replace(" Of ", " of ")


def community(lat, lon, state, county_name, place, firm_table):
    """The NFIP community at a point and its initial FIRM date.

    First choice: FEMA's NFHL 'Political Jurisdictions' layer gives the community ID (CID) at the
    point, looked up in the bundled Community Status Book table. Fallback: match the geocoder's town /
    city, or else the county's unincorporated area, by name within the county.
    firm_table: rows with cid, state, community, county, initial_firm (a pandas DataFrame).
    Returns {"status": "ok", "cid", "name", "initial_firm" (date or None), "how"} or
    {"status": "not_found", "message"}."""
    def row_out(r, how):
        d = r["initial_firm"]
        try:
            d = date.fromisoformat(str(d)) if d and str(d) not in ("nan", "NaT", "None") else None
        except ValueError:
            d = None
        return {"status": "ok", "cid": str(r["cid"]), "name": nice_name(r["community"]), "initial_firm": d,
                "how": how}

    data = _point_query(JURISDICTION_LAYER, lat, lon, "CID,POL_NAME1")
    if data and "error" not in data:
        for f in data.get("features", []):
            cid = str(f.get("attributes", {}).get("CID") or "").strip()
            hit = firm_table[firm_table["cid"].astype(str) == cid]
            if cid and len(hit):
                return row_out(hit.iloc[0], "community ID from FEMA's flood map")
    # Fallback: names within the county (Census county names vs Community Status Book county names)
    sub = firm_table[(firm_table["state"] == state) & (firm_table["county"].map(_norm) == _norm(county_name))]
    if place:
        p = _norm(place)
        hit = sub[sub["community"].map(_norm).str.startswith(p + " ")
                  | (sub["community"].map(_norm) == p)]
        hit = hit[~hit["community"].str.upper().str.contains("COUNTY")] if len(hit) > 1 else hit
        if len(hit) == 1:
            return row_out(hit.iloc[0], "matched by town or city name")
    else:
        hit = sub[sub["community"].map(_norm) == _norm(county_name)]
        if len(hit) == 1:
            return row_out(hit.iloc[0], "matched by county name")
    return {"status": "not_found", "message": "We couldn't tell which flood insurance community this address "
                                              "is in, so we can't check the date of its first flood map."}


def firm_status(year_built, firm_date):
    """Pre- or post-FIRM from the year built, using FEMA's rule: a building is post-FIRM if it was
    built (construction started) after December 31, 1974, or on or after the date of the community's
    initial FIRM, whichever is later.

    With only a YEAR we can be sure except in the cutoff year itself (unless the cutoff is Jan 1).
    Returns {"status": "post" | "pre" | "unsure", "cutoff": date or None, "rule": text}."""
    if firm_date is None:
        return {"status": "unsure", "cutoff": None,
                "rule": "We couldn't find the date of this area's first flood insurance rate map."}
    cutoff = max(FIRST_FIRM_RULE_DATE, firm_date)
    rule = (f"FEMA's rule: post-FIRM means built after December 31, 1974, or on or after the area's first "
            f"flood insurance rate map ({firm_date.strftime('%B')} {firm_date.day}, {firm_date.year}), "
            f"whichever is later. Here that means built on or after "
            f"{cutoff.strftime('%B')} {cutoff.day}, {cutoff.year}.")
    if year_built is None:
        return {"status": "unsure", "cutoff": cutoff, "rule": rule + " Enter the year built to apply it."}
    y = int(year_built)
    if y > cutoff.year or (y == cutoff.year and cutoff.month == 1 and cutoff.day == 1):
        return {"status": "post", "cutoff": cutoff, "rule": rule}
    if y < cutoff.year:
        return {"status": "pre", "cutoff": cutoff, "rule": rule}
    return {"status": "unsure", "cutoff": cutoff,
            "rule": rule + f" A building from {y} could be either, so please choose."}


def year_to_band(year):
    """Year built -> the model's age group, or None if the year is missing or not believable."""
    if year is None:
        return None
    y = int(year)
    if y < 1700 or y > date.today().year + 1:
        return None
    if y < 1960:
        return "before 1960"
    if y < 1980:
        return "1960-1979"
    if y < 2000:
        return "1980-1999"
    return "2000 or later"
