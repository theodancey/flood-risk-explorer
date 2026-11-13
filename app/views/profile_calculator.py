"""
Page 2: Property Profile Calculator.

Describe a TYPE of insured building (flood zone, age, occupancy, ...) in a state or county and see
the model's expected claim frequency, claim size and annual loss per policy-year for the average
building of that type, plus a waterfall showing each characteristic's effect, starting from the
state average. It describes a group of similar insured buildings, never one specific property.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

ANY = "__any__"     # "anywhere in the state": the state's mix of counties

st.title("Property Profile Calculator")
st.caption("Expected flood losses for the average insured building with the characteristics you choose.")
C.disclaimer()

ex = C.explainer()
bases = ex.t.d["bases"]

# Options in a natural reading order, with plain-English labels (values are the model's levels)
ZONE = {"A/V high risk": "A/V: high-risk zone", "B/C/X moderate-low risk": "B/C/X: moderate or low risk"}
BUILT = {"before 1960": "Before 1960", "1960-1979": "1960-1979", "1980-1999": "1980-1999",
         "2000 or later": "2000 or later"}
OCC = {"single-unit residential": "Single-unit home", "multi-unit residential": "Multi-unit residential (2+ units)",
       "non-residential": "Non-residential (business, etc.)"}
COVER = {"$0 (contents only)": "$0 (contents-only policy)", "under $100K": "Under $100K",
         "$100K-$249K": "$100K-$249K", "$250K or more": "$250K or more (the NFIP maximum for homes)"}
FLOORS = {"1": "1", "2": "2", "3+": "3 or more"}
YES_NO = {"True": "Yes", "False": "No"}
FIRM = {"unknown": "Not sure / skip", "False": "No: pre-FIRM", "True": "Yes: post-FIRM"}

# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------
state = st.segmented_control("State", list(C.STATES), format_func=C.STATES.get, default="VA",
                             required=True, key="pc_state")
fips_list, names = C.counties_in(state)
county = st.selectbox("County or independent city (type to search)", [ANY] + fips_list,
                      format_func=lambda f: f"Anywhere in {C.STATES[state]} (state mix of counties)"
                      if f == ANY else names[f], key="pc_county",
                      help="With no county factor in the model, the county changes the result only through "
                           "its state. Choosing a county also shows that county's certainty label.")

# ---------------------------------------------------------------------------
# The building type
# ---------------------------------------------------------------------------
st.markdown("#### Building characteristics")
c1, c2 = st.columns(2)
with c1:
    zone = st.radio("Flood zone", list(ZONE), format_func=ZONE.get, key="pc_zone",
                    index=list(ZONE).index(bases["flood_zone_group"]),
                    help="FEMA's flood maps put each building in a zone. A and V zones are high-risk "
                         "('Special Flood Hazard Areas', at least a 1% chance of flooding each year; V = coastal "
                         "with waves). B, C and X are moderate- or low-risk zones outside them.")
    built = st.selectbox("Year built", list(BUILT), format_func=BUILT.get, key="pc_built",
                         index=list(BUILT).index(bases["construction_band"]),
                         help="Older buildings claim far more often: many were built before flood-safe "
                              "building rules (1960-79 is the riskiest band in the data).")
    firm = st.radio("Built after the area's first flood map (post-FIRM)?", list(FIRM), format_func=FIRM.get,
                    key="pc_firm", help="A FIRM is a Flood Insurance Rate Map. 'Post-FIRM' buildings were built "
                                        "after the community's first map, so under flood-safe building rules. "
                                        "The model doesn't use this directly (year built and flood zone carry "
                                        "its effect); it's only checked for consistency with year built.")
    elevated = st.radio("Elevated building?", list(YES_NO), format_func=YES_NO.get, horizontal=True,
                        key="pc_elev", index=list(YES_NO).index(bases["elevated"]),
                        help="Raised on piles, piers, columns or walls so the lowest floor is above "
                             "expected flood levels. Elevated buildings have smaller claims.")
with c2:
    occ = st.selectbox("Occupancy", list(OCC), format_func=OCC.get, key="pc_occ",
                       index=list(OCC).index(bases["occupancy_group"]),
                       help="What the building is used for. Apartment and business buildings have much "
                            "bigger claims than single homes.")
    cover = st.selectbox("Building coverage", list(COVER), format_func=COVER.get, key="pc_cover",
                         index=list(COVER).index(bases["building_coverage_band"]),
                         help="The policy's building coverage limit: the most the policy pays for the "
                              "structure. It also stands in for building value (bigger, pricier buildings "
                              "buy more cover), so it is not purely causal.")
    floors = st.segmented_control("Floors", list(FLOORS), format_func=FLOORS.get, key="pc_floors",
                                  default=bases["floors_group"], required=True,
                                  help="Number of floors. Barely changes expected losses overall.")
    nonres = occ == "non-residential"
    primary = st.radio("Primary residence?", list(YES_NO), format_func=YES_NO.get, horizontal=True,
                       key="pc_primary", index=list(YES_NO).index(bases["primary_residence"]),
                       disabled=nonres,
                       help="Is it someone's main home (vs a second home, rental or holiday home)? "
                            "Only applies to residential buildings.")
    if nonres:
        primary = "False"
        st.caption("Set to **No**: a non-residential building isn't anyone's primary residence.")

features = {"flood_zone_group": zone, "construction_band": built, "occupancy_group": occ,
            "building_coverage_band": cover, "elevated": elevated, "floors_group": floors,
            "primary_residence": primary}
if firm != "unknown":
    features["post_firm"] = firm

# ---------------------------------------------------------------------------
# Checks on the combination (warnings, never errors)
# ---------------------------------------------------------------------------
port = ex.portfolio
in_state = port[port["state"] == state]
if firm != "unknown":
    band = in_state[in_state["construction_band"] == built]
    share = 100 * band.loc[band["post_firm"] == firm, "exposure"].sum() / band["exposure"].sum()
    if share < 2:
        st.warning(f"Unusual combination: only {share:.1f}% of insured buildings in {C.STATES[state]} built "
                   f"{BUILT[built].lower()} are {'post' if firm == 'True' else 'pre'}-FIRM (most first flood maps "
                   "date from the 1970s-80s). Check the year built. The estimate below uses year built, not "
                   "post-FIRM.", icon=":material/warning:")
where = in_state if county == ANY else in_state[in_state["county"] == county]
mask = None
for f in C.explain.FEATURES:                 # insured buildings matching all 7 model characteristics
    m = where[f].astype(str) == features[f]
    mask = m if mask is None else (mask & m)
same = where.loc[mask, "exposure"].sum()
where_name = C.STATES[state] if county == ANY else names[county]
if same < 50:
    st.info(f"Few insured buildings exactly like this in {where_name} ({same:,.0f} policy-years in 2010-2025). "
            "The estimate still works, because the model combines each characteristic's effect learned from "
            "all three states, but there's little local data on this exact type.", icon=":material/info:")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
try:
    res = C.profile_result(state, None if county == ANY else county, tuple(sorted(features.items())))
except ValueError as e:                      # e.g. a level the model doesn't know
    st.error(f"Sorry, this profile couldn't be rated: {e}")
    st.stop()

st.markdown("#### Result")
sname = res["state_name"]
r = res["ratio_to_state"]
m1, m2, m3 = st.columns(3)
m1.metric("Expected annual loss", C.money(res["pure_premium"]),
          delta=f"{100 * (r - 1):+.0f}% vs {sname} avg", delta_color="inverse" if abs(r - 1) >= 0.1 else "off",
          help="Pure premium: expected claim cost per policy-year = frequency × severity (× small loads), "
               "average year, 2025 dollars. Not a premium or quote: real prices add expenses, catastrophe "
               "loads and NFIP rules.")
m2.metric("Claim frequency", C.per_thousand(res["frequency"]),
          help=f"Expected paid claims per 1,000 policy-years in an average year ({C.one_in(res['frequency'])}).")
m3.metric("Claim severity", C.money(res["severity"]),
          help="Expected size of one paid claim (building + contents), 2025 dollars.")
st.markdown(res["sentence"])
if county != ANY:
    C.certainty_badge(res["certainty"])
    st.caption(f"County certainty for {names[county]}: {C.CERTAINTY_STYLE[res['certainty']][2]} "
               "See the County Explorer for details.")

st.markdown(f"#### Each characteristic's effect, starting from the {sname} average")
C.show_chart(C.waterfall_figure(res["waterfall"]), key="pc_wf")
st.caption(f"Starts at the average expected loss over all insured buildings in {sname} "
           f"({C.money(res['state_average_pure_premium'])}). Each bar is one characteristic's effect "
           "(orange = raises, blue = lowers; hover for the multiplier). The effects are Shapley values: each "
           "one is averaged over every order of adding the characteristics, so the order of the bars "
           "doesn't matter and they multiply exactly to the result.")
with st.expander("Show the numbers"):
    wf = res["waterfall"]
    tbl = wf.assign(**{"Multiplier": wf["multiplier"].map(lambda v: "" if v != v else f"×{v:.2f}"),
                       "Running total": wf["after"].map(C.money)})[["step", "Multiplier", "Running total"]]
    st.dataframe(tbl.rename(columns={"step": "Step"}), hide_index=True)
    for note in res["notes"]:
        st.caption(note)
