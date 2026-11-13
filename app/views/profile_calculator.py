"""
Page 2: Property Profile Calculator.

Describe a TYPE of insured building (flood zone, age, use, ...) in a state or county and see the
model's expected yearly flood loss for the average building of that type, how often such buildings
claim and how big a claim is, plus a waterfall showing each characteristic's effect, starting from
the state average. It describes a group of similar insured buildings, never one specific property.
Milestone 15: plain wording, a fairer comparison (state average AND a typical lower-risk building),
the address lookup and the "what if?" mitigation section.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

ANY = "__any__"     # "anywhere in the state": the state's mix of counties

st.title("Property Profile Calculator")
st.caption("Describe a kind of building. See the average yearly flood loss for insured buildings like it.")
C.disclaimer()

ex = C.explainer()
bases = ex.t.d["bases"]

# Options in a natural reading order, with plain-English labels (values are the model's levels)
ZONE = {"A/V high risk": "High risk (zone A or V)", "B/C/X moderate-low risk": "Lower risk (zone B, C or X)"}
BUILT = {"before 1960": "Before 1960", "1960-1979": "1960-1979", "1980-1999": "1980-1999",
         "2000 or later": "2000 or later"}
OCC = {"single-unit residential": "Single-family home", "multi-unit residential": "Building with 2 or more homes",
       "non-residential": "Business or other non-home building"}
COVER = {"$0 (contents only)": "$0 (belongings only)", "under $100K": "Under $100K",
         "$100K-$249K": "$100K-$249K", "$250K or more": "$250K or more (the most a home policy allows)"}
FLOORS = {"1": "1", "2": "2", "3+": "3 or more"}
YES_NO = {"True": "Yes", "False": "No"}
FIRM = {"unknown": "Not sure / skip", "False": "No (built before)", "True": "Yes (built after)"}

# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------
state = st.segmented_control("State", list(C.STATES), format_func=C.STATES.get, default="VA",
                             required=True, key="pc_state")
fips_list, names = C.counties_in(state)
county = st.selectbox("County or independent city (type to search)", [ANY] + fips_list,
                      format_func=lambda f: f"Anywhere in {C.STATES[state]} (state mix of counties)"
                      if f == ANY else names[f], key="pc_county",
                      help="The estimate depends on the state, not the exact county. Choosing a county also "
                           "shows how much local claims data there is.")

# ---------------------------------------------------------------------------
# The building type
# ---------------------------------------------------------------------------
st.markdown("#### About the building")
c1, c2 = st.columns(2)
with c1:
    zone = st.radio("Flood zone", list(ZONE), format_func=ZONE.get, key="pc_zone",
                    index=list(ZONE).index(bases["flood_zone_group"]),
                    help="FEMA's flood maps put each building in a zone. Zones A and V are high risk: at least "
                         "a 1-in-100 chance of flooding each year (V = on the coast, with waves). Zones B, C and X "
                         "are lower risk.")
    built = st.selectbox("Year built", list(BUILT), format_func=BUILT.get, key="pc_built",
                         index=list(BUILT).index(bases["construction_band"]),
                         help="Older buildings claim far more often. Many were built before flood-safe "
                              "building rules. Buildings from 1960-1979 have the most claims in the data.")
    firm = st.radio("Built after the area's first flood map?", list(FIRM), format_func=FIRM.get,
                    key="pc_firm", help="Insurers call this 'post-FIRM' (FIRM = Flood Insurance Rate Map). "
                                        "Buildings from after the area's first flood map had to follow "
                                        "flood-safe building rules. The estimate doesn't use this answer "
                                        "directly (year built covers it); it's only used to check your answers.")
    elevated = st.radio("Raised building?", list(YES_NO), format_func=YES_NO.get, horizontal=True,
                        key="pc_elev", index=list(YES_NO).index(bases["elevated"]),
                        help="Raised on posts, piers or walls so the lowest floor sits above expected flood "
                             "water. Raised buildings tend to have smaller claims.")
with c2:
    occ = st.selectbox("What is it used for?", list(OCC), format_func=OCC.get, key="pc_occ",
                       index=list(OCC).index(bases["occupancy_group"]),
                       help="Apartment and business buildings have much bigger claims than single homes.")
    cover = st.selectbox("Building coverage", list(COVER), format_func=COVER.get, key="pc_cover",
                         index=list(COVER).index(bases["building_coverage_band"]),
                         help="The most the policy pays to repair the building itself. Bigger, pricier "
                              "buildings usually buy more, so this also hints at the building's value.")
    floors = st.segmented_control("Floors", list(FLOORS), format_func=FLOORS.get, key="pc_floors",
                                  default=bases["floors_group"], required=True,
                                  help="Number of floors. This barely changes the estimate.")
    nonres = occ == "non-residential"
    primary = st.radio("Someone's main home?", list(YES_NO), format_func=YES_NO.get, horizontal=True,
                       key="pc_primary", index=list(YES_NO).index(bases["primary_residence"]),
                       disabled=nonres,
                       help="A main home, not a second home, rental or vacation home. Only for homes.")
    if nonres:
        primary = "False"
        st.caption("Set to **No**: a business building isn't anyone's main home.")

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
                   f"{BUILT[built].lower()} were built {'after' if firm == 'True' else 'before'} the area's first "
                   "flood map (most first maps are from the 1970s-80s). Please check the year built. The estimate "
                   "below uses the year built.", icon=":material/warning:")
where = in_state if county == ANY else in_state[in_state["county"] == county]
mask = None
for f in C.explain.FEATURES:                 # insured buildings matching all 7 model characteristics
    m = where[f].astype(str) == features[f]
    mask = m if mask is None else (mask & m)
same = where.loc[mask, "exposure"].sum()
where_name = C.STATES[state] if county == ANY else names[county]
if same < 50:
    st.info(f"Few insured buildings exactly like this in {where_name} (about {same:,.0f} building-years in "
            "2010-2025). The estimate still works: the model learns each feature's effect from all three "
            "states. But there's little local data on this exact kind of building.", icon=":material/info:")

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
pp, r = res["pure_premium"], res["ratio_to_state"]
low = C.zone_average(state, "B/C/X moderate-low risk")
with st.container(border=True, key="hero"):
    m1, m2 = st.columns([1.25, 1])
    m1.metric("Expected yearly loss", C.money(pp),
              delta=f"{C.ratio_words(r, sname)} ({C.money(res['state_average_pure_premium'])})",
              delta_color="inverse" if abs(r - 1) >= 0.1 else "off",
              delta_arrow="auto" if abs(r - 1) >= 0.1 else "off",
              help="The average flood claim cost per year for insured buildings like this, in a typical year "
                   "of storms, in 2025 dollars. It is not a price or a quote: real prices add costs, storm "
                   "cushions and NFIP rules.")
    rl = pp / low
    m2.metric("Typical lower-risk insured building", C.money(low),
              delta=(f"Yours is {rl:.1f} times this" if rl >= 1.1 else
                     f"Yours is {100 * (1 - rl):.0f}% lower" if rl <= 1 / 1.1 else "About the same as yours"),
              delta_color="off", delta_arrow="off",
              help=f"The average insured building in a lower-risk flood zone (B, C or X) in {sname}.")
    st.caption(f"About {C.av_share(state):.0f}% of insured buildings in {sname} are in high-risk flood zones, "
               f"so the {sname} average ({C.money(res['state_average_pure_premium'])}) is already high. "
               "That's why a typical lower-risk building is shown too.")
st.markdown(C.profile_summary(res))
m3, m4 = st.columns(2)
m3.metric("Chance of a claim each year", C.per_thousand(res["frequency"]),
          help=f"Expected paid claims per 1,000 insured buildings like this in a typical year ({C.one_in(res['frequency'])}).")
m4.metric("Typical claim size", C.money(res["severity"]),
          help="Expected size of one paid claim (building + belongings), 2025 dollars.")
if county != ANY:
    crow = C.county_summary().set_index("county").loc[county]
    st.markdown(f"**Local data for {names[county]}**")
    C.show_certainty(crow["Z_sev"], crow["paid_claims_2010_2025"], crow["history_vs_model"], names[county])

st.markdown(f"#### What raises or lowers it, starting from the {sname} average")
C.show_chart(C.waterfall_figure(res["waterfall"]), key="pc_wf")
st.caption(f"The first bar is the average for all insured buildings in {sname} "
           f"({C.money(res['state_average_pure_premium'])}). Each bar after it is one feature you chose. "
           "Orange raises the estimate, blue lowers it.")
with st.expander("Learn more: how the bars are worked out"):
    st.markdown(
        "Features overlap (for example, older buildings are more often in high-risk zones), so there's no "
        "single right order to add them. Each bar is a **fair-share split**: the feature's effect averaged "
        "over every order of adding the features. Statisticians call these **Shapley values**. The bars don't "
        "depend on their order, and they multiply exactly to the result. Hover a bar for its multiplier "
        "(×1.20 = 20% higher). Features that change the estimate by less than 2% are grouped as "
        "**Other small effects**.")
    wf = C.group_small(res["waterfall"])
    tbl = wf.assign(**{"Multiplier": wf["multiplier"].map(lambda v: "" if v != v else f"×{v:.2f}"),
                       "Running total": wf["after"].map(C.money)})[["label", "Multiplier", "Running total"]]
    st.dataframe(tbl.rename(columns={"label": "Step"}), hide_index=True)
    if res["notes"]:
        st.caption("Your answer to \"built after the first flood map\" isn't used in the estimate: the year "
                   "built and flood zone already carry its effect.")

C.footer()
