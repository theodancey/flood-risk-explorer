"""
Page 1: County Explorer.

Pick a state and a county to see the model's expected annual flood loss per insured building
(an area-level average), how it compares with the state, how certain it is, the county's mix of
insured buildings, its own claims history, and WHY it differs from the state average.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

st.title("County Explorer")
st.caption("Expected flood insurance losses for the insured buildings in a county, compared with its state.")
C.disclaimer()

# ---------------------------------------------------------------------------
# Inputs: state, then county (the county list is searchable: click it and type)
# ---------------------------------------------------------------------------
state = st.segmented_control("State", list(C.STATES), format_func=C.STATES.get, default="VA",
                             required=True, key="ce_state")
fips_list, names = C.counties_in(state)
default_fips = C.biggest_county(state)
fips = st.selectbox("County or independent city (type to search)", fips_list,
                    index=fips_list.index(default_fips), format_func=lambda f: names[f], key="ce_county",
                    help="Virginia's independent cities (e.g. Norfolk city) are counted separately from "
                         "counties, as in Census and FEMA data.")

cs = C.county_summary()
row = cs[cs["county"] == fips].iloc[0]
res = C.county_result(fips)
sname = C.STATES[state]

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------
st.subheader(f"{row['name']}, {state}")
c1, c2 = st.columns([1.3, 1])
with c1:
    st.metric("Expected annual loss per policy-year", C.money(res["pure_premium"]),
              help="The model's expected flood claim cost per insured building per year, in an average "
                   "year, in 2025 dollars, for this county's mix of insured buildings. Actuaries call this "
                   "the pure premium: the part of an insurance price that pays for claims, before expenses.")
with c2:
    r = res["ratio_to_state"]
    st.metric(f"vs the {sname} average", C.ratio(r),
              delta=f"{100 * (r - 1):+.0f}% vs {C.money(res['state_average_pure_premium'])}",
              delta_color="inverse" if abs(r - 1) >= 0.1 else "off",
              help="Ratio of the county's expected loss per policy-year to its state's average. "
                   "Above 1 = riskier than the state's typical insured building.")
c3, c4, c5 = st.columns(3)
c3.metric("Claim frequency", C.per_thousand(res["frequency"]),
          help=f"Expected paid claims per 1,000 insured policy-years in an average year ({C.one_in(res['frequency'])}).")
c4.metric("Claim severity", C.money(res["severity"]),
          help="Expected size of a paid claim (building + contents), 2025 dollars.")
c5.metric("Insured policy-years", f"{res['policy_years']:,.0f}",
          help="2010-2025 total. One policy-year = one building insured for one year.")

st.markdown(row["summary"])

# ---------------------------------------------------------------------------
# Certainty
# ---------------------------------------------------------------------------
st.markdown("#### How certain is this?")
C.certainty_badge(row["certainty"])
st.markdown(f"**{row['certainty']}** means {C.CERTAINTY_STYLE[row['certainty']][2]}")
with st.expander("Why this label?"):
    st.markdown(row["uncertainty_note"])
    st.caption("Rule: High = credibility Z ≥ 0.8 and the county's 2010-2025 claims within 0.8-1.25× what "
               "the model expects; Low = no paid claims, Z < 0.5, or history outside 0.5-2.0×; Medium = "
               "everything else. Credibility Z (0 to 1) measures how much weight the county's own claim "
               "history would deserve, given how many claims it has (Bühlmann-Straub method).")

# ---------------------------------------------------------------------------
# Property mix
# ---------------------------------------------------------------------------
st.markdown("#### What kind of buildings are insured here?")
MIX = [("pct_av_zone", "In a high-risk A/V flood zone"), ("pct_pre_firm", "Built before the first flood map (pre-FIRM)"),
       ("pct_built_pre_1980", "Built before 1980"), ("pct_elevated", "Elevated"),
       ("pct_multi_or_nonres", "Multi-unit or non-residential"), ("pct_not_primary", "Not a primary residence")]
labels = [lab for _, lab in MIX]
county_vals = [res["mix"][k] for k, _ in MIX]
state_vals = [res["state_mix"][k] for k, _ in MIX]
fig = go.Figure()
fig.add_bar(y=labels, x=county_vals, name=row["name"], orientation="h", marker_color=C.BLUE,
            text=[C.pct(v) for v in county_vals], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + row["name"] + ": %{x:.1f}%<extra></extra>")
fig.add_bar(y=labels, x=state_vals, name=f"{sname} (all insured)", orientation="h", marker_color=C.LIGHT_GREY,
            text=[C.pct(v) for v in state_vals], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + sname + ": %{x:.1f}%<extra></extra>")
C._base_layout(fig, height=380)
fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                  legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
fig.update_xaxes(range=[0, 118], ticksuffix="%", title="Share of insured policy-years, 2010-2025")
C.show_chart(fig, key="mix")
st.caption("A/V zones are FEMA's high-risk flood zones (V = coastal with waves). \"Pre-FIRM\" buildings were "
           "built before the community's first Flood Insurance Rate Map, so usually without flood-safe "
           "building rules.")

# ---------------------------------------------------------------------------
# Claims history
# ---------------------------------------------------------------------------
st.markdown("#### This county's own claims history")
h = C.county_history()
h = h[h["county"] == fips].sort_values("year")
events = C.MAJOR_EVENTS[state]
if h["paid_claims"].sum() == 0:
    st.info(f"{row['name']} had no paid NFIP flood claims in 2010-2025 "
            f"({h['policy_years'].sum():,.0f} insured policy-years). That's why its certainty is Low: there's "
            "no local history to check the estimate against.", icon=":material/info:")
else:
    lc = (h["losses_2025"] / h["policy_years"]).where(h["policy_years"] > 0, 0)
    fig = go.Figure(go.Bar(
        x=h["year"], y=h["losses_2025"], marker_color=C.BLUE,
        customdata=list(zip(h["paid_claims"], h["policy_years"], lc)),
        hovertemplate="%{x}<br>Paid losses: $%{y:,.0f}<br>Paid claims: %{customdata[0]:,}<br>"
                      "Policy-years: %{customdata[1]:,.0f}<br>Loss per policy-year: $%{customdata[2]:,.0f}"
                      "<extra></extra>"))
    top = h["losses_2025"].max()
    for yr, name in events.items():
        val = h.loc[h["year"] == yr, "losses_2025"]
        if len(val):
            fig.add_annotation(x=yr, y=float(val.iloc[0]), text=name, showarrow=False, yshift=12,
                               font=dict(size=11, color=C.GREY), textangle=-30 if len(name) > 10 else 0)
    C._base_layout(fig, height=340)
    fig.update_xaxes(dtick=1, tickangle=-45, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title="Paid losses (2025 $)", tickprefix="$", range=[0, top * 1.3 if top > 0 else 1])
    C.show_chart(fig, key="history")
    actual = row["actual_loss_cost_2010_2025"]
    hv = row["history_vs_model"]
    st.markdown(
        f"Over 2010-2025 this county had **{int(row['paid_claims_2010_2025']):,} paid claims** and an actual "
        f"loss cost of **{C.money(actual)} per policy-year**, vs the model's {C.money(res['pure_premium'])} "
        f"for an average year." + (f" Allowing for each year's storms, its claims cost about **{hv:.2f}×** "
                                   "what the model expects." if hv == hv else ""))
    st.caption(f"Labels mark the major flood events behind {sname}'s worst years "
               f"({', '.join(f'{y} {n}' for y, n in events.items())}). A few storm years dominate most "
               "counties' history, so actual history is noisy.")

# ---------------------------------------------------------------------------
# Why it differs
# ---------------------------------------------------------------------------
st.markdown(f"#### Why it differs from the {sname} average")
C.show_chart(C.waterfall_figure(res["waterfall"]), key="why")
st.caption(
    f"Each bar shows how much one part of the county's **property mix** moves the expected loss from the "
    f"{sname} average (orange = raises, blue = lowers; hover for the multiplier). \"Overlap and interactions\" "
    "is the part that comes from how the characteristics combine. There is **no county factor**: the price "
    "doesn't adjust for the county's own claims history, because in testing that made predictions for unseen "
    "years worse (a county's history is often one storm's footprint, and the next storm goes elsewhere). "
    "The history and certainty above show where the county's experience differs.")
