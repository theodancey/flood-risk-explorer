"""
Page 1: County Explorer (the default page, also at /county).

Pick a state and a county (or click the map, or search an address) to see the model's expected
yearly flood loss for the insured buildings there (an area-level average), next to what actually
happened in 2010-2025, how it compares with the state, how much local data backs it up, and WHY it
differs from the state average. Milestone 15: plain wording, a hero number, tabs.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

st.title("County Explorer")
st.caption("How much flood damage insured buildings in a county are expected to have each year, "
           "compared with the rest of the state.")
C.disclaimer()

# ---------------------------------------------------------------------------
# Inputs: state, then county (the county list is searchable: click it and type)
# ---------------------------------------------------------------------------
state = st.segmented_control("State", list(C.STATES), format_func=C.STATES.get, default="VA",
                             required=True, key="ce_state")
fips_list, names = C.counties_in(state)
if st.session_state.get("ce_county") not in fips_list:        # first visit, or the state changed
    st.session_state["ce_county"] = C.biggest_county(state)
fips = st.selectbox("County or independent city (type to search)", fips_list,
                    format_func=lambda f: names[f], key="ce_county",
                    help="In Virginia, cities like Norfolk are separate from counties, as in Census and FEMA data.")

cs = C.county_summary()
row = cs[cs["county"] == fips].iloc[0]
res = C.county_result(fips)
sname = C.STATES[state]
name = row["name"]
h = C.county_history()
h = h[h["county"] == fips].sort_values("year")

# ---------------------------------------------------------------------------
# Headline: the model's estimate (hero) next to what actually happened
# ---------------------------------------------------------------------------
st.subheader(f"{name}, {state}")
model, actual = res["pure_premium"], float(row["actual_loss_cost_2010_2025"])
r = res["ratio_to_state"]
with st.container(border=True, key="hero"):
    c1, c2 = st.columns([1.25, 1])
    c1.metric("Model estimate (typical year)", C.money(model),
              delta=f"{C.ratio_words(r, sname)} ({C.money(res['state_average_pure_premium'])})",
              delta_color=("inverse" if abs(r - 1) >= 0.1 else "off"),
              delta_arrow=("auto" if abs(r - 1) >= 0.1 else "off"),
              help="Expected flood claim cost per insured building per year, in 2025 dollars, for a typical "
                   "year of storms and this county's mix of insured buildings.")
    c2.metric("Actual 2010-2025 average", C.money(actual),
              help="What flood claims really cost per insured building per year here, averaged over "
                   "2010-2025 (2025 dollars). A few big storms can push this far above or below a typical year.")
    gap = actual / model if model > 0 else 1.0
    if gap > 1.5 or gap < 1 / 1.5:
        top = h[h["losses_2025"] > 0].nlargest(2, "losses_2025")
        events = C.MAJOR_EVENTS[state]
        storms = [f"{events[y]} ({y})" if y in events else str(y) for y in top["year"]]
        if gap > 1.5 and storms:
            st.markdown(f"Past losses here were about **{gap:.1f} times** the estimate. A few storms, such as "
                        f"{' and '.join(storms)}, caused most of them. The model estimates a typical year.")
        elif gap > 1.5:
            st.markdown(f"Past losses here were about **{gap:.1f} times** the estimate. The model estimates a typical year.")
        else:
            st.markdown("Past losses here were well **below** the estimate. Big storms mostly missed this area "
                        "in 2010-2025, but that may not last. The model estimates a typical year.")
st.markdown(C.short_summary(res))

# ---------------------------------------------------------------------------
# Tabs: overview / claims history / why it differs
# ---------------------------------------------------------------------------
tab_over, tab_hist, tab_why = st.tabs(["Overview", "Claims history", "Why it differs"])

with tab_over:
    c3, c4 = st.columns(2)
    c3.metric("Chance of a claim each year", C.per_thousand(res["frequency"]),
              help=f"Expected paid flood claims per 1,000 insured buildings in a typical year ({C.one_in(res['frequency'])}).")
    c4.metric("Typical claim size", C.money(res["severity"]),
              help="Expected size of one paid claim (building + belongings), 2025 dollars.")

    st.markdown("#### How much local data backs this up?")
    C.show_certainty(row["Z_sev"], row["paid_claims_2010_2025"], row["history_vs_model"], name)
    st.caption(f"{res['policy_years']:,.0f} years of insured buildings here in 2010-2025 (one building insured "
               "for one year counts as one).")
    with st.expander("Learn more about this check"):
        st.markdown(row["uncertainty_note"])
        st.caption("The exact rule (credibility Z and the Bühlmann-Straub method) is on the How It Works page.")

    st.markdown("#### What kind of buildings are insured here?")
    MIX = [("pct_av_zone", "In a high-risk flood zone (A or V)"),
           ("pct_pre_firm", "Built before the area's first flood map"),
           ("pct_built_pre_1980", "Built before 1980"), ("pct_elevated", "Raised above the ground"),
           ("pct_multi_or_nonres", "Apartments or businesses"), ("pct_not_primary", "Second homes or rentals")]
    labels = [lab for _, lab in MIX]
    county_vals = [res["mix"][k] for k, _ in MIX]
    state_vals = [res["state_mix"][k] for k, _ in MIX]
    fig = go.Figure()
    fig.add_bar(y=labels, x=county_vals, name=name, orientation="h", marker_color=C.BLUE,
                text=[C.pct(v) for v in county_vals], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>" + name + ": %{x:.1f}%<extra></extra>")
    fig.add_bar(y=labels, x=state_vals, name=f"All of {sname}", orientation="h", marker_color=C.LIGHT_GREY,
                text=[C.pct(v) for v in state_vals], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>" + sname + ": %{x:.1f}%<extra></extra>")
    C._base_layout(fig, height=380)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, 118], ticksuffix="%", title="Share of insured buildings, 2010-2025")
    C.show_chart(fig, key="mix")
    st.caption("FEMA's flood maps mark A and V zones as high risk (V = on the coast, with waves). Buildings "
               "from before the area's first flood map were usually built without flood-safe rules.")

with tab_hist:
    events = C.MAJOR_EVENTS[state]
    if h["paid_claims"].sum() == 0:
        st.info(f"{name} had no paid NFIP flood claims in 2010-2025 "
                f"({h['policy_years'].sum():,.0f} years of insured buildings). So there's no local history to "
                "check the estimate against.", icon=":material/info:")
    else:
        lc = (h["losses_2025"] / h["policy_years"]).where(h["policy_years"] > 0, 0)
        fig = go.Figure(go.Bar(
            x=h["year"], y=h["losses_2025"], marker_color=C.BLUE,
            customdata=list(zip(h["paid_claims"], h["policy_years"], lc)),
            hovertemplate="%{x}<br>Paid losses: $%{y:,.0f}<br>Paid claims: %{customdata[0]:,}<br>"
                          "Insured buildings: %{customdata[1]:,.0f}<br>Loss per building: $%{customdata[2]:,.0f}"
                          "<extra></extra>"))
        top = h["losses_2025"].max()
        for yr, ev in events.items():
            val = h.loc[h["year"] == yr, "losses_2025"]
            if len(val):
                fig.add_annotation(x=yr, y=float(val.iloc[0]), text=ev, showarrow=False, yshift=12,
                                   font=dict(size=11, color=C.GREY), textangle=-30 if len(ev) > 10 else 0)
        C._base_layout(fig, height=340)
        fig.update_xaxes(dtick=1, tickangle=-45, gridcolor="rgba(0,0,0,0)")
        fig.update_yaxes(title="Paid losses (2025 $)", tickprefix="$", range=[0, top * 1.3 if top > 0 else 1])
        C.show_chart(fig, key="history")
        st.markdown(f"In 2010-2025, {name} had **{int(row['paid_claims_2010_2025']):,} paid flood claims**. "
                    f"They cost **{C.money(actual)} per insured building per year** on average. The model's "
                    f"estimate for a typical year is {C.money(model)}.")
        st.caption(f"Labels mark the big storms behind {sname}'s worst years. In most counties, a few storm "
                   "years cause most of the losses, so past claims jump around a lot.")
        hv = row["history_vs_model"]
        if hv == hv:
            with st.expander("Learn more: allowing for each year's storms"):
                st.markdown(f"If we compare each year with the model's estimate *for that year's storms* (not a "
                            f"typical year), this county's claims cost about **{hv:.2f}×** what the model expects "
                            "(1.00 = a perfect match).")

with tab_why:
    st.markdown(f"#### Why it differs from the {sname} average")
    C.show_chart(C.waterfall_figure(res["waterfall"]), key="why")
    st.caption(f"Each bar shows how one feature of the buildings insured here moves the estimate away from the "
               f"{sname} average. Orange bars raise it, blue bars lower it.")
    with st.expander("Learn more about this chart"):
        st.markdown(
            "Each bar compares this county's **mix** of insured buildings with the state's, one feature at a "
            "time (hover a bar to see its multiplier, for example ×1.20 = 20% higher). **Combined effects** "
            "is the extra that comes from how the features combine (the model's *interactions*). Features "
            "that move the estimate by less than 2% are grouped as **Other small effects**.\n\n"
            "There is **no county adjustment**: the estimate doesn't follow the county's own past claims, "
            "because in testing that made predictions for new years worse (a county's history is often one "
            "storm's path, and the next storm goes elsewhere).")

C.footer()
