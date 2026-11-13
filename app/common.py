"""
Shared code for the Flood Risk Explorer Streamlit app (Milestones 14-15).

Every page imports this module. It:
  * finds the app's data folder RELATIVE to this file (app/data), so the app runs the same on
    any computer or free host, whatever folder it is started from;
  * loads the small precomputed files once and caches them (st.cache_data / st.cache_resource);
  * holds the colours, number formatting, the disclaimer and the chart helpers, so every page
    looks and reads the same.

The app reads ONLY app/data/ (about 1.3 MB). It never touches the raw downloads or the processed
modelling tables in the project's top-level data folder (not in Git, and not on a free host) and never loads the statsmodels/XGBoost model objects: the
production model is in app/data/rating_factors.json as small lookup tables.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data"
ROOT = APP_DIR.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import explain  # noqa: E402  (src/explain.py: numpy + pandas only when loaded with with_model=False)

STATES = {"VA": "Virginia", "NC": "North Carolina", "MD": "Maryland"}
GITHUB_URL = "https://github.com/theodancey/flood-risk-explorer"
LAST_UPDATED = "2026-11-13"        # date the app and its data were last updated

# ---------------------------------------------------------------------------
# Colours: the dataviz reference palette (pre-validated for colour-blind readers).
# Orange = pushes expected loss UP, blue = pulls it DOWN, grey = totals / the state comparison.
# ---------------------------------------------------------------------------
BLUE = explain.DOWN        # "#2a78d6"
ORANGE = explain.UP        # "#eb6834"
GREY = explain.TOTAL       # "#52514e"
LIGHT_GREY = "#b9b8b1"
GRID = "#e1e0d9"
INK = "#0b0b0b"

# Major flood events behind each state's worst loss years in 2010-2025 (checked once against the
# claims' dates of loss and FEMA's event names during development; the app does not read claims).
MAJOR_EVENTS = {
    "VA": {2011: "Irene", 2012: "Sandy", 2016: "Matthew", 2018: "Michael"},
    "NC": {2011: "Irene", 2016: "Matthew", 2018: "Florence", 2024: "Helene"},
    "MD": {2011: "Irene", 2012: "Sandy", 2016: "Ellicott City flash flood",
           2018: "Ellicott City flash flood", 2021: "Oct. nor'easter"},
}

# The FULL disclaimer (footer of every page) and the SHORT one (top of every page). Both keep the
# same meaning: area-level averages from public data, never a prediction or quote for one property.
DISCLAIMER_MD = (
    "**Model-based, area-level averages** from public FEMA National Flood Insurance Program (NFIP) data "
    "(Virginia, North Carolina, Maryland; 2010-2025). Not a prediction for any specific property, not an "
    "insurance quote, and not financial advice. Not affiliated with or endorsed by FEMA."
)
SHORT_DISCLAIMER_MD = (
    "**Area-level averages** from public FEMA flood insurance data. Not a prediction or a quote for any "
    "specific property."
)


# ---------------------------------------------------------------------------
# Loading (cached: the files are read once per server, not on every click)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def county_summary():
    """One row per county (257): expected values, certainty, property mix, summary sentence."""
    df = pd.read_csv(DATA / "county_summary.csv", dtype={"county": str})
    return df.sort_values(["state", "name"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def county_history():
    """County x year 2010-2025: policy-years, paid claims, losses in 2025 dollars."""
    return pd.read_csv(DATA / "county_history.csv", dtype={"county": str})


@st.cache_data(show_spinner=False)
def state_averages():
    return json.loads((DATA / "state_averages.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading the model tables...")
def explainer():
    """The rating tables + insured portfolio, built once and shared by every visitor.
    cache_resource (not cache_data): it's one model-like object we don't want copied per call."""
    return explain.load(DATA, with_model=False)


@st.cache_data(show_spinner=False, max_entries=500)
def county_result(fips):
    return explainer().explain_county(fips)


@st.cache_data(show_spinner="Working out each factor's effect...", max_entries=500)
def profile_result(state, county, features):
    """features: a tuple of (name, value) pairs (hashable, so results can be cached)."""
    return explainer().explain_profile({**dict(features), "county": county}, state=state)


def counties_in(state):
    """FIPS codes of the counties in a state, sorted by name, and a FIPS -> name lookup."""
    cs = county_summary()
    sub = cs[cs["state"] == state].sort_values("name")
    return list(sub["county"]), dict(zip(cs["county"], cs["name"]))


def biggest_county(state):
    cs = county_summary()
    sub = cs[cs["state"] == state]
    return sub.loc[sub["policy_years_2010_2025"].idxmax(), "county"]

@st.cache_data(show_spinner=False)
def zone_average(state, zone):
    """The model's expected yearly loss for the AVERAGE insured building in one flood-zone group of a
    state (for example 'B/C/X moderate-low risk' in Virginia): an exposure-weighted average over that
    state's insured buildings in the zone, i.e. 'a typical insured low-risk building'."""
    ex = explainer()
    p = ex.portfolio
    rows = p.index[(p["state"] == state) & (p["flood_zone_group"].astype(str) == zone)]
    w = p.loc[rows, "exposure"].to_numpy(float)
    return float((w * ex.p_rates.loc[rows, "pure_premium"].to_numpy()).sum() / w.sum())


def av_share(state):
    """% of the state's insured policy-years in high-risk A/V zones (2010-2025)."""
    return float(state_averages()["states"][state]["mix_pct"]["pct_av_zone"])


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def money(x, decimals=0):
    return f"${x:,.{decimals}f}"


def ratio(x):
    return f"{x:.2f}×"


def pct(x, decimals=0):
    return f"{x:.{decimals}f}%"


def per_thousand(freq):
    """0.0072 claims per building per year -> '7.2 in 1,000'."""
    return f"{1000 * freq:.1f} in 1,000"


def one_in(freq):
    return f"about 1 in {1 / freq:,.0f} insured buildings a year" if freq > 0 else "no claims expected"


def ratio_words(r, state_name):
    """Plain comparison with the state average: '1.6 times', '15% below', 'about the same as'."""
    if r >= 1.1:
        return f"{r:.1f} times the {state_name} average"
    if r <= 1 / 1.1:
        return f"{100 * (1 - r):.0f}% below the {state_name} average"
    return f"about the same as the {state_name} average"


def change_words(m):
    """A multiplier in plain words: 1.44 -> 'raises it by 44%', 0.81 -> 'lowers it by 19%'."""
    if m >= 1:
        return f"raises it by {100 * (m - 1):.0f}%"
    return f"lowers it by {100 * (1 - m):.0f}%"


# ---------------------------------------------------------------------------
# Plain-English labels for the model's characteristics (Milestone 15). The model and src/explain.py
# keep their technical labels; the app shows these instead.
# ---------------------------------------------------------------------------
PLAIN_LEVEL = {
    ("flood_zone_group", "A/V high risk"): "Flood zone: high risk (A or V)",
    ("flood_zone_group", "B/C/X moderate-low risk"): "Flood zone: lower risk (B, C or X)",
    ("construction_band", "before 1960"): "Built before 1960",
    ("construction_band", "1960-1979"): "Built 1960-1979",
    ("construction_band", "1980-1999"): "Built 1980-1999",
    ("construction_band", "2000 or later"): "Built 2000 or later",
    ("occupancy_group", "single-unit residential"): "Single-family home",
    ("occupancy_group", "multi-unit residential"): "Building with 2+ homes",
    ("occupancy_group", "non-residential"): "Business or other non-home",
    ("building_coverage_band", "$0 (contents only)"): "Coverage: belongings only",
    ("building_coverage_band", "under $100K"): "Coverage: under $100K",
    ("building_coverage_band", "$100K-$249K"): "Coverage: $100K-$249K",
    ("building_coverage_band", "$250K or more"): "Coverage: $250K or more",
    ("elevated", "True"): "Raised building: Yes",
    ("elevated", "False"): "Raised building: No",
    ("floors_group", "1"): "Floors: 1",
    ("floors_group", "2"): "Floors: 2",
    ("floors_group", "3+"): "Floors: 3 or more",
    ("primary_residence", "True"): "Main home: Yes",
    ("primary_residence", "False"): "Main home: No",
}
PLAIN_MIX = {   # county waterfall steps: how the county's MIX of insured buildings differs from the state
    "flood_zone_group": "How many are in high-risk zones",
    "construction_band": "How old the buildings are",
    "occupancy_group": "Homes vs. apartments and businesses",
    "building_coverage_band": "How much coverage people buy",
    "elevated": "How many are raised",
    "floors_group": "Number of floors",
    "primary_residence": "Main homes vs. second homes",
    "overlap": "Combined effects",
}


def pretty_step(player, step):
    """Plain label for one waterfall row. Never shows True/False: booleans become Yes/No."""
    if player in ("start", "end", "small"):
        return step
    level = step.split(": ", 1)[1] if ": " in step else None
    if level is not None and (player, level) in PLAIN_LEVEL:          # a profile step
        return PLAIN_LEVEL[(player, level)]
    if step.startswith("Mix: ") or player == "overlap":                # a county step
        return PLAIN_MIX.get(player, step.removeprefix("Mix: ").capitalize())
    return step.replace(": True", ": Yes").replace(": False", ": No")


SMALL_EFFECT = 0.02     # steps that change the result by less than 2% are grouped together


def group_small(wf, threshold=SMALL_EFFECT):
    """Waterfall table with plain labels, and the steps within +/-2% merged into one 'Other small
    effects' row (its multiplier = the product of the merged ones, so the total is unchanged)."""
    wf = wf.copy()
    wf["label"] = [pretty_step(p, s) for p, s in zip(wf["player"], wf["step"])]
    wf["merged"] = None
    start, end = wf.iloc[[0]], wf.iloc[[-1]]
    steps = wf.iloc[1:-1]
    small = steps[(steps["multiplier"] - 1).abs() < threshold]
    if len(small) < 2:                                     # nothing (or only one bar) to group
        return wf
    big = steps.drop(small.index)
    merged = pd.DataFrame([{"step": "Other small effects", "player": "small",
                            "multiplier": float(small["multiplier"].prod()),
                            "label": "Other small effects", "merged": ", ".join(small["label"])}])
    out = pd.concat([start, big, merged, end], ignore_index=True)
    run = float(start["after"].iloc[0])                   # recompute the running totals
    for i in range(1, len(out) - 1):
        out.loc[i, "before"] = run
        run *= out.loc[i, "multiplier"]
        out.loc[i, "after"] = run
    return out


# ---------------------------------------------------------------------------
# Certainty: two plain signals instead of one label (Milestone 15)
# ---------------------------------------------------------------------------
def certainty_signals(z, claims, history_vs_model):
    """Returns (amount, agreement, shaky).
    amount: 'plenty' / 'some' / 'limited': how much local claim history there is to check the
            estimate against (credibility Z >= 0.8, 0.5-0.8, below 0.5 or no claims).
    agreement: None when past claims roughly agree with the model (within 0.5-2x), otherwise the
            ratio (past claims / model).
    shaky: True only when the data is limited AND past claims are more than 3x off the model in either
            direction (the only red case: about 65 small counties, under 3% of insured buildings)."""
    no_claims = claims is None or claims <= 0 or z is None or z != z
    amount = "limited" if no_claims else ("plenty" if z >= 0.8 else ("some" if z >= 0.5 else "limited"))
    hv = history_vs_model
    differs = (not no_claims) and hv is not None and hv == hv and not 0.5 <= hv <= 2.0
    agreement = hv if differs else None
    shaky = amount == "limited" and agreement is not None and not 1 / 3 <= agreement <= 3
    return amount, agreement, shaky


AMOUNT_TEXT = {"plenty": ("Plenty of local claims data", "green", ":material/check_circle:"),
               "some": ("Some local claims data", "blue", ":material/info:"),
               "limited": ("Limited local claims data", "gray", ":material/info:")}


def agreement_words(hv):
    if hv >= 1:
        return f"Past claims were about {hv:.1f} times the estimate"
    return f"Past claims were about {100 * hv:.0f}% of the estimate"


def show_certainty(z, claims, history_vs_model, where):
    """Two badges and 1-2 plain sentences. The technical rule is on the How It Works page."""
    amount, agreement, shaky = certainty_signals(z, claims, history_vs_model)
    label, colour, icon = AMOUNT_TEXT[amount]
    if shaky:
        colour, icon = "red", ":material/warning:"
    badges = f":{colour}-badge[{icon} {label}]"
    if agreement is not None:
        badges += f" :{'red' if shaky else 'gray'}-badge[{agreement_words(agreement)}]"
    st.markdown(badges)
    if claims is None or claims <= 0:
        why = f"{where} had no paid flood claims in 2010-2025, so there is nothing local to check the estimate against."
    else:
        why = (f"We checked the estimate against {where}'s own past claims ({claims:,.0f} paid claims). "
               "More claims make that check more reliable.")
    if agreement is not None:
        why += " Past claims here differ a lot from the estimate, often because one storm hit or missed the area."
        if shaky:
            why += " Treat the number as a rough guide."
    st.caption(why)
    return amount, agreement, shaky


# ---------------------------------------------------------------------------
# Short plain-English county summary (about 200 characters)
# ---------------------------------------------------------------------------
MIX_PLAIN = {   # mix share -> plain phrase (completes "X% of insured buildings here ...")
    "pct_av_zone": "are in high-risk flood zones",
    "pct_built_pre_1980": "were built before 1980",
    "pct_multi_or_nonres": "are apartments or businesses",
    "pct_cover_250k_plus": "have $250K+ of coverage",
    "pct_elevated": "are raised",
    "pct_one_floor": "have one floor",
    "pct_not_primary": "are second homes or rentals",
}


def short_summary(res):
    """One short sentence: how the county compares with its state, and the biggest reason."""
    r, sn = res["ratio_to_state"], res["state_name"]
    head = f"Expected yearly losses here are {ratio_words(r, sn)}"
    if abs(r - 1) < 0.1:
        return head + "."
    d = 1 if r >= 1 else -1
    drv = explain._drivers(res["waterfall"], d, k=1)
    drv = drv[drv["player"].isin(list(explain.FACTOR_MIX))]
    if not len(drv):
        return head + "."
    key = explain.FACTOR_MIX[drv.iloc[0]["player"]]
    return (head + f", mainly because {res['mix'][key]:.0f}% of insured buildings here {MIX_PLAIN[key]} "
            f"({sn}: {res['state_mix'][key]:.0f}%).")


def profile_summary(res):
    """One short sentence for the calculator: the result and its two biggest reasons."""
    r = res["ratio_to_state"]
    head = f"Insured buildings like this have about {money(res['pure_premium'])} of flood claims a year on average"
    if abs(r - 1) < 0.1:
        return head + f", {ratio_words(r, res['state_name'])}."
    drv = explain._drivers(res["waterfall"], 1 if r >= 1 else -1)
    reasons = [pretty_step(p, s).lower() for p, s in zip(drv["player"], drv["step"])]
    word = "raise" if r >= 1 else "lower"
    return head + "." + (f" The biggest things that {word} it: {' and '.join(reasons)}." if reasons else "")


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------
def disclaimer():
    """Short, visible disclaimer: shown at the top of every page (the full one is in the footer)."""
    st.info(SHORT_DISCLAIMER_MD, icon=":material/info:")


def footer():
    """Footer on every page: the full disclaimer, who built it, the data source and the date."""
    st.divider()
    st.caption(DISCLAIMER_MD)
    st.caption(f"Built by **Theo Dancey** · [Source code on GitHub]({GITHUB_URL}) · Data: FEMA OpenFEMA "
               f"National Flood Insurance Program (public) · Last updated {LAST_UPDATED}")


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _base_layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=30, b=8), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(size=13, color=INK), hoverlabel=dict(font_size=13))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def waterfall_figure(wf, title=None):
    """Plotly waterfall from the state average to the result.

    wf: the 'waterfall' table from explain.explain_county / explain_profile (start row, one row per
    characteristic with a multiplier and the running $ before/after, end row). Labels are made plain
    (Yes/No, no jargon) and effects within +/-2% are grouped into one 'Other small effects' bar.
    Each bar shows the dollar change it causes; hover shows the % change and the multiplier."""
    wf = group_small(wf)
    steps = wf.iloc[1:-1]
    start, end = wf.iloc[0], wf.iloc[-1]
    labels = [start["label"]] + list(steps["label"]) + [end["label"]]
    measure = ["absolute"] + ["relative"] * len(steps) + ["total"]
    deltas = list(steps["after"] - steps["before"])
    values = [start["after"]] + deltas + [end["after"]]
    text = ([money(start["after"])] + [("+" if d >= 0 else "−") + money(abs(d)) for d in deltas]
            + [money(end["after"])])
    hover = [f"{start['label']}: {money(start['after'])} a year"]
    for s in steps.itertuples():
        extra = f"<br>Includes: {s.merged}" if isinstance(s.merged, str) else ""
        hover.append(f"{s.label}<br>{change_words(s.multiplier).capitalize()} "
                     f"({'+' if s.after >= s.before else '−'}{money(abs(s.after - s.before))})"
                     f"<br>Multiplier ×{s.multiplier:.2f}{extra}")
    hover.append(f"{end['label']}: {money(end['after'])} a year")
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measure, text=text, textposition="outside",
        hovertext=hover, hoverinfo="text", cliponaxis=False,
        increasing=dict(marker=dict(color=ORANGE)), decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color=GREY)), connector=dict(line=dict(color=LIGHT_GREY, width=1)),
    ))
    _base_layout(fig, height=70 + 34 * len(labels))
    xmax = max(wf["after"].max(), pd.to_numeric(wf["before"], errors="coerce").max())
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Expected yearly loss per building (2025 dollars)", tickprefix="$",
                     range=[0, xmax * 1.3], separatethousands=True)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14), x=0, xanchor="left"))
    return fig


def show_chart(fig, key=None, **kwargs):
    return st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key, **kwargs)
