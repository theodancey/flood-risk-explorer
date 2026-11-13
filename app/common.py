"""
Shared code for the Flood Risk Explorer Streamlit app (Milestone 14).

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

CERTAINTY_STYLE = {   # label -> (badge colour, icon, one-line meaning)
    "High": ("green", ":material/check_circle:",
             "plenty of local claim history, and it agrees with the model."),
    "Medium": ("orange", ":material/info:",
               "some local claim history; it roughly agrees with the model, but the estimate could be off."),
    "Low": ("red", ":material/warning:",
            "little local claim history, or the county's history differs a lot from the model. Treat the "
            "number as a rough guide."),
}

# Major flood events behind each state's worst loss years in 2010-2025 (checked once against the
# claims' dates of loss and FEMA's event names during development; the app does not read claims).
MAJOR_EVENTS = {
    "VA": {2011: "Irene", 2012: "Sandy", 2016: "Matthew", 2018: "Michael"},
    "NC": {2011: "Irene", 2016: "Matthew", 2018: "Florence", 2024: "Helene"},
    "MD": {2011: "Irene", 2012: "Sandy", 2016: "Ellicott City flash flood",
           2018: "Ellicott City flash flood", 2021: "Oct. nor'easter"},
}

DISCLAIMER_MD = (
    "**Model-based, area-level averages** from public FEMA National Flood Insurance Program (NFIP) data "
    "(Virginia, North Carolina, Maryland; 2010-2025). Not a prediction for any specific property, not an "
    "insurance quote, and not financial advice. Not affiliated with or endorsed by FEMA."
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
    """0.0072 claims per policy-year -> '7.2 per 1,000 policy-years'."""
    return f"{1000 * freq:.1f} per 1,000"


def one_in(freq):
    return f"about 1 claim every {1 / freq:,.0f} policy-years" if freq > 0 else "no claims expected"


def ratio_words(r, state_name):
    if r >= 1.1:
        return f"{r:.1f}× the {state_name} average"
    if r <= 1 / 1.1:
        return f"{100 * (1 - r):.0f}% below the {state_name} average"
    return f"close to the {state_name} average"


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------
def disclaimer():
    """Short, visible disclaimer: shown at the top of every page."""
    st.info(DISCLAIMER_MD, icon=":material/info:")


def certainty_badge(label):
    colour, icon, _ = CERTAINTY_STYLE.get(label, ("gray", None, ""))
    st.badge(f"{label} certainty", icon=icon, color=colour)


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
    factor with a multiplier and the running $ before/after, end row). Each step is shown as the
    dollar change it causes, in order of size; hover shows the multiplier."""
    steps = wf[~wf["player"].isin(["start", "end"])]
    start, end = wf.iloc[0], wf.iloc[-1]
    labels = [start["step"]] + list(steps["step"]) + [end["step"]]
    measure = ["absolute"] + ["relative"] * len(steps) + ["total"]
    values = [start["after"]] + list(steps["after"] - steps["before"]) + [end["after"]]
    text = [money(start["after"])] + [f"×{m:.2f}" for m in steps["multiplier"]] + [money(end["after"])]
    hover = ([f"{start['step']}: {money(start['after'])} per policy-year"]
             + [f"{s.step}<br>×{s.multiplier:.2f} ({'+' if s.after >= s.before else '−'}"
                f"{money(abs(s.after - s.before))})<br>running total {money(s.after)}"
                for s in steps.itertuples()]
             + [f"{end['step']}: {money(end['after'])} per policy-year ({end['multiplier']:.2f}× the start)"])
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measure, text=text, textposition="outside",
        hovertext=hover, hoverinfo="text", cliponaxis=False,
        increasing=dict(marker=dict(color=ORANGE)), decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color=GREY)), connector=dict(line=dict(color=LIGHT_GREY, width=1)),
    ))
    _base_layout(fig, height=70 + 34 * len(labels))
    xmax = max(wf["after"].max(), pd.to_numeric(wf["before"], errors="coerce").max())
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Expected loss per policy-year (2025 $, average year)", tickprefix="$",
                     range=[0, xmax * 1.25], separatethousands=True)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14), x=0, xanchor="left"))
    return fig


def show_chart(fig, key=None):
    st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key)
