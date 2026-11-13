r"""
Flood Risk Explorer: Streamlit app entry point (Milestone 14).

Run it from the project folder (Windows, PowerShell):
    .venv\Scripts\python.exe -m streamlit run app\app.py

This file only sets up the page and the menu. Each page is its own script in app/views/;
shared loading, formatting and charts live in app/common.py.
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
import common  # noqa: E402

st.set_page_config(page_title="Flood Risk Explorer", page_icon=":material/water:", layout="centered")

pages = [
    st.Page(APP_DIR / "views" / "county_explorer.py", title="County Explorer", icon=":material/map:",
            url_path="county", default=True),
    st.Page(APP_DIR / "views" / "profile_calculator.py", title="Property Profile Calculator",
            icon=":material/calculate:", url_path="calculator"),
    st.Page(APP_DIR / "views" / "how_it_works.py", title="How It Works", icon=":material/menu_book:",
            url_path="how-it-works"),
]
page = st.navigation(pages)

with st.sidebar:
    st.markdown("### Flood Risk Explorer")
    st.caption("Area-level flood insurance risk for Virginia, North Carolina and Maryland, "
               "from public NFIP data (2010-2025).")
    st.caption(common.DISCLAIMER_MD)
    st.caption(f"[Source code on GitHub]({common.GITHUB_URL})")

page.run()
