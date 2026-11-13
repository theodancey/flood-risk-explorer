r"""
Flood Risk Explorer: Streamlit app entry point (Milestones 14-15).

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

# The County Explorer is the default page, served at "/". Streamlit ignores a default page's own
# url_path, so the same script is ALSO registered as a hidden page at "/county": shared links to
# /county then open the County Explorer instead of "Page not found" (Milestone 15).
EXPLORER = APP_DIR / "views" / "county_explorer.py"
pages = [
    st.Page(EXPLORER, title="County Explorer", icon=":material/map:", url_path="explorer", default=True),
    st.Page(EXPLORER, title="County Explorer", icon=":material/map:", url_path="county", visibility="hidden"),
    st.Page(APP_DIR / "views" / "profile_calculator.py", title="Property Profile Calculator",
            icon=":material/calculate:", url_path="calculator"),
    st.Page(APP_DIR / "views" / "how_it_works.py", title="How It Works", icon=":material/menu_book:",
            url_path="how-it-works"),
]
page = st.navigation(pages)

with st.sidebar:
    st.markdown("### Flood Risk Explorer")
    st.caption("Flood insurance losses by county for Virginia, North Carolina and Maryland, "
               "from public FEMA data (2010-2025).")
    st.caption(f"[Source code on GitHub]({common.GITHUB_URL})")

page.run()
