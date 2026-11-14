r"""
Flood Risk Explorer: Streamlit app entry point (Milestones 14-15).

Run it from the project folder (Windows, PowerShell):
    .venv\Scripts\python.exe -m streamlit run app\app.py

This file only sets up the page and the menu. Each page is its own script in app/views/;
shared loading, formatting and charts live in app/common.py.
"""

import importlib
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ---------------------------------------------------------------------------
# Stale-module guard (Milestone 16). Streamlit re-runs the page SCRIPTS on every click, but the
# helper modules they import (common, lookup, src/explain) are imported only once per server, and
# Python keeps that first copy in memory (sys.modules). After a push, Streamlit Cloud pulls the new
# files and re-runs the pages, yet the OLD common.py can still be in memory, which gives errors
# like "module 'common' has no attribute ...". So on every run we check each helper's file: if it
# changed on disk since it was loaded, reload it. Dependencies first (common imports explain and
# lookup), so common picks up their new versions too. Unchanged files are not reloaded, so normal
# clicks cost only a few file-date checks. Cached results (st.cache_data) survive a reload when
# the function's code is unchanged, because Streamlit keys its cache on the function's source.
_HELPERS = ("explain", "lookup", "common")


def _mtime(module):
    try:
        return Path(module.__file__).stat().st_mtime_ns
    except (AttributeError, TypeError, OSError):
        return None


def _load_fresh_helpers():
    for name in _HELPERS:
        module = sys.modules.get(name)
        if module is None:
            if name == "common":
                importlib.import_module(name)     # first run: common also imports explain + lookup
            continue
        now = _mtime(module)
        if now is not None and now != getattr(module, "_loaded_mtime_ns", now):
            module = importlib.reload(module)
        module._loaded_mtime_ns = _mtime(module)
    for name in _HELPERS:                         # stamp modules imported for the first time above
        module = sys.modules.get(name)
        if module is not None and not hasattr(module, "_loaded_mtime_ns"):
            module._loaded_mtime_ns = _mtime(module)


_load_fresh_helpers()
import common  # noqa: E402  (the up-to-date copy; the pages' "import common" gets this same one)

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
