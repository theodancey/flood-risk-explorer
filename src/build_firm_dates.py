r"""
Build app/data/firm_dates.csv: each NFIP community's INITIAL Flood Insurance Rate Map (FIRM) date
for Virginia, North Carolina and Maryland (Milestone 15).

Why: the app's address lookup needs the date of the community's first flood insurance rate map to
tell whether a building is "pre-FIRM" or "post-FIRM" (see app/lookup.py: firm_status). These dates
almost never change, so a small snapshot bundled with the app is faster and more reliable than
calling the API on every lookup.

Source: OpenFEMA "NFIP Community Status Book" (NfipCommunityStatusBook v1), free and public, no key:
    https://www.fema.gov/api/open/v1/NfipCommunityStatusBook

Run from the project folder (needs the internet; uses only requests + pandas):
    .venv\Scripts\python.exe src\build_firm_dates.py

The full download is saved to data/raw/community_status_book/ (not in Git); the small extract goes
to app/data/firm_dates.csv (in Git, a few KB).
"""

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "community_status_book"
OUT = ROOT / "app" / "data" / "firm_dates.csv"
URL = "https://www.fema.gov/api/open/v1/NfipCommunityStatusBook"
STATES = ["VA", "NC", "MD"]


def download(state, page=1000):
    """All community records for one state (OpenFEMA pages through results with $skip)."""
    rows, skip = [], 0
    while True:
        for attempt in range(3):
            try:
                r = requests.get(URL, params={"$filter": f"state eq '{state}'", "$top": page, "$skip": skip},
                                 timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(3)
        batch = r.json()["NfipCommunityStatusBook"]
        rows += batch
        if len(batch) < page:
            return rows
        skip += page


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    allrows = []
    for st in STATES:
        rows = download(st)
        (RAW / f"csb_{st}.json").write_text(json.dumps(rows), encoding="utf-8")
        print(f"{st}: {len(rows)} communities")
        allrows += rows
    df = pd.DataFrame(allrows)
    out = pd.DataFrame({
        "cid": df["communityIdNumber"].astype(str).str.zfill(6),
        "state": df["state"],
        "community": df["communityName"].str.replace("*", "", regex=False).str.strip(),
        "county": df["county"].str.strip(),
        # dates only (no time); missing = the community has no FIRM yet
        "initial_firm": pd.to_datetime(df["initialFloodInsuranceRateMap"], errors="coerce", utc=True).dt.date,
        "initial_fhbm": pd.to_datetime(df["initialFloodHazardBoundaryMap"], errors="coerce", utc=True).dt.date,
        "participating": df["participatingInNFIP"],
    }).sort_values(["state", "cid"]).drop_duplicates("cid")
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows, {OUT.stat().st_size / 1024:.0f} KB, "
          f"{out['initial_firm'].isna().sum()} without a FIRM date; downloaded {date.today()}")


if __name__ == "__main__":
    main()
