r"""
Milestone 16: does a REGIONAL level inside North Carolina predict unseen years better than one
statewide level?

Plain idea: today the model gives every NC county the same "storm level" (the NC state factor).
Hurricanes hit the coast much more than the mountains, so maybe the Coastal Plain, Piedmont and
Mountains each deserve their own level. This script tests that with the SAME pre-stated rule used
for the county factor (PROGRESS.md, "Milestone 16 decisions"): fit on 15 years, predict the 16th
(leave-one-year-out, LOYO), repeat for every year, plus the 2021-2025 holdout.

Regions: NOAA NCEI's county-to-climate-division table (official, county-level). NC divisions 1-2 are
the Mountains, 3-5 the Piedmont, 6-8 the Coastal Plain.

Run from the project folder (takes a while; prints progress):
    .venv\Scripts\python.exe src\nc_region_test.py
Writes reports/11_nc_region_scores.csv, reports/11_nc_region_summary.csv,
reports/11_nc_region_relativities.csv and reports/11_nc_region_ae.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import multistate as ms  # noqa: E402

ROOT = SRC.parent
CLIMDIV = ROOT / "data" / "raw" / "noaa_climdiv" / "county-to-climdivs.txt"
BASE = "Coastal Plain or outside NC"
DIVISION_REGION = {1: "NC Mountains", 2: "NC Mountains", 3: "NC Piedmont", 4: "NC Piedmont",
                   5: "NC Piedmont", 6: BASE, 7: BASE, 8: BASE}
NO_REGION, REGION = "no region (production)", "NC regions"


def nc_regions():
    """County FIPS (5 digits) -> region, for North Carolina's 100 counties."""
    rows = []
    for line in CLIMDIV.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[0].startswith("37"):
            rows.append((parts[0], DIVISION_REGION[int(parts[2][-2:])]))
    out = dict(rows)
    assert len(out) == 100, f"expected 100 NC counties, found {len(out)}"
    return out


def add_region(df, regions):
    """nc_region column: the NC region for NC rows, the base level everywhere else (incl. NC rows
    with an unknown county, which keep the statewide level)."""
    r = df["county"].astype(str).map(regions)
    return df.assign(nc_region=r.where(df["state"] == "NC", BASE).fillna(BASE))


def region_spec(spec):
    s = {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)) for k, v in spec.items()}
    s["freq_features"] = s["freq_features"] + ["nc_region"]
    s["sev_features"] = s["sev_features"] + ["nc_region"]
    s["freq_bases"] = {**s["freq_bases"], "nc_region": BASE}
    s["sev_bases"] = {**s["sev_bases"], "nc_region": BASE}
    return s


def compare(cells, claims, spec, spec_r, verbose=True):
    """16-year LOYO and the 2021-2025 holdout for both models, scored within each state."""
    scores, oof, t0 = [], [], time.time()
    splits = [(yr, cells["year"] != yr, claims["year"] != yr, cells["year"] == yr, "LOYO")
              for yr in ms.val.ALL_YEARS]
    splits.append((0, cells["year"] <= ms.val.LAST_TRAIN_YEAR, claims["year"] <= ms.val.LAST_TRAIN_YEAR,
                   cells["year"] > ms.val.LAST_TRAIN_YEAR, "holdout"))
    for yr, tr_mc, tr_mk, te_m, design in splits:
        tr_c, tr_k = cells[tr_mc], claims[tr_mk]
        te = cells[te_m & (cells["exposure"] > 0)].reset_index(drop=True)
        m0, _, _ = ms.fit3(tr_c, tr_k, spec, county=False)
        m1, _, _ = ms.fit3(tr_c, tr_k, spec_r, county=False)
        preds = {NO_REGION: m0.predict(te, county=False), REGION: m1.predict(te, county=False)}
        del m0, m1
        scores += ms._score_states(te, preds, tr_c, {"design": design, **({"test_year": 0} if yr == 0 else {})},
                                   by_year=(design == "LOYO"))
        if design == "LOYO":
            oof.append(te[["state", "year", "county", "nc_region", "exposure", ms.LOSS]].assign(
                pred_no_region=preds[NO_REGION], pred_region=preds[REGION]))
        if verbose:
            print(f"  {design} {yr or '2021-25'} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            ms.mem("after this step")
    return pd.concat(scores, ignore_index=True), pd.concat(oof, ignore_index=True)


def decide(scores):
    """The pre-stated rule (PROGRESS.md, Milestone 16 decisions). Returns (adopt, summary table)."""
    lo = scores[scores["design"] == "LOYO"].groupby(["model", "state"])["improvement_rebased_pct"].mean().unstack()
    ho = scores[scores["design"] == "holdout"].set_index(["model", "state"])["improvement_rebased_pct"].unstack()
    summ = lo.add_prefix("LOYO ").join(ho.add_prefix("holdout "))
    summ["LOYO average of states"] = lo.mean(axis=1)
    summ["holdout states below no-region"] = ho.lt(ho.loc[NO_REGION], axis=1).sum(axis=1)
    years = scores[scores["design"] == "LOYO"].pivot_table(index=["state", "test_year"], columns="model",
                                                             values="improvement_rebased_pct")
    summ["NC years better than no-region"] = [
        int((years.loc["NC", m] > years.loc["NC", NO_REGION]).sum()) if m != NO_REGION else np.nan
        for m in summ.index]
    r = summ.loc[REGION]
    rule1 = r["LOYO average of states"] > summ.loc[NO_REGION, "LOYO average of states"]
    rule2 = r["holdout states below no-region"] <= 1
    rule3 = r["LOYO NC"] >= summ.loc[NO_REGION, "LOYO NC"]
    adopt = bool(rule1 and rule2 and rule3)
    summ = summ.reset_index()
    summ["rule 1 (LOYO average higher)"] = [None if m == NO_REGION else bool(rule1) for m in summ["model"]]
    summ["rule 2 (holdout below in <= 1 state)"] = [None if m == NO_REGION else bool(rule2) for m in summ["model"]]
    summ["rule 3 (NC LOYO not lower)"] = [None if m == NO_REGION else bool(rule3) for m in summ["model"]]
    summ["adopted"] = [(m == REGION) == adopt for m in summ["model"]]
    return adopt, summ


def region_ae(oof):
    """Actual vs expected (A/E) by NC region over the 16 unseen years, for both models. A/E above 1
    = the model was too low there; below 1 = too high (e.g. inland counties storms mostly missed)."""
    nc = oof[oof["state"] == "NC"]
    g = nc.groupby("nc_region")[[ms.LOSS, "pred_no_region", "pred_region", "exposure"]].sum()
    return pd.DataFrame({"policy_years": g["exposure"], "actual_losses_2025": g[ms.LOSS],
                         "ae_no_region": g[ms.LOSS] / g["pred_no_region"],
                         "ae_region": g[ms.LOSS] / g["pred_region"]}).reset_index()


def main():
    t0 = time.time()
    regions = nc_regions()
    print("NC counties per region:", pd.Series(regions).value_counts().to_dict())
    cells, claims = ms.load_data()
    cells, claims = add_region(cells, regions), add_region(claims, regions)
    spec = ms.production_spec()
    spec["county_parts"] = []
    spec_r = region_spec(spec)

    print("Full fit with regions (relativities) ...", flush=True)
    m, _, _ = ms.fit3(cells, claims, spec_r, county=False)
    rel = ms.pp.combined_relativity_table(m.freq, m.sev)
    rel = rel[rel["feature"] == "nc_region"]
    rel.to_csv(ms.REPORTS / "11_nc_region_relativities.csv", index=False)
    print(rel.to_string())
    del m

    print("LOYO + holdout, with and without regions ...", flush=True)
    scores, oof = compare(cells, claims, spec, spec_r)
    scores.to_csv(ms.REPORTS / "11_nc_region_scores.csv", index=False)
    adopt, summ = decide(scores)
    summ.to_csv(ms.REPORTS / "11_nc_region_summary.csv", index=False)
    ae = region_ae(oof)
    ae.to_csv(ms.REPORTS / "11_nc_region_ae.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string())
    print(ae.round(3).to_string())
    print("ADOPT REGIONS:" if adopt else "NOT ADOPTED:", adopt)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
