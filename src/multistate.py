"""
Milestone 13: from Virginia to Virginia + North Carolina + Maryland.

Three questions, in order:
  1. TRANSFER TEST: does the Virginia production model, applied UNCHANGED, work in North
     Carolina and Maryland? (A/E overall, by year, by segment; lift and Gini.)
  2. REFIT: the same production GLM approach fitted on all three states, with state x year
     effects and a STATE FACTOR, per-state off-balance and Buhlmann-Straub county factors
     pooled over all counties of the three states. Validated by leave-one-year-out (LOYO),
     the 2021-2025 holdout, and leave-one-STATE-out.
  3. CONSISTENCY: the same model fitted on each state alone. Do the relativities agree?
     (Pre-stated rule in PROGRESS.md, Milestone 13 decisions.)

Run from the project folder (results go to reports/10_*.csv and data/processed/m13_*):
    .venv\\Scripts\\python.exe src\\multistate.py              # self-test (simulated data)
    .venv\\Scripts\\python.exe src\\multistate.py transfer     # step 1 (about a minute)
    .venv\\Scripts\\python.exe src\\multistate.py refit        # steps 2-3, all three stages
    .venv\\Scripts\\python.exe src\\multistate.py refit main   # ... or one stage at a time
    .venv\\Scripts\\python.exe src\\multistate.py refit consistency
    .venv\\Scripts\\python.exe src\\multistate.py refit validate
    .venv\\Scripts\\python.exe src\\multistate.py xgb          # XGBoost check (frozen settings)
    .venv\\Scripts\\python.exe src\\multistate.py production   # save the 3-state production model
Long steps print "[memory X GB]" after each fit (needs psutil, in requirements.txt).

Area level only: every number describes groups of insured buildings, never one property.
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SRC = Path(__file__).resolve().parent
for p in (SRC, SRC / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import actuarial as A                         # noqa: E402
import build_dataset as B                     # noqa: E402
import validation as val                      # noqa: E402
import frequency_glm as fg                    # noqa: E402
import severity_glm as sv                     # noqa: E402
import pure_premium as pp                     # noqa: E402
import production                             # noqa: E402

ROOT = SRC.parent
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
STATES = ("VA", "NC", "MD")
LOSS = pp.LOSS
SEGMENTS = ["flood_zone_group", "construction_band", "occupancy_group", "building_coverage_band",
            "elevated", "floors_group", "primary_residence", "post_firm", "coastal_tidal"]


# ---------------------------------------------------------------------------
# Data and the production specification
# ---------------------------------------------------------------------------
# Memory (Milestone 13). statsmodels result objects point back at their model and vice versa
# (a "reference cycle"). Python frees such objects only when its cycle collector happens to run,
# so in a loop of hundreds of fits the old design matrices piled up (about +1 GB per fit; the
# first 3-state refit was stopped by the system at ~10 GB). Forcing a collection before every
# fit keeps memory flat (~0.5 GB): see PROGRESS.md, Milestone 13 decisions.
try:
    import psutil
    _PROC = psutil.Process()
except ImportError:                                    # memory printing is optional
    _PROC = None


def mem(tag=""):
    """Free unreachable objects now, and print this process's memory use if psutil is installed."""
    gc.collect()
    if _PROC is not None and tag:
        print(f"      [memory {_PROC.memory_info().rss / 1e9:.2f} GB] {tag}", flush=True)


# Columns of the cell table that no Milestone 13 step uses (dropped on load to save memory)
UNUSED_CELL_COLS = ["losses_nominal", "claims_all", "cpi_factor"]


def load_data(verbose=True):
    """(cells, claims) for the three states, only the columns the models use. The claim table is
    built on first use."""
    import pyarrow.parquet as pq
    names = [c for c in pq.read_schema(B.OUT_FILE_3).names if c not in UNUSED_CELL_COLS]
    cells = pd.read_parquet(B.OUT_FILE_3, columns=names)
    if Path(sv.CLAIMS_PATH_3).exists():
        claims = pd.read_parquet(sv.CLAIMS_PATH_3)
    else:
        claims = sv.build_claims_table(B.OUT_FILE_3, sv.CLAIMS_PATH_3, verbose, states=STATES)
    mem("data loaded" if verbose else "")
    return cells, claims


def production_spec(path=production.VA_PATH):
    """Features, interactions and base levels of the Virginia production model (Milestone 11),
    so the refit uses exactly the same structure."""
    m = production.load(path)
    return {"freq_features": list(m.freq.spec.features), "sev_features": list(m.sev.spec.features),
            "freq_interactions": [list(x) for x in m.freq.spec.interactions],
            "sev_interactions": [list(x) for x in m.sev.spec.interactions],
            "freq_bases": pp.bases_without_year(m.freq.spec), "sev_bases": pp.bases_without_year(m.sev.spec)}


def fit3(cells, claims, spec, county=True):
    """The production approach on whatever states are in `cells`: frequency x severity GLMs
    (state x year effects + state factor when there are several states), loads, per-state
    off-balance, and Buhlmann-Straub county factors re-centred within each state.
    Returns (model, county table, credibility parameters)."""
    mem()                                              # free earlier fits first (see mem())
    m = pp.fit_combined(cells, claims, spec["freq_features"], spec["sev_features"], spec["freq_bases"],
                        spec["sev_bases"], spec["freq_interactions"], spec["sev_interactions"], by_state=True)
    if not county:
        return m, None, None
    multi = cells["state"].nunique() > 1
    return pp.with_county_factors(m, cells, claims, "buhlmann_straub",
                                  recentre_by="state" if multi else None)


def cat_years_by_state(cells):
    """Pre-stated rule: each state's 3 worst years by 2025-dollar losses."""
    out = {st: val.cat_years(cells[cells["state"] == st]) for st in cells["state"].unique()}
    out["pooled"] = val.cat_years(cells)
    return out


def top_events(states=STATES):
    """The flood event with the most paid claims in each state and year (FEMA's floodEvent
    field), so storms are labelled from the data rather than from memory."""
    rows = []
    for st in states:
        c = A.load_claims(st, first_year=A.FIRST_FULL_YEAR)
        c = c[c["is_paid"] & (c["yearOfLoss"] <= A.LAST_FULL_YEAR)]
        g = c.groupby(["yearOfLoss", c["floodEvent"].fillna("(no event name)")]).agg(
            n=("id", "size"), paid=("total_paid", "sum")).reset_index()
        tot = g.groupby("yearOfLoss")["n"].transform("sum")
        g["share_of_year_claims"] = g["n"] / tot
        top = g.sort_values("n", ascending=False).drop_duplicates("yearOfLoss")
        top = top.rename(columns={"yearOfLoss": "year", "floodEvent": "top_event", "n": "top_event_claims"})
        top.insert(0, "state", st)
        rows.append(top[["state", "year", "top_event", "top_event_claims", "share_of_year_claims"]])
    return pd.concat(rows, ignore_index=True).sort_values(["state", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def rebase_by(pred, actual, groups):
    """Scale predictions so they add up to the actual total within each group (e.g. each
    state-year): removes the year's overall level, which no model can predict."""
    d = pd.DataFrame({"g": np.asarray(groups), "p": np.asarray(pred, float), "a": np.asarray(actual, float)})
    s = d.groupby("g")[["p", "a"]].transform("sum")
    return (d["p"] * s["a"] / s["p"].where(s["p"] > 0)).fillna(0).to_numpy()


def segment_ae(df, actual, pred_rebased, features):
    """A/E by level of each feature, after re-basing (1.0 = the segment is right)."""
    rows = []
    for f in features:
        g = df.assign(_a=actual, _p=pred_rebased).groupby(df[f].astype(str))[["_a", "_p"]].sum()
        for lvl, r in g.iterrows():
            rows.append({"feature": f, "level": lvl, "actual": r["_a"], "expected": r["_p"],
                         "ae": r["_a"] / r["_p"] if r["_p"] > 0 else np.nan})
    return pd.DataFrame(rows)


def score_by_year(test, pred, train_rate, all_losses_by_year=None):
    """pp.score_cells for each test year (rows already one state)."""
    out = []
    for yr, t in test.groupby("year"):
        mask = (test["year"] == yr).to_numpy()
        al = None if all_losses_by_year is None else all_losses_by_year.get(yr)
        out.append({"test_year": int(yr), **pp.score_cells(t, np.asarray(pred)[mask], train_rate, all_test_losses=al)})
    return pd.DataFrame(out)


def summarise(scores, cat, by=("model", "state")):
    """Mean improvement and Gini: all years, non-cat, cat; years won are added by the caller."""
    rows = []
    for key, g in scores.groupby(list(by)):
        key = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        st = key.get("state", "pooled")
        cy = cat.get(st, cat["pooled"])
        c = g["test_year"].isin(cy)
        rows.append({**key, "folds": len(g), "mean_improvement": g["improvement_rebased_pct"].mean(),
                     "non_cat_mean": g.loc[~c, "improvement_rebased_pct"].mean(),
                     "cat_mean": g.loc[c, "improvement_rebased_pct"].mean(),
                     "median": g["improvement_rebased_pct"].median(), "mean_gini": g["gini"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Transfer test
# ---------------------------------------------------------------------------
def transfer(cells, claims, verbose=True):
    """Apply the Virginia production model unchanged to each state (county factor 1.0 outside
    Virginia; Virginia's average-year level everywhere). Virginia itself is shown in-sample,
    for reference. Returns a dict of tables."""
    mv = production.load(production.VA_PATH).combined
    va_rate = None
    overall, by_year, segs, lifts, scores = [], [], [], [], []
    for st in STATES:
        c_all = cells[cells["state"] == st]
        c = c_all[c_all["exposure"] > 0].reset_index(drop=True)
        k = sv.severity_rows(claims[claims["state"] == st])
        county = st == "VA"            # NC/MD counties get 1.0 (the VA table's 37xxx codes are VA errors)
        pred_pp = mv.predict(c, county=county)
        pred_f = mv.freq.predict(c, "average")
        pred_s = mv.sev.predict(k, "average")
        if st == "VA":
            va_rate = c_all[LOSS].sum() / c_all["exposure"].sum()
        overall.append({"state": st, "in_sample": st == "VA", "policy_years": c["exposure"].sum(),
                        "paid_claims": int(c_all["claims"].sum()), "losses_2025": c_all[LOSS].sum(),
                        "expected_losses": pred_pp.sum(), "ae_pure_premium": c_all[LOSS].sum() / pred_pp.sum(),
                        "ae_frequency": c["claims"].sum() / pred_f.sum(),
                        "ae_severity": k[sv.TARGET].sum() / pred_s.sum(),
                        "actual_loss_cost": c_all[LOSS].sum() / c_all["exposure"].sum(),
                        "expected_loss_cost": pred_pp.sum() / c["exposure"].sum()})
        # by year (raw A/E: the model predicts an AVERAGE year, so storm years are > 1)
        yf = c.assign(pf=pred_f, pp_=pred_pp).groupby("year").agg(
            claims=("claims", "sum"), pred_claims=("pf", "sum"), pred_losses=("pp_", "sum"))
        yl = c_all.groupby("year")[LOSS].sum()
        ys = k.assign(ps=pred_s).groupby("year").agg(sev_act=(sv.TARGET, "sum"), sev_pred=("ps", "sum"))
        y = yf.join(yl).join(ys)
        y["ae_frequency"] = y["claims"] / y["pred_claims"]
        y["ae_pure_premium"] = y[LOSS] / y["pred_losses"]
        y["ae_severity"] = y["sev_act"] / y["sev_pred"]
        by_year.append(y.reset_index().assign(state=st))
        # segments, re-based within each year
        rb_pp = rebase_by(pred_pp, c[LOSS], c["year"])
        rb_f = rebase_by(pred_f, c["claims"], c["year"])
        rb_s = rebase_by(pred_s, k[sv.TARGET], k["year"])
        seg_feats = [f for f in SEGMENTS if f in c]
        for kind, d, a, p in (("pure_premium", c, c[LOSS], rb_pp), ("frequency", c, c["claims"], rb_f),
                              ("severity", k, k[sv.TARGET], rb_s)):
            segs.append(segment_ae(d, a.to_numpy(), p, [f for f in seg_feats if f in d]).assign(state=st, kind=kind))
        # lift over all years (average-year rates, so years don't drive the ranking)
        lt = pp.lift_table(c[LOSS], rebase_by(pred_pp, c[LOSS], np.zeros(len(c))), c["exposure"])
        lt["pred_relative"] = lt["pred_loss_cost"] / (c[LOSS].sum() / c["exposure"].sum())
        lifts.append(lt.reset_index().assign(state=st))
        al = c_all.groupby("year")[LOSS].sum().to_dict()
        scores.append(score_by_year(c, pred_pp, va_rate, al).assign(state=st, model="VA model, unchanged"))
        if verbose:
            o = overall[-1]
            print(f"  {st}: A/E pure premium {o['ae_pure_premium']:.2f}, frequency {o['ae_frequency']:.2f}, "
                  f"severity {o['ae_severity']:.2f}; mean per-year improvement "
                  f"{scores[-1]['improvement_rebased_pct'].mean():.1f}%, Gini {scores[-1]['gini'].mean():.3f}")
    return {"overall": pd.DataFrame(overall), "by_year": pd.concat(by_year, ignore_index=True),
            "segments": pd.concat(segs, ignore_index=True), "lift": pd.concat(lifts, ignore_index=True),
            "scores": pd.concat(scores, ignore_index=True)}


# ---------------------------------------------------------------------------
# 2. Refit and its validation
# ---------------------------------------------------------------------------
def _score_states(test_cells, preds, train_cells, label_rows, by_year=True):
    """Score each model's predictions per state (and per year if by_year)."""
    rows = []
    for st, t in test_cells.groupby("state"):
        m = (test_cells["state"] == st).to_numpy()
        tr = train_cells[train_cells["state"] == st] if (train_cells["state"] == st).any() else train_cells
        rate = tr[LOSS].sum() / tr["exposure"].sum()
        for name, pr in preds.items():
            if pr is None or (isinstance(pr, dict) and st not in pr):
                continue
            p = pr[st] if isinstance(pr, dict) else np.asarray(pr)[m]
            if by_year:
                s = score_by_year(t, p, rate)
            else:
                s = pd.DataFrame([pp.score_cells(t, p, rate)])
            rows.append(s.assign(model=name, state=st, **label_rows))
    return rows


def refit_validation(cells, claims, spec, verbose=True):
    """LOYO (16 folds), the 2021-2025 holdout and leave-one-state-out.

    Models scored on each state's held-out rows:
      "3-state GLM + county"   the refit (state factor, county factors re-centred per state)
      "3-state GLM"            the same without county factors
      "own-state GLM + county" the same approach fitted on that state alone
      "VA model, unchanged"    (NC, MD only) the transfer-test model, for comparison
    Returns (scores, out-of-fold predictions)."""
    mv = production.load(production.VA_PATH).combined
    scores, oof = [], []
    t0 = time.time()

    def preds_for(train_c, train_k, test):
        m3, _, _ = fit3(train_c, train_k, spec)
        p = {"3-state GLM + county": m3.predict(test), "3-state GLM": m3.predict(test, county=False)}
        own = {}
        for st in STATES:
            tm = test["state"] == st
            if not tm.any() or not (train_c["state"] == st).any():
                continue
            ms, _, _ = fit3(train_c[train_c["state"] == st], train_k[train_k["state"] == st], spec)
            own[st] = ms.predict(test[tm])
        p["own-state GLM + county"] = own
        p["VA model, unchanged"] = {st: mv.predict(test[test["state"] == st], county=False)
                                    for st in ("NC", "MD") if (test["state"] == st).any()}
        return p

    for yr in val.ALL_YEARS:
        tr_c, tr_k = cells[cells["year"] != yr], claims[claims["year"] != yr]
        te = cells[(cells["year"] == yr) & (cells["exposure"] > 0)].reset_index(drop=True)
        p = preds_for(tr_c, tr_k, te)
        scores += _score_states(te, p, tr_c, {"design": "LOYO"})
        o = te[["state", "year", "county", "county_group", *SEGMENTS, "exposure", "claims", LOSS]].copy()
        o["pred_3state_county"] = p["3-state GLM + county"]
        o["pred_3state"] = p["3-state GLM"]
        o["pred_own"] = np.nan
        for st, v in p["own-state GLM + county"].items():
            o.loc[o["state"] == st, "pred_own"] = v
        oof.append(o)
        if verbose:
            print(f"  LOYO {yr} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            mem("after this step")

    # holdout: train 2010-2020, test 2021-2025 (scored over the whole window, once)
    tr_c, tr_k = cells[cells["year"] <= val.LAST_TRAIN_YEAR], claims[claims["year"] <= val.LAST_TRAIN_YEAR]
    te = cells[(cells["year"] > val.LAST_TRAIN_YEAR) & (cells["exposure"] > 0)].reset_index(drop=True)
    scores += _score_states(te, preds_for(tr_c, tr_k, te), tr_c, {"design": "holdout", "test_year": 0}, by_year=False)
    if verbose:
        print(f"  holdout done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        mem("after this step")

    # leave-one-state-out: fit on two states (all years), predict the third. The held-out
    # state has no state factor, so it gets the base state's level (Virginia's, or North
    # Carolina's when Virginia is held out); raw A/E shows that level, re-based scores ignore it.
    for st in STATES:
        tr_c, tr_k = cells[cells["state"] != st], claims[claims["state"] != st]
        te = cells[(cells["state"] == st) & (cells["exposure"] > 0)].reset_index(drop=True)
        m2, _, _ = fit3(tr_c, tr_k, spec)
        p = {"2 other states GLM + county": m2.predict(te)}
        rows = _score_states(te, p, tr_c, {"design": "leave-one-state-out"})
        scores += rows
        if verbose:
            print(f"  leave out {st} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            mem("after this step")
    return pd.concat(scores, ignore_index=True), pd.concat(oof, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Relativities by state, heterogeneity tests, consistency labels
# ---------------------------------------------------------------------------
def _component_fit(cells, claims, spec, kind, cov="cluster"):
    """One GLM component (frequency or severity) with cluster-robust errors (county group x
    year), state x year effects. Returns (model, training rows)."""
    mem()                                              # free earlier fits first (see mem())
    if kind == "frequency":
        rows = fg.aggregate(cells[cells["exposure"] > 0],
                            [*spec["freq_features"], "county_group", "state", "state_year"])
        m = fg.fit_glm(rows, spec["freq_features"], spec["freq_interactions"], cov=cov,
                       cluster_by=("county_group", "year"), bases=spec["freq_bases"], year_col="state_year")
    else:
        rows = sv.severity_rows(claims)
        m = sv.fit_gamma(rows, spec["sev_features"], spec["sev_interactions"], cov=cov,
                         cluster_by=("county_group", "year"), bases=spec["sev_bases"], year_col="state_year")
    return m, rows


def term_groups(model):
    """Design columns grouped by term: each feature, each interaction (no const, no year)."""
    groups = {}
    for c in model.spec.columns:
        if c == "const" or c.startswith("year["):
            continue
        parts = [p.split("[")[0] for p in c.split(":")]
        groups.setdefault(" x ".join(parts), []).append(c)
    return groups


def relativities_with_ci(model, label):
    ci = model.result.conf_int()
    ci.index = model.params.index
    rows = []
    for term, cols in term_groups(model).items():
        for c in cols:
            rows.append({"fit": label, "term": term, "column": c, "relativity": float(np.exp(model.params[c])),
                         "ci_low": float(np.exp(ci.loc[c, 0])), "ci_high": float(np.exp(ci.loc[c, 1]))})
    return rows


def _state_dummies(rows, states):
    s = rows["state"].astype(str).to_numpy()
    return {st: (s == st).astype(float) for st in states}


def fit_with_state_interaction(rows, model, kind, term_cols, states=("NC", "MD"), cov="cluster"):
    """Refit `model`'s design plus (term columns x state) interactions, cluster-robust errors.
    Returns (statsmodels result, list of the new column names, design builder)."""
    mem()
    def design(df, include_year=True):
        X = fg.design_matrix(df, model.spec, include_year)
        sd = _state_dummies(df, states)
        new = {}
        for c in term_cols:
            for st in states:
                new[f"{c}:state[{st}]"] = X[c].to_numpy() * sd[st]
        return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
    X = design(rows)
    if kind == "frequency":
        glm = sm.GLM(rows["claims"].to_numpy(float), X, family=sm.families.Poisson(),
                     offset=np.log(rows["exposure"].to_numpy(float)))
        start = None
    else:
        glm = sm.GLM(rows[sv.TARGET].to_numpy(float), X, family=sm.families.Gamma(sm.families.links.Log()))
        start = np.zeros(X.shape[1])
        start[0] = np.log(rows[sv.TARGET].mean())
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": fg.cluster_codes(rows, ("county_group", "year"))}} \
        if cov == "cluster" else {}
    res = glm.fit(start_params=start, **kw)
    newcols = [c for c in X.columns if ":state[" in c]
    return res, newcols, design


def heterogeneity_tests(cells, claims, spec, verbose=True):
    """For every term of each component: cluster-robust Wald test that the term's effect is the
    same in all three states (term x state interactions = 0)."""
    out = []
    for kind in ("frequency", "severity"):
        m, rows = _component_fit(cells, claims, spec, kind)
        for term, cols in term_groups(m).items():
            res, newcols, _ = fit_with_state_interaction(rows, m, kind, cols)
            names = list(res.params.index)
            R = np.zeros((len(newcols), len(names)))
            for i, c in enumerate(newcols):
                R[i, names.index(c)] = 1.0
            w = res.wald_test(R, scalar=True)
            out.append({"kind": kind, "term": term, "df": len(newcols), "wald_chi2": float(w.statistic),
                        "p_value": float(w.pvalue)})
            if verbose:
                print(f"    {kind:9s} {term:45s} p = {float(w.pvalue):.4f}", flush=True)
                mem("after this step")
    return pd.DataFrame(out)


def consistency_labels(rel, het):
    """The pre-stated rule (PROGRESS.md, Milestone 13 decisions), per term:
      consistent       every level on the same side of 1 in all three states, and Wald p >= 0.05
      differs in size  same direction in all states, Wald p < 0.05
      inconsistent     some level has a state whose CI lies entirely on the OTHER side of 1
                       from the pooled estimate
      unclear          point estimates on different sides of 1, but no CI excludes 1 the other way
    """
    out = []
    pooled = rel[rel["fit"] == "pooled"].set_index(["kind", "column"])
    for (kind, term), g in rel[rel["fit"] != "pooled"].groupby(["kind", "term"]):
        p = het.set_index(["kind", "term"]).loc[(kind, term), "p_value"]
        same_side, opposite_ci = True, False
        for col, gg in g.groupby("column"):
            ref = pooled.loc[(kind, col), "relativity"]
            up = ref >= 1
            if not ((gg["relativity"] >= 1) == up).all():
                same_side = False
            if up and (gg["ci_high"] < 1).any() or (not up) and (gg["ci_low"] > 1).any():
                opposite_ci = True
        label = ("inconsistent" if opposite_ci else
                 "consistent" if same_side and p >= 0.05 else
                 "differs in size" if same_side else "unclear")
        out.append({"kind": kind, "term": term, "wald_p": p, "label": label})
    return pd.DataFrame(out)


def interaction_keep_rule(cells, claims, spec, terms, verbose=True):
    """For terms whose state effect is significant: does a term x state interaction also
    improve dev-LOYO (2010-2020) deviance? Scored per state and test year (re-based within
    each), averaged. Keep rule = Wald p < 0.05 AND better dev-LOYO mean."""
    rows = []
    for kind, term in terms:
        imp = {"base": [], "with": []}
        for yr in val.DEV_YEARS:
            tr_c = cells[(cells["year"] != yr) & (cells["year"] <= val.LAST_TRAIN_YEAR)]
            tr_k = claims[(claims["year"] != yr) & (claims["year"] <= val.LAST_TRAIN_YEAR)]
            m, rows_tr = _component_fit(tr_c, tr_k, spec, kind, cov="none")
            cols = term_groups(m)[term]
            res, _, design = fit_with_state_interaction(rows_tr, m, kind, cols, cov="none")
            if kind == "frequency":
                te = fg.aggregate(cells[(cells["year"] == yr) & (cells["exposure"] > 0)],
                                  [*spec["freq_features"], "county_group", "state", "state_year"])
            else:
                te = sv.severity_rows(claims[claims["year"] == yr])
            eta_b = fg.design_matrix(te, m.spec, False).to_numpy() @ m.params.to_numpy()
            Xw = design(te, include_year=False)
            eta_w = Xw.to_numpy() @ res.params.reindex(Xw.columns).fillna(0).to_numpy()
            for st, t in te.groupby("state"):
                mk = (te["state"] == st).to_numpy()
                if kind == "frequency":
                    e = t["exposure"].to_numpy()
                    if t["claims"].sum() == 0:
                        continue
                    for name, eta in (("base", eta_b), ("with", eta_w)):
                        imp[name].append(fg.score_fold(t, e * np.exp(eta[mk]), e)["improvement_rebased_pct"])
                else:
                    if len(t) < 5:
                        continue
                    for name, eta in (("base", eta_b), ("with", eta_w)):
                        imp[name].append(sv.score_fold(t, np.exp(eta[mk]), np.ones(len(t)))["improvement_rebased_pct"])
        b, w = np.mean(imp["base"]), np.mean(imp["with"])
        wins = int(np.sum(np.array(imp["with"]) > np.array(imp["base"])))
        rows.append({"kind": kind, "term": term, "dev_loyo_base": b, "dev_loyo_with_state_interaction": w,
                     "gain_points": w - b, "state_years_better": wins, "state_years": len(imp["base"]),
                     "passes_loyo": w > b})
        if verbose:
            print(f"    keep rule {kind} {term}: {b:.2f}% -> {w:.2f}% ({wins}/{len(imp['base'])} better)", flush=True)
            mem("after this step")
    return pd.DataFrame(rows)


def relativity_comparison(cells, claims, spec, verbose=True):
    """Same-spec fits: pooled (3 states, state x year) and each state alone, with
    cluster-robust 95% CIs; the shared-year sensitivity; heterogeneity tests and labels."""
    rel = []
    for kind in ("frequency", "severity"):
        m, _ = _component_fit(cells, claims, spec, kind)
        rel += [dict(r, kind=kind) for r in relativities_with_ci(m, "pooled")]
        for st in STATES:
            ms, _ = _component_fit(cells[cells["state"] == st], claims[claims["state"] == st], spec, kind)
            rel += [dict(r, kind=kind) for r in relativities_with_ci(ms, st)]
    rel = pd.DataFrame(rel)
    if verbose:
        print("  heterogeneity tests (term x state, cluster-robust Wald):", flush=True)
    het = heterogeneity_tests(cells, claims, spec, verbose)
    lab = consistency_labels(rel, het)
    flagged = [(r.kind, r.term) for r in lab.itertuples() if r.wald_p < 0.05]
    keep = interaction_keep_rule(cells, claims, spec, flagged, verbose) if flagged else pd.DataFrame()
    return rel, het, lab, keep


def candidate_spec(spec, keep):
    """The production spec plus every term x state interaction that passed the keep rule
    (`keep`: the interaction_keep_rule table; it holds only terms with Wald p < 0.05, so
    passes_loyo = passes the keep rule). A term "a x b" becomes the three-way term a x b x state."""
    out = {k: [list(x) for x in v] if "interactions" in k else v for k, v in spec.items()}
    for _, r in keep[keep["passes_loyo"]].iterrows():
        key = "freq_interactions" if r["kind"] == "frequency" else "sev_interactions"
        out[key].append([*r["term"].split(" x "), "state"])
    out["freq_bases"] = {**spec["freq_bases"], "state": "VA"}
    out["sev_bases"] = {**spec["sev_bases"], "state": "VA"}
    return out


def state_interaction_candidate(cells, claims, spec, keep, verbose=True):
    """Milestone 11's hybrid rule, second half: the model with ALL passing term x state
    interactions replaces the production spec only if its pure premium is not worse on the
    16-year LOYO (mean, per state) or the 2021-2025 holdout. Both fitted the same way (fit3,
    county factors included). Returns (scores, candidate spec)."""
    cand = candidate_spec(spec, keep)
    scores, t0 = [], time.time()
    splits = [(yr, cells["year"] != yr, claims["year"] != yr, cells["year"] == yr, "LOYO")
              for yr in val.ALL_YEARS]
    splits.append((0, cells["year"] <= val.LAST_TRAIN_YEAR, claims["year"] <= val.LAST_TRAIN_YEAR,
                   cells["year"] > val.LAST_TRAIN_YEAR, "holdout"))
    for yr, tr_mc, tr_mk, te_m, design in splits:
        tr_c, tr_k = cells[tr_mc], claims[tr_mk]
        te = cells[te_m & (cells["exposure"] > 0)].reset_index(drop=True)
        preds = {}
        for name, s in (("production spec", spec), ("with state interactions", cand)):
            m, _, _ = fit3(tr_c, tr_k, s)
            preds[name] = m.predict(te)
            del m
        scores += _score_states(te, preds, tr_c, {"design": design, **({"test_year": 0} if yr == 0 else {})},
                                by_year=(design == "LOYO"))
        if verbose:
            print(f"  candidate {design} {yr or '2021-25'} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            mem("after this step")
    return pd.concat(scores, ignore_index=True), cand


def shared_year_sensitivity(cells, claims, spec):
    """Alternative: ONE year factor for all states + a state main effect. How much do the
    relativities move compared with state x year effects?"""
    out = []
    for kind in ("frequency", "severity"):
        m_sy, rows = _component_fit(cells, claims, spec, kind, cov="none")
        if kind == "frequency":
            m_y = fg.fit_glm(rows, [*spec["freq_features"], "state"], spec["freq_interactions"],
                             bases={**spec["freq_bases"], "state": "VA"})
        else:
            m_y = sv.fit_gamma(rows, [*spec["sev_features"], "state"], spec["sev_interactions"],
                               bases={**spec["sev_bases"], "state": "VA"})
        for c in m_sy.params.index:
            if c == "const" or c.startswith("year["):
                continue
            out.append({"kind": kind, "column": c, "state_x_year": float(np.exp(m_sy.params[c])),
                        "shared_year": float(np.exp(m_y.params[c]))})
        for c in [c for c in m_y.params.index if c.startswith("state[")]:
            out.append({"kind": kind, "column": c, "state_x_year": np.nan,
                        "shared_year": float(np.exp(m_y.params[c]))})
    d = pd.DataFrame(out)
    d["ratio"] = d["shared_year"] / d["state_x_year"]
    return d


# ---------------------------------------------------------------------------
# Credibility then vs now
# ---------------------------------------------------------------------------
def credibility_summary(cells, claims, spec):
    """Buhlmann-Straub parameters for: Virginia alone (Milestone 9's setting), each state
    alone, and the three states pooled (the production choice)."""
    rows, tables = [], {}
    fits = {"VA alone": ("VA",), "NC alone": ("NC",), "MD alone": ("MD",), "3 states pooled": STATES}
    for label, sts in fits.items():
        c, k = cells[cells["state"].isin(sts)], claims[claims["state"].isin(sts)]
        m, tbl, par = fit3(c, k, spec)
        tables[label] = tbl
        for part in ("freq", "sev"):
            p = par[part]
            rows.append({"fit": label, "part": part, "epv": p["epv"], "vhm": p["vhm"], "k": p["k"],
                         "counties": int((tbl["f_exp"] > 0).sum()),
                         "median_Z": float(tbl[f"Z_{part}"][tbl["s_n"] > 0].median())})
    return pd.DataFrame(rows), tables


# ---------------------------------------------------------------------------
# 5. XGBoost check (frozen Milestone 10 settings, no re-tuning)
# ---------------------------------------------------------------------------
class StateLevelXGB:
    """An XGBModel fitted with state x year levels in base_margin, plus an average-year level
    for EACH state (the same idea as the GLM's state factor). Average-year predictions only."""

    def __init__(self, m, train):
        self.m, self.kind = m, m.kind
        tm = np.exp(m.tree_margin(train).astype(float))
        g = train["state"].astype(str).to_numpy()
        if m.kind == "frequency":
            a, b = train["claims"].to_numpy(float), train["exposure"].to_numpy(float) * tm
        elif m.kind == "severity":
            a, b = train[sv.TARGET].to_numpy(float), tm
        else:
            a, b = train[LOSS].to_numpy(float), train["exposure"].to_numpy(float) * tm
        d = pd.DataFrame({"g": g, "a": a, "b": b}).groupby("g")[["a", "b"]].sum()
        self.levels = np.log(d["a"] / d["b"]).to_dict()

    def rate(self, df, year_level="average"):
        lvl = df["state"].astype(str).map(self.levels).fillna(np.mean(list(self.levels.values()))).to_numpy(float)
        r = np.exp(self.m.tree_margin(df).astype(float) + lvl)
        return r * (self.m.load if self.kind == "severity" else self.m.off_balance)

    def predict(self, df, year_level="average"):
        r = self.rate(df)
        return r if self.kind == "severity" else df["exposure"].to_numpy(float) * r


def xgb_cell_table(cells):
    """XGBoost's frequency / Tweedie rows: cells with exposure summed over the features + state-year."""
    import xgb_models as X
    mem()
    m = cells[cells["exposure_matched"] & (cells["exposure"] > 0)]
    keys = X.ALL_FEATURES + ["state", "state_year", "year"]
    return m.groupby(keys, observed=True)[["exposure", "claims", LOSS]].sum().reset_index()


def fit_xgb3(cells, claims, xcells, settings, kinds=("frequency", "severity", "tweedie")):
    """Frozen-settings XGBoost on the given training rows. Severity and Tweedie get 'state' added
    to their (geography) features; frequency keeps its no-geography features. Returns
    {"pp/XGB fxs": CombinedModel, "pp/XGB Tweedie": StateLevelXGB, "freq": ..., "sev": ...}."""
    import xgb_models as X
    mem()
    out = {}
    sev_rows = sv.severity_rows(claims)

    def fit(kind, tbl, unmatched=0.0):
        s = settings[kind]
        feats = list(s["features"]) + (["state"] if kind != "frequency" and cells["state"].nunique() > 1 else [])
        tr = tbl.assign(year=tbl["state_year"])          # state x year levels in base_margin
        m = X.fit_xgb(tr, kind, dict(s["params"]), s["n_rounds"], feats, s.get("monotone"),
                      unmatched_losses=unmatched)
        return StateLevelXGB(m, tr)

    if "frequency" in kinds:
        out["freq"] = fit("frequency", xcells)
    if "severity" in kinds:
        out["sev"] = fit("severity", sev_rows)
    if "frequency" in kinds and "severity" in kinds:
        xc = pp.CombinedModel(out["freq"], out["sev"], sv.icc_bridge_factor(claims))
        xc.off_balance_by = pp.off_balance_factor(xc, cells, by="state")
        xc.off_balance = xc.off_balance_by.get("VA", 1.0)
        out["pp/XGB fxs"] = xc
    if "tweedie" in kinds:
        unm = cells.loc[~(cells["exposure_matched"] & (cells["exposure"] > 0))]
        tw = fit("tweedie", xcells, float(unm[LOSS].sum()))
        # per-state off-balance (puts back each state's unmatched losses)
        pred = tw.predict(cells[cells["exposure"] > 0])
        a = cells.groupby("state")[LOSS].sum()
        p = pd.Series(pred, index=cells[cells["exposure"] > 0].index).groupby(cells["state"]).sum()
        tw.ob = (a / p).to_dict()
        out["pp/XGB Tweedie"] = tw
    return out


def _pp_pred(model, df):
    if isinstance(model, StateLevelXGB):        # Tweedie with per-state off-balance
        return model.predict(df) * df["state"].map(model.ob).to_numpy(float)
    return model.predict(df)


def xgb_validation(cells, claims, spec, verbose=True):
    """LOYO (16 folds) and the holdout: the 3-state hybrid GLM + county vs XGBoost freq x sev and
    XGBoost Tweedie, identical rows. Returns (scores, out-of-fold predictions)."""
    import xgb_models as X
    mem()
    settings = X.read_settings()
    xcells = xgb_cell_table(cells)
    scores, oof = [], []
    t0 = time.time()
    splits = [(yr, [y for y in val.ALL_YEARS if y != yr], [yr], "LOYO") for yr in val.ALL_YEARS]
    splits.append((0, list(val.DEV_YEARS), list(val.HOLDOUT_YEARS), "holdout"))
    for label, tr_y, te_y, design in splits:
        tr_c, tr_k = cells[cells["year"].isin(tr_y)], claims[claims["year"].isin(tr_y)]
        te = cells[cells["year"].isin(te_y) & (cells["exposure"] > 0)].reset_index(drop=True)
        glm, _, _ = fit3(tr_c, tr_k, spec)
        xm = fit_xgb3(tr_c, tr_k, xcells[xcells["year"].isin(tr_y)], settings)
        preds = {"3-state hybrid GLM + county": glm.predict(te),
                 "XGB freq x sev": _pp_pred(xm["pp/XGB fxs"], te),
                 "XGB Tweedie": _pp_pred(xm["pp/XGB Tweedie"], te)}
        scores += _score_states(te, preds, tr_c, {"design": design}, by_year=(design == "LOYO"))
        if design == "LOYO":
            o = te[["state", "year", "county", "county_group", *SEGMENTS, "exposure", "claims", LOSS]].copy()
            for k, v in preds.items():
                o[k] = v
            oof.append(o)
        if verbose:
            print(f"  XGB check {design} {label or ''}: {(time.time() - t0) / 60:.1f} min", flush=True)
            mem("after this step")
    return pd.concat(scores, ignore_index=True), pd.concat(oof, ignore_index=True)


def xgb_explainability(cells, claims, spec):
    """Milestone 11's flip check on the 2025 in-force cells: flip one feature on every cell and
    record the share of exposure whose prediction moves AGAINST the actuarial direction."""
    import xgb_models as X
    mem()
    settings = X.read_settings()
    xm = fit_xgb3(cells, claims, xgb_cell_table(cells), settings, kinds=("frequency", "severity"))
    grid = cells[(cells["year"] == 2025) & (cells["exposure"] > 0)].reset_index(drop=True)
    checks = [("elevated", "False", "True", "down", "frequency"), ("elevated", "False", "True", "down", "severity"),
              ("flood_zone_group", "A/V high risk", "B/C/X moderate-low risk", "down", "frequency"),
              ("flood_zone_group", "A/V high risk", "B/C/X moderate-low risk", "down", "severity"),
              ("building_coverage_band", "$100K-$249K", "$250K or more", "up", "severity")]
    rows = []
    for f, a, b, direction, kind in checks:
        g = grid[grid[f].astype(str) == a]
        if not len(g):
            continue
        m = xm["freq" if kind == "frequency" else "sev"]
        g1 = g.assign(exposure=1.0)
        r0, r1 = m.rate(g1), m.rate(g1.assign(**{f: b}))
        wrong = (r1 > r0 * 1.0001) if direction == "down" else (r1 < r0 * 0.9999)
        for st in STATES:
            mk = (g["state"] == st).to_numpy()
            e = g["exposure"].to_numpy()[mk]
            rows.append({"kind": kind, "flip": f"{f}: {a} -> {b}", "expected_direction": direction, "state": st,
                         "share_exposure_wrong_way": float((e * wrong[mk]).sum() / e.sum()) if e.sum() else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _save(name, df):
    path = REPORTS / f"10_{name}.csv"
    df.to_csv(path, index=False)
    print(f"  saved {path.relative_to(ROOT)} ({len(df)} rows)")


def cmd_transfer():
    t0 = time.time()
    cells, claims = load_data()
    print("Cat years by state (top 3 by 2025-dollar losses):", cat_years_by_state(cells))
    res = transfer(cells, claims)
    ev = top_events()
    res["by_year"] = res["by_year"].merge(ev, on=["state", "year"], how="left")
    for k, v in res.items():
        _save(f"transfer_{k}", v)
    print(f"transfer done in {(time.time() - t0) / 60:.1f} min")


def cmd_refit(stage="all"):
    """The refit runs in three stages, each of which can be run on its own (one at a time keeps
    memory low): main (full fit, credibility), consistency (by-state fits, tests), validate
    (LOYO, holdout, leave-one-state-out). stage="all" runs them in order."""
    t0 = time.time()
    cells, claims = load_data()
    spec = production_spec()
    cat = cat_years_by_state(cells)
    print("Cat years:", cat)
    (PROCESSED / "m13_cat_years.json").write_text(json.dumps({k: list(v) for k, v in cat.items()}), encoding="utf-8")
    if stage in ("all", "main"):
        _refit_main(cells, claims, spec)
    if stage in ("all", "consistency"):
        _refit_consistency(cells, claims, spec)
    if stage in ("all", "candidate"):
        _refit_candidate(cells, claims, spec)
    if stage in ("all", "validate"):
        _refit_validate(cells, claims, spec, cat)
    print(f"refit ({stage}) done in {(time.time() - t0) / 60:.1f} min")


def _refit_main(cells, claims, spec):
    print("Full fit on all years ...", flush=True)
    m3, tbl, par = fit3(cells, claims, spec)
    rel = pp.combined_relativity_table(m3.freq, m3.sev)
    _save("refit_relativities", rel)
    tbl.reset_index().to_csv(PROCESSED / "m13_county_table.csv", index=False)
    (PROCESSED / "m13_credibility_params.json").write_text(json.dumps(par, default=float, indent=1), encoding="utf-8")

    mem("full fit")
    print("Credibility then vs now ...", flush=True)
    cs, _ = credibility_summary(cells, claims, spec)
    _save("credibility", cs)
    mem("credibility")


def _refit_consistency(cells, claims, spec):
    print("Relativities by state + heterogeneity tests ...", flush=True)
    rel_s, het, lab, keep = relativity_comparison(cells, claims, spec)
    _save("relativities_by_state", rel_s)
    _save("heterogeneity_tests", het)
    _save("consistency_labels", lab)
    if len(keep):
        _save("state_interaction_keep_rule", keep)
    _save("shared_year_sensitivity", shared_year_sensitivity(cells, claims, spec))
    mem("consistency")


def _refit_candidate(cells, claims, spec):
    kp = REPORTS / "10_state_interaction_keep_rule.csv"
    if not kp.exists():
        print("No term x state interaction passed the Wald test: nothing to try.")
        return
    keep = pd.read_csv(kp)
    if not keep["passes_loyo"].any():
        print("No term x state interaction passed the keep rule: nothing to try.")
        return
    print("Hybrid rule, second half: all passing state interactions together vs the production spec ...",
          flush=True)
    sc, cand = state_interaction_candidate(cells, claims, spec, keep)
    _save("state_interaction_candidate_scores", sc)
    (PROCESSED / "m13_candidate_spec.json").write_text(json.dumps(cand, indent=1), encoding="utf-8")
    lo = sc[sc["design"] == "LOYO"].groupby(["state", "model"])["improvement_rebased_pct"].mean().unstack()
    ho = sc[sc["design"] == "holdout"].set_index(["state", "model"])["improvement_rebased_pct"].unstack()
    print("  16-year LOYO mean improvement (%):")
    print(lo.round(2).to_string())
    print("  2021-2025 holdout improvement (%):")
    print(ho.round(2).to_string())


def _refit_validate(cells, claims, spec, cat):
    print("LOYO / holdout / leave-one-state-out ...", flush=True)
    scores, oof = refit_validation(cells, claims, spec)
    _save("refit_scores", scores)
    oof.to_parquet(PROCESSED / "m13_oof_loyo.parquet", index=False)
    _save("refit_summary", summarise(scores[scores["design"] == "LOYO"], cat))
    mem("validation")


def cmd_production():
    """Refit the 3-state production model on all years and save models/production_model.json
    (run only after the XGBoost check confirms the GLM recommendation; see PROGRESS.md)."""
    import hashlib
    cells, claims = load_data()
    spec = production_spec()
    m, tbl, par = fit3(cells, claims, spec)
    names = cells.drop_duplicates("county").set_index("county")["county_name"].to_dict()
    names.update({k: v for k, v in {"51515": "Bedford city (former; part of Bedford County since 2013)"}.items()})
    tbl = tbl.copy()
    tbl["name"] = [names.get(c, "Unknown county") for c in tbl.index.astype(str)]
    tbl.index = tbl.index.astype(str).rename("county")
    summ = pd.read_csv(REPORTS / "10_refit_summary.csv")
    sc = pd.read_csv(REPORTS / "10_refit_scores.csv")
    main = "3-state GLM + county"
    loyo = summ[summ["model"] == main].set_index("state")
    ho = sc[(sc["design"] == "holdout") & (sc["model"] == main)].set_index("state")
    sha = hashlib.sha256(Path(B.OUT_FILE_3).read_bytes()).hexdigest()[:16]
    sf = {st: {"frequency": float(np.exp(m.freq.params.get(f"state[{st}]", 0.0))),
               "severity": float(np.exp(m.sev.params.get(f"state[{st}]", 0.0))),
               "off_balance_ratio": float(m.off_balance_by[st] / m.off_balance_by["VA"])} for st in STATES}
    for v in sf.values():
        v["pure_premium"] = v["frequency"] * v["severity"] * v["off_balance_ratio"]
    card = {
        "name": "Flood loss cost model: Hybrid GLM fxs + state factor + county (VA, NC, MD)",
        "version": "2.0", "milestone": 13, "created": time.strftime("%Y-%m-%d"),
        "states": list(STATES),
        "structure": "frequency GLM (Poisson, log link, exposure offset) x severity GLM (Gamma, log link), "
                     "both with state x year effects absorbed into a state factor, x large-loss load x ICC "
                     "bridge x off-balance (per state) x Buhlmann-Straub county factor (pooled over the three "
                     "states' counties, re-centred within each state)",
        "frequency_features": spec["freq_features"], "frequency_interactions": spec["freq_interactions"],
        "severity_features": spec["sev_features"], "severity_interactions": spec["sev_interactions"],
        "state_factor_vs_virginia": sf,
        "fitted_on_years": [2010, 2025], "dollars": "2025 (CPI-U)", "level": "average year",
        "data_file": "data/processed/model_data_3states.parquet", "data_file_sha256_16": sha,
        "validation": {
            "loyo16_mean_improvement_pct_by_state": loyo["mean_improvement"].to_dict(),
            "loyo16_gini_by_state": loyo["mean_gini"].to_dict(),
            "holdout_2021_2025_improvement_pct_by_state": ho["improvement_rebased_pct"].to_dict(),
            "holdout_gini_by_state": ho["gini"].to_dict(),
            "yardstick": "re-based Tweedie (p = 1.5) deviance improvement vs no-features model; each unseen "
                         "year once, scored within each state"},
        "county_method": "buhlmann_straub", "county_parameters": par,
        "why_not_xgboost": "Milestone 11's decision rule re-applied on three states in Milestone 13 "
                           "(reports/10_multistate.md).",
        "report": "reports/10_multistate.md",
        "previous_virginia_model": "models/production_model_va.json",
    }
    row = {"flood_zone_group": "A/V high risk", "construction_band": "1960-1979", "floors_group": "1",
           "occupancy_group": spec["sev_bases"]["occupancy_group"],
           "building_coverage_band": spec["sev_bases"]["building_coverage_band"],
           "elevated": spec["sev_bases"]["elevated"], "primary_residence": spec["sev_bases"]["primary_residence"],
           "post_firm": "False"}
    checks = {}
    for st, county in (("VA", "51810"), ("NC", "37055"), ("MD", "24047")):
        r = pd.DataFrame([{**row, "county": county, "state": st}])
        checks[f"{st} {names.get(county)} 1960s house"] = {"county": county,
                                                           "pure_premium": float(m.rate(r)[0])}
    card["check_virginia_beach_worksheet"] = checks["VA Virginia Beach city 1960s house"]["pure_premium"]
    card["check_worksheets"] = checks
    production.save(m, card, tbl, production.PATH, production.DISCLAIMER_3)
    print("saved models/production_model.json")
    for k, v in checks.items():
        print(f"  {k}: ${v['pure_premium']:,.2f}")
    print("  state factors vs Virginia:", json.dumps(sf, indent=1))


def cmd_xgb():
    t0 = time.time()
    cells, claims = load_data()
    spec = production_spec()
    print("XGBoost (frozen Milestone 10 settings) vs the 3-state GLM ...", flush=True)
    scores, oof = xgb_validation(cells, claims, spec)
    _save("xgb_scores", scores)
    oof.to_parquet(PROCESSED / "m13_oof_xgb.parquet", index=False)
    cat = cat_years_by_state(cells)
    _save("xgb_summary", summarise(scores[scores["design"] == "LOYO"], cat))
    print("Explainability flip check ...", flush=True)
    _save("xgb_flip_check", xgb_explainability(cells, claims, spec))
    print(f"xgb done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Self-test on simulated data with known answers
# ---------------------------------------------------------------------------
def _self_test():
    """Two simulated 'states' with the same relativities but different levels and storm years:
    the state x year fit + absorb_group_levels must recover the zone relativity and the state
    factor, fitted predictions must be unchanged by absorbing, and re-centring by state must
    leave each state's total unchanged."""
    rng = np.random.default_rng(0)
    rows = []
    for st, lvl, storm in (("VA", 0.01, 2012), ("NC", 0.02, 2018)):
        for yr in range(2010, 2021):
            for zone, zr in (("A", 1.0), ("B", 0.4)):
                for county in (f"{st}1", f"{st}2"):
                    e = 2000.0
                    mu = e * lvl * zr * (6.0 if yr == storm else 1.0)
                    rows.append({"state": st, "year": yr, "state_year": f"{st}|{yr}", "zone": zone,
                                 "county": county, "county_group": county, "exposure": e,
                                 "claims": rng.poisson(mu)})
    d = pd.DataFrame(rows)
    m = fg.fit_glm(d, ["zone"], bases={"zone": "A"}, year_col="state_year")
    ma = fg.absorb_group_levels(m, d, "state", "VA")
    assert np.allclose(m.predict(d, "fitted"), ma.predict(d, "fitted")), "absorbing changed fitted values"
    zrel = float(np.exp(ma.params["zone[B]"]))
    assert abs(zrel - 0.4) < 0.05, zrel
    # state factor = ratio of average-year levels (NC's level x 2, its storm year x 6 averaged in)
    true = (0.02 * (10 + 6) / 11) / (0.01 * (10 + 6) / 11)
    sf = float(np.exp(ma.params["state[NC]"]))
    assert abs(sf / true - 1) < 0.08, (sf, true)
    # each state's average-year predictions add up to its actual claims
    pa = ma.predict(d, "average")
    for st in ("VA", "NC"):
        mk = d["state"] == st
        assert abs(pa[mk].sum() / d.loc[mk, "claims"].sum() - 1) < 1e-9
    # re-centring county factors within each state
    fac = pd.Series({"VA1": 1.5, "VA2": 0.9, "NC1": 1.2, "NC2": 1.2})

    class _M:      # minimal stand-in for CombinedModel.predict
        def predict(self, df, county=False):
            return ma.predict(df, "average")
    f2, scales = pp.recentre(fac, _M(), d, by="state")
    for st in ("VA", "NC"):
        mk = d["state"] == st
        base = ma.predict(d[mk], "average")
        assert abs((base * d.loc[mk, "county"].map(f2)).sum() / base.sum() - 1) < 1e-9
    # rebase_by and segment A/E
    rb = rebase_by([1, 1, 2, 2], [3, 1, 0, 6], ["a", "a", "b", "b"])
    assert np.allclose(rb, [2, 2, 3, 3])
    print(f"  zone relativity {zrel:.3f} (true 0.4); state factor {sf:.3f} (true {true:.3f})")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        print("multistate self-test passed:", _self_test())
    elif cmd == "transfer":
        cmd_transfer()
    elif cmd == "refit":
        cmd_refit(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif cmd == "xgb":
        cmd_xgb()
    elif cmd == "production":
        cmd_production()
    else:
        print(__doc__)
        sys.exit(1)
