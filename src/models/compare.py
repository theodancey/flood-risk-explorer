"""
MODEL COMPARISON (Milestone 11): the GLMs vs XGBoost, head to head, on identical folds.

Plain-English summary (full explanations in LEARNING_NOTES.md, Milestone 11):
  * A FAIR comparison means every model sees the same training rows, is scored on the same
    unseen rows with the same yardstick, and gets the same loss treatment (2025 dollars, no
    cap, ICC bridge, off-balance). `fit_split` refits EVERY model on one training split, and
    `run_splits` scores them all on the matching test split. Nothing is re-tuned: XGBoost uses
    the hyperparameters frozen in models/xgb_settings.json (chosen on 2010-2020 only).
  * XGBoost frequency x severity is assembled with the SAME code as the GLM version
    (pure_premium.CombinedModel), so the off-balance and the optional Buhlmann-Straub county
    factor are computed identically for both.
  * "Is the difference real?" BOOTSTRAP: resample the evidence many times and see how much
    the difference between two models moves.
      - year bootstrap: resample the per-year scores (asks "would it hold with other weather?")
      - cluster bootstrap: inside each test year, resample county groups with replacement and
        rescore (asks "is it within-year sampling noise?"). Deviances are computed from
        per-cluster sums, so 1,000 replicates take seconds.
  * STABILITY: how much do a model's predictions for the same cells move when one year is
    dropped from training, five years are added, or (XGBoost) only the random seed changes.

Run the self-test:  .venv\\Scripts\\python.exe src\\models\\compare.py
"""

import sys
import time
from pathlib import Path

import json
import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parent
SRC = MODELS.parent
for p in (SRC, MODELS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import validation as val                      # noqa: E402
import frequency_glm as fg                    # noqa: E402
import severity_glm as sv                     # noqa: E402
import pure_premium as pp                     # noqa: E402
import xgb_models as X                        # noqa: E402

ROOT = SRC.parent
LOSS = pp.LOSS
OOF_DIR = ROOT / "data" / "processed"         # cached out-of-fold predictions (not in Git)
P_REF = pp.REF_POWER                           # Tweedie p = 1.5, the common pure-premium yardstick
CHAMPION = "GLM fxs + county"

# Model names (kept short: they label chart lines and table columns).
FREQ_MODELS = ["GLM Poisson", "GLM NB", "XGB", "Hybrid GLM"]
SEV_MODELS = ["GLM Gamma", "XGB", "XGB no-geo", "Hybrid GLM"]
PP_MODELS = ["GLM fxs + county", "GLM fxs", "GLM Tweedie", "XGB fxs", "XGB fxs no-geo + county",
             "XGB Tweedie", "Hybrid GLM fxs + county"]


# ---------------------------------------------------------------------------
# Data and settings
# ---------------------------------------------------------------------------
def xgb_table(raw):
    """XGBoost's frequency/Tweedie table (xgb_models.load_cell_table) built from raw cells."""
    m = raw[raw["exposure_matched"] & (raw["exposure"] > 0)]
    return m.groupby(X.ALL_FEATURES + ["year"], observed=True)[["exposure", "claims", LOSS]].sum().reset_index()


def nogeo_severity_settings():
    """Settings for severity XGBoost WITHOUT geography: the Milestone 10 tuning table for that
    variant, with the same pre-stated tie rule (no new search)."""
    t = pd.read_csv(X.tuning_csv("severity", "no geography"))
    row = X.choose_config(t)
    return {"params": X.params_of(row), "n_rounds": int(row["best_rounds"]), "features": X.BUILDING_FEATURES,
            "monotone": None, "dev_loyo_score": float(row["score"])}


def load_data(verbose=True):
    """Everything the comparison needs, loaded once."""
    raw = pd.read_parquet(X.CELLS_PATH)
    claims = sv.load_claims(verbose=False)              # all paid claims (ICC bridge needs ICC-only ones)
    pj = json.loads((ROOT / "models" / "pure_premium.json").read_text(encoding="utf-8"))
    freq, sev, _ = pp.load_saved_models()
    S = X.read_settings()
    D = {"raw": raw, "claims": claims, "sevdf": sv.severity_rows(claims), "xcells": xgb_table(raw),
         "settings": S, "sev_nogeo": nogeo_severity_settings(),
         "FF": list(freq.spec.features), "SF": list(sev.spec.features),
         "FB": pp.bases_without_year(freq.spec), "SB": pp.bases_without_year(sev.spec),
         "TWF": pj["tweedie_challenger"]["features"], "TW_POWER": float(pj["tweedie_challenger"]["power"]),
         "hybrid": {"freq_features": list(freq.spec.features), "freq_inter": [],
                    "sev_features": list(sev.spec.features), "sev_inter": []}}
    if verbose:
        print(f"raw cells {len(raw):,} | XGB table {len(D['xcells']):,} | severity claims {len(D['sevdf']):,} "
              f"| all paid claims {len(claims):,}")
    return D


def _years(df, years):
    return df[df["year"].isin(list(years))]


# ---------------------------------------------------------------------------
# Fitting every model on one training split
# ---------------------------------------------------------------------------
def _xgb(D, train_tbl, kind, unmatched=0.0, seed=None, settings=None):
    s = settings or D["settings"][kind]
    params = dict(s["params"])
    m_kw = {}
    if seed is not None:                      # seed sensitivity: same settings, other random seed
        old = X.SEED
        X.SEED = seed
    try:
        m = X.fit_xgb(train_tbl, kind, params, s["n_rounds"], s["features"], s.get("monotone"),
                      unmatched_losses=unmatched, **m_kw)
    finally:
        if seed is not None:
            X.SEED = old
    return m


def fit_split(D, train_years, models=None, seed=None):
    """Refit every model on `train_years`. Returns a dict of fitted models:
    freq/<name>, sev/<name>, pp/<name> (pure-premium models have .predict(cells) -> losses
    and .rate(cells) -> loss cost per policy-year)."""
    raw_tr = _years(D["raw"], train_years)
    cl_tr = _years(D["claims"], train_years)
    sev_tr = _years(D["sevdf"], train_years)
    x_tr = _years(D["xcells"], train_years)
    unm_tr = float(raw_tr.loc[~(raw_tr["exposure_matched"] & (raw_tr["exposure"] > 0)), LOSS].sum())
    H = D["hybrid"]
    out = {}
    want = (lambda n: True) if models is None else (lambda n: n in models)

    # --- GLMs (frequency x severity, with and without the county factor; hybrid; NB; Tweedie)
    glm = pp.fit_combined(raw_tr, cl_tr, D["FF"], D["SF"], D["FB"], D["SB"])
    out["freq/GLM Poisson"], out["sev/GLM Gamma"] = glm.freq, glm.sev
    out["pp/GLM fxs"] = glm
    if want("pp/GLM fxs + county"):
        out["pp/GLM fxs + county"] = pp.with_county_factors(glm, raw_tr, cl_tr, "buhlmann_straub")[0]
    hyb_differs = (H["freq_features"] != D["FF"] or H["freq_inter"] or H["sev_features"] != D["SF"] or H["sev_inter"])
    if hyb_differs and want("pp/Hybrid GLM fxs + county"):
        hb = pp.fit_combined(raw_tr, cl_tr, H["freq_features"], H["sev_features"], D["FB"], D["SB"],
                             H["freq_inter"], H["sev_inter"])
        out["freq/Hybrid GLM"], out["sev/Hybrid GLM"] = hb.freq, hb.sev
        out["pp/Hybrid GLM fxs + county"] = pp.with_county_factors(hb, raw_tr, cl_tr, "buhlmann_straub")[0]
    if want("freq/GLM NB"):
        out["freq/GLM NB"] = fg.fit_negbin(fg.aggregate(raw_tr[raw_tr["exposure"] > 0], D["FF"]), D["FF"],
                                           bases=D["FB"])
    if want("pp/GLM Tweedie"):
        out["pp/GLM Tweedie"] = pp.fit_tweedie(raw_tr, D["TWF"], D["TW_POWER"])

    # --- XGBoost (frozen settings). Frequency x severity goes through the SAME CombinedModel
    # code as the GLM, so ICC bridge, off-balance and county factor are computed identically.
    xf = _xgb(D, x_tr, "frequency", seed=seed)
    xs = _xgb(D, sev_tr, "severity", seed=seed)
    out["freq/XGB"], out["sev/XGB"] = xf, xs
    xc = pp.CombinedModel(xf, xs, sv.icc_bridge_factor(cl_tr))
    xc.off_balance = pp.off_balance_factor(xc, raw_tr)
    out["pp/XGB fxs"] = xc
    if want("pp/XGB fxs no-geo + county") or want("sev/XGB no-geo"):
        xs0 = _xgb(D, sev_tr, "severity", seed=seed, settings=D["sev_nogeo"])
        out["sev/XGB no-geo"] = xs0
        xc0 = pp.CombinedModel(xf, xs0, sv.icc_bridge_factor(cl_tr))
        xc0.off_balance = pp.off_balance_factor(xc0, raw_tr)
        out["pp/XGB fxs no-geo + county"] = pp.with_county_factors(xc0, raw_tr, cl_tr, "buhlmann_straub")[0]
    if want("pp/XGB Tweedie"):
        out["pp/XGB Tweedie"] = _xgb(D, x_tr, "tweedie", unmatched=unm_tr, seed=seed)

    out["_train"] = {"years": sorted(int(y) for y in train_years),
                     "pp_rate": float(raw_tr[LOSS].sum() / raw_tr["exposure"].sum()),
                     "freq_rate": float(x_tr["claims"].sum() / x_tr["exposure"].sum()),
                     "sev_mean": float(sev_tr[sv.TARGET].mean()), "unmatched": unm_tr}
    return out


def pp_rate(model, df):
    """Loss cost per policy-year for any pure-premium model (CombinedModel, TweedieGLM, XGBModel)."""
    return np.asarray(model.rate(df), float)


def freq_rate(model, df):
    return np.asarray(model.predict(df.assign(exposure=1.0), "average"), float)


def sev_rate(model, df):
    return np.asarray(model.predict(df, "average"), float)


# ---------------------------------------------------------------------------
# Scoring one split
# ---------------------------------------------------------------------------
def _score_groups(kind, test, preds, train_info, all_losses_by_year=None, holdout=False):
    """Score rows: one per test year (and a 'pooled' row for the holdout)."""
    rows = []
    years = sorted(test["year"].unique())
    groups = [(int(y), test["year"].to_numpy() == y) for y in years]
    if holdout:
        groups.append(("pooled", np.ones(len(test), bool)))
    for g, mask in groups:
        t = test[mask]
        if kind == "frequency":
            pnull = t["exposure"].to_numpy(float) * train_info["freq_rate"]
        elif kind == "severity":
            pnull = np.full(len(t), train_info["sev_mean"])
        for name, pr in preds.items():
            pr = pr[mask]
            if kind == "frequency":
                s = fg.score_fold(t, pr, pnull)
            elif kind == "severity":
                s = sv.score_fold(t, pr, pnull)
            else:
                yrs = t["year"].unique()
                allt = float(sum(all_losses_by_year[int(y)] for y in yrs))
                s = pp.score_cells(t, pr, train_info["pp_rate"], all_test_losses=allt)
            rows.append({"kind": kind, "model": name, "test_year": g,
                         "cat_year": val.is_cat_year(g) if g != "pooled" else False, **s})
    return rows


def score_split(D, models, test_years, holdout=False, label=None):
    """Scores and out-of-fold predictions on the test years, for every fitted model."""
    tr = models["_train"]
    # frequency: XGBoost's 10-feature table (as Milestone 10); severity: individual claims;
    # pure premium: the raw county-level cells with exposure (as Milestone 9).
    ft = _years(D["xcells"], test_years).reset_index(drop=True)
    st = _years(D["sevdf"], test_years).reset_index(drop=True)
    raw_te = _years(D["raw"], test_years)
    pt = raw_te[raw_te["exposure"] > 0].reset_index(drop=True)
    all_losses = raw_te.groupby("year")[LOSS].sum().to_dict()

    fp = {k.split("/", 1)[1]: freq_rate(m, ft) * ft["exposure"].to_numpy(float)
          for k, m in models.items() if k.startswith("freq/")}
    sp = {k.split("/", 1)[1]: sev_rate(m, st) for k, m in models.items() if k.startswith("sev/")}
    prate = {k.split("/", 1)[1]: pp_rate(m, pt) for k, m in models.items() if k.startswith("pp/")}
    ppred = {k: r * pt["exposure"].to_numpy(float) for k, r in prate.items()}

    rows = _score_groups("frequency", ft, fp, tr, holdout=holdout) + \
        _score_groups("severity", st, sp, tr, holdout=holdout) + \
        _score_groups("pure_premium", pt, ppred, tr, all_losses, holdout=holdout)
    sc = pd.DataFrame(rows)
    sc["split"] = label

    keep_f = ["year", "exposure", "claims", *X.ALL_FEATURES]
    of = ft[keep_f].copy()
    for k, v in fp.items():
        of[k] = v
        of["rank:" + k] = v / ft["exposure"].to_numpy(float) / tr["freq_rate"]   # level-free
    os_ = st[["year", sv.TARGET, *X.ALL_FEATURES]].copy()
    for k, v in sp.items():
        os_[k] = v
        os_["rank:" + k] = v / tr["sev_mean"]
    op = pt[["year", "county", "exposure", "claims", LOSS, *X.ALL_FEATURES]].copy()
    for k, v in ppred.items():
        op[k] = v
        op["rank:" + k] = prate[k] / tr["pp_rate"]
    for o in (of, os_, op):
        o["split"] = label
    return sc, {"frequency": of, "severity": os_, "pure_premium": op}


def run_splits(D, design, grid=None, verbose=True):
    """design: "loyo16" (16 folds), "dev" (11 development folds) or "holdout".
    Returns (scores, {kind: OOF table}, {split label: {model: grid rates}} if grid given)."""
    if design == "holdout":
        splits = [("holdout", val.DEV_YEARS, val.HOLDOUT_YEARS)]
    else:
        years = val.ALL_YEARS if design == "loyo16" else val.DEV_YEARS
        splits = [(y, [z for z in years if z != y], [y]) for y in years]
    scores, oof, grids = [], {"frequency": [], "severity": [], "pure_premium": []}, {}
    t0 = time.time()
    for label, tr_years, te_years in splits:
        assert not set(tr_years) & set(te_years)
        m = fit_split(D, tr_years)
        sc, o = score_split(D, m, te_years, holdout=(design == "holdout"), label=label)
        scores.append(sc)
        for k in oof:
            oof[k].append(o[k])
        if grid is not None:
            grids[label] = grid_rates(m, grid)
        if verbose:
            print(f"  {design} split {label}: {time.time() - t0:.0f}s", flush=True)
    return (pd.concat(scores, ignore_index=True), {k: pd.concat(v, ignore_index=True) for k, v in oof.items()},
            grids)


def grid_rates(models, grid):
    """Each model's average-year rate on a fixed grid of cells (for stability checks)."""
    out = {}
    for k, m in models.items():
        if k.startswith("_"):
            continue
        kind, name = k.split("/", 1)
        f = {"freq": freq_rate, "sev": sev_rate, "pp": pp_rate}[kind]
        out[k] = f(m, grid)
    return out


def cache_path(name):
    return OOF_DIR / f"m11_{name}.parquet"


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(scores, kind, split=None):
    """Mean improvement (all / normal / cat years), Gini, raw A/E, years beating the champion."""
    s = scores[(scores["kind"] == kind) & (scores["test_year"] != "pooled")]
    if split is not None:
        s = s[s["split"].astype(str).isin([str(x) for x in split])] if isinstance(split, (list, tuple)) else s
    g = s.groupby("model", sort=False)
    t = pd.DataFrame({"mean_improvement": g["improvement_rebased_pct"].mean(),
                      "normal_years": s[~s["cat_year"]].groupby("model", sort=False)["improvement_rebased_pct"].mean(),
                      "cat_years": s[s["cat_year"]].groupby("model", sort=False)["improvement_rebased_pct"].mean(),
                      "gini": g["gini"].mean(), "folds": g.size()})
    return t


def wins_vs(scores, kind, a, b, metric="improvement_rebased_pct"):
    """Per-year comparison of model b vs model a: years b wins, mean difference, sign-test p."""
    from scipy import stats
    s = scores[(scores["kind"] == kind) & (scores["test_year"] != "pooled")]
    pa = s[s["model"] == a].set_index("test_year")[metric]
    pb = s[s["model"] == b].set_index("test_year")[metric]
    d = (pb - pa).dropna()
    wins = int((d > 0).sum())
    p = float(stats.binomtest(wins, len(d), 0.5).pvalue) if len(d) else np.nan
    return {"a": a, "b": b, "years": len(d), "b_wins": wins, "mean_diff_pts": float(d.mean()), "sign_test_p": p}


def rebase_by_year(o, col, actual):
    """Predictions scaled so each year's total equals that year's actual total."""
    return o[col] * o.groupby("year")[actual].transform("sum") / o.groupby("year")[col].transform("sum")


def segment_ae(o, models, features, actual, champion=None):
    """Out-of-fold A/E by feature level, predictions re-based within each year, pooled."""
    rows = []
    rb = {m: rebase_by_year(o, m, actual) for m in models}
    for f in features:
        a = o.groupby(f, observed=True)[actual].sum()
        for m in models:
            e = rb[m].groupby(o[f], observed=True).sum()
            for lvl in a.index:
                rows.append({"feature": f, "level": str(lvl), "model": m, "actual": float(a[lvl]),
                             "expected": float(e[lvl]), "ae": float(a[lvl] / e[lvl])})
    return pd.DataFrame(rows)


def end_decile_error(dl):
    """Mean |model - actual| in the two end deciles of a double-lift table (a and b)."""
    end = dl.iloc[[0, -1]]
    return {"a": float((end["a_relative"] - end["losses_relative"]).abs().mean()),
            "b": float((end["b_relative"] - end["losses_relative"]).abs().mean())}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def year_bootstrap(scores, kind, a, b, metric="improvement_rebased_pct", B=10_000, seed=11):
    """Resample the test YEARS with replacement; CI of mean(b) - mean(a) over the years."""
    s = scores[(scores["kind"] == kind) & (scores["test_year"] != "pooled")]
    pa = s[s["model"] == a].set_index("test_year")[metric]
    pb = s[s["model"] == b].set_index("test_year")[metric].reindex(pa.index)
    d = (pb - pa).to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    reps = d[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"estimate": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "share_reps_positive": float((reps > 0).mean())}


def _dev_parts(kind, y, mu, w):
    """Per-row pieces whose cluster sums give the re-based deviance for ANY rescaling
    (re-basing multiplies mu by c = sum(y w) / sum(mu w), which depends on the resample)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    if kind == "frequency":                   # Poisson: D = A - 2 Y log(Y/M) + 2Y, A = sum 2[y log(y/mu) - y]
        ylog = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
        return np.column_stack([2 * (ylog - y), y, mu])
    if kind == "severity":                    # Gamma: D = 2[(1/c) sum y/mu - n - sum log(y/mu) + n log c]
        return np.column_stack([y / mu, np.ones_like(y), np.log(y / mu), y, mu])
    p = P_REF                                 # Tweedie (rates r = y/e, weights e)
    e = np.asarray(w, float)
    r, m = y / e, mu / e
    return np.column_stack([e * r ** (2 - p) / ((1 - p) * (2 - p)), e * r * m ** (1 - p), e * m ** (2 - p),
                            e * r, e * m])


def _dev_from_sums(kind, S):
    """Re-based deviance from summed parts; S has shape (..., n_parts)."""
    if kind == "frequency":
        A, Y, M = S[..., 0], S[..., 1], S[..., 2]
        return A - 2 * Y * np.log(Y / M) + 2 * Y
    if kind == "severity":
        q, n, lg, Y, M = (S[..., i] for i in range(5))
        c = Y / M
        return 2 * (q / c - n - lg + n * np.log(c))
    p = P_REF
    T0, T1, T2, Y, M = (S[..., i] for i in range(5))
    c = Y / M
    return 2 * (T0 - c ** (1 - p) * T1 / (1 - p) + c ** (2 - p) * T2 / (2 - p))


def _weighted_gini(y, rank, w, mult):
    """Gini for many replicates at once: rows of `mult` are per-row replicate multiplicities.
    Rows are sorted by `rank` once; the replicate weights then rescale exposure and actuals."""
    order = np.argsort(rank, kind="mergesort")
    y, w, mult = y[order], w[order], mult[:, order]
    ew, ey = mult * w, mult * y
    x = np.cumsum(ew, axis=1)
    x = np.concatenate([np.zeros((len(mult), 1)), x / x[:, -1:]], axis=1)
    c = np.cumsum(ey, axis=1)
    c = np.concatenate([np.zeros((len(mult), 1)), c / c[:, -1:]], axis=1)
    return 1 - 2 * np.trapezoid(c, x, axis=1)


def cluster_bootstrap(o, kind, models, null_col=None, B=1000, seed=7, cluster="county_group",
                      group="year", chunk=100):
    """Within each `group` (test year), resample `cluster`s with replacement B times and
    recompute every model's re-based deviance improvement and Gini; then average over groups
    (each year once). Returns {model: {"improvement": (B,), "gini": (B,)}}.
    o: an OOF table with actual column, exposure (not severity), model prediction columns and
    the null-model prediction in `null_col` (computed here if None)."""
    rng = np.random.default_rng(seed)
    actual = {"frequency": "claims", "severity": sv.TARGET, "pure_premium": LOSS}[kind]
    dkind = "tweedie" if kind == "pure_premium" else kind
    res = {m: {"improvement": np.zeros(B), "gini": np.zeros(B)} for m in models}
    groups = list(o.groupby(group, sort=True))
    for g, t in groups:
        cl = pd.factorize(t[cluster].astype(str))[0]
        K = cl.max() + 1
        y = t[actual].to_numpy(float)
        w = t["exposure"].to_numpy(float) if kind != "severity" else np.ones(len(t))
        if kind == "severity":
            null = np.full(len(t), y.mean())          # within-year rebased null = flat
        else:
            null = w * y.sum() / w.sum()
        counts = rng.multinomial(K, np.full(K, 1 / K), size=B).astype(float)     # (B, K)
        P0 = np.zeros((K, _dev_parts(dkind, y[:1] + 1, null[:1], w[:1]).shape[1]))
        np.add.at(P0, cl, _dev_parts(dkind, y, null, w))
        D0 = _dev_from_sums(dkind, counts @ P0)
        for m in models:
            mu = t[m].to_numpy(float)
            P = np.zeros_like(P0)
            np.add.at(P, cl, _dev_parts(dkind, y, mu, w))
            D = _dev_from_sums(dkind, counts @ P)
            res[m]["improvement"] += 100 * (1 - D / D0) / len(groups)
            rank = mu / w
            gi = np.concatenate([_weighted_gini(y, rank, w, counts[i:i + chunk][:, cl])
                                 for i in range(0, B, chunk)])
            res[m]["gini"] += gi / len(groups)
    return res


def bootstrap_diff(res, a, b, metric):
    d = res[b][metric] - res[a][metric]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"a": a, "b": b, "metric": metric, "ci_low": float(lo), "ci_high": float(hi),
            "share_reps_positive": float((d > 0).mean()), "boot_mean": float(d.mean())}


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def relative_shift(rate_a, rate_b, w):
    """Level-free change between two predictions of the same cells: |log(relative a / relative b)|,
    each normalised to its exposure-weighted mean. Returns weighted mean and P95 (as % change)."""
    ra = np.asarray(rate_a, float) / np.average(rate_a, weights=w)
    rb = np.asarray(rate_b, float) / np.average(rate_b, weights=w)
    d = np.abs(np.log(ra / rb))
    order = np.argsort(d)
    cw = np.cumsum(np.asarray(w, float)[order]) / np.sum(w)
    p95 = d[order][np.searchsorted(cw, 0.95)]
    return {"mean_pct": float(100 * (np.exp(np.average(d, weights=w)) - 1)),
            "p95_pct": float(100 * (np.exp(p95) - 1)), "max_pct": float(100 * (np.exp(d.max()) - 1))}


# ---------------------------------------------------------------------------
# Hybrid: GLM + interactions / second-pass features (dev years only)
# ---------------------------------------------------------------------------
def frequency_second_pass(df, features, inter, dropped, loyo_years, alpha=0.05):
    """frequency_glm has no second pass (Milestone 8 added one for severity); same logic:
    re-test each dropped feature against the CURRENT model with the keep rule, add the best
    passer, repeat. Returns (table, final features)."""
    cache, rows, feats, left, rnd = {}, [], list(features), list(dropped), 1
    while left:
        tested = []
        for f in left:
            r = fg.compare_models(df, feats, inter, feats + [f], inter, loyo_years, cache=cache)
            ok = bool(r["cluster_Wald_p"] < alpha and r["d_devLOYO_pct"] > 0)
            tested.append((f, ok, r))
            rows.append({"round": rnd, "candidate": f, "passes": ok, "added": False, **r})
        passing = [(f, r) for f, ok, r in tested if ok]
        if not passing:
            break
        best = max(passing, key=lambda t: t[1]["d_devLOYO_pct"])[0]
        rows[len(rows) - len(tested) + [t[0] for t in tested].index(best)]["added"] = True
        feats.append(best)
        left.remove(best)
        rnd += 1
    return pd.DataFrame(rows), feats


HYBRID_FREQ_INTER = [["construction_band", "flood_zone_group"]]          # top XGB frequency interaction
HYBRID_SEV_INTER = [["building_coverage_band", "occupancy_group"],       # top XGB severity interaction
                    ["elevated", "flood_zone_group"]]                    # flagged in Milestone 8 (small)
FREQ_DROPPED = ["building_coverage_band", "occupancy_group", "elevated", "post_firm", "primary_residence"]


def hybrid_tests(D, verbose=True):
    """The hybrid search, on the 11 development years only, with the existing keep rule.
    Frequency: (1) year built x zone; (2) second pass over the features Milestone 7 dropped.
    Severity: (1) coverage x occupancy; (2) elevated x zone. Each step is tested against the
    model as it stands after the previous step. Returns (table, hybrid spec dict)."""
    rows = []
    fdf = _years(D["xcells"], val.DEV_YEARS)
    sdf = _years(D["sevdf"], val.DEV_YEARS)
    ff, fi = list(D["FF"]), []
    fcache = {}
    for inter in HYBRID_FREQ_INTER:
        r = fg.compare_models(fdf, ff, fi, ff, fi + [inter], val.DEV_YEARS, cache=fcache)
        keep = bool(r["cluster_Wald_p"] < 0.05 and r["d_devLOYO_pct"] > 0)
        rows.append({"model": "frequency", "step": " x ".join(inter), "kept": keep, **r})
        if keep:
            fi = fi + [inter]
        if verbose:
            print(f"  frequency + {' x '.join(inter)}: Wald p {r['cluster_Wald_p']:.3g}, "
                  f"dev-LOYO {r['d_devLOYO_pct']:+.2f} pts ({r['devLOYO_folds_better']}) -> {keep}", flush=True)
    sp_tbl, ff2 = frequency_second_pass(fdf, ff, fi, FREQ_DROPPED, val.DEV_YEARS)
    for _, r in sp_tbl.iterrows():
        rows.append({"model": "frequency", "step": f"second pass round {r['round']}: + {r['candidate']}",
                     "kept": bool(r["added"]), **{k: r[k] for k in r.index if k not in ("round", "candidate", "passes", "added")}})
        if verbose:
            print(f"  frequency second pass r{r['round']} + {r['candidate']}: Wald p {r['cluster_Wald_p']:.3g}, "
                  f"dev-LOYO {r['d_devLOYO_pct']:+.2f} ({r['devLOYO_folds_better']}) passes={r['passes']} "
                  f"added={r['added']}", flush=True)
    sf, si = list(D["SF"]), []
    scache = {}
    for inter in HYBRID_SEV_INTER:
        r = sv.compare_models(sdf, sf, sf, val.DEV_YEARS, cache=scache, cur_inter=si, new_inter=si + [inter])
        keep = bool(r["cluster_Wald_p"] < 0.05 and r["d_devLOYO_pct"] > 0)
        rows.append({"model": "severity", "step": " x ".join(inter), "kept": keep, **r})
        if keep:
            si = si + [inter]
        if verbose:
            print(f"  severity + {' x '.join(inter)}: Wald p {r['cluster_Wald_p']:.3g}, "
                  f"dev-LOYO {r['d_devLOYO_pct']:+.2f} pts ({r['devLOYO_folds_better']}) -> {keep}", flush=True)
    spec = {"freq_features": ff2, "freq_inter": fi, "sev_features": sf, "sev_inter": si}
    return pd.DataFrame(rows), spec


def cmd_run():
    """Everything expensive, cached for notebook 09 (about 20-40 minutes on a laptop)."""
    t0 = time.time()
    D = load_data()
    print("1. Hybrid tests (development years only)")
    ht, spec = hybrid_tests(D)
    ht.to_csv(ROOT / "reports" / "08_hybrid_tests.csv", index=False)
    (OOF_DIR / "m11_hybrid_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    D["hybrid"] = spec
    print("   hybrid spec:", spec, f"({time.time() - t0:.0f}s)")
    raw = D["raw"]
    grid = raw[(raw["year"] == 2025) & (raw["exposure"] > 0)].reset_index(drop=True)
    grids = {}
    for design in ("loyo16", "dev", "holdout"):
        print(f"2. {design}")
        sc, oof, g = run_splits(D, design, grid=grid if design != "dev" else None)
        sc.to_csv(ROOT / "reports" / f"08_scores_{design}.csv", index=False)
        for k, o in oof.items():
            o.to_parquet(cache_path(f"oof_{k}_{design}"), index=False)
        for lab, v in g.items():
            grids[f"{design}:{lab}"] = v
    print("3. All-years fit and XGBoost seed refits")
    grids["all"] = grid_rates(fit_split(D, val.ALL_YEARS), grid)
    for seed in (1, 2, 3):
        m = fit_split(D, val.ALL_YEARS, models={"pp/XGB Tweedie"}, seed=seed)
        grids[f"seed:{seed}"] = {k: v for k, v in grid_rates(m, grid).items() if "XGB" in k}
        print(f"   seed {seed} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for split, d in grids.items():
        for model, r in d.items():
            rows.append(pd.DataFrame({"split": split, "model": model, "cell": np.arange(len(r)), "rate": r}))
    pd.concat(rows, ignore_index=True).to_parquet(cache_path("grid_rates"), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} minutes")


# ---------------------------------------------------------------------------
# Self-test on simulated data with known answers
# ---------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    ok = True

    # 1. Re-based deviance from cluster sums == direct computation, for all three kinds.
    n = 400
    e = rng.uniform(0.5, 3, n)
    mu_f = e * rng.uniform(0.01, 0.2, n)
    y_f = rng.poisson(mu_f * 1.3).astype(float)
    c = y_f.sum() / mu_f.sum()
    direct = fg.poisson_deviance(y_f, mu_f * c)
    via = _dev_from_sums("frequency", _dev_parts("frequency", y_f, mu_f, e).sum(0))
    ok &= np.isclose(direct, via)
    y_s = rng.gamma(1.2, 20_000, n)
    mu_s = rng.uniform(10_000, 40_000, n)
    direct = sv.gamma_deviance(y_s, sv.rebase(mu_s, y_s.sum()))
    via = _dev_from_sums("severity", _dev_parts("severity", y_s, mu_s, None).sum(0))
    ok &= np.isclose(direct, via)
    y_t = np.where(rng.random(n) < 0.1, rng.gamma(1, 30_000, n), 0.0)
    mu_t = e * rng.uniform(50, 400, n)
    c = y_t.sum() / mu_t.sum()
    direct = pp.tweedie_deviance(y_t / e, mu_t * c / e, P_REF, e)
    via = _dev_from_sums("tweedie", _dev_parts("tweedie", y_t, mu_t, e).sum(0))
    ok &= np.isclose(direct, via)
    print(f"  cluster-sum deviances match direct formulas: {ok}")

    # 2. Weighted Gini with all-ones multiplicities == fg.gini.
    g1 = _weighted_gini(y_f, mu_f / e, e, np.ones((1, n)))[0]
    ok2 = np.isclose(g1, fg.gini(y_f, mu_f, e))
    print(f"  replicate Gini (weights 1) = fg.gini: {ok2}")
    ok &= ok2

    # 3. Cluster bootstrap: a model that knows the truth beats a flat model with a CI above 0;
    #    two identical models give a CI of exactly 0.
    years = np.repeat(np.arange(5), n // 5)
    true = rng.lognormal(0, 0.8, n)
    expo = rng.uniform(1, 5, n)
    y = rng.poisson(0.2 * true * expo).astype(float)
    o = pd.DataFrame({"year": years, "county_group": rng.integers(0, 20, n), "exposure": expo, "claims": y,
                      "good": 0.2 * true * expo, "flat": 0.2 * expo, "good2": 0.2 * true * expo})
    r = cluster_bootstrap(o, "frequency", ["good", "flat", "good2"], B=200)
    d = bootstrap_diff(r, "flat", "good", "improvement")
    z = bootstrap_diff(r, "good", "good2", "gini")
    ok3 = d["ci_low"] > 0 and abs(z["ci_low"]) < 1e-12 and abs(z["ci_high"]) < 1e-12
    print(f"  cluster bootstrap: truth beats flat (CI {d['ci_low']:.1f} to {d['ci_high']:.1f}); identical = 0: {ok3}")
    ok &= ok3

    # 4. Year bootstrap and sign test on a toy score table.
    sc = pd.DataFrame([{"kind": "frequency", "model": m, "test_year": yr, "cat_year": False,
                        "improvement_rebased_pct": v + (2 if m == "B" else 0)}
                       for yr, v in enumerate(rng.normal(5, 1, 12)) for m in ("A", "B")])
    yb = year_bootstrap(sc, "frequency", "A", "B", B=2000)
    wv = wins_vs(sc, "frequency", "A", "B")
    ok4 = np.isclose(yb["estimate"], 2) and np.isclose(yb["ci_low"], 2) and wv["b_wins"] == 12
    print(f"  year bootstrap of a constant +2 difference = [2, 2], 12/12 wins: {ok4}")
    ok &= ok4

    # 5. Stability: identical predictions -> 0; a pure level shift -> 0; a real change -> > 0.
    w = rng.uniform(1, 2, 50)
    a = rng.lognormal(0, 0.5, 50)
    s0, s1, s2 = relative_shift(a, a, w), relative_shift(a, 3 * a, w), relative_shift(a, a * rng.lognormal(0, 0.1, 50), w)
    ok5 = s0["mean_pct"] < 1e-9 and s1["mean_pct"] < 1e-9 and s2["mean_pct"] > 1
    print(f"  stability metric ignores level, detects change: {ok5}")
    ok &= ok5
    return bool(ok)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        cmd_run()
        sys.exit(0)
    passed = _self_test()
    print("compare self-test passed:", passed)
    sys.exit(0 if passed else 1)
