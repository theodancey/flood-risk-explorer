"""
GLOBAL EXPLANATIONS (Milestone 12): SHAP values for the XGBoost challengers and the exact
equivalent for the production GLM, on the same rows and the same (log) scale, so the two
models' explanations can be compared like for like.

SHAP in one line: a prediction's SHAP values split "prediction minus the average prediction"
into one share per feature, fairly (each feature's average extra contribution over every
order in which features could be added: the Shapley value from game theory).

- XGBoost: exact TreeSHAP from XGBoost itself (`pred_contribs`, `pred_interactions`). Values are
  on the log scale of the trees (the model's "margin"), so exp(value) is a multiplier. One-hot
  columns (occupancy, county group) are summed back into their feature.
- GLM: on the log scale the GLM is a sum of terms, so its SHAP values have a closed form:
  main effect: term(x) - average term; an interaction g(a, b) is shared between a and b by the
  exact two-player Shapley formula. Background = the exposure-weighted rows given.

    .venv\\Scripts\\python.exe src\\global_explain.py      # self-test
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
for p in (SRC, SRC / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import explain                                 # noqa: E402

FEATURES = explain.FEATURES


# ---------------------------------------------------------------------------
# GLM: exact interventional SHAP on the log scale
# ---------------------------------------------------------------------------
def _wmean(x, w):
    return float((x * w).sum() / w.sum())


def glm_shap(tables, df, weight=None, parts=("frequency", "severity", "pure_premium")):
    """SHAP values (log scale) of the production GLM for each row of df.
    tables: explain.RatingTables; df needs FEATURES (+ 'county' for the pure premium).
    weight: background weights (exposure for cells, 1 per claim for claims).
    Returns {part: DataFrame (rows x players)}; players = FEATURES (+ 'county' for pure premium).
    Each row's values sum to log(prediction) - weighted mean of log(prediction)."""
    w = np.ones(len(df)) if weight is None else np.asarray(weight, float)
    codes = tables.codes(df)
    out = {}
    for part in parts:
        phi = {}
        for f in FEATURES:
            if part == "frequency":
                term = tables.main_f[f][codes[f]]
            elif part == "severity":
                term = tables.main_s[f][codes[f]]
            else:
                term = tables.main_f[f][codes[f]] + tables.main_s[f][codes[f]]
            phi[f] = term - _wmean(term, w)
        for ipart, a, b, tab in tables.pairs:
            if part != "pure_premium" and ipart != part:
                continue
            ca, cb = codes[a], codes[b]
            pa = np.bincount(ca, weights=w, minlength=tab.shape[0]) / w.sum()   # marginal of a
            pb = np.bincount(cb, weights=w, minlength=tab.shape[1]) / w.sum()   # marginal of b
            g = tab[ca, cb]
            e_all = _wmean(g, w)                    # v(empty): joint average
            v_a = (tab @ pb)[ca]                    # E over b of g(x_a, B)
            v_b = (pa @ tab)[cb]                    # E over a of g(A, x_b)
            phi[a] = phi[a] + 0.5 * ((v_a - e_all) + (g - v_b))
            phi[b] = phi[b] + 0.5 * ((v_b - e_all) + (g - v_a))
        if part == "pure_premium" and "county" in df:
            lcf = tables.county_log_factor(tables.county_codes(df["county"]))
            phi["county"] = lcf - _wmean(lcf, w)
        out[part] = pd.DataFrame(phi, index=df.index)
    return out


# ---------------------------------------------------------------------------
# XGBoost: TreeSHAP grouped to features
# ---------------------------------------------------------------------------
def _groups(encoder):
    cols = encoder.columns
    feats = list(dict.fromkeys(encoder.source(c) for c in cols))
    idx = {f: [i for i, c in enumerate(cols) if encoder.source(c) == f] for f in feats}
    return feats, idx


def xgb_shap(model, df):
    """(SHAP DataFrame rows x features, bias) on the tree (log) scale, without the year level /
    exposure offset. Rows sum (with the bias) to model.tree_margin(df) (checked)."""
    import xgboost as xgb
    X = model.encoder.transform(df)
    d = xgb.DMatrix(X, feature_names=model.encoder.columns, base_margin=np.zeros(len(df)))
    c = model.booster.predict(d, pred_contribs=True)
    feats, idx = _groups(model.encoder)
    out = pd.DataFrame({f: c[:, idx[f]].sum(axis=1) for f in feats}, index=df.index).astype(float)
    bias = c[:, -1].astype(float)
    err = np.abs(out.sum(axis=1).to_numpy() + bias - model.tree_margin(df)).max()
    assert err < 1e-3, f"SHAP additivity failed ({err})"
    return out, bias


def xgb_interactions(model, df):
    """Mean |SHAP interaction value| between each pair of features (off-diagonal, both halves
    added: the full interaction effect), grouped to features. Returns a features x features
    DataFrame."""
    import xgboost as xgb
    X = model.encoder.transform(df)
    d = xgb.DMatrix(X, feature_names=model.encoder.columns, base_margin=np.zeros(len(df)))
    inter = model.booster.predict(d, pred_interactions=True)[:, :-1, :-1]
    feats, idx = _groups(model.encoder)
    M = np.zeros((len(feats), len(feats)))
    for i, a in enumerate(feats):
        for j, b in enumerate(feats):
            block = inter[:, idx[a]][:, :, idx[b]].sum(axis=(1, 2))
            M[i, j] = np.abs(block).mean() * (2 if i != j else 1)
    return pd.DataFrame(M, index=feats, columns=feats), inter


def interaction_values_by_feature(model, inter, a, b):
    """Per-row interaction value between features a and b (both halves), from xgb_interactions."""
    feats, idx = _groups(model.encoder)
    return inter[:, idx[a]][:, :, idx[b]].sum(axis=(1, 2)) * 2


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def importance(phi, weight=None):
    """Weighted mean |SHAP| per feature (log scale), plus the 'typical multiplier' exp(mean |phi|):
    how far, as a factor, this feature typically moves a prediction from the average."""
    w = np.ones(len(phi)) if weight is None else np.asarray(weight, float)
    m = {c: _wmean(np.abs(phi[c].to_numpy()), w) for c in phi.columns}
    s = pd.Series(m).sort_values(ascending=False)
    return pd.DataFrame({"mean_abs_shap": s, "typical_multiplier": np.exp(s), "share_pct": 100 * s / s.sum()})


def effect_by_level(phi, df, feature, weight=None):
    """Weighted mean SHAP value of `feature` for each of its levels, as a multiplier vs average."""
    w = np.ones(len(df)) if weight is None else np.asarray(weight, float)
    g = pd.DataFrame({"lvl": df[feature].astype(str).to_numpy(), "phi": phi[feature].to_numpy() * w, "w": w})
    a = g.groupby("lvl").sum()
    return np.exp(a["phi"] / a["w"])


def _self_test():
    ok = True
    t = explain.load(with_model=False).t
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f: rng.choice(t.levels[f], 4000) for f in FEATURES})
    df["county"] = rng.choice(t.counties, 4000)
    w = rng.uniform(0.1, 3, 4000)
    phi = glm_shap(t, df, w)
    lp = np.log(t.rate(df)["pure_premium"].to_numpy())
    good = np.allclose(phi["pure_premium"].sum(axis=1).to_numpy(), lp - _wmean(lp, w))
    print(f"  GLM SHAP rows sum to log(prediction) - average: {good}")
    ok &= good
    # brute-force exact interventional Shapley on a 3-player toy for the interaction formula
    tab = np.array([[0.0, 0.3], [0.2, -0.5]])
    a, b = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    ww = rng.uniform(0.5, 2, 500)
    pa, pb = np.bincount(a, ww) / ww.sum(), np.bincount(b, ww) / ww.sum()
    i = 7
    v0 = (tab[a, b] * ww).sum() / ww.sum()
    va, vb, vab = (tab[a[i], b] * ww).sum() / ww.sum(), (tab[a, b[i]] * ww).sum() / ww.sum(), tab[a[i], b[i]]
    brute = 0.5 * ((va - v0) + (vab - vb))
    formula = 0.5 * (((tab @ pb)[a[i]] - v0) + (vab - (pa @ tab)[b[i]]))
    good = np.isclose(brute, formula)
    print(f"  interaction share formula = brute-force Shapley: {good}")
    return bool(ok & good)


if __name__ == "__main__":
    passed = _self_test()
    print("global_explain self-test passed:", passed)
    sys.exit(0 if passed else 1)
