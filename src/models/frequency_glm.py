"""
Claim FREQUENCY model: a Poisson GLM with a log link and log(exposure) as the offset.

Plain-English summary (full explanations in LEARNING_NOTES.md, Milestone 7):
  * We predict the number of paid claims in each rating cell (county group x year x
    building features). A Poisson GLM says:
        expected claims = exposure x exp(b0 + b1*x1 + b2*x2 + ...)
    so each feature MULTIPLIES the claim rate by exp(b), its "relativity".
  * log(exposure) is the OFFSET: a term with its coefficient fixed at 1. It turns the model
    into a model of the RATE (claims per policy-year), so a cell with twice the exposure
    expects twice the claims, all else equal.
  * Calendar year enters as a factor ("C(year)"): it soaks up the storm-year swings
    (Irene, Sandy, Matthew), so they don't distort the building-feature relativities.
    A year the model has never seen gets the AVERAGE-YEAR level (see FreqGLM.predict).

Why hand-built dummy columns instead of statsmodels formulas? Speed: the formula interface
took ~18 s per fit on this data and ~0.2 s this way, and leave-one-year-out needs hundreds
of fits. The maths is identical.

Only rows with exposure > 0 are used (the 111 "unmatched" cells with claims but no exposure
are dropped: a rate needs exposure, and log(0) is undefined). See docs/modeling_dataset.md.

Run the self-test:  .venv\\Scripts\\python.exe src\\models\\frequency_glm.py
"""

import itertools
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:                  # so "import validation" works when run as a script
    sys.path.insert(0, str(SRC))
import validation as val                      # noqa: E402

ROOT = SRC.parent
DATA_PATH = ROOT / "data" / "processed" / "va_model_data.parquet"

# Every candidate feature from the Milestone 6 shortlist, plus the two coastal flags.
CANDIDATES = ["flood_zone_group", "construction_band", "building_coverage_band", "occupancy_group",
              "elevated", "post_firm", "primary_residence", "floors_group",
              "coastal_tidal", "coastal", "county_group"]

# Natural display order for the ordered features (others: base first, then by exposure).
LEVEL_ORDER = {
    "construction_band": ["before 1960", "1960-1979", "1980-1999", "2000 or later"],
    "building_coverage_band": ["$0 (contents only)", "under $100K", "$100K-$249K", "$250K or more"],
    "floors_group": ["1", "2", "3+"],
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_frequency_data(path=DATA_PATH, verbose=True):
    """The modelling table with zero-exposure (unmatched) rows removed."""
    df = pd.read_parquet(path)
    keep = df["exposure_matched"] & (df["exposure"] > 0)
    if verbose:
        d = df[~keep]
        print(f"Dropped {len(d)} zero-exposure cells holding {int(d['claims'].sum())} paid claims "
              f"({d['claims'].sum() / df['claims'].sum():.2%}). Kept {keep.sum():,} cells, "
              f"{df.loc[keep, 'exposure'].sum():,.2f} policy-years, {int(df.loc[keep, 'claims'].sum()):,} paid claims.")
    return df[keep].reset_index(drop=True)


def aggregate(df, keys):
    """Add up exposure and claims over rows that share the same `keys` (+ year).

    For a Poisson GLM whose features are all in `keys`, this changes nothing that matters:
    the coefficients, fitted totals and DIFFERENCES in deviance between models are identical
    (the cell sums are "sufficient statistics"). It only makes fitting faster. What it does
    change is the absolute deviance and the cell-level dispersion, so all models that are
    compared with each other must be fitted on the SAME aggregated table."""
    keys = [k for k in keys if k != "year"] + ["year"]
    return df.groupby(keys, observed=True)[["exposure", "claims"]].sum().reset_index()


# ---------------------------------------------------------------------------
# Design matrix: one 0/1 column per non-base level
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    """Which features are in the model, their levels and base levels."""
    features: list
    interactions: list = field(default_factory=list)   # list of [feature_a, feature_b]
    year_factor: bool = True
    levels: dict = field(default_factory=dict)          # feature -> levels (str), base first
    bases: dict = field(default_factory=dict)           # feature -> base level (str)
    # Milestone 13 (several states). year_col: the data column holding the "year" key, e.g.
    # "state_year" for state x year effects (a storm year in one state only). absorbed:
    # features whose level is carried by that key in fitted mode, e.g. "state": in "fitted"
    # predictions the state-year effect already holds the state's level, so these columns are
    # 0; in "average" predictions they hold each state's average-year level vs the base state
    # (the STATE FACTOR). See absorb_group_levels().
    year_col: str = "year"
    absorbed: list = field(default_factory=list)

    @property
    def columns(self):
        cols = ["const"]
        for f in self.features:
            cols += [f"{f}[{l}]" for l in self.levels[f][1:]]
        for term in self.interactions:
            cols += interaction_columns(term, self.levels)
        if self.year_factor:
            cols += [f"year[{l}]" for l in self.levels["year"][1:]]
        return cols


def interaction_columns(term, levels):
    """Column names of one interaction term: every combination of non-base levels of its
    members, e.g. "a[x]:b[y]". Two-way terms between model features are the usual case;
    Milestone 13 also allows longer terms and members that are not main features (e.g.
    construction_band x flood_zone_group x state, where the state's own level is carried by
    the state x year effects)."""
    combos = itertools.product(*[levels[m][1:] for m in term])
    return [":".join(f"{m}[{l}]" for m, l in zip(term, c)) for c in combos]


def ordered_levels(df, feature, base, weight="exposure"):
    """Levels as strings: base first, then natural order (if defined) or by `weight`
    (exposure for frequency; claim count for severity)."""
    s = df[feature].astype(str)
    exp = df.groupby(s, observed=True)[weight].sum().sort_values(ascending=False)
    order = LEVEL_ORDER.get(feature, list(exp.index))
    if feature == "year":
        order = sorted(exp.index)
    lv = [l for l in order if l in exp.index] + [l for l in exp.index if l not in order]
    return [base] + [l for l in lv if l != base]


def make_spec(train, features, interactions=(), year_factor=True, bases=None, weight="exposure",
              year_col="year"):
    """Base level = the level with the most exposure (as in the one-way analysis),
    unless given in `bases`. `weight` names the column that decides "most" (the severity
    model passes its claim-count column). year_col: see Spec."""
    bases = dict(bases or {})
    if year_col != "year":
        train = train.assign(year=train[year_col])
    spec = Spec(list(features), [list(p) for p in interactions], year_factor, year_col=year_col)
    extra = [m for term in interactions for m in term if m not in features]   # e.g. "state"
    for f in list(features) + list(dict.fromkeys(extra)) + (["year"] if year_factor else []):
        s = train[f].astype(str)
        base = str(bases.get(f, train.groupby(s, observed=True)[weight].sum().idxmax()))
        spec.bases[f] = base
        spec.levels[f] = ordered_levels(train, f, base, weight)
    return spec


def design_matrix(df, spec, include_year=True):
    """0/1 columns in spec.columns order. A level not seen in training gets all zeros
    (i.e. it is treated like the base level); with include_year=False the year columns are
    all zero (the base year), which predict() then moves to the average-year level."""
    cols = {"const": np.ones(len(df))}
    dummies = {}
    members = list(dict.fromkeys(m for term in spec.interactions for m in term if m not in spec.features))
    for f in spec.features + members + (["year"] if spec.year_factor else []):
        s = df[spec.year_col if f == "year" else f].astype(str).to_numpy()
        for l in spec.levels[f][1:]:
            dummies[(f, l)] = (s == l).astype(float)
    for f in spec.features:
        off = include_year and f in spec.absorbed     # carried by the state-year effect instead
        for l in spec.levels[f][1:]:
            cols[f"{f}[{l}]"] = np.zeros(len(df)) if off else dummies[(f, l)]
    for term in spec.interactions:
        for c in itertools.product(*[spec.levels[m][1:] for m in term]):
            col = np.ones(len(df))
            for m, l in zip(term, c):
                col = col * dummies[(m, l)]
            cols[":".join(f"{m}[{l}]" for m, l in zip(term, c))] = col
    if spec.year_factor:
        for l in spec.levels["year"][1:]:
            cols[f"year[{l}]"] = dummies[("year", l)] if include_year else np.zeros(len(df))
    return pd.DataFrame(cols, index=df.index)[spec.columns]


# ---------------------------------------------------------------------------
# The fitted model
# ---------------------------------------------------------------------------
@dataclass
class FreqGLM:
    spec: Spec
    params: pd.Series            # coefficients b (log scale), indexed by column name
    family: str = "poisson"      # "poisson" or "negbin"
    alpha: float = None          # negative binomial shape parameter (variance = mu + alpha*mu^2)
    avg_year_log: float = 0.0    # log of the average-year level (see predict)
    result: object = None        # the statsmodels results object (not saved to JSON)

    def eta(self, df, include_year=True):
        """Linear predictor without the offset: b0 + sum(b x)."""
        return design_matrix(df, self.spec, include_year).to_numpy() @ self.params.to_numpy()

    def predict(self, df, year_level="average"):
        """Expected claims for each row of df.

        year_level="fitted":  use each row's own fitted year effect (only for training years).
        year_level="average": ignore the year effects and use the AVERAGE-YEAR level instead,
            the level at which the training years' total predicted claims equal their total
            actual claims. This is how a rate is used: it is set from past years and charged
            for a future year whose weather nobody knows yet."""
        if year_level == "fitted" or not self.spec.year_factor:
            return df["exposure"].to_numpy() * np.exp(self.eta(df, True))
        return df["exposure"].to_numpy() * np.exp(self.eta(df, False) + self.avg_year_log)

    def relative_rate(self, df):
        """Product of each row's feature relativities (exp of the linear predictor without the
        intercept and the year terms). Year-free and level-free: the right score for RANKING
        cells, e.g. in a lift chart that pools several years or several models' predictions."""
        return np.exp(self.eta(df, include_year=False) - self.params["const"])

    def relativities(self):
        """exp(b) for every non-year column (the multiplicative effect vs the base level)."""
        p = self.params[[c for c in self.params.index if not c.startswith("year[") and c != "const"]]
        return np.exp(p)

    def to_json_dict(self):
        return {"spec": asdict(self.spec), "params": self.params.to_dict(), "family": self.family,
                "alpha": self.alpha, "avg_year_log": self.avg_year_log}


def _avg_year_log(train, spec, params):
    """log c, with c chosen so that sum over training rows of exposure x exp(features part) x c
    equals the training claims. With year in the model this is the exposure-weighted average
    of the year effects (on the scale of claims)."""
    base_eta = design_matrix(train, spec, include_year=False).to_numpy() @ params.to_numpy()
    return float(np.log(train["claims"].sum() / (train["exposure"].to_numpy() * np.exp(base_eta)).sum()))


def absorb_group_levels(model, train, group="state", base=None, actual=None, weight=None):
    """Turn state x year effects into an average-year level for each state (Milestone 13).

    A model fitted with year_col="state_year" has one effect per state and year, and one
    average-year level for all states together. This returns a copy whose spec has `group`
    as an extra ABSORBED feature: its coefficient for each non-base state = log of that
    state's average-year level minus the base state's, each level chosen so the state's
    training predictions add up to its training actuals (claims for frequency, dollars for
    severity). avg_year_log becomes the base state's level. Fitted-year predictions are
    unchanged (absorbed columns are 0 there); average-year predictions now differ by state.

    actual: the actual totals per training row (default: train["claims"]).
    weight: per-row multiplier of exp(eta) (default: train["exposure"]; severity: claim count)."""
    from dataclasses import replace
    actual = train["claims"].to_numpy(float) if actual is None else np.asarray(actual, float)
    weight = train["exposure"].to_numpy(float) if weight is None else np.asarray(weight, float)
    eta0 = design_matrix(train, model.spec, include_year=False).to_numpy() @ model.params.to_numpy()
    g = train[group].astype(str).to_numpy()
    d = pd.DataFrame({"g": g, "a": actual, "m": weight * np.exp(eta0), "w": weight}).groupby("g")[["a", "m", "w"]].sum()
    lv = np.log(d["a"] / d["m"])
    if base not in lv.index:            # e.g. Virginia left out (leave-one-state-out): use the
        base = str(d["w"].idxmax())     # training state with the most weight (exposure / claims)
    spec = replace(model.spec, features=[*model.spec.features, group],
                   absorbed=[*model.spec.absorbed, group],
                   levels={**model.spec.levels, group: [base] + [l for l in lv.index if l != base]},
                   bases={**model.spec.bases, group: base})
    extra = pd.Series({f"{group}[{l}]": float(lv[l] - lv[base]) for l in lv.index if l != base})
    # keep the column order of spec.columns (features before interactions and year)
    params = pd.concat([model.params, extra]).reindex(spec.columns)
    return replace(model, spec=spec, params=params, avg_year_log=float(lv[base]))


def cluster_codes(df, cluster_by):
    return pd.factorize(df[list(cluster_by)].astype(str).agg("|".join, axis=1))[0]


def fit_glm(train, features, interactions=(), year_factor=True, family="poisson", alpha=None,
            cov="poisson", cluster_by=("county_group", "year"), bases=None, spec=None, year_col="year"):
    """Fit a Poisson (or negative binomial) GLM with log link and log(exposure) offset.

    cov (how the standard errors / confidence intervals are computed; the coefficients are
    the same for all three):
      "poisson"  assumes variance = mean (plain Poisson).
      "quasi"    quasi-Poisson: variance = phi x mean, phi from the Pearson chi-square
                 (all standard errors x sqrt(phi)).
      "cluster"  cluster-robust ("sandwich") errors: rows in the same cluster (default:
                 county group x year) may move together, as they do when a storm hits."""
    spec = spec or make_spec(train, features, interactions, year_factor, bases, year_col=year_col)
    X = design_matrix(train, spec)
    y = train["claims"].to_numpy(float)
    off = np.log(train["exposure"].to_numpy(float))
    fam = sm.families.Poisson() if family == "poisson" else sm.families.NegativeBinomial(alpha=alpha)
    model = sm.GLM(y, X, family=fam, offset=off)
    if cov == "quasi":
        res = model.fit(scale="X2")
    elif cov == "cluster":
        res = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_codes(train, cluster_by)})
    else:
        res = model.fit()
    params = pd.Series(res.params, index=X.columns)
    return FreqGLM(spec, params, family, alpha, _avg_year_log(train, spec, params), res)


def fit_negbin(train, features, interactions=(), year_factor=True, alpha=None, bases=None, spec=None):
    """Negative binomial (NB2) GLM: variance = mu + alpha x mu^2.

    alpha is estimated by maximum likelihood ("profile likelihood": refit the GLM for
    different alpha values and keep the one with the highest log-likelihood)."""
    spec = spec or make_spec(train, features, interactions, year_factor, bases)
    if alpha is None:
        def neg_llf(log_a):
            return -fit_glm(train, features, family="negbin", alpha=float(np.exp(log_a)),
                            spec=spec).result.llf
        opt = optimize.minimize_scalar(neg_llf, bounds=(np.log(1e-3), np.log(50)), method="bounded",
                                       options={"xatol": 1e-3})
        alpha = float(np.exp(opt.x))
    return fit_glm(train, features, family="negbin", alpha=alpha, spec=spec)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def poisson_deviance(y, mu):
    """Poisson deviance: 2 x sum[ y log(y/mu) - (y - mu) ]. 0 = perfect; lower is better.
    It is "how surprised a Poisson model is by what actually happened"."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * (t - (y - mu)).sum())


def rebase(pred, actual_total):
    """Scale predictions so they add up to the actual total (removes the year's level)."""
    pred = np.asarray(pred, float)
    return pred * actual_total / pred.sum()


def null_prediction(train, test):
    """The 'no features' baseline: every policy-year gets the training-average frequency."""
    return test["exposure"].to_numpy() * train["claims"].sum() / train["exposure"].sum()


def gini(claims, pred, exposure):
    """Exposure-weighted Gini of the model's ranking (0 = random, higher = better).

    Sort cells from lowest to highest predicted frequency; plot cumulative share of exposure
    (x) vs cumulative share of actual claims (y). A model that ranks well bunches the claims
    at the end, so the curve sags below the diagonal. Gini = 1 - 2 x (area under the curve)."""
    claims, pred, exposure = (np.asarray(a, float) for a in (claims, pred, exposure))
    order = np.argsort(pred / exposure, kind="mergesort")
    x = np.concatenate([[0], np.cumsum(exposure[order]) / exposure.sum()])
    y = np.concatenate([[0], np.cumsum(claims[order]) / claims.sum()])
    return float(1 - 2 * np.trapezoid(y, x))


def lift_table(claims, pred, exposure, n=10, rank_by=None):
    """Exposure-weighted n-tiles of predicted frequency: each bucket holds 1/n of the exposure.
    Returns actual and predicted frequency per bucket (per 1,000 policy-years).

    rank_by: the score used to sort cells (default pred / exposure). When pooling several
    years, sort by the year-free (average-year) rate, so the ranking reflects the features
    only and not which years were storm years; `pred` can then be the re-based prediction."""
    claims, pred, exposure = (np.asarray(a, float) for a in (claims, pred, exposure))
    score = pred / exposure if rank_by is None else np.asarray(rank_by, float)
    order = np.argsort(score, kind="mergesort")
    cum = np.cumsum(exposure[order]) / exposure.sum()
    bucket = np.minimum((cum * n - 1e-12).astype(int), n - 1) + 1
    t = pd.DataFrame({"bucket": bucket, "exposure": exposure[order], "claims": claims[order],
                      "pred": pred[order]}).groupby("bucket")[["exposure", "claims", "pred"]].sum()
    t["actual_freq_per_1000"] = 1000 * t["claims"] / t["exposure"]
    t["pred_freq_per_1000"] = 1000 * t["pred"] / t["exposure"]
    t["actual_relative_to_average"] = (t["claims"] / t["exposure"]) / (claims.sum() / exposure.sum())
    return t


def actual_vs_expected(df, pred, by):
    """Actual / expected claims by the levels of `by` (a column name or list)."""
    t = df.assign(expected=np.asarray(pred, float)).groupby(by, observed=True)[
        ["exposure", "claims", "expected"]].sum()
    t["actual_over_expected"] = t["claims"] / t["expected"]
    return t


def score_fold(test, pred, pred_null):
    """All per-fold metrics from the validation plan, for a model and the null baseline."""
    y, actual = test["claims"].to_numpy(float), test["claims"].sum()
    pr, nr = rebase(pred, actual), rebase(pred_null, actual)
    d = {"test_claims": int(actual), "test_exposure": float(test["exposure"].sum()),
         "pred_claims": float(np.sum(pred)), "null_claims": float(np.sum(pred_null)),
         "actual_over_expected": float(actual / np.sum(pred)),
         "deviance_raw": poisson_deviance(y, pred), "null_deviance_raw": poisson_deviance(y, pred_null),
         "deviance_rebased": poisson_deviance(y, pr), "null_deviance_rebased": poisson_deviance(y, nr),
         "gini": gini(y, pred, test["exposure"])}
    d["improvement_rebased_pct"] = 100 * (1 - d["deviance_rebased"] / d["null_deviance_rebased"])
    d["improvement_raw_pct"] = 100 * (1 - d["deviance_raw"] / d["null_deviance_raw"])
    return d


def loyo_scores(df, fit_fn, years=None, keep_cols=()):
    """Leave-one-year-out: for each year, fit_fn(train) -> FreqGLM, predict the held-out year
    at the average-year level, and score it. Returns (one row per fold, out-of-fold
    predictions for every test row)."""
    rows, oof = [], []
    for y, train, test in val.leave_one_year_out(df, years):
        m = fit_fn(train)
        pred, pnull = m.predict(test, "average"), null_prediction(train, test)
        rows.append({"test_year": y, "cat_year": val.is_cat_year(y), **score_fold(test, pred, pnull)})
        o = test[["year", "exposure", "claims", *keep_cols]].copy()
        o["pred"], o["pred_null"] = pred, pnull
        o["pred_rebased"] = rebase(pred, test["claims"].sum())
        o["relative_rate"] = m.relative_rate(test)
        oof.append(o)
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


# ---------------------------------------------------------------------------
# Comparing nested models (the step-by-step build)
# ---------------------------------------------------------------------------
def _loyo_cached(df, features, interactions, years, cache):
    key = (tuple(features), tuple(tuple(p) for p in interactions))
    if key not in cache:
        sc, _ = loyo_scores(df, lambda tr: fit_glm(tr, features, interactions), years)
        cache[key] = sc.set_index("test_year")
    return cache[key]


def compare_models(df, cur_features, cur_inter, new_features, new_inter, loyo_years=None,
                  cluster_by=("county_group", "year"), cache=None):
    """Compare a candidate model (new_*) with the current one (cur_*), both fitted on `df`.

    In-sample measures (the classic toolkit):
      * deviance and AIC (AIC = deviance-type misfit + 2 x number of parameters; lower is better);
      * LRT: likelihood-ratio test, chi-square on the deviance drop (assumes plain Poisson);
      * quasi-F: the deviance drop per added parameter divided by phi (the candidate's Pearson
        dispersion), i.e. the LRT corrected for cell-level over-dispersion;
      * cluster Wald: are the added coefficients jointly zero, using cluster-robust errors
        (lets a whole county-year move together, as it does in a storm).
    Out-of-sample (if loyo_years given): leave-one-year-out on those years; each fold's
    re-based deviance improvement over the no-features model, in %, averaged with every year
    counting once; the change versus the current model; and how many folds got better."""
    cache = {} if cache is None else cache
    m0, m1 = fit_glm(df, cur_features, cur_inter), fit_glm(df, new_features, new_inter)
    r0, r1 = m0.result, m1.result
    ddev, ddf = r0.deviance - r1.deviance, len(m1.params) - len(m0.params)
    phi = r1.pearson_chi2 / r1.df_resid
    row = {"n_params": len(m1.params), "deviance": r1.deviance, "AIC": r1.aic,
           "d_params": ddf, "d_deviance": ddev, "d_AIC": r1.aic - r0.aic, "phi": phi,
           "LRT_p": stats.chi2.sf(ddev, ddf) if ddf > 0 else np.nan,
           "quasi_F": ddev / ddf / phi if ddf > 0 else np.nan,
           "quasi_F_p": stats.f.sf(ddev / ddf / phi, ddf, r1.df_resid) if ddf > 0 else np.nan,
           "cluster_Wald_p": np.nan}
    new_cols = [c for c in m1.params.index if c not in m0.params.index]
    if set(m0.params.index) <= set(m1.params.index) and new_cols:      # nested: Wald test possible
        rc = fit_glm(df, new_features, new_inter, cov="cluster", cluster_by=cluster_by).result
        R = np.zeros((len(new_cols), len(m1.params)))
        for i, c in enumerate(new_cols):
            R[i, list(m1.params.index).index(c)] = 1.0
        row["cluster_Wald_p"] = float(rc.wald_test(R, scalar=True).pvalue)
    if loyo_years is not None:
        s1 = _loyo_cached(df, new_features, new_inter, loyo_years, cache)
        row["devLOYO_improvement_pct"] = s1["improvement_rebased_pct"].mean()
        row["devLOYO_improvement_noncat_pct"] = s1.loc[~s1["cat_year"], "improvement_rebased_pct"].mean()
        if cur_features or cur_inter:
            s0 = _loyo_cached(df, cur_features, cur_inter, loyo_years, cache)
            row["d_devLOYO_pct"] = row["devLOYO_improvement_pct"] - s0["improvement_rebased_pct"].mean()
            row["devLOYO_folds_better"] = f"{int((s1['deviance_rebased'] < s0['deviance_rebased']).sum())}/{len(s1)}"
        else:   # versus the year-only model, whose re-based predictions equal the null's
            row["d_devLOYO_pct"] = row["devLOYO_improvement_pct"]
            row["devLOYO_folds_better"] = f"{int((s1['improvement_rebased_pct'] > 0).sum())}/{len(s1)}"
    return row


def forward_selection(df, start_features, candidates, loyo_years, cluster_by=("county_group", "year"),
                      alpha=0.05):
    """Step-by-step model building ("forward selection").

    Start from `start_features`; then for each candidate in order, test it against the CURRENT
    model and keep it only if BOTH:
      1. it is statistically real after allowing for storm clumping: cluster-robust Wald p < alpha
         (skipped when the candidate replaces a feature, because the models aren't nested); and
      2. it improves out-of-sample fit: the mean leave-one-year-out deviance improvement goes up.
    candidates: list of dicts {"name", "add": [...], "remove": [...], "interactions": [[a, b]]}.
    Returns (table with one row per step and the decision, final features, final interactions)."""
    cache, rows = {}, []
    feats, inter = list(start_features), []
    row = compare_models(df, [], [], feats, inter, loyo_years, cluster_by, cache)
    rows.append({"step": "start: " + " + ".join(feats), "decision": "start", **row})
    for c in candidates:
        nf = [f for f in feats if f not in c.get("remove", [])] + [f for f in c.get("add", []) if f not in feats]
        ni = inter + [list(p) for p in c.get("interactions", [])]
        row = compare_models(df, feats, inter, nf, ni, loyo_years, cluster_by, cache)
        p = row["cluster_Wald_p"]
        wald_ok = bool(c.get("remove")) if np.isnan(p) else p < alpha   # replacements: LOYO decides
        loyo_ok = row["d_devLOYO_pct"] > 0
        keep = bool(wald_ok and loyo_ok)
        why = [] if keep else (["not significant after clumping"] if not wald_ok else []) + \
            (["worse on unseen years"] if not loyo_ok else [])
        rows.append({"step": c["name"], "decision": "KEEP" if keep else "drop: " + "; ".join(why), **row})
        if keep:
            feats, inter = nf, ni
    return pd.DataFrame(rows).set_index("step"), feats, inter


# ---------------------------------------------------------------------------
# Over-dispersion
# ---------------------------------------------------------------------------
def overdispersion_tests(model, df):
    """Does the variance of claim counts exceed the mean (as Poisson assumes)?
      * Pearson phi = sum((y - mu)^2 / mu) / degrees of freedom. About 1 if Poisson is right.
      * deviance / df: a second estimate (less reliable when most cells have 0 or 1 claims).
      * Cameron-Trivedi test: regress ((y - mu)^2 - y) / mu on mu (no intercept). Under Poisson
        the slope is 0; a significantly positive slope (estimate of alpha) = over-dispersion
        of the negative binomial form variance = mu + alpha mu^2."""
    r = model.result
    y = df["claims"].to_numpy(float)
    mu = np.asarray(r.fittedvalues, float)
    z = ((y - mu) ** 2 - y) / mu
    ols = sm.OLS(z, mu).fit()
    return {"pearson_phi": r.pearson_chi2 / r.df_resid, "deviance_over_df": r.deviance / r.df_resid,
            "CT_alpha": float(ols.params[0]), "CT_t": float(ols.tvalues[0]),
            "CT_p_one_sided": float(stats.norm.sf(ols.tvalues[0]))}


# ---------------------------------------------------------------------------
# Relativity table
# ---------------------------------------------------------------------------
def relativity_table(model, df, ci_results=None, level=0.95):
    """One row per level of every feature (base levels included, relativity 1):
    exposure, claims, relativity exp(b), and a 95% CI for each entry of ci_results
    (dict label -> statsmodels result with the same columns; default: the model's own)."""
    ci_results = ci_results or {"": model.result}
    cis = {lab: pd.DataFrame(np.asarray(r.conf_int(alpha=1 - level)), index=model.params.index)
           for lab, r in ci_results.items()}
    rows = []
    for f in model.spec.features:
        sums = df.groupby(df[f].astype(str), observed=True)[["exposure", "claims"]].sum()
        for l in model.spec.levels[f]:
            col = f"{f}[{l}]"
            base = l == model.spec.bases[f]
            row = {"feature": f, "level": l, "is_base": base,
                   "exposure_share": sums["exposure"].get(l, 0) / sums["exposure"].sum(),
                   "claims": int(sums["claims"].get(l, 0)),
                   "relativity": 1.0 if base else float(np.exp(model.params[col]))}
            for lab, ci in cis.items():
                suf = f"_{lab}" if lab else ""
                row[f"lo{suf}"] = 1.0 if base else float(np.exp(ci.loc[col, 0]))
                row[f"hi{suf}"] = 1.0 if base else float(np.exp(ci.loc[col, 1]))
            rows.append(row)
    for a, b in model.spec.interactions:
        for la in model.spec.levels[a][1:]:
            for lb in model.spec.levels[b][1:]:
                col = f"{a}[{la}]:{b}[{lb}]"
                row = {"feature": f"{a} x {b}", "level": f"{la} & {lb}", "is_base": False,
                       "exposure_share": np.nan, "claims": np.nan,
                       "relativity": float(np.exp(model.params[col]))}
                for lab, ci in cis.items():
                    suf = f"_{lab}" if lab else ""
                    row[f"lo{suf}"], row[f"hi{suf}"] = (float(np.exp(ci.loc[col, 0])), float(np.exp(ci.loc[col, 1])))
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------
def save_model(model, stem, extra=None):
    """Write <stem>.json (portable: spec, coefficients, average-year level; what the app reads)
    and <stem>.pickle (the full statsmodels result, without the data, for later analysis)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    d = model.to_json_dict()
    d.update(extra or {})
    stem.with_suffix(".json").write_text(json.dumps(d, indent=2, default=float), encoding="utf-8")
    if model.result is not None:
        model.result.save(str(stem.with_suffix(".pickle")), remove_data=True)


def load_model_json(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return FreqGLM(Spec(**d["spec"]), pd.Series(d["params"]), d["family"], d["alpha"], d["avg_year_log"])


# ---------------------------------------------------------------------------
# Self-test on simulated data with known answers
# ---------------------------------------------------------------------------
def _simulate(seed=1, rows_per_combo=60):
    rng = np.random.default_rng(seed)
    zone_rel = {"A": 1.0, "X": 0.5}
    band_rel = {"old": 2.0, "mid": 1.0, "new": 0.6}
    year_rel = {2010: 1.0, 2011: 3.0, 2012: 0.7}
    rows = []
    for z, rz in zone_rel.items():
        for b, rb in band_rel.items():
            for yr, ry in year_rel.items():
                e = rng.uniform(20, 200, rows_per_combo)
                rows.append(pd.DataFrame({"zone": z, "band": b, "year": yr, "exposure": e,
                                          "grp": rng.integers(0, 10, rows_per_combo),
                                          "claims": rng.poisson(e * 0.01 * rz * rb * ry)}))
    df = pd.concat(rows, ignore_index=True)
    df.loc[df["zone"] == "A", "exposure"] *= 1.5          # make A and "mid" the base levels
    df.loc[df["band"] == "mid", "exposure"] *= 1.5
    df["claims"] = rng.poisson(df["exposure"] * 0.05 * df["zone"].map(zone_rel)
                               * df["band"].map(band_rel) * df["year"].map(year_rel))
    return df


def _self_test():
    df = _simulate()
    m = fit_glm(df, ["zone", "band"])
    assert m.spec.bases["zone"] == "A" and m.spec.bases["band"] == "mid"
    rel = m.relativities()
    assert abs(rel["zone[X]"] / 0.5 - 1) < 0.06, rel
    assert abs(rel["band[old]"] / 2.0 - 1) < 0.06 and abs(rel["band[new]"] / 0.6 - 1) < 0.08, rel

    # Balance: predictions add up to the actual claims (fitted years AND average-year level).
    assert np.isclose(m.predict(df, "fitted").sum(), df["claims"].sum(), rtol=1e-6)
    assert np.isclose(m.predict(df, "average").sum(), df["claims"].sum(), rtol=1e-6)

    # Offset: doubling every exposure and claim count leaves the rate model unchanged.
    m2 = fit_glm(df.assign(exposure=2 * df["exposure"], claims=2 * df["claims"]), ["zone", "band"])
    assert np.allclose(m2.params, m.params, atol=1e-6)

    # Aggregating cells with the same features changes neither the coefficients nor
    # the deviance difference between nested models.
    agg = aggregate(df, ["zone", "band"])
    ma = fit_glm(agg, ["zone", "band"])
    assert np.allclose(ma.params, m.params, atol=1e-6)
    d_full = fit_glm(df, ["zone"]).result.deviance - m.result.deviance
    d_agg = fit_glm(agg, ["zone"]).result.deviance - ma.result.deviance
    assert np.isclose(d_full, d_agg, rtol=1e-6)

    # Scores: a model beats the null; deviance(y, y) = 0; rebase hits the total; Gini > 0.
    y = df["claims"].to_numpy(float)
    assert poisson_deviance(y, m.predict(df, "fitted")) < poisson_deviance(y, null_prediction(df, df))
    assert poisson_deviance(np.array([0.0, 3.0]), np.array([1e-12, 3.0])) < 1e-9
    assert np.isclose(rebase([1, 2, 3], 12).sum(), 12)
    assert gini(y, m.predict(df), df["exposure"]) > 0.1
    lt = lift_table(y, m.predict(df), df["exposure"])
    assert len(lt) == 10 and np.isclose(lt["exposure"].sum(), df["exposure"].sum())
    assert lt["actual_freq_per_1000"].iloc[-1] > lt["actual_freq_per_1000"].iloc[0]
    lt2 = lift_table(y, m.predict(df), df["exposure"], rank_by=m.relative_rate(df))
    # Same ranking up to ties (the two scores can differ in the last floating-point digit,
    # which moves tied cells across a decile boundary), so compare the ranking quality.
    assert np.isclose(lt2["claims"].sum(), lt["claims"].sum())
    assert np.isclose(gini(y, m.relative_rate(df) * df["exposure"], df["exposure"]),
                      gini(y, m.predict(df), df["exposure"]), atol=0.01)

    # LOYO runs, each fold predicts an unseen year at the average-year level.
    sc, oof = loyo_scores(df, lambda tr: fit_glm(tr, ["zone", "band"]))
    assert len(sc) == 3 and len(oof) == len(df)

    # Model comparison / forward selection: a real feature ("band") is kept with a huge,
    # significant deviance drop; a pure-noise column is dropped.
    df["noise"] = (df["grp"] % 2).astype(str)
    row = compare_models(df, ["zone"], [], ["zone", "band"], [], loyo_years=None, cluster_by=("grp", "year"))
    assert row["LRT_p"] < 1e-6 and row["d_deviance"] > 50 and row["cluster_Wald_p"] < 0.05
    tbl, feats, _ = forward_selection(df, ["zone"], [{"name": "+band", "add": ["band"]},
                                                     {"name": "+noise", "add": ["noise"]}],
                                      loyo_years=(2010, 2011, 2012), cluster_by=("grp", "year"))
    assert tbl.loc["+band", "decision"] == "KEEP" and feats == ["zone", "band"], tbl

    # Over-dispersion: Poisson data should give phi near 1.
    od = overdispersion_tests(m, df)
    assert 0.8 < od["pearson_phi"] < 1.25, od

    # Negative binomial on Poisson data: alpha near 0, same relativities.
    nb = fit_negbin(df, ["zone", "band"])
    assert nb.alpha < 0.05 and abs(nb.relativities()["zone[X]"] / rel["zone[X]"] - 1) < 0.03

    # Relativity table includes base levels at exactly 1.
    rt = relativity_table(m, df)
    assert (rt.loc[rt["is_base"], "relativity"] == 1).all() and len(rt) == 5

    # JSON round trip predicts the same.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        save_model(m, Path(tmp) / "m")
        m3 = load_model_json(Path(tmp) / "m.json")
        assert np.allclose(m3.predict(df), m.predict(df))
    return True


if __name__ == "__main__":
    print("frequency_glm self-test passed:", _self_test())
