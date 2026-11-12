"""
XGBoost CHALLENGER models (Milestone 10): frequency, severity and pure premium (Tweedie).

Plain-English summary (full explanations in LEARNING_NOTES.md, Milestone 10):
  * Gradient boosting builds a model as a SUM OF MANY SMALL DECISION TREES. Each new tree is
    fitted to what the trees so far still get wrong, and only a small step of it is added
    (the learning rate). XGBoost is a fast, regularised implementation of this idea.
  * The OBJECTIVE tells XGBoost what kind of target it is predicting, exactly like the GLM
    family does:
        frequency   count:poisson   claim counts (Poisson), log link
        severity    reg:gamma       claim sizes (Gamma), log link
        pure prem.  reg:tweedie     loss cost (compound Poisson-Gamma, power 1.7), log link
    With a log link the trees add up on the LOG scale, so their effects multiply, just like
    GLM relativities.
  * BASE_MARGIN is XGBoost's version of the GLM OFFSET: a fixed starting value per row that
    the trees add to. For frequency it is log(earned exposure), so the trees model claims per
    policy-year. We ALSO put the calendar year's level into it (log of the year's average
    claim rate / claim size / loss cost, from the training years only), the same job the GLMs'
    year factor does: storm years don't distort the building effects, and an unseen year is
    predicted at the AVERAGE-YEAR level. Year is never a tree feature.
  * Same data, same features and the same validation design as the GLMs
    (docs/validation_plan.md): hyperparameters are tuned with leave-one-year-out on
    2010-2020 only; 2021-2025 is scored once at the end.
  * EARLY STOPPING on the cross-validation mean: the 11 fold models are trained side by side,
    10 trees at a time. After each chunk we score every held-out year and average. When that
    average hasn't improved for 150 trees we stop, and the best tree count is n_estimators.
  * MONOTONIC CONSTRAINTS force a feature's effect to go only one way (e.g. a higher coverage
    limit can never lower the expected claim size), whatever the other features are.

Encoding (numbers XGBoost can split on):
  * ordered bands -> integer codes (year built 0 = before 1960 ... 3 = 2000+; coverage
    0 = $0 ... 3 = $250K+; floors 0 = 1 ... 2 = 3+). Codes are needed for monotone constraints.
  * yes/no features -> 0/1 (flood zone: 0 = A/V, 1 = B/C/X).
  * occupancy and county group -> one 0/1 column per level ("one-hot"). No target encoding,
    so no claim information leaks into the features. A level not seen in training -> all 0.

Run the self-test:                .venv\\Scripts\\python.exe src\\models\\xgb_models.py
Tune (one search per geography variant; writes reports/07_xgb_tuning_<kind>_<variant>.csv):
                                  .venv\\Scripts\\python.exe src\\models\\xgb_models.py tune frequency
Geography + monotone choice:      .venv\\Scripts\\python.exe src\\models\\xgb_models.py compare frequency
Fit and save the final models:    .venv\\Scripts\\python.exe src\\models\\xgb_models.py fit-final
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODELS = Path(__file__).resolve().parent
SRC = MODELS.parent
for p in (SRC, MODELS):                       # so the imports below work when run as a script
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import validation as val                      # noqa: E402
import frequency_glm as fg                    # noqa: E402
import severity_glm as sv                     # noqa: E402
import pure_premium as pp                     # noqa: E402

ROOT = SRC.parent
CELLS_PATH = ROOT / "data" / "processed" / "va_model_data.parquet"
MODEL_DIR = ROOT / "models"
SETTINGS_PATH = MODEL_DIR / "xgb_settings.json"
REPORTS = ROOT / "reports"
LOSS = pp.LOSS                                # cell losses, 2025 dollars, incl. ICC
TWEEDIE_POWER = 1.7                           # Milestone 9's chosen Tweedie power

# ---------------------------------------------------------------------------
# Features and their encoding
# ---------------------------------------------------------------------------
BUILDING_FEATURES = ["flood_zone_group", "construction_band", "building_coverage_band", "occupancy_group",
                     "elevated", "post_firm", "primary_residence", "floors_group"]
ALL_FEATURES = BUILDING_FEATURES + ["coastal_tidal", "county_group"]
GEOGRAPHY_VARIANTS = {                        # the Q7 geography test
    "no geography": BUILDING_FEATURES,
    "+ Tidewater": BUILDING_FEATURES + ["coastal_tidal"],
    "+ Tidewater + county group": ALL_FEATURES,
}
ORDINAL = {                                   # code = position in the list
    "construction_band": ["before 1960", "1960-1979", "1980-1999", "2000 or later"],
    "building_coverage_band": ["$0 (contents only)", "under $100K", "$100K-$249K", "$250K or more"],
    "floors_group": ["1", "2", "3+"],
}
BINARY = {                                    # the value coded as 1
    "flood_zone_group": "B/C/X moderate-low risk",
    "elevated": "True", "post_firm": "True", "primary_residence": "True", "coastal_tidal": "True",
}
ONEHOT = ["occupancy_group", "county_group", "state"]     # "state": Milestone 13 (three states)

# Monotonic constraints (-1 = the prediction may only go DOWN as the code goes up; +1 = only up).
# They hold for every combination of the other features. Chosen on actuarial grounds:
MONOTONE = {
    # frequency: lower-risk zone, elevated, newer and post-FIRM buildings never claim MORE often
    "frequency": {"flood_zone_group": -1, "elevated": -1, "construction_band": -1, "post_firm": -1},
    # severity: a higher coverage limit never lowers the claim size; elevated and B/C/X never raise it.
    # Year built, floors, occupancy stay free (the GLM shows non-monotone patterns, e.g. 1960-79 peak).
    "severity": {"building_coverage_band": +1, "elevated": -1, "flood_zone_group": -1},
    # pure premium: the directions both parts agree on. Year built is left free because the
    # GLM loss cost peaks in 1960-79 (3.5x) above before-1960 (2.8x).
    "tweedie": {"flood_zone_group": -1, "elevated": -1, "post_firm": -1, "building_coverage_band": +1},
}

KINDS = {
    "frequency": {"objective": "count:poisson", "target": "claims"},
    "severity": {"objective": "reg:gamma", "target": sv.TARGET},
    "tweedie": {"objective": "reg:tweedie", "target": LOSS},
}
SEED = 2026


@dataclass
class Encoder:
    """Turns the feature columns into numbers, remembering the one-hot levels seen in training."""
    features: list
    onehot_levels: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, df, features):
        enc = cls(list(features))
        for f in features:
            if f in ONEHOT:
                enc.onehot_levels[f] = sorted(df[f].astype(str).unique())
        return enc

    @property
    def columns(self):
        cols = []
        for f in self.features:
            cols += [f"{f}={l}" for l in self.onehot_levels[f]] if f in ONEHOT else [f]
        return cols

    def source(self, col):
        """Which original feature a column belongs to (for importance summed over one-hots)."""
        return col.split("=", 1)[0]

    def transform(self, df):
        out = {}
        for f in self.features:
            s = df[f].astype(str).to_numpy()
            if f in ORDINAL:
                codes = pd.Series(s).map({l: i for i, l in enumerate(ORDINAL[f])}).to_numpy(float)
                if np.isnan(codes).any():
                    raise ValueError(f"unknown level in {f}: {set(s[np.isnan(codes)])}")
                out[f] = codes
            elif f in BINARY:
                out[f] = (s == BINARY[f]).astype(float)
            elif f in ONEHOT:
                for l in self.onehot_levels[f]:
                    out[f"{f}={l}"] = (s == l).astype(float)
            else:
                raise ValueError(f"no encoding rule for {f}")
        return pd.DataFrame(out, index=df.index)[self.columns].to_numpy(np.float32)

    def monotone_string(self, monotone):
        """XGBoost's monotone_constraints parameter: one entry per column, in column order."""
        monotone = monotone or {}
        # Constrained features are ordinal or yes/no, so their column name IS the feature name.
        return "(" + ",".join(str(int(monotone.get(c, 0))) for c in self.columns) + ")"

    def to_dict(self):
        return {"features": self.features, "onehot_levels": self.onehot_levels}

    @classmethod
    def from_dict(cls, d):
        return cls(list(d["features"]), dict(d["onehot_levels"]))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cell_table(path=CELLS_PATH, verbose=True):
    """Rating cells WITH exposure, summed over rows that share all 10 features + year (the
    county itself is dropped; county_group stays). Used for frequency and the Tweedie.
    Summing rows with identical features changes nothing a tree can see (gradient and hessian
    sums are the same), it just makes fitting faster. Every model is scored on THIS table, so
    their deviances are comparable."""
    cells = pd.read_parquet(path)
    m = cells[cells["exposure_matched"] & (cells["exposure"] > 0)]
    agg = m.groupby(ALL_FEATURES + ["year"], observed=True)[["exposure", "claims", LOSS]].sum().reset_index()
    if verbose:
        print(f"Cell table: {len(agg):,} rows ({len(m):,} cells with exposure), "
              f"{agg['exposure'].sum():,.0f} policy-years, {int(agg['claims'].sum()):,} paid claims, "
              f"${agg[LOSS].sum():,.0f}.")
    return agg


def unmatched_losses_by_year(path=CELLS_PATH):
    """Losses in the zero-exposure ("unmatched") cells per year: they can't enter a rate model,
    but the Tweedie's off-balance adds them back (as in Milestone 9)."""
    cells = pd.read_parquet(path)
    u = cells[~(cells["exposure_matched"] & (cells["exposure"] > 0))]
    return u.groupby("year")[LOSS].sum().reindex(val.ALL_YEARS, fill_value=0.0)


def load_severity_table(verbose=True):
    """Individual paid claims with building + contents > $0 (same rows as the severity GLM)."""
    c = sv.severity_rows(sv.load_claims(verbose=verbose))
    if verbose:
        print(f"Severity table: {len(c):,} paid claims, ${c[sv.TARGET].sum():,.0f} building + contents.")
    return c


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def year_levels(train, kind, y=None):
    """log of each training year's average (claim rate / claim size / loss cost): the
    year-only model, used as the year part of base_margin."""
    if kind == "frequency":
        g = train.groupby("year")[["claims", "exposure"]].sum()
        return np.log(np.maximum(g["claims"], 0.5) / g["exposure"]).to_dict()
    if kind == "severity":
        y = train[sv.TARGET] if y is None else pd.Series(y, index=train.index)
        return np.log(y.groupby(train["year"]).mean()).to_dict()
    g = train.groupby("year")[[LOSS, "exposure"]].sum()
    return np.log(np.maximum(g[LOSS], 1.0) / g["exposure"]).to_dict()


def xgb_params(kind, params, encoder=None, monotone=None, nthread=None):
    """Our search parameters -> XGBoost's native parameter dict."""
    p = {"objective": KINDS[kind]["objective"], "tree_method": "hist", "seed": SEED,
         "max_depth": int(params["max_depth"]), "eta": float(params["learning_rate"]),
         "min_child_weight": float(params["min_child_weight"]), "subsample": float(params["subsample"]),
         "colsample_bytree": float(params["colsample_bytree"]), "lambda": float(params["reg_lambda"]),
         "alpha": float(params["reg_alpha"]), "gamma": float(params["gamma"]), "verbosity": 0}
    if kind == "tweedie":
        p["tweedie_variance_power"] = TWEEDIE_POWER
    if monotone and encoder is not None:
        p["monotone_constraints"] = encoder.monotone_string(monotone)
    if nthread:
        p["nthread"] = nthread
    return p


def _dmatrix(X, encoder, label=None, weight=None, margin=None):
    return xgb.DMatrix(X, label=label, weight=weight, base_margin=margin, feature_names=encoder.columns)


def _train_parts(train, kind, encoder, ylog, cap=None):
    """(X, label, weight, base_margin) for training rows."""
    X = encoder.transform(train)
    yr = train["year"].map(ylog).to_numpy(float)
    if kind == "frequency":
        return X, train["claims"].to_numpy(float), None, np.log(train["exposure"].to_numpy(float)) + yr
    if kind == "severity":
        return X, sv.cap_amounts(train[sv.TARGET], cap), None, yr
    e = train["exposure"].to_numpy(float)
    return X, train[LOSS].to_numpy(float) / e, e, yr


@dataclass
class XGBModel:
    kind: str
    features: list
    params: dict
    n_rounds: int
    encoder: Encoder
    monotone: dict = None
    booster: object = None
    year_log: dict = None        # year -> log level used in training (the "year factor")
    avg_year_log: float = 0.0    # average-year level for unseen years (balances training totals)
    load: float = 1.0            # severity: large-loss load (1.0 = no cap, as the GLM)
    cap: float = None            # severity: cap in 2025 $ (None = uncapped)
    off_balance: float = 1.0     # Tweedie: puts back the unmatched-cell losses (as Milestone 9)
    train_years: list = None
    train_rate: float = None     # training average rate (for level-free ranking scores)

    def tree_margin(self, df):
        """Sum of the trees only (log scale), with no exposure and no year level. A zero
        base_margin is passed so XGBoost's global intercept (base_score) is never added:
        training always used a base_margin, so the trees are fitted relative to it."""
        X = self.encoder.transform(df)
        d = _dmatrix(X, self.encoder, margin=np.zeros(len(df)))
        return self.booster.predict(d, output_margin=True)

    def rate(self, df, year_level="average"):
        """Per policy-year (frequency, Tweedie) or per claim (severity), 2025 $ for money."""
        lvl = (df["year"].map(self.year_log).to_numpy(float) if year_level == "fitted"
               else self.avg_year_log)
        r = np.exp(self.tree_margin(df).astype(float) + lvl)
        return r * (self.load if self.kind == "severity" else self.off_balance)

    def predict(self, df, year_level="average"):
        """Expected claims (frequency), claim size (severity) or losses (Tweedie) per row."""
        r = self.rate(df, year_level)
        return r if self.kind == "severity" else df["exposure"].to_numpy(float) * r

    def relative_rate(self, df):
        """Average-year rate / training average rate: level-free, for ranking pooled folds."""
        return self.rate(df) / self.train_rate

    # ---- saving
    def meta(self):
        return {"kind": self.kind, "features": self.features, "params": self.params,
                "n_rounds": self.n_rounds, "encoder": self.encoder.to_dict(), "monotone": self.monotone,
                "year_log": {str(k): v for k, v in self.year_log.items()},
                "avg_year_log": self.avg_year_log, "load": self.load, "cap": self.cap,
                "off_balance": self.off_balance, "train_years": self.train_years,
                "train_rate": self.train_rate, "xgboost_version": xgb.__version__,
                "tweedie_power": TWEEDIE_POWER if self.kind == "tweedie" else None}

    def save(self, stem):
        """<stem>.ubj (the trees, binary; gitignored) and <stem>.meta.json (everything else)."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(stem.with_suffix(".ubj")))
        Path(str(stem) + ".meta.json").write_text(json.dumps(self.meta(), indent=2, default=float),
                                                  encoding="utf-8")

    @classmethod
    def load_model(cls, stem):
        stem = Path(stem)
        d = json.loads(Path(str(stem) + ".meta.json").read_text(encoding="utf-8"))
        b = xgb.Booster()
        b.load_model(str(stem.with_suffix(".ubj")))
        return cls(d["kind"], d["features"], d["params"], d["n_rounds"], Encoder.from_dict(d["encoder"]),
                   d["monotone"], b, {int(k): v for k, v in d["year_log"].items()}, d["avg_year_log"],
                   d["load"], d["cap"], d["off_balance"], d["train_years"], d["train_rate"])


def fit_xgb(train, kind, params, n_rounds, features=ALL_FEATURES, monotone=None, cap_quantile=None,
            unmatched_losses=0.0, year_backfit=0, nthread=None):
    """Fit one XGBoost model on `train` (all its years).

    cap_quantile (severity only): cap claims at this quantile of the training claims and add
        the capped-off dollars back as a flat load (the GLM's large-loss machinery).
    unmatched_losses (Tweedie only): training losses in zero-exposure cells, added back by the
        off-balance factor.
    year_backfit: extra rounds of re-estimating the year levels from the fitted model's A/E by
        year and refitting (0 = year-only levels, the default)."""
    enc = Encoder.fit(train, features)
    cap = None
    if kind == "severity" and cap_quantile is not None:
        cap = float(np.quantile(train[sv.TARGET], cap_quantile))
    y_for_levels = sv.cap_amounts(train[sv.TARGET], cap) if kind == "severity" else None
    ylog = year_levels(train, kind, y_for_levels)
    p = xgb_params(kind, params, enc, monotone, nthread)
    for it in range(year_backfit + 1):
        X, y, w, margin = _train_parts(train, kind, enc, ylog, cap)
        booster = xgb.train(p, _dmatrix(X, enc, y, w, margin), int(n_rounds))
        m = XGBModel(kind, list(features), dict(params), int(n_rounds), enc, monotone, booster, ylog)
        if it < year_backfit:                 # re-estimate year levels: log(actual / fitted) by year
            fitted = pd.Series(m.predict(train, "fitted") if kind != "severity" else m.rate(train, "fitted"),
                               index=train.index)
            act = pd.Series(y if kind == "severity" else (train["claims"] if kind == "frequency" else train[LOSS]),
                            index=train.index)
            ae = act.groupby(train["year"]).sum() / fitted.groupby(train["year"]).sum()
            ylog = {k: v + float(np.log(ae[k])) for k, v in ylog.items()}
    # Average-year level: training predictions (without year levels) add up to training actuals.
    tm = np.exp(m.tree_margin(train).astype(float))
    if kind == "frequency":
        m.avg_year_log = float(np.log(train["claims"].sum() / (train["exposure"].to_numpy() * tm).sum()))
        m.train_rate = float(train["claims"].sum() / train["exposure"].sum())
    elif kind == "severity":
        yc = sv.cap_amounts(train[sv.TARGET], cap)
        m.avg_year_log = float(np.log(yc.sum() / tm.sum()))
        m.cap, m.load = cap, (sv.large_loss_load(train[sv.TARGET], cap) if cap is not None else 1.0)
        m.train_rate = float(train[sv.TARGET].mean())
    else:
        m.avg_year_log = float(np.log(train[LOSS].sum() / (train["exposure"].to_numpy() * tm).sum()))
        m.off_balance = float((train[LOSS].sum() + unmatched_losses) / train[LOSS].sum())
        m.train_rate = float((train[LOSS].sum() + unmatched_losses) / train["exposure"].sum())
    # "year" may hold state-year keys such as "NC|2018" (Milestone 13), which stay as text.
    m.train_years = sorted(int(y) if str(y).isdigit() else str(y) for y in train["year"].unique())
    return m


@dataclass
class XGBFreqSev:
    """Pure premium from the two XGBoost models, built exactly like pure_premium.CombinedModel:
    frequency x severity x ICC bridge x off-balance (no county factor: geography, if any, is
    inside the trees)."""
    freq: XGBModel
    sev: XGBModel
    icc_bridge: float = 1.0
    off_balance: float = 1.0
    train_rate: float = None

    def rate(self, df):
        return self.freq.rate(df) * self.sev.rate(df) * self.icc_bridge * self.off_balance

    def predict(self, df):
        return df["exposure"].to_numpy(float) * self.rate(df)

    def relative_rate(self, df):
        return self.rate(df) / self.train_rate


def combine_freq_sev(freq, sev, cells_train, claims_train, unmatched_train=0.0):
    """cells_train: cell table rows of the training years (exposure > 0); claims_train: ALL paid
    claims of those years (incl. ICC-only, for the bridge); unmatched_train: losses in their
    zero-exposure cells. The off-balance makes predicted = actual total training losses."""
    m = XGBFreqSev(freq, sev, sv.icc_bridge_factor(claims_train))
    actual = cells_train[LOSS].sum() + unmatched_train
    m.off_balance = float(actual / m.predict(cells_train).sum())
    m.train_rate = float(actual / cells_train["exposure"].sum())
    return m


# ---------------------------------------------------------------------------
# Scoring (the validation plan's metrics, via the GLM modules so they're identical)
# ---------------------------------------------------------------------------
def null_prediction(train, test, kind, unmatched_train=0.0):
    if kind == "frequency":
        return fg.null_prediction(train, test)
    if kind == "severity":
        return sv.null_prediction(train, test)
    return test["exposure"].to_numpy(float) * (train[LOSS].sum() + unmatched_train) / train["exposure"].sum()


def score(test, pred, kind, train=None, pred_null=None, unmatched_train=0.0, unmatched_test=0.0):
    """One fold's metrics. Frequency: Poisson deviance; severity: Gamma deviance; Tweedie:
    Tweedie deviance at the common yardstick p = 1.5 (as in Milestone 9)."""
    if kind == "frequency":
        return fg.score_fold(test, pred, pred_null)
    if kind == "severity":
        return sv.score_fold(test, pred, pred_null)
    train_rate = (train[LOSS].sum() + unmatched_train) / train["exposure"].sum()
    return pp.score_cells(test, pred, train_rate, all_test_losses=float(test[LOSS].sum() + unmatched_test))


def _unm(unmatched, years):
    return 0.0 if unmatched is None else float(unmatched.reindex(list(years), fill_value=0.0).sum())


def evaluate(df, kind, params=None, n_rounds=None, features=ALL_FEATURES, monotone=None, years=None,
             holdout=False, unmatched=None, cap_quantile=None, keep_cols=(), year_backfit=0, fit_fn=None):
    """Fit on each training split and score the held-out part.
    fit_fn: optional fit_fn(train) -> any model with .predict(test) and .relative_rate(test)
        (e.g. a GLM), scored by exactly the same code on the same rows. Default: fit_xgb.
    holdout=False: leave-one-year-out over `years` (default DEV_YEARS), one row per fold.
    holdout=True: train 2010-2020, test 2021-2025; one row per test year plus a "pooled" row.
    Returns (scores, out-of-fold predictions, the fitted models)."""
    if holdout:
        tr, te = val.time_holdout(df)
        splits = [("holdout", tr, te)]
    else:
        splits = list(val.leave_one_year_out(df, val.DEV_YEARS if years is None else years))
    target = KINDS[kind]["target"]
    rows, oof, models = [], [], {}
    for label, train, test in splits:
        m = fit_fn(train) if fit_fn is not None else fit_xgb(
            train, kind, params, n_rounds, features, monotone, cap_quantile,
            _unm(unmatched, train["year"].unique()), year_backfit)
        models[label] = m
        pred = m.predict(test)
        pnull = null_prediction(train, test, kind, _unm(unmatched, train["year"].unique()))
        o = test[["year", target, *[c for c in ("exposure", "claims") if c in test and c != target],
                  *keep_cols]].copy()
        o["pred"], o["pred_null"], o["relative_rate"] = pred, pnull, m.relative_rate(test)
        oof.append(o)
        groups = [(label, test, pred, pnull)] if not holdout else \
            [(int(y), test[test["year"] == y], pred[(test["year"] == y).to_numpy()],
              pnull[(test["year"] == y).to_numpy()]) for y in sorted(test["year"].unique())] + \
            [("pooled", test, pred, pnull)]
        for g, t, pr, pn in groups:
            un_t = _unm(unmatched, t["year"].unique())
            rows.append({"test_year": g, "cat_year": val.is_cat_year(g) if g != "pooled" else False,
                         **score(t, pr, kind, train, pn, _unm(unmatched, train["year"].unique()), un_t)})
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True), models


# ---------------------------------------------------------------------------
# Tuning: cross-validation with early stopping on the fold MEAN
# ---------------------------------------------------------------------------
def cv_lockstep(df, kind, params, features=ALL_FEATURES, monotone=None, years=val.DEV_YEARS,
                chunk=None, max_rounds=None, patience=150, unmatched=None, nthread=None):
    """Train one model per leave-one-year-out fold, all in step, `chunk` trees at a time.
    After each chunk, score each held-out year (re-based deviance improvement over the
    no-features model, the validation plan's main metric) and average over folds, each year
    counting once. Stop when the average hasn't improved for `patience` trees.

    Only the folds in `years` are used, so with DEV_YEARS no 2021-2025 row is ever touched.
    Returns dict: best_rounds, best_score, curve (round x fold), fold_scores at the best round."""
    max_rounds, chunk = max_rounds or MAX_ROUNDS[kind], chunk or CHUNK[kind]
    folds = []
    for y, train, test in val.leave_one_year_out(df, years):
        assert y not in set(train["year"]) and set(test["year"]) == {y}
        enc = Encoder.fit(train, features)
        ylog = year_levels(train, kind)
        X, lab, w, margin = _train_parts(train, kind, enc, ylog)
        Xt = enc.transform(test)
        un_tr = _unm(unmatched, train["year"].unique())
        folds.append({"year": y, "train": train, "test": test, "enc": enc,
                      "dtrain": _dmatrix(X, enc, lab, w, margin),
                      "dtest": _dmatrix(Xt, enc, margin=np.zeros(len(test))),
                      "base": test["exposure"].to_numpy(float) if kind != "severity" else np.ones(len(test)),
                      "null": null_prediction(train, test, kind, un_tr), "un_tr": un_tr,
                      "un_te": _unm(unmatched, [y]),
                      "p": xgb_params(kind, params, enc, monotone, nthread), "booster": None})
    curve, best, best_r, r = [], -np.inf, 0, 0
    while r < max_rounds:
        r += chunk
        for f in folds:
            f["booster"] = xgb.train(f["p"], f["dtrain"], chunk, xgb_model=f["booster"])
        for f in folds:
            pred = f["base"] * np.exp(f["booster"].predict(f["dtest"], output_margin=True).astype(float))
            s = score(f["test"], pred, kind, f["train"], f["null"], f["un_tr"], f["un_te"])
            curve.append({"round": r, "test_year": f["year"], "cat_year": val.is_cat_year(f["year"]),
                          "improvement_rebased_pct": s["improvement_rebased_pct"], "gini": s["gini"]})
        mean = float(np.mean([c["improvement_rebased_pct"] for c in curve if c["round"] == r]))
        if mean > best + 1e-9:
            best, best_r = mean, r
        elif r - best_r >= patience:
            break
    curve = pd.DataFrame(curve)
    at = curve[curve["round"] == best_r]
    return {"best_rounds": int(best_r), "best_score": best, "stopped_at": int(r), "curve": curve,
            "fold_scores": at.reset_index(drop=True),
            "non_cat": float(at.loc[~at["cat_year"], "improvement_rebased_pct"].mean()),
            "cat": float(at.loc[at["cat_year"], "improvement_rebased_pct"].mean()),
            "gini": float(at["gini"].mean()),
            "folds_beating_null": int((at["improvement_rebased_pct"] > 0).sum())}


# min_child_weight = the smallest total HESSIAN allowed in a leaf, and the hessian's unit
# depends on the objective: Poisson ~ expected claims (x e^0.7, XGBoost's max_delta_step),
# Gamma ~ 1 per claim, Tweedie ~ exposure x loss cost^0.3 (so about 1.5 per policy-year).
SEARCH_MCW = {"frequency": (1.0, 300.0), "severity": (3.0, 150.0), "tweedie": (100.0, 100_000.0)}
MAX_ROUNDS = {"frequency": 1500, "severity": 4000, "tweedie": 1500}
CHUNK = {"frequency": 10, "severity": 20, "tweedie": 10}     # trees added between CV checks


def sample_params(rng, kind):
    """One random draw from the search space (log-uniform where the scale matters)."""
    def logu(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return {"max_depth": int(rng.integers(2, 7)), "learning_rate": logu(0.01, 0.1),
            "min_child_weight": logu(*SEARCH_MCW[kind]), "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)), "reg_lambda": logu(0.1, 50.0),
            "reg_alpha": 0.0 if rng.random() < 0.5 else logu(0.1, 5.0),
            "gamma": 0.0 if rng.random() < 0.5 else logu(0.01, 1.0)}


PARAM_COLS = ["max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree",
              "reg_lambda", "reg_alpha", "gamma"]


def random_search(df, kind, n_configs, features=ALL_FEATURES, monotone=None, out_csv=None, seed=SEED,
                  unmatched=None, **cv_kw):
    """Randomized search over n_configs parameter draws, each scored by cv_lockstep on the
    development folds. Results are appended to out_csv after every config (so an interrupted
    run resumes where it stopped). Prints progress."""
    rng = np.random.default_rng(seed)
    draws = [sample_params(rng, kind) for _ in range(n_configs)]
    done = pd.read_csv(out_csv) if out_csv and Path(out_csv).exists() else pd.DataFrame()
    rows = done.to_dict("records")
    t0 = time.time()
    for i, prm in enumerate(draws):
        if i < len(done):
            continue
        t = time.time()
        res = cv_lockstep(df, kind, prm, features, monotone, unmatched=unmatched, **cv_kw)
        rows.append({"config": i, **prm, "best_rounds": res["best_rounds"], "stopped_at": res["stopped_at"],
                     "score": res["best_score"], "non_cat": res["non_cat"], "cat": res["cat"],
                     "gini": res["gini"], "folds_beating_null": res["folds_beating_null"],
                     "seconds": time.time() - t})
        best = max(r["score"] for r in rows)
        print(f"  [{kind}] config {i + 1}/{n_configs}: depth {prm['max_depth']}, lr {prm['learning_rate']:.3f}, "
              f"mcw {prm['min_child_weight']:.1f} -> {res['best_score']:.2f}% at {res['best_rounds']} trees "
              f"({time.time() - t:.0f}s; best so far {best:.2f}%; elapsed {(time.time() - t0) / 60:.1f} min)",
              flush=True)
        if out_csv:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def choose_config(results, tolerance=0.1):
    """The best mean development-LOYO score; among configs within `tolerance` points of it,
    the SIMPLEST (shallowest trees, then fewest trees). Decided before tuning."""
    top = results[results["score"] >= results["score"].max() - tolerance]
    return top.sort_values(["max_depth", "best_rounds", "score"], ascending=[True, True, False]).iloc[0]


def params_of(row):
    return {c: (int(row[c]) if c == "max_depth" else float(row[c])) for c in PARAM_COLS}


def paired_comparison(a, b):
    """Fold-by-fold comparison of two cv_lockstep results: mean difference (b - a) in points
    and the number of folds where b is better."""
    fa = a["fold_scores"].set_index("test_year")["improvement_rebased_pct"]
    fb = b["fold_scores"].set_index("test_year")["improvement_rebased_pct"]
    return {"mean_diff_pts": float((fb - fa).mean()), "folds_better": int((fb > fa).sum()),
            "folds": int(len(fa))}


# ---------------------------------------------------------------------------
# Looking inside the model
# ---------------------------------------------------------------------------
def importance_by_feature(model):
    """Total gain per ORIGINAL feature (one-hot columns summed), as % of all gain. Gain = how
    much the splits on that feature improved the training objective, added over all trees."""
    g = pd.Series(model.booster.get_score(importance_type="total_gain"))
    by = g.groupby(g.index.map(model.encoder.source)).sum()
    by = by.reindex(model.features, fill_value=0.0)
    return (100 * by / by.sum()).sort_values(ascending=False)


def interaction_pairs(model, top=10):
    """A rough interaction measure from the tree structure: whenever a split on feature B sits
    directly below a split on feature A, the tree is treating B differently depending on A.
    Adds up the child splits' gain for each (A, B) pair (unordered, different features only),
    as % of all split gain."""
    t = model.booster.trees_to_dataframe()
    splits = t[t["Feature"] != "Leaf"].set_index("ID")
    src = splits["Feature"].map(model.encoder.source)
    rows = []
    for nid, r in splits.iterrows():
        for child in (r["Yes"], r["No"]):
            if child in splits.index:
                a, b = src[nid], src[child]
                if a != b:
                    rows.append((tuple(sorted((a, b))), splits.at[child, "Gain"]))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=["pair", "gain"]).groupby("pair")["gain"].sum()
    s = 100 * s / splits["Gain"].sum()
    s.index = [f"{a} x {b}" for a, b in s.index]
    return s.sort_values(ascending=False).head(top)


def levels_of(feature, df):
    if feature in ORDINAL:
        return ORDINAL[feature]
    if feature in BINARY:
        return sorted(df[feature].astype(str).unique())
    return sorted(df[feature].astype(str).unique())


def _set_level(df, feature, level):
    d = df.copy()
    if df[feature].dtype == bool:
        d[feature] = level == "True"
    else:
        d[feature] = level
    return d


def partial_dependence(model, df, feature, weight=None, base=None):
    """For each level of `feature`: set EVERY row to that level, keep everything else as it is,
    and take the (exposure- or claim-) weighted average predicted rate. Divided by the base
    level's value, this is the model's average "relativity" for the feature, directly
    comparable with a GLM relativity."""
    w = np.ones(len(df)) if weight is None else df[weight].to_numpy(float)
    out = {}
    for l in levels_of(feature, df):
        out[l] = float(np.average(model.rate(_set_level(df, feature, l)), weights=w))
    s = pd.Series(out)
    return s / s[base] if base is not None else s


def two_way_interaction(model, df, fa, fb, base_a, base_b, weight=None):
    """Two-way partial dependence relative to (base_a, base_b), and the INTERACTION RATIO:
    PD(a, b) x PD(base) / (PD(a, base) x PD(base, b)). A purely multiplicative model (a GLM
    without interaction terms) gives exactly 1 everywhere."""
    w = np.ones(len(df)) if weight is None else df[weight].to_numpy(float)
    pd2 = pd.DataFrame(index=levels_of(fa, df), columns=levels_of(fb, df), dtype=float)
    for la in pd2.index:
        da = _set_level(df, fa, la)
        for lb in pd2.columns:
            pd2.loc[la, lb] = np.average(model.rate(_set_level(da, fb, lb)), weights=w)
    rel = pd2 / pd2.loc[base_a, base_b]
    ratio = rel.div(rel[base_b], axis=0).div(rel.loc[base_a], axis=1)
    return rel, ratio


# ---------------------------------------------------------------------------
# Settings file and final fits
# ---------------------------------------------------------------------------
def read_settings(path=SETTINGS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def write_settings(s, path=SETTINGS_PATH):
    Path(path).write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")


def load_kind_data(kind, verbose=True):
    """(table, unmatched losses by year or None) for a model kind."""
    if kind == "severity":
        return load_severity_table(verbose), None
    return load_cell_table(verbose=verbose), (unmatched_losses_by_year() if kind == "tweedie" else None)


N_CONFIGS = {"frequency": 40, "severity": 40, "tweedie": 20}
VARIANT_TAGS = {"no geography": "nogeo", "+ Tidewater": "tidal", "+ Tidewater + county group": "county"}


def tuning_csv(kind, variant):
    return REPORTS / f"07_xgb_tuning_{kind}_{VARIANT_TAGS[variant]}.csv"


def cmd_tune(kind):
    """A separate randomized search for EACH geography variant (the same random draws, so the
    variants are compared config by config), all scored on the development folds only.
    Why separately: a dry run showed county group makes unseen years much worse, so a search
    run with county group in picks settings that work around it; the geography question is
    only fair if each variant gets its own best settings."""
    df, unm = load_kind_data(kind)
    for variant, feats in GEOGRAPHY_VARIANTS.items():
        print(f"Tuning {kind} / {variant}: {N_CONFIGS[kind]} random configs x 11 development folds "
              f"(2010-2020), {len(feats)} features, no monotone constraints.", flush=True)
        random_search(df, kind, N_CONFIGS[kind], feats, None, tuning_csv(kind, variant), unmatched=unm)


def cmd_compare(kind, top_k=5):
    """(1) Geography: the best tuned config of each variant, compared fold by fold with the
    GLMs' keep rule. (2) Monotone constraints on the winning variant's top_k configs.
    Writes reports/07_xgb_{geography,monotone}_<kind>*.csv and models/xgb_settings.json."""
    df, unm = load_kind_data(kind)
    s = read_settings()

    # ---- (1) geography
    geo_rows, geo_runs, chosen = [], {}, {}
    for name, feats in GEOGRAPHY_VARIANTS.items():
        res = pd.read_csv(tuning_csv(kind, name))
        ch = choose_config(res)
        chosen[name] = (ch, res)
        cv = cv_lockstep(df, kind, params_of(ch), feats, None, unmatched=unm)
        geo_runs[name] = cv
        geo_rows.append({"variant": name, "features": "|".join(feats), "config": int(ch["config"]),
                         **params_of(ch), "best_rounds": cv["best_rounds"], "score": cv["best_score"],
                         "best_config_score_in_search": float(res["score"].max()),
                         "median_config_score": float(res["score"].median()),
                         "non_cat": cv["non_cat"], "cat": cv["cat"], "gini": cv["gini"],
                         "folds_beating_null": cv["folds_beating_null"]})
        print(f"  geography '{name}': config {int(ch['config'])}, {cv['best_score']:.2f}% at "
              f"{cv['best_rounds']} trees", flush=True)
    # Paired search comparison: how often does each variant beat 'no geography' on the SAME draw?
    base_res = chosen["no geography"][1].set_index("config")["score"]
    for r in geo_rows:
        other = chosen[r["variant"]][1].set_index("config")["score"]
        r["draws_better_than_no_geography"] = int((other > base_res).sum())
        r["draws"] = int(len(other))
    names = list(GEOGRAPHY_VARIANTS)
    # Same keep rule as the GLMs: a geography step is kept if it improves the mean development
    # LOYO score AND is better in at least 6 of the 11 folds than the current feature set.
    current, steps = names[0], []
    for nxt in names[1:]:
        c = paired_comparison(geo_runs[current], geo_runs[nxt])
        keep = c["mean_diff_pts"] > 0 and c["folds_better"] >= 6
        steps.append({"from": current, "to": nxt, **c, "kept": bool(keep)})
        if keep:
            current = nxt
    pd.DataFrame(geo_rows).to_csv(REPORTS / f"07_xgb_geography_{kind}.csv", index=False)
    pd.concat([v["fold_scores"].assign(variant=k) for k, v in geo_runs.items()]).to_csv(
        REPORTS / f"07_xgb_geography_folds_{kind}.csv", index=False)
    pd.DataFrame(steps).to_csv(REPORTS / f"07_xgb_geography_steps_{kind}.csv", index=False)
    print(f"Geography rule -> '{current}'. Steps: {steps}", flush=True)
    feats = GEOGRAPHY_VARIANTS[current]

    # ---- (2) monotone constraints on the chosen variant's top configs
    res = chosen[current][1]
    top = res.sort_values("score", ascending=False).head(top_k)
    rows, runs = [], {}
    print(f"Monotone comparison for {kind}: top {top_k} configs of '{current}', constraints {MONOTONE[kind]}",
          flush=True)
    chosen_cfg = int(chosen[current][0]["config"])
    if chosen_cfg not in set(top["config"].astype(int)):          # always include the chosen config
        top = pd.concat([top, res[res["config"] == chosen_cfg]])
    for _, r in top.iterrows():
        prm = params_of(r)
        for label, mono in (("unconstrained", None), ("constrained", MONOTONE[kind])):
            if label == "unconstrained" and int(r["config"]) == chosen_cfg:
                cv = geo_runs[current]                              # already run above
            else:
                cv = cv_lockstep(df, kind, prm, feats, mono, unmatched=unm)
            runs[(int(r["config"]), label)] = cv
            rows.append({"config": int(r["config"]), "variant": label, **prm, "best_rounds": cv["best_rounds"],
                         "score": cv["best_score"], "non_cat": cv["non_cat"], "cat": cv["cat"],
                         "gini": cv["gini"], "folds_beating_null": cv["folds_beating_null"]})
            print(f"  config {int(r['config'])} {label}: {cv['best_score']:.2f}% at {cv['best_rounds']} trees",
                  flush=True)
    mono_tbl = pd.DataFrame(rows)
    mono_tbl.to_csv(REPORTS / f"07_xgb_monotone_{kind}.csv", index=False)
    pd.concat([v["fold_scores"].assign(config=k[0], variant=k[1]) for k, v in runs.items()]).to_csv(
        REPORTS / f"07_xgb_monotone_folds_{kind}.csv", index=False)
    # Pre-stated rule, on the chosen config's pair: use the constraints unless they cost more than
    # 0.25 points of mean development-LOYO improvement, or are worse in 8 or more of the 11 folds.
    pair = paired_comparison(runs[(chosen_cfg, "unconstrained")], runs[(chosen_cfg, "constrained")])
    piv = mono_tbl.pivot(index="config", columns="variant", values="score")
    avg_diff = float((piv["constrained"] - piv["unconstrained"]).mean())
    use_mono = not (pair["mean_diff_pts"] < -0.25 or (pair["folds"] - pair["folds_better"]) >= 8)
    variant = "constrained" if use_mono else "unconstrained"
    ch = choose_config(mono_tbl[mono_tbl["variant"] == variant])
    final = runs[(int(ch["config"]), variant)]
    print(f"Monotone rule: config {chosen_cfg} constrained - unconstrained = {pair['mean_diff_pts']:+.2f} pts, "
          f"better in {pair['folds_better']}/{pair['folds']} folds -> {variant}; final config "
          f"{int(ch['config'])}, {final['best_rounds']} trees, {final['best_score']:.2f}%.", flush=True)

    s[kind] = {
        "objective": KINDS[kind]["objective"],
        **({"tweedie_variance_power": TWEEDIE_POWER} if kind == "tweedie" else {}),
        "tuning": {"configs_per_variant": int(N_CONFIGS[kind]), "folds": "LOYO 2010-2020 (11 folds)",
                   "metric": "mean re-based deviance improvement vs the no-features model, % (each year once)",
                   "early_stopping": f"lockstep folds, {CHUNK[kind]}-tree chunks, stop after 150 trees without improvement, max {MAX_ROUNDS[kind]}",
                   "tie_rule": "within 0.1 pts of the best: shallowest, then fewest trees"},
        "geography_test": {"steps": steps, "chosen": current,
                           "rule": "keep a geography step if mean dev-LOYO improves and >= 6 of 11 folds are better",
                           "scores": {r["variant"]: r["score"] for r in geo_rows}},
        "monotone_test": {"constraints": MONOTONE[kind], "configs_compared": int(len(top)),
                          "chosen_config_pair": pair, "avg_diff_pts_top_configs": avg_diff,
                          "rule": "use constraints unless > 0.25 pts worse or worse in >= 8 of 11 folds",
                          "use_constraints": bool(use_mono)},
        "features": feats, "monotone": MONOTONE[kind] if use_mono else None,
        "config": int(ch["config"]), "params": params_of(ch), "n_rounds": int(final["best_rounds"]),
        "dev_loyo_score": float(final["best_score"]), "dev_loyo_non_cat": final["non_cat"],
        "dev_loyo_cat": final["cat"], "dev_loyo_gini": final["gini"],
    }
    write_settings(s)


def fit_final(kind, settings=None, years=None, unmatched=None, df=None):
    s = (settings or read_settings())[kind]
    if df is None:
        df, unmatched = load_kind_data(kind, verbose=False)
    if years is not None:
        df = df[df["year"].isin(years)]
    return fit_xgb(df, kind, s["params"], s["n_rounds"], s["features"], s.get("monotone"),
                   unmatched_losses=_unm(unmatched, df["year"].unique()))


def cmd_fit_final():
    s = read_settings()
    for kind in KINDS:
        if kind not in s or "features" not in s[kind]:
            print(f"skip {kind}: not tuned/compared yet")
            continue
        df, unm = load_kind_data(kind, verbose=False)
        for tag, yrs in (("dev", val.DEV_YEARS), ("all", val.ALL_YEARS)):
            m = fit_final(kind, s, yrs, unm, df)
            m.save(MODEL_DIR / f"xgb_{kind}_{tag}")
            print(f"saved models/xgb_{kind}_{tag}.ubj ({m.n_rounds} trees, years {yrs[0]}-{yrs[-1]})")


# ---------------------------------------------------------------------------
# Self-test on simulated data with known answers
# ---------------------------------------------------------------------------
def _simulate(seed=5, n=6000):
    rng = np.random.default_rng(seed)
    years = np.array(val.ALL_YEARS[:6])
    d = pd.DataFrame({
        "year": rng.choice(years, n),
        "flood_zone_group": rng.choice(["A/V high risk", "B/C/X moderate-low risk"], n),
        "construction_band": rng.choice(ORDINAL["construction_band"], n),
        "building_coverage_band": rng.choice(ORDINAL["building_coverage_band"], n),
        "occupancy_group": rng.choice(["single-unit residential", "multi-unit residential", "non-residential"], n),
        "elevated": rng.random(n) < 0.4, "post_firm": rng.random(n) < 0.5,
        "primary_residence": rng.random(n) < 0.7,
        "floors_group": rng.choice(ORDINAL["floors_group"], n),
        "coastal_tidal": rng.random(n) < 0.8,
        "county_group": rng.choice(["A county", "B county", "C county"], n),
        "exposure": rng.uniform(5, 50, n)})
    year_eff = dict(zip(years, [1.0, 3.0, 0.5, 1.0, 0.7, 1.5]))
    # True frequency: B/C/X 0.5x; year built bumps UP for 1980-99 (so a -1 constraint must flatten it)
    cons = {"before 1960": 2.0, "1960-1979": 2.0, "1980-1999": 2.4, "2000 or later": 0.6}
    rate = 0.02 * d["year"].map(year_eff) * np.where(d["flood_zone_group"].str.startswith("B"), 0.5, 1.0) \
        * d["construction_band"].map(cons)
    d["claims"] = rng.poisson(d["exposure"] * rate)
    # True severity: coverage band multiplies size
    cov = dict(zip(ORDINAL["building_coverage_band"], [0.3, 0.7, 0.85, 1.0]))
    mu = 30000 * d["building_coverage_band"].map(cov) * d["year"].map(year_eff) ** 0.3
    d[sv.TARGET] = rng.gamma(2.0, mu / 2.0)
    d[LOSS] = d["claims"] * d[sv.TARGET]
    return d


def _self_test():
    d = _simulate()
    prm = {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 5.0, "subsample": 1.0,
           "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0}
    feats = ["flood_zone_group", "construction_band", "building_coverage_band", "occupancy_group", "county_group"]

    # 1. Encoder: codes, one-hot, unseen level -> all zeros, monotone string in column order
    enc = Encoder.fit(d, feats)
    X = enc.transform(d.head(3))
    assert X.shape == (3, len(enc.columns)) and enc.columns[-3:] == ["county_group=A county",
                                                                    "county_group=B county", "county_group=C county"]
    new = d.head(1).assign(county_group="Z county")
    assert enc.transform(new)[0, -3:].sum() == 0
    ms = enc.monotone_string({"flood_zone_group": -1, "building_coverage_band": 1})
    assert ms.startswith("(-1,0,1,") and ms.count(",") == len(enc.columns) - 1

    # 2. Frequency: base_margin semantics, average-year balance, exposure scaling, known effect
    m = fit_xgb(d, "frequency", prm, 150, feats)
    tm1 = m.tree_margin(d.head(50))
    dm = _dmatrix(m.encoder.transform(d.head(50)), m.encoder, margin=np.full(50, 2.5))
    assert np.allclose(m.booster.predict(dm, output_margin=True) - tm1, 2.5, atol=1e-5), "base_margin not additive"
    assert abs(m.predict(d).sum() / d["claims"].sum() - 1) < 1e-6, "average-year level doesn't balance"
    dd = d.head(200).assign(exposure=d.head(200)["exposure"] * 2)
    assert np.allclose(m.predict(dd), 2 * m.predict(d.head(200)))
    pdz = partial_dependence(m, d, "flood_zone_group", "exposure", "A/V high risk")
    assert 0.4 < pdz["B/C/X moderate-low risk"] < 0.62, pdz
    # Fitted year levels track the simulated year effects (year 2 = 3x year 1)
    fy = pd.Series(m.year_log)
    assert 2.4 < np.exp(fy[2011] - fy[2010]) < 3.7, fy

    # 3. Monotone constraint: the 1980-99 bump must not appear when construction is -1
    mc = fit_xgb(d, "frequency", prm, 150, feats, {"construction_band": -1})
    pdc = partial_dependence(mc, d, "construction_band", "exposure")
    assert (np.diff(pdc.to_numpy()) <= 1e-9).all(), pdc
    pdu = partial_dependence(m, d, "construction_band", "exposure")
    assert pdu["1980-1999"] > pdu["1960-1979"], "unconstrained model should find the bump"

    # 4. Save / load round trip gives identical predictions
    tmp = Path(__file__).resolve().parent / "_xgb_selftest_tmp"
    try:
        m.save(tmp)
        m2 = XGBModel.load_model(tmp)
        assert np.allclose(m2.predict(d), m.predict(d)) and m2.encoder.columns == m.encoder.columns
    finally:
        tmp.with_suffix(".ubj").unlink(missing_ok=True)
        Path(str(tmp) + ".meta.json").unlink(missing_ok=True)

    # 5. Severity: coverage effect recovered; monotone +1 respected; cap + load
    c = d[d["claims"] > 0].copy()
    ms_ = fit_xgb(c, "severity", prm, 100, feats, {"building_coverage_band": 1})
    pc = partial_dependence(ms_, c, "building_coverage_band")
    assert (np.diff(pc.to_numpy()) >= -1e-9).all() and 0.15 < pc.iloc[0] / pc.iloc[-1] < 0.5, pc
    assert abs(ms_.predict(c).sum() / c[sv.TARGET].sum() - 1) < 1e-6
    mcap = fit_xgb(c, "severity", prm, 50, feats, cap_quantile=0.95)
    assert mcap.load > 1 and abs(mcap.predict(c).sum() / c[sv.TARGET].sum() - 1) < 1e-6

    # 6. Tweedie: off-balance puts back the extra (unmatched) losses
    mt = fit_xgb(d, "tweedie", prm, 100, feats, unmatched_losses=0.05 * d[LOSS].sum())
    assert abs(mt.predict(d).sum() / (1.05 * d[LOSS].sum()) - 1) < 1e-6

    # 7. Lockstep CV: uses only the given years, picks a round, scores beat the null
    cv = cv_lockstep(d, "frequency", prm, feats, years=val.ALL_YEARS[:6], chunk=25, max_rounds=300, patience=75)
    assert 0 < cv["best_rounds"] <= 300 and cv["best_score"] > 0 and len(cv["fold_scores"]) == 6

    # 8. evaluate() returns one row per fold and out-of-fold predictions for every test row
    sc, oof, _ = evaluate(d, "frequency", prm, 50, feats, years=val.ALL_YEARS[:6])
    assert len(sc) == 6 and len(oof) == len(d)
    imp = importance_by_feature(m)
    assert imp.index[0] in ("construction_band", "flood_zone_group") and abs(imp.sum() - 100) < 1e-6
    assert len(interaction_pairs(m)) > 0

    # 9. Frequency x severity combination balances to actual losses (+ unmatched)
    cc = c.assign(total_2025=c[sv.TARGET] * 1.01)               # pretend 1% ICC on top
    fs = combine_freq_sev(m, ms_, d, cc, unmatched_train=100.0)
    assert abs(fs.icc_bridge - 1.01) < 1e-9
    assert abs(fs.predict(d).sum() / (d[LOSS].sum() + 100.0) - 1) < 1e-9
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="XGBoost challenger models (Milestone 10)")
    ap.add_argument("command", nargs="?", default="selftest", choices=["selftest", "tune", "compare", "fit-final"])
    ap.add_argument("kind", nargs="?", choices=list(KINDS))
    a = ap.parse_args()
    if a.command == "selftest":
        print("xgb_models self-test passed:", _self_test())
    elif a.command == "tune":
        cmd_tune(a.kind)
    elif a.command == "compare":
        cmd_compare(a.kind)
    else:
        cmd_fit_final()
