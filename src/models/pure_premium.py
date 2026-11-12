"""
PURE PREMIUM: expected loss cost per policy-year = frequency x severity (Milestone 9).

Plain-English summary (full explanations in LEARNING_NOTES.md, Milestone 9):
  * The frequency GLM (src/models/frequency_glm.py) gives expected paid claims per policy-year;
    the severity GLM (src/models/severity_glm.py) gives the expected size of a paid claim.
    Their product is the expected loss per policy-year, the "pure premium":
        pure premium = base rate x base severity x (product of every factor's relativity)
                       x large-loss load x ICC bridge x off-balance x county factor
  * Large-loss load: Milestone 8 chose NOT to cap claims, so this load is 1.0 (it is stored
    inside the severity model and applied by SevGLM.predict).
  * ICC bridge: severity models building + contents only; this factor adds ICC back
    (severity_glm.icc_bridge_factor).
  * Off-balance correction: one multiplier so that predicted total losses equal actual total
    losses on the data the model was built from. It is needed because the two models are each
    calibrated on their own (the average of a product is not the product of averages), the
    Gamma log link balances ratios rather than dollars, and 1% of claims have no matching
    exposure.
  * County factor: each county's actual-to-expected (A/E) ratio against the GLM, shrunk toward
    1 by CREDIBILITY (how much we trust the county's own experience): limited fluctuation
    (Z = min(1, sqrt(n / n_full))) or Buhlmann-Straub (Z = w / (w + k)).
  * Tweedie GLM: one model for pure premium directly, as a cross-check.
  * Earned premium per cell, for loss ratios (actual NFIP premiums vs losses).

Run the self-test:  .venv\\Scripts\\python.exe src\\models\\pure_premium.py
"""

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

MODELS = Path(__file__).resolve().parent
SRC = MODELS.parent
for p in (SRC, MODELS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import validation as val                      # noqa: E402
import frequency_glm as fg                    # noqa: E402
import severity_glm as sv                     # noqa: E402

ROOT = SRC.parent
CELLS_PATH = ROOT / "data" / "processed" / "va_model_data.parquet"
PREMIUM_PATH = ROOT / "data" / "processed" / "va_cell_premium.parquet"
FREQ_JSON = ROOT / "models" / "frequency_glm.json"
SEV_JSON = ROOT / "models" / "severity_glm.json"
LOSS = "losses_2025"           # cell losses in 2025 dollars, incl. ICC
REF_POWER = 1.5                # Tweedie power used as the COMMON yardstick when scoring models


# ---------------------------------------------------------------------------
# Loading the saved models
# ---------------------------------------------------------------------------
def load_saved_models(freq_path=FREQ_JSON, sev_path=SEV_JSON):
    """(FreqGLM, SevGLM, severity JSON dict) from the committed model files."""
    return (fg.load_model_json(freq_path), sv.load_model_json(sev_path),
            json.loads(Path(sev_path).read_text(encoding="utf-8")))


def bases_without_year(spec):
    """A model's base levels, minus the year (a refit on other years needs its own year base)."""
    return {k: v for k, v in spec.bases.items() if k != "year"}


# ---------------------------------------------------------------------------
# The combined model
# ---------------------------------------------------------------------------
@dataclass
class CombinedModel:
    freq: object                  # frequency_glm.FreqGLM
    sev: object                   # severity_glm.SevGLM (its .load is the large-loss load)
    icc_bridge: float = 1.0
    off_balance: float = 1.0
    county_factors: pd.Series = None   # index = county FIPS (str), value = factor (missing -> 1)

    @property
    def large_loss_load(self):
        return self.sev.load

    def components(self, df, county=True):
        """Per-row rates (per policy-year, average year): frequency, severity, and the pure
        premium step by step."""
        one = df.assign(exposure=1.0)
        out = pd.DataFrame(index=df.index)
        out["frequency"] = self.freq.predict(one, "average")          # claims per policy-year
        out["severity"] = self.sev.predict(df, "average")             # incl. the large-loss load
        out["pp_raw"] = out["frequency"] * out["severity"] * self.icc_bridge
        out["pp_balanced"] = out["pp_raw"] * self.off_balance
        cf = self.county_factor(df) if county else np.ones(len(df))
        out["county_factor"] = cf
        out["pure_premium"] = out["pp_balanced"] * cf
        return out

    def county_factor(self, df):
        if self.county_factors is None:
            return np.ones(len(df))
        return df["county"].astype(str).map(self.county_factors).fillna(1.0).to_numpy(float)

    def rate(self, df, county=True):
        """Expected loss per policy-year (2025 dollars, average year)."""
        return self.components(df, county)["pure_premium"].to_numpy()

    def predict(self, df, county=True):
        """Expected losses for each row = exposure x rate."""
        return df["exposure"].to_numpy(float) * self.rate(df, county)

    def base_pure_premium(self):
        """Loss cost of the all-base-levels building in an average year (no county factor)."""
        f = np.exp(self.freq.params["const"] + self.freq.avg_year_log)
        s = np.exp(self.sev.params["const"] + self.sev.avg_year_log) * self.sev.load
        return float(f * s * self.icc_bridge * self.off_balance)


def off_balance_factor(model, cells):
    """Actual total losses / predicted total losses (before this correction, no county).
    `cells` must include the zero-exposure ("unmatched") rows so their losses count."""
    pred = model.predict(cells.assign(), county=False) / model.off_balance
    return float(cells[LOSS].sum() / pred.sum())


def fit_combined(cells, claims, freq_features, sev_features, freq_bases=None, sev_bases=None,
                 freq_interactions=(), sev_interactions=()):
    """Refit both GLMs on the given years (cells and claims for the same years), then the ICC
    bridge and the off-balance on those same years.
    freq_interactions / sev_interactions: optional [[feature_a, feature_b], ...] interaction
    terms (Milestone 11's "GLM + interactions" hybrid); both features must be in the model."""
    fc = fg.aggregate(cells[cells["exposure"] > 0], freq_features)
    freq = fg.fit_glm(fc, freq_features, freq_interactions, bases=freq_bases)
    sev = sv.fit_gamma(sv.severity_rows(claims), sev_features, sev_interactions, bases=sev_bases)
    m = CombinedModel(freq, sev, sv.icc_bridge_factor(claims))
    m.off_balance = off_balance_factor(m, cells)
    return m


def combined_relativity_table(freq, sev):
    """Every factor level in either model: frequency relativity, severity relativity (1.0 when the
    factor isn't in that model) and their product, the pure-premium relativity."""
    rows = []
    feats = list(dict.fromkeys(list(freq.spec.features) + list(sev.spec.features)))
    for f in feats:
        spec = freq.spec if f in freq.spec.features else sev.spec
        for lvl in spec.levels[f]:
            col = f"{f}[{lvl}]"
            fr = float(np.exp(freq.params[col])) if col in freq.params.index else 1.0
            sr = float(np.exp(sev.params[col])) if col in sev.params.index else 1.0
            rows.append({"feature": f, "level": lvl, "is_base": lvl == spec.bases[f],
                         "in_frequency": f in freq.spec.features, "in_severity": f in sev.spec.features,
                         "frequency_rel": fr, "severity_rel": sr, "pure_premium_rel": fr * sr})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Credibility-weighted county factor
# ---------------------------------------------------------------------------
def county_ae(model, cells, claims):
    """Actual vs GLM-expected by county (and by county x year, for Buhlmann-Straub).

    Frequency: paid claims vs expected claims, with each year's FITTED year effect, so a
        storm year's overall level is not mistaken for a county effect.
    Severity: building + contents dollars vs expected dollars (fitted year effects), over the
        claims with b+c > 0.
    Returns (by_county, by_county_year)."""
    c = cells[cells["exposure"] > 0]
    fe = model.freq.predict(c, "fitted")
    s = sv.severity_rows(claims)
    se = model.sev.predict(s, "fitted")
    fy = c.assign(f_exp=fe).groupby(["county", "year"]).agg(f_act=("claims", "sum"), f_exp=("f_exp", "sum"))
    sy = s.assign(s_exp=se).groupby(["county", "year"]).agg(
        s_n=(sv.TARGET, "size"), s_act=(sv.TARGET, "sum"), s_exp=("s_exp", "sum"))
    cy = fy.join(sy, how="outer").fillna(0.0)
    by = cy.groupby("county").sum()
    by["freq_ae"] = by["f_act"] / by["f_exp"]
    by["sev_ae"] = (by["s_act"] / by["s_exp"]).where(by["s_n"] > 0, 1.0)
    return by, cy


def full_credibility_standard(p=0.90, k=0.05, cv=None):
    """Expected claims needed for FULL credibility (limited fluctuation):
        frequency:  n_full = (z / k)^2, with z the normal quantile for probability p
                    (p = 90%, k = 5% gives (1.645 / 0.05)^2 = 1,082 claims);
        severity:   n_full = (z / k)^2 x CV^2, with CV = sd / mean of claim size."""
    z = stats.norm.ppf((1 + p) / 2)
    n = (z / k) ** 2
    return float(n if cv is None else n * cv ** 2)


def limited_fluctuation_z(n, n_full):
    """Z = min(1, sqrt(n / n_full)): the square-root rule."""
    return np.minimum(1.0, np.sqrt(np.asarray(n, float) / n_full))


def buhlmann_straub(x, w, group):
    """Buhlmann-Straub credibility from repeated observations per group (here: county x year).

    x: observed ratio (A/E) per group-period, w: its weight (expected claims or claim count).
      EPV ("expected process variance") = average variance of a group's ratio from period to
          period, per unit weight: how noisy one group's experience is.
      VHM ("variance of hypothetical means") = how much the groups' TRUE ratios differ.
      k = EPV / VHM;  Z_i = w_i / (w_i + k).
    Returns dict(k, epv, vhm, mean) and a DataFrame per group (w, xbar, Z)."""
    d = pd.DataFrame({"x": np.asarray(x, float), "w": np.asarray(w, float), "g": np.asarray(group)})
    d = d[d["w"] > 0]
    g = d.groupby("g")
    wi = g["w"].sum()
    xi = g.apply(lambda t: np.average(t["x"], weights=t["w"]), include_groups=False)
    ni = g.size()
    W = wi.sum()
    xbar = float((wi * xi).sum() / W)
    within = d.assign(xi=d["g"].map(xi))
    keep = ni[ni > 1].index
    within = within[within["g"].isin(keep)]
    epv = float((within["w"] * (within["x"] - within["xi"]) ** 2).sum() / (ni[keep] - 1).sum())
    r = len(wi)
    vhm = float(((wi * (xi - xbar) ** 2).sum() - (r - 1) * epv) / (W - (wi ** 2).sum() / W))
    k = epv / vhm if vhm > 0 else np.inf
    z = wi / (wi + k) if np.isfinite(k) else wi * 0.0
    return {"k": k, "epv": epv, "vhm": vhm, "mean": xbar}, pd.DataFrame({"w": wi, "xbar": xi, "Z": z})


def credibility_factors(by, cy, method="limited_fluctuation", p=0.90, k=0.05, sev_cv=None):
    """County factors = (Z_f x freq A/E + (1 - Z_f)) x (Z_s x sev A/E + (1 - Z_s)).

    method "limited_fluctuation": Z from claim counts with full-credibility standards
        n_full (frequency) and n_full x CV^2 (severity).
    method "buhlmann_straub": Z from county x year A/E (weights: expected claims for frequency,
        claim count for severity).
    Returns (table per county, parameters dict). Factors are NOT yet re-centred (see recentre)."""
    t = by.copy()
    params = {"method": method}
    if method == "limited_fluctuation":
        nf, ns = full_credibility_standard(p, k), full_credibility_standard(p, k, sev_cv)
        t["Z_freq"] = limited_fluctuation_z(t["f_act"], nf)
        t["Z_sev"] = limited_fluctuation_z(t["s_n"], ns)
        params.update({"p": p, "k": k, "n_full_freq": nf, "sev_cv": sev_cv, "n_full_sev": ns})
    elif method == "buhlmann_straub":
        c = cy.reset_index()
        cf = c[c["f_exp"] > 0]
        pf, tf = buhlmann_straub(cf["f_act"] / cf["f_exp"], cf["f_exp"], cf["county"])
        cs = c[c["s_n"] > 0]
        ps, ts = buhlmann_straub(cs["s_act"] / cs["s_exp"], cs["s_n"], cs["county"])
        t["Z_freq"] = tf["Z"].reindex(t.index).fillna(0.0)
        t["Z_sev"] = ts["Z"].reindex(t.index).fillna(0.0)
        params.update({"freq": pf, "sev": ps})
    else:
        raise ValueError(method)
    t["freq_factor"] = t["Z_freq"] * t["freq_ae"] + (1 - t["Z_freq"])
    t["sev_factor"] = t["Z_sev"] * t["sev_ae"] + (1 - t["Z_sev"])
    t["factor_raw"] = t["freq_factor"] * t["sev_factor"]
    return t, params


def recentre(factors, model, cells):
    """Scale county factors so they don't change the statewide total: the second off-balance.
    Returns (re-centred factors as a Series, the scale applied)."""
    base = model.predict(cells, county=False)
    f = cells["county"].astype(str).map(factors).fillna(1.0).to_numpy()
    scale = float(base.sum() / (base * f).sum())
    return factors * scale, scale


def with_county_factors(model, cells, claims, method, **kw):
    """Copy of `model` with re-centred credibility county factors estimated on (cells, claims)."""
    by, cy = county_ae(model, cells, claims)
    tbl, params = credibility_factors(by, cy, method, **kw)
    fac, scale = recentre(tbl["factor_raw"].rename(index=str), model, cells)
    tbl["factor"] = fac.to_numpy()
    m = CombinedModel(model.freq, model.sev, model.icc_bridge, model.off_balance, fac)
    params["recentre_scale"] = scale
    return m, tbl, params


# ---------------------------------------------------------------------------
# Tweedie GLM (pure premium in one model)
# ---------------------------------------------------------------------------
@dataclass
class TweedieGLM:
    spec: fg.Spec
    params: pd.Series
    power: float
    avg_year_log: float = 0.0
    off_balance: float = 1.0
    phi: float = None
    result: object = None

    def eta(self, df, include_year=True):
        return fg.design_matrix(df, self.spec, include_year).to_numpy() @ self.params.to_numpy()

    def rate(self, df):
        return np.exp(self.eta(df, False) + self.avg_year_log) * self.off_balance

    def predict(self, df):
        return df["exposure"].to_numpy(float) * self.rate(df)

    def relative_rate(self, df):
        return np.exp(self.eta(df, False) - self.params["const"])

    def relativities(self):
        p = self.params[[c for c in self.params.index if not c.startswith("year[") and c != "const"]]
        return np.exp(p)


def fit_tweedie(cells, features, power, bases=None):
    """Tweedie GLM, log link: loss cost (losses / exposure) with exposure as the weight,
    calendar year as a factor, average-year level, and an off-balance for the losses in
    zero-exposure cells. Cells with identical features + year are summed first, which leaves
    the coefficients unchanged (the cell sums are sufficient for a weighted Tweedie GLM)."""
    m = cells[cells["exposure"] > 0]
    agg = m.groupby(list(features) + ["year"], observed=True)[["exposure", LOSS]].sum().reset_index()
    spec = fg.make_spec(agg, features, (), True, bases)
    X = fg.design_matrix(agg, spec)
    w = agg["exposure"].to_numpy(float)
    y = agg[LOSS].to_numpy(float) / w
    fam = sm.families.Tweedie(var_power=power, link=sm.families.links.Log())
    start = np.zeros(X.shape[1])
    start[0] = np.log(np.average(y, weights=w))
    res = sm.GLM(y, X, family=fam, var_weights=w).fit(start_params=start)
    params = pd.Series(res.params, index=X.columns)
    base_eta = fg.design_matrix(agg, spec, include_year=False).to_numpy() @ params.to_numpy()
    avg = float(np.log(agg[LOSS].sum() / (w * np.exp(base_eta)).sum()))
    tw = TweedieGLM(spec, params, power, avg, 1.0, float(res.scale), res)
    tw.off_balance = float(cells[LOSS].sum() / tw.predict(cells).sum())
    return tw


def tweedie_power_from_gamma_shape(alpha):
    """In a compound Poisson-Gamma (the Tweedie with 1 < p < 2), the power and the Gamma shape
    alpha of each claim are linked: p = (alpha + 2) / (alpha + 1)."""
    return (alpha + 2) / (alpha + 1)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def tweedie_deviance(y, mu, p=REF_POWER, w=None):
    """Tweedie deviance for 1 < p < 2 (y may be 0):
        2 x sum w [ y^(2-p) / ((1-p)(2-p)) - y mu^(1-p) / (1-p) + mu^(2-p) / (2-p) ].
    Used on loss COSTS (losses / exposure) with exposure as the weight."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    d = y ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)
    return float(2 * (w * d).sum())


def loss_gini(losses, pred, exposure):
    """Exposure-weighted Gini of a loss-cost ranking (frequency_glm.gini with losses)."""
    return fg.gini(losses, pred, exposure)


def score_cells(test, pred, train_rate, p=REF_POWER, all_test_losses=None):
    """Per-fold scores on test cells with exposure > 0: raw A/E (vs ALL test losses when given,
    so unmatched claims count), re-based Tweedie deviance and its improvement over the
    no-features model (everyone at the training loss cost), and the Gini."""
    e = test["exposure"].to_numpy(float)
    y = test[LOSS].to_numpy(float)
    pred = np.asarray(pred, float)
    actual = y.sum()
    pr = pred * actual / pred.sum()
    nl = e * actual / e.sum()
    d = {"test_losses": float(actual if all_test_losses is None else all_test_losses),
         "pred_losses": float(pred.sum()), "null_pred_losses": float(e.sum() * train_rate)}
    d["actual_over_expected"] = d["test_losses"] / d["pred_losses"]
    d["deviance_rebased"] = tweedie_deviance(y / e, pr / e, p, e)
    d["null_deviance_rebased"] = tweedie_deviance(y / e, nl / e, p, e)
    d["improvement_rebased_pct"] = 100 * (1 - d["deviance_rebased"] / d["null_deviance_rebased"])
    d["gini"] = loss_gini(y, pred, e) if actual > 0 else np.nan
    return d


def lift_table(losses, pred, exposure, rank_by=None, n=10):
    """Exposure deciles of the ranking score: actual and predicted loss cost per policy-year."""
    losses, pred, exposure = (np.asarray(a, float) for a in (losses, pred, exposure))
    score = pred / exposure if rank_by is None else np.asarray(rank_by, float)
    order = np.argsort(score, kind="mergesort")
    cum = np.cumsum(exposure[order]) / exposure.sum()
    bucket = np.minimum((cum * n - 1e-12).astype(int), n - 1) + 1
    t = pd.DataFrame({"decile": bucket, "exposure": exposure[order], "losses": losses[order],
                      "pred": pred[order]}).groupby("decile").sum()
    t["actual_loss_cost"] = t["losses"] / t["exposure"]
    t["pred_loss_cost"] = t["pred"] / t["exposure"]
    t["actual_relative"] = t["actual_loss_cost"] / (losses.sum() / exposure.sum())
    return t


def double_lift(losses, pred_a, pred_b, exposure, n=10):
    """Double-lift chart data: sort cells by the RATIO pred_a / pred_b (where the two models
    disagree most at the ends), cut into exposure deciles, and compare actual loss cost with
    each model's prediction (both re-based to the actual total). The model whose line tracks
    the actual bars is right where they disagree."""
    losses, pred_a, pred_b, exposure = (np.asarray(a, float) for a in (losses, pred_a, pred_b, exposure))
    pa, pb = pred_a * losses.sum() / pred_a.sum(), pred_b * losses.sum() / pred_b.sum()
    order = np.argsort(pa / pb, kind="mergesort")
    cum = np.cumsum(exposure[order]) / exposure.sum()
    bucket = np.minimum((cum * n - 1e-12).astype(int), n - 1) + 1
    t = pd.DataFrame({"decile": bucket, "exposure": exposure[order], "losses": losses[order],
                      "a": pa[order], "b": pb[order]}).groupby("decile").sum()
    avg = losses.sum() / exposure.sum()
    for c in ["losses", "a", "b"]:
        t[f"{c}_relative"] = t[c] / t["exposure"] / avg
    t["ratio_a_over_b"] = t["a"] / t["b"]
    return t


# ---------------------------------------------------------------------------
# Leave-one-year-out for the combined model, county factors and Tweedie
# ---------------------------------------------------------------------------
def loyo_combined(cells, claims, freq_features, sev_features, freq_bases=None, sev_bases=None,
                  tweedie_features=None, tweedie_power=None, county_methods=(), years=None,
                  sev_cv=None, keep_cols=()):
    """For each held-out year: refit frequency, severity (and Tweedie) on the other years,
    estimate county factors on the other years only, and predict the held-out year.
    Returns (scores: one row per fold x model, out-of-fold predictions for matched test cells)."""
    rows, oof = [], []
    years = sorted(cells["year"].unique()) if years is None else sorted(years)
    for yr in years:
        tr_c, te_c = cells[cells["year"].isin(years) & (cells["year"] != yr)], cells[cells["year"] == yr]
        tr_k, _ = claims[claims["year"].isin(years) & (claims["year"] != yr)], None
        m = fit_combined(tr_c, tr_k, freq_features, sev_features, freq_bases, sev_bases)
        test = te_c[te_c["exposure"] > 0]
        all_losses = float(te_c[LOSS].sum())
        train_rate = tr_c[LOSS].sum() / tr_c["exposure"].sum()
        preds = {"freq x sev": m.predict(test, county=False)}
        for meth in county_methods:
            mc, _, _ = with_county_factors(m, tr_c, tr_k, meth, sev_cv=sev_cv)
            preds[f"freq x sev + county ({meth})"] = mc.predict(test)
        if tweedie_power is not None:
            tw = fit_tweedie(tr_c, tweedie_features, tweedie_power)
            preds[f"Tweedie p={tweedie_power:g}"] = tw.predict(test)
        o = test[["year", "county", "exposure", LOSS, "claims", *keep_cols]].copy()
        for name, pr in preds.items():
            rows.append({"model": name, "test_year": yr, "cat_year": val.is_cat_year(yr),
                         **score_cells(test, pr, train_rate, all_test_losses=all_losses)})
            o[name] = pr
        # Ranking scores RELATIVE to each fold's base level, so pooling folds in a lift chart
        # isn't distorted by fold-to-fold level differences (e.g. the fold without 2016 has a
        # much lower average-year level, which would push all 2016 cells down the ranking).
        o["rank_freq_x_sev"] = m.rate(test, county=False) / m.base_pure_premium()
        if tweedie_power is not None:
            o["rank_tweedie"] = tw.relative_rate(test)
        oof.append(o)
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def county_ae_out_of_fold(oof, pred_col, min_claims=30, names=None):
    """Held-out A/E by county from LOYO predictions, re-based within each year (so the storm
    year's level is removed) and pooled over years. Returns the table and two summaries:
    claim-weighted mean |log A/E| and share of counties within 0.8-1.25 (min_claims or more)."""
    o = oof.copy()
    o["rb"] = o[pred_col] * o.groupby("year")[LOSS].transform("sum") / o.groupby("year")[pred_col].transform("sum")
    t = o.groupby("county").agg(claims=("claims", "sum"), actual=(LOSS, "sum"), expected=("rb", "sum"))
    t["ae"] = t["actual"] / t["expected"]
    if names is not None:
        t["name"] = t.index.map(names)
    big = t[t["claims"] >= min_claims]
    la = np.log(big["ae"].clip(lower=1e-3))
    summ = {"counties": len(big), "mean_abs_log_ae": float(np.average(la.abs(), weights=big["claims"])),
            "share_within_0.8_1.25": float(((big["ae"] >= 0.8) & (big["ae"] <= 1.25)).mean())}
    return t, summ


# ---------------------------------------------------------------------------
# Earned premium per rating cell (for loss ratios)
# ---------------------------------------------------------------------------
PREMIUM_COLS = ["totalInsurancePremiumOfThePolicy", "policyCost"]


def build_cell_premium(cells_path=CELLS_PATH, save_path=PREMIUM_PATH, verbose=True):
    """Earned premium per rating cell (nominal dollars), on the same keys as the cell table.

    Each policy term's premium is earned evenly over its term: a year earns
    premium x (days of cover in that year) / (term length). We reuse
    actuarial.earned_exposure with weight = premium x days-per-year / term days.
      totalInsurancePremiumOfThePolicy: the premium (before fees and surcharges)
      policyCost: what the policyholder paid, incl. the federal policy fee, HFIAA surcharge,
                  reserve fund assessment etc."""
    import actuarial as A
    import build_dataset as B

    pol = A.load_va_policies(extra_cols=[*B.FEATURE_SOURCE_COLS, B.POLICY_FLOORS_COL, *PREMIUM_COLS])
    feat, _ = B.make_features(pol, B.POLICY_FLOORS_COL)
    pol = pd.concat([pol[["id", "policyEffectiveDate", "policyTerminationDate",
                          "cancellationDateOfFloodPolicy", "county", *PREMIUM_COLS]], feat], axis=1)
    pol["county"] = pol["county"].fillna(B.UNKNOWN_COUNTY)
    term_days = (pol["policyTerminationDate"] - pol["policyEffectiveDate"]).dt.days
    for c in PREMIUM_COLS:     # per-policy-year rate of the premium, so earned_exposure earns it
        pol[f"w_{c}"] = np.where(term_days > 0, pol[c].fillna(0) * 365.25 / term_days.clip(lower=1), 0.0)
    keys = ["county", *B.CELL_FEATURES]
    out = A.earned_exposure(pol, B.YEARS, by=keys)
    for c, name in [("totalInsurancePremiumOfThePolicy", "earned_premium"), ("policyCost", "earned_policy_cost")]:
        e = A.earned_exposure(pol, B.YEARS, by=keys, weight=f"w_{c}").rename(columns={"exposure": name})
        out = out.merge(e, on=[*keys, "year"])
    out = out[out["exposure"] > 0]
    cells = pd.read_parquet(cells_path)
    j = cells[cells["exposure"] > 0].merge(out, on=B.CELL_KEYS, how="left", suffixes=("", "_p"))
    assert j["exposure_p"].notna().all(), "a cell with exposure has no premium row"
    assert np.allclose(j["exposure"], j["exposure_p"], atol=1e-6), "premium exposure differs from the cell table"
    res = j[[*B.CELL_KEYS, "earned_premium", "earned_policy_cost"]]
    if verbose:
        print(f"Earned premium {res['earned_premium'].sum():,.0f} and policy cost {res['earned_policy_cost'].sum():,.0f} "
              f"(nominal) over {len(res):,} cells; exposure reconciles with the cell table.")
    if save_path:
        res.to_parquet(save_path, index=False)
    return res


def load_cell_premium(path=PREMIUM_PATH, rebuild=False):
    if rebuild or not Path(path).exists():
        return build_cell_premium(save_path=path)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_combined(model, path, county_table=None, extra=None):
    """JSON the app reads: references to the two GLM files, the loads, the off-balance, the base
    loss cost, and the county factors (with their credibility Z)."""
    d = {"frequency_model": "models/frequency_glm.json", "severity_model": "models/severity_glm.json",
         "large_loss_load": model.large_loss_load, "icc_bridge": model.icc_bridge,
         "off_balance": model.off_balance, "base_pure_premium_average_year": model.base_pure_premium(),
         "county_factors": {} if model.county_factors is None else
         {str(k): float(v) for k, v in model.county_factors.items()}}
    if county_table is not None:
        d["county_table"] = json.loads(county_table.reset_index().to_json(orient="records"))
    d.update(extra or {})
    Path(path).write_text(json.dumps(d, indent=2, default=float), encoding="utf-8")


def load_combined(path, freq_path=FREQ_JSON, sev_path=SEV_JSON):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    freq, sev, _ = load_saved_models(freq_path, sev_path)
    cf = pd.Series(d["county_factors"], dtype=float) if d["county_factors"] else None
    return CombinedModel(freq, sev, d["icc_bridge"], d["off_balance"], cf)


# ---------------------------------------------------------------------------
# Self-test on simulated data with known answers
# ---------------------------------------------------------------------------
def _simulate(seed=3):
    rng = np.random.default_rng(seed)
    zone_f, occ_s = {"A": 1.0, "X": 0.5}, {"single": 1.0, "multi": 2.0}
    county_f = {f"c{i}": v for i, v in enumerate([1.0] * 6 + [1.6, 0.6])}
    years = [2010, 2011, 2012, 2013]
    cells, claims = [], []
    for yr in years:
        for z in zone_f:
            for o in occ_s:
                for c, cfac in county_f.items():
                    e = rng.uniform(300, 800)
                    n = rng.poisson(e * 0.02 * zone_f[z] * cfac)
                    amt = rng.gamma(1.0, 20_000 * occ_s[o], n)
                    icc = np.where(rng.random(n) < 0.02, 5_000.0, 0.0)
                    cells.append({"year": yr, "county": c, "county_group": c, "zone": z, "occ": o,
                                  "exposure": e, "claims": n, LOSS: float((amt + icc).sum()), "exposure_matched": True})
                    for a, ic in zip(amt, icc):
                        claims.append({"year": yr, "county": c, "county_group": c, "zone": z, "occ": o,
                                       sv.TARGET: a, "total_2025": a + ic, sv.WEIGHT: 1.0})
    return pd.DataFrame(cells), pd.DataFrame(claims)


def _self_test():
    cells, claims = _simulate()
    m = fit_combined(cells, claims, ["zone"], ["occ"], freq_bases={"zone": "A"}, sev_bases={"occ": "single"})
    # Off-balance: predictions add up to actual losses.
    assert np.isclose(m.predict(cells, county=False).sum(), cells[LOSS].sum(), rtol=1e-9)
    # Combined relativities multiply; the frequency relativity equals the data's own ratio
    # (one feature, balanced design) and is near the true 0.5 (sampling noise ~7%).
    rt = combined_relativity_table(m.freq, m.sev).set_index("level")
    assert np.allclose(rt["pure_premium_rel"], rt["frequency_rel"] * rt["severity_rel"])
    raw = cells.groupby("zone")["claims"].sum() / cells.groupby("zone")["exposure"].sum()
    assert abs(rt.loc["X", "frequency_rel"] / (raw["X"] / raw["A"]) - 1) < 0.02
    assert abs(rt.loc["X", "frequency_rel"] / 0.5 - 1) < 0.2
    assert abs(rt.loc["multi", "severity_rel"] / 2.0 - 1) < 0.15

    # Limited fluctuation: 1,082 claims standard; Z capped at 1; square-root rule.
    assert round(full_credibility_standard()) == 1082
    assert np.allclose(limited_fluctuation_z([0, 270.6, 1082.4, 5000], 1082.4), [0, 0.5, 1, 1], atol=1e-3)

    # Buhlmann-Straub recovers a known k on simulated groups: true means differ with variance a,
    # each period's ratio has variance s2 / w.
    rng = np.random.default_rng(7)
    a, s2 = 0.04, 4.0
    mu = 1 + rng.normal(0, np.sqrt(a), 400)
    g = np.repeat(np.arange(400), 10)
    w = rng.uniform(5, 50, len(g))
    x = mu[g] + rng.normal(0, np.sqrt(s2 / w))
    par, tab = buhlmann_straub(x, w, g)
    assert abs(par["k"] / (s2 / a) - 1) < 0.25, par
    assert ((tab["Z"] > 0) & (tab["Z"] < 1)).all()

    # County factors: the high county gets a factor > 1, the low one < 1, and re-centring keeps
    # the statewide total.
    for meth in ["limited_fluctuation", "buhlmann_straub"]:
        mc, tbl, _ = with_county_factors(m, cells, claims, meth, sev_cv=1.0)
        assert np.isclose(mc.predict(cells).sum(), m.predict(cells, county=False).sum(), rtol=1e-9)
        assert tbl.loc["c6", "factor"] > 1.1 and tbl.loc["c7", "factor"] < 0.9, (meth, tbl)

    # Tweedie recovers the pure-premium relativities (zone 0.5 on frequency, occ 2 on severity).
    tw = fit_tweedie(cells, ["zone", "occ"], 1.5)
    r = tw.relativities()
    zx = r["zone[X]"] if "zone[X]" in r.index else 1 / r["zone[A]"]
    om = r["occ[multi]"] if "occ[multi]" in r.index else 1 / r["occ[single]"]
    assert abs(zx / rt.loc["X", "frequency_rel"] - 1) < 0.1 and abs(om / 2.0 - 1) < 0.15, r
    assert np.isclose(tw.predict(cells).sum(), cells[LOSS].sum(), rtol=1e-9)
    assert abs(tweedie_power_from_gamma_shape(1.0) - 1.5) < 1e-12

    # Scores: deviance(y, y) = 0; double-lift totals equal actual; lift has 10 deciles.
    y = cells[LOSS].to_numpy() / cells["exposure"].to_numpy()
    assert tweedie_deviance(y + 1, y + 1, 1.5) < 1e-6
    dl = double_lift(cells[LOSS], m.predict(cells), tw.predict(cells), cells["exposure"])
    assert np.isclose(dl["a"].sum(), cells[LOSS].sum()) and np.isclose(dl["b"].sum(), cells[LOSS].sum())
    lt = lift_table(cells[LOSS], m.predict(cells), cells["exposure"])
    assert len(lt) == 10 and np.isclose(lt["exposure"].sum(), cells["exposure"].sum())

    # LOYO runs and tests every matched cell once.
    sc, oof = loyo_combined(cells, claims, ["zone"], ["occ"], tweedie_features=["zone", "occ"],
                            tweedie_power=1.5, county_methods=("limited_fluctuation",), sev_cv=1.0)
    assert len(oof) == len(cells) and sc["model"].nunique() == 3

    # JSON round trip.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mc, tbl, _ = with_county_factors(m, cells, claims, "limited_fluctuation", sev_cv=1.0)
        fg.save_model(m.freq, Path(tmp) / "f")
        sv.save_model(m.sev, Path(tmp) / "s")
        save_combined(mc, Path(tmp) / "pp.json", tbl)
        m2 = load_combined(Path(tmp) / "pp.json", Path(tmp) / "f.json", Path(tmp) / "s.json")
        assert np.allclose(m2.predict(cells), mc.predict(cells))
    return True


if __name__ == "__main__":
    print("pure_premium self-test passed:", _self_test())
