"""
One-way analysis: how frequency, severity and pure premium vary across the levels of
ONE feature at a time (the classic first step of an actuarial rating review).

Plain-English definitions (see LEARNING_NOTES.md, Task 5):
  * Level        one category of a feature, e.g. "A/V high risk" for flood zone.
  * Base level   the reference level everything is compared with. By convention the
                 level with the most exposure (its estimate is the most reliable).
  * Relativity   a level's measure divided by the base level's: 1.30 = "30% higher".
  * Credibility  how much we can trust a level's own numbers. Classic actuarial rule
                 ("limited fluctuation"): 1,082 claims give 90% confidence that the
                 observed frequency is within 5% of the truth. Fewer claims -> partial
                 credibility Z = sqrt(claims / 1082).
  * Confidence interval (CI)  a range that, with 95% confidence, contains the true value.

All tables use every row of va_model_data.parquet: the 1% of claims with no matching
exposure (exposure_matched = False) add claims but no exposure. Using them everywhere keeps
frequency x severity = pure premium exactly. The GLMs (week 3+) will drop them for frequency.
Dollars are 2025 dollars (losses_2025).

Run the self-test:  .venv\\Scripts\\python.exe src\\one_way.py
"""

import numpy as np
import pandas as pd
from scipy import stats

FULL_CREDIBILITY_CLAIMS = 1082     # limited-fluctuation standard: P = 90%, k = 5%
LOW_EXPOSURE_SHARE = 0.02          # flag levels with < 2% of the feature's exposure ...
LOW_CLAIMS = 30                    # ... or fewer than 30 paid claims
Z95 = stats.norm.ppf(0.975)        # 1.96

SUMS = ["exposure", "claims", "losses_2025"]


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------
def poisson_ci(claims, exposure, level=0.95):
    """Exact ("Garwood") confidence interval for a Poisson rate = claims / exposure.

    Uses the link between the Poisson and chi-square distributions, so it works even for
    0 or 1 claims, where the usual "estimate +/- 1.96 x standard error" breaks down."""
    n = np.asarray(claims, float)
    e = np.asarray(exposure, float)
    a = 1 - level
    lo = np.where(n > 0, stats.chi2.ppf(a / 2, 2 * n) / 2, 0.0) / e
    hi = stats.chi2.ppf(1 - a / 2, 2 * n + 2) / 2 / e
    return lo, hi


def _measures(t):
    """Frequency, severity, pure premium on a table of sums."""
    t = t.copy()
    t["frequency"] = t["claims"] / t["exposure"]
    t["severity"] = t["losses_2025"] / t["claims"].where(t["claims"] > 0)
    t["pure_premium"] = t["losses_2025"] / t["exposure"]
    return t


def poisson_dispersion(df, feature, with_year):
    """Over-dispersion factor phi for claim counts, measured on the year x level table.

    phi = Pearson chi-square / degrees of freedom. phi = 1 means counts vary as much as a
    Poisson model expects; phi = 20 means 20 times more variance (storms hit many policies
    at once, so claims arrive in clumps). Confidence intervals are widened by sqrt(phi).

    with_year=False: expected = exposure x level frequency (all years alike). The
        year-to-year swings count as noise, which is right for a level's long-run frequency.
    with_year=True: expected = exposure x level effect x year effect (fitted by a small
        Poisson GLM). Storm years lift all levels together, and that common part cancels in
        a relativity, so this smaller phi is the one for relativities."""
    import statsmodels.api as sm
    t = df.groupby([feature, "year"], observed=True)[["exposure", "claims"]].sum().reset_index()
    t = t[t["exposure"] > 0]
    if with_year:
        X = pd.get_dummies(t[[feature, "year"]].astype(str), drop_first=True, dtype=float)
        X = sm.add_constant(X, has_constant="add")
        fit = sm.GLM(t["claims"], X, family=sm.families.Poisson(),
                     offset=np.log(t["exposure"])).fit()
        mu, n_par = fit.fittedvalues, X.shape[1]
    else:
        f = t.groupby(feature, observed=True)[["claims", "exposure"]].transform("sum")
        mu, n_par = t["exposure"] * f["claims"] / f["exposure"], t[feature].nunique()
    dof = len(t) - n_par
    return float(((t["claims"] - mu) ** 2 / mu).sum() / dof) if dof > 0 else 1.0


def gamma_dispersion(df, feature):
    """Dispersion of average claim size (Gamma GLM sense) around each level's severity.

    Each cell with paid claims contributes its average claim, weighted by its claim count.
    phi is roughly (coefficient of variation of one claim)^2: claim sizes are very spread out."""
    c = df[df["claims"] > 0]
    y = c["losses_2025"] / c["claims"]
    lvl = c.groupby(feature, observed=True)[["losses_2025", "claims"]].transform("sum")
    mu = lvl["losses_2025"] / lvl["claims"]
    dof = len(c) - c[feature].nunique()
    return float((c["claims"] * (y - mu) ** 2 / mu ** 2).sum() / dof)


# ---------------------------------------------------------------------------
# The one-way table
# ---------------------------------------------------------------------------
def one_way(df, feature, base=None, phi_freq=None, phi_rel=None, phi_sev=None):
    """One row per level of `feature`: exposure, claims, losses, frequency (with 95% CIs),
    severity, pure premium, their relativities to `base`, and credibility flags.

    base: the reference level (default: the level with the most exposure).
    phi_*: over-dispersion factors (computed from df if not given; see poisson_dispersion
           and gamma_dispersion). Pass phi_*=1 to get plain Poisson / no widening.

    Columns of CIs:
      freq_lo/freq_hi        exact Poisson 95% CI (assumes claims are independent)
      freq_lo_od/freq_hi_od  widened for over-dispersion (more honest for flood)
      freq_rel_lo/_hi        95% CI for the frequency relativity (log scale, widened)
      sev_rel_lo/_hi         95% CI for the severity relativity (Gamma, widened)"""
    t = _measures(df.groupby(feature, observed=True)[SUMS].sum())
    if base is None:
        base = t["exposure"].idxmax()
    phi_freq = poisson_dispersion(df, feature, with_year=False) if phi_freq is None else phi_freq
    phi_rel = poisson_dispersion(df, feature, with_year=True) if phi_rel is None else phi_rel
    phi_sev = gamma_dispersion(df, feature) if phi_sev is None else phi_sev

    t["exposure_share"] = t["exposure"] / t["exposure"].sum()
    t["freq_lo"], t["freq_hi"] = poisson_ci(t["claims"], t["exposure"])
    se_log = np.sqrt(phi_freq / t["claims"].clip(lower=0.5))
    t["freq_lo_od"] = t["frequency"] * np.exp(-Z95 * se_log)
    t["freq_hi_od"] = t["frequency"] * np.exp(Z95 * se_log)

    b = t.loc[base]
    for m in ["frequency", "severity", "pure_premium"]:
        t[f"{m}_rel"] = t[m] / b[m]
    # Relativity CI on the log scale: Var(log ratio) = Var(log level) + Var(log base).
    # The base level is the reference, so its own relativity is exactly 1 (no interval).
    n, nb = t["claims"].clip(lower=0.5), b["claims"]
    se_f = np.sqrt(phi_rel * (1 / n + 1 / nb))
    se_s = np.sqrt(phi_sev * (1 / n + 1 / nb))
    for m, se in [("frequency", se_f), ("severity", se_s)]:
        short = "freq" if m == "frequency" else "sev"
        t[f"{short}_rel_lo"] = t[f"{m}_rel"] * np.exp(-Z95 * se)
        t[f"{short}_rel_hi"] = t[f"{m}_rel"] * np.exp(Z95 * se)
        t.loc[base, [f"{short}_rel_lo", f"{short}_rel_hi"]] = 1.0

    t["credibility_Z"] = np.minimum(1.0, np.sqrt(t["claims"] / FULL_CREDIBILITY_CLAIMS))
    t["low_credibility"] = (t["exposure_share"] < LOW_EXPOSURE_SHARE) | (t["claims"] < LOW_CLAIMS)
    t["is_base"] = t.index == base
    t.attrs.update(feature=feature, base=base, phi_freq=phi_freq, phi_rel=phi_rel, phi_sev=phi_sev)
    return t


def importance(df, feature):
    """How much a feature explains, on its own, for frequency and for severity.

    Uses deviance (a goodness-of-fit score for Poisson / Gamma data; lower = better fit).
    A one-way model ("each level has its own frequency") vs a null model ("one frequency for
    everyone"). The drop in deviance, per extra parameter, divided by the over-dispersion
    phi, is a quasi-likelihood F statistic: about 1 = "no better than noise", large = matters.

    For a one-way Poisson model the drop has a closed form: 2 * sum_l n_l * log(f_l / f_all)."""
    t = _measures(df.groupby(feature, observed=True)[SUMS].sum())
    t = t[t["exposure"] > 0]
    L = len(t)
    f_all = t["claims"].sum() / t["exposure"].sum()
    n = t["claims"]
    dev_f = 2 * float((n[n > 0] * np.log(t.loc[n > 0, "frequency"] / f_all)).sum())
    phi_f = poisson_dispersion(df, feature, with_year=True)

    # Gamma severity deviance, cell level, weights = claims.
    c = df[df["claims"] > 0]
    y = c["losses_2025"] / c["claims"]
    w = c["claims"]
    lvl = c.groupby(feature, observed=True)[["losses_2025", "claims"]].transform("sum")
    mu1 = lvl["losses_2025"] / lvl["claims"]
    mu0 = c["losses_2025"].sum() / w.sum()
    gdev = lambda mu: 2 * float((w * (-np.log(y / mu) + (y - mu) / mu)).sum())
    dev_s = gdev(mu0) - gdev(mu1)
    phi_s = gamma_dispersion(df, feature)
    return {"feature": feature, "levels": L,
            "freq_dev_drop": dev_f, "freq_F": dev_f / (L - 1) / phi_f, "phi_freq_rel": phi_f,
            "sev_dev_drop": dev_s, "sev_F": dev_s / (L - 1) / phi_s, "phi_sev": phi_s}


# ---------------------------------------------------------------------------
# Stability, confounding, association
# ---------------------------------------------------------------------------
def by_year(df, feature, base=None):
    """Frequency/severity relativities for each year separately (long table).
    The base level is fixed across years (the all-years highest-exposure level)."""
    if base is None:
        base = df.groupby(feature, observed=True)["exposure"].sum().idxmax()
    out = []
    for y, d in df.groupby("year"):
        t = _measures(d.groupby(feature, observed=True)[SUMS].sum())
        b = t.loc[base]
        t["frequency_rel"] = t["frequency"] / b["frequency"]
        t["severity_rel"] = t["severity"] / b["severity"]
        t["pure_premium_rel"] = t["pure_premium"] / b["pure_premium"]
        out.append(t.assign(year=y).reset_index())
    return pd.concat(out, ignore_index=True)


def exclude_years(df, years):
    """The table without some years (e.g. the catastrophe years)."""
    return df[~df["year"].isin(list(years))]


def stratified_relativity(df, feature, by="flood_zone_group", base=None):
    """Frequency relativity of each level vs base, comparing like with like within each
    stratum of `by` (a Mantel-Haenszel rate ratio).

    Idea: inside each flood zone separately, compare the level's claim rate with the base
    level's, then pool the zones with weights. If the one-way relativity was partly caused by
    the feature being concentrated in one zone (confounding), this "zone-adjusted" number
    removes that part. by can be a column name or a list of names.

    Formula, for a level (1) vs base (0) over strata s with claims a, b and exposures E1, E0:
        RR_MH = sum_s(a_s * E0_s / E_s) / sum_s(b_s * E1_s / E_s),   E_s = E0_s + E1_s"""
    by = [by] if isinstance(by, str) else list(by)
    by = [b for b in by if b != feature]           # adjusting a feature for itself is meaningless
    if base is None:
        base = df.groupby(feature, observed=True)["exposure"].sum().idxmax()
    if not by:                                     # nothing left to adjust for: plain one-way
        t = df.groupby(feature, observed=True)[["exposure", "claims"]].sum()
        f = t["claims"] / t["exposure"]
        return (f / f[base]).rename(f"freq_rel_adj_for_{feature}")
    t = df.groupby([*by, feature], observed=True)[["exposure", "claims"]].sum()
    out = {}
    for lvl in df[feature].unique():
        if lvl == base:
            out[lvl] = 1.0
            continue
        num = den = 0.0
        for _, g in t.groupby(level=by):
            g = g.droplevel(by)
            if lvl not in g.index or base not in g.index:
                continue
            a, e1 = g.loc[lvl, "claims"], g.loc[lvl, "exposure"]
            b, e0 = g.loc[base, "claims"], g.loc[base, "exposure"]
            num += a * e0 / (e0 + e1)
            den += b * e1 / (e0 + e1)
        out[lvl] = num / den if den > 0 else np.nan
    return pd.Series(out, name=f"freq_rel_adj_for_{'+'.join(by)}")


def cramers_v(df, a, b, weight="exposure"):
    """Strength of association between two categorical features, from 0 (unrelated)
    to 1 (one completely determines the other), weighted by exposure.

    Cramer's V = sqrt(chi-square / (N * (min(rows, cols) - 1))). It's the categorical
    cousin of a correlation coefficient. Because both chi-square and N scale with the total
    weight, V doesn't depend on whether we count policy-years or shares."""
    tab = df.pivot_table(index=a, columns=b, values=weight, aggfunc="sum", fill_value=0, observed=True)
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    N = tab.to_numpy().sum()
    expected = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / N
    chi2 = float(((tab.to_numpy() - expected) ** 2 / expected).sum())
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (N * k))) if k > 0 else 0.0


# ---------------------------------------------------------------------------
# Self-test on made-up data with known answers
# ---------------------------------------------------------------------------
def _self_test():
    # Two zones, a feature "x" that sits mostly in the risky zone. Inside each zone x has
    # NO effect, so the true zone-adjusted relativity is exactly 1, but the one-way
    # relativity is > 1 (confounding).
    rows = []
    for zone, rate, sev in [("high", 0.02, 30_000.0), ("low", 0.005, 10_000.0)]:
        for x, expo in [("yes", 900 if zone == "high" else 100), ("no", 100 if zone == "high" else 900)]:
            for year in [2020, 2021]:
                e = expo / 2
                n = rate * e
                rows.append({"zone": zone, "x": x, "year": year, "exposure": e,
                             "claims": n, "losses_2025": n * sev})
    d = pd.DataFrame(rows)

    t = one_way(d, "x", base="no", phi_freq=1, phi_rel=1, phi_sev=1)
    # one-way: yes = (0.02*900 + 0.005*100)/1000 = 0.0185, no = (0.02*100+0.005*900)/1000 = 0.0065
    assert abs(t.loc["yes", "frequency"] - 0.0185) < 1e-12
    assert abs(t.loc["yes", "frequency_rel"] - 0.0185 / 0.0065) < 1e-12
    assert t.loc["no", "frequency_rel"] == 1 and t.loc["no", "freq_rel_lo"] == 1
    # frequency x severity = pure premium
    assert np.allclose(t["frequency"] * t["severity"], t["pure_premium"])
    adj = stratified_relativity(d, "x", by="zone", base="no")
    assert abs(adj["yes"] - 1.0) < 1e-12, adj           # confounding removed
    # Exact Poisson CI: 0 claims -> lower 0; 10 claims in 1000 -> known values
    lo, hi = poisson_ci([0, 10], [1000, 1000])
    assert lo[0] == 0 and abs(hi[0] - 3.6889 / 1000) < 1e-6
    assert abs(lo[1] - 4.7954 / 1000) < 1e-6 and abs(hi[1] - 18.3904 / 1000) < 1e-6
    # Cramer's V: x is strongly tied to zone (0.8 here), and a feature with itself gives 1
    assert abs(cramers_v(d, "zone", "x") - 0.8) < 1e-12
    assert abs(cramers_v(d.assign(zone2=d["zone"]), "zone", "zone2") - 1.0) < 1e-12
    # Credibility: 1082 claims -> Z = 1; 270.5 claims -> Z = 0.5
    t2 = d.assign(claims=d["claims"] * 0 + 270.5)
    assert abs(one_way(t2, "x", phi_freq=1, phi_rel=1, phi_sev=1)["credibility_Z"].iloc[0] - 1) < 1e-12
    return True


if __name__ == "__main__":
    print("one_way self-test passed:", _self_test())
