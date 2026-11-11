"""
Train/test splits for checking models on years they have NOT seen (weeks 3-6).

Why split by YEAR, never by random row? Flood claims come in storms: one hurricane makes
claims in thousands of cells in the same year. If random rows from 2016 sit in both the
training and the test data, the model has already "seen" Hurricane Matthew, and the test
score is too optimistic. Holding out whole years means the test really is new weather.

The design is explained in docs/validation_plan.md. In short:
  * MAIN comparison: leave-one-year-out (LOYO) cross-validation. 16 folds; in each one,
    the model is trained on 15 years and scored on the 1 year it never saw. Every metric is
    reported per fold, with the catastrophe folds flagged and summaries with/without them.
  * SECONDARY check: a forward-in-time holdout, train 2010-2020 / test 2021-2025 (scored
    once, at the end), because pricing is always about future years.
  * Model development (feature choice, XGBoost tuning) uses LOYO on the training years
    2010-2020 only, so 2021-2025 stays untouched until the final check.

Run the self-test:  .venv\\Scripts\\python.exe src\\validation.py
"""

import pandas as pd

ALL_YEARS = tuple(range(2010, 2026))           # 2010..2025, the modelling years
LAST_TRAIN_YEAR = 2020                         # forward holdout: train <= 2020, test >= 2021
DEV_YEARS = tuple(y for y in ALL_YEARS if y <= LAST_TRAIN_YEAR)       # 2010..2020
HOLDOUT_YEARS = tuple(y for y in ALL_YEARS if y > LAST_TRAIN_YEAR)    # 2021..2025

# The three worst years by paid losses in 2025 dollars (checked by cat_years() below):
# 2016 Hurricane Matthew ($104M), 2011 Hurricane Irene ($60M), 2012 Hurricane Sandy ($35M).
# Together they are 68% of 2010-2025 losses.
CAT_YEARS = (2011, 2012, 2016)


def cat_years(df, n=3, loss_col="losses_2025"):
    """The n years with the largest losses (sorted). Used to confirm CAT_YEARS."""
    by_year = df.groupby("year")[loss_col].sum()
    return tuple(sorted(by_year.nlargest(n).index.tolist()))


def is_cat_year(year):
    return year in CAT_YEARS


def time_holdout(df, last_train_year=LAST_TRAIN_YEAR):
    """Forward-in-time split: (train = years <= last_train_year, test = later years)."""
    train = df[df["year"] <= last_train_year]
    test = df[df["year"] > last_train_year]
    return train, test


def leave_one_year_out(df, years=None):
    """Yield (held_out_year, train, test) for each year in `years` (default: all years in df).

    Only years in `years` are used at all. E.g. years=DEV_YEARS gives 11 folds inside
    2010-2020 and never touches 2021-2025."""
    years = sorted(df["year"].unique()) if years is None else sorted(years)
    pool = df[df["year"].isin(years)]
    for y in years:
        yield y, pool[pool["year"] != y], pool[pool["year"] == y]


def forward_chaining(df, first_test_year=2016, last_year=None):
    """Rolling-origin ("expanding window") folds: train on every year BEFORE y, test on y.
    Never uses the future, but early folds have little training data. Kept as an
    alternative for comparison (see docs/validation_plan.md)."""
    last_year = int(df["year"].max()) if last_year is None else last_year
    for y in range(first_test_year, last_year + 1):
        yield y, df[df["year"] < y], df[df["year"] == y]


def fold_summary(folds):
    """One row per fold: test year, cat flag, training and test exposure/claims/losses.
    `folds` is any iterable of (year, train, test), e.g. leave_one_year_out(df)."""
    rows = []
    for y, train, test in folds:
        rows.append({"test_year": y, "cat_year": is_cat_year(y),
                     "train_years": train["year"].nunique(),
                     "train_exposure": train["exposure"].sum(), "train_claims": int(train["claims"].sum()),
                     "test_exposure": test["exposure"].sum(), "test_claims": int(test["claims"].sum()),
                     "test_losses_2025": test["losses_2025"].sum()})
    return pd.DataFrame(rows)


def summarize_scores(scores, metric, year_col="test_year"):
    """Summarize a per-fold metric the way the validation plan asks: mean and median over all
    folds (each year counts once), without the cat years, and for the cat years only.
    `scores` is a DataFrame with one row per fold (and per model, if a 'model' column exists)."""
    s = scores.assign(cat=scores[year_col].isin(CAT_YEARS))
    keys = ["model"] if "model" in s.columns else []
    parts = {"all folds (mean)": s, "non-cat folds (mean)": s[~s["cat"]], "cat folds (mean)": s[s["cat"]]}
    out = {name: (p.groupby(keys)[metric].mean() if keys else pd.Series({metric: p[metric].mean()}))
           for name, p in parts.items()}
    out["all folds (median)"] = s.groupby(keys)[metric].median() if keys else pd.Series({metric: s[metric].median()})
    return pd.DataFrame(out)


def _self_test():
    d = pd.DataFrame({"year": [y for y in ALL_YEARS for _ in range(3)]})
    d["exposure"], d["claims"] = 1.0, 1
    d["losses_2025"] = d["year"].map(lambda y: 100.0 if y in CAT_YEARS else 1.0)
    assert cat_years(d) == CAT_YEARS

    train, test = time_holdout(d)
    assert set(train["year"]) == set(DEV_YEARS) and set(test["year"]) == set(HOLDOUT_YEARS)
    assert len(train) + len(test) == len(d)

    folds = list(leave_one_year_out(d))
    assert len(folds) == 16
    for y, tr, te in folds:
        assert set(te["year"]) == {y} and y not in set(tr["year"])        # no leakage of year y
        assert len(tr) + len(te) == len(d)                               # every row used once
    assert sorted(i for _, _, te in folds for i in te.index) == sorted(d.index)   # each row tested once

    dev = list(leave_one_year_out(d, DEV_YEARS))
    assert len(dev) == 11 and all(tr["year"].max() <= LAST_TRAIN_YEAR for _, tr, _ in dev)

    for y, tr, te in forward_chaining(d, 2016):
        assert tr["year"].max() < y and set(te["year"]) == {y}

    fs = fold_summary(leave_one_year_out(d))
    assert fs["cat_year"].sum() == 3 and (fs["test_claims"] == 3).all()

    sc = pd.DataFrame({"test_year": list(ALL_YEARS), "score": [10.0 if y in CAT_YEARS else 1.0 for y in ALL_YEARS]})
    summ = summarize_scores(sc, "score")
    assert summ.loc["score", "non-cat folds (mean)"] == 1.0 and summ.loc["score", "cat folds (mean)"] == 10.0
    return True


if __name__ == "__main__":
    print("validation self-test passed:", _self_test())
