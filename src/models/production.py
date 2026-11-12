"""
PRODUCTION MODEL: the one model the Streamlit app loads (chosen in Milestone 11).

Everything the app needs is in ONE small, readable JSON file, models/production_model.json:
the frequency GLM and severity GLM (specs + coefficients), the loads (large-loss, ICC bridge,
off-balance), the Buhlmann-Straub county factors, and a "model card" (what it is, what data,
how it was validated, and the area-level disclaimer). No pickles, no tree files.

    import production
    m = production.load()
    m.predict(cells)          # per-row expected frequency, severity and loss cost per policy-year
    m.explain(row)            # factor-by-factor worksheet for one group of buildings

Area level only: a prediction describes the AVERAGE insured building with these
characteristics in this county, in an average year. It is not a quote for any property.

Run the self-test:  .venv\\Scripts\\python.exe src\\models\\production.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parent
SRC = MODELS.parent
for p in (SRC, MODELS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import frequency_glm as fg                    # noqa: E402
import severity_glm as sv                     # noqa: E402
import pure_premium as pp                     # noqa: E402

ROOT = SRC.parent
PATH = ROOT / "models" / "production_model.json"
VA_PATH = ROOT / "models" / "production_model_va.json"     # the Virginia-only model (Milestone 11)
DISCLAIMER = ("Area-level estimate from aggregated public NFIP data (Virginia, 2010-2025). It describes the "
              "average insured building with these characteristics in this county in an average year. It is "
              "NOT a quote, a premium, or a prediction for any individual property.")
DISCLAIMER_3 = ("Area-level estimate from aggregated public NFIP data (Virginia, North Carolina and Maryland, "
                "2010-2025). It describes the average insured building with these characteristics in this "
                "county in an average year. It is NOT a quote, a premium, or a prediction for any individual "
                "property.")
# First two digits of a county FIPS code -> state postal code (the states this project models).
FIPS_STATE = {"51": "VA", "37": "NC", "24": "MD"}


def state_of_county(county):
    """State postal code for each county code: "37183" -> "NC", "UNKNOWN-MD" -> "MD"."""
    c = pd.Series(county).astype(str)
    st = c.str[:2].map(FIPS_STATE)
    unk = c.str.extract(r"^UNKNOWN-([A-Z]{2})$")[0]
    return st.fillna(unk)


def save(model, card, county_table=None, path=PATH, disclaimer=DISCLAIMER):
    """Write the production JSON from a pure_premium.CombinedModel of GLMs."""
    d = {"model_card": card, "disclaimer": disclaimer,
         "frequency_model": model.freq.to_json_dict(), "severity_model": model.sev.to_json_dict(),
         "large_loss_load": model.large_loss_load, "icc_bridge": model.icc_bridge,
         "off_balance": model.off_balance, "off_balance_by_state": model.off_balance_by,
         "base_pure_premium_average_year": model.base_pure_premium(),
         "county_factors": {} if model.county_factors is None else
         {str(k): float(v) for k, v in model.county_factors.items()}}
    if county_table is not None:
        d["county_table"] = json.loads(county_table.reset_index().to_json(orient="records"))
    Path(path).write_text(json.dumps(d, indent=2, default=float), encoding="utf-8")


class ProductionModel:
    def __init__(self, d):
        self.raw = d
        f, s = d["frequency_model"], d["severity_model"]
        self.freq = fg.FreqGLM(fg.Spec(**f["spec"]), pd.Series(f["params"]), f["family"], f["alpha"],
                               f["avg_year_log"])
        self.sev = sv.SevGLM(fg.Spec(**s["spec"]), pd.Series(s["params"]), s["phi"], s["avg_year_log"],
                             s["cap"], s["load"])
        cf = pd.Series(d["county_factors"], dtype=float) if d["county_factors"] else None
        self.combined = pp.CombinedModel(self.freq, self.sev, d["icc_bridge"], d["off_balance"], cf,
                                         d.get("off_balance_by_state"))
        self.card, self.disclaimer = d["model_card"], d["disclaimer"]
        self.county_table = pd.DataFrame(d.get("county_table", []))

    @property
    def by_state(self):
        """True for the multi-state model (it has a state factor)."""
        return "state" in self.freq.spec.features or "state" in self.sev.spec.features

    @property
    def features(self):
        """Columns the input table needs (plus 'county' as a 5-digit FIPS string). 'state' is
        not needed: it is derived from the county code when missing."""
        return sorted((set(self.freq.spec.features) | set(self.sev.spec.features)) - {"state"})

    def _with_state(self, df):
        if "state" in df and df["state"].notna().all():
            return df
        st = state_of_county(df["county"]).to_numpy()
        if self.by_state and pd.isna(st).any():
            raise ValueError("county codes must be FIPS codes from VA (51), NC (37) or MD (24)")
        return df.assign(state=st)

    def predict(self, cells):
        """Per row: frequency (claims per policy-year), severity ($ per claim, 2025 dollars, incl.
        large-loss load), county factor, and pure_premium ($ per policy-year, average year)."""
        need = set(self.features) | {"county"}
        missing = need - set(cells.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        # The GLM design matrix reads a 'year' column, but the average-year prediction zeroes the
        # year terms, so any value gives the same answer; supply one if the caller has none.
        df = self._with_state(cells.assign(exposure=1.0))
        for col, spec in (("year", self.freq.spec), (self.freq.spec.year_col, self.freq.spec)):
            if col not in df:
                df = df.assign(**{col: spec.bases.get("year", "2025")})
        c = self.combined.components(df)
        return c[["frequency", "severity", "county_factor", "pure_premium"]]

    def _steps(self, model, row):
        """(label, multiplier) for each non-base term of one GLM, for one row."""
        spec, out = model.spec, []
        for f in spec.features:
            lvl = str(row[f])
            col = f"{f}[{lvl}]"
            out.append((f"{f} = {lvl}", float(np.exp(model.params[col])) if col in model.params else 1.0))
        for a, b in spec.interactions:
            col = f"{a}[{row[a]}]:{b}[{row[b]}]"
            if col in model.params:
                out.append((f"{a} = {row[a]} AND {b} = {row[b]} (interaction)", float(np.exp(model.params[col]))))
        return out

    def explain(self, row):
        """A rating worksheet for one row (dict or Series): every multiplier, in order."""
        row = dict(row)
        if self.by_state and not row.get("state"):
            row["state"] = state_of_county([row.get("county", "")]).iloc[0]
        base_f = float(np.exp(self.freq.params["const"] + self.freq.avg_year_log))
        base_s = float(np.exp(self.sev.params["const"] + self.sev.avg_year_log))
        lines = [("frequency", "base claims per policy-year (average year)", base_f)]
        lines += [("frequency", l, m) for l, m in self._steps(self.freq, row)]
        lines += [("severity", "base claim size, 2025 $ (average year)", base_s)]
        lines += [("severity", l, m) for l, m in self._steps(self.sev, row)]
        one = pd.DataFrame([{"county": str(row.get("county", "")), "state": row.get("state")}])
        cf = float(self.combined.county_factor(one)[0])
        ob = float(np.atleast_1d(self.combined.off_balance_rows(one))[0])
        lines += [("loads", "large-loss load", self.combined.large_loss_load),
                  ("loads", "ICC bridge", self.combined.icc_bridge),
                  ("loads", "off-balance" + (f" ({row['state']})" if self.combined.off_balance_by else ""), ob),
                  ("county", "credibility county factor (less certain)", cf)]
        t = pd.DataFrame(lines, columns=["part", "term", "value"])
        f = t.loc[t.part == "frequency", "value"].prod()
        s = t.loc[t.part == "severity", "value"].prod()
        total = f * s * t.loc[t.part.isin(["loads", "county"]), "value"].prod()
        t.attrs.update({"frequency": f, "severity": s, "pure_premium": total})
        return t


def load(path=PATH):
    return ProductionModel(json.loads(Path(path).read_text(encoding="utf-8")))


def _self_test():
    if not PATH.exists():
        print("models/production_model.json not found (created by notebook 09)")
        return False
    m = load()
    # Milestone 9's worked example: Virginia Beach, 1960s 1-storey house, A/V zone, base levels otherwise.
    row = {"flood_zone_group": "A/V high risk", "construction_band": "1960-1979", "floors_group": "1",
           "occupancy_group": m.sev.spec.bases["occupancy_group"],
           "building_coverage_band": m.sev.spec.bases["building_coverage_band"],
           "elevated": m.sev.spec.bases["elevated"], "primary_residence": m.sev.spec.bases["primary_residence"],
           "post_firm": "False", "county": "51810"}
    for f in m.features:                       # any feature the hybrid added: use its base level
        row.setdefault(f, (m.freq.spec.bases | m.sev.spec.bases)[f])
    w = m.explain(row)
    p = m.predict(pd.DataFrame([row]))
    ok = np.isclose(w.attrs["pure_premium"], p["pure_premium"].iloc[0])
    print(f"  worksheet total ${w.attrs['pure_premium']:,.2f} = predict() ${p['pure_premium'].iloc[0]:,.2f}: {ok}")
    target = m.card.get("check_virginia_beach_worksheet")
    if target is not None:
        ok2 = abs(w.attrs["pure_premium"] - target) < 0.01
        print(f"  matches the model card's stored check (${target:,.2f}): {ok2}")
        ok &= ok2
    # Multi-state model: the stored check rows for the other states (same house, one county each).
    for label, chk in m.card.get("check_worksheets", {}).items():
        r = {**row, "county": chk["county"]}
        got = float(m.predict(pd.DataFrame([r]))["pure_premium"].iloc[0])
        okc = abs(got - chk["pure_premium"]) < 0.01 and np.isclose(m.explain(r).attrs["pure_premium"], got)
        print(f"  {label}: ${got:,.2f} (stored ${chk['pure_premium']:,.2f}): {okc}")
        ok &= okc
    # The Virginia-only model must still load and reproduce its own stored check.
    if VA_PATH.exists() and VA_PATH != PATH:
        mv = load(VA_PATH)
        got = float(mv.predict(pd.DataFrame([row]))["pure_premium"].iloc[0])
        okv = abs(got - mv.card["check_virginia_beach_worksheet"]) < 0.01
        print(f"  Virginia-only model (production_model_va.json): ${got:,.2f}: {okv}")
        ok &= okv
    return bool(ok)


if __name__ == "__main__":
    passed = _self_test()
    print("production self-test passed:", passed)
    sys.exit(0 if passed else 1)
