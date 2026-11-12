"""
LOCAL EXPLANATIONS for the production flood model (Milestone 12; three states since Milestone 13),
plus the builder for the small app data files in app/data/.

Two questions the app answers:

    import explain
    explain.explain_profile({"county": "Norfolk city", "flood_zone_group": "A/V high risk", ...})
        -> expected frequency, severity and pure premium for that GROUP of insured buildings,
           and a step-by-step "waterfall" from the STATE average to the result.
    explain.explain_county("Norfolk city")
        -> the county's expected loss cost, its credibility county factor and Z, its mix of
           insured buildings, and how much of its difference from its state's average is due to
           the mix vs the county factor.
    Counties are identified by FIPS code (unique) or name; a name shared by several states
    ("Montgomery County") needs state="MD" or "Montgomery County, MD".

AREA LEVEL ONLY. Every number describes the average insured building with those
characteristics (or the county's insured buildings as a whole) in an average year. Nothing
here is a quote, a premium or a prediction for any individual property.

How the profile waterfall works (multiplicative Shapley values, the idea behind SHAP):
    Start from the state average expected loss per policy-year. "Learning" a factor of the
    profile (e.g. its flood zone) moves the expected value: set that factor on every insured
    building in that state and re-average. The move depends on which factors you already know,
    so we average it over EVERY order of learning the factors (2^8 = 256 combinations, exact).
    Each factor's step is a multiplier; the steps multiply exactly from the state average to
    the profile's expected loss cost, and they don't depend on any chosen order.

At run time this module reads only committed files: models/production_model.json and
app/data/*. Only `build` reads data/processed/ (which is not in Git).

    .venv\\Scripts\\python.exe src\\explain.py build     # rebuild app/data from data/processed
    .venv\\Scripts\\python.exe src\\explain.py           # self-test
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
for p in (SRC, SRC / "models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import production                              # noqa: E402

ROOT = SRC.parent
APP_DATA = ROOT / "app" / "data"
CELLS_PATH = ROOT / "data" / "processed" / "model_data_3states.parquet"     # Milestone 13
LOSS = "losses_2025"
VA_STATE_FIPS = "51"
STATE_FIPS = {"VA": "51", "NC": "37", "MD": "24"}
STATE_NAMES = {"VA": "Virginia", "NC": "North Carolina", "MD": "Maryland"}
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}


def state_of(county):
    """State postal code of a county FIPS code ('37183' -> 'NC'; 'UNKNOWN-MD' -> 'MD')."""
    c = str(county)
    if c.startswith("UNKNOWN-"):
        return c.split("-", 1)[1]
    return FIPS_STATE.get(c[:2])
# Census 2020 names miss localities that no longer exist; FEMA records still use their codes.
FORMER_NAMES = {"51515": "Bedford city (former; part of Bedford County since 2013)"}

# The 7 building features the production model uses (frequency and/or severity).
FEATURES = ["flood_zone_group", "construction_band", "occupancy_group", "building_coverage_band",
            "elevated", "floors_group", "primary_residence"]
# post_firm is NOT in the model (it failed the keep rule: its effect is carried by year built
# and zone), but it's kept for the county mix and accepted as input.
EXTRA = ["post_firm"]

LABELS = {"flood_zone_group": "Flood zone", "construction_band": "Year built",
          "occupancy_group": "Occupancy", "building_coverage_band": "Building coverage",
          "elevated": "Elevated building", "floors_group": "Floors",
          "primary_residence": "Primary residence", "post_firm": "Built after the first flood map (post-FIRM)",
          "county": "County factor (credibility, less certain)"}

# Plain-English phrases for a profile's level ("because it ..."), used in sentences.
LEVEL_PHRASES = {
    ("flood_zone_group", "A/V high risk"): "it is in a high-risk A/V flood zone",
    ("flood_zone_group", "B/C/X moderate-low risk"): "it is in a moderate/low-risk B/C/X zone",
    ("construction_band", "before 1960"): "it was built before 1960",
    ("construction_band", "1960-1979"): "it was built in 1960-1979",
    ("construction_band", "1980-1999"): "it was built in 1980-1999",
    ("construction_band", "2000 or later"): "it was built in 2000 or later",
    ("occupancy_group", "single-unit residential"): "it is a single-unit home (claims are smaller than for apartment and commercial buildings)",
    ("occupancy_group", "multi-unit residential"): "it is a multi-unit residential building",
    ("occupancy_group", "non-residential"): "it is a non-residential building",
    ("building_coverage_band", "$0 (contents only)"): "it has contents-only cover",
    ("building_coverage_band", "under $100K"): "it has building cover under $100K",
    ("building_coverage_band", "$100K-$249K"): "it has building cover of $100K-$249K",
    ("building_coverage_band", "$250K or more"): "it has building cover of $250K or more",
    ("elevated", "True"): "it is elevated",
    ("elevated", "False"): "it is not elevated",
    ("floors_group", "1"): "it has one floor",
    ("floors_group", "2"): "it has two floors",
    ("floors_group", "3+"): "it has three or more floors",
    ("primary_residence", "True"): "it is a primary residence",
    ("primary_residence", "False"): "it is not a primary residence",
}

CERTAINTY_RULE = "High: Z >= 0.8; Medium: 0.5 <= Z < 0.8; Low: Z < 0.5 or no claims (Z = severity credibility)"


def _str_levels(df, cols):
    """Booleans become 'True'/'False' text, as in the model specs."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(str)
    return out


# ---------------------------------------------------------------------------
# Rating tables: the production GLMs as lookup arrays (log scale)
# ---------------------------------------------------------------------------
class RatingTables:
    """The production model as small lookup tables:
    log pp = const_f + const_s + log(ICC) + log(off-balance)
             + sum over features of (freq + sev log relativity of the level)
             + sum over interactions of the two-way log relativity
             + log(county factor).
    Built from models/production_model.json; also written to app/data/rating_factors.json."""

    def __init__(self, d):
        self.d = d
        self.levels = d["levels"]                               # feature -> list of levels
        self.idx = {f: {l: i for i, l in enumerate(lv)} for f, lv in self.levels.items()}
        self.main_f = {f: np.array(d["frequency"]["log_relativity"][f]) for f in FEATURES}
        self.main_s = {f: np.array(d["severity"]["log_relativity"][f]) for f in FEATURES}
        self.pairs = [(p["part"], p["a"], p["b"], np.array(p["log_relativity"])) for p in d["interactions"]]
        self.log_base_f = math.log(d["frequency"]["base_per_policy_year"])
        self.log_base_s = math.log(d["severity"]["base_claim_size"])     # includes the large-loss load
        self.log_loads = math.log(d["loads"]["icc_bridge"]) + math.log(d["loads"]["off_balance"])
        self.counties = list(d["county_factors"].keys())
        self.cidx = {c: i for i, c in enumerate(self.counties)}
        self.log_cf = np.log(np.array([d["county_factors"][c] for c in self.counties]))
        self.default_cf = d["county_factor_default"]
        # Milestone 13: state factor (frequency and severity log relativities by state) and one
        # off-balance per state. A Virginia-only file has neither (every state term = 0).
        st = d.get("state") or {}
        self.states = list(st.get("levels", []))
        self.state_f = st.get("frequency_log_relativity", {})
        self.state_s = st.get("severity_log_relativity", {})
        self.log_ob_state = {k: math.log(v) for k, v in (d["loads"].get("off_balance_by_state") or {}).items()}

    @classmethod
    def from_production(cls, m):
        """Export the production model (a production.ProductionModel) to the table form."""
        levels, out_f, out_s = {}, {}, {}
        for f in FEATURES:
            spec = m.freq.spec if f in m.freq.spec.features else m.sev.spec
            levels[f] = list(spec.levels[f])
            for model, out in ((m.freq, out_f), (m.sev, out_s)):
                out[f] = [float(model.params.get(f"{f}[{l}]", 0.0)) for l in levels[f]]
        inter = []
        for part, model in (("frequency", m.freq), ("severity", m.sev)):
            for a, b in model.spec.interactions:
                tab = [[float(model.params.get(f"{a}[{la}]:{b}[{lb}]", 0.0)) for lb in levels[b]]
                       for la in levels[a]]
                inter.append({"part": part, "a": a, "b": b, "log_relativity": tab})
        ct = m.county_table.copy()
        ct["county"] = ct["county"].astype(str)
        state = None
        if m.by_state:
            lv = list((m.freq.spec.levels.get("state") or m.sev.spec.levels["state"]))
            state = {"levels": lv, "base": lv[0],
                     "frequency_log_relativity": {l: float(m.freq.params.get(f"state[{l}]", 0.0)) for l in lv},
                     "severity_log_relativity": {l: float(m.sev.params.get(f"state[{l}]", 0.0)) for l in lv},
                     "note": "state factor = the state's average-year level vs the base state (from state x "
                             "year effects); the state comes from the county code"}
        d = {
            "description": "Production model (hybrid GLM) as lookup tables. pure premium per policy-year = "
                           "base_per_policy_year x base_claim_size x exp(sum of frequency and severity log "
                           "relativities, incl. interactions and the state factor) x icc_bridge x off_balance "
                           "(per state when off_balance_by_state is given) x county factor. "
                           "2025 dollars, average year. Area-level only.",
            "levels": levels,
            "bases": {f: (m.freq.spec.bases | m.sev.spec.bases)[f] for f in FEATURES},
            "frequency": {"base_per_policy_year": float(np.exp(m.freq.params["const"] + m.freq.avg_year_log)),
                          "features": list(m.freq.spec.features), "log_relativity": out_f},
            "severity": {"base_claim_size": float(np.exp(m.sev.params["const"] + m.sev.avg_year_log) * m.sev.load),
                         "large_loss_load_included": float(m.sev.load),
                         "features": list(m.sev.spec.features), "log_relativity": out_s},
            "interactions": inter,
            "loads": {"large_loss_load": float(m.sev.load), "icc_bridge": float(m.combined.icc_bridge),
                      "off_balance": float(m.combined.off_balance),
                      "off_balance_by_state": m.combined.off_balance_by},
            "state": state,
            "county_factors": {str(k): float(v) for k, v in m.combined.county_factors.items()},
            "county_factor_default": 1.0,
            "county_credibility": {r.county: {"Z_freq": float(r.Z_freq), "Z_sev": float(r.Z_sev),
                                              "claims": float(r.s_n), "freq_ae": float(r.freq_ae),
                                              "sev_ae": float(r.sev_ae), "name": r.name}
                                   for r in ct.itertuples()},
            "labels": {f: LABELS[f] for f in FEATURES},
            "disclaimer": m.disclaimer,
        }
        return cls(d)

    # ---- encoding
    def codes(self, df):
        """Integer level codes for each feature (raises on an unknown level)."""
        out = {}
        for f in FEATURES:
            s = df[f].astype(str)
            c = s.map(self.idx[f])
            if c.isna().any():
                raise ValueError(f"unknown level for {f}: {sorted(set(s[c.isna()]))}; allowed {self.levels[f]}")
            out[f] = c.to_numpy(int)
        return out

    def county_codes(self, county):
        """Index into the county factor table; -1 = not in the table (factor 1.0)."""
        return pd.Series(county).astype(str).map(self.cidx).fillna(-1).to_numpy(int)

    def county_log_factor(self, ccodes):
        return np.where(ccodes >= 0, self.log_cf[np.maximum(ccodes, 0)], math.log(self.default_cf))

    # ---- states
    def state_codes(self, df_or_counties):
        """State of each row: the 'state' column if present, else from the county code."""
        if isinstance(df_or_counties, pd.DataFrame):
            if "state" in df_or_counties:
                return df_or_counties["state"].astype(str).to_numpy()
            df_or_counties = df_or_counties["county"]
        return np.array([state_of(c) or "" for c in pd.Series(df_or_counties).astype(str)], dtype=object)

    def _state_terms(self, states, n):
        """(log freq, log sev, log off-balance) state terms per row (zeros for a one-state file)."""
        if states is None or not self.states:
            return 0.0, 0.0, np.full(n, self.log_loads - math.log(self.d["loads"]["icc_bridge"]))
        bad = sorted(set(states) - set(self.states))
        if bad:
            raise ValueError(f"unknown state {bad}; the model covers {self.states}")
        f = np.array([self.state_f[s] for s in states])
        g = np.array([self.state_s[s] for s in states])
        ob = np.array([self.log_ob_state[s] for s in states])
        return f, g, ob

    # ---- rating
    def log_parts(self, codes, states=None):
        """(log frequency, log severity) per row, average year, without the county factor.
        states: each row's state (needed for the multi-state model's state factor)."""
        n = len(next(iter(codes.values())))
        sf, ss, _ = self._state_terms(states, n)
        lf = self.log_base_f + sf + sum(self.main_f[f][codes[f]] for f in FEATURES)
        ls = self.log_base_s + ss + sum(self.main_s[f][codes[f]] for f in FEATURES)
        for part, a, b, tab in self.pairs:
            v = tab[codes[a], codes[b]]
            if part == "frequency":
                lf = lf + v
            else:
                ls = ls + v
        return lf, ls

    def log_loads_rows(self, states, n):
        """log(ICC bridge x off-balance) per row (the off-balance is per state when available)."""
        _, _, ob = self._state_terms(states, n)
        return math.log(self.d["loads"]["icc_bridge"]) + ob

    def log_pp(self, codes, ccodes, states=None):
        lf, ls = self.log_parts(codes, states)
        return lf + ls + self.log_loads_rows(states, len(lf)) + self.county_log_factor(ccodes)

    def rate(self, df):
        """frequency, severity, county_factor, pure_premium per row (same as production.predict)."""
        c = self.codes(df)
        st = self.state_codes(df) if self.states else None
        lf, ls = self.log_parts(c, st)
        cf = np.exp(self.county_log_factor(self.county_codes(df["county"])))
        loads = np.exp(self.log_loads_rows(st, len(df)))
        return pd.DataFrame({"frequency": np.exp(lf), "severity": np.exp(ls), "county_factor": cf,
                             "pure_premium": np.exp(lf + ls) * loads * cf}, index=df.index)

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.d, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Shapley values (exact, all coalitions)
# ---------------------------------------------------------------------------
def shapley_from_values(v, n):
    """v[mask] = value of the coalition whose members are the set bits of mask (n players).
    Returns each player's Shapley value: the average, over all orders of adding players, of how
    much the value changes when that player is added. They sum to v[full] - v[empty]."""
    fact = [math.factorial(k) for k in range(n + 1)]
    phi = np.zeros(n)
    for mask in range(1 << n):
        size = bin(mask).count("1")
        for j in range(n):
            if not mask & (1 << j):
                wgt = fact[size] * fact[n - size - 1] / fact[n]
                phi[j] += wgt * (v[mask | (1 << j)] - v[mask])
    return phi


# ---------------------------------------------------------------------------
# The explainer
# ---------------------------------------------------------------------------
class Explainer:
    """Holds the rating tables, the insured portfolio of each state (the "background" for
    averages) and the county summary. Create once (load()) and reuse; the app should cache it."""

    def __init__(self, tables, portfolio, county_summary=None, averages=None, model=None):
        self.t = tables
        self.model = model
        p = _str_levels(portfolio, FEATURES + EXTRA)
        p["county"] = p["county"].astype(str)
        if "state" not in p:
            p["state"] = [state_of(c) for c in p["county"]]
        p["state"] = p["state"].astype(str)
        self.portfolio = p
        # Row-level expected values for the whole portfolio (with post_firm, for the mix)
        self.p_rates = tables.rate(p)
        self.state_list = [s for s in STATE_NAMES if s in set(p["state"])]
        # Shapley background per state: aggregated over the model features + county.
        self.bg = {}
        for st in self.state_list:
            bg = p[p["state"] == st].groupby(FEATURES + ["county"], observed=True)["exposure"].sum().reset_index()
            self.bg[st] = {"df": bg, "codes": tables.codes(bg), "w": bg["exposure"].to_numpy(float),
                           "lcf": tables.county_log_factor(tables.county_codes(bg["county"])),
                           "states": np.array([st] * len(bg), dtype=object) if tables.states else None}
        self.state_stats = {st: self._area_stats(p.index[p["state"] == st]) for st in self.state_list}
        self.all_stats = self._area_stats(p.index)
        self.va = self.state_stats.get("VA", self.all_stats)       # kept for older code
        self.county_summary = county_summary
        self.averages = averages
        names = p[["county", "county_name"]].drop_duplicates().set_index("county")["county_name"]
        self.names = names.to_dict()

    # ---- area statistics (state or county)
    def _area_stats(self, rows):
        p, r = self.portfolio.loc[rows], self.p_rates.loc[rows]
        w = p["exposure"].to_numpy(float)
        f, s, cf = r["frequency"].to_numpy(), r["severity"].to_numpy(), r["county_factor"].to_numpy()
        pp = r["pure_premium"].to_numpy()
        W = w.sum()
        out = {
            "policy_years": float(W),
            "expected_frequency": float((w * f).sum() / W),
            "expected_severity": float((w * f * s).sum() / (w * f).sum()),   # expected losses / expected claims
            "expected_pure_premium": float((w * pp).sum() / W),
            "expected_pure_premium_no_county": float((w * pp / cf).sum() / W),
        }
        out["mix"] = mix_shares(p)
        # mean relativity of each factor (main effects, frequency x severity), for mix indices
        codes = self.t.codes(p)
        out["mean_relativity"] = {f: float((w * np.exp(self.t.main_f[f][codes[f]] + self.t.main_s[f][codes[f]])).sum() / W)
                                  for f in FEATURES}
        return out

    # ---- resolving a county
    def label(self, fips):
        """'Montgomery County, MD'."""
        return f"{self.names.get(fips, fips)}, {state_of(fips)}"

    def resolve_county(self, county, state=None):
        """FIPS (e.g. '51710') or a name ('Norfolk city', 'Norfolk', 'Montgomery County, MD').
        state: optional postal code ('VA', 'NC', 'MD') to pick between states. Ambiguous names
        (e.g. Fairfax, Richmond: both a county and a city in Virginia; Montgomery County: in all
        three states) raise with the options."""
        if county is None:
            return None
        s = str(county).strip()
        if s in self.names:
            if state and state_of(s) != state.upper():
                raise ValueError(f"county {s} is in {state_of(s)}, not {state}")
            return s
        if "," in s:                                   # "Name, ST"
            name, st = (x.strip() for x in s.rsplit(",", 1))
            if st.upper() in STATE_NAMES:
                if state and state.upper() != st.upper():
                    raise ValueError(f"'{county}' is not in {state}")
                s, state = name, st
        pool = {k: v for k, v in self.names.items() if state is None or state_of(k) == state.upper()}
        cands = [k for k, v in pool.items() if v.lower() == s.lower()]
        if not cands:
            cands = [k for k, v in pool.items()
                     if v.lower().replace(" city", "").replace(" county", "") == s.lower()]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            raise ValueError(f"'{county}' is ambiguous: {sorted(self.label(c) for c in cands)}")
        where = STATE_NAMES.get((state or "").upper(), "Virginia, North Carolina or Maryland")
        raise ValueError(f"unknown county or city in {where}: '{county}'")

    def county_credibility(self, fips):
        cc = self.t.d["county_credibility"].get(str(fips))
        if cc is None:
            return {"Z_freq": 0.0, "Z_sev": 0.0, "claims": 0.0}
        return cc

    # ---- profile
    def normalise_profile(self, features, state=None):
        """Fill missing features with the model's base levels; booleans -> 'True'/'False'.
        The state comes from the county; without a county, `state` (or features['state']) is
        needed when the portfolio covers several states."""
        prof = {}
        for f in FEATURES + EXTRA:
            v = features.get(f, self.t.d["bases"].get(f, "False") if f in FEATURES else None)
            if v is None:
                prof[f] = None
                continue
            v = str(v)
            if f in FEATURES and v not in self.t.idx[f]:
                raise ValueError(f"unknown level for {f}: '{v}'; allowed {self.t.levels[f]}")
            prof[f] = v
        state = state or features.get("state")
        prof["county"] = self.resolve_county(features.get("county"), state)
        if prof["county"] is not None:
            prof["state"] = state_of(prof["county"])
        elif state:
            if str(state).upper() not in self.state_list:
                raise ValueError(f"unknown state '{state}'; choose from {self.state_list}")
            prof["state"] = str(state).upper()
        elif len(self.state_list) == 1:
            prof["state"] = self.state_list[0]
        else:
            raise ValueError("give a county or a state (VA, NC or MD)")
        return prof

    def _coalition_values(self, prof):
        """v[mask] = log of the exposure-weighted STATE average pure premium when the players in
        mask are set to the profile's values on every insured building of that state (8 players:
        the 7 features + county; without a county, 7 players). The state itself is never a
        player: every building in the background is already in the profile's state."""
        players = FEATURES + (["county"] if prof["county"] is not None else [])
        n = len(players)
        B = self.bg[prof["state"]]
        nb = len(B["df"])
        pcode = {f: self.t.idx[f][prof[f]] for f in FEATURES}
        pc_log = (float(self.t.county_log_factor(self.t.county_codes([prof["county"]]))[0])
                  if prof["county"] is not None else None)
        loads = self.t.log_loads_rows(B["states"], nb)
        v = np.zeros(1 << n)
        W = B["w"].sum()
        for mask in range(1 << n):
            codes = {f: (np.full(nb, pcode[f]) if mask & (1 << j) else B["codes"][f])
                     for j, f in enumerate(FEATURES)}
            lf, ls = self.t.log_parts(codes, B["states"])
            lcf = pc_log if (n == 8 and mask & (1 << 7)) else B["lcf"]
            lp = lf + ls + loads + lcf
            mx = lp.max()
            v[mask] = mx + math.log((B["w"] * np.exp(lp - mx)).sum() / W)
        return players, v

    def explain_profile(self, features, state=None):
        """Expected frequency, severity and pure premium (per policy-year, average year, 2025 $)
        for a GROUP of insured buildings with these characteristics in this county, and the
        waterfall from the state average. `features`: dict with any of FEATURES, 'post_firm'
        and 'county' (FIPS or name; None = anywhere in the state given by `state`)."""
        prof = self.normalise_profile(features, state)
        st = prof["state"]
        sname = STATE_NAMES.get(st, st)
        players, v = self._coalition_values(prof)
        phi = shapley_from_values(v, len(players))
        va_pp = float(math.exp(v[0]))
        pp = float(math.exp(v[-1]))
        row = pd.DataFrame([{**{f: prof[f] for f in FEATURES}, "county": prof["county"] or "", "state": st}])
        r = self.t.rate(row).iloc[0]
        freq, sev = float(r["frequency"]), float(r["severity"])
        if prof["county"] is None:          # county drawn from the state's mix: average factor
            cf = pp / float(r["pure_premium"] / r["county_factor"])
        else:
            cf = float(r["county_factor"])
            assert abs(pp - float(r["pure_premium"])) < 1e-6 * pp, "Shapley end point != rating"
        steps = []
        for j, name in enumerate(players):
            lvl = prof[name] if name != "county" else (self.names.get(prof["county"], prof["county"]))
            steps.append({"player": name, "step": f"{LABELS[name]}: {lvl}",
                          "multiplier": float(math.exp(phi[j])), "log_effect": float(phi[j])})
        wf = _waterfall_table(steps, va_pp, pp, f"{sname} average", "This profile")
        cred = self.county_credibility(prof["county"]) if prof["county"] else None
        any_county = f"{sname} (any county)"
        res = {
            "kind": "profile", "profile": prof, "state": st, "state_name": sname,
            "county_name": self.names.get(prof["county"], any_county) if prof["county"] else any_county,
            "frequency": freq, "severity": sev, "county_factor": cf, "pure_premium": pp,
            "state_average_pure_premium": va_pp, "ratio_to_state": pp / va_pp,
            "ratio_to_three_states": pp / self.all_stats["expected_pure_premium"],
            "Z_sev": cred["Z_sev"] if cred else None, "Z_freq": cred["Z_freq"] if cred else None,
            "certainty": certainty_label(cred) if cred else None,
            "waterfall": wf,
            "notes": ["post_firm is not in the model: its effect is carried by year built and flood zone."]
                     if prof.get("post_firm") is not None else [],
            "disclaimer": self.t.d["disclaimer"],
        }
        if self.model is not None and prof["county"] is not None:
            res["worksheet"] = self.model.explain({**{f: prof[f] for f in FEATURES}, "county": prof["county"],
                                                   "state": st})
        res["sentence"] = profile_sentence(res)
        return res

    # ---- county
    def explain_county(self, county, state=None):
        """The county's expected loss cost per policy-year for its own mix of insured buildings,
        its county factor and credibility, its mix vs its state, and the breakdown of WHY it
        differs from the state average (property mix vs county factor)."""
        fips = self.resolve_county(county, state)
        rows = self.portfolio.index[self.portfolio["county"] == fips]
        if len(rows) == 0:
            raise ValueError(f"no insured exposure for {county}")
        st = state_of(fips)
        sname = STATE_NAMES.get(st, st)
        c, va = self._area_stats(rows), self.state_stats[st]
        cf = float(np.exp(self.t.county_log_factor(self.t.county_codes([fips]))[0]))
        cf_bar = va["expected_pure_premium"] / va["expected_pure_premium_no_county"]
        mix_effect = c["expected_pure_premium_no_county"] / va["expected_pure_premium_no_county"]
        cf_effect = cf / cf_bar
        ratio = c["expected_pure_premium"] / va["expected_pure_premium"]
        assert abs(ratio - mix_effect * cf_effect) < 1e-9 * ratio
        steps = []
        idx_prod = 1.0
        for f in FEATURES:
            m = c["mean_relativity"][f] / va["mean_relativity"][f]
            idx_prod *= m
            steps.append({"player": f, "step": f"Mix: {LABELS[f].lower()}", "multiplier": m,
                          "log_effect": math.log(m)})
        overlap = mix_effect / idx_prod
        steps.append({"player": "overlap", "step": "Mix: overlap and interactions", "multiplier": overlap,
                      "log_effect": math.log(overlap)})
        steps.append({"player": "county", "step": LABELS["county"], "multiplier": cf_effect,
                      "log_effect": math.log(cf_effect)})
        wf = _waterfall_table(steps, va["expected_pure_premium"], c["expected_pure_premium"],
                              f"{sname} average", self.names[fips])
        cred = self.county_credibility(fips)
        res = {
            "kind": "county", "county": fips, "county_name": self.names[fips], "state": st, "state_name": sname,
            "policy_years": c["policy_years"],
            "frequency": c["expected_frequency"], "severity": c["expected_severity"],
            "pure_premium": c["expected_pure_premium"],
            "state_average_pure_premium": va["expected_pure_premium"], "ratio_to_state": ratio,
            "ratio_to_three_states": c["expected_pure_premium"] / self.all_stats["expected_pure_premium"],
            "mix_effect": mix_effect, "county_factor_effect": cf_effect, "overlap_effect": overlap,
            "county_factor": cf, "Z_sev": cred["Z_sev"], "Z_freq": cred["Z_freq"],
            "claims": cred["claims"], "certainty": certainty_label(cred),
            "mix": c["mix"], "state_mix": va["mix"], "waterfall": wf,
            "disclaimer": self.t.d["disclaimer"],
        }
        res["uncertainty_note"] = uncertainty_note(res)
        res["sentence"] = county_sentence(res)
        return res


def _waterfall_table(steps, start, end, start_label, end_label):
    """Steps sorted by size of effect (order doesn't change Shapley values), with running $."""
    st = sorted(steps, key=lambda s: -abs(s["log_effect"]))
    rows = [{"step": start_label, "player": "start", "multiplier": np.nan, "before": np.nan, "after": start}]
    run = start
    for s in st:
        new = run * s["multiplier"]
        rows.append({"step": s["step"], "player": s["player"], "multiplier": s["multiplier"],
                     "before": run, "after": new})
        run = new
    assert abs(run - end) < 1e-6 * max(end, 1e-9), f"waterfall does not reach the total: {run} vs {end}"
    rows.append({"step": end_label, "player": "end", "multiplier": end / start, "before": np.nan, "after": end})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mix, certainty and sentences
# ---------------------------------------------------------------------------
MIX_DEFS = {   # name -> (column, level(s) counted, phrase for a sentence)
    "pct_av_zone": ("flood_zone_group", ["A/V high risk"], "are in high-risk A/V zones"),
    "pct_elevated": ("elevated", ["True"], "are elevated"),
    "pct_pre_firm": ("post_firm", ["False"], "were built before the area's first flood map (pre-FIRM)"),
    "pct_built_pre_1980": ("construction_band", ["before 1960", "1960-1979"], "were built before 1980"),
    "pct_multi_or_nonres": ("occupancy_group", ["multi-unit residential", "non-residential"],
                            "are multi-unit or non-residential buildings"),
    "pct_not_primary": ("primary_residence", ["False"], "are not primary residences"),
    "pct_cover_250k_plus": ("building_coverage_band", ["$250K or more"], "carry building cover of $250K or more"),
    "pct_cover_under_100k": ("building_coverage_band", ["under $100K", "$0 (contents only)"],
                             "carry building cover under $100K (or contents only)"),
    "pct_one_floor": ("floors_group", ["1"], "have one floor"),
}
# Which mix share best describes each factor's mix index in a sentence.
FACTOR_MIX = {"flood_zone_group": "pct_av_zone", "construction_band": "pct_built_pre_1980",
              "occupancy_group": "pct_multi_or_nonres", "building_coverage_band": "pct_cover_250k_plus",
              "elevated": "pct_elevated", "floors_group": "pct_one_floor", "primary_residence": "pct_not_primary"}


def mix_shares(p):
    """Exposure-weighted shares (%) of the insured buildings with each characteristic."""
    w = p["exposure"].to_numpy(float)
    W = w.sum()
    return {k: float(100 * w[p[col].astype(str).isin(lv).to_numpy()].sum() / W) for k, (col, lv, _) in MIX_DEFS.items()}


def certainty_label(cred):
    z, n = cred.get("Z_sev", 0.0), cred.get("claims", 0.0)
    if n <= 0:
        return "Low"
    return "High" if z >= 0.8 else ("Medium" if z >= 0.5 else "Low")


def uncertainty_note(res):
    z, n, lab = res["Z_sev"], res["claims"], res["certainty"]
    zf = res.get("Z_freq") or 0.0
    what = "claim-size" if zf == 0 else "claim-size and claim-frequency"
    base = (f"{lab} certainty. The county factor ({res['county_factor']:.2f}) blends this county's own {what} "
            f"history ({n:,.0f} paid claims, credibility Z = {z:.2f}) with the model for the state. ")
    if n <= 0:
        base = (f"Low certainty. There are no paid claims here in 2010-2025, so the county factor "
                f"({res['county_factor']:.2f}) is just the state's re-centring value. ")
    elif lab == "Low":
        base += "With so few claims, most of the weight is on the statewide model and the factor could move a lot with one storm. "
    elif lab == "Medium":
        base += "There is some local evidence, but the factor is still uncertain. "
    else:
        base += "There is plenty of local claim history, but county factors still move in storm years. "
    if zf == 0:
        return base + ("Claim frequency gets no county credit (Z = 0): county differences in how often "
                       "claims happen were no bigger than storm-to-storm noise.")
    return base + (f"Claim frequency gets some county credit (Z = {zf:.2f}), but frequency differences between "
                   "counties are mostly storm-to-storm noise.")


def _ratio_phrase(r, state_name="Virginia"):
    if r >= 1.1:
        return f"about {r:.1f}x the {state_name} average"
    if r <= 1 / 1.1:
        return f"about {100 * (1 - r):.0f}% below the {state_name} average"
    return f"close to the {state_name} average"


def _join(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _drivers(wf, direction, k=2, min_log=math.log(1.05)):
    """Top-k steps pushing in the overall direction (+1 up, -1 down), ignoring tiny ones."""
    s = wf[~wf["player"].isin(["start", "end"])].copy()
    s["lg"] = np.log(s["multiplier"])
    s = s[(s["lg"] * direction) > min_log].sort_values("lg", key=lambda x: -x.abs())
    return s.head(k)


def county_sentence(res):
    r = res["ratio_to_state"]
    sn = res["state_name"]
    d = 1 if r >= 1 else -1
    head = (f"Model-based expected flood losses for insured buildings in {res['county_name']}, {res['state']} are "
            f"{_ratio_phrase(r, sn)} (about ${res['pure_premium']:,.0f} vs ${res['state_average_pure_premium']:,.0f} "
            f"per policy-year in an average year)")
    def phrase(row):
        if row.player in FACTOR_MIX:
            key = FACTOR_MIX[row.player]
            return (f"{res['mix'][key]:.0f}% of insured buildings here {MIX_DEFS[key][2]} "
                    f"({sn}: {res['state_mix'][key]:.0f}%)")
        if row.player == "county":
            return (f"its own claims history has had {'larger' if row.multiplier > 1 else 'smaller'} claims "
                    f"than the model expects (county factor {res['county_factor']:.2f}, "
                    f"{res['certainty'].lower()} certainty)")
        return "of how its building characteristics combine"          # the overlap step

    if abs(math.log(r)) > math.log(1.1):
        parts = [phrase(row) for row in _drivers(res["waterfall"], d).itertuples()]
        if parts:
            head += ", mainly because " + _join(parts)
    else:                                         # near average: say what offsets what, if anything big
        up = _drivers(res["waterfall"], 1, k=1, min_log=math.log(1.1))
        down = _drivers(res["waterfall"], -1, k=1, min_log=math.log(1.1))
        if len(up) and len(down):
            head += (f": pushing it up, {phrase(next(up.itertuples()))}; pulling it down, "
                     f"{phrase(next(down.itertuples()))}")
        else:
            head += "; no single factor moves it much"
    return head + ". This is an area-level estimate, not a prediction for any individual property."


def profile_sentence(res):
    r = res["ratio_to_state"]
    d = 1 if r >= 1 else -1
    where = res["county_name"] + (f", {res['state']}" if res["profile"]["county"] else "")
    head = (f"For insured buildings with this profile in {where}, the model's expected flood loss is about "
            f"${res['pure_premium']:,.0f} per policy-year in an average year, {_ratio_phrase(r, res['state_name'])}")
    drv = _drivers(res["waterfall"], d)
    parts = []
    for row in drv.itertuples():
        if row.player == "county":
            parts.append(f"the county's claims have been {'larger' if row.multiplier > 1 else 'smaller'} than "
                         f"expected (a less certain factor)")
        else:
            parts.append(LEVEL_PHRASES.get((row.player, res["profile"][row.player]), row.step))
    if parts and abs(math.log(r)) > math.log(1.1):
        head += ", mainly because " + _join(parts)
    return head + ". This describes the average for a group of similar insured buildings, not a quote or a " \
                  "prediction for any individual property."


# ---------------------------------------------------------------------------
# Waterfall chart
# ---------------------------------------------------------------------------
UP, DOWN, TOTAL = "#eb6834", "#2a78d6", "#52514e"      # dataviz reference palette slots 2, 1, secondary ink


def plot_waterfall(res, path=None, ax=None, title=None):
    """Bars from the state average to the result; orange = raises expected loss, blue = lowers."""
    import matplotlib.pyplot as plt
    wf = res["waterfall"]
    n = len(wf)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * n + 1.6))
    else:
        fig = ax.figure
    for i, row in enumerate(wf.itertuples()):
        y = n - 1 - i
        if row.player in ("start", "end"):
            ax.barh(y, row.after, color=TOTAL, height=0.6)
            ax.text(row.after, y, f"  ${row.after:,.0f}", va="center", fontsize=9, color="#0b0b0b")
        else:
            lo, hi = sorted([row.before, row.after])
            ax.barh(y, hi - lo, left=lo, color=UP if row.after > row.before else DOWN, height=0.6)
            ax.text(hi, y, f"  x{row.multiplier:.2f}", va="center", fontsize=9, color="#52514e")
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(wf["step"])[::-1], fontsize=9)
    ax.set_xlabel("Expected loss per policy-year (2025 $, average year)")
    xmax = max(wf["after"].max(), np.nanmax(wf["before"].to_numpy(dtype=float)))
    ax.set_xlim(0, xmax * 1.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ttl = title or (res["county_name"] if res["kind"] == "county" else f"Profile in {res['county_name']}")
    ax.set_title(f"{ttl}: {res['ratio_to_state']:.2f}x the {res['state_name']} average\n"
                 f"orange = raises, blue = lowers; area-level model estimate, not a property prediction",
                 fontsize=10, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ---------------------------------------------------------------------------
# Loading and building app data
# ---------------------------------------------------------------------------
_CACHE = {}


def load(app_data=APP_DATA, with_model=True):
    """The Explainer from committed files only (app/data + models/production_model.json)."""
    key = (str(app_data), with_model)
    if key not in _CACHE:
        app_data = Path(app_data)
        t = RatingTables.from_json(app_data / "rating_factors.json")
        port = pd.read_parquet(app_data / "portfolio.parquet")
        cs = pd.read_csv(app_data / "county_summary.csv", dtype={"county": str}) \
            if (app_data / "county_summary.csv").exists() else None
        av_path = app_data / "state_averages.json"
        av = json.loads(av_path.read_text(encoding="utf-8")) if av_path.exists() else None
        _CACHE[key] = Explainer(t, port, cs, av, production.load() if with_model else None)
    return _CACHE[key]


def explain_profile(features, state=None):
    return load().explain_profile(features, state)


def explain_county(county, state=None):
    return load().explain_county(county, state)


def build_portfolio(cells_path=CELLS_PATH):
    """Insured exposure 2010-2025 in the three states by model features + post_firm + county
    (no claims or losses: just where the insured buildings are). Returns (portfolio, cells)."""
    c = pd.read_parquet(cells_path)
    if "state" not in c:                                 # a Virginia-only table
        c = c[c["county"].astype(str).str.startswith(VA_STATE_FIPS)].assign(state="VA")
    c = c[c["state"].isin(list(STATE_NAMES))]
    c = c.assign(county=c["county"].astype(str),
                 county_name=[FORMER_NAMES.get(k, n) for k, n in zip(c["county"].astype(str), c["county_name"])])
    m = c[c["exposure_matched"] & (c["exposure"] > 0)]
    m = _str_levels(m, FEATURES + EXTRA)
    keys = FEATURES + EXTRA + ["state", "county", "county_name"]
    port = m.groupby(keys, observed=True)["exposure"].sum().reset_index()
    for col in keys:
        port[col] = port[col].astype("category")
    return port, c


def _averages(ex, st_stats, cells, t, label, m):
    """Average-year expected values for one area (a state or all three), plus actuals."""
    c25 = cells[cells["exposure_matched"] & (cells["exposure"] > 0) & (cells["year"] == 2025)]
    r25 = t.rate(_str_levels(c25, FEATURES).assign(county=c25["county"].astype(str), state=c25["state"].astype(str)))
    pp25 = float((c25["exposure"].to_numpy() * r25["pure_premium"].to_numpy()).sum() / c25["exposure"].sum())
    return {
        "basis": f"Exposure-weighted average of the production model's average-year expected values over all "
                 f"insured buildings in {label}, 2010-2025 mix. 2025 dollars.",
        "expected_frequency_per_policy_year": st_stats["expected_frequency"],
        "expected_severity_per_claim": st_stats["expected_severity"],
        "expected_pure_premium_per_policy_year": st_stats["expected_pure_premium"],
        "expected_pure_premium_2025_mix": pp25,
        "actual_loss_cost_2010_2025": float(cells[LOSS].sum() / cells["exposure"].sum()),
        "policy_years_2010_2025": st_stats["policy_years"],
        "paid_claims_2010_2025": int(cells["claims"].sum()),
        "losses_2025_dollars_2010_2025": float(cells[LOSS].sum()),
        "mix_pct": st_stats["mix"],
        "mean_relativity": st_stats["mean_relativity"],
        "average_county_factor": st_stats["expected_pure_premium"] / st_stats["expected_pure_premium_no_county"],
    }


def build_app_data(out=APP_DATA, cells_path=CELLS_PATH, verbose=True):
    """Write every file in app/data/. Reads data/processed (local only)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = production.load()
    t = RatingTables.from_production(m)
    t.to_json(out / "rating_factors.json")
    port, cells = build_portfolio(cells_path)
    port.to_parquet(out / "portfolio.parquet", index=False)

    # county x year history (actual experience; all rows, so unmatched claims count too)
    hist = (cells.groupby(["state", "county", "county_name", "year"], observed=True)
            .agg(policy_years=("exposure", "sum"), paid_claims=("claims", "sum"), losses_2025=(LOSS, "sum"))
            .reset_index())
    hist = hist[hist["policy_years"] > 0]
    hist.to_csv(out / "county_history.csv", index=False, float_format="%.4f")

    ex = Explainer(t, port, model=m)
    states = {st: _averages(ex, ex.state_stats[st], cells[cells["state"] == st], t, STATE_NAMES[st], m)
              for st in ex.state_list}
    averages = {
        "states": {st: {"name": STATE_NAMES[st], **v} for st, v in states.items()},
        "three_states": _averages(ex, ex.all_stats, cells, t, "Virginia, North Carolina and Maryland", m),
        "state_factor": t.d.get("state"),
        "comparison_rule": "Each county and profile is compared with its OWN state's average "
                           "(ratio_to_state); ratio_to_three_states compares with all three states together.",
        "certainty_rule": CERTAINTY_RULE,
        "disclaimer": m.disclaimer,
    }
    (out / "state_averages.json").write_text(json.dumps(averages, indent=1), encoding="utf-8")
    old = out / "virginia_averages.json"                   # replaced by state_averages.json
    if old.exists():
        old.unlink()

    rows = []
    obs = hist.groupby("county").agg(py=("policy_years", "sum"), cl=("paid_claims", "sum"), ls=("losses_2025", "sum"))
    for fips in sorted(port["county"].astype(str).unique()):
        e = ex.explain_county(fips)
        wf = e["waterfall"].set_index("player")
        row = {"county": fips, "state": e["state"], "name": e["county_name"],
               "policy_years_2010_2025": e["policy_years"],
               "avg_policies_per_year": e["policy_years"] / 16,
               "paid_claims_2010_2025": int(obs.loc[fips, "cl"]),
               "actual_loss_cost_2010_2025": float(obs.loc[fips, "ls"] / obs.loc[fips, "py"]),
               "expected_frequency": e["frequency"], "expected_severity": e["severity"],
               "expected_pure_premium": e["pure_premium"], "ratio_to_state": e["ratio_to_state"],
               "ratio_to_three_states": e["ratio_to_three_states"],
               "county_factor": e["county_factor"], "Z_sev": e["Z_sev"], "Z_freq": e["Z_freq"],
               "certainty": e["certainty"], "mix_effect": e["mix_effect"],
               "county_factor_effect": e["county_factor_effect"], "overlap_effect": e["overlap_effect"]}
        row.update({f"mix_index_{f}": float(wf.loc[f, "multiplier"]) for f in FEATURES})
        row.update(e["mix"])
        row["summary"] = e["sentence"]
        row["uncertainty_note"] = e["uncertainty_note"]
        rows.append(row)
    cs = pd.DataFrame(rows).sort_values(["state", "expected_pure_premium"], ascending=[True, False])
    cs.to_csv(out / "county_summary.csv", index=False, float_format="%.6g")
    if verbose:
        for f in sorted(out.iterdir()):
            print(f"  {f.name:28s} {f.stat().st_size / 1024:8.1f} KB")
        for st, v in states.items():
            print(f"  {st}: {int((cs['state'] == st).sum())} counties/cities; average "
                  f"${v['expected_pure_premium_per_policy_year']:,.2f} per policy-year (2025 mix "
                  f"${v['expected_pure_premium_2025_mix']:,.2f}; actual ${v['actual_loss_cost_2010_2025']:,.2f})")
    _CACHE.clear()
    return cs


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    ok = True
    # 1. Shapley on a toy game: v = log of a product -> each player's own log factor.
    v = np.array([0.0, math.log(2), math.log(3), math.log(6)])
    phi = shapley_from_values(v, 2)
    ok &= np.allclose(np.exp(phi), [2, 3])
    print(f"  toy Shapley (2 x 3): {np.exp(phi).round(6)}: {np.allclose(np.exp(phi), [2, 3])}")
    if not (APP_DATA / "rating_factors.json").exists():
        print("  app/data not built yet (run: explain.py build)")
        return bool(ok)
    ex = load()
    m = ex.model
    # 2. rating tables = production.predict on every level combination (one county per state)
    combos = pd.DataFrame(list(itertools.product(*[ex.t.levels[f] for f in FEATURES])), columns=FEATURES)
    examples = {"VA": "51810", "NC": "37055", "MD": "24047"}
    for st in ex.state_list:
        cc = combos.assign(county=examples[st])
        a = ex.t.rate(cc)["pure_premium"].to_numpy()
        b = m.predict(cc)["pure_premium"].to_numpy()
        good = np.allclose(a, b, rtol=1e-9)
        print(f"  rating tables = production.predict on all {len(combos):,} level combinations ({st}): {good}")
        ok &= good
    # 3. Virginia Beach worksheet through explain_profile (the model card's stored check)
    vb = {"county": "51810", "flood_zone_group": "A/V high risk", "construction_band": "1960-1979",
          "floors_group": "1"}
    r = ex.explain_profile(vb)
    good = abs(r["pure_premium"] - m.card["check_virginia_beach_worksheet"]) < 0.01
    good &= abs(r["worksheet"].attrs["pure_premium"] - r["pure_premium"]) < 1e-6
    print(f"  Virginia Beach profile ${r['pure_premium']:,.2f} (model card check): {good}")
    ok &= good
    wf = r["waterfall"]
    good = np.isclose(wf["after"].iloc[0] * wf["multiplier"].iloc[1:-1].prod(), r["pure_premium"])
    print(f"  waterfall steps multiply to the result: {good}")
    ok &= good
    print("  " + r["sentence"])
    for name in ["Norfolk"] + (["Dare County, NC", "Worcester County, MD"] if len(ex.state_list) > 1 else []):
        e = ex.explain_county(name)
        good = np.isclose(e["mix_effect"] * e["county_factor_effect"], e["ratio_to_state"])
        print(f"  {name}: mix x county factor = ratio to {e['state_name']}: {good}")
        print("  " + e["sentence"])
        ok &= good
    return bool(ok)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_app_data()
        sys.exit(0)
    passed = _self_test()
    print("explain self-test passed:", passed)
    sys.exit(0 if passed else 1)
