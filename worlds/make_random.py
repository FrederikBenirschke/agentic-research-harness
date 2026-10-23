"""Procedural random worlds: each seed draws its own mechanisms, decoys, noise level and noise features.

    python worlds/make_random.py SEED [SEED ...]

writes worlds/rand/r<SEED>/{public,sealed} and evals/rand/r<SEED>.json. The panel format, the public/sealed
split and the truth.json schema are those of the planted worlds (make_planted4.py, make_planted5.py).

Seed convention
---------------
    seeds 0..999        DEV   use these to tune prompts and agents; reading their truth.json is allowed
    seeds >= 100000     EXAM  never tune on these and never read their truth.json while tuning; they are
                              scored once, after tuning is frozen
    seeds 1000..99999   unassigned
The draw depends only on the seed, so a world can be regenerated exactly.

What a seed draws
-----------------
- 0-5 mechanisms (P(0) = 0.1: a null world with no forecastable mean), families drawn without replacement
  from the library below, each with random feature indices, sign, strength and shape parameters;
- 0-2 decoys (a proxy of a true signal, a near-duplicate of a true feature, or a feature that leaks the
  target in the training dates only);
- a target oracle R2, log-uniform on [0.005, 0.3], which sets the noise level; the noise is split into an
  unforecastable market shock and idiosyncratic noise with random tails;
- 16-24 features; the unused ones are noise with random persistence, scales (some tiny or huge), t(3)
  tails, and sometimes camouflage that mimics a mechanism's footprint (an integer code column, a
  date-constant column, random missingness) so that the footprint alone does not give a mechanism away.
Each feature is also shown through a random positive affine map (loc, scale); the truth text writes the
formula on the underlying standardized value, e.g. (x05 - 2.1)/0.3.

Mechanism library (families)
----------------------------
linear, rank (per-date rank of a fat-tailed feature), csz (per-date z-score under scale drift), winsor
(outlier-corrupted feature), scale (tiny or huge sd), hinge, step, tanh (monotone saturating), sine, bump
(bump or U-shape), product (interaction), gated (interaction with a date-constant feature), regime (latent
Markov regime or a structural break in a coefficient), decay (coefficient fading over time), lag_y
(momentum or reversal in the asset's own y), leadlag (one leader asset's lagged y moves the others),
sector (sector-relative effect, with a sector code column), missing (informative missingness), publag
(feature published with a lag: today's value moves y one period later), hetero (heteroskedastic noise
driven by a feature; no effect on the mean), market (date-constant feature that times the common
component).
Decoys: proxy, dup, leak.

truth.json records per mechanism id, family, features, a family-generic regex pattern for the grader,
text, share (variance share of y) and params; decoys; noise_features; oracle_holdout_r2 (the R2 of the true
conditional mean on the holdout); and baselines fit on train, scored on the holdout: ridge on standardized
raw features (alpha 1, NaN set to 0 after standardizing), HistGradientBoostingRegressor with default params,
and zero.
"""

import json
import os
import re
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np
import polars as pl

from make_planted import N_ASSETS, N_HOLD, N_TRAIN, ar1

ROOT = Path(__file__).resolve().parent.parent
T, N = N_TRAIN + N_HOLD, N_ASSETS
EXAM_MIN, DEV_MAX = 100_000, 999

PATTERNS = {
    "linear": r"linear|ridge|\bols\b|regress|coef|slope|beta|proportional|monoton|correlat",
    "rank": r"rank|quantile.?transform|percentile|gauss.?rank",
    "csz": r"z.?score|standardi[sz]|normali[sz]|over\(.?date|per.?date|cross.?section|scale (drift|change)",
    "winsor": r"winsor|clip|outlier|spike|robust|huber",
    "scale": r"scal|standardi[sz]|z.?score|normali[sz]",
    "hinge": r"threshold|hinge|kink|max\(|min\(|relu|piecewise|one.?sided|asymmetr|only (when|for|above|below)|breakpoint|cutoff",
    "step": r"threshold|step|jump|indicator|binary|discontinu|cutoff|above|below|piecewise",
    "tanh": r"tanh|saturat|plateau|flatten|diminishing|concave|sigmoid|logistic|levels? off|capped|clip",
    "sine": r"\bsin\b|\bsine\b|\bcos\b|cosin|oscillat|periodic|\bwave|several (peaks|turns)|rises?,? (then )?falls?,? (and |then )?rises?",
    "bump": r"bump|hump|peak|inverted.?u|u.?shape|non.?monoton|quadratic|squar|\*\* ?2|\^2|abs\(|absolute|extreme|gaussian",
    "product": r"interact|product|multipl|cross.?term|x\d\d\s*[*×·]\s*x\d\d|same sign|jointly",
    "gated": r"gate|gating|regime|only (when|on dates|if)|condition|interact|switch|state|depends on",
    "regime": r"regime|break|switch|unstable|time.?vary|flip|rolling|sub.?period|non.?stationar|changes? (sign|over time)",
    "decay": r"decay|fad|weaken|shrink|declin|diminish|time.?vary|rolling|recent|half.?life|non.?stationar|over time",
    "lag_y": r"momentum|revers|autocorr|mean.?revert|lag",
    "leadlag": r"lead|cross.?asset|other assets?|leader|spill.?over",
    "sector": r"sector|industry|group|cluster|categor|relative to|demean|within",
    "missing": r"missing|\bnan\b|null|impute|absent|indicator",
    "publag": r"lag|shift|delay|previous|prior|yesterday|stale|t-1|one period",
    "hetero": r"heterosk|heterosc|variance|volatil|noise (scale|level)|weighted|\bwls\b|dispersion",
    "market": r"market|date.?level|date.?constant|common|aggregate|time.?series|timing|all assets|same for all",
}
N_FEATS = {"product": 2, "gated": 2, "sector": 2}
FAMILIES = list(PATTERNS)
LAGGED = ("lag_y", "leadlag")


def csrank(v):
    r = v.argsort(axis=1).argsort(axis=1) + 0.5
    return r / v.shape[1] - 0.5


def off(k):
    return f"- {k:.2f}" if k >= 0 else f"+ {-k:.2f}"


def unit(a):
    a = a - np.nanmean(a)
    return a / np.nanstd(a)


class World:
    def __init__(self, seed):
        self.seed, self.rng = seed, np.random.default_rng(seed)

    def name(self, f):
        return f"x{f:02d}"

    def ref(self, f):
        loc, sc = self.disp[f]
        n = self.name(f)
        body = n if loc == 0 else f"{n} - {loc:g}" if loc > 0 else f"{n} + {-loc:g}"
        if sc == 1:
            return body if loc == 0 else f"({body})"
        return f"({body})/{sc:g}" if loc else f"{n}/{sc:g}"

    def latent(self, rho=None):
        rho = self.rng.uniform(0, 0.9) if rho is None else rho
        return ar1(self.rng, (T, N), rho)

    def date_level(self):
        return ar1(self.rng, (T, 1), self.rng.uniform(0.9, 0.99)) * np.ones((1, N))

    # each builder returns g (the unnormalized mean contribution, or None) and text(c), the formula with the
    # final coefficient c on g; it may overwrite self.z / self.raw / self.disp for its features
    def b_linear(self, f):
        (a,) = f
        return self.z[:, :, a], lambda c: f"linear: y += {c:+.4g} * {self.ref(a)}"

    def b_rank(self, f):
        (a,) = f
        df = self.rng.choice([1.5, 2, 3])
        self.z[:, :, a] = self.rng.standard_t(df, size=(T, N))
        return csrank(self.z[:, :, a]), lambda c: (
            f"per-date rank: {self.name(a)} is t({df:g}) distributed; y += {c:+.4g} * (per-date rank of "
            f"{self.name(a)} scaled to (-0.5, 0.5))")

    def b_csz(self, f):
        (a,) = f
        vol = self.rng.uniform(0.1, 0.2)
        s = np.exp(np.cumsum(self.rng.normal(0, vol, T)).clip(-1.6, 1.6))[:, None]
        g = self.z[:, :, a].copy()
        self.z[:, :, a] = g * s
        return g, lambda c: (f"per-date z-score: the per-date scale of {self.name(a)} drifts (about "
                             f"{s.min():.2g} to {s.max():.2g}); y += {c:+.4g} * per-date z-score of {self.name(a)}")

    def b_winsor(self, f):
        (a,) = f
        p = self.rng.uniform(0.005, 0.02)
        g = self.z[:, :, a].copy()
        spikes = self.rng.random((T, N)) < p
        self.z[:, :, a] = g + spikes * self.rng.choice([-1, 1], size=(T, N)) * self.rng.uniform(50, 150, (T, N))
        return g, lambda c: (f"outlier-corrupted: {p:.1%} of {self.name(a)}'s rows carry spikes of 50-150 sd; "
                             f"y += {c:+.4g} * the clean value of {self.ref(a)} (clip/winsorize to recover it)")

    def b_scale(self, f):
        (a,) = f
        sc = 10 ** (self.rng.uniform(-3, -2) if self.rng.random() < 0.5 else self.rng.uniform(2.5, 4))
        self.disp[a] = (0, sc)
        return self.z[:, :, a], lambda c: (f"scale: {self.name(a)} has sd about {sc:.3g}; y += {c:+.4g} * "
                                           f"{self.ref(a)}, so a fit on unscaled features mis-weights it")

    def b_hinge(self, f):
        (a,) = f
        k, up = self.rng.uniform(-1, 1), self.rng.random() < 0.5
        z = self.z[:, :, a]
        g = np.maximum(z - k, 0) if up else np.maximum(k - z, 0)
        side = "above" if up else "below"
        expr = f"max({self.ref(a)} {off(k)}, 0)" if up else f"max({k:.2f} - {self.ref(a)}, 0)"
        return g, lambda c: f"hinge: y += {c:+.4g} * {expr}; {self.name(a)} matters only {side} that threshold"

    def b_step(self, f):
        (a,) = f
        k = self.rng.uniform(-1.2, 1.2)
        return (self.z[:, :, a] > k).astype(float), lambda c: (
            f"step: y += {c:+.4g} * [{self.ref(a)} > {k:.2f}], a jump at the threshold, flat on either side")

    def b_tanh(self, f):
        (a,) = f
        k = self.rng.uniform(1.5, 4)
        return np.tanh(k * self.z[:, :, a]), lambda c: (
            f"tanh: y += {c:+.4g} * tanh({k:.2f} * {self.ref(a)}), monotone and saturating beyond about +-{1.5 / k:.2f}")

    def b_sine(self, f):
        (a,) = f
        w, ph = self.rng.uniform(2, 4), self.rng.uniform(0, 2 * np.pi)
        return np.sin(w * self.z[:, :, a] + ph), lambda c: (
            f"sine: y += {c:+.4g} * sin({w:.2f} * {self.ref(a)} + {ph:.2f}), oscillating over {self.name(a)}'s range")

    def b_bump(self, f):
        (a,) = f
        m, s = self.rng.uniform(-0.8, 0.8), self.rng.uniform(0.4, 0.9)
        return np.exp(-((self.z[:, :, a] - m) ** 2) / (2 * s * s)), lambda c: (
            f"bump: y += {c:+.4g} * exp(-({self.ref(a)} {off(m)})^2 / (2*{s:.2f}^2)), non-monotone "
            f"({'a peak' if c > 0 else 'a U-shape'} near {m:.2f})")

    def b_product(self, f):
        a, b = f
        return self.z[:, :, a] * self.z[:, :, b], lambda c: (
            f"product interaction: y += {c:+.4g} * {self.ref(a)} * {self.ref(b)}; neither predicts y alone")

    def b_gated(self, f):
        a, gf = f
        self.z[:, :, gf] = self.date_level()
        k, up = self.rng.uniform(-0.5, 0.5), self.rng.random() < 0.5
        gate = self.z[:, :, gf] > k if up else self.z[:, :, gf] < k
        return self.z[:, :, a] * gate, lambda c: (
            f"gated: y += {c:+.4g} * {self.ref(a)} only on dates when the date-constant {self.ref(gf)} is "
            f"{'above' if up else 'below'} {k:.2f} (gate open {gate[:, 0].mean():.0%} of dates)")

    def b_regime(self, f):
        (a,) = f
        lo = self.rng.uniform(-1, 0.3)
        if self.rng.random() < 0.5:
            stay = self.rng.uniform(0.97, 0.995)
            st = np.empty(T, bool)
            st[0] = True
            for t in range(1, T):
                st[t] = st[t - 1] if self.rng.random() < stay else not st[t - 1]
            coef = np.where(st, 1.0, lo)
            how = f"a latent Markov regime (stay prob {stay:.3f}, not observed): 1 in state A, {lo:.2f} in state B"
        else:
            br = int(self.rng.integers(150, 450))
            coef = np.where(np.arange(T) < br, lo, 1.0)
            how = f"a structural break at date {br}: {lo:.2f} before, 1 after"
        return self.z[:, :, a] * coef[:, None], lambda c: (
            f"regime switch: y += {c:+.4g} * b_t * {self.ref(a)}, where b_t follows {how}")

    def b_decay(self, f):
        (a,) = f
        tau, floor = self.rng.uniform(T / 4, T), self.rng.uniform(0, 0.4)
        coef = floor + (1 - floor) * np.exp(-np.arange(T) / tau)
        return self.z[:, :, a] * coef[:, None], lambda c: (
            f"decaying coefficient: y += {c:+.4g} * b_t * {self.ref(a)}, b_t = {floor:.2f} + {1 - floor:.2f} * "
            f"exp(-date/{tau:.0f}): the effect fades over time")

    def b_sector(self, f):
        a, sf = f
        k = int(self.rng.integers(4, 9))
        sec = self.rng.integers(0, k, N)
        sec[:k] = np.arange(k)
        self.sector_col(sf, sec)
        share = self.rng.uniform(0.5, 0.85)
        common = np.stack([self.latent(0.9) for _ in range(k)], axis=-1)[:, 0, :][:, sec]  # (T, N) per-sector
        e = self.z[:, :, a]
        self.z[:, :, a] = np.sqrt(share) * common + np.sqrt(1 - share) * e
        x = self.z[:, :, a]
        g = x - np.stack([x[:, sec == j].mean(axis=1) for j in range(k)], axis=1)[:, sec]
        return g, lambda c: (
            f"sector-relative: {self.name(sf)} is a sector code ({k} sectors); y += {c:+.4g} * ({self.ref(a)} minus "
            f"its per-date sector mean); {share:.0%} of {self.name(a)}'s variance is a sector-date common part")

    def sector_col(self, sf, sec):
        self.z[:, :, sf] = np.broadcast_to(sec.astype(float), (T, N))
        self.disp[sf] = (0, 1)

    def b_missing(self, f):
        (a,) = f
        p = self.rng.uniform(0.1, 0.35)
        z = self.z[:, :, a].copy()
        if self.rng.random() < 0.5:
            m = self.rng.random((T, N)) < p
            self.z[:, :, a] = np.where(m, np.nan, z)
            return m.astype(float), lambda c: (
                f"informative missingness: {self.name(a)} is missing on {p:.0%} of rows at random; y += {c:+.4g} * "
                f"[{self.name(a)} is missing]; the observed values carry no signal")
        q = np.quantile(z, 1 - p)
        self.z[:, :, a] = np.where(z > q, np.nan, z)
        return z, lambda c: (
            f"informative missingness: {self.name(a)} is missing exactly when its latent value is in its top {p:.0%}; "
            f"y += {c:+.4g} * latent {self.ref(a)}, so a missing indicator carries the high values")

    def b_publag(self, f):
        (a,) = f
        self.z[:, :, a] = self.latent(self.rng.uniform(0, 0.3))
        g = np.zeros((T, N))
        g[1:] = self.z[:-1, :, a]
        return g, lambda c: (
            f"published with a lag: y at date t += {c:+.4g} * {self.ref(a)} at date t-1 (the previous row); "
            f"today's value has little contemporaneous relation to y")

    def b_market(self, f):
        (a,) = f
        self.z[:, :, a] = self.date_level()
        return self.z[:, :, a], lambda c: (
            f"market timing: the date-constant {self.ref(a)} moves y of all assets alike: y += {c:+.4g} * {self.ref(a)}")

    def b_hetero(self, f):
        (a,) = f
        k = self.rng.uniform(0.3, 0.8)
        self.hetero = np.exp(k * self.z[:, :, a])
        self.hetero /= np.sqrt(np.mean(self.hetero ** 2))
        return None, lambda c: (
            f"heteroskedastic noise: the idiosyncratic noise sd is proportional to exp({k:.2f} * {self.ref(a)}); "
            "no effect on the mean of y")

    def b_lag_y(self, f):
        self.lag = int(self.rng.choice([1, 1, 2]))
        return "lag", lambda c: (
            f"lagged own y: y_t += {c:+.4g} * y_(t-{self.lag}) of the same asset "
            f"({'momentum' if c > 0 else 'reversal'})")

    def b_leadlag(self, f):
        self.leader = int(self.rng.integers(0, N))
        return "lead", lambda c: (
            f"lead-lag: asset {self.leader} is a leader; every other asset's y_t += {c:+.4g} * y_(t-1) of asset "
            f"{self.leader} (cross-asset)")

    def build(self):
        rng = self.rng
        n_mech = 0 if rng.random() < 0.1 else int(rng.integers(1, 6))
        fams = list(rng.choice(FAMILIES, n_mech, replace=False))
        n_dec = int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.2]))
        camo = [k for k, p in (("sector", 0.35), ("date", 0.4), ("mcar", 0.3)) if rng.random() < p]
        need = sum(N_FEATS.get(m, 0 if m in LAGGED else 1) for m in fams) + n_dec + len(camo)
        F = max(int(rng.integers(16, 25)), need + 4)
        free = list(rng.permutation(F))
        self.F = F
        self.z = np.stack([self.latent() for _ in range(F)], axis=-1)
        self.disp = {}
        for f in range(F):
            sc = 10 ** rng.uniform(-1, 1) if rng.random() < 0.5 else 1.0
            loc = 0.0 if rng.random() < 0.7 else float(rng.uniform(-3, 3)) * sc
            self.disp[f] = (round(loc, 3), float(f"{sc:.3g}"))
        self.hetero, self.lag, self.leader = None, None, None

        r2_target = float(np.exp(rng.uniform(np.log(0.005), np.log(0.3))))
        mechs = []
        for fam in fams:
            feats = [int(free.pop()) for _ in range(N_FEATS.get(fam, 0 if fam in LAGGED else 1))]
            g, text = getattr(self, f"b_{fam}")(feats)
            mechs.append({"family": fam, "feats": feats, "g": g, "text": text})
        mean_m = [m for m in mechs if m["g"] is not None]
        w = rng.dirichlet(np.full(len(mean_m), 2.0)) if mean_m else []
        static = np.zeros((T, N))
        for m, wk in zip(mean_m, w):
            amp = np.sqrt(wk * r2_target) * rng.choice([-1, 1])
            if isinstance(m["g"], str):          # lagged y: amp is the coefficient on a roughly unit-variance y
                m["coef"] = float(np.clip(amp, -0.4, 0.4))
            else:
                sd = np.std(m["g"])
                m["coef"] = amp / sd
                m["part"] = m["coef"] * (m["g"] - np.mean(m["g"]))
                static += m["part"]

        # noise: market shock + idiosyncratic with random tails, total variance 1 - r2_target
        nv = 1 - r2_target
        mkt_frac = 0.0 if rng.random() < 0.4 else rng.uniform(0.1, 0.6)
        df = rng.choice([np.inf, 3, 4, 6])
        idio = rng.normal(size=(T, N)) if np.isinf(df) else rng.standard_t(df, (T, N)) * np.sqrt((df - 2) / df)
        if self.hetero is not None:
            idio = idio * self.hetero
        mdf = rng.choice([np.inf, 3])
        mk = rng.normal(size=(T, 1)) if np.isinf(mdf) else rng.standard_t(mdf, (T, 1)) / np.sqrt(3)
        eps = np.sqrt(nv * (1 - mkt_frac)) * idio + np.sqrt(nv * mkt_frac) * mk

        lagm = next((m for m in mechs if m["family"] == "lag_y"), None)
        leadm = next((m for m in mechs if m["family"] == "leadlag"), None)
        y, mu = np.zeros((T, N)), static.copy()
        lagpart, leadpart = np.zeros((T, N)), np.zeros((T, N))
        others = np.arange(N) != (self.leader if self.leader is not None else -1)
        for t in range(T):
            if lagm and t >= self.lag:
                lagpart[t] = lagm["coef"] * y[t - self.lag]
            if leadm and t >= 1:
                leadpart[t] = leadm["coef"] * y[t - 1, self.leader] * others
            mu[t] = static[t] + lagpart[t] + leadpart[t]
            y[t] = mu[t] + eps[t]
        if lagm:
            lagm["part"] = lagpart
        if leadm:
            leadm["part"] = leadpart

        yscale = float(f"{10 ** rng.uniform(-2.3, 0):.3g}")
        y, mu = y * yscale, mu * yscale

        # decoys, built after y
        dec = []
        kinds = []
        for _ in range(n_dec):
            opts = ["leak"]
            if any(not isinstance(m["g"], str) and m["g"] is not None for m in mechs):
                opts += ["proxy", "proxy"]
            if any(m["feats"] for m in mechs if m["family"] not in ("sector", "missing")):
                opts += ["dup"]
            kinds.append(rng.choice(opts))
        for kind in kinds:
            d = int(free.pop())
            dn = self.name(d)
            if kind == "proxy":
                m = mechs[rng.choice([i for i, m in enumerate(mechs) if m["g"] is not None and not isinstance(m["g"], str)])]
                noise = rng.uniform(0.6, 1.5)
                self.z[:, :, d] = unit(m["part"]) + noise * rng.normal(size=(T, N))
                txt = (f"{dn} is a noisy proxy of the {m['family']} mechanism's contribution (noise sd {noise:.2f} per "
                       f"unit signal): correlated with y only through it")
                dec.append({"feature": dn, "family": "proxy", "of": m["family"], "text": txt})
            elif kind == "dup":
                m = mechs[rng.choice([i for i, m in enumerate(mechs) if m["feats"] and m["family"] not in ("sector", "missing")])]
                src = m["feats"][0]
                noise = rng.uniform(0.05, 0.25)
                self.z[:, :, d] = self.z[:, :, src] + noise * np.nanstd(self.z[:, :, src]) * rng.normal(size=(T, N))
                self.disp[d] = self.disp[src]
                txt = (f"{dn} is a near-duplicate of {self.name(src)} (noise {noise:.2f} sd): it has no effect of its own")
                dec.append({"feature": dn, "family": "dup", "of": self.name(src), "text": txt})
            else:
                rho = rng.uniform(0.05, 0.25)
                v = rng.normal(size=(T, N))
                v[:N_TRAIN] = rho * unit(y[:N_TRAIN]) + np.sqrt(1 - rho ** 2) * v[:N_TRAIN]
                self.z[:, :, d] = v
                txt = (f"{dn} leaks the target in the training dates only (corr {rho:.2f} with same-date y); "
                       f"in the holdout it is pure noise: a trap")
                dec.append({"feature": dn, "family": "leak", "text": txt})

        # noise features and camouflage
        noise_feats = [int(f) for f in free]
        for f in noise_feats:
            if rng.random() < 0.4:
                self.z[:, :, f] = rng.standard_t(3, (T, N))
            if rng.random() < 0.15:
                self.disp[f] = (0, float(f"{10 ** (rng.uniform(-3, -2) if rng.random() < 0.5 else rng.uniform(2.5, 4)):.3g}"))
        pool = list(noise_feats)
        rng.shuffle(pool)
        for c in camo:
            if not pool:
                break
            f = pool.pop()
            if c == "sector":
                k = int(rng.integers(4, 9))
                sec = rng.integers(0, k, N)
                sec[:k] = np.arange(k)
                self.sector_col(f, sec)
            elif c == "date":
                self.z[:, :, f] = self.date_level()
            else:
                self.z[:, :, f] = np.where(rng.random((T, N)) < rng.uniform(0.05, 0.3), np.nan, self.z[:, :, f])

        x = np.stack([self.disp[f][0] + self.disp[f][1] * self.z[:, :, f] for f in range(F)], axis=-1)
        self.x, self.y, self.mu = x, y, mu
        self.mechs, self.decoys, self.noise_feats = mechs, dec, noise_feats
        self.r2_target, self.yscale = r2_target, yscale
        self.noise = {"market_share_of_noise": round(float(mkt_frac), 3),
                      "idio_tail_df": None if np.isinf(df) else int(df),
                      "market_tail_df": None if np.isinf(mdf) else int(mdf), "y_scale": yscale}
        for m in mechs:
            if "coef" in m and not isinstance(m["g"], str):
                m["coef_text"] = m["coef"] * yscale
            else:
                m["coef_text"] = m.get("coef", 0.0)


def baselines(train, hold, feats):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    Xtr, Xho = train.select(feats).to_numpy().astype(float), hold.select(feats).to_numpy().astype(float)
    ytr, yho = train["y"].to_numpy(), hold["y"].to_numpy()
    r2 = lambda yh: float(1 - np.sum((yho - yh) ** 2) / np.sum((yho - yho.mean()) ** 2))
    mean, sd = np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0)
    sd[sd == 0] = 1
    std = lambda X: np.nan_to_num((X - mean) / sd)
    ridge = Ridge(alpha=1.0).fit(std(Xtr), ytr)
    hgb = HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    return {"ridge": r2(ridge.predict(std(Xho))), "hgb": r2(hgb.predict(Xho)), "zero": r2(np.zeros_like(yho))}


def main(seed):
    w = World(seed)
    w.build()
    out = ROOT / "worlds" / "rand" / f"r{seed}"
    names = [w.name(f) for f in range(w.F)]
    d, s = np.meshgrid(np.arange(T), np.arange(N), indexing="ij")
    df = pl.DataFrame({"date": d.ravel(), "asset": s.ravel(),
                       **{n: w.x[:, :, i].ravel() for i, n in enumerate(names)}, "y": w.y.ravel()}).fill_nan(None)
    for sub in ("public", "sealed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    train, hold = df.filter(pl.col("date") < N_TRAIN), df.filter(pl.col("date") >= N_TRAIN)
    train.write_parquet(out / "public" / "train.parquet")
    hold.write_parquet(out / "sealed" / "holdout.parquet")
    yh, mh = w.y[N_TRAIN:].ravel(), w.mu[N_TRAIN:].ravel()
    oracle = float(1 - np.sum((yh - mh) ** 2) / np.sum((yh - yh.mean()) ** 2))
    vy = np.var(w.y)
    mechanisms, used = [], set()
    for i, m in enumerate(w.mechs):
        feats = [w.name(f) for f in m["feats"]] if m["family"] not in LAGGED else ["y"]
        used |= set(m["feats"])
        part = m.get("part")
        share = 0.0 if part is None else float(np.var(part * w.yscale) / vy)
        text = m["text"](m["coef_text"])
        mechanisms.append({"id": f"{m['family']}_{i}", "family": m["family"], "features": feats,
                           "pattern": PATTERNS[m["family"]], "text": text, "share": round(share, 4)})
    truth = {
        "seed": seed, "split": "exam" if seed >= EXAM_MIN else "dev" if seed <= DEV_MAX else "unassigned",
        "mechanisms": mechanisms,
        "decoys": w.decoys,
        "noise_features": [w.name(f) for f in sorted(w.noise_feats)],
        "noise": w.noise,
        "target_r2": round(w.r2_target, 4),
        "oracle_holdout_r2": oracle,
        "baselines": baselines(train, hold, names),
    }
    for m in mechanisms:     # self-check: the grader would credit the truth's own text
        re.compile(m["pattern"])
        if not re.search(m["pattern"], m["text"].lower()):
            print(f"warning: seed {seed} {m['id']}: pattern does not match its own text", file=sys.stderr)
    (out / "sealed" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "public" / "README.md").write_text(
        f"Panel of {N_ASSETS} assets x {N_TRAIN} dates. Columns: date, asset, x00..x{w.F - 1:02d} (features known "
        "at the close of `date`), y (the asset's return over the NEXT period).\nTask: forecast y from information "
        "available at `date`.\n")
    spec = {"world": f"../worlds/rand/r{seed}",     # evals.py resolves spec["world"] against evals/
            "conditions": {"analyst-open": {"agent": "analyst", "about": "how the features relate to y", "notes": []},
                           "forecaster-open": {"agent": "forecaster", "notes": []}}}
    (ROOT / "evals" / "rand").mkdir(parents=True, exist_ok=True)
    (ROOT / "evals" / "rand" / f"r{seed}.json").write_text(json.dumps(spec, indent=1) + "\n")
    b = truth["baselines"]
    # exam seeds print no mechanism count, so running the generator does not reveal the draw
    head = f"n_mech {len(mechanisms)}, " if truth["split"] != "exam" else ""
    print(f"wrote {out} ({truth['split']}): {head}oracle {oracle:.4f}, ridge {b['ridge']:.4f}, hgb {b['hgb']:.4f}, "
          f"zero {b['zero']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        main(int(a))
