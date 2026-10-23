"""Generate worlds/planted7: a two-factor panel, in vol-normalised returns, whose predictable part lives only in
the residual.

    y = s_{a,t} * r                        s: each asset's volatility, persistent (log AR(1) with rho 0.98), spread
                                           across assets (about 0.4 to 2.5) and with a common vol-clustering part;
                                           not given, it has to be estimated from y's history
    r = b1_a * f1_t + b2_a * f2_t          two latent factors with per-asset loadings (not given); ~75% of var(r)
      + g_{c(a), t}                        a latent industry factor; the 5 industries are not given
      + mu + u                             u: idiosyncratic t(4) shock
    mu = -0.25 * u[t-1]                    residual reversal: last period's idiosyncratic shock reverses; raw lagged
                                           y barely shows it because the factor part of y does not reverse
       + 0.25 * z(x_a - industry mean of x_a)   x_a predicts the residual relative to its hidden industry only
    x_b = b2 + noise                       decoy: a characteristic that loads on f2; f2 drifts up in the training
                                           dates (fading to zero by their end) and has mean zero in the holdout

Per-date demeaning does not remove the factors (loadings differ by asset), and loadings, correlations and
residuals estimated on raw y are dominated by the high-volatility assets. Estimating each asset's volatility from
history and dividing it out, then estimating loadings (rolling regressions on the date mean, PCA of normalised
returns), clustering residual correlations into industries, and modelling the normalised residual recover the
oracle; the forecast is multiplied back by the volatility estimate.

    python worlds/make_planted7.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from make_planted import N_ASSETS, N_FEAT, N_HOLD, N_TRAIN, ar1

SEED = 7
N_IND = 5


def main(out=Path(__file__).with_name("planted7"), seed=SEED, fa=6, fb=10, b1_sd=0.4, f2_sd=0.9, g_sd=0.6, xa_ind=0.0):
    rng = np.random.default_rng(seed)
    T = N_TRAIN + N_HOLD
    x = ar1(rng, (T, N_ASSETS, N_FEAT), 0.6)
    for i in range(N_FEAT):
        if i not in (fa, fb):
            x[:, :, i] *= 10 ** rng.uniform(-1, 1)
    b1 = 1 + b1_sd * rng.normal(size=N_ASSETS)
    b2 = rng.normal(size=N_ASSETS)
    ind = rng.permutation(np.arange(N_ASSETS) % N_IND)
    f1 = 1.3 * rng.standard_t(4, size=T) / np.sqrt(2)
    drift = np.where(np.arange(T) < N_TRAIN, 0.3 * (1 - np.arange(T) / N_TRAIN), 0.0)
    f2 = drift + f2_sd * rng.normal(size=T)
    g = g_sd * rng.normal(size=(T, N_IND))[:, ind]
    u = rng.standard_t(4, size=(T, N_ASSETS)) / np.sqrt(2)
    xa = x[:, :, fa]
    rel = xa - np.stack([xa[:, ind == c].mean(axis=1) for c in range(N_IND)], axis=1)[:, ind]
    rel /= rel.std()
    if xa_ind:  # an industry-level part of x_a that does not predict: x_a works only relative to its industry
        level = ar1(np.random.default_rng(seed + 1000), (T, N_IND), 0.95)[:, ind]
        x[:, :, fa] = xa + xa_ind * level / level.std() * xa.std()
    u_lag = np.vstack([np.zeros((1, N_ASSETS)), u[:-1]])
    common = np.zeros(T)                                    # vol clustering shared by all assets
    own = np.zeros((T, N_ASSETS))
    for d in range(1, T):
        common[d] = 0.97 * common[d - 1] + 0.3 * np.sqrt(1 - 0.97 ** 2) * rng.normal()
        own[d] = 0.98 * own[d - 1] + 0.35 * np.sqrt(1 - 0.98 ** 2) * rng.normal(size=N_ASSETS)
    s = np.exp(0.45 * rng.normal(size=N_ASSETS)[None] + common[:, None] + own)
    s /= np.median(s)
    parts = {"reversal": -0.25 * u_lag, "industry_rel": 0.25 * rel}
    parts = {k: s * v for k, v in parts.items()}
    mu = sum(parts.values())
    factors = s * (b1[None] * f1[:, None] + b2[None] * (f2 - drift)[:, None] + g)
    y = factors + s * (b2[None] * drift[:, None] + u) + mu
    x[:, :, fb] = b2[None] + 0.5 * rng.normal(size=(T, N_ASSETS))
    d_, a_ = np.meshgrid(np.arange(T), np.arange(N_ASSETS), indexing="ij")
    df = pl.DataFrame({"date": d_.ravel(), "asset": a_.ravel(),
                       **{f"x{i:02d}": x[:, :, i].ravel() for i in range(N_FEAT)}, "y": y.ravel()})
    for sub in ("public", "sealed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("date") < N_TRAIN).write_parquet(out / "public" / "train.parquet")
    df.filter(pl.col("date") >= N_TRAIN).write_parquet(out / "sealed" / "holdout.parquet")
    yh, mh = y[N_TRAIN:].ravel(), mu[N_TRAIN:].ravel()
    oracle = float(1 - np.sum((yh - mh) ** 2) / np.sum((yh - yh.mean()) ** 2))
    fA, fB = f"x{fa:02d}", f"x{fb:02d}"
    truth = {
        "mechanisms": [
            {"id": "reversal", "features": [],
             "pattern": r"residual.{0,40}(reversal|revers|mean.?revert|negative.{0,20}lag)|idiosyncratic.{0,40}revers|"
                        r"(factor|beta|pca|principal).{0,80}revers",
             "text": "last period's idiosyncratic return (after removing two latent factors with per-asset loadings "
                     "and a latent industry factor, all in returns divided by each asset's volatility) reverses: "
                     "-0.25 times it, times the asset's current volatility"},
            {"id": "industry_rel", "features": [fA],
             "pattern": r"industr|cluster|group|sector|peer|relative to",
             "text": f"{fA} predicts y only relative to the mean of {fA} within the asset's latent industry "
                     f"(5 industries found by clustering residual correlations), 0.25 per unit of vol-normalised return"},
        ],
        "decoys": [{"feature": fB, "text": f"{fB} is a noisy copy of the assets' loading on factor 2, whose mean is "
                                          "positive early in the training dates and zero in the holdout"}],
        "noise_features": [f"x{i:02d}" for i in range(N_FEAT) if i not in (fa, fb)],
        "oracle_holdout_r2": oracle,
    }
    (out / "sealed" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "public" / "README.md").write_text(
        f"Panel of {N_ASSETS} assets x {N_TRAIN} dates. Columns: date, asset, x00..x{N_FEAT - 1:02d} (features known "
        "at the close of `date`), y (the asset's return over the NEXT period).\nTask: forecast y from information "
        "available at `date`.\n")
    share = {k: f"{np.var(v) / np.var(y):.4f}" for k, v in parts.items()}
    print(f"wrote {out}, oracle holdout R2 {oracle:.4f}, variance share {share}, "
          f"factor share {np.var(factors) / np.var(y):.3f}, vol range {np.quantile(s, [0.05, 0.5, 0.95]).round(2)}")


if __name__ == "__main__":
    main()
    main(Path(__file__).with_name("planted7b"), seed=29, fa=13, fb=2)  # held out
    # planted8: each step pays (loadings spread, larger factor 2 and industry factor, x_a with an industry level)
    main(Path(__file__).with_name("planted8"), seed=8, fa=4, fb=15, b1_sd=0.8, f2_sd=1.5, g_sd=1.0, xa_ind=1.5)
    main(Path(__file__).with_name("planted8b"), seed=31, fa=12, fb=9, b1_sd=0.8, f2_sd=1.5, g_sd=1.0, xa_ind=1.5)
    # planted8c: planted8 with a stronger industry factor, so the industry eigenvalues (~4) clear the noise edge (~1.7)
    main(Path(__file__).with_name("planted8c"), seed=8, fa=4, fb=15, b1_sd=0.8, f2_sd=1.5, g_sd=2.2, xa_ind=1.5)
