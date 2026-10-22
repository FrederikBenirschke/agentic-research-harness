"""Generate worlds/planted5: linear effects that only show after standard preprocessing.

    y = m_t                                  market factor: same for all assets on a date, fat-tailed, unforecastable
      + 0.20 * csrank(x02)                   x02 is t(2) distributed; y depends on its per-date rank
      + 0.20 * clean(x07)                    x07 has 1% corrupted rows (spikes ~100 sd); clipping recovers it
      + 0.20 * x10 / s_t                     x10's per-date scale s_t drifts (0.2 to 5); y depends on its per-date z-score
      + 0.20 * x13 / 0.01                    x13 has sd 0.01; ridge on unscaled features shrinks it to zero
      + t(3) noise
    csrank is scaled to unit variance. Noise features get random scales and some get t(3) tails.

Raw-feature OLS on raw y does badly here; per-date ranks, winsorizing, per-date z-scores, scaling before
ridge and a demeaned or winsorized target each recover part of the oracle.

    python worlds/make_planted5.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from make_planted import N_ASSETS, N_FEAT, N_HOLD, N_TRAIN, ar1

SEED = 5


def csrank(v):
    r = v.argsort(axis=1).argsort(axis=1) + 0.5
    u = r / v.shape[1] - 0.5
    return u / u.std()


def main(out=Path(__file__).with_name("planted5"), seed=SEED, rk=2, wz=7, cz=10, sc=13):
    rng = np.random.default_rng(seed)
    T = N_TRAIN + N_HOLD
    x = ar1(rng, (T, N_ASSETS, N_FEAT), 0.6)
    for i in range(N_FEAT):  # noise features: mixed scales, some fat-tailed
        if i in (rk, wz, cz, sc):
            continue
        if rng.random() < 0.4:
            x[:, :, i] = rng.standard_t(3, size=(T, N_ASSETS))
        x[:, :, i] *= 10 ** rng.uniform(-1, 1.5)
    # rank: t(2) feature, effect through its per-date rank
    x[:, :, rk] = rng.standard_t(2, size=(T, N_ASSETS)) * 3
    f_rank = csrank(x[:, :, rk])
    # winsor: clean feature plus 1% spikes
    clean = x[:, :, wz].copy()
    spikes = rng.random((T, N_ASSETS)) < 0.01
    x[:, :, wz] = clean + spikes * rng.choice([-1, 1], size=(T, N_ASSETS)) * rng.uniform(50, 150, size=(T, N_ASSETS))
    # csz: per-date scale drifts in slow regimes
    s_t = np.exp(np.cumsum(rng.normal(0, 0.15, T)).clip(-1.6, 1.6))[:, None]
    z = x[:, :, cz].copy()
    x[:, :, cz] = z * s_t
    # scale: tiny-sd feature
    x[:, :, sc] *= 0.01
    parts = {"rank": 0.20 * f_rank, "winsor": 0.20 * clean, "csz": 0.20 * z, "scale": 0.20 * x[:, :, sc] / 0.01}
    mu = sum(parts.values())
    market = 0.8 * rng.standard_t(3, size=(T, 1)) * np.ones((1, N_ASSETS))
    y = mu + market + 0.8 * rng.standard_t(3, size=(T, N_ASSETS))
    d, s = np.meshgrid(np.arange(T), np.arange(N_ASSETS), indexing="ij")
    df = pl.DataFrame({"date": d.ravel(), "asset": s.ravel(),
                       **{f"x{i:02d}": x[:, :, i].ravel() for i in range(N_FEAT)}, "y": y.ravel()})
    for sub in ("public", "sealed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("date") < N_TRAIN).write_parquet(out / "public" / "train.parquet")
    df.filter(pl.col("date") >= N_TRAIN).write_parquet(out / "sealed" / "holdout.parquet")
    yh, mh = y[N_TRAIN:].ravel(), mu[N_TRAIN:].ravel()
    oracle = float(1 - np.sum((yh - mh) ** 2) / np.sum((yh - yh.mean()) ** 2))
    fr, fw, fc, fs = (f"x{i:02d}" for i in (rk, wz, cz, sc))
    truth = {
        "mechanisms": [
            {"id": "rank", "features": [fr], "pattern": r"rank|quantile.?transform|percentile|gauss.?rank",
             "text": f"{fr} is t(2) distributed; y rises with its per-date rank (0.20 per unit-variance rank)"},
            {"id": "winsor", "features": [fw], "pattern": r"winsor|clip|outlier|spike|robust|huber",
             "text": f"{fw} has 1% corrupted rows (spikes of 50-150); y rises with the clean value (0.20)"},
            {"id": "csz", "features": [fc],
             "pattern": r"z.?score|standardi[sz]|normali[sz]|over\(.?date|per.?date|cross.?section|scale (drift|change)",
             "text": f"{fc}'s per-date scale drifts from about 0.2 to 5; y rises with its per-date z-score (0.20)"},
            {"id": "scale", "features": [fs], "pattern": r"scal|standardi[sz]|z.?score|normali[sz]",
             "text": f"{fs} has sd 0.01; y = 0.20 * {fs} / 0.01, so ridge on unscaled features shrinks it away"},
        ],
        "decoys": [],
        "noise_features": [f"x{i:02d}" for i in range(N_FEAT) if i not in (rk, wz, cz, sc)],
        "oracle_holdout_r2": oracle,
    }
    (out / "sealed" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "public" / "README.md").write_text(
        f"Panel of {N_ASSETS} assets x {N_TRAIN} dates. Columns: date, asset, x00..x{N_FEAT - 1:02d} (features known "
        "at the close of `date`), y (the asset's return over the NEXT period).\nTask: forecast y from information "
        "available at `date`.\n")
    share = {k: f"{np.var(v) / np.var(y):.3f}" for k, v in parts.items()}
    print(f"wrote {out}, oracle holdout R2 {oracle:.3f}, variance share {share}, market share {np.var(market) / np.var(y):.3f}")


if __name__ == "__main__":
    main()
    main(Path(__file__).with_name("planted5b"), seed=19, rk=11, wz=4, cz=0, sc=8)  # held out
