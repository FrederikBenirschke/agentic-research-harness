"""Generate worlds/planted1: a panel with known mechanisms, for grading agents.

    y = 0.35 * rank_z(x03)               cross-sectional rank effect
      + 0.45 * x07 * [x11 > 0]           x07 acts only when date-constant x11 is positive
      + 0.15 * y_lag1                    per-asset momentum
      + noise
    x05 = x03 + noise                    decoy: correlated with x03, no effect of its own

The truth (and the oracle R2 of the true conditional mean on the holdout) is
written to sealed/truth.json, which agents cannot read.

    python worlds/make_planted.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

N_ASSETS, N_TRAIN, N_HOLD, N_FEAT, SEED = 40, 500, 200, 16, 1


def ar1(rng, shape, rho):
    out = np.empty(shape)
    out[0] = rng.normal(size=shape[1:])
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + np.sqrt(1 - rho**2) * rng.normal(size=shape[1:])
    return out


def rank_z(a):
    r = a.argsort(axis=1).argsort(axis=1)
    return (r - r.mean(axis=1, keepdims=True)) / r.std(axis=1, keepdims=True)


def main(out=Path(__file__).with_name("planted1"), seed=SEED):
    rng = np.random.default_rng(seed)
    T = N_TRAIN + N_HOLD
    x = ar1(rng, (T, N_ASSETS, N_FEAT), 0.6)
    x[:, :, 11] = ar1(rng, (T, 1), 0.95)                    # date-constant regime
    x[:, :, 5] = x[:, :, 3] + 0.6 * rng.normal(size=(T, N_ASSETS))
    y, mu = np.zeros((T, N_ASSETS)), np.zeros((T, N_ASSETS))
    for t in range(T):
        mu[t] = (0.35 * rank_z(x[t:t + 1, :, 3])[0] + 0.45 * x[t, :, 7] * (x[t, 0, 11] > 0)
                 + (0.15 * y[t - 1] if t else 0))
        y[t] = mu[t] + rng.normal(size=N_ASSETS)
    d, a = np.meshgrid(np.arange(T), np.arange(N_ASSETS), indexing="ij")
    df = pl.DataFrame({"date": d.ravel(), "asset": a.ravel(),
                       **{f"x{i:02d}": x[:, :, i].ravel() for i in range(N_FEAT)}, "y": y.ravel()})
    for sub in ("public", "sealed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("date") < N_TRAIN).write_parquet(out / "public" / "train.parquet")
    df.filter(pl.col("date") >= N_TRAIN).write_parquet(out / "sealed" / "holdout.parquet")
    yh, mh = y[N_TRAIN:].ravel(), mu[N_TRAIN:].ravel()
    oracle = float(1 - np.sum((yh - mh) ** 2) / np.sum((yh - yh.mean()) ** 2))
    truth = {
        "mechanisms": [
            {"id": "rank_x03", "features": ["x03"], "text": "y rises with the within-date rank of x03"},
            {"id": "x07_when_x11", "features": ["x07", "x11"],
             "text": "x07 predicts y only on dates where the date-constant x11 is positive"},
            {"id": "momentum", "features": ["y"], "text": "y is positively autocorrelated per asset at lag 1 (0.15)"},
        ],
        "decoys": [{"feature": "x05", "text": "x05 is x03 plus noise: correlated with y only through x03"}],
        "noise_features": [f"x{i:02d}" for i in range(N_FEAT) if i not in (3, 5, 7, 11)],
        "oracle_holdout_r2": oracle,
    }
    (out / "sealed" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "public" / "README.md").write_text(
        f"Panel of {N_ASSETS} assets x {N_TRAIN} dates. Columns: date, asset, x00..x{N_FEAT - 1:02d} (features known "
        "at the close of `date`), y (the asset's return over the NEXT period).\nTask: forecast y from information "
        "available at `date`.\n")
    print(f"wrote {out}, oracle holdout R2 {oracle:.3f}")


if __name__ == "__main__":
    main()
    main(Path(__file__).with_name("planted1b"), seed=11)  # held out: never tuned on
