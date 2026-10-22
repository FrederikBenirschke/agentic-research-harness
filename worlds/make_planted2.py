"""Generate worlds/planted2: a control for planted1 with NO regime effect.

    y = 0.40 * x09                       plain linear effect
      + 0.30 * rank_z(x02)               cross-sectional rank effect
      - 0.10 * y_lag1                    per-asset reversal
      + noise
    x06 = x09 + noise                    decoy
    x13 date-constant, no effect         a regime feature that gates nothing

An agent that reports an effect living in one x13 regime is wrong here.

    python worlds/make_planted2.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from make_planted import N_ASSETS, N_FEAT, N_HOLD, N_TRAIN, ar1, rank_z

SEED = 2


def main(out=Path(__file__).with_name("planted2")):
    rng = np.random.default_rng(SEED)
    T = N_TRAIN + N_HOLD
    x = ar1(rng, (T, N_ASSETS, N_FEAT), 0.6)
    x[:, :, 13] = ar1(rng, (T, 1), 0.95)
    x[:, :, 6] = x[:, :, 9] + 0.6 * rng.normal(size=(T, N_ASSETS))
    y, mu = np.zeros((T, N_ASSETS)), np.zeros((T, N_ASSETS))
    for t in range(T):
        mu[t] = 0.40 * x[t, :, 9] + 0.30 * rank_z(x[t:t + 1, :, 2])[0] - (0.10 * y[t - 1] if t else 0)
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
            {"id": "linear_x09", "features": ["x09"], "text": "y rises linearly with x09, the same on every date"},
            {"id": "rank_x02", "features": ["x02"], "text": "y rises with the within-date rank of x02"},
            {"id": "reversal", "features": ["y"], "text": "y is negatively autocorrelated per asset at lag 1 (-0.10)"},
        ],
        "decoys": [{"feature": "x06", "text": "x06 is x09 plus noise: correlated with y only through x09"},
                   {"feature": "x13", "text": "x13 is date-constant and gates nothing: no effect depends on its regime"}],
        "noise_features": [f"x{i:02d}" for i in range(N_FEAT) if i not in (2, 6, 9, 13)],
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
