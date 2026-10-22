"""Generate worlds/planted3: mechanisms the agent prompts never mention.

    y = 0.30 * (x04**2 - 1)              U-shape: linear correlation with y is ~0
      + 0.25 * x10[t-2]                  x10 acts with a two-period delay
      + 0.30 * y[t-1, a-20]  (a >= 20)   cross-asset lead-lag: asset a-20 leads asset a
      + noise
    x12 = x04**2 + noise                 decoy: a linear screen finds it, not x04

The prompts' checklist (regime split, proxy pair, lag-1 autocorrelation) finds none of these.

    python worlds/make_planted3.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from make_planted import N_ASSETS, N_FEAT, N_HOLD, N_TRAIN, ar1

SEED = 3
LEAD = N_ASSETS // 2


def main(out=Path(__file__).with_name("planted3"), seed=SEED, u=4, lag=10, decoy=12):
    """u: U-shape feature, lag: two-period-delay feature, decoy: noisy copy of u**2."""
    rng = np.random.default_rng(seed)
    T = N_TRAIN + N_HOLD
    x = ar1(rng, (T, N_ASSETS, N_FEAT), 0.6)
    x[:, :, decoy] = x[:, :, u] ** 2 + 0.7 * rng.normal(size=(T, N_ASSETS))
    y, mu = np.zeros((T, N_ASSETS)), np.zeros((T, N_ASSETS))
    for t in range(T):
        mu[t] = 0.30 * (x[t, :, u] ** 2 - 1) + (0.25 * x[t - 2, :, lag] if t >= 2 else 0)
        if t:
            mu[t, LEAD:] += 0.30 * y[t - 1, :LEAD]
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
            {"id": "ushape", "features": [f"x{u:02d}"], "pattern": r"squar|quadratic|u-shape|u shape|\^2|\*\*2|abs|convex|magnitude|nonlinear|non-linear",
             "text": f"y rises with x{u:02d} squared (a U-shape): extreme x{u:02d} either way predicts higher y; linear correlation is ~0"},
            {"id": "lag2", "features": [f"x{lag:02d}"], "pattern": r"lag.?2|t-2|two.?(period|date|day|step)|shift\(2\)|2-period|2 period",
             "text": f"y depends on x{lag:02d} from two dates earlier (coefficient 0.25), not on today's x10"},
            {"id": "leadlag", "features": ["y"], "pattern": r"cross-asset|cross asset|lead|other asset|another asset|pair|asset \d+|a-20|\+ ?20|- ?20",
             "text": f"assets {LEAD}..{N_ASSETS - 1} follow asset a-{LEAD}'s previous-period y (coefficient 0.30)"},
        ],
        "decoys": [{"feature": f"x{decoy:02d}", "text": f"x{decoy:02d} is x{u:02d} squared plus noise: correlated with y only through x{u:02d}"}],
        "noise_features": [f"x{i:02d}" for i in range(N_FEAT) if i not in (u, lag, decoy)],
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
    main(Path(__file__).with_name("planted3b"), seed=13, u=8, lag=14, decoy=1)  # held out: other seed and features
