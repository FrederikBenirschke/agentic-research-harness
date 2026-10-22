"""Generate worlds/planted4: nonlinearities that are not squares.

    y = 0.25 * x03 * x09                 interaction: neither feature predicts y alone
      + 0.35 * sin(2.5 * x06)            oscillation: non-monotone, a quadratic fit misses it
      + 0.55 * max(x11 - 0.5, 0)         threshold: x11 matters only above 0.5
      + noise
    x13 = x03 * x09 + noise              decoy: a linear screen finds it, not x03 or x09

planted3's confound move ("regress your strongest feature on the others and their squares")
names the planted3 answer; here it does not.

    python worlds/make_planted4.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from make_planted import N_ASSETS, N_FEAT, N_HOLD, N_TRAIN, ar1

SEED = 4


def main(out=Path(__file__).with_name("planted4"), seed=SEED, a=3, b=9, osc=6, hinge=11, decoy=13):
    rng = np.random.default_rng(seed)
    T = N_TRAIN + N_HOLD
    x = ar1(rng, (T, N_ASSETS, N_FEAT), 0.6)
    x[:, :, decoy] = x[:, :, a] * x[:, :, b] + 0.7 * rng.normal(size=(T, N_ASSETS))
    parts = {"interact": 0.25 * x[:, :, a] * x[:, :, b],
             "oscillate": 0.35 * np.sin(2.5 * x[:, :, osc]),
             "threshold": 0.55 * np.maximum(x[:, :, hinge] - 0.5, 0)}
    mu = sum(parts.values())
    y = mu + rng.normal(size=(T, N_ASSETS))
    d, s = np.meshgrid(np.arange(T), np.arange(N_ASSETS), indexing="ij")
    df = pl.DataFrame({"date": d.ravel(), "asset": s.ravel(),
                       **{f"x{i:02d}": x[:, :, i].ravel() for i in range(N_FEAT)}, "y": y.ravel()})
    for sub in ("public", "sealed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("date") < N_TRAIN).write_parquet(out / "public" / "train.parquet")
    df.filter(pl.col("date") >= N_TRAIN).write_parquet(out / "sealed" / "holdout.parquet")
    yh, mh = y[N_TRAIN:].ravel(), mu[N_TRAIN:].ravel()
    oracle = float(1 - np.sum((yh - mh) ** 2) / np.sum((yh - yh.mean()) ** 2))
    fa, fb, fo, fh, fd = (f"x{i:02d}" for i in (a, b, osc, hinge, decoy))
    truth = {
        "mechanisms": [
            {"id": "interact", "features": [fa, fb],
             "pattern": rf"interact|product|multipl|cross.?term|{fa}\s*[*×x·]\s*{fb}|{fb}\s*[*×x·]\s*{fa}|same sign|jointly",
             "text": f"y rises with the product {fa}*{fb} (coefficient 0.25); neither feature predicts y on its own"},
            {"id": "oscillate", "features": [fo],
             "pattern": r"\bsin\b|\bsine\b|\bcos\b|cosin|oscillat|periodic|\bwave|several (peaks|turns)|rises?,? (then )?falls?,? (and |then )?rises?",
             "text": f"y follows sin(2.5 * {fo}) (coefficient 0.35): rises, falls and rises again across {fo}'s range"},
            {"id": "threshold", "features": [fh],
             "pattern": r"threshold|hinge|kink|max\(|clip|relu|piecewise|> ?0?\.5|above 0?\.5|one.?sided|asymmetr|only (when|for|above)",
             "text": f"y rises with {fh} only above 0.5: 0.55 * max({fh} - 0.5, 0); below that {fh} has no effect"},
        ],
        "decoys": [{"feature": fd, "text": f"{fd} is {fa}*{fb} plus noise: correlated with y only through the interaction"}],
        "noise_features": [f"x{i:02d}" for i in range(N_FEAT) if i not in (a, b, osc, hinge, decoy)],
        "oracle_holdout_r2": oracle,
    }
    (out / "sealed" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "public" / "README.md").write_text(
        f"Panel of {N_ASSETS} assets x {N_TRAIN} dates. Columns: date, asset, x00..x{N_FEAT - 1:02d} (features known "
        "at the close of `date`), y (the asset's return over the NEXT period).\nTask: forecast y from information "
        "available at `date`.\n")
    share = {k: f"{np.var(v) / np.var(y):.3f}" for k, v in parts.items()}
    corr = {f: f"{np.corrcoef(x[:, :, i].ravel(), y.ravel())[0, 1]:+.3f}" for f, i in
            ((fa, a), (fb, b), (fo, osc), (fh, hinge), (fd, decoy))}
    print(f"wrote {out}, oracle holdout R2 {oracle:.3f}, variance share {share}, linear corr {corr}")


if __name__ == "__main__":
    main()
    main(Path(__file__).with_name("planted4b"), seed=17, a=1, b=12, osc=8, hinge=14, decoy=5)  # held out
