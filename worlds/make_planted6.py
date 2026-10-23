"""Generate worlds/planted6: scheduled events ("earnings announcements").

Each asset announces once every 60 dates on its own fixed schedule, so the dates are known in advance.
y on date t is the return from t to t+1; the announcement period of an asset is the date whose y contains
the announcement.

    x_to   = dates until the asset's next announcement period (0 on that date)       public calendar
    x_since = dates since the asset's last announcement period                          public calendar
    x_rev  = analyst revision, drifts during the quarter (AR), resets after an announcement
    s      = 0.6 * x_rev + 0.8 * eps                                                    the surprise
    x_surp = the last announcement's surprise s, published at the close of the announcement date

    y = 2.0 * s * [x_to == 0]                              announcement jump; 1.2 * x_rev of it is predictable
      + 0.25 * x_surp * exp(-(x_since - 1) / 8)            post-announcement drift, decays over ~8 dates
      + noise, with sd 1 normally and sd 2.5 on announcement dates
    x_dec = x_rev + noise                                  decoy: a noisy proxy of the revision

Pooled linear models on all rows dilute the jump into ~60x more non-event rows, and the event-date noise
dominates squared error; separating "is it an event" from "how big" recovers it.

    python worlds/make_planted6.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from make_planted import N_ASSETS, N_FEAT, N_HOLD, N_TRAIN, ar1

SEED = 6
PERIOD = 60


def main(out=Path(__file__).with_name("planted6"), seed=SEED, to=2, since=9, rev=5, surp=12, dec=14):
    rng = np.random.default_rng(seed)
    T = N_TRAIN + N_HOLD
    x = ar1(rng, (T, N_ASSETS, N_FEAT), 0.6)
    for i in range(N_FEAT):  # noise features: mixed scales
        if i not in (to, since, rev, surp, dec):
            x[:, :, i] *= 10 ** rng.uniform(-1, 1)
    offset = rng.integers(0, PERIOD, N_ASSETS)
    t = np.arange(T)[:, None]
    x_to = (offset[None, :] - t) % PERIOD                      # 0 on the announcement date
    event = x_to == 0
    x_since = np.where(event, PERIOD, PERIOD - x_to)           # dates since the last announcement date
    # revision: AR(1) within the quarter, reset after each announcement; unit variance on event dates
    r = np.zeros((T, N_ASSETS))
    for d in range(T):
        prev = np.zeros(N_ASSETS) if d == 0 else np.where(event[d - 1], 0.0, r[d - 1])
        r[d] = 0.97 * prev + 0.24 * rng.normal(size=N_ASSETS)
    r /= r[event].std()
    s = np.where(event, 0.6 * r + 0.8 * rng.normal(size=(T, N_ASSETS)), 0.0)
    # last surprise, known from the close of the date after the announcement date
    x_surp = np.zeros((T, N_ASSETS))
    for d in range(1, T):
        x_surp[d] = np.where(event[d - 1], s[d - 1], x_surp[d - 1])
    parts = {"event": np.where(event, 2.0 * 0.6 * r, 0.0),
             "drift": 0.25 * x_surp * np.exp(-(x_since - 1) / 8)}
    mu = sum(parts.values())
    noise = np.where(event, 2.5, 1.0) * rng.normal(size=(T, N_ASSETS))
    y = mu + np.where(event, 2.0 * (s - 0.6 * r), 0.0) + noise   # plus the unpredictable part of the jump
    x[:, :, to], x[:, :, since], x[:, :, rev], x[:, :, surp] = x_to, x_since, r, x_surp
    x[:, :, dec] = r + 1.0 * rng.normal(size=(T, N_ASSETS))
    d_, a_ = np.meshgrid(np.arange(T), np.arange(N_ASSETS), indexing="ij")
    df = pl.DataFrame({"date": d_.ravel(), "asset": a_.ravel(),
                       **{f"x{i:02d}": x[:, :, i].ravel() for i in range(N_FEAT)}, "y": y.ravel()})
    for sub in ("public", "sealed"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("date") < N_TRAIN).write_parquet(out / "public" / "train.parquet")
    df.filter(pl.col("date") >= N_TRAIN).write_parquet(out / "sealed" / "holdout.parquet")
    yh, mh = y[N_TRAIN:].ravel(), mu[N_TRAIN:].ravel()
    oracle = float(1 - np.sum((yh - mh) ** 2) / np.sum((yh - yh.mean()) ** 2))
    ft, fs, fr, fu, fd = (f"x{i:02d}" for i in (to, since, rev, surp, dec))
    truth = {
        "mechanisms": [
            {"id": "event", "features": [ft, fr],
             "pattern": r"announce|earnings|event|calendar|schedul|jump|" + rf"{ft}\s*==\s*0",
             "text": f"{ft} counts dates to a scheduled event (0 on the event date); on event dates y jumps by "
                     f"1.2 * {fr} plus large noise; {fr} has no effect on other dates"},
            {"id": "drift", "features": [fu, fs],
             "pattern": r"drift|post.?(event|announce|earnings)|pead|decay|after the (event|announce)",
             "text": f"after an event y drifts by 0.25 * {fu} * exp(-({fs} - 1) / 8): the last surprise {fu} "
                     f"predicts returns for about 8 dates, fading with {fs}"},
        ],
        "decoys": [{"feature": fd, "text": f"{fd} is {fr} plus noise: a weaker proxy of the revision"}],
        "noise_features": [f"x{i:02d}" for i in range(N_FEAT) if i not in (to, since, rev, surp, dec)],
        "oracle_holdout_r2": oracle,
    }
    (out / "sealed" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "public" / "README.md").write_text(
        f"Panel of {N_ASSETS} assets x {N_TRAIN} dates. Columns: date, asset, x00..x{N_FEAT - 1:02d} (features known "
        "at the close of `date`), y (the asset's return over the NEXT period).\nTask: forecast y from information "
        "available at `date`.\n")
    share = {k: f"{np.var(v) / np.var(y):.4f}" for k, v in parts.items()}
    print(f"wrote {out}, oracle holdout R2 {oracle:.4f}, variance share {share}, event share {event.mean():.4f}")


if __name__ == "__main__":
    main()
    main(Path(__file__).with_name("planted6b"), seed=23, to=11, since=0, rev=7, surp=3, dec=13)  # held out
