# Worlds

Each world is a panel of assets by dates with features `x00`, `x01`, ... and a
target `y`. `arh init` copies `public/train.parquet` into the run's work
directory. `sealed/holdout.parquet` is used only to compute the holdout score,
which is stored but not returned to the agents. The real worlds also have
`final/test.parquet`, which the harness does not read.

## Planted worlds

Synthetic panels with known mechanisms. `sealed/truth.json` lists the
mechanisms, the decoy feature and the oracle holdout R² (the R² of the true
conditional mean). The generator scripts in this directory document each
formula and regenerate the worlds.

| world | generator | contents |
|---|---|---|
| `planted1` | `make_planted.py` | rank effect, regime-gated effect, momentum, decoy |
| `planted1b` | `make_planted.py` | as planted1, seed 11 |
| `planted2` | `make_planted2.py` | control for planted1: linear and rank effects, reversal, a regime feature with no effect, decoy |
| `planted3` | `make_planted3.py` | U-shape, two-period delayed effect, cross-asset lead-lag, decoy |
| `planted3b` | `make_planted3.py` | as planted3, seed 13, different feature indices |
| `planted4` | `make_planted4.py` | interaction, sine, threshold, decoy |
| `planted4b` | `make_planted4.py` | as planted4, seed 17, different feature indices |
| `planted5` | `make_planted5.py` | four linear effects that appear only after preprocessing (per-date rank of a t(2) feature, clipping a feature with 1% spikes, per-date z-score of a feature with drifting scale, scaling a feature with sd 0.01 before ridge), a fat-tailed market factor that is 45–49% of var(y), no decoy; oracle holdout R² 0.049 |
| `planted5b` | `make_planted5.py` | as planted5, seed 19, different feature indices; oracle holdout R² 0.052 |

All planted worlds have 40 assets, 16 features, 500 training dates and 200
holdout dates.

## Real worlds

The generation details and the mechanism in each real world are not in this
repository.

| world | assets | features | train / holdout / final dates |
|---|---|---|---|
| `rank100v2` | 60 | 28 | 525 / 225 / 250 |
| `regime200` | 60 | 28 | 525 / 225 / 250 |
| `seed0v2` | 24 | 40 | 525 / 225 / 250 |

SHA-256 of each split (the run manifest records the same hashes):

```
d25a1ebfec540802927522094b5377e7136731b27c9433f5c8c7e1ee47042aa6  rank100v2/public/train.parquet
2d98b4f93cb03043b769d24853ee5f798f8d5ca84d94edbff992cc3c278fea45  rank100v2/sealed/holdout.parquet
6c93c97f7f41bc8e69cbe828ae98789acea180b18a12a683369a8a4f9c0ea5ad  rank100v2/final/test.parquet
eed3321df2c857c07ce9ddaeefeccd68a3d58e0dbbeb8f8136b0983c817a74ef  regime200/public/train.parquet
652db54eeea36dae38aba9af01185c4f654bd7dceb4f2a10e14d10d4ee409f15  regime200/sealed/holdout.parquet
fc0547a0801586536572b12627ab99870266cb6a4cf3f4cd07a0cecebbed7edb  regime200/final/test.parquet
45b696146b65e54a48cbea1b0227f5e734080511ca4a928933b140d5219ee2be  seed0v2/public/train.parquet
2e442e87545545a95b427d3f5a0c177ad4ef13cf964cca509a9ab100b6607775  seed0v2/sealed/holdout.parquet
e57c716badf9ddc62de2b06fede575bd1e4cb0ea3f01f8ba0e239515a847a289  seed0v2/final/test.parquet
```
