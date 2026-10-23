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
| `planted6` | `make_planted6.py` | scheduled events: a jump on each asset's announcement date (partly predictable from a drifting revision), post-announcement drift, decoy |
| `planted6b` | `make_planted6.py` | as planted6, seed 23, different feature indices |
| `planted7` | `make_planted7.py` | returns with per-asset volatility, two latent factors and five latent industries; the forecastable part is residual reversal and a feature that works relative to its industry; decoy that loads on a drifting factor |
| `planted7b` | `make_planted7.py` | as planted7, seed 29, different feature indices |
| `planted8` | `make_planted7.py` | as planted7 with larger factor spread and an industry-level part in the feature, so only vol + factors + industries + residual pays; industry eigenvalues at the noise edge |
| `planted8b` | `make_planted7.py` | as planted8, seed 31, different feature indices |
| `planted8c` | `make_planted7.py` | planted8 with a stronger industry factor (eigenvalues well above the noise edge) |

`make_random.py SEED` draws a procedural world (0 to 5 mechanisms from a
library, decoys, noise level, noise features) into `rand/r<SEED>`. Seeds below
1000 are for development; seeds from 100000 are exam worlds whose truth files
are not read while tuning.

## Exam worlds

`exam/exam1` to `exam/exam6` were a sealed test set, built by an agent so that
whoever tunes the prompts had not seen them. They are not in the repository.
After pass 1 they were retired to dev. See [docs/exam.md](../docs/exam.md) for
the protocol and [results/exam/ledger.jsonl](../results/exam/ledger.jsonl) for
every run.

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
