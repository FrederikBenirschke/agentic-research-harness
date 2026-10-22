# Worlds

Panels the agent is asked to forecast. Each world has the three splits the harness
expects; the agent's work dir gets only the early part of `public/train.parquet`.
How the data was generated, and the hidden rule in each world, are kept out of this
repository until the reported runs are finished.

| world | assets | features | train / validation / final dates |
|---|---|---|---|
| `rank100v2` | 60 | 28 | 525 / 225 / 250 |
| `regime200` | 60 | 28 | 525 / 225 / 250 |
| `seed0v2` | 24 | 40 | 525 / 225 / 250 |

SHA-256 of every split (the run manifest pins the same hashes):

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
