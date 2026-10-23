# Sealed exam protocol

The exam worlds are a test set that nobody tuning the prompts may inspect. The
generator and the worlds live under `worlds/exam/`, which is not in the
repository.

- **Only frozen prompt arms are run on exam worlds.** An arm is frozen before
  its exam pass, with sha256 hashes of its prompt files and of the harness code,
  and it is not edited during or after the pass.
- **Every run is appended to the ledger,** including failed, crashed and
  discarded runs. Nothing is deleted from it.
- **Exam results are never used for tuning.** They are not used to choose,
  edit or rank prompt variants for further development.
- **After a pass the exam worlds are retired to dev.** They may then be used
  for tuning like any other dev world. The next pass needs a fresh set with new
  seeds and new feature assignments, built by someone who has not seen the
  prompts' tuning.

## Pass 1, 2026-10-22/23

Arms `explore3` (primer only) and `plancall` (separate plan call), Haiku 5.5
and Opus 5.5, worlds exam1 to exam6, 3 runs per cell, 72 runs. Each cell ran
with `--max-turns 60 --timeout 2400 --budget 6`. The ledger is
[results/exam/ledger.jsonl](../results/exam/ledger.jsonl), and
`python scripts/tables.py` rebuilds the README table from it.

Frozen at git `3dbc292` plus the working tree. The hashes match these files as
committed in `c2764ff` and `29d44ad`:

```
3a5b40f451c715acf2b2d589ea866aed82cb84bc27305fcef24d23683070aad0  evals/prompts/explore3/analyst.md
23af37e2cef41a1d01a82b667798e43ef134a9ea41fde93ebbf0e4c0410dcd0b  evals/prompts/explore3/forecaster.md
3a5b40f451c715acf2b2d589ea866aed82cb84bc27305fcef24d23683070aad0  evals/prompts/plancall/analyst.md
6bd6e0f4a47d8f5f4860f28794af965b7d56a4ee4d61d580d1a6afdf386b62ef  evals/prompts/plancall/forecaster.md
82ccd569f4bd84136ad8cf87e3db06b04a53d6fafc70ee9210e9e884d7492cd7  src/arh/runner.py
f244a5f52af9c765f8c7d0d185a66cf9a9cf060fa4c6de1697d76e9401d409ce  src/arh/evals.py
97e9c691820ee9edc5db2a95bcfbdc44a25505dbbd2c54088493f0406aee4aa0  src/arh/tools.py
5ec3a392d489674a1a403b641a84bffc7b206e7293728b64d738715c20ce68f3  src/arh/score.py
```

After the pass, exam1 to exam6 were retired to dev and their generator was
read to explain the scores above 1.0 of the oracle (README, Section 5).
