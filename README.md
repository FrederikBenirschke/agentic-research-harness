# agentic-research-harness

This project tests whether LLM agents can do quantitative research: given a
table of data, can they find the patterns in it and build a forecast? To check
their answers we need to know the truth, so the code makes the data itself.

## What the code does

1. **Makes a dataset with hidden rules.** A Python script generates a table
   of 40 assets over 700 days. Each row has 16 numeric columns (`x00` to
   `x15`) and the asset's next-day return (`y`). The script builds `y` from a
   few rules it writes down in a hidden file, for example "`x07` raises
   tomorrow's return, but only on days when `x11` is positive". It also adds a
   trap: a column that looks useful but only because it is a noisy copy of a
   real one. The rest of the columns are noise. The last 200 days are held
   back for scoring. Each generated dataset has a name (`planted1`, `exam3`,
   `r55`, ...), and we call one a *world*.
2. **Lets agents work on it.** Claude agents (built on the Claude Agent SDK)
   get the first 500 days and a shell where they can run Python. The
   *analyst* explores the data and writes up what it found. The *forecaster*
   reads those notes and writes a forecasting model as a Python file. The
   agents pass notes to each other through a small database. Each agent's
   instructions are one Markdown file, and we keep different versions of
   these instructions side by side to compare them.
3. **Keeps the answers out of reach.** The shell runs in a sandbox that
   cannot see the held-back days, the hidden rules or the generator script,
   and has no internet.
4. **Scores the forecast.** The code runs the agent's model on the held-back
   days, one day at a time so it cannot look ahead, and compares it with the
   best possible forecast: the one computed from the hidden rules (the
   *oracle*). A score of 1.0 means the agent did as well as the oracle, 0
   means no better than guessing the average. The code also checks which of
   the hidden rules the agent mentions and actually uses in its model,
   whether it fell for the trap, and whether the numbers in its notes come
   from code it really ran.

We ran Haiku, Sonnet, Opus and Fable, and one open model (gpt-oss) served
locally, usually 3 to 10 times per setting. In the tables, ± is the standard
deviation over runs, or over datasets where stated.

## What we found

- On six datasets built by a separate agent and never looked at before the
  run, Opus came within 8% of the oracle on every dataset that had a pattern
  to find. On the one with no pattern, no model fooled itself into finding
  one.
- A cheat sheet of research methods in the instructions made the model forget
  methods it already knew. Asked to plan, Haiku included PCA in 5 of 5 plans
  without the sheet and in 0 of 5 with it. Having the model write its own
  plan before it sees the sheet brings these methods back into the code it
  runs (PCA in 17 of 18 runs, against 1 of 30).
- Haiku ignores instructions but reacts to numbers. Telling it to check
  whether a column works within each sector changed nothing. Printing the
  result of that check in the data summary it reads at the start improved its
  forecasts on 6 of 8 new datasets.
- The agents cannot peek at the answers. We checked the logs of all 472 runs
  and told one agent to try to break in; every attempt was blocked.

## What was tuned on what

Some arms were built after looking at the world they were then scored on.
This decides how much each result says:

| result | built on | tested on | what it shows |
|---|---|---|---|
| tuned prompt (Section 1) | planted1 | planted1b: same mechanisms, new seed | nothing about discovery: the prompt names the mechanisms |
| group decomposition (Section 5) | exam3, after the exam | fresh random worlds, same kind of sector effect | a generic diagnostic transfers within that mechanism family |
| plan call (Section 3) | planted8 | planted8c, sealed exam | first result, n=3 |
| sealed exam (Section 4) | nothing: frozen prompts, worlds built blind | | the clean test |

An earlier arm (`evals/prompts/move`) told the analyst to regress its strongest
feature "on the other features and their squares". That names the planted form
in planted3 (`y` depends on `x04²`), so its hits are leakage, and we no longer
report them.

## Results

### 1. A prompt that names the mechanism helps only on that mechanism

The tuned prompt (`cur`) adds two checks to the original (`v1`): split an
effect by a date-level regime, and fit `y` on both features of a correlated
pair. These are planted1's two mechanisms (the `x07` gate on `x11`, and the
decoy `x05`, a noisy copy of `x03`), so the prompt describes the answer. On
planted1b (same mechanisms, new seed) it adds 0.11 to 0.15 of the oracle for
Haiku, Sonnet and Opus. That is instruction following, not discovery.

On planted3, whose mechanisms it does not describe (a U-shape in `x04` with
near-zero linear correlation, a two-period lag, a cross-asset lead), the tuned
Haiku, Sonnet and Opus analysts missed the U-shape in 15 of 15 runs, and their
forecasters used the decoy `x12 = x04² + noise` in 15 of 15. A generic EDA
instruction ("correlation ≈ 0 is not no effect: check deciles, lags > 1,
other assets") raised lag-2 hits from 1/5 to 3/5 and found no U-shape. A
forecaster given a note naming all three mechanisms scores 1.00, so the gap is
in discovery, not modelling. gpt-oss fits a linear model on all 16 features,
and none of the numbers in its notes appear in its tool output.

### 2. The primer pushes out what the model knows

Later prompts (`explore3`) append a reference primer on quant research
practice. To see what it does to the model's own ideas, we asked the model,
with no tools or data, to list the checks it would run on a returns panel
(5 plans per setting):

| model | primer | plans with PCA | plans with industries |
|---|---|---|---|
| Haiku 5.5 | none | 5/5 | 5/5 |
| Haiku 5.5 | in system prompt, or before/after the question | 0/5 | 0–2/5 |
| Haiku 5.5 | "write your own list first" in the same answer | 2/5 | 1/5 |
| Haiku 5.5 | primer framed as additive | 1/5 | 1/5 |
| Opus 5.5 | any of the above | 5/5 | 5/5 |

Haiku's plans stay at 15 to 17 items either way, so the primer's items replace
its own. This resembles part-list cueing in human memory and the finding that
models follow context over what they know (Xie et al. 2023).

Opus keeps the items in the plan but does not run them. planted8
(`worlds/make_planted7.py`) pays only for the full chain: divide out each
asset's volatility, remove two latent factors, cluster residual correlations
into industries, model the residual. In 6 Opus runs, none did PCA.

### 3. A separate plan call

The fix moves the plan out of the primer's context. A separate call with no
primer writes the model's own plan (`plan:` in the prompt frontmatter), and
the agent then runs with that plan on top of its prompt. On planted8, runs
whose code computes an eigendecomposition or clusters assets rose from 1/30
without the plan call to 17/18 with it. Scores stayed low (≤ 0.21) because
planted8's industry eigenvalues sit at the Marchenko–Pastur noise edge.
planted8c raises the industry factor above the edge:

| model | arm | PCA / clustering runs | score |
|---|---|---|---|
| Opus 5.5 | primer only | 2/3 / 0/3 | −0.08, 0.01, 0.00 |
| Opus 5.5 | plan call | 3/3 / 3/3 | 0.67, 0.15, 0.24 |
| Haiku 5.5 | primer only | 0/3 / 0/3 | −0.84, −0.09, −0.06 |
| Haiku 5.5 | plan call | 2/3 / 1/3 | −0.18, −0.12, −0.12 |

n=3 per cell. On the exam worlds (no factor structure) the plan call neither
helps nor hurts (Haiku, n=10: within ±0.15 on every world, sign mixed), but it
has a collapse mode: 3 of 10 Haiku runs on exam6 scored about 0.1.

### 4. Sealed exam

`worlds/exam/exam1`–`exam6` were built by a separate agent; nobody tuning
prompts opened their sealed files or generator before the pass
([protocol and prompt hashes](docs/exam.md)). Two frozen forecaster arms, 3
runs each per world with Haiku and Opus, 72 runs, all in
[results/exam/ledger.jsonl](results/exam/ledger.jsonl) (– = turn limit or
timeout).

| world | Haiku, primer | Haiku, plan call | Opus, primer | Opus, plan call |
|---|---|---|---|---|
| exam1 | 0.64, 0.67, 0.67 | 0.61, 0.67, 0.85 | 0.96, 1.00, 0.99 | 0.99, 0.96, 1.00 |
| exam2 | 0.55, 0.90, 0.56 | 0.78, 0.89, – | 1.04, 1.04, 1.04 | 1.04, 1.04, 1.04 |
| exam3 | 0.08, 0.04, 0.11 | 0.14, 0.09, 0.03 | 0.96, 0.95, 0.94 | 0.92, 0.92, – |
| exam4 (null, raw R²) | −0.002, −0.003, −0.003 | −0.002, −0.008, −0.003 | −0.003, 0.007, – | −0.003, −0.003, −0.005 |
| exam5 | 0.91, 1.12, 1.06 | 1.04, 1.29, 1.10 | 1.19, 1.11, 1.33 | 1.29, 1.01, 1.18 |
| exam6 | 0.95, 0.81, 0.84 | 0.89, 0.93, 0.69 | 0.96, 0.96, – | 0.96, 0.96, 0.96 |

Opus is near the oracle everywhere there is signal; neither model overfits
the null world. Scores above 1 have two causes found after the pass. On exam2
the stated oracle ignores past `y`, which also carries the signal; with it the
oracle rises and Opus is at 1.01. On exam5 the oracle's holdout R² is low by
chance (0.026 vs 0.038 on training dates, bootstrap sd 0.010). Opus runs cost
about $2 to $3 each, Haiku about $0.07. The exam worlds are now dev worlds.

### 5. Why Haiku fails exam3

In exam3, `x11` predicts `y` only relative to its date × sector mean (sector
code `x06`), scaled by volatility. Every Opus run subtracts the date × sector
mean; no Haiku run does. One Haiku run grouped by `x06` across all dates, saw
93% of the variance within sectors and dropped the idea (by date × sector it
is 71%). Haiku used 32 to 44 of 60 turns, so the turn limit is not the cause.
It can build the model when given the Opus note (7 of 7 runs at 0.93 to 0.96),
and a prompt line asking it to split strong features by date × group did
nothing (0/3).

The group decomposition (`ARH_SUMMARY=groups`) adds one line to the harness's
data summary: for each column constant per asset with few values, every
feature's share of within-date variance in its date × group mean, and the rank
IC of the deviation. It names no feature and fires on any such column.

Haiku on exam3, `explore3`, 10 runs per arm:

| arm | score | runs ≥ 0.7 |
|---|---|---|
| default summary | 0.18 ± 0.20 | 1/10 |
| group decomposition | 0.54 ± 0.41 | 6/10 |
| group decomposition + plan call | 0.43 ± 0.36 | 4/10 |

Since the line was built after exam3, we tested it on random worlds from
`worlds/make_random.py`, 3 runs per arm per world, mean change in score (as
ΔR² / oracle R², since a few worlds have oracle R² < 0.03), ± sd over worlds:

| worlds | Haiku | Sonnet |
|---|---|---|
| 8 fresh worlds with a sector effect | +0.15 ± 0.24, 6/8 up | +0.16 ± 0.11, 8/8 up |
| 6 fresh controls (no sector effect) | −0.06 ± 0.09, 2/6 up | −0.05 (2 worlds) |

It helps on sector worlds and costs a little on controls, where Haiku
sometimes models a group with no signal. Sonnet without the line is no better
than Haiku (−0.02 over 11 worlds); on one world it took the decoy.

### Access to the holdout and truth files

Bash runs in the Claude Code OS sandbox, which denies reads under the home
directory except the run's work dir and the Python runtime, with no network
and `setting_sources=[]`. `scripts/audit_access.py` checked 472 runs and 1,248
Bash commands: none named a sealed file, truth file, generator or `worlds/`,
and every read outside the work dir returned "Operation not permitted". A
Haiku analyst told to break in (relative and absolute paths, `open()`, `find`,
symlinks) was denied 76 times, and its trace has no truth-file content.

## Limitations

- 3 to 10 runs per cell; MDE not computed. Differences of one or two runs out
  of five are not results.
- The tuned prompt and the group decomposition were built on worlds of the
  same kind they were tested on (see the table above). Only the sealed exam is
  a blind test, and it has no factor or industry structure.
- Mechanism detection uses regexes; the lead-lag regex is unreliable, and a
  name-only regex credited the sector effect in 10 of 13 Haiku runs that never
  used it (`mechanisms_in_model` fixes this by also requiring the features in
  the model code).

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
scripts/reproduce.sh          # rebuilds worlds/planted*, the tables and docs/results.png from results/ (no model calls)
.venv/bin/pytest -q
```

Rerunning evals calls the models (Haiku about $0.005 per 15-turn run):

```bash
.venv/bin/arh eval all --spec evals/planted3.json --model claude-haiku-5-5 --n 5 --parallel 5 \
  --max-turns 15 --prompts evals/prompts/cur --out runs/evals/planted3/haiku-cur.jsonl
.venv/bin/arh report --scorecard runs/evals/planted3/haiku-cur.jsonl
scripts/ablation.sh           # the full model x prompt grid (~$30)
```

`ARH_SUMMARY=none|groups` changes the data summary. A `plan:` line in a
prompt's frontmatter turns on the separate plan call.
[docs/eval-log.md](docs/eval-log.md) has the full eval log.

## Next

- A second blind exam with factor and industry structure, to test the plan
  call outside planted8c, and an oracle that uses past `y`.
- Worlds with nonlinearities no prompt names (|x|, thresholds) to measure
  discovery without leakage.
- Several analysts in parallel, each assigned one direction, against the same
  number unassigned.
