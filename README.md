# agentic-research-harness

LLM agents (Claude Agent SDK) look for planted mechanisms in synthetic panels
with a known truth, so every claim and forecast is graded against the answer.

- **Sealed exam:** on six worlds built by a separate agent and run once with
  frozen prompts, Opus scored 0.92 or more of the oracle on every world with
  signal, and no model overfit the null world.
- **A reference primer pushes out methods the model knows:** with the primer,
  Haiku's plans drop PCA from 5/5 to 0/5. A separate plan call before the
  primer restores them: on planted8, PCA in agent code went from 1 of 30 runs
  to 17 of 18.
- **Numbers reach Haiku where instructions do not:** one line in the data
  summary (a per-group variance decomposition) raised Haiku's score on sector
  worlds by 0.15 ± 0.24 of the oracle (8 fresh worlds), where a prompt line
  asking for the same check did nothing.
- **The agents cannot see the answers:** an audit of 472 run traces and a
  deliberate break-in attempt found no read of the holdout or truth files.

## Setup

A world is a panel of 40 assets × 500 dates with 16 features. The task is to
forecast next-period `y`. Each world plants a few mechanisms (e.g. a feature
that matters only when another is positive) and a decoy correlated with `y`
only through a planted one. The generator writes a sealed truth file and
holdout that the agents cannot read.

Three agents share a SQLite note store (full-text search, exposed over MCP,
tools scoped per agent): the **analyst** studies the data and writes findings,
the **ideator** proposes mechanisms without raw data, and the **forecaster**
files a model. Analyst and forecaster run Python through Bash in the Claude
Code OS sandbox (no network, no reads outside their work dir). The harness
scores a forecast in a separate worker that receives one date at a time, shows
the agent walk-forward CV on training data, and stores the holdout score
without returning it. Local models (gpt-oss) run through Ollama's
Anthropic-compatible API with the same tools.

Each agent is one Markdown prompt; prompt versions live in
`evals/prompts/<arm>/`. An eval is one agent × a fixed set of starting notes ×
a world, so an ablation is a grid of model × prompt × world.

**Score** = forecaster holdout R² / oracle holdout R², where the oracle is the
true conditional mean (1.00 = as good as the truth). Graders also record which
mechanisms appear in the note and in the model code, whether the decoy is a
model input, whether the note's numbers appear in the agent's tool output, and
an Opus judge's correct/partial/wrong rating against the truth. In the tables,
± is the standard deviation over runs, or over worlds where stated.

## What was tuned on what

Some arms were built after looking at the world they were then scored on.
This decides how much each result says:

| result | built on | tested on | what it shows |
|---|---|---|---|
| tuned prompt (Section 1) | planted1 | planted1b: same mechanisms, new seed | no seed overfit; not that it finds new kinds of mechanism |
| group decomposition (Section 5) | exam3, after the exam | fresh random worlds, same kind of sector effect | a generic diagnostic transfers within that mechanism family |
| plan call (Section 3) | planted8 | planted8c, sealed exam | first result, n=3 |
| sealed exam (Section 4) | nothing: frozen prompts, worlds built blind | | the clean test |

An earlier arm (`evals/prompts/move`) told the analyst to regress its strongest
feature "on the other features and their squares". That names the planted form
in planted3 (`y` depends on `x04²`), so its hits are leakage, and we no longer
report them.

## Results

![results](docs/results.png)

### 1. Tuned prompts on a new seed

The tuned prompt (`cur`) adds two checks to the original (`v1`): split an
effect by regime (a date-level feature above or below its median), and fit `y`
on both features of a correlated pair to see which keeps its effect.

| model | prompt | planted1b score | forecaster found x07 gate | used decoy | planted2 (control) | $/run |
|---|---|---|---|---|---|---|
| gpt-oss (local) | original | 0.64 ± 0.14 | 0/3 | 0/3 | 0.90 ± 0.00 | - |
| gpt-oss (local) | tuned | 0.63 ± 0.14 | 0/3 | 0/3 | 0.90 ± 0.00 | - |
| Haiku 5.5 | original | 0.70 ± 0.08 | 1/5 | 4/5 | 0.90 ± 0.00 | 0.003 |
| Haiku 5.5 | tuned | 0.81 ± 0.00 | 5/5 | 0/5 | 0.90 ± 0.02 | 0.005 |
| Sonnet 5.5 | original | 0.74 ± 0.00 | 4/5 | 5/5 | 0.91 ± 0.00 | 0.043 |
| Sonnet 5.5 | tuned | 0.85 ± 0.06 | 5/5 | 4/5 | 0.91 ± 0.00 | 0.048 |
| Opus 5.5 | original | 0.80 ± 0.08 | 5/5 | 3/5 | 0.92 ± 0.01 | 0.153 |
| Opus 5.5 | tuned | 0.95 ± 0.00 | 5/5 | 0/5 | 0.91 ± 0.00 | 0.149 |
| Fable 5.1 | original | 0.95 ± 0.00 | 5/5 | 1/5 | 0.92 ± 0.03 | 0.594 |
| Fable 5.1 | tuned | 0.92 ± 0.06 | 5/5 | 0/5 | 0.91 ± 0.01 | 0.532 |

The tuned prompt adds 0.11 to 0.15 for Haiku, Sonnet and Opus and costs
nothing on the control world planted2. Fable already finds the gate with the
original prompt. gpt-oss finds it with neither; it fits a linear model on all
16 features, and none of the numbers in its notes appear in its tool output
(Claude models: 0.70 to 0.99). The gain does not come from a wider search:
Claude forecasters score 1.0 to 1.3 models per run under both prompts.

On planted3, whose mechanisms these checks do not target (a U-shape in `x04`
with near-zero linear correlation, a two-period lag, a cross-asset lead), the
tuned Haiku, Sonnet and Opus analysts missed the U-shape in 15 of 15 runs and
their forecasters used the decoy `x12 = x04² + noise` in 15 of 15. A generic
EDA instruction ("correlation ≈ 0 is not no effect: check deciles, lags > 1,
other assets") raised lag-2 hits from 1/5 to 3/5 but found no U-shape. A
forecaster given a note naming all three mechanisms scores 1.00, so the gap is
in discovery, not modelling.

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
- One real dataset (`rank100v2`, no truth file): the regime instruction left
  holdout R² unchanged (0.042 vs 0.041).
- An earlier planted1b batch (n=4, [docs/eval-log.md](docs/eval-log.md))
  found no gain from the tuned prompt (0.73 vs 0.74); unresolved.

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
