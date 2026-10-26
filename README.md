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
- A list of research methods in the instructions made the model drop
  methods it already knew. The agents' instructions end with a reference
  list of things to check. To see what the list does, we sent the model one
  message with the dataset's description and asked which checks it would
  run, with no data and no tools. Without the reference list in that message,
  Haiku's answer included PCA (finding the common factors that move many
  assets together) 5 of 5 times; with the list included, 0 of 5 times. The
  fix: before an agent starts, one separate call without the list asks the
  model for its own checklist. The agent then gets that checklist on top of
  its usual instructions and is told to run every check in it. On a dataset
  built around common factors, the share of agent runs whose code actually
  did PCA or grouped the assets went from 1 of 30 to 17 of 18.
- Haiku ignores instructions but reacts to numbers. Telling it to check
  whether a column works within each sector changed nothing. Printing the
  result of that check in the data summary it reads at the start improved its
  forecasts on 6 of 8 new datasets.
- The agents cannot peek at the answers. We checked the logs of all 472 runs
  and told one agent to try to break in; every attempt was blocked.

## Which results are a fair test

Some changes to the instructions were written after we had looked at the
dataset they were then tested on. That limits what the result can show:

| change | written while looking at | tested on | what the result shows |
|---|---|---|---|
| improved instructions (Section 1) | `planted1` | `planted1b`, same hidden rules | only that models follow instructions |
| extra numbers in the data summary (Section 5) | `exam3`, after the exam | new random datasets with the same kind of sector rule | that it carries over to new datasets of that kind |
| separate planning call (Section 3) | `planted8` | a variant of `planted8`, the exam | that agents then run the checks; too few runs to judge scores |
| sealed exam (Section 4) | nothing: instructions frozen, datasets built by someone else | | a fair test |

One earlier version told the analyst to check whether its strongest column is
explained by "the other features and their squares". `planted3`'s hidden rule
is a square, so that line gave away the answer. We dropped those results.

## Results

### 1. Telling the agent what to look for

Our first improved instructions added two checks: see whether a column
predicts returns only on some days (for example when another column is
positive), and when two columns are correlated, fit both and see which one
still matters. These are exactly the two rules hidden in `planted1`, the
dataset we wrote them on: `x07` matters only when `x11` is positive, and `x05`
is a noisy copy of `x03`. On a second dataset with the same rules and
different random numbers (`planted1b`), the new instructions raised the score
by 0.11 to 0.15 for Haiku, Sonnet and Opus. This shows the models follow
instructions. It does not show they discover anything.

To test discovery we made `planted3`, whose rules the instructions do not
describe. The main rule is a U-shape: `y` rises with `x04` squared, so large
values of `x04` in either direction mean higher returns, and a plain
correlation between `x04` and `y` is close to zero. The trap column `x12` is
`x04` squared plus noise. Haiku, Sonnet and Opus missed the U-shape in all 15
runs, and their models used the trap column in all 15. Adding a general hint
("a correlation near zero does not mean no effect: look at averages by
decile, longer lags, other assets") did not help them find it. When we gave a
forecaster a note that names all three rules, it matched the oracle, so the
models can build the right model; they just do not find the rules.

### 2. The reference list makes the model forget what it knows

Later versions of the instructions end with a reference list of research
practice: how the score works, fat tails, removing the average return of each
day, columns that stand in for other columns, effects that depend on the
state of the market, testing many ideas at once. We wanted to know whether
this list adds to what the model would check on its own, or replaces it.

So we sent the model one message, with no data and no tools: the dataset's
description and a request to list the structures it would check for in a
panel of asset returns, with a concrete check for each. We asked 5 times per setting and counted
the answers that mention PCA and the ones that mention grouping assets into
industries.

| model | reference list in the message | answers with PCA | answers with industries |
|---|---|---|---|
| Haiku 5.5 | no | 5/5 | 5/5 |
| Haiku 5.5 | yes, in any position | 0/5 | 0–2/5 |
| Haiku 5.5 | yes, and asked to write its own list first | 2/5 | 1/5 |
| Haiku 5.5 | yes, introduced as "in addition to what you know" | 1/5 | 1/5 |
| Opus 5.5 | any of the above | 5/5 | 5/5 |

Haiku's answers have 15 to 17 items with or without the list, so the list's
items take the place of its own instead of adding to them. Opus keeps its own
items in the answer, but when it then works on real data it does not act on
them: on `planted8`, a dataset where returns are driven by two common factors
and industry groups, none of 6 Opus runs did PCA.

### 3. Letting the model plan before it reads the list

The fix is to ask for the model's plan before the list is in front of it.
Before the agent starts, the code makes one extra call to the same model. That
call gets only the dataset's description and the question above, with no
reference list, and the model answers with its own checklist. The agent then
starts as usual, with its normal instructions and the reference list, but the
checklist is placed at the top with the instruction to run every check on it
and report which ones showed something.

On `planted8`, Opus and Haiku runs whose code actually computed PCA or
grouped the assets went from 1 of 30 without the extra call to 17 of 18 with
it. Whether this also gives better forecasts we cannot say yet. On `planted8`
the scores stayed near zero either way, and on a variant with stronger
industry groups we have only 3 runs per setting, too few to tell. On the exam
datasets, which have no common factors, 10 Haiku runs per setting showed no
difference in score either way.

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
