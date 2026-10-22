# Eval log

This log records each prompt change and its scores, the graders, and the note
and model formats.

## Prompt changes

All runs use Haiku 5.5 with 15 turns and n=3 per cell unless noted. We did not
compute the MDE.

The prompt snapshots are in `runs/evals/{analyst,forecaster}-v*.md`.

| change | effect |
|---|---|
| v2: data summary in every system prompt | agents had skipped note #1 when searching by kind; no x07 gain alone |
| v3: "check whether the effect depends on a regime (date-level feature above vs below its median)" | planted1 x07×x11 found 0/3 to 3/3 (analyst and forecaster), forecaster holdout/oracle 0.51 to 0.68; planted2 control: regime split run and correctly reported null 6/6; rank100v2 holdout unchanged (0.042 vs 0.041) |
| v4 analyst: "for a correlated pair, fit y on both and report which keeps its effect" | decoy flagged as proxy: planted2 0/3 to 3/3, planted1 0/3 to 1/3. Forecaster version had no effect (v3 already drops decoys 6/6), reverted |
| held-out seed planted1b (never tuned on), n=4, v1 vs current, an earlier batch than the README ablation | x07×x11 found: analyst 0/4 to 3/4, forecaster 1/4 to 3/4; decoy in model 3/4 to 0/4; holdout/oracle 0.74 to 0.73 (no gain: linear x07 already gets most of it with the gate on 28% of holdout dates; one run split on the wrong date-level variable, 0.56) |

The next table is for planted3, whose mechanisms no prompt names: a U-shape in x04, x10 at lag 2, and a
cross-asset lead-lag. The decoy is x12 = x04² + noise. All arms use Haiku 5.5 with 15 turns and n=5 per arm.
planted3b uses seed 13 and moves the mechanisms to x08 and x14 and the decoy to x01.

| arm | analyst U-shape | forecaster holdout/oracle | x12 decoy in model |
|---|---|---|---|
| cur (also Sonnet, Opus: U-shape 0/5 analysts and 0/5 forecasters each, 0.51–0.56) | 0/5 | 0.51 | 5/5 |
| no data summary | 0/5 | 0.33–0.55 | 4/5 |
| eda line ("corr ≈ 0 is not no effect: deciles, lags > 1, other assets") | 0/5 | 0.37–0.56 | 5/5 |
| move line ("ask what explains your strongest feature: regress it on the others and their squares") | 5/5 | 0.51–0.63 | 5/5 |
| planted3b cur / move | 0/5 to 4/5 | 0.51 / 0.42–0.61 | 4/5 / 4/5 |
| chain: forecaster given one move-arm finding (U-shape only) | - | 0.56–0.63 | 2/5 |
| control: forecaster given a finding naming all three | - | 1.00 ×5 | 0/5 |

Agents select x12 because they rank features by linear correlation. Removing the summary does not change this,
because they compute the same ranking themselves. The v4 proxy instruction does not apply, since
corr(x04, x12) ≈ 0. The move line makes analysts find the U-shape in 5/5 runs, and forecasters given the same line find it in 1/5.
A forecaster told all three mechanisms scores 1.00 of oracle, so the score is limited by which mechanisms are
found.

The README describes quick mode and the grader bugs.

## What is graded

Run with `arh eval <condition|all> --model <model>`, summarise with `arh report`.

A condition (`evals/planted1.json`) is one agent plus the frozen notes it
starts from, on `worlds/planted1` (known mechanisms in `sealed/truth.json`:
rank effect of x03, x07 gated by date-constant x11, lag-1 momentum; x05 decoy).

| Group | Metrics |
|---|---|
| process | finished, protocol_ok (exactly one note of an allowed kind), turns, wall time, tool calls per tool, tool error rate, redundant-call rate, read-before-write, trials |
| grounding | refs_grounded (cited notes were seen), numbers_grounded (note's numbers appear in tool output), cv_reported_ok, rerun_ok / rerun_numbers (analyst script reruns and reproduces) |
| accuracy | mechanisms named (note + model code), decoy mentioned / in model, noise features named, cv, holdout, holdout / oracle |
| judge | specificity, evidence, actionability, clarity, correct vs truth (Opus 5.5 by default, or an Ollama model at temperature 0) |

`arh metrics <run>` computes the process and grounding metrics for any research run.
Local models go through Ollama's Anthropic-compatible API (any model not
named `claude-*`).

## Note format

`kind` is one of `summary` (written by the harness), `finding`, `idea`,
`question`, `forecast`. A note has text, `refs` (ids of notes it builds on or
answers), an optional artifact (a copy of a script or model file), and for
forecasts a CV score. A question is open if no finding refs it, and an idea is
open if no forecast refs it.

The ideator asks for analysis by writing a `question`, and the analyst prompt
asks analysts to answer open questions first.

## Model interface

```python
def fit(train: pl.DataFrame) -> Any: ...
def predict(state, history: pl.DataFrame, today: pl.DataFrame) -> np.ndarray: ...
```

`history` has every row (with `y`) before `today`; `today` has no `y`.
