# Detailed results

See the [README](../README.md) for what the code does. Scores are the
agent's forecast quality on the held-back days divided by the oracle's, so
1.0 matches the best possible forecast. ± is the standard deviation over runs
unless stated.

## Which results are a fair test

Some changes to the instructions were written after we had looked at the
dataset they were then tested on. That limits what the result can show:

| change | written while looking at | tested on | what the result shows |
|---|---|---|---|
| improved instructions (section 1) | `planted1` | `planted1b`, same hidden rules | only that models follow instructions |
| extra numbers in the data summary (section 5) | `exam3`, after the exam | new random datasets with the same kind of sector rule | that it carries over to new datasets of that kind |
| separate planning call (section 3) | `planted8` | a variant of `planted8`, the exam | that agents then run the checks; too few runs to judge scores |
| sealed exam (section 4) | nothing: instructions frozen, datasets built by someone else | | a fair test |

One earlier version told the analyst to check whether its strongest column is
explained by "the other features and their squares". `planted3`'s hidden rule
is a square, so that line gave away the answer. We dropped those results.

## 1. Telling the agent what to look for

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

## 2. The reference list makes the model forget what it knows

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

## 3. Letting the model plan before it reads the list

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

## 4. Sealed exam

A separate agent built six new datasets (`worlds/exam/exam1` to `exam6`) and
sealed their hidden rules. Nobody working on the instructions looked at them
before the run, and the instructions were frozen beforehand
([protocol and file hashes](exam.md)). We ran two versions of the forecaster
(the normal one, and the one with the planning call), 3 times each per
dataset, with Haiku and Opus: 72 runs, all listed in
[results/exam/ledger.jsonl](../results/exam/ledger.jsonl). A dash is a run that
ran out of turns or time.

| dataset | Haiku | Haiku + planning call | Opus | Opus + planning call |
|---|---|---|---|---|
| exam1 | 0.64, 0.67, 0.67 | 0.61, 0.67, 0.85 | 0.96, 1.00, 0.99 | 0.99, 0.96, 1.00 |
| exam2 | 0.55, 0.90, 0.56 | 0.78, 0.89, – | 1.04, 1.04, 1.04 | 1.04, 1.04, 1.04 |
| exam3 | 0.08, 0.04, 0.11 | 0.14, 0.09, 0.03 | 0.96, 0.95, 0.94 | 0.92, 0.92, – |
| exam5 | 0.91, 1.12, 1.06 | 1.04, 1.29, 1.10 | 1.19, 1.11, 1.33 | 1.29, 1.01, 1.18 |
| exam6 | 0.95, 0.81, 0.84 | 0.89, 0.93, 0.69 | 0.96, 0.96, – | 0.96, 0.96, 0.96 |

`exam4` has no pattern at all, so there is no oracle to divide by. Every
forecast there explained about none of the returns on the held-back days
(R² between −0.008 and 0.007), so no model invented a pattern that was not
there.

Opus is close to the oracle on every dataset. Haiku is lower everywhere and
fails `exam3`. Scores above 1 have two causes, found after the run. On `exam2`
the oracle ignores past returns, which also carry the signal, so it is not
quite the best possible forecast; with past returns added, Opus is at 1.01.
On `exam5` the oracle happens to do badly on the held-back days (it explains
2.6% of the variance there against 3.8% on the training days), and the
forecasts are within that noise. Opus runs cost about $2 to $3 each, Haiku
about $0.07.

## 5. Why Haiku fails exam3

In `exam3`, column `x11` predicts returns only relative to the other assets
in the same sector on the same day (column `x06` is the sector). Every Opus
run computed that difference; no Haiku run did. One Haiku run checked the idea
the wrong way, by grouping on sector across all days instead of per day, saw
that the sector explained little, and dropped it. Haiku did not run out of
turns (it used 32 to 44 of 60). Given the note an Opus run wrote, Haiku built
the right model in 7 of 7 runs, so it can do it when told what to do. Adding a
line to its instructions telling it to check features within each sector and
day changed nothing (0 of 3 runs).

What did work was a number instead of an instruction. Every agent starts with
a short summary of each column that the code computes. We added one line to
it: for every column that looks like a group label (fixed per asset, few
values), how much of each feature's daily variation is shared within the
group, and how well the feature predicts once the group's average is
subtracted. The line names no feature and is computed the same way for any
dataset (`ARH_SUMMARY=groups`).

Haiku on `exam3`, 10 runs each (± is the spread over runs). The group line
raised the score by +0.39 (95% interval +0.08 to +0.70):

| data summary | score | runs at 0.7 or more |
|---|---|---|
| normal | 0.18 ± 0.20 | 1/10 |
| with the group line | 0.54 ± 0.41 | 6/10 |
| with the group line + planning call | 0.43 ± 0.36 | 4/10 |

We wrote the line after seeing `exam3`, so we also tested it on new random
datasets from `worlds/make_random.py`: 8 with a sector rule and 6 without, 3
runs per setting. The table gives the average change in score with a 95%
interval. The interval comes from a bootstrap that resamples runs and blocks
of 10 held-back days, with both settings scored on the same days:

| datasets | Haiku | Sonnet |
|---|---|---|
| 8 with a sector rule | +0.15 (+0.04 to +0.26), better on 6 of 8 | +0.16 (+0.06 to +0.27), better on 8 of 8 |
| 6 without one | −0.06 (−0.17 to +0.03), better on 2 of 6 | −0.05 (−0.13 to +0.04), 2 datasets run |

The gain varies a lot between datasets (standard deviation 0.24 for Haiku),
but the average is clearly above zero.

It helps where there is a sector rule and costs a little where there is none,
because Haiku then sometimes builds on a group that predicts nothing. The test
datasets have the same kind of sector rule as `exam3`, so this does not show
the idea works for other kinds of rule.

## Can the agents see the answers?

The agents run Python in the Claude Code sandbox. It blocks reading any file
under the home directory except the run's own folder and the Python install,
and blocks the network. `scripts/audit_access.py` went through the logs of 472
runs (1,248 shell commands): none touched a hidden-rule file, held-back data
or generator script, and every attempt to read outside the run's folder was
refused. We also told one Haiku agent to try to break in by any means
(relative and absolute paths, Python `open()`, `find`, symlinks). It was
refused 76 times, and its log contains nothing from the hidden files.

## Limitations

- 3 to 10 runs per setting, and we did not compute how large a difference
  these could detect. A difference of one or two runs out of five means
  nothing.
- Only the sealed exam is a fair test, and it has no common factors or
  industries.
- Whether a rule was found is checked with text patterns over the agent's
  notes and code. An early version credited Haiku with the sector rule in 10
  of 13 runs that never used it, because their notes said "group"; the check
  now also requires the rule's columns to appear in the model code.
