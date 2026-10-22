# agentic-research-harness

Three LLM research agents share one SQLite note store and nothing else. Each
agent gets its own eval on synthetic worlds where the true mechanisms are
known, so a prompt change can be measured on one agent instead of judged by
watching a whole loop.

The task is panel forecasting: 40 assets × 500 dates, 16 features, predict
next-period `y`. The core is about 780 lines of Python, plus a 410-line eval
module. Its predecessor, an 8-stage orchestrated chain (~8k lines), had no
way to tell which stage was failing. This repo is the rewrite.

## Design

- **Agents never call each other.** They coordinate only through notes
  (`finding`, `idea`, `question`, `forecast`) in a shared store with FTS5
  search. Any agent can run alone, any number of times, in any mix, in
  parallel.
- **Agents never score themselves.** Model code runs in a separate sandboxed
  worker that is fed one date at a time. Agents see walk-forward CV on their
  training data. The holdout score is recorded but never shown to them.
- **One eval per agent, on frozen inputs.** An eval condition is one agent
  plus the notes it starts from, run on a world with a sealed `truth.json`.
  To judge a prompt edit, rerun that agent's eval, not the loop.

| agent | sees | tools | writes |
|---|---|---|---|
| analyst | data, all notes | sandboxed Bash, store | finding |
| ideator | data summary, notes (no raw data) | store | idea, question |
| forecaster | data, all notes | sandboxed Bash, store, `score_model` | forecast |

| file | job |
|---|---|
| `store.py` | notes (FTS5), runs, trials |
| `tools.py` | store and scoring as MCP tools, scoped per agent |
| `runner.py` | one agent run: prompt, tools, sandbox, trace |
| `score.py`, `worker.py` | walk-forward CV, hidden holdout, sandboxed model worker |
| `evals.py` | eval conditions, graders, scorecards |
| `agents/*.md` | one prompt per agent (frontmatter: tools, writable kinds) |

The agents run on the Claude Agent SDK. Non-Claude models (gpt-oss) go
through Ollama's Anthropic-compatible API.

## What the evals measure

Each run is graded against the world's sealed truth.

- **Accuracy**
  - mechanisms found: the right features named and the pattern described, in
    the note or the model code;
  - decoy used in the model;
  - forecaster holdout R² as a fraction of the oracle's (the true conditional
    mean).
- **Process**
  - protocol followed (exactly one note of an allowed kind);
  - turns, tool calls, tool error rate, redundant calls;
  - which exploration checks were run (regime split, lags, nonlinearity,
    cross-asset).
- **Honesty**
  - cited notes were actually read;
  - the note's numbers appear in tool output;
  - the analyst's script reruns and reproduces its numbers.
- **Judge**: Opus 5.5 scores specificity, evidence, actionability and
  correctness against the truth.

## Results

![results](docs/results.png)

Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.1 and gpt-oss (local), with 15 turns
per run and n=5 per cell (n=3 for gpt-oss). Ranges are over runs. MDE was
not computed, so differences of one or two runs out of five are not evidence.

### 1. Model × prompt ablation

`v1` is the first prompt. `cur` adds the three lines that eval iterations on
`planted1` earned:
- regime split;
- a proxy check on correlated pairs;
- the data summary in the system prompt.

Prompts were tuned on `planted1`. The columns below are:
- **planted1b**: a held-out seed, never tuned on. Mechanisms: a rank effect,
  x07 active only when the date-level x11 > 0, and momentum; x05 is a decoy.
- **planted2**: a control with no regime effect.

| model | prompt | forecaster holdout/oracle (planted1b) | forecaster x07 gate | forecaster uses decoy | analyst x07 gate | control holdout/oracle (planted2) | $/run |
|---|---|---|---|---|---|---|---|
| gpt-oss-32k | v1 | 0.64 ± 0.14 (n=3) | 0/3 | 0/3 | 0/3 | 0.90 ± 0.00 | local |
| gpt-oss-32k | cur | 0.63 ± 0.14 (n=3) | 0/3 | 0/3 | 0/3 | 0.90 ± 0.00 | local |
| claude-haiku-5-5 | v1 | 0.70 ± 0.08 (n=5) | 1/5 | 4/5 | 0/5 | 0.90 ± 0.00 | $0.003 |
| claude-haiku-5-5 | cur | 0.81 ± 0.00 (n=5) | 5/5 | 0/5 | 4/5 | 0.90 ± 0.02 | $0.005 |
| claude-sonnet-5-5 | v1 | 0.74 ± 0.00 (n=5) | 4/5 | 5/5 | 4/5 | 0.91 ± 0.00 | $0.043 |
| claude-sonnet-5-5 | cur | 0.85 ± 0.06 (n=5) | 5/5 | 4/5 | 5/5 | 0.91 ± 0.00 | $0.048 |
| claude-opus-5-5 | v1 | 0.80 ± 0.08 (n=5) | 5/5 | 3/5 | 5/5 | 0.92 ± 0.01 | $0.153 |
| claude-opus-5-5 | cur | 0.95 ± 0.00 (n=5) | 5/5 | 0/5 | 5/5 | 0.91 ± 0.00 | $0.149 |
| claude-fable-5-1 | v1 | 0.95 ± 0.00 (n=5) | 5/5 | 1/5 | 3/5 | 0.92 ± 0.03 | $0.594 |
| claude-fable-5-1 | cur | 0.92 ± 0.06 (n=5) | 5/5 | 0/5 | 5/5 | 0.91 ± 0.01 | $0.532 |

The prompt lines transfer to the held-out seed.
- Haiku, Sonnet and Opus gain 0.11–0.15 of oracle.
- Haiku's x07 gate goes from 1/5 to 5/5.
- Fable already finds the gate under `v1`, and the lines don't help it.
- gpt-oss never finds the gate under either prompt.
- On the control, which has no gate to find, every model scores 0.90–0.92,
  so the regime line does not make agents invent effects.
- Separately, a Haiku forecaster given one Opus analyst finding scores
  0.95 × 5. Cheap forecasters can ride on an expensive analyst through the
  store.

### 2. How each model works

These are the same runs as the ablation table, pooled over planted1b and
planted2 (n=10 per cell, 6 for gpt-oss). Cells show means per run.
- **numbers grounded**: share of the note's numbers that appear in tool output.
- **rerun ok**: the analyst's script reproduces its numbers.
- **judge correct**: share judged correct against the truth.

| model | prompt | agent | n | turns | tool calls | Bash | tool errors | trials | wall s | out tokens | numbers grounded | rerun ok | judge evidence | judge correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-32k | v1 | analyst | 6 | 9.8 | 9.5 | 6.2 | 0.33 | - | 356 | 1758 | 0.67 | 3/6 | 3.5 | 0.75 |
| gpt-oss-32k | v1 | forecaster | 6 | 3.8 | 2.8 | 0.0 | 0.00 | 0.0 | 290 | 682 | 0.00 | - | 1.2 | 0.00 |
| gpt-oss-32k | cur | analyst | 6 | 10.2 | 9.2 | 5.0 | 0.52 | - | 305 | 2101 | 0.17 | 2/6 | 3.3 | 0.33 |
| gpt-oss-32k | cur | forecaster | 6 | 5.0 | 4.0 | 0.3 | 0.15 | 0.2 | 240 | 899 | 0.00 | - | 1.2 | 0.00 |
| claude-haiku-5-5 | v1 | analyst | 10 | 8.2 | 7.2 | 3.8 | 0.09 | - | 70 | 4033 | 0.98 | 10/10 | 4.9 | 0.90 |
| claude-haiku-5-5 | v1 | forecaster | 10 | 7.8 | 6.8 | 1.0 | 0.00 | 1.0 | 35 | 1971 | 0.70 | - | 3.9 | 0.30 |
| claude-haiku-5-5 | cur | analyst | 10 | 8.8 | 7.8 | 4.3 | 0.15 | - | 41 | 4675 | 0.99 | 8/10 | 4.7 | 0.80 |
| claude-haiku-5-5 | cur | forecaster | 10 | 9.6 | 8.6 | 2.5 | 0.03 | 1.1 | 42 | 2933 | 0.90 | - | 4.3 | 0.70 |
| claude-sonnet-5-5 | v1 | analyst | 10 | 5.6 | 4.6 | 1.6 | 0.04 | - | 31 | 2297 | 0.97 | 9/10 | 4.6 | 0.70 |
| claude-sonnet-5-5 | v1 | forecaster | 10 | 5.0 | 4.0 | 1.0 | 0.00 | 1.0 | 37 | 1030 | 0.97 | - | 3.4 | 0.30 |
| claude-sonnet-5-5 | cur | analyst | 10 | 6.0 | 5.0 | 2.0 | 0.06 | - | 30 | 2567 | 0.97 | 10/10 | 4.9 | 0.70 |
| claude-sonnet-5-5 | cur | forecaster | 10 | 7.0 | 6.0 | 3.0 | 0.17 | 1.0 | 32 | 1675 | 0.99 | - | 3.7 | 0.80 |
| claude-opus-5-5 | v1 | analyst | 10 | 6.2 | 5.2 | 2.1 | 0.03 | - | 42 | 3512 | 0.98 | 10/10 | 4.9 | 1.00 |
| claude-opus-5-5 | v1 | forecaster | 10 | 8.6 | 7.6 | 3.5 | 0.10 | 1.0 | 61 | 3132 | 0.94 | - | 4.1 | 0.40 |
| claude-opus-5-5 | cur | analyst | 10 | 6.0 | 5.0 | 2.0 | 0.07 | - | 35 | 2919 | 0.95 | 10/10 | 5.0 | 1.00 |
| claude-opus-5-5 | cur | forecaster | 10 | 8.7 | 7.7 | 3.7 | 0.07 | 1.0 | 57 | 2912 | 0.97 | - | 4.9 | 0.90 |
| claude-fable-5-1 | v1 | analyst | 10 | 7.2 | 6.2 | 3.2 | 0.03 | - | 86 | 5820 | 0.96 | 10/10 | 5.0 | 1.00 |
| claude-fable-5-1 | v1 | forecaster | 10 | 10.6 | 9.6 | 4.2 | 0.02 | 1.0 | 155 | 7213 | 0.98 | - | 4.6 | 0.60 |
| claude-fable-5-1 | cur | analyst | 10 | 7.5 | 6.5 | 3.4 | 0.15 | - | 73 | 4908 | 0.98 | 10/10 | 5.0 | 1.00 |
| claude-fable-5-1 | cur | forecaster | 10 | 9.4 | 8.4 | 3.1 | 0.02 | 1.3 | 87 | 4985 | 0.95 | - | 4.9 | 0.60 |

What changes by model:

- **No forecaster searches.** Every Claude forecaster submits about one model
  to `score_model` (trials ≈ 1.0) and stops. CV is used as a final check, not
  as a search signal. So the prompt gains come from what the forecaster looks
  at before modelling, not from trying more models.
- **The `cur` prompt changes what forecasters do, not just what they say.**
  Bash calls per forecaster rise for Haiku (1.0 → 2.5) and Sonnet
  (1.0 → 3.0). Judge-correct rises for Haiku (0.3 → 0.7), Sonnet (0.3 → 0.8)
  and Opus (0.4 → 0.9). Analyst effort barely moves.
- **Cost tracks tokens, not quality, at the top end.** Fable writes the most
  (5–7k output tokens, 87–155 s per forecaster run). It costs about 3.5× Opus
  per run and is no better under `cur`. Sonnet is the leanest (5–7 turns,
  about 30 s), but its forecaster still keeps the decoy in 4/5 runs.
- **Claude models report honestly; gpt-oss does not do the work.** Claude
  notes have 90–99% of their numbers grounded in tool output, and analyst
  scripts rerun and reproduce in 8–10 of 10 runs. gpt-oss forecasters almost
  never open the data: they fit a linear regression on all 16 features
  (0–0.3 Bash calls per run, nothing grounded, judge evidence 1.2/5). Under
  `cur`, gpt-oss analysts make more tool errors (0.33 → 0.52) and ground
  fewer numbers (0.67 → 0.17). The longer prompt hurts the small local model.

### 3. planted3: mechanisms no prompt mentions

This world tests whether the gains above are just "the prompt names the
answer". It has three mechanisms that the prompts' checklist (regime split,
proxy pair, lag-1) cannot find:
- **U-shape**: 0.30·(x04² − 1), which has a linear correlation of about 0;
- **lag 2**: x10 at t−2;
- **lead-lag**: assets 20–39 follow asset a−20's previous return.

The decoy is x12 = x04² + noise, which a linear screen picks up instead of
x04. planted3b is the held-out copy: another seed, with the mechanisms moved
to x08 and x14 and the decoy to x01.

| world | arm | analyst finds U-shape | analyst finds lag 2 | forecaster holdout/oracle | forecaster uses decoy |
|---|---|---|---|---|---|
| planted3 | Haiku, current prompts | 0/5 | 1/5 | 0.51–0.51 | 5/5 |
| planted3 | Sonnet, current prompts | 0/5 | 0/5 | 0.51–0.54 | 5/5 |
| planted3 | Opus, current prompts | 0/5 | 0/5 | 0.55–0.56 | 5/5 |
| planted3 | Haiku, no data summary | 0/5 | 0/5 | 0.33–0.55 | 4/5 |
| planted3 | Haiku, + EDA line | 0/5 | 3/5 | 0.37–0.56 | 5/5 |
| planted3 | Haiku, + EDA line, no summary | 0/5 | 2/5 | 0.51–0.63 | 5/5 |
| planted3 | Haiku, + confound move | 5/5 | 2/5 | 0.51–0.63 | 5/5 |
| planted3b | Haiku, current prompts | 0/5 | 0/5 | 0.51–0.51 | 4/5 |
| planted3b | Haiku, + confound move | 4/5 | 0/5 | 0.42–0.61 | 4/5 |
| planted3 | forecaster given one move-arm finding | - | - | 0.56–0.63 | 2/5 |
| planted3 | forecaster given a finding naming all three | - | - | 1.00–1.00 | 0/5 |

What the arms showed, in order:

1. **All three Claude models fail the same way.** They rank features by
   linear correlation, take x12 as the top signal, and never look at x04.
   Opus is no better than Haiku here.
2. **The data summary is not the cause.** Removing it changes nothing,
   because the agents rebuild the same linear screen themselves.
3. **A generic EDA line doesn't fix it either.** The line was
   "correlation ≈ 0 is not no effect: check deciles, lags > 1, other assets".
   Agents run more checks (mean 3 → 5.2, lags checked 5/5) and find lag 2
   more often, but still miss the U-shape.
4. **A "confound move" does fix it.** The line is "ask what explains your
   strongest feature: regress it on other features and their squares". With
   it, analysts find x12 ≈ x04² (residual correlation ≈ 0.001) in 5/5 runs.
   On the held-out planted3b it works in 4/5.
5. **The forecaster then models the truth perfectly once it is told it.**
   Given a note naming all three mechanisms, it scores 1.00 of oracle × 5.
   Given one real move-arm finding (U-shape only), it scores 0.56–0.63, and
   the decoy drops from 5/5 to 2/5.

The bottleneck is discovery, not modelling. A forecaster gains only as much
as the store's findings cover. That points the next work at running many
analysts in parallel on different directions, rather than at a better
forecaster prompt.

### Grader bugs found along the way

The evals are code too, and these graders were wrong until fixed:
- `decoy_in_model` counted comments such as `# x05 dropped` as using the
  decoy;
- `numbers_grounded` ignored numbers that came from the task text;
- the judge confused a measured autocorrelation with the true coefficient.

Also, quick mode (1 turn, notes inlined) tracks mechanism naming but not
forecaster holdout (0.37 vs 0.68 agentic). It is fine for plumbing, but not
for ranking forecaster prompts.

## Limitations

- **n=5 per cell and MDE not computed.** Treat 4/5 vs 5/5 as a tie.
- **Loose regex detection.** U-shape, lag-2 and lead-lag detection uses
  regexes over the note and code. Lead-lag hits in particular are unreliable.
- **The confound move may be a hint.** The "and their squares" wording is
  close to the planted mechanism. planted3b moves the features but keeps the
  same functional form, so a world with a different nonlinearity (e.g. |x|
  or a threshold) is the real test.
- **Synthetic worlds only.** The one real dataset (`rank100v2`) has no truth
  file, so it only shows that the prompts do no harm there (holdout R² 0.042
  vs 0.041).

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
scripts/reproduce.sh          # rebuilds worlds/planted*, the tables and docs/results.png from results/ (no model calls)
.venv/bin/pytest -q
```

Rerunning the evals calls models; Haiku costs about $0.005 per run.

```bash
# one cell: agent condition x model x prompt version
.venv/bin/arh eval all --spec evals/planted3.json --model claude-haiku-5-5 --n 5 --parallel 5 \
  --max-turns 15 --prompts evals/prompts/move --out runs/evals/planted3/haiku-move.jsonl
.venv/bin/arh report --scorecard runs/evals/planted3/haiku-move.jsonl
scripts/ablation.sh           # the full model x prompt grid (~$30)
```

`ARH_SUMMARY=none` runs the no-summary arm. `evals/*.json` defines the
conditions, `evals/prompts/{v1,cur,eda,move}` holds the prompt versions, and `PLAN.md` has the full eval log.

## Next

Run several analysts in parallel, each assigned a direction (nonlinearity,
lags, cross-asset, regimes). Then measure whether coverage of the truth
grows with the number of agents, compared with unassigned agents that
duplicate each other.
