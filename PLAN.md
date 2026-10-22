# Plan

A minimal research harness: independent agents, one shared store, one eval per
agent. The sibling repo `agentic-research-loop` is the counter-example: an
8-stage chain per cycle, ~8k lines, most of the effort spent on scaffolding
rather than on making any one agent better.

## Principles

- **Agents never call each other.** They coordinate only through notes in the
  store. Any agent runs alone, any number of times, in any mix, in parallel.
- **Improve agents one at a time.** Each agent has its own eval on frozen
  inputs; a prompt change is judged by rerunning that agent's eval, not a full
  loop.
- **The agent never scores itself.** Model code runs in a separate,
  sandboxed process fed one date at a time. Agents see only walk-forward CV on
  their training data; the holdout score is recorded but never shown to them.
- **Small.** Core under ~1000 lines. Anything not earning its keep stays out.

## Pieces

| File | Job |
|---|---|
| `store.py` | SQLite notes table with FTS5 search; runs and trials tables |
| `tools.py` | The store and scoring as MCP tools, scoped per agent |
| `runner.py` | Run one agent: prompt + tools + sandbox, trace, record the run |
| `score.py` / `worker.py` | Walk-forward CV and hidden holdout; the sandboxed model worker |
| `summary.py` | Deterministic data summary, stored as note #1 of a run |
| `cli.py` | `arh init / run / loop / status / notes` |
| `agents/*.md` | One prompt per agent: frontmatter (tools, writable kinds) + text |

## Notes

`kind` is one of `summary` (harness), `finding`, `idea`, `question`, `forecast`.
A note has text, `refs` (ids it builds on or answers), an optional artifact
(frozen copy of a script or model), and for forecasts a CV score. "Open" is
computed: a question nobody's finding refs, an idea no forecast refs.

## Agents

| Agent | Sees | Tools | Writes |
|---|---|---|---|
| analyst | data, all notes | Bash (sandboxed), store | finding |
| ideator | data summary, all notes, no raw data | store | idea, question |
| forecaster | data, all notes | Bash, store, `score_model` | forecast |

The ideator asks for analysis it needs by writing a `question`; analysts work
open questions first. Optional later: a distiller that merges/rewrites notes,
only if the store gets too noisy for search.

## Model interface

```python
def fit(train: pl.DataFrame) -> Any: ...
def predict(state, history: pl.DataFrame, today: pl.DataFrame) -> np.ndarray: ...
```

`history` has every row (with `y`) before `today`; `today` has no `y`.

## Milestones

1. Store, runner, scoring, forecaster alone on `rank100v2`. Target: match or
   beat the old loop's ~0.049 holdout R² with a tenth of the code.
2. Analyst + ideator; does a forecaster with notes beat one without?
3. Per-agent evals on planted-signal worlds (analyst: finds the structure;
   ideator: names the mechanism; forecaster: holdout R² vs oracle given the
   right idea). Iterate prompts.
4. Distiller, only if milestone 2 shows store noise.

## Evals (`arh eval <condition|all> --model <ollama model>`; `arh report`)

A condition (`evals/planted1.json`) is one agent plus the frozen notes it
starts from, on `worlds/planted1` (known mechanisms in `sealed/truth.json`:
rank effect of x03, x07 gated by date-constant x11, lag-1 momentum; x05 decoy).

| Group | Metrics |
|---|---|
| process | finished, protocol_ok (exactly one note of an allowed kind), turns, wall time, tool calls per tool, tool error rate, redundant-call rate, read-before-write, trials |
| honesty | refs_grounded (cited notes were seen), numbers_grounded (note's numbers appear in tool output), cv_reported_ok, rerun_ok / rerun_numbers (analyst script reruns and reproduces) |
| accuracy | mechanisms named (note + model code), decoy mentioned / in model, noise features named, cv, holdout, holdout / oracle |
| judge | specificity, evidence, actionability, clarity, correct vs truth (Ollama model, temperature 0) |

`arh metrics <run>` gives process + honesty for any research run.
Local models go through Ollama's Anthropic-compatible API (any model not
named `claude-*`).

## Eval log (Haiku 5.5, 15 turns, n=3 per cell unless noted; MDE not computed)

Prompt snapshots in `runs/evals/{analyst,forecaster}-v*.md`.

| change | effect |
|---|---|
| v2: data summary in every system prompt | agents had skipped note #1 when searching by kind; no x07 gain alone |
| v3: "check whether the effect depends on a regime (date-level feature above vs below its median)" | planted1 x07×x11 found 0/3 → 3/3 (analyst and forecaster), forecaster holdout/oracle 0.51 → 0.68; planted2 control: regime split run and correctly reported null 6/6; rank100v2 holdout unchanged (0.042 vs 0.041) |
| v4 analyst: "for a correlated pair, fit y on both and report which keeps its effect" | decoy flagged as proxy: planted2 0/3 → 3/3, planted1 0/3 → 1/3. Forecaster version had no effect (v3 already drops decoys 6/6), reverted |
| held-out seed planted1b (never tuned on), n=4, v1 vs current | x07×x11 found: analyst 0/4 → 3/4, forecaster 1/4 → 3/4; decoy in model 3/4 → 0/4; holdout/oracle 0.74 → 0.73 (no gain: linear x07 already gets most of it with the gate on 28% of holdout dates; one run split on the wrong date-level variable, 0.56) |

planted3 (mechanisms no prompt names: U-shape x04², x10 at lag 2, cross-asset lead-lag; decoy x12 = x04² + noise),
Haiku 5.5, t15, n=5 per arm. planted3b = seed 13, mechanisms moved to x08 / x14, decoy x01.

| arm | analyst U-shape | forecaster holdout/oracle | x12 decoy in model |
|---|---|---|---|
| cur (also Sonnet, Opus: U-shape 0/10 each, 0.51–0.56) | 0/5 | 0.51 | 5/5 |
| no data summary | 0/5 | 0.33–0.55 | 4/5 |
| eda line ("corr ≈ 0 is not no effect: deciles, lags > 1, other assets") | 0/5 | 0.37–0.56 | 5/5 |
| move line ("ask what explains your strongest feature: regress it on the others and their squares") | 5/5 | 0.51–0.63 | 5/5 |
| planted3b cur / move | 0/5 → 4/5 | 0.51 / 0.42–0.61 | 4/5 / 4/5 |
| chain: forecaster given one move-arm finding (U-shape only) | — | 0.56–0.63 | 2/5 |
| control: forecaster given a finding naming all three | — | 1.00 ×5 | 0/5 |

The anchor is the linear screen, not the summary: agents rebuild it themselves. The v4 proxy rule cannot fire
(corr(x04, x12) ≈ 0). The move line fixes discovery in analysts but forecasters ignore it; the forecaster models
the truth perfectly once told it, so the bottleneck is discovery, and the chain is limited by what one finding covers.

Quick mode (1 turn, notes inlined) tracks mechanism naming but not forecaster holdout
(v3 quick 0.37 vs agentic 0.68): use it for the ideator and plumbing, not to rank forecaster prompts.
Grader bugs found along the way: judge confused a measured autocorrelation with the true coefficient;
`numbers_grounded` ignored the task text; `decoy_in_model` counted comments ("# x05 dropped").

## Out on purpose

LangGraph, judge, skeptic, preregistration, a second backend, checkpoint and
resume, multiple memory stores, provenance manifests.
