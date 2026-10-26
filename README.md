# agentic-research-harness

A multi-agent research system built on the Claude Agent SDK, plus the eval
harness that tells whether a change to an agent made it better. The agents do
quantitative research: they explore a table of data, write up findings and
build a forecasting model. The data is synthetic, with hidden rules that a
generator writes down, so every claim an agent makes and every forecast it
files can be graded against the truth.

## The agents

- **Three roles.** The *analyst* explores the data and writes findings. The
  *ideator* proposes ideas from a data summary without seeing raw data. The
  *forecaster* reads the notes and files a model as a Python file. Each agent
  is one Markdown file whose frontmatter lists its tools and the kinds of
  note it may write ([src/arh/agents/](src/arh/agents/)).
- **Shared notes instead of messages.** Agents never talk directly. They read
  and write notes in a SQLite store with full-text search, exposed to them as
  an MCP server. Notes cite other notes, and the store checks permissions:
  the ideator has no shell, and a forecast filed by the ideator is rejected.
  Because agents share only the store, each one can be evaluated alone.
- **Sandboxed shell.** Analyst and forecaster run Python through Bash inside
  the Claude Code OS sandbox: read access to their own folder only, no
  network, no host settings or CLAUDE.md, a private temp dir, and one BLAS
  thread so parallel runs do not starve each other.
- **Isolated scoring.** A forecast is a model file. The harness runs it in a
  separate sandboxed process that talks over stdin/stdout and receives one
  day of data at a time, so the model cannot look ahead. The agent sees
  cross-validation scores on its training data; the score on held-back data
  is stored and never returned to it.
- **Limits and traces.** Every run has a turn limit, a cost cap and a
  timeout, and every message goes to a JSONL trace that the graders and the
  access audit read.
- **Planning stage.** A `plan:` line in an agent's frontmatter makes the
  runner call the model once, before the agent starts, to write its own
  checklist; the agent then runs with that checklist at the top
  ([why](docs/results.md#2-the-reference-list-makes-the-model-forget-what-it-knows)).
- **Any model.** Non-Claude models (here gpt-oss) run through Ollama's
  Anthropic-compatible API with the same tools.

## The evals

A generator script builds each dataset: 40 assets over 700 days, 16 numeric
columns and the next-day return. It plants a few rules, adds a trap column
that only looks useful (a noisy copy of a real one), writes both to a hidden
file, and holds back the last 200 days. One eval is one agent, a fixed set of
starting notes, one dataset and one prompt version; `arh eval` runs it N times
in parallel and grades every run. Prompt versions sit side by side in
`evals/prompts/`, so an ablation is a grid of model × prompt × dataset.

**How the agents work together, and how that is tested.** In a full
research run (`arh loop`) analysts, ideators and forecasters take turns over
one shared note store: analysts post findings, the ideator posts ideas and
questions built on them, and the forecaster reads everything and files a
model. To benchmark this, an eval runs one agent at a time and fixes what the
others would have written as its starting notes. That makes a handoff
repeatable: the same Haiku forecaster scores 0.81 from the data alone and 0.95
in 5 of 5 runs when handed a finding written by an Opus analyst
([section 7](docs/results.md#7-handing-notes-between-agents)). We have not
benchmarked the full loop end to end.

Every run is graded on four groups of metrics ([src/arh/evals.py](src/arh/evals.py)).
The last group, forecast quality, applies only to forecasters.

**How did it work?** Whether the run finished and filed exactly one note of
an allowed kind; turns, tool calls, tool error rate, repeated identical calls,
whether it read notes before writing, models scored before filing, output
tokens, wall time and cost. Pattern checks over its code record which
analyses it ran (regime split, lags, nonlinearity, cross-asset, multiple
testing). Per-model tables are in
[section 6](docs/results.md#6-how-each-model-uses-its-tools).

**Is the note backed by work it did?** The share of numbers in the note that
appear in the agent's own tool output; whether the notes it cites were
actually returned to it by a tool; whether a forecaster reports the CV score
the harness measured; and for an analyst, whether its script, rerun in the
sandbox, prints the numbers in its note. An Opus judge also rates each note
1 to 5 for specificity, evidence, actionability and clarity, and, given the
hidden rules, as correct, partial or wrong.

**Did it find the truth?** For each hidden rule, whether the agent's note
names it (all the rule's columns plus a pattern for its form, so naming `x04`
alone does not count as finding a U-shape in `x04`), and for a forecaster
whether its model code actually uses those columns. A note can say "sector"
while the model never reads the sector column, so the second check is the
one we report. Also: whether the trap column is an input to the model
(comments do not count), and how many pure-noise columns the note mentions.

**Forecast quality.** The main benchmark number is the *score*: the
forecaster's R² on the 200 held-back days divided by the R² of the best
possible forecast, the *oracle*, which computes the expected return from the
hidden rules. 1.0 means the agent matched the oracle, 0 means it did no better
than predicting the average. Dividing by the oracle makes datasets with weak
and strong signal comparable. Since the forecaster builds on the notes it is
given, this score measures the analyst's work too when an analyst's finding
is the starting note. We also record the daily rank correlation between
forecast and return and the Sharpe ratio of a long-short portfolio built from
the forecast. On datasets with no signal, the raw R² is reported, since there
is nothing to divide by.

Each setting runs 3 to 10 times. Tables give the mean ± the standard
deviation over runs, and comparisons between settings give a 95% interval
from a bootstrap over runs and blocks of held-back days.

Two checks guard the benchmark itself. `scripts/audit_access.py` scans every
trace for reads of hidden files: across 472 runs none happened, and an agent
told to break in was refused 76 times. And a sealed exam, six datasets built
by a separate agent and run once with frozen prompts
([protocol](docs/exam.md)), is the only test the prompts were not tuned on.

## What we learned

The plan was to make the agents better by telling them more. Later prompts
ended with a reference list of quant research practice: how the score works,
fat tails, removing each day's average, columns that stand in for others,
effects that depend on the market's state, testing many ideas at once. This
backfired. With the list in the prompt, Haiku stopped proposing methods it
proposes on its own. Asked which checks it would run on a dataset, it named
PCA in 5 of 5 answers without the list and in 0 of 5 with it. Its answers
stayed the same length, so the list's items replaced its own ideas instead of
adding to them. Opus kept its own ideas in the plan but did not act on them:
on a dataset driven by common factors, none of 6 Opus runs did PCA.

More instructions did not fix this. Telling the model to write its own list
first, or introducing the reference as "in addition to what you know", barely
moved it. What worked was changing the structure around the model. The
runner now asks for the model's own checklist in a separate call that never
sees the reference list, and puts that checklist at the top of the agent's
prompt. Agents went from running PCA or grouping assets in 1 of 30 runs to 17
of 18 ([details](docs/results.md#2-the-reference-list-makes-the-model-forget-what-it-knows)).

The same held for a single instruction. Haiku failed one exam dataset because
it never compared a column with its sector's average on the same day, and a
prompt line telling it to do that changed nothing. Computing the comparison
and adding the numbers to the data summary every agent starts with did work:
on new datasets with a sector rule the score rose by 0.15 for Haiku (95%
interval 0.04 to 0.26) and 0.16 for Sonnet (0.06 to 0.27), at a small cost on
datasets without one ([details](docs/results.md#5-why-haiku-fails-exam3)).

We also made mistakes in the evals. Our first improved prompt raised
scores by 0.11 to 0.15, but we had written it while looking at the test
dataset, and its two new checks were exactly the two hidden rules. On a
dataset whose rules no prompt describes, Haiku, Sonnet and Opus missed the
main rule in 15 of 15 runs ([details](docs/results.md#1-telling-the-agent-what-to-look-for)).
We now record what each change was written against, and count only the sealed
exam as a fair test: six datasets built by a separate agent and run once with
frozen prompts, where Opus came within 8% of the best possible forecast on
every dataset with a pattern ([details](docs/results.md#4-sealed-exam)). The
graders needed the same scrutiny: a text-match check credited Haiku with a
rule in 10 of 13 runs that never used it, because the notes said "group".

## Code

| file | job |
|---|---|
| `runner.py` | one agent run: prompt, planning stage, tools, sandbox, limits, trace |
| `tools.py` | the store and scoring as MCP tools, scoped per agent |
| `store.py` | notes (full-text search), runs, scored models, requests |
| `score.py`, `worker.py` | walk-forward CV, hidden holdout, sandboxed model process |
| `summary.py` | the data summary every agent starts with |
| `evals.py` | eval cells, graders, LLM judge, scorecards |
| `cli.py` | `arh init / run / loop / eval / regrade / report` |
| `worlds/` | dataset generators |

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
scripts/reproduce.sh     # rebuilds the datasets and tables from results/, no model calls
```

Running agents calls the models (Haiku costs about $0.005 to $0.07 per run):

```bash
.venv/bin/arh eval all --spec evals/planted3.json --model claude-haiku-5-5 --n 5 \
  --max-turns 15 --prompts evals/prompts/cur --out runs/evals/planted3/haiku.jsonl
.venv/bin/arh report --scorecard runs/evals/planted3/haiku.jsonl
```
