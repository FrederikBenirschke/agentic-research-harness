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

- **Known truth.** A generator script builds each dataset (40 assets, 700
  days, 16 columns, next-day return), plants a few rules and a trap column
  that only looks useful, writes them to a hidden file, and holds back the
  last 200 days.
- **Eval = one agent × fixed starting notes × dataset × prompt version.**
  `arh eval` runs a cell N times in parallel and grades each run. Prompt
  versions live side by side in `evals/prompts/`, so an ablation is a grid of
  model × prompt × dataset.
- **Graders.** Forecast score against the best possible forecast (the
  *oracle*, computed from the hidden rules); which hidden rules the agent
  names and which it actually uses in its model code; whether it fell for
  the trap; whether the numbers in its note appear in its tool output and the
  notes it cites were returned to it; whether the analyst's script reproduces
  its numbers on rerun; an Opus judge rating each note against the truth; and
  process metrics (turns, tool calls, errors, models tried, cost).
- **Access audit.** `scripts/audit_access.py` scans every trace for reads of
  hidden files. Across 472 runs none happened, and an agent told to break in
  was refused 76 times.
- **Sealed exam.** Six datasets built by a separate agent, run once with
  frozen prompts ([protocol](docs/exam.md)).

## What the evals caught

Full tables: [docs/results.md](docs/results.md).

- **Prompts that leak the answer.** Instructions written while looking at a
  dataset raised scores on that kind of dataset by 0.11 to 0.15, but on a
  dataset whose rules no prompt describes, Haiku, Sonnet and Opus all missed
  the main rule in 15 of 15 runs
  ([section 1](docs/results.md#1-telling-the-agent-what-to-look-for)).
- **Context that pushes out what the model knows.** A reference list of
  methods in the prompt made Haiku drop PCA from 5 of 5 plans to 0 of 5. The
  planning stage above was the fix: agents then ran those checks in 17 of 18
  runs instead of 1 of 30. Whether forecasts improve is not settled yet
  ([section 3](docs/results.md#3-letting-the-model-plan-before-it-reads-the-list)).
- **Small models follow numbers, not instructions.** Telling Haiku to check
  a column within each sector did nothing. Adding that check's result to the
  data summary the harness computes raised its score on 6 of 8 new datasets
  with a sector rule (+0.15 ± 0.24)
  ([section 5](docs/results.md#5-why-haiku-fails-exam3)).
- **Graders that lie.** A text-match grader credited Haiku with a rule in 10
  of 13 runs that never used it; the grader now also requires the rule's
  columns in the model code. An earlier bug counted a comment like
  `# x05 dropped` as using the trap column.
- **Sealed exam.** Opus came within 8% of the oracle on every exam dataset
  with a pattern, and no model invented a pattern in the one without
  ([section 4](docs/results.md#4-sealed-exam)).

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
