# agentic-research-harness

A multi-agent research system built on the Claude Agent SDK, plus the eval
harness that tells whether a change to an agent made it better. The agents do
quantitative research: they explore a table of data, write up findings and
build a forecasting model. The data is synthetic, with hidden rules that a
generator writes down, so every claim an agent makes and every forecast it
files can be graded against the truth.

## How it evolved

Each stage started from a problem the previous one exposed, and fixed it by
changing the structure around the agents rather than adding instructions.
Scores are holdout R² divided by the oracle's (the best possible forecast,
computed from the hidden rules), so 1.0 is perfect; the agents and the evals
are described below.

1. **Role agents over a shared store.** Three agents (analyst, ideator,
   forecaster), each one Markdown file, that share only a SQLite note store
   exposed as an MCP server. Sandboxed shell, sandboxed scoring that feeds the
   model one day at a time, traces for everything. An audit of 472 runs found
   no read of a hidden file.
2. **The evals caught our own cheating.** Our first improved prompt raised
   scores by 0.11 to 0.15, but its two new checks were exactly the two hidden
   rules of the dataset we wrote it on. On a dataset no prompt describes,
   Haiku, Sonnet and Opus missed the main rule in 15 of 15 runs. Since then
   every change records what it was written against, and only a sealed exam
   (datasets built by a separate agent, run once with frozen prompts) counts as
   a fair test. Opus came within 8% of the oracle on every exam dataset with a
   pattern ([details](docs/results.md#1-telling-the-agent-what-to-look-for)).
3. **Instructions displace knowledge.** A reference list of quant practice in
   the prompt made Haiku stop proposing PCA (5 of 5 answers without it, 0 of 5
   with it). Rewording did not help; moving the model's own checklist into a
   separate call that never sees the list did: agents ran PCA or grouped
   assets in 17 of 18 runs, against 1 of 30
   ([details](docs/results.md#2-the-reference-list-makes-the-model-forget-what-it-knows)).
4. **Tools beat instructions.** A prompt line telling Haiku to compare a
   column with its sector average changed nothing; computing that comparison
   into the data summary raised the score on new sector datasets by 0.15 for
   Haiku and 0.16 for Sonnet ([details](docs/results.md#5-why-haiku-fails-exam3)).
5. **Handoffs.** Because agents share only the store, a handoff can be fixed
   and replayed. A Haiku forecaster scores 0.81 alone and 0.95 in 5 of 5 runs
   when handed one finding by an Opus analyst: much of the gap between models
   is in the research, not the modelling
   ([details](docs/results.md#7-handing-notes-between-agents)).
6. **Tree search.** `arh search` replaces fixed roles with a tree of notes
   and model files grown by a bandit over typed moves (survey, residuals,
   draft, act on a finding, simplify, merge, ...). It reaches 0.94 to 0.97 of
   the oracle on one random world and 0.66 to 0.80 on a harder one, but does
   not yet beat a greedy search, and its screening tool turned out to mirror
   the generator's families, so it now runs with a `--no-screen` control arm
   ([details](docs/results.md#8-tree-search-over-models)).
7. **From a score to an explanation.** A good model file does not say why it
   works. Models can now be a sum of named pieces, each with a claim and a
   measured necessity (CV lost when it is dropped). One Haiku call turns a
   tree's findings into such a theory; pruning pieces that do not earn their
   place gives, on one world, four pieces that find both planted rules with no
   false positive and score 0.147 against an oracle of 0.153, as well as the
   best free-code model ([details](docs/results.md#9-from-a-score-to-an-explanation)).

Three lessons carried through:

- **Structure beats instructions.** Separate calls, computed summaries and
  typed moves changed behaviour; prompt lines mostly did not.
- **Check every change for leaking the answer.** A prompt line, a summary
  statistic or a tool package written after seeing a dataset measures
  instruction following on that dataset, not discovery.
- **Grade the graders.** A text-match check credited Haiku with a sector rule
  in 10 of 13 runs that never used it; the rule now has to appear in the
  model code. The same kind of check still credits a lead-lag rule too
  easily (section 9).

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

## Code

| file | job |
|---|---|
| `runner.py` | one agent run: prompt, planning stage, tools, sandbox, limits, trace |
| `tools.py` | the store and scoring as MCP tools, scoped per agent |
| `store.py` | notes (full-text search), runs, scored models, requests |
| `score.py`, `worker.py` | walk-forward CV, hidden holdout, sandboxed model process |
| `summary.py` | the data summary every agent starts with |
| `search.py`, `screen.py` | tree search over notes and models; the feature screen |
| `mechanisms.py`, `theory.py` | mechanism templates, `Pieces` models, necessity, grading against the truth |
| `evals.py` | eval cells, graders, LLM judge, scorecards |
| `cli.py` | `arh init / run / loop / search / eval / regrade / report` |
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

A tree search, then a theory from its findings (about $0.35 to $0.55 for 60 Haiku steps):

```bash
.venv/bin/python worlds/make_random.py 19     # a random world; the draw depends only on the seed
.venv/bin/arh search r19-tree --world worlds/rand/r19 --steps 60 --parallel 3
.venv/bin/python scripts/theory_from_findings.py runs/r19-tree worlds/rand/r19 runs/r19-tree/theory
```

## Next steps

- A prune move inside the tree, so the search itself keeps only pieces with
  positive necessity, and sufficiency tables for `Pieces` models.
- A full `--pieces` search against free code on several worlds and seeds.
- A template for gating on a feature's own trailing correlation, the gap that
  made the `r112` theory fail.
- A benchmark of tree against greedy search on 6 worlds × 3 seeds, with
  families the screen covers reported apart from those it does not.
