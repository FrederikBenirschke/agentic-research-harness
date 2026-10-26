# agentic-research-harness

Can LLM agents do quantitative research: given a table of data, find the
patterns in it and build a forecast? To check their answers we need to know
the truth, so the code makes the data itself.

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

Details, tables and every number: [docs/results.md](docs/results.md).

- **Fair test.** On six datasets built by a separate agent and not looked at
  before the run, Opus came within 8% of the oracle on every dataset with a
  pattern, and no model invented a pattern in the one without.
  ([section 4](docs/results.md#4-sealed-exam))
- **The models do not discover rules they are not told about.** A U-shaped
  rule (returns rise with a column squared) was missed in 15 of 15 Haiku,
  Sonnet and Opus runs. Instructions that named the hidden rules helped, but
  that only shows the models follow instructions.
  ([section 1](docs/results.md#1-telling-the-agent-what-to-look-for))
- **A reference list in the instructions made Haiku forget methods it knew.**
  Asked which checks it would run, Haiku named PCA in 5 of 5 answers without
  the list and 0 of 5 with it. Asking for the model's own checklist in a
  separate call first got agents to run those checks (17 of 18 runs, against
  1 of 30). Whether that improves the forecasts is not yet clear.
  ([sections 2 and 3](docs/results.md#2-the-reference-list-makes-the-model-forget-what-it-knows))
- **Haiku ignores instructions but uses numbers.** Telling it to check a
  column within each sector did nothing. Printing that check's result in the
  data summary improved its score on 6 of 8 new datasets with a sector rule
  (+0.15 ± 0.24) and cost a little on datasets without one.
  ([section 5](docs/results.md#5-why-haiku-fails-exam3))
- **The agents cannot see the answers.** Logs of 472 runs and a deliberate
  break-in attempt show no read of the hidden files.
  ([details](docs/results.md#can-the-agents-see-the-answers))

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

The agents' instructions are in `src/arh/agents/`, other versions in
`evals/prompts/`. The dataset generators are in `worlds/`. The full log of
every eval is [docs/eval-log.md](docs/eval-log.md).
