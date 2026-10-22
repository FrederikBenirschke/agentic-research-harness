"""arh init | run | loop | status | notes | metrics | eval | report"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import os
import signal
import subprocess
from pathlib import Path

import polars as pl

from arh.runner import DEFAULT_MODEL, run_agent
from arh.store import Store, render
from arh.summary import summarize

RUNS = Path(__file__).resolve().parents[2] / "runs"


def init(name: str, world: Path) -> Path:
    root = RUNS / name
    if root.exists():
        raise SystemExit(f"{root} exists")
    (root / "work" / "data").mkdir(parents=True)
    shutil.copy2(world / "public" / "train.parquet", root / "work" / "data" / "train.parquet")
    for extra in ("README.md",):
        if (world / "public" / extra).exists():
            shutil.copy2(world / "public" / extra, root / "work" / "data" / extra)
    (root / "config.json").write_text(json.dumps({"world": str(world.resolve())}, indent=2))
    Store(root / "store.db").add("summary", "harness", summarize(pl.read_parquet(world / "public" / "train.parquet")))
    return root


async def loop(root: Path, mix: dict[str, int], rounds: int, parallel: int, model: str | None) -> None:
    gate = asyncio.Semaphore(parallel)

    async def one(agent: str) -> None:
        async with gate:
            run = await run_agent(root, agent, model=model)
            print(f"done {run}", flush=True)

    for r in range(rounds):
        batch = [a for a, n in mix.items() for _ in range(n)]
        random.shuffle(batch)
        print(f"round {r + 1}/{rounds}: {batch}", flush=True)
        await asyncio.gather(*(one(a) for a in batch))


def status(root: Path) -> None:
    store = Store(root / "store.db")
    runs = store.runs()
    cost = sum(r["cost"] or 0 for r in runs)
    print(f"notes {store.counts()}  runs {len(runs)}  cost ${cost:.2f}")
    for r in runs:
        print(f"  {r['id']:12} {r['agent']:10} turns {r['turns']}  ${r['cost'] or 0:.2f}  {r['error'] or ''}")
    print("forecasts (holdout is harness-only):")
    for f in store.forecasts():
        cv = "  -   " if f["cv"] is None else f"{f['cv']:+.4f}"
        ho = "  -   " if f["holdout"] is None else f"{f['holdout']:+.4f}"
        print(f"  #{f['id']:<4} cv {cv}  holdout {ho}  {f['text']}")


def _terminate(*_) -> None:
    """On SIGTERM, also stop the agent CLI subprocesses; otherwise they keep running on the local model."""
    subprocess.run(["pkill", "-TERM", "-P", str(os.getpid())])
    os._exit(143)


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate)
    p = argparse.ArgumentParser(prog="arh")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("name")
    s.add_argument("--world", type=Path, required=True)
    for cmd in ("run", "loop", "status", "notes"):
        s = sub.add_parser(cmd)
        s.add_argument("name")
        if cmd in ("run", "loop"):
            s.add_argument("--model", default=DEFAULT_MODEL)
    sub.choices["run"].add_argument("agent")
    sub.choices["run"].add_argument("--task", default="")
    sub.choices["loop"].add_argument("--mix", default="analyst:1,ideator:1,forecaster:1")
    sub.choices["loop"].add_argument("--rounds", type=int, default=1)
    sub.choices["loop"].add_argument("--parallel", type=int, default=3)
    sub.choices["notes"].add_argument("--kind")
    sub.choices["notes"].add_argument("--query", default="")
    s = sub.add_parser("metrics", help="process and grounding metrics for every agent run of a research run")
    s.add_argument("name")
    s.add_argument("--judge", help="Ollama model to grade each note")
    s = sub.add_parser("eval", help="run one agent N times on a frozen eval condition and grade each run")
    s.add_argument("condition")
    s.add_argument("--spec", type=Path, default=RUNS.parent / "evals" / "planted1.json")
    s.add_argument("--model", default="gpt-oss-32k")
    s.add_argument("--n", type=int, default=1)
    s.add_argument("--parallel", type=int, default=4)
    s.add_argument("--judge", default="claude-opus-5-5", help="claude-* or Ollama model, or 'none'")
    s.add_argument("--max-turns", type=int, default=0, help="0: quick, fewest turns per agent, notes inlined")
    s.add_argument("--prompts", type=Path, help="dir of prompt versions to use instead of agents/")
    s.add_argument("--out", type=Path, help="results file (default runs/evals/results.jsonl)")
    s = sub.add_parser("regrade", help="grade recorded eval runs again with the current grader")
    s.add_argument("files", nargs="+", type=Path)
    s.add_argument("--judge", default="claude-opus-5-5")
    s.add_argument("--out-dir", type=Path, default=RUNS / "evals" / "regraded")
    s = sub.add_parser("report", help="aggregate runs/evals/results.jsonl by condition and model")
    s.add_argument("files", nargs="*", type=Path, help="results files; with --scorecard, a table per agent")
    s.add_argument("--scorecard", action="store_true")
    a = p.parse_args()

    if a.cmd in ("metrics", "eval", "report", "regrade"):
        from arh import evals
        if a.cmd == "metrics":
            root = RUNS / a.name
            for r in Store(root / "store.db").runs():
                print(json.dumps(evals.grade(root, r["id"], a.judge)))
        elif a.cmd == "eval":
            (RUNS / "evals").mkdir(parents=True, exist_ok=True)
            conds = list(json.loads(a.spec.read_text())["conditions"]) if a.condition == "all" else [a.condition]
            asyncio.run(evals.run_conditions(a.spec, conds, a.model, a.n, None if a.judge == "none" else a.judge,
                                             a.parallel, RUNS, a.max_turns, prompts=a.prompts, out=a.out))
            print(evals.report(a.out or RUNS / "evals" / "results.jsonl"))
        elif a.cmd == "regrade":
            evals.regrade(a.files, None if a.judge == "none" else a.judge, a.out_dir)
        elif a.scorecard:
            print(evals.scorecard(a.files or sorted((RUNS / "evals").glob("results-*.jsonl"))))
        else:
            for f in a.files or [RUNS / "evals" / "results.jsonl"]:
                print(evals.report(f))
        return

    if a.cmd == "init":
        print(init(a.name, a.world))
        return
    root = RUNS / a.name
    if a.cmd == "run":
        print(asyncio.run(run_agent(root, a.agent, a.task, a.model)))
    elif a.cmd == "loop":
        mix = {k: int(v) for k, v in (x.split(":") for x in a.mix.split(","))}
        asyncio.run(loop(root, mix, a.rounds, a.parallel, a.model))
    elif a.cmd == "status":
        status(root)
    elif a.cmd == "notes":
        for n in Store(root / "store.db").search(a.query, a.kind, k=200):
            print(render(n, full=True), "\n")


if __name__ == "__main__":
    main()
