"""From a finished run's findings to a graded mechanism theory.

One Haiku call turns the run's findings and the template library (arh.mechanisms) into a theory, claim by
claim. The theory is scored by walk-forward CV with the necessity of every instance; instances whose necessity
is not above one standard error are dropped and the rest refitted. Both theories are scored on the holdout and
graded against the world's planted mechanisms (the grade reads the sealed truth; nothing upstream does).

    python scripts/theory_from_findings.py runs/r19-tree2 worlds/rand/r19 out/r19-theory
"""

import asyncio
import json
import re
import sqlite3
import sys
from pathlib import Path

import polars as pl

from arh import mechanisms, theory
from arh.runner import one_turn

MODEL = "claude-haiku-5-5"

PROMPT = """A research team studied a panel dataset (columns date, asset, features {feats}, target y = next-period \
return). Below are their findings, then a library of mechanism templates. Write the theory the findings support: \
the target transforms and the terms, each a template with its parameters filled in. First go through the findings \
claim by claim and say, for each claim about how y depends on the data, which template expresses it, or "none" if \
no template can. Then put every expressible claim into the theory, including ones the findings rate as weaker: each \
piece is tested afterwards on its own and dropped if it does not help, so leaving a claim out costs more than \
putting it in.

# Findings
{findings}

# Templates
Kinds: "target" changes what is fitted; "term" adds a column to the fit. Parameter types: feature = a column name, \
series = "frame" (y after the target transforms) or a column name.
{library}

Answer with JSON only: {{"claims": [{{"finding": <id>, "claim": "...", "template": "<label or none>"}}], \
"targets": [{{"template": ..., "params": {{...}}}}], "terms": [...], "why": {{"<template label>": "<finding ids and \
the evidence>"}}}}"""


def findings(run: Path) -> str:
    db = sqlite3.connect(run / "store.db")
    rows = db.execute("select id, text from nodes where kind='finding' and status='done' and move!='playbook' "
                      "order by id").fetchall()
    return "\n\n".join(f"## Finding #{i}\n{t}" for i, t in rows)


async def propose(prompt: str, feats: list[str]) -> dict:
    """The model's theory; one retry with the validation error if an instance is invalid."""
    text = await one_turn(prompt, MODEL)
    for attempt in range(2):
        spec = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        spec = {"targets": spec.get("targets", []), "terms": spec.get("terms", [])}
        try:
            for _, inst in theory.instances(spec):
                mechanisms.check(inst, feats)
            return spec
        except ValueError as e:
            if attempt:
                raise
            text = await one_turn(f"{prompt}\n\nYour previous answer:\n{text}\n\nIt was invalid: {e}. Fix it.", MODEL)


def report(title: str, spec: dict, res: dict, world: Path) -> None:
    print(f"\n# {title}")
    for r in res["instances"]:
        (n, se), (a, _) = r["necessity"], r["sufficiency"]
        print(f"{r['instance']:55s} necessity {n:+.4f} ± {se:.4f}  alone {a:+.4f}")
    g = theory.grade(spec, res, world)
    print(f"cv {res['cv']:+.4f}  holdout {res['holdout']:+.4f}  recall {g['recall']:.2f}  spurious {g['spurious']}")


def main(run: Path, world: Path, out: Path) -> None:
    train = pl.read_parquet(world / "public" / "train.parquet")
    feats = [c for c in train.columns if c not in ("date", "asset", "y")]
    library = "\n".join(f"- {n} ({t.kind}): {t.says} Parameters: {t.params}; defaults {t.defaults}"
                        for n, t in mechanisms.TEMPLATES.items())
    spec = asyncio.run(propose(PROMPT.format(feats=", ".join(feats), findings=findings(run), library=library), feats))
    res = theory.evaluate(spec, train, out / "proposed", world)
    report("proposed", spec, res, world)

    keep = [(p, i) for (p, i), r in zip(theory.instances(spec), res["instances"]) if r["necessity"][0] > r["necessity"][1]]
    pruned = {part: [i for p, i in keep if p == part] for part in ("targets", "terms")}
    res = theory.evaluate(pruned, train, out / "pruned", world)
    report("pruned (necessity > 1 SE on training CV)", pruned, res, world)
    (out / "theory.json").write_text(json.dumps({"proposed": spec, "pruned": pruned}, indent=1))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
