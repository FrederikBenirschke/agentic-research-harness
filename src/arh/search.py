"""Tree search over models and findings, run by the harness (after AIDE and R&D-Agent).

Many step agents work in parallel on one shared tree. Every node is one step's note, and its kind
says what the step produced: a model (a forecast model with its CV and out-of-sample residuals;
hypothesis, result, learned), a finding (a claim about the data with the script that shows it), or
an idea (a proposal read off the ledger, nothing run).

Each step, the harness picks a move from a menu with a Thompson-sampling bandit and the nodes it
starts from, then a bounded searcher agent does that one step. Analysis moves file a finding, idea
moves an idea, model moves a model; the `act` move changes a model using a given finding or idea. A
model wins when it beats its parent; a finding or idea wins when a model built on it beats its
parent, so every node earns its place by what it leads to. Every step sees a ledger of the tree,
so what one branch learned reaches the others.

Nodes live in the run's store.db (table nodes); models and residuals in work/nodes/, readable by
every agent.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from arh import mechanisms as mech, score, theory
from arh.runner import run_agent
from arh.store import Store

# Generic moves, about method, never about a particular dataset. Each move needs a starting model
# node ("model"), a finding or idea to work from ("input"), both, or neither.
ANALYSIS = {
    "survey": ("Screen every feature, including those the ledger calls noise, with every package of the screen "
               "tool (python ../lib/screen.py raw cs lag cslag inter shape cond --target TARGET), then check the "
               "strongest and most surprising lines with your own script: is the relation linear, does it hold "
               "over time and across assets?", ()),
    "factor": ("Fit common factor models to the target and study what they leave (python ../lib/screen.py "
               "factor --target TARGET: principal components, and stepwise peeling of the strongest features, each "
               "followed by a screen of the residual). Then say which factors a model should use and what in the "
               "residual is still structured, and check that with your own script.", ("model?",)),
    "explore": ("Study the data for structure that no finding in the ledger covers yet.", ()),
    "residuals": ("Study where the starting node's residuals are large or structured, and what that says "
                  "the model is missing: screen them with the packages that fit (python ../lib/screen.py ... "
                  "--target TARGET), then dig into what stands out.", ("model",)),
    "check": ("Test the given note's claim another way (other periods, assets or ranges, another statistic, a "
              "control): does it hold, how strong is it, and where does it break?", ("input",)),
}
IDEAS = {
    "orthogonal": ("Propose one approach to forecasting y unlike any model or idea in the ledger.", ()),
    "grounded": ("Propose one change to the starting node that follows from the findings and residuals in the "
                 "ledger.", ("model",)),
}
MODELS = {
    "draft": ("Write a new model from scratch, built on an idea different from the models in the ledger.", ()),
    "act": ("Change the starting node to use what the given note (a finding or an idea) says.", ("model", "input")),
    "feature": ("Find an input the starting node does not use (a transformation, an interaction, a use of "
                "history) and add it.", ("model",)),
    "conditioning": ("Make the starting node's relation between the features and y depend on part of the data "
                     "(periods, assets, ranges of a feature) where that helps.", ("model",)),
    "form": ("Change the starting node's model form (functional form, model class, regularisation), keeping its "
             "inputs.", ("model",)),
    "simplify": ("Remove parts of the starting node that do not earn their place, and check whether its CV "
                 "holds.", ("model",)),
    "merge": ("Take what works in another model of the ledger and combine it with the starting node.", ("model",)),
}
FREE = {"free": ("Choose the step you think is most useful: study the data, or change the starting node.",
                 ("model",))}
MOVES = {**ANALYSIS, **IDEAS, **MODELS, **FREE, "playbook": ("", ())}

# Standard ideas of the field, filed as idea nodes when a tree is made, so `act` can apply them like any idea
# and the search learns which pay off. They are general practice, never about a particular dataset; the step
# that applies one judges whether and how it fits the starting model.
PLAYBOOK = [
    "Idea: Normalise by volatility. Fit y divided by a forecast of its size (per asset, per date, or from "
    "features that predict |y|), or weight the fit by the inverse of that variance, then map back to y.\nWhy: "
    "when the size of y varies, a fit on raw y is dominated by the noisiest rows.\nTest: screen |y| (screen.py "
    "vol); a model fitted this way should beat the starting node's CV.",
    "Idea: Remove common factors. Take the market, sector or principal-component part out of y, or out of the "
    "features, model what is left, and add back the factor part only if it is predictable.\nWhy: common moves "
    "can hide asset-specific signal and make features look weaker or stronger than they are.\nTest: screen the "
    "factor residual (screen.py factor); a model of it should beat the starting node's CV.",
    "Idea: Mean reversion. Treat a level as an Ornstein-Uhlenbeck / AR(1) process per asset: a feature, or the "
    "cumulative sum over a trailing window of what the node models (y, or y net of factors when the node removes "
    "them, as in statistical arbitrage). Estimate its speed and mean, and use the deviation from the mean in "
    "units of its equilibrium sd (the s-score) as a forecast or feature.\nWhy: a level that reverts pulls the "
    "next change towards its mean.\nTest: screen.py ou, with --target set to what the node models, gives "
    "half-lives and the s-score; a model using it should beat the starting node's CV.",
    "Idea: Change what is fitted to a cross-sectional target. Fit the within-date rank or z-score of y, with "
    "features chosen and standardised for that target, then rescale the forecast to y.\nWhy: a rank or z "
    "target ignores the date-wide level and outliers and keeps the ordering across assets.\nTest: the rescaled "
    "forecast should beat the starting node's CV.",
    "Idea: Split direction and size. Predict the sign of y with a classifier and its magnitude with a "
    "regressor, and combine them into a forecast of y.\nWhy: what drives the direction and what drives the "
    "size can differ.\nTest: screen sign(y) and |y| (screen.py vol); the combined forecast should beat the "
    "starting node's CV.",
]
SCREEN = Path(__file__).with_name("screen.py")   # copied to work/lib/ for every agent, as is mechanisms.py
IDEA_TURNS = 6   # an idea step reads the ledger and may glance at the data; it does not build anything

FILE_FINDING = ("Do not build a forecast model in this step. End with one add_note(kind=\"finding\", "
                "artifact=<your script>) whose text has three short parts:\n\nClaim: what you found, with the "
                "numbers.\nEvidence: how you tested it, and how sure it is.\nImplication: what a model should do "
                "about it, if anything.")
FILE_IDEA = ("Do not build or score a model, and look at the data only as much as a quick check needs. End "
             "with one add_note(kind=\"idea\") whose text has three short parts:\n\nIdea: what to try.\nWhy: what "
             "in the ledger or the data suggests it, or what it covers that nothing else does.\nTest: what model "
             "change would test it, and what result would show it wrong.")
FILE_FORECAST = ("End with one add_note(kind=\"forecast\", artifact=\"model.py\") whose text has three short "
                 "parts:\n\nHypothesis: what you expected and why.\nResult: the CV you saw, compared with the "
                 "starting node.\nLearned: what this tells the team about the data.\n\nFile it even if it did "
                 "not beat the starting node.")
FILE_EITHER = ("If you built a model, end with one add_note(kind=\"forecast\", artifact=\"model.py\", text=\""
               "Hypothesis: ... Result: ... Learned: ...\"); file it even if it did not beat the starting node. "
               "If you only studied the data, end with one add_note(kind=\"finding\", artifact=<your script>, "
               "text=\"Claim: ... Evidence: ... Implication: ...\").")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY, kind TEXT, parent INTEGER, uses INTEGER, move TEXT, run TEXT, status TEXT,
    note INTEGER, cv REAL, folds TEXT, ic REAL, holdout REAL, same INTEGER, text TEXT, error TEXT, frame TEXT,
    pieces TEXT);
"""
DEFAULT_FRAME = "y; the features as given"


class Tree:
    def __init__(self, root: Path):
        self.root, self.store = root, Store(root / "store.db")
        self.dir = root / "work" / "nodes"
        self.dir.mkdir(exist_ok=True)
        (root / "work" / "lib").mkdir(exist_ok=True)
        shutil.copy2(SCREEN, root / "work" / "lib" / "screen.py")
        shutil.copy2(mech.__file__, root / "work" / "lib" / "mechanisms.py")
        with self.store._db() as db:
            db.executescript(SCHEMA)
            cols = {r[1] for r in db.execute("PRAGMA table_info(nodes)")}
            for c in ("frame", "pieces"):   # trees made before these existed
                if c not in cols:
                    db.execute(f"ALTER TABLE nodes ADD COLUMN {c} TEXT")

    def seed(self, playbook: list[str]) -> None:
        """File the playbook as idea nodes, once per tree."""
        with self.store._db() as db:
            if not db.execute("SELECT 1 FROM nodes WHERE move = 'playbook'").fetchone():
                db.executemany("INSERT INTO nodes (kind, move, status, text, frame) VALUES ('idea', 'playbook', "
                               "'done', ?, 'y')", [(t,) for t in playbook])

    def reset(self) -> None:
        """Steps left running by an earlier, killed search never finish."""
        with self.store._db() as db:
            db.execute("UPDATE nodes SET status = 'failed', error = 'interrupted' WHERE status = 'running'")

    def nodes(self) -> list[dict]:
        with self.store._db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM nodes ORDER BY id")]

    def open(self, parent: int | None, uses: int | None, move: str) -> int:
        with self.store._db() as db:
            return db.execute("INSERT INTO nodes (parent, uses, move, status) VALUES (?,?,?, 'running')",
                              (parent, uses, move)).lastrowid

    def close(self, nid: int, **cols) -> None:
        with self.store._db() as db:
            db.execute(f"UPDATE nodes SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?", (*cols.values(), nid))


def done(nodes: list[dict], kind: str) -> list[dict]:
    """Done nodes of a kind; "input" means findings and ideas, the nodes a model step can be given."""
    kinds = ("finding", "idea") if kind == "input" else (kind,)
    return [n for n in nodes if n["status"] == "done" and n["kind"] in kinds]


def beats(child: dict, parent: dict) -> bool:
    """The child's fold R2s beat the parent's by more than one standard error of the paired difference,
    so a change that moves CV by noise does not count as a win."""
    d = np.array(json.loads(child["folds"])) - np.array(json.loads(parent["folds"]))
    return d.mean() > d.std(ddof=1) / np.sqrt(len(d))


def model_reward(node: dict, by_id: dict[int, dict], models: list[dict]) -> float:
    """1 if a model node beat its parent (a draft: the median model's CV), else 0."""
    if node["parent"] is not None:
        return float(not node["same"] and beats(node, by_id[node["parent"]]))
    cvs = sorted(n["cv"] for n in models if n["id"] != node["id"])
    return float(node["cv"] > (cvs[len(cvs) // 2] if cvs else 0.0))


def credit(nodes: list[dict]) -> dict[int, list[float]]:
    """Per node, the rewards it has earned. A model node earns its own reward; a finding earns the rewards
    of the model nodes built on it (none yet means its value is still unknown)."""
    by_id = {n["id"]: n for n in nodes}
    models = done(nodes, "model")
    out: dict[int, list[float]] = {n["id"]: [] for n in nodes}
    for n in models:
        r = model_reward(n, by_id, models)
        out[n["id"]].append(r)
        if n["uses"] in out:
            out[n["uses"]].append(r)
    return out


def pick_move(nodes: list[dict], eps: float, rng: random.Random, allowed: set[str] | None = None) -> str:
    """Thompson sampling with Beta(1 + wins, 1 + losses) per move. Running steps count as losses until they
    finish (virtual loss), so parallel workers spread out; findings nobody has built on yet count as
    nothing. With probability ``eps`` a uniform move, so no move's chance falls to zero."""
    arms = [m for m in MOVES if m != "playbook" and (allowed is None or m in allowed)]   # playbook nodes are ideas
    if rng.random() < eps:
        return rng.choice(arms)
    got = credit(nodes)
    wins = {m: 0.0 for m in MOVES}
    losses = {m: 0.0 for m in MOVES}
    for n in nodes:
        if n["status"] == "running":
            losses[n["move"]] += 1
        elif n["status"] == "failed":
            losses[n["move"]] += 1
        else:
            wins[n["move"]] += sum(got[n["id"]])
            losses[n["move"]] += len(got[n["id"]]) - sum(got[n["id"]])
    return max(arms, key=lambda m: rng.betavariate(1 + wins[m], 1 + losses[m]))


EXHAUSTED = 4   # a model with this many finished children and none better is closed as a parent
BLOCKED = 2     # a move tried this many times on one parent without a win is blocked there
RETIRED = 2     # a finding or idea this many models were built on, none better than its parent, is retired
BACKLOG = 2     # with this many findings and ideas no model has been built on, the next step acts on one


def pruned(nodes: list[dict]) -> tuple[set[int], set[tuple[int, str]], set[int]]:
    """Dead ends: model nodes closed as parents, (parent, move) pairs blocked, findings and ideas retired.
    The search skips them, except on its uniform-chance steps, so nothing is ruled out for good."""
    got = credit(nodes)
    kids: dict[int, list[float]] = {}
    pairs: dict[tuple[int, str], list[float]] = {}
    for n in done(nodes, "model"):
        if n["parent"] is not None:
            kids.setdefault(n["parent"], []).extend(got[n["id"]])
            pairs.setdefault((n["parent"], n["move"]), []).extend(got[n["id"]])
    closed = {p for p, r in kids.items() if len(r) >= EXHAUSTED and not any(r)}
    blocked = {pm for pm, r in pairs.items() if len(r) >= BLOCKED and not any(r)}
    retired = {f["id"] for f in done(nodes, "input") if len(got[f["id"]]) >= RETIRED and not any(got[f["id"]])}
    return closed, blocked, retired


def pick_model(nodes: list[dict], top_k: int, rng: random.Random, move: str = "",
               prune: bool = True) -> dict | None:
    """One of the ``top_k`` open model nodes by CV, weighted down by how many children it has and has running.
    ``prune``: skip nodes closed as parents, and those on which ``move`` is blocked."""
    closed, blocked, _ = pruned(nodes) if prune else (set(), set(), set())
    top = sorted((n for n in done(nodes, "model") if n["id"] not in closed and (n["id"], move) not in blocked),
                 key=lambda n: -n["cv"])[:top_k]
    if not top:
        return None
    kids = {n["id"]: 0.0 for n in top}
    for n in nodes:
        if n["parent"] in kids:
            kids[n["parent"]] += 1.0 if n["status"] == "running" else 0.3
    return rng.choices(top, weights=[1 / (1 + kids[n["id"]]) for n in top])[0]


def backlog(nodes: list[dict]) -> int:
    """Findings and ideas no step has built on or checked yet."""
    used = {n["uses"] for n in nodes if n["uses"] is not None}
    return sum(f["id"] not in used and f["move"] != "playbook" for f in done(nodes, "input"))


def pick_input(nodes: list[dict], rng: random.Random, prune: bool = True, move: str = "") -> dict | None:
    """A finding or idea nobody has built on or checked yet if there is one, else Thompson sampling on the
    rewards of the models built on each. ``prune``: skip retired ones. A check is not given another check."""
    retired = pruned(nodes)[2] if prune else set()
    found = [f for f in done(nodes, "input") if f["id"] not in retired and not (move == "check" and f["move"] == "check")]
    if not found:
        return None
    used = {n["uses"] for n in nodes if n["uses"] is not None}
    fresh = [f for f in found if f["id"] not in used and f["move"] != "playbook"]   # the playbook waits its turn
    if fresh:
        return rng.choice(fresh)
    got = credit(nodes)
    return max(found, key=lambda f: rng.betavariate(1 + sum(got[f["id"]]), 1 + len(got[f["id"]]) - sum(got[f["id"]])))


def short(n: dict, k: int = 500) -> str:
    return " ".join((n["text"] or "").split())[:k]


def ledger(nodes: list[dict], parent: dict | None, rng: random.Random, k: int = 8) -> str:
    """Models: the starting node's ancestry, the best, the latest and a random sample of the rest. Findings:
    the most useful, the latest and those nobody has built on yet."""
    models = done(nodes, "model")
    if not models and not done(nodes, "input"):
        return "Nothing yet: this is one of the first steps."
    by_id = {n["id"]: n for n in models}
    got = credit(nodes)
    pick, p = [], parent
    while p:
        pick.append(p)
        p = by_id.get(p["parent"])
    pick += sorted(models, key=lambda n: -n["cv"])[:k] + models[-k:]
    rest = [n for n in models if n not in pick]
    pick += rng.sample(rest, min(k, len(rest)))
    shown = sorted({n["id"]: n for n in pick}.values(), key=lambda n: n["id"])
    out = [f"## Models ({len(models)} so far, {len(shown)} shown)"]
    for n in shown:
        gain = (" (same forecasts as its parent)" if n["same"] else
                f", {n['cv'] - by_id[n['parent']]['cv']:+.4f} vs parent" if n["parent"] in by_id else "")
        origin = f"from #{n['parent']}" if n["parent"] else "draft"
        via = f" using #{n['uses']}" if n["uses"] else ""
        frame = f" [frame: {n['frame']}]" if n["frame"] and n["frame"] != DEFAULT_FRAME else ""
        ps = f" [pieces: {theory.table(json.loads(n['pieces']))}]" if n.get("pieces") else ""
        out.append(f"- model #{n['id']} ({origin}, move {n['move']}{via}) cv {n['cv']:+.4f}{gain}{frame}{ps}: {short(n)}")
    found = done(nodes, "input")
    used = [f for f in found if got[f["id"]]]
    show = sorted(used, key=lambda f: -sum(got[f["id"]]))[:k] + found[-k:] + [f for f in found if not got[f["id"]]][:k]
    show = sorted({f["id"]: f for f in show}.values(), key=lambda f: f["id"])
    out.append(f"\n## Findings and ideas ({len(found)} so far, {len(show)} shown)")
    for f in show:
        on = f" on model #{f['parent']}" if f["parent"] else ""
        of = f" checking #{f['uses']}" if f["uses"] else ""
        g = got[f["id"]]
        use = f"{len(g)} models built on it, {sum(g):.0f} beat their parent" if g else "no model built on it yet"
        tgt = f"; target {f['frame']}" if f["frame"] and f["frame"] != "y" else ""
        out.append(f"- {f['kind']} #{f['id']} ({f['move']}{on}{of}{tgt}; {use}): {short(f)}")
    closed, blocked, retired = pruned(nodes)
    if closed or blocked or retired:
        out.append("\n## Dead ends so far (tried, no improvement)")
        if closed:
            out.append(f"- models whose children never beat them ({EXHAUSTED}+ tries): "
                       + ", ".join(f"#{i}" for i in sorted(closed)))
        if blocked:
            out.append(f"- moves that failed {BLOCKED}+ times on one model: "
                       + ", ".join(f"{m} on #{p}" for p, m in sorted(blocked)))
        if retired:
            out.append(f"- findings and ideas no model built on them improved ({RETIRED}+ tries): "
                       + ", ".join(f"#{i}" for i in sorted(retired)))
    failed = sum(n["status"] == "failed" for n in nodes)
    out.append(f"\n{failed} steps failed to file anything. Every node's full note is in the shared notes "
               f"(get_note with the node's note id, or search_notes).")
    return "\n".join(out)


SCREEN_HELP = ("A screen tool is at ../lib/screen.py (python ../lib/screen.py --help): it scores every feature "
               "under a package of transforms (raw, cs, lag, cslag, inter, shape, cond, factor, vol, ou) by per-date "
               "rank correlation with a target (y, z, pcaK, ols:x01,x02|lag1, or a node's ../nodes/N_resid.parquet), in "
               "one call. Use it for broad passes and your own scripts for the follow-up.")


PIECES_HELP = """Write the model as a sum of named pieces, so the team can see what each part adds:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import polars as pl
from mechanisms import Pieces, Target, template, z, past

def my_piece(d):
    # d: rows sorted by asset, date, with the features, y and frame (the target being fitted); y and frame are
    # null on the dates being forecast. Return a polars expression, a series or an array with one column or
    # several, built only from earlier rows wherever it uses y or frame. Polars cannot nest a window over
    # date inside one over asset: materialise each step with d = d.with_columns(...) first.
    d = d.with_columns(z("xa").alias("za"))
    return d.select(past(pl.col("za"), 1)).to_series()

MODEL = Pieces({
    "a short name": ("one line: what in the data this piece expresses", my_piece),
    "another": ("...", template("linear", feature="xb")),
}, lookback=120)   # dates of history a forecast needs
fit, predict = MODEL.fit, MODEL.predict
```

A piece is any code that makes columns: the model is a ridge of the target on all pieces' columns, calibrated
to y, so an effect that is not additive (a product, a switch, a nonlinear map, a fitted sub-model) goes inside
one piece. A piece can instead change what is fitted: Target(forward, scale=None), where forward(d, y) is the
new target made from column y (scale(d), if given, maps a forecast of it back). Ready-made pieces are in
../lib/mechanisms.py (python -c "import sys; sys.path.insert(0, '../lib'); import mechanisms; print(mechanisms.help())").
After scoring, the harness measures each piece's necessity: the CV lost when it is dropped and the rest refitted.
A piece that adds nothing is a claim the data does not support."""


def frame_of(node: dict | None) -> str:
    return (node or {}).get("frame") or DEFAULT_FRAME


def chain(nodes: list[dict], parent: dict | None) -> str:
    """The findings and ideas the starting node's line of models was built on, oldest first."""
    by_id, out, p = {n["id"]: n for n in nodes}, [], parent
    while p:
        if p["uses"] in by_id:
            u = by_id[p["uses"]]
            out.append(f"#{u['id']} ({'playbook' if u['move'] == 'playbook' else u['kind']}: {short(u, 80).removeprefix('Idea: ')}) "
                       f"in model #{p['id']}")
        p = by_id.get(p["parent"])
    return "; ".join(reversed(out))


def task(nid: int, move: str, parent: dict | None, finding: dict | None, led: str, built_on: str = "",
         screen: bool = True, pieces: bool = False) -> str:
    parts = [f"You are step #{nid} of the search."]
    target = f"../nodes/{parent['id']}_resid.parquet" if parent else "y"
    if parent:
        parts.append(f"Start from model #{parent['id']}: its model is ../nodes/{parent['id']}.py (cv "
                     f"{parent['cv']:+.4f}), its out-of-sample residuals (date, asset, pred, resid) over the CV "
                     f"dates are in ../nodes/{parent['id']}_resid.parquet, and its note is note #{parent['note']}.")
        if parent.get("pieces"):
            parts.append("Its pieces, each with its necessity (CV lost without it): "
                         + theory.table(json.loads(parent["pieces"])))
        if move in MODELS or move in FREE:
            parts.append(f"Its frame (what it fits; its feature space) is: {frame_of(parent)}. Work in that frame, "
                         "unless your step is to change it.")
        if built_on:
            parts.append(f"Its line of models already applies: {built_on}. Build on that: ideas compose, and "
                         "a new one applies to what the line models now, not to raw y.")
    if finding:
        src = "from the playbook of standard ideas" if finding["move"] == "playbook" else f"note #{finding['note']}"
        parts.append(f"The given note is {finding['kind']} #{finding['id']} ({src}):\n{finding['text']}\n\nJudge "
                     "whether and how it applies to the starting node; adapt it to what the data shows.")
    how = MOVES[move][0].replace('TARGET', target)
    parts.append(f"Your step: {how if screen else re.sub(r' ?[(]python [.][.]/lib/screen[.]py[^)]*[)]', '', how)}")
    if screen:
        parts.append(SCREEN_HELP)
    if pieces and (move in MODELS or move in FREE):
        parts.append(PIECES_HELP)
    parts.append(FILE_FINDING if move in ANALYSIS else FILE_IDEA if move in IDEAS else
                 FILE_EITHER if move in FREE else FILE_FORECAST)
    parts.append(f"# Ledger of the search so far\n\n{led}")
    return "\n\n".join(parts)


def filed(store: Store, run: str, kind: str) -> dict | None:
    rows = [n for n in store.search(kind=kind, k=100_000) if n["run"] == run]
    return store.get(rows[0]["id"]) if rows else None   # newest first: the last one filed


async def step(tree: Tree, nid: int, move: str, parent: dict | None, finding: dict | None, led: str, model: str,
               max_turns: int, built_on: str = "", screen: bool = True, pieces: bool = False) -> None:
    """One searcher run. A forecast makes this a model node, which the harness scores itself so every model has
    residuals for later steps; otherwise the finding or idea it filed is the node."""
    run = await run_agent(tree.root, "searcher", task(nid, move, parent, finding, led, built_on, screen, pieces), model=model,
                          max_turns=min(max_turns, IDEA_TURNS) if move in IDEAS else max_turns,
                          max_budget_usd=1.0, timeout_s=1200)
    fc = filed(tree.store, run, "forecast")
    if fc is None:
        for kind in ("finding", "idea"):
            if fd := filed(tree.store, run, kind):
                tree.close(nid, run=run, kind=kind, note=fd["id"], status="done", text=fd["text"],
                           frame=f"residuals of #{parent['id']}" if parent and kind == "finding" else "y")
                return
        tree.close(nid, run=run, status="failed", error="nothing filed")
        return
    dest, oof = tree.dir / f"{nid}.py", tree.dir / f"{nid}_resid.parquet"
    shutil.copy2(tree.root / "work" / fc["artifact"], dest)
    train = pl.read_parquet(tree.root / "work" / "data" / "train.parquet")
    table = None
    try:
        folds, rank = await asyncio.to_thread(score.cv_full, dest, train, oof=oof)
        if pieces:
            table = await asyncio.to_thread(theory.pieces, dest, train, folds)
            if table is None:
                raise score.ScoreError("the model is not written as Pieces")
    except score.ScoreError as e:
        tree.close(nid, run=run, kind="model", note=fc["id"], status="failed", error=str(e)[:500])
        return
    same = 0
    if parent:
        a, b = pl.read_parquet(oof)["pred"].to_numpy(), pl.read_parquet(tree.dir / f"{parent['id']}_resid.parquet")
        same = int(np.allclose(a, b["pred"].to_numpy(), rtol=1e-6, atol=1e-9))
    line = next((ln for ln in (fc["text"] or "").splitlines() if ln.strip().lower().startswith("frame:")), "")
    frame = line.split(":", 1)[1].strip()[:200] if line else frame_of(parent)
    tree.close(nid, run=run, kind="model", note=fc["id"], status="done", cv=sum(folds) / len(folds),
               folds=json.dumps(folds), ic=rank["ic"], holdout=fc["holdout"], same=same, text=fc["text"], frame=frame,
               pieces=json.dumps(table) if table else None)


async def search(root: Path, steps: int, parallel: int, model: str, max_turns: int, drafts: int, top_k: int,
                 eps: float, seed: int = 0, show_ledger: bool = True, survey: bool = True,
                 playbook: bool = True, moves: set[str] | None = None, screen: bool = True,
                 pieces: bool = False) -> None:
    """Run until the tree has ``steps`` nodes besides the playbook (failed ones count), at most ``parallel`` steps at once. The
    first ``drafts`` steps are drafts; moves and starting nodes are chosen when a step starts. ``show_ledger``
    False is the control: each step sees only its starting node and finding, not the rest of the tree.
    ``survey``: the first step after the drafts is a survey of every feature (the move stays in the menu
    either way; off reproduces runs made before it existed only approximately). ``playbook``: file the
    standard ideas of PLAYBOOK as idea nodes before the first step. ``moves``: the only moves the bandit may
    pick after the drafts (an ablation of the menu); None is all of them. ``screen`` False is the control for
    the screen tool, whose packages mirror known families: no tool, no help text, no survey. ``pieces``: every
    model is written as a sum of named pieces (mechanisms.Pieces), and each piece's necessity is measured."""
    tree, rng = Tree(root), random.Random(seed)
    tree.reset()
    if not screen:
        (root / "work" / "lib" / "screen.py").unlink(missing_ok=True)   # the copy Tree() just made
        survey = False
    if playbook:
        tree.seed(PLAYBOOK)
    running: set[asyncio.Task] = set()

    def launch() -> None:
        nodes = tree.nodes()
        move = ("draft" if sum(n["move"] == "draft" for n in nodes) < drafts else
                "survey" if survey and not any(n["move"] == "survey" for n in nodes) else
                "act" if backlog(nodes) >= BACKLOG and (moves is None or "act" in moves) else
                pick_move(nodes, eps, rng, moves))   # findings get tested
        need = MOVES[move][1]
        prune = rng.random() >= eps   # on uniform-chance steps dead ends are open again
        parent = pick_model(nodes, top_k, rng, move, prune) if {"model", "model?"} & set(need) else None
        finding = pick_input(nodes, rng, prune, move) if "input" in need else None
        if ("model" in need and parent is None) or ("input" in need and finding is None):
            move, parent, finding = (("explore", None, None) if move in ANALYSIS else
                                     ("orthogonal", None, None) if move in IDEAS else
                                     ("draft", None, None) if parent is None else ("feature", parent, None))
        nid = tree.open(parent and parent["id"], finding and finding["id"], move)
        print(f"start #{nid} {move}" + (f" from #{parent['id']}" if parent else "")
              + (f" with #{finding['id']}" if finding else ""), flush=True)
        led = ledger(nodes if show_ledger else [], parent, rng)
        running.add(asyncio.create_task(step(tree, nid, move, parent, finding, led, model, max_turns,
                                             chain(nodes, parent), screen, pieces)))

    while True:
        while len(running) < parallel and sum(n["move"] != "playbook" for n in tree.nodes()) < steps:
            launch()
        if not running:
            break
        finished, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for t in finished:
            running.discard(t)
            if t.exception():
                print(f"step crashed: {t.exception()!r}", flush=True)
        report(root)


def mechanisms(root: Path, nodes: list[dict]) -> dict[int, list[str]]:
    """Per done node, the planted mechanisms it found: for a model, those its model uses; for a finding or idea,
    those its text names (needs the world's sealed truth)."""
    from arh import evals
    world = Path(json.loads((root / "config.json").read_text())["world"])
    truth = json.loads((world / "sealed" / "truth.json").read_text())
    store, out = Store(root / "store.db"), {}
    for n in nodes:
        if n["move"] == "playbook":
            out[n["id"]] = []   # standard ideas, not the search's own
            continue
        note = store.get(n["note"])
        if n["kind"] == "model":
            code = (root / "work" / "nodes" / f"{n['id']}.py").read_text()
            out[n["id"]] = evals.accuracy(note, truth, code).get("mechanisms_in_model", [])
        else:
            out[n["id"]] = evals.accuracy(note, truth)["mechanisms_hit"]
    return out


def report(root: Path, full: bool = False) -> None:
    tree = Tree(root)
    nodes = tree.nodes()
    models = sorted(done(nodes, "model"), key=lambda n: -n["cv"])
    found = done(nodes, "input")
    cost = sum(r["cost"] or 0 for r in tree.store.runs())
    print(f"nodes {len(nodes)}: {len(models)} models, {len(found)} findings and ideas, "
          f"{sum(n['status'] == 'failed' for n in nodes)} failed, {sum(n['status'] == 'running' for n in nodes)} "
          f"running; cost ${cost:.2f}", flush=True)
    mech = mechanisms(root, models + found) if full else {}
    for n in models[:10 if not full else None]:
        ho = "  -   " if n["holdout"] is None else f"{n['holdout']:+.4f}"
        print(f"  #{n['id']:<4} {n['move']:12} parent {str(n['parent']):5} finding {str(n['uses']):5} "
              f"cv {n['cv']:+.4f} holdout {ho}{' same' if n['same'] else ''}"
              + (f"  {','.join(mech[n['id']])}" if full else ""), flush=True)
        if full and n.get("pieces"):
            for r in json.loads(n["pieces"]):
                print(f"      {r['necessity']:+.4f} ± {r['se']:.4f}  {r['name']}: {r['claim']}", flush=True)
    if not full:
        return
    got = credit(nodes)
    for f in found:
        print(f"  {f['kind']:7} #{f['id']:<4} {f['move']:10} built on {len(got[f['id']])}, won {sum(got[f['id']]):.0f}  "
              f"{','.join(mech[f['id']])}  {short(f, 120)}")
    stats = {m: [0.0, 0] for m in MOVES}
    for n in nodes:
        stats[n["move"]][1] += 1
        if n["status"] == "done":
            stats[n["move"]][0] += sum(got[n["id"]])
    print("moves (wins/pulls):", {m: f"{w:.0f}/{p}" for m, (w, p) in stats.items()})
    top = [n["holdout"] for n in models[:5] if n["holdout"] is not None]
    print(f"best-by-cv holdout {models[0]['holdout'] if models else None}; top-5 mean holdout "
          f"{sum(top) / len(top) if top else None}")
    print(f"mechanisms in any model: {sorted(set().union(*(mech[n['id']] for n in models)))}; "
          f"named by any finding or idea: {sorted(set().union(*(mech[f['id']] for f in found)))}")
