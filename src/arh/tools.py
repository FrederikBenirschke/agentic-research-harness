"""Agent tools: search and read notes, add a note of a kind the agent may
write, and (forecaster only) score a model. They are served in-process over
MCP, so the store and the scorer run in the harness process, outside the
agent's sandbox."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import polars as pl
from claude_agent_sdk import create_sdk_mcp_server, tool

from arh import score
from arh.store import KINDS, Store, render

SERVER = "store"


def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def build(store: Store, *, run: str, agent: str, writes: list[str], cwd: Path, work: Path, world: Path,
          can_score: bool, has_files: bool) -> tuple[object, list[str]]:
    train = work / "data" / "train.parquet"

    def resolve(path: str) -> Path:
        p = (cwd / path).resolve()
        if not p.is_file() or work.resolve() not in p.parents:
            raise ValueError(f"{path}: not a file under the work dir")
        return p

    @tool("search_notes", "Search the shared notes. Empty query lists the newest. open_only: questions no "
          "finding answers yet, ideas no forecast has used yet.",
          {"type": "object", "properties": {
              "query": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)},
              "open_only": {"type": "boolean"}, "k": {"type": "integer"}}})
    async def search_notes(args):
        rows = store.search(args.get("query", ""), args.get("kind"), bool(args.get("open_only")),
                            min(int(args.get("k", 8)), 30))
        return _ok("\n".join(render(r) for r in rows) or "no notes match")

    @tool("get_note", "Read one note in full, with the notes that cite it.",
          {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]})
    async def get_note(args):
        note = store.get(int(args["id"]))
        return _ok(render(note, full=True)) if note else _err("no such note")

    props = {"kind": {"type": "string", "enum": writes}, "text": {"type": "string"},
             "refs": {"type": "array", "items": {"type": "integer"}}}
    if has_files:
        props["artifact"] = {"type": "string"}
        props["code"] = {"type": "string"}

    @tool("add_note", f"Add a note to the shared store. You may write kinds: {', '.join(writes)}. refs: ids of "
          "notes this builds on or answers." + (" artifact: a file you wrote (script or model.py), path relative "
          "to your working dir; it is frozen into the store. Or pass the file's source as code instead (saved as "
          "model.py for a forecast, script.py otherwise). A forecast's artifact is scored by the harness."
          if has_files else ""),
          {"type": "object", "properties": props, "required": ["kind", "text"]})
    async def add_note(args):
        kind = args["kind"]
        if kind not in writes:
            return _err(f"{agent} may write only {writes}")
        try:
            artifact, cv = None, None
            if args.get("code"):
                args["artifact"] = Path(args.get("artifact") or ("model.py" if kind == "forecast" else "script.py")).name
                (cwd / args["artifact"]).write_text(args["code"])
            if args.get("artifact"):
                src = resolve(args["artifact"])
                dest = work / "notes" / f"{run}-{store.counts().get(kind, 0)}-{src.name}"
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(src, dest)
                artifact = str(dest.relative_to(work))
            if kind == "forecast":
                if artifact is None:
                    return _err("a forecast needs its model file as artifact")
                folds = await asyncio.to_thread(score.cv, work / artifact, pl.read_parquet(train))
                cv = sum(folds) / len(folds)
            note_id = store.add(kind, agent, args["text"], args.get("refs", []), artifact, run, cv)
        except (ValueError, score.ScoreError) as e:
            return _err(str(e))
        if kind == "forecast":
            try:                                    # stored, not returned to the agent
                h = await asyncio.to_thread(score.holdout_full, work / artifact, world)
                store.set_holdout(note_id, h["r2"], h["ic"], h["ls_sharpe"])
            except score.ScoreError:
                pass
            return _ok(f"added #{note_id}, cv R2 {cv:.4f}")
        return _ok(f"added #{note_id}")

    @tool("score_model", "Walk-forward CV of a model file (expanding window, 5 folds over the last half of "
          "the training dates). Returns R2 per fold. Every call is counted as a trial.",
          {"type": "object", "properties": {
              "path": {"type": "string"},
              "relation": {"type": "string", "enum": ["follow-up", "new"],
                           "description": "follow-up: refines an earlier trial or note; new: a different idea"},
              "builds_on": {"type": "string", "description": "the trial or note it follows up, or what it differs from"},
              "why": {"type": "string", "description": "a paragraph of prose: what you saw in the data or in earlier "
                      "results, the hypothesis this model tests, why this model tests it, and what CV would confirm or "
                      "refute it"},
              "learned": {"type": "string", "description": "a paragraph of prose: what the previous trial's result "
                          "told you about the data, and how that led to this one"}},
           "required": ["path"]})
    async def score_model(args):
        why = {k: args[k] for k in ("relation", "builds_on", "why", "learned") if args.get(k)}
        why = json.dumps(why) if why else None
        try:
            path = resolve(args["path"])
            sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            folds, rank = await asyncio.to_thread(score.cv_full, path, pl.read_parquet(train))
        except (ValueError, score.ScoreError) as e:
            store.add_trial(run, "", None, str(e)[:500], why)
            return _err(str(e))
        tid, n = store.add_trial(run, sha, sum(folds) / len(folds), None, why)
        return _ok(f"trial #{tid}: cv R2 mean {sum(folds) / len(folds):.4f}, folds {[round(f, 4) for f in folds]}; "
                   f"rank IC {rank['ic']:.4f}, top-minus-bottom quintile Sharpe per date {rank['ls_sharpe']:.3f}; "
                   f"{n} distinct models scored in this research run so far")

    @tool("request", "Record a library, tool or data you wanted for an idea but do not have (there is no network "
          "and no pip install). The maintainer reviews these. Then carry on with your fallback.",
          {"type": "object", "properties": {
              "want": {"type": "string", "description": "what you wanted, e.g. a library and the method in it"},
              "why": {"type": "string", "description": "the idea it was for and why it should help on this data"},
              "instead": {"type": "string", "description": "what you will do instead with what is available"}},
           "required": ["want", "why", "instead"]})
    async def request(args):
        rid = store.add_request(run, args["want"], args["why"], args["instead"])
        return _ok(f"request #{rid} recorded. Go ahead with your fallback: {args['instead']}")

    tools = [search_notes, get_note, add_note] + ([request] if has_files else []) + ([score_model] if can_score else [])
    return (create_sdk_mcp_server(name=SERVER, version="1.0", tools=tools),
            [f"mcp__{SERVER}__{t.name}" for t in tools])
