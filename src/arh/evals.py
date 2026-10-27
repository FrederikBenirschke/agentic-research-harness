"""Grade agent runs. Each run gets three groups of metrics:

process    whether it finished and wrote exactly one note; tool use (calls, errors,
           repeated identical calls, whether it read notes before writing); effort
           (turns, trials, tokens, wall time)
grounding  whether the notes it cites were returned to it by a tool; whether the
           numbers in its note appear in tool output; whether a forecast reports
           the CV the harness measured; whether an analyst's script reruns
accuracy   using the world's sealed truth.json: which planted mechanisms the note
           names, whether it names the decoy, and for a forecast the holdout R2
           divided by the oracle's

An optional LLM judge (a claude-* or Ollama model) scores specificity, evidence,
actionability, clarity and, given the truth, correctness.

An eval condition (evals/<world>.json) is one agent plus a fixed set of starting
notes, so one agent can be tested without running the others.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from arh import score
from arh.runner import OLLAMA, load_agent, run_agent
from arh.store import Store, render

NUM = re.compile(r"(?<![\w.])[-+]?\d*\.\d+(?:e[-+]?\d+)?%?|(?<![\w.])[-+]?\d+%", re.I)
FEAT = re.compile(r"\bx\d\d\b")


# trace parsing

def events(trace: Path) -> tuple[list[dict], dict | None]:
    """Tool calls in order, each with its result text and error flag; the final result message."""
    calls, by_id, final = [], {}, None
    for line in trace.read_text().splitlines():
        msg = json.loads(line)
        if not isinstance(msg, dict):
            continue
        if "num_turns" in msg:
            final = msg
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if "name" in block and "input" in block and "id" in block:
                by_id[block["id"]] = c = {"name": block["name"].removeprefix("mcp__store__"),
                                          "input": block["input"], "result": "", "error": False}
                calls.append(c)
            elif "tool_use_id" in block and block["tool_use_id"] in by_id:
                c, content = by_id[block["tool_use_id"]], block.get("content")
                c["result"] = content if isinstance(content, str) else json.dumps(content)
                c["error"] = bool(block.get("is_error"))
    return calls, final


def numbers(text: str) -> list[str]:
    return [n for n in NUM.findall(text) if n.strip("+-%") not in ("", "0", "1")]


def grounded(claimed: list[str], source: str) -> float | None:
    """Share of claimed numbers that appear in ``source`` at the precision written."""
    if not claimed:
        return None
    pool = [n.rstrip("%") for n in NUM.findall(source)]
    def hit(n: str) -> bool:
        v, pct = n.rstrip("%"), n.endswith("%")
        d = len(v.split(".")[1]) if "." in v and "e" not in v.lower() else 0
        try:
            target = float(v) / (100 if pct else 1)
            dd = d + (2 if pct else 0)
            return any(abs(round(float(p) * k, dd) - round(target, dd)) < 10 ** -dd / 2 + 1e-12
                       for p in pool for k in (1, 0.01, 100) if p not in ("", "+", "-"))
        except ValueError:
            return False
    return sum(map(hit, claimed)) / len(claimed)


# metrics

def run_metrics(root: Path, run: str) -> dict:
    store = Store(root / "store.db")
    meta = next(r for r in store.runs() if r["id"] == run)
    calls, final = events(root / "traces" / f"{run}.jsonl")
    agent_meta, _ = load_agent(meta["agent"])
    notes = [store.get(n["id"]) for n in store.search(k=10_000) if n["run"] == run]
    writes = [c for c in calls if c["name"] == "add_note"]
    first_write = calls.index(writes[0]) if writes else len(calls)
    seen = " ".join([meta.get("task") or ""] + [c["result"] for c in calls[:first_write]])   # quick mode puts notes in the task
    keys = [json.dumps([c["name"], c["input"]], sort_keys=True) for c in calls]
    m = {"run": run, "agent": meta["agent"], "model": meta["model"],
         "finished": final is not None and not final.get("is_error") and not meta["error"],
         "error": meta["error"], "n_notes": len(notes),
         # an agent that may write forecasts and other kinds (researcher) may write several notes but ends on a forecast
         "protocol_ok": all(n["kind"] in agent_meta["writes"] for n in notes) and (
             len(notes) == 1 if len(agent_meta["writes"]) == 1 or "forecast" not in agent_meta["writes"]
             else bool(notes) and notes[0]["kind"] == "forecast"),
         "turns": meta["turns"], "wall_s": (final or {}).get("duration_ms", 0) / 1000 or None,
         "out_tokens": ((final or {}).get("usage") or {}).get("output_tokens"), "cost": meta["cost"],
         "tool_calls": len(calls), "tools": {k: sum(c["name"] == k for c in calls) for k in {c["name"] for c in calls}},
         "tool_error_rate": sum(c["error"] for c in calls) / len(calls) if calls else None,
         "redundant_rate": (len(keys) - len(set(keys))) / len(keys) if keys else None,
         "read_before_write": any(c["name"] in ("search_notes", "get_note") for c in calls[:first_write]),
         "trials": sum(c["name"] == "score_model" for c in calls)}
    code = "\n".join([c["input"].get("command", "") for c in calls if c["name"] == "Bash"]
                     + [p.read_text(errors="replace") for p in (root / "work" / run).glob("*.py")])
    if any(c["name"] == "Bash" for c in calls):
        m.update(exploration(code))
    if not notes:
        return m
    # notes are newest first; grade the latest forecast if there is one, else the latest note
    note = next((n for n in notes if n["kind"] == "forecast"), notes[0])
    m["note_id"], m["kind"], m["text_len"] = note["id"], note["kind"], len(note["text"])
    seen_ids = {int(i) for i in re.findall(r"#(\d+)", seen)}
    m["refs_grounded"] = (sum(r in seen_ids for r in note["refs"]) / len(note["refs"])) if note["refs"] else None
    sources = seen + " ".join(json.dumps(c["input"]) for c in calls[:first_write])
    m["numbers_grounded"] = grounded(numbers(note["text"]), sources)
    if note["kind"] == "forecast":
        m["cv"], m["holdout"] = note["cv"], note["holdout"]
        m["holdout_ic"], m["holdout_ls"] = note.get("holdout_ic"), note.get("holdout_ls")
        m["cv_reported_ok"] = note["cv"] is not None and any(
            abs(float(n.rstrip("%")) - note["cv"]) < 0.002 for n in numbers(note["text"]) if not n.endswith("%"))
    if note["kind"] == "finding" and note["artifact"]:
        m.update(rerun(root, note))
    return m


# Checks detected by regex in the agent's code (Bash commands and .py files), regardless of what its note says.
CHECKS = {
    "time_split": r"half|first_half|second_half|date\"?\)? *[<>]=? *\d{2,}|by_year|period|rolling",
    "regime_split": r"median\(\)|quantile|> *0\b.*date|regime",
    "lags": r"shift\(",
    "multi_lag": r"shift\((-?[2-9]|lag)|for +lag|range\(1, *\d",
    "nonlinear": r"\*\* *2|\^2|square|abs\(|pow\(|qcut|cut\(|bins|decile|quintile|spline|poly|tree|forest|boost",
    "interaction": r"x\d\d\"?\)? *\* *\(?pl\.col\(\"?x\d\d|interaction|x\d\d *\* *x\d\d",
    "cross_asset": r"pivot|corrcoef\(.*\.t|lead|cross.?asset|other asset|asset *!=",
    "multiple_testing": r"bonferroni|fdr|benjamini|permut|shuffle|placebo|multiple",
}


def exploration(code: str) -> dict:
    low = code.lower()
    checks = {k: bool(re.search(p, low)) for k, p in CHECKS.items()}
    return {"checks": [k for k, v in checks.items() if v], "n_checks": sum(checks.values()),
            "n_features_examined": len(set(FEAT.findall(low)))}


def rerun(root: Path, note: dict) -> dict:
    """Rerun a finding's script in the model sandbox from its author's working dir."""
    script = (root / "work" / note["artifact"]).resolve()
    cwd = root / "work" / note["run"]
    if script.suffix != ".py":
        return {"rerun_ok": None}
    allow = [root / "work", *score.runtime_paths()]
    try:
        p = subprocess.run(["sandbox-exec", "-p", score.profile(allow), sys.executable, "-I", str(script)],
                           cwd=cwd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"rerun_ok": False, "rerun_numbers": None}
    return {"rerun_ok": p.returncode == 0, "rerun_numbers": grounded(numbers(note["text"]), p.stdout)}


def code_tokens(code: str) -> str:
    """The names and string literals a model's code uses, without comments or docstrings
    ("# x05 dropped" must not count as using x05)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return re.sub(r"#.*", "", code)
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return " ".join(n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else n.value
                    for n in ast.walk(tree)
                    if isinstance(n, (ast.Name, ast.Attribute))
                    or (isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs))


def accuracy(note: dict, truth: dict, code: str = "") -> dict:
    """Regex grading against the truth. A mechanism counts as found if all its features are named (for y,
    a lag word such as "lag" or "momentum") and, if the truth gives a ``pattern``, the pattern matches
    (naming x04 alone does not count as the U-shape). ``code`` is a forecast's model source; only names and
    strings used in the code count, not comments or docstrings."""
    code = code_tokens(code) if code else ""
    text = (note["text"] + " " + code).lower()
    feats = set(FEAT.findall(text))
    lagged = bool(re.search(r"lag|autocorr|momentum|previous y|prior y|shift", text))
    hits = [m["id"] for m in truth["mechanisms"]
            if all((f in feats) if f != "y" else lagged for f in m["features"])
            and (not m.get("pattern") or re.search(m["pattern"], text))]
    out = {"mechanisms_hit": hits, "n_mechanisms_hit": len(hits),
           "decoy_mentioned": any(d["feature"] in feats for d in truth["decoys"]),
           "noise_named": len(feats & set(truth["noise_features"]))}
    if code:
        out["decoy_in_model"] = any(d["feature"] in code for d in truth["decoys"])
        # named is not used: a note can say "sector" while the model never reads the sector code
        used = set(FEAT.findall(code.lower()))
        out["mechanisms_in_model"] = [m["id"] for m in truth["mechanisms"] if m["id"] in hits
                                      and all(f in used for f in m["features"] if f != "y")]
    if note["kind"] == "forecast" and note.get("holdout") is not None:
        oracle = truth["oracle_holdout_r2"]   # null worlds: the ratio is meaningless, report raw holdout only
        out["holdout_vs_oracle"] = note["holdout"] / oracle if oracle > 1e-3 else None
    return out


JUDGE = """You grade one note written by a research agent ({agent}) on a team forecasting y in a panel dataset.
Score each 1-5: specificity (concrete features, numbers, conditions), evidence (claims backed by stated
measurements), actionability (a teammate could act on it directly), clarity.
{truth_part}
Reply with JSON only: {{"specificity": n, "evidence": n, "actionability": n, "clarity": n, "correct": "{correct_options}", "why": "one sentence"}}

NOTE:
{note}"""


def judge(note: dict, agent: str, model: str, truth: dict | None) -> dict:
    truth_part = ("The TRUE data-generating process is:\n" + "\n".join(
        f"- {m['text']}" for m in truth["mechanisms"]) + "\n" + "\n".join(f"- {d['text']}" for d in truth["decoys"])
        + "\nAll other features are noise. correct: is what the note claims consistent with the truth?"
        + " Coefficients above are not measured statistics: a correlation, IC or autocorrelation the note"
        + " measured may differ from a coefficient without being wrong."
        if truth else "correct: n/a (no truth available).")
    prompt = JUDGE.format(agent=agent, truth_part=truth_part, note=note["text"],
                          correct_options="correct|partial|wrong|n/a")
    try:
        out = json.loads(_claude(prompt, model) if model.startswith("claude-") else _ollama(prompt, model))
        return {f"judge_{k}": out.get(k) for k in ("specificity", "evidence", "actionability", "clarity",
                                                     "correct", "why")}
    except Exception as e:                       # record a judge failure as a missing grade
        return {"judge_error": f"{type(e).__name__}: {e}"[:200]}


def _ollama(prompt: str, model: str) -> str:
    body = json.dumps({"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


def _claude(prompt: str, model: str) -> str:
    """One tool-less turn through the agent SDK; returns the JSON object in the reply."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    async def ask() -> str:
        opts = ClaudeAgentOptions(model=model, tools=[], max_turns=1, setting_sources=[],
                                  system_prompt="You grade research notes. Reply with one JSON object only.",
                                  extra_args={"strict-mcp-config": None})
        text = ""
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, AssistantMessage):
                text += "".join(b.text for b in msg.content if isinstance(b, TextBlock))
        return text

    text = asyncio.run(ask())
    return text[text.find("{"):text.rfind("}") + 1]


def grade(root: Path, run: str, judge_model: str | None = None) -> dict:
    m = run_metrics(root, run)
    world = Path(json.loads((root / "config.json").read_text())["world"])
    truth_path = world / "sealed" / "truth.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else None
    if "note_id" in m:
        note = Store(root / "store.db").get(m["note_id"])
        code = (root / "work" / note["artifact"]).read_text() if note["kind"] == "forecast" and note["artifact"] else ""
        if truth:
            m.update(accuracy(note, truth, code))
        if judge_model:
            m.update(judge(note, m["agent"], judge_model, truth))
    return m


def regrade(files: list[Path], judge_model: str | None, out_dir: Path) -> None:
    """Grade the recorded runs again with the current grader: same runs, fresh metrics and judge."""
    keep = ("condition", "mode", "world", "rep", "root", "prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        with open(out_dir / f.name, "w") as out:
            for line in f.read_text().splitlines():
                old = json.loads(line)
                m = grade(Path(old["root"]), old["run"], judge_model)
                m.update({k: old[k] for k in keep if k in old}, judge_model=judge_model)
                m.setdefault("prompts", f.stem.removeprefix("results-"))
                out.write(json.dumps(m) + "\n")
                print(f"{f.name} {old['run']}: {summary_line(m)}", flush=True)


# running conditions

QUICK_TURNS = {"analyst": 2}     # fewest turns that can produce the note: analyst must run a script first

async def run_conditions(spec_path: Path, conditions: list[str], model: str, n: int, judge_model: str | None,
                         parallel: int, runs_dir: Path, max_turns: int = 15, timeout_s: float = 600,
                         prompts: Path | None = None, out: Path | None = None,
                         max_budget_usd: float = 2.0) -> list[dict]:
    """Run n reps of each condition, at most ``parallel`` at a time, and append each graded run to ``out``.
    All (condition, rep) pairs share one pool because a local Ollama server has about 2.5x the throughput
    when serving several requests at once (requires OLLAMA_NUM_PARALLEL >= parallel)."""
    from arh.cli import init
    spec = json.loads(spec_path.read_text())
    world = (spec_path.parent.parent / spec["world"]).resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    mode = f"t{max_turns}" if max_turns else "quick"
    gate = asyncio.Semaphore(parallel)
    out = out or runs_dir / "evals" / "results.jsonl"
    label = prompts.name if prompts else "agents"
    if os.environ.get("ARH_SUMMARY"):
        label += "-sum" + os.environ["ARH_SUMMARY"]

    async def rep(condition: str, i: int) -> dict:
        # "<agent>-open": the agent with no starting notes, for specs that do not list it
        cond = spec["conditions"].get(condition) or {"agent": condition.removesuffix("-open"), "notes": []}
        async with gate:
            name = f"evals/{Path(spec['world']).name}-{condition}-{model.replace(':', '_')}-{label}-{mode}-{stamp}-r{i}"
            root = init(name, world)
            store = Store(root / "store.db")
            for note in cond["notes"]:
                store.add(note["kind"], note["author"], note["text"], note.get("refs", []))
            turns, task = max_turns or QUICK_TURNS.get(cond["agent"], 1), ""
            if not max_turns:                   # quick mode: put all notes in the task, ask for the note directly
                notes = "\n\n".join(render(n, full=True) for n in reversed(store.search(k=30)))
                task = (f"All notes so far:\n\n{notes}\n\nYou have {turns} turn(s), so do not search. "
                        + ("Run one script, then call add_note." if turns > 1 else "Call add_note now."))
            run = await run_agent(root, cond["agent"], task, model=model, max_turns=turns, timeout_s=timeout_s,
                                  max_budget_usd=max_budget_usd, prompts=prompts)
            m = await asyncio.to_thread(grade, root, run, judge_model)
            m.update(condition=condition, mode=mode, world=spec["world"], rep=i, root=str(root), judge_model=judge_model,
                     prompts=label)
            with open(out, "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"{condition} r{i}: {summary_line(m)}", flush=True)
            return m

    return await asyncio.gather(*(rep(c, i) for i in range(n) for c in conditions))


def summary_line(m: dict) -> str:
    keys = ["finished", "protocol_ok", "turns", "tool_calls", "tool_error_rate", "numbers_grounded",
            "refs_grounded", "n_mechanisms_hit", "decoy_in_model", "cv", "holdout", "holdout_vs_oracle", "judge_correct"]
    return "  ".join(f"{k}={_fmt(m.get(k))}" for k in keys if k in m)


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


REPORT = ["n", "finished", "protocol_ok", "turns", "wall_s", "tool_calls", "tool_error_rate", "redundant_rate",
          "read_before_write", "trials", "refs_grounded", "numbers_grounded", "cv_reported_ok", "rerun_ok",
          "rerun_numbers", "n_mechanisms_hit", "decoy_mentioned", "decoy_in_model", "noise_named", "cv", "holdout", "holdout_vs_oracle",
          "judge_specificity", "judge_evidence", "judge_actionability", "judge_clarity", "judge_correct_rate"]


def report(results: Path) -> str:
    """One row per (condition, model, mode): mean (sd) of every metric over its reps."""
    rows: dict[tuple, list[dict]] = {}
    for line in results.read_text().splitlines():
        m = json.loads(line)
        rows.setdefault((m["condition"], m["model"], m.get("mode", "?")), []).append(m)
    out = []
    for (cond, model, mode), ms in sorted(rows.items()):
        out.append(f"## {cond}  [{model}, {mode}]")
        for k in REPORT:
            if k == "n":
                vals = [len(ms)]
            elif k == "judge_correct_rate":
                vals = [m["judge_correct"] == "correct" for m in ms if m.get("judge_correct") not in (None, "n/a")]
            else:
                vals = [m[k] for m in ms if m.get(k) is not None]
            if not vals:
                continue
            nums = [float(v) for v in vals]
            mean = statistics.fmean(nums)
            sd = f" (sd {statistics.stdev(nums):.3f})" if len(nums) > 1 else ""
            out.append(f"  {k:20} {mean:.3f}{sd}  [n={len(nums)}]")
        hits = [h for m in ms for h in m.get("mechanisms_hit", [])]
        if hits:
            out.append(f"  {'mechanisms':20} " + ", ".join(f"{h}:{hits.count(h)}/{len(ms)}" for h in sorted(set(hits))))
    return "\n".join(out)


# Scorecard columns per agent: the main metric first, then related ones.
HEADLINE = {
    "analyst": ["protocol_ok", "x07_found", "n_mechanisms_hit", "numbers_grounded", "rerun_numbers", "judge_correct_rate"],
    "ideator": ["protocol_ok", "n_mechanisms_hit", "numbers_grounded", "judge_actionability", "judge_correct_rate"],
    "forecaster": ["protocol_ok", "holdout_vs_oracle", "x07_found", "decoy_in_model", "cv_reported_ok", "trials",
                   "judge_correct_rate"],
    "researcher": ["protocol_ok", "holdout_vs_oracle", "x07_found", "decoy_in_model", "cv_reported_ok", "trials",
                   "n_notes"],
}
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def scorecard(files: list[Path]) -> str:
    """Markdown table per agent: one row per (model, mode, prompt, world), mean of the headline metrics.
    The prompt label is the results file's name (results-<label>.jsonl)."""
    rows: dict[tuple, list[dict]] = {}
    for f in files:
        label = f.stem.removeprefix("results-")
        for line in f.read_text().splitlines():
            m = json.loads(line)
            m["x07_found"] = any("x07" in h for h in m.get("mechanisms_hit", []))
            m["judge_correct_rate"] = SCORE.get(m.get("judge_correct"))
            key = (m["agent"], m["model"], m.get("mode", "?"), label, Path(m.get("world", "?")).name)
            rows.setdefault(key, []).append(m)
    out = []
    for agent, cols in HEADLINE.items():
        out += [f"\n### {agent}\n", "| model | mode | prompt | world | n | " + " | ".join(cols) + " |",
                "|" + "---|" * (5 + len(cols))]
        for key, ms in sorted(rows.items()):
            if key[0] != agent:
                continue
            cells = []
            for k in cols:
                vals = [float(m[k]) for m in ms if m.get(k) is not None]
                cells.append(f"{statistics.fmean(vals):.2f}" if vals else "-")
            out.append("| " + " | ".join([*key[1:], str(len(ms)), *cells]) + " |")
    return "\n".join(out)
