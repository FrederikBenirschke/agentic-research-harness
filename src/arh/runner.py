"""Run one agent once: its prompt, its tools, a sandboxed shell in its own
working dir, a trace of every message, and a row in the store's runs table.

A research run directory holds:
  config.json    the world it runs on
  store.db       notes, runs, trials (unreadable from agent shells)
  work/data/     the training panel
  work/notes/    frozen note artifacts
  work/<run>/    one agent run's working dir; agents read all of work/, write only their own dir
  traces/<run>.jsonl
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query

from arh import score, tools
from arh.store import Store

AGENTS = Path(__file__).with_name("agents")
DEFAULT_MODEL = "claude-sonnet-5-5"
REF = "## Reference: a map of quant research practice"   # where an arm's primer starts
OLLAMA = os.environ.get("ARH_OLLAMA", "http://localhost:11434")


def model_env(model: str | None) -> dict:
    """A model not named claude-* is served by Ollama's Anthropic-compatible API."""
    if model is None or model.startswith("claude-"):
        return {}
    return {"ANTHROPIC_BASE_URL": OLLAMA, "ANTHROPIC_AUTH_TOKEN": "ollama", "ANTHROPIC_API_KEY": ""}


def load_agent(name: str, prompts: Path | None = None) -> tuple[dict, str]:
    """Frontmatter (``tools``: Bash or none, ``writes``: note kinds) and the prompt. ``prompts``: a dir of
    prompt versions to use instead of agents/ (an agent missing there falls back to agents/)."""
    path = prompts / f"{name}.md" if prompts and (prompts / f"{name}.md").exists() else AGENTS / f"{name}.md"
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(), re.DOTALL)
    if not m:
        raise ValueError(f"{name}.md: no frontmatter")
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): [v.strip() for v in val.split(",") if v.strip()] for k, val in meta.items()}
    return meta, m.group(2).strip()


def primer(prompt: str) -> str:
    """The arm's reference section: from REF up to the appended data summary, or the end."""
    if REF not in prompt:
        return ""
    rest = prompt[prompt.index(REF):]
    return rest[:rest.index("\n\n# Data summary")] if "\n\n# Data summary" in rest else rest


async def one_turn(text: str, model: str | None) -> str:
    """One tool-less turn with no system prompt."""
    out = ""
    async for msg in query(prompt=text, options=ClaudeAgentOptions(
            model=model, max_turns=1, tools=[], allowed_tools=[], setting_sources=[], env=model_env(model),
            extra_args={"strict-mcp-config": None})):
        if isinstance(msg, AssistantMessage):
            out += "".join(getattr(b, "text", "") for b in msg.content)
    return out.strip()


def sandbox(work: Path) -> dict:
    """The CLI's OS sandbox for Bash: no network, the private roots unreadable
    except the work dir and the Python runtime; writes stay in the cwd."""
    allow = [str(work.resolve()), *map(str, score.runtime_paths())]
    return {"enabled": True, "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False,
            "network": {"allowedDomains": []},
            "filesystem": {"denyRead": [str(p) for p in score.private_roots()], "allowRead": allow}}


def _json(msg) -> str:
    return json.dumps(asdict(msg) if is_dataclass(msg) else str(msg), default=str)


async def run_agent(root: Path, agent: str, task: str = "", model: str | None = DEFAULT_MODEL,
                    max_turns: int = 40, max_budget_usd: float = 2.0, timeout_s: float = 1800,
                    prompts: Path | None = None) -> str:
    root = root.resolve()
    world = Path(json.loads((root / "config.json").read_text())["world"])
    store, work = Store(root / "store.db"), root / "work"
    meta, prompt = load_agent(agent, prompts)
    shell = "Bash" in meta.get("tools", [])
    run = f"{agent[:3]}-{secrets.token_hex(3)}"
    cwd = work / run
    (cwd / "tmp").mkdir(parents=True)
    server, names = tools.build(store, run=run, agent=agent, writes=meta["writes"], cwd=cwd, work=work,
                                world=world, can_score="score" in meta.get("tools", []), has_files=shell)
    store.start_run(run, agent, task, model)
    summary = store.search(kind="summary", k=1)
    if summary:                                   # append the data summary to every agent's prompt
        prompt += f"\n\n# Data summary (note #{summary[0]['id']})\n\n{summary[0]['text']}"
    if wanted := store.requests():             # so later agents do not plan around what is missing
        prompt += "\n\n# Not available (requested by earlier agents)\n\n" + "\n".join(f"- {w['want']}" for w in wanted)
    if ask := ", ".join(meta.get("plan", [])):    # a plan from the model's own knowledge, before it sees the primer
        readme = (world / "public" / "README.md").read_text()
        plan = await one_turn(f"{readme}\n\n{ask}", model)
        if review := ", ".join(meta.get("review", [])):   # then, in a second call, what the primer adds to it
            ref = primer(prompt) or prompt
            plan += "\n\n## Added after reading the reference notes\n\n" + await one_turn(
                f"{readme}\n\nYour plan, written earlier:\n\n{plan}\n\nReference notes:\n\n<notes>\n{ref}\n</notes>\n\n{review}", model)
        if "drop_reference" in meta.get("plan_mode", []) and (ref := primer(prompt)):
            prompt = prompt.replace(ref, "")
        (cwd / "own_plan.md").write_text(plan)
        prompt = (f"# Your own plan\n\nYou wrote this from your own knowledge before reading anything below. "
                  f"It comes first: run every check in it, and say in the note which ones showed something.\n\n"
                  f"{plan}\n\n# Instructions\n\n{prompt}")
    opts = ClaudeAgentOptions(
        system_prompt=prompt,
        tools=["Bash"] if shell else [],
        mcp_servers={tools.SERVER: server},
        allowed_tools=(["Bash"] if shell else []) + names,
        setting_sources=[],                       # no host CLAUDE.md, settings or hooks
        cwd=str(cwd),
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        # no background Bash: an agent that backgrounds a slow fit waits on it and ends without a note.
        # Own TMPDIR: the default one is shared, so parallel runs overwrote and ran each other's scripts.
        # One BLAS/OpenMP thread: parallel runs oversubscribe the cores (a 0.3 s HGB fit took 37 s).
        env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
             "BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "600000", "TMPDIR": str(cwd / "tmp"),
             **score.ONE_THREAD, **model_env(model)},
        sandbox=sandbox(work),
        extra_args={"strict-mcp-config": None},   # no claude.ai connectors or host MCP servers
        **({} if model is None or model.startswith("claude-")   # local models decode at ~14 tok/s, so limit reasoning
           else {"effort": "low", "max_thinking_tokens": 1024}),
    )
    (root / "traces").mkdir(exist_ok=True)
    final, error = None, None
    with open(root / "traces" / f"{run}.jsonl", "w") as trace:
        try:
            async with asyncio.timeout(timeout_s):
                async for msg in query(prompt=task or ", ".join(meta.get("task", [])) or "Do one unit of work, then stop.",
                                       options=opts):
                    trace.write(_json(msg) + "\n")
                    if isinstance(msg, ResultMessage):
                        final = msg
                    elif isinstance(msg, AssistantMessage):
                        trace.flush()
        except Exception as e:                    # record the run with its error (budget cap, timeout, crash)
            error = f"{type(e).__name__}: {e}"[:1000]
    store.end_run(run, getattr(final, "total_cost_usd", None), getattr(final, "num_turns", None),
                  error or (None if final is None or not final.is_error else final.subtype))
    return run
