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
    cwd.mkdir(parents=True)
    server, names = tools.build(store, run=run, agent=agent, writes=meta["writes"], cwd=cwd, work=work,
                                world=world, can_score="score" in meta.get("tools", []), has_files=shell)
    store.start_run(run, agent, task, model)
    summary = store.search(kind="summary", k=1)
    if summary:                                   # append the data summary to every agent's prompt
        prompt += f"\n\n# Data summary (note #{summary[0]['id']})\n\n{summary[0]['text']}"
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
        # no background Bash: an agent that backgrounds a slow fit waits on it and ends without a note
        env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
             "BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "600000", **model_env(model)},
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
                async for msg in query(prompt=task or "Do one unit of work, then stop.", options=opts):
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
