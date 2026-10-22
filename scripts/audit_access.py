"""Scan agent traces for reads outside the run's own work dir. No model calls.

Flags a Bash command that names a path outside its run (after removing the run's own root),
and any tool output that shows sealed files, truth, generators or other worlds.

    python scripts/audit_access.py runs/evals
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CMD = re.compile(r"sealed|truth|holdout\.parquet|make_planted|worlds|/Users/|~/|\$HOME|\.\./\.\./\.\.|find /|ls /")
OUT = re.compile(r"sealed|truth\.json|holdout\.parquet|make_planted|worlds/|Operation not permitted|Permission denied")


def blocks(trace: Path):
    for line in trace.read_text().splitlines():
        content = json.loads(line).get("content")
        for b in content if isinstance(content, list) else []:
            if isinstance(b, dict) and b.get("name") == "Bash":
                yield "cmd", b["input"].get("command", "")
            elif isinstance(b, dict) and "tool_use_id" in b:
                c = b.get("content")
                yield "out", c if isinstance(c, str) else json.dumps(c)


if __name__ == "__main__":
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/evals").resolve()
    traces = sorted(base.rglob("traces/*.jsonl"))
    n_cmd, flags = 0, []
    for t in traces:
        root = str(t.parent.parent)
        for kind, text in blocks(t):
            n_cmd += kind == "cmd"
            clean = text.replace(root, "<run>")
            hit = (CMD if kind == "cmd" else OUT).search(clean)
            if hit:
                flags.append(f"{t.parent.parent.name}/{t.stem} {kind}: {clean[max(0, hit.start() - 100):hit.end() + 150]!r}")
    print("\n".join(flags))
    print(f"{len(traces)} traces, {n_cmd} Bash commands, {len(flags)} flagged")
