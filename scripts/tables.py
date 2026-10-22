"""Rebuild the README's result tables from results/*.jsonl. No model calls.

    python scripts/tables.py            # prints markdown
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = ["gpt-oss-32k", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def agent(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["condition"].startswith(name)]


def mean_sd(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.fmean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def frac(rows: list[dict], test) -> str:
    return f"{sum(bool(test(r)) for r in rows)}/{len(rows)}" if rows else "-"


def hit(rows: list[dict], key: str) -> str:
    return frac(rows, lambda r: any(key in h for h in r.get("mechanisms_hit") or []))


def ratios(rows: list[dict]) -> list[float]:
    return [r["holdout_vs_oracle"] for r in rows if r.get("holdout_vs_oracle") is not None]


def cost(rows: list[dict]) -> str:
    c = [r["cost"] for r in rows if r.get("cost")]
    return f"${statistics.fmean(c):.3f}" if c else "-"


def ablation() -> str:
    out = ["| model | prompt | forecaster holdout/oracle (planted1b) | forecaster x07 gate | forecaster uses decoy | "
           "analyst x07 gate | control holdout/oracle (planted2) | $/run |",
           "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for v in ("v1", "cur"):
            held, ctrl = RESULTS / "ablation" / f"{m}-{v}-planted1b.jsonl", RESULTS / "ablation" / f"{m}-{v}-planted2.jsonl"
            if not held.exists():
                continue
            rows = load(held)
            f, a = agent(rows, "forecaster"), agent(rows, "analyst")
            c = agent(load(ctrl), "forecaster") if ctrl.exists() else []
            out.append(f"| {m} | {v} | {mean_sd(ratios(f))} (n={len(f)}) | {hit(f, 'x07')} | "
                       f"{frac(f, lambda r: r.get('decoy_in_model'))} | {hit(a, 'x07')} | {mean_sd(ratios(c))} | "
                       f"{cost(rows) if m.startswith('claude') else 'local'} |")
    return "\n".join(out)


def avg(rows: list[dict], key: str, fmt: str = ".1f") -> str:
    v = [float(r[key]) for r in rows if r.get(key) is not None]
    return format(statistics.fmean(v), fmt) if v else "-"


def behaviour() -> str:
    """How each model works, pooled over both worlds: process, honesty and judge metrics per agent."""
    out = ["| model | prompt | agent | n | turns | tool calls | Bash | tool errors | trials | wall s | out tokens | "
           "numbers grounded | rerun ok | judge evidence | judge correct |", "|---|" * 15]
    for m in MODELS:
        for v in ("v1", "cur"):
            rows = [r for w in ("planted1b", "planted2") if (f := RESULTS / "ablation" / f"{m}-{v}-{w}.jsonl").exists()
                    for r in load(f)]
            for name in ("analyst", "forecaster"):
                a = agent(rows, name)
                for r in a:
                    r["bash"] = (r.get("tools") or {}).get("Bash", 0)
                    r["correct"] = r.get("judge_correct") == "correct" if r.get("judge_correct") else None
                out.append(f"| {m} | {v} | {name} | {len(a)} | {avg(a, 'turns')} | {avg(a, 'tool_calls')} | "
                           f"{avg(a, 'bash')} | {avg(a, 'tool_error_rate', '.2f')} | "
                           f"{avg(a, 'trials') if name == 'forecaster' else '-'} | {avg(a, 'wall_s', '.0f')} | "
                           f"{avg(a, 'out_tokens', '.0f')} | {avg(a, 'numbers_grounded', '.2f')} | "
                           f"{frac(a, lambda r: r.get('rerun_ok')) if name == 'analyst' else '-'} | "
                           f"{avg(a, 'judge_evidence')} | {avg(a, 'correct', '.2f')} |")
    return "\n".join(out[:1] + ["|" + "---|" * 15] + out[2:])


PLANTED3 = [  # (file stem, world, arm description)
    ("claude-haiku-5-5-cur", "planted3", "Haiku, current prompts"),
    ("claude-sonnet-5-5-cur", "planted3", "Sonnet, current prompts"),
    ("claude-opus-5-5-cur", "planted3", "Opus, current prompts"),
    ("claude-haiku-5-5-cur-nosummary", "planted3", "Haiku, no data summary"),
    ("claude-haiku-5-5-eda", "planted3", "Haiku, + EDA line"),
    ("claude-haiku-5-5-eda-nosum", "planted3", "Haiku, + EDA line, no summary"),
    ("claude-haiku-5-5-move", "planted3", "Haiku, + confound move"),
    ("claude-haiku-5-5-3b-cur", "planted3b", "Haiku, current prompts"),
    ("claude-haiku-5-5-3b-move", "planted3b", "Haiku, + confound move"),
    ("claude-haiku-5-5-chain", "planted3", "forecaster given one move-arm finding"),
    ("claude-haiku-5-5-control", "planted3", "forecaster given a finding naming all three"),
]


def planted3() -> str:
    out = ["| world | arm | analyst finds U-shape | analyst finds lag 2 | forecaster holdout/oracle | "
           "forecaster uses decoy |", "|---|---|---|---|---|---|"]
    for stem, world, arm in PLANTED3:
        rows = load(RESULTS / "planted3" / f"{stem}.jsonl")
        a, f = agent(rows, "analyst"), agent(rows, "forecaster")
        r = ratios(f)
        rng = f"{min(r):.2f}–{max(r):.2f}" if r else "-"
        out.append(f"| {world} | {arm} | {hit(a, 'ushape')} | {hit(a, 'lag2')} | {rng} | "
                   f"{frac(f, lambda r: r.get('decoy_in_model'))} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("## Model x prompt ablation\n\n" + ablation() + "\n\n## Behaviour by model\n\n" + behaviour() +
          "\n\n## planted3: mechanisms no prompt names\n\n" + planted3())
