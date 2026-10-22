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


NAME = {"gpt-oss-32k": "gpt-oss (local)", "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5",
        "claude-opus-5-5": "Opus 5.5", "claude-fable-5-1": "Fable 5.1"}
PROMPT = {"v1": "original", "cur": "tuned"}


def ablation() -> str:
    out = ["| model | prompt | score on planted1b | forecaster found x07 gate | analyst found x07 gate | "
           "forecaster used decoy | score on planted2 (control) | cost per run |",
           "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for v in ("v1", "cur"):
            held, ctrl = RESULTS / "ablation" / f"{m}-{v}-planted1b.jsonl", RESULTS / "ablation" / f"{m}-{v}-planted2.jsonl"
            if not held.exists():
                continue
            rows = load(held)
            f, a = agent(rows, "forecaster"), agent(rows, "analyst")
            c = agent(load(ctrl), "forecaster") if ctrl.exists() else []
            out.append(f"| {NAME[m]} | {PROMPT[v]} | {mean_sd(ratios(f))} | {hit(f, 'x07')} | {hit(a, 'x07')} | "
                       f"{frac(f, lambda r: r.get('decoy_in_model'))} | {mean_sd(ratios(c))} | "
                       f"{cost(rows) if m.startswith('claude') else '-'} |")
    return "\n".join(out)


def avg(rows: list[dict], key: str, fmt: str = ".1f") -> str:
    v = [float(r[key]) for r in rows if r.get(key) is not None]
    return format(statistics.fmean(v), fmt) if v else "-"


def pooled(m: str, v: str, name: str) -> list[dict]:
    rows = [r for w in ("planted1b", "planted2") if (f := RESULTS / "ablation" / f"{m}-{v}-{w}.jsonl").exists()
            for r in load(f)]
    a = agent(rows, name)
    for r in a:
        r["bash"] = (r.get("tools") or {}).get("Bash", 0)
        r["correct"] = r.get("judge_correct") == "correct" if r.get("judge_correct") else None
    return a


def behaviour(name: str) -> str:
    """Process, grounding and judge metrics for one agent per model and prompt, pooled over both worlds."""
    last = "models tried" if name == "forecaster" else "script reproduces"
    out = [f"| model | prompt | runs | turns | Bash calls | tool error rate | seconds | output tokens | "
           f"numbers grounded | judge: correct | {last} |", "|" + "---|" * 11]
    for m in MODELS:
        for v in ("v1", "cur"):
            a = pooled(m, v, name)
            if not a:
                continue
            extra = avg(a, "trials") if name == "forecaster" else frac(a, lambda r: r.get("rerun_ok"))
            out.append(f"| {NAME[m]} | {PROMPT[v]} | {len(a)} | {avg(a, 'turns')} | {avg(a, 'bash')} | "
                       f"{avg(a, 'tool_error_rate', '.2f')} | {avg(a, 'wall_s', '.0f')} | "
                       f"{avg(a, 'out_tokens', '.0f')} | {avg(a, 'numbers_grounded', '.2f')} | "
                       f"{avg(a, 'correct', '.2f')} | {extra} |")
    return "\n".join(out)


PLANTED3 = [  # (file stem, world, model, prompt or condition)
    ("claude-haiku-5-5-cur", "planted3", "Haiku 5.5", "tuned"),
    ("claude-sonnet-5-5-cur", "planted3", "Sonnet 5.5", "tuned"),
    ("claude-opus-5-5-cur", "planted3", "Opus 5.5", "tuned"),
    ("claude-haiku-5-5-cur-nosummary", "planted3", "Haiku 5.5", "tuned, no data summary"),
    ("claude-haiku-5-5-eda", "planted3", "Haiku 5.5", "tuned + EDA instruction"),
    ("claude-haiku-5-5-eda-nosum", "planted3", "Haiku 5.5", "tuned + EDA instruction, no data summary"),
    ("claude-haiku-5-5-move", "planted3", "Haiku 5.5", "tuned + squares instruction"),
    ("claude-haiku-5-5-3b-cur", "planted3b", "Haiku 5.5", "tuned"),
    ("claude-haiku-5-5-3b-move", "planted3b", "Haiku 5.5", "tuned + squares instruction"),
    ("claude-haiku-5-5-chain", "planted3", "Haiku 5.5", "forecaster given one finding from the squares arm"),
    ("claude-haiku-5-5-control", "planted3", "Haiku 5.5", "forecaster given a note naming all three mechanisms"),
]


def planted3() -> str:
    out = ["| world | model | prompt or condition | analyst found U-shape | analyst found lag 2 | score (min–max) | "
           "forecaster used decoy |", "|---|---|---|---|---|---|---|"]
    for stem, world, model, arm in PLANTED3:
        rows = load(RESULTS / "planted3" / f"{stem}.jsonl")
        a, f = agent(rows, "analyst"), agent(rows, "forecaster")
        r = ratios(f)
        lo, hi = (f"{min(r):.2f}", f"{max(r):.2f}") if r else ("-", "-")
        rng = lo if lo == hi else f"{lo}–{hi}"
        out.append(f"| {world} | {model} | {arm} | {hit(a, 'ushape')} | {hit(a, 'lag2')} | {rng} | "
                   f"{frac(f, lambda r: r.get('decoy_in_model'))} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("## Tuned prompts on new data\n\n" + ablation() + "\n\n## Forecasters\n\n" + behaviour("forecaster") +
          "\n\n## Analysts\n\n" + behaviour("analyst") + "\n\n## planted3\n\n" + planted3())
