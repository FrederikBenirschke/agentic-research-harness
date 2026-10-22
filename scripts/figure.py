"""Draw docs/results.png from results/*.jsonl. No model calls.

    python scripts/figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tables import MODELS, PLANTED3, PROMPT, RESULTS, agent, load, ratios

OUT = Path(__file__).resolve().parents[1] / "docs" / "results.png"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
V1, CUR = "#eb6834", "#2a78d6"   # categorical slots 2 and 1 (validated, light surface)
SHORT = {"gpt-oss-32k": "gpt-oss (n=3)", "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5",
         "claude-opus-5-5": "Opus 5.5", "claude-fable-5-1": "Fable 5.1"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ablation(ax):
    """Per-run forecaster holdout/oracle on held-out planted1b: original vs tuned prompt, per model."""
    for i, m in enumerate(MODELS):
        for v, color, dy in (("v1", V1, 0.17), ("cur", CUR, -0.17)):
            r = ratios(agent(load(RESULTS / "ablation" / f"{m}-{v}-planted1b.jsonl"), "forecaster"))
            counts: dict[float, int] = {}
            for x in r:   # runs often land on the same score: one dot per value, labelled with its count
                counts[round(x, 2)] = counts.get(round(x, 2), 0) + 1
            ax.scatter(list(counts), [i + dy] * len(counts), s=42, color=color, edgecolor=SURFACE, linewidth=1.5,
                       zorder=5, label=f"{PROMPT[v]} prompt" if i == 0 else None)
            for x, n in counts.items():
                if n > 1:
                    ax.text(x + 0.012, i + dy, f"×{n}", va="center", fontsize=8, color=MUTED)
            ax.plot([sum(r) / len(r)] * 2, [i + dy - 0.13, i + dy + 0.13], color=color, linewidth=2, zorder=4)
    ax.set_yticks(range(len(MODELS)), [SHORT[m] for m in MODELS], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("score: forecaster holdout R² / oracle (dots = runs, ×n = n runs at that value; bar = mean)",
                  color=MUTED, fontsize=9)
    ax.set_title("planted1b, not used for tuning: original vs tuned prompt", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    style(ax)


def discovery(ax):
    """planted3 analysts that found the U-shape (no prompt names it), per arm."""
    arms = [(f"{model.split()[0]}, {arm}" + ("" if w == "planted3" else " (planted3b)"),
             load(RESULTS / "planted3" / f"{s}.jsonl"))
            for s, w, model, arm in PLANTED3 if "forecaster" not in arm]
    names, hits = [], []
    for name, rows in arms:
        a = agent(rows, "analyst")
        names.append(name)
        hits.append(sum(any("ushape" in h for h in r.get("mechanisms_hit") or []) for r in a))
    colors = [CUR if "squares" in n else MUTED for n in names]
    ax.barh(range(len(names)), hits, height=0.6, color=colors, zorder=3)
    for i, h in enumerate(hits):
        ax.text(h + 0.08, i, f"{h}/5", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)), names, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 5.6)
    ax.set_xticks(range(6))
    ax.set_xlabel("analyst runs (of 5) that found the U-shape in x04", color=MUTED, fontsize=9)
    ax.set_title("planted3: a U-shape that no prompt describes", loc="left", color=INK, fontsize=11)
    style(ax)


if __name__ == "__main__":
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    ablation(a)
    discovery(b)
    fig.tight_layout(w_pad=3)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT}")
