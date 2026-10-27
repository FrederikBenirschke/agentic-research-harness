"""Draw the figures in docs/fig/ from results/. No model calls.

    python scripts/figure.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from tables import MODELS, RESULTS, agent, load, ratios

OUT = Path(__file__).resolve().parents[1] / "docs" / "fig"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, FADED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"   # categorical slots 1-3; gray = not kept
SHORT = {"gpt-oss-32k": "gpt-oss (n=3)", "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5",
         "claude-opus-5-5": "Opus 5.5", "claude-fable-5-1": "Fable 5.1"}
ORACLE = {"r19": 0.1530738203124029, "r112": 0.1218126913954799}


def style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=10.5)


def dots(ax, xs, y, color, label=None):
    """One dot per distinct value (runs often tie), labelled ×n, and a bar at the mean."""
    counts: dict[float, int] = {}
    for x in xs:
        counts[round(x, 2)] = counts.get(round(x, 2), 0) + 1
    ax.scatter(list(counts), [y] * len(counts), s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=5,
               label=label)
    for x, n in counts.items():
        if n > 1 and any(0 < o - x < 0.06 for o in counts):   # a dot just right of it: label above instead
            ax.text(x, y - 0.17, f"×{n}", ha="center", va="bottom", fontsize=8, color=MUTED)
        elif n > 1:
            ax.text(x + 0.012, y, f"×{n}", va="center", fontsize=8, color=MUTED)
    ax.plot([np.mean(xs)] * 2, [y - 0.13, y + 0.13], color=color, linewidth=2, zorder=4)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def leak():
    """A prompt tuned on planted1 helps on planted1b (same rules) and does nothing on planted3 (rules it does
    not describe)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.25, 1]}, sharey=True)
    models = MODELS[1:4]
    for i, m in enumerate(models):
        for v, color, dy, lab in (("v1", ORANGE, 0.17, "original prompt"), ("cur", BLUE, -0.17, "tuned prompt")):
            dots(a, ratios(agent(load(RESULTS / "ablation" / f"{m}-{v}-planted1b.jsonl"), "forecaster")), i + dy,
                 color, lab if i == 0 else None)
        dots(b, ratios(agent(load(RESULTS / "planted3" / f"{m}-cur.jsonl"), "forecaster")), i - 0.17, BLUE)
    a.set_yticks(range(len(models)), [SHORT[m] for m in models], color=INK)
    a.invert_yaxis()
    for ax in (a, b):
        ax.set_xlim(0.3, 1.02)
        ax.set_xlabel("holdout R² / oracle (dot = run, bar = mean)", color=MUTED, fontsize=9)
        style(ax)
    title(a, "planted1b: same hidden rules the prompt was tuned on")
    title(b, "planted3: rules no prompt describes")
    b.text(0.62, 1.2, "U-shape found in 0 of 15 runs;\ntrap column used in 15 of 15", fontsize=8.5, color=MUTED,
           va="center")
    a.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    save(fig, "1-leak.png")


def planning():
    """Counts from docs/results.md sections 2 and 3 (one-message answers and agent runs)."""
    rows = [("Haiku answers naming PCA", [("no reference list", 5, 5), ("reference list in prompt", 0, 5)]),
            ("planted8 runs: PCA or grouped assets", [("list in prompt", 1, 30),
                                                                            ("own checklist from a\nseparate call", 17, 18)])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6))
    for ax, (name, bars) in zip(axes, rows):
        for j, (lab, k, n) in enumerate(bars):
            ax.barh(j, k / n, height=0.55, color=[ORANGE, BLUE][j], zorder=3)
            ax.text(k / n + 0.02, j, f"{k}/{n}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(bars)), [b[0] for b in bars], color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        title(ax, name.replace("\n", " "))
        style(ax)
    save(fig, "2-planning.png")


def groups():
    """Mean score per random world, normal data summary vs the summary with the computed group line."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, sharex=True)
    worlds = {}
    for f in sorted((RESULTS / "groups").glob("r*-default.jsonl")):
        w = f.name.split("-")[0]
        if w == "r10":   # from an earlier batch; results.md section 5 reports the other eight
            continue
        if (RESULTS / "groups" / f"sonnet-{w}-default.jsonl").exists() or w in ("r45", "r46", "r47", "r50", "r51", "r52"):
            truth = json.loads((Path(__file__).resolve().parents[1] / "worlds" / "rand" / w / "sealed" / "truth.json")
                               .read_text())
            worlds[w] = any(m["family"] == "sector" for m in truth["mechanisms"])
    order = sorted(worlds, key=lambda w: (not worlds[w], int(w[1:])))
    for ax, prefix, name in ((axes[0], "", "Haiku 5.5"), (axes[1], "sonnet-", "Sonnet 5.5")):
        for i, w in enumerate(order):
            pair = []
            for v in ("default", "groups"):
                f = RESULTS / "groups" / f"{prefix}{w}-{v}.jsonl"
                pair.append(np.mean(ratios(agent(load(f), "forecaster"))) if f.exists() else None)
            if None in pair:
                continue
            ax.plot(pair, [i, i], color=GRID, linewidth=2, zorder=2)
            ax.scatter([pair[0]], [i], s=40, color=ORANGE, zorder=4, edgecolor=SURFACE, linewidth=1.5,
                       label="normal summary" if i == 0 else None)
            ax.scatter([pair[1]], [i], s=40, color=BLUE, zorder=4, edgecolor=SURFACE, linewidth=1.5,
                       label="with group line" if i == 0 else None)
        n = sum(worlds.values())
        ax.axhline(n - 0.5, color=MUTED, linewidth=0.8)
        ax.text(1.02, n / 2 - 0.5, "sector rule", rotation=-90, va="center", fontsize=8.5, color=MUTED,
                transform=ax.get_yaxis_transform())
        ax.text(1.02, n + (len(order) - n) / 2 - 0.5, "no sector rule", rotation=-90, va="center", fontsize=8.5,
                color=MUTED, transform=ax.get_yaxis_transform())
        ax.set_xlabel("holdout R² / oracle, mean of 3 runs", color=MUTED, fontsize=9)
        title(ax, name)
        style(ax)
    axes[0].set_yticks(range(len(order)), order, color=INK)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    save(fig, "3-groups.png")


def handoff():
    """A Haiku forecaster on planted1b alone, and handed a finding by a Haiku or an Opus analyst."""
    alone = ratios(agent(load(RESULTS / "ablation" / "claude-haiku-5-5-cur-planted1b.jsonl"), "forecaster"))
    chain = load(RESULTS / "ablation" / "chain-haiku.jsonl")
    rows = [("from the data alone", alone, FADED),
            ("+ finding by a Haiku analyst", ratios([r for r in chain if r["condition"] == "forecaster-haikufinding"]), ORANGE),
            ("+ finding by an Opus analyst", ratios([r for r in chain if r["condition"] == "forecaster-opusfinding"]), BLUE)]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for i, (_, xs, color) in enumerate(rows):
        dots(ax, xs, i, color)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("holdout R² / oracle (dot = run, bar = mean)", color=MUTED, fontsize=9)
    title(ax, "Haiku forecaster on planted1b, 5 runs each")
    style(ax)
    save(fig, "4-handoff.png")


def best_so_far(path: Path, oracle: float) -> tuple[list[int], list[float]]:
    """After each model, the holdout of the best model by CV so far (what the search would hand in)."""
    rows = [r for r in csv.DictReader(path.open()) if r["cv"] and r["holdout"]]
    xs, ys, best = [], [], None
    for k, r in enumerate(rows, 1):
        if best is None or float(r["cv"]) > float(best["cv"]):
            best = r
        xs.append(k)
        ys.append(float(best["holdout"]) / oracle)
    return xs, ys


def search():
    runs = json.loads((RESULTS / "search" / "runs.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, world in zip(axes, ("r19", "r112")):
        ends = []
        for run, label in runs.items():
            if not run.startswith(world + "-"):
                continue
            color = BLUE if label.startswith("full tree") else ORANGE
            ls = "--" if ("playbook" in label or "no screen" in label or label.startswith("greedy,")) else "-"
            xs, ys = best_so_far(RESULTS / "search" / f"{run}.csv", ORACLE[world])
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, zorder=3)
            ends.append([ys[-1], xs[-1], label])
        ends.sort(reverse=True)
        col = max(x for _, x, _ in ends) + 4   # labels in one column right of the lines, joined by leaders
        ly = min(ends[0][0], 0.98)
        for k, (y, x, label) in enumerate(ends):   # keep labels at least 0.05 apart, top down
            ly = y if k == 0 else min(y, ly - 0.05)
            ly = min(ly, 0.98)
            ax.plot([x, col - 0.5], [y, ly], color=GRID, linewidth=0.8, zorder=1)
            ax.text(col, ly, label, fontsize=7.5, color=MUTED, va="center")
        ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlabel("models scored", color=MUTED, fontsize=9)
        ax.set_xlim(0, 110 if world == "r19" else 82)
        title(ax, f"{world} (oracle R² {ORACLE[world]:.3f})")
        style(ax, grid="y")
    axes[0].set_ylabel("holdout / oracle of the best model by CV", color=MUTED, fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].plot([], [], color=BLUE, linewidth=2, label="full tree")
    axes[0].plot([], [], color=ORANGE, linewidth=2, label="greedy (+ survey)")
    axes[0].plot([], [], color=MUTED, linewidth=2, linestyle="--", label="variant: playbook, no screen, no survey")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    save(fig, "5-search.png")


def theory():
    t = json.loads((RESULTS / "theory" / "r19.json").read_text())
    kept = {r["instance"] for r in t["pruned"]}
    rows = sorted(t["proposed"], key=lambda r: r["necessity"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(rows):
        color = BLUE if r["instance"] in kept else FADED
        ax.barh(i, r["necessity"], height=0.6, color=color, zorder=3)
        ax.errorbar(r["necessity"], i, xerr=r["se"], color=INK, linewidth=1, capsize=2, zorder=4)
    ax.set_yticks(range(len(rows)), [r["instance"] for r in rows], color=INK, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("necessity: CV R² lost when the piece is dropped and the rest refitted (± 1 SE)",
                  color=MUTED, fontsize=9)
    h = t["holdout"]
    title(ax, f"r19: {len(rows)} pieces proposed from the findings, holdout {h['proposed']:.4f}\n"
              f"{len(kept)} kept, refitted: holdout {h['pruned']:.4f} (oracle {t['oracle']:.4f})")
    ax.legend(handles=[Patch(color=BLUE, label="kept: necessity > 1 SE"), Patch(color=FADED, label="dropped")],
              frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    style(ax)
    save(fig, "6-theory.png")


if __name__ == "__main__":
    for f in (leak, planning, groups, handoff, search, theory):
        f()
