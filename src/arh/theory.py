"""Score a theory: a list of mechanism instances (arh.mechanisms) fitted together.

A theory is compiled into an ordinary model directory (model.py, mechanisms.py, theory.json), so it is scored
by the same sandboxed walk-forward CV and holdout as every agent's model. On top of the score, each instance
gets two numbers, both fold-paired differences in CV R^2:

  necessity    the theory's CV minus its CV without the instance (what the instance adds to the rest)
  sufficiency  the CV of the instance alone minus the CV of no instance at all

``pieces`` does the same for a model written with ``mechanisms.Pieces`` (any code, scored as an ordinary
model file): the necessity of each named piece.

``grade`` matches instances to a world's planted mechanisms. It reads the sealed truth and is for evals only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from arh import mechanisms, score

MODEL = '''import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import mechanisms

SPEC = json.loads((Path(__file__).parent / "theory.json").read_text())


def fit(train):
    feats = [c for c in train.columns if c not in ("date", "asset", "y")]
    return mechanisms.Theory(SPEC, feats).fit(train)


def predict(state, history, today):
    return state.predict(history, today)
'''


def compile(spec: dict, out: Path) -> Path:
    """A model directory for ``spec`` ({"targets": [...], "terms": [...]}); returns its model.py."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mechanisms.__file__, out / "mechanisms.py")
    (out / "theory.json").write_text(json.dumps(spec, indent=1))
    (out / "model.py").write_text(MODEL)
    return out / "model.py"


def instances(spec: dict) -> list[tuple[str, dict]]:
    return [("targets", i) for i in spec.get("targets", [])] + [("terms", i) for i in spec.get("terms", [])]


def without(spec: dict, k: int) -> dict:
    keep = [x for j, x in enumerate(instances(spec)) if j != k]
    return {part: [i for p, i in keep if p == part] for part in ("targets", "terms")}


def only(spec: dict, k: int) -> dict:
    part, inst = instances(spec)[k]
    return {"targets": [], "terms": [], part: [inst]}


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    d = np.array(a) - np.array(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def evaluate(spec: dict, train: pl.DataFrame, out: Path, world: Path | None = None) -> dict:
    """CV of the theory, and necessity and sufficiency of each instance; holdout too if ``world`` is given.
    Every variant is compiled under ``out``."""
    feats = [c for c in train.columns if c not in ("date", "asset", "y")]
    for _, inst in instances(spec):
        mechanisms.check(inst, feats)   # fail before any CV runs
    full, _ = score.cv_full(compile(spec, out / "full"), train, oof=out / "full" / "resid.parquet")
    empty, _ = score.cv_full(compile({"targets": [], "terms": []}, out / "empty"), train)
    rows = []
    for k, (part, inst) in enumerate(instances(spec)):
        rest = score.cv(compile(without(spec, k), out / f"without{k}"), train) if len(instances(spec)) > 1 else empty
        alone = score.cv(compile(only(spec, k), out / f"only{k}"), train)
        rows.append({"instance": mechanisms.label(mechanisms.check(inst, feats)), "part": part,
                     "necessity": paired(full, rest), "sufficiency": paired(alone, empty)})
    res = {"cv": float(np.mean(full)), "folds": full, "instances": rows}
    if world:
        res["holdout"] = score.holdout(out / "full" / "model.py", world)
    return res


# ---- grading against the planted truth (evals only) ----------------------------------------------------

# Which templates can express each family of the world generator, and which of the family's features the
# instance must name. Only the grader sees this; no agent prompt does.
EXPRESSES = {
    "linear": {"linear", "saturating", "lag"}, "tanh": {"saturating", "linear"}, "publag": {"lag"},
    "lag_y": {"own_lag", "ou"}, "decay": {"own_lag", "ou", "lag"}, "hinge": {"hinge"}, "step": {"hinge", "gate"},
    "bump": {"bump"}, "sine": {"periodic"}, "product": {"interaction"}, "gated": {"gate", "interaction"},
    "regime": {"gate"}, "sector": {"group", "factor"}, "market": {"market", "factor"}, "hetero": {"vol"},
    "csz": {"linear", "group"}, "rank": {"linear", "saturating"}, "winsor": {"saturating", "hinge"},
}


def named(inst: dict) -> set[str]:
    p = inst["params"]
    return {v for k, v in p.items() if isinstance(v, str) and v not in ("", "frame")
            and mechanisms.TEMPLATES[inst["template"]].params.get(k) in ("feature", "series")}


def grade(spec: dict, result: dict, world: Path) -> dict:
    """For each planted mechanism, the instances that could express it, name one of its features and add CV
    (necessity more than one standard error above 0); for each instance, the mechanism it matches or None."""
    truth = json.loads((world / "sealed" / "truth.json").read_text())
    insts = [i for _, i in instances(spec)]
    feats = [c for c in pl.read_parquet(world / "public" / "train.parquet").columns if c.startswith("x")]
    insts = [mechanisms.check(i, feats) for i in insts]
    found, match = {}, [None] * len(insts)
    for m in truth["mechanisms"]:
        hits = []
        for j, (inst, row) in enumerate(zip(insts, result["instances"])):
            mean, se = row["necessity"]
            if inst["template"] in EXPRESSES.get(m["family"], set()) and named(inst) & set(m["features"]) \
                    and mean > se:
                hits.append(j)
                match[j] = m["id"]
        found[m["id"]] = hits
    return {"mechanisms": found, "instances": match,
            "recall": sum(bool(h) for h in found.values()) / max(1, len(found)),
            "spurious": sum(m is None and r["necessity"][0] > r["necessity"][1]
                            for m, r in zip(match, result["instances"]))}


# ---- necessity of the pieces of a model file -----------------------------------------------------------

DROP = '''import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("base", Path(__file__).with_name({base!r}))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
MODEL = _base.MODEL.without({k})
fit, predict = MODEL.fit, MODEL.predict
'''


def pieces(model: Path, train: pl.DataFrame, full: list[float]) -> list[dict] | None:
    """For a model file written with ``Pieces`` whose CV folds are ``full``: per piece its name, claim and
    necessity (fold-paired CV lost when it is dropped and the rest refitted, mean and SE). None if the model
    has no pieces. The variants are written next to the model and removed after."""
    names = score.pieces(model)
    if not names:
        return None
    rows = []
    for k, (name, claim) in enumerate(names):
        drop = model.with_name(f"{model.stem}_drop{k}.py")
        drop.write_text(DROP.format(base=model.name, k=k))
        try:
            rest = score.cv(drop, train)
        finally:
            drop.unlink(missing_ok=True)
        mean, se = paired(full, rest)
        rows.append({"name": name, "claim": claim, "necessity": mean, "se": se})
    return rows


def table(rows: list[dict] | None) -> str:
    return "; ".join(f"{r['name']} {r['necessity']:+.4f} ± {r['se']:.4f}" for r in rows or [])
