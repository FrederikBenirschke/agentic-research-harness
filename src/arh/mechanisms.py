"""Mechanism templates: general hypotheses about what drives y, each with typed parameters.

A template is written by hand; a search step fills in its parameters, and the harness computes what the
filled-in template (an instance) contributes. A theory is a list of target transforms (what the model fits)
and terms (the columns it fits with). ``Theory`` fits one on training rows and forecasts new dates, using
only rows before each date, so a theory is scored like any model.

This file is copied next to a compiled theory model and must import nothing from arh.

Kinds:
  target  changes what is fitted, and maps a forecast of it back to y (factor, vol, rank)
  term    adds a column to the fit (linear, lag, saturating, hinge, bump, periodic, interaction, group,
          market, lead, own_lag, ou, gate)

``Pieces`` is a model written as a sum of named pieces, each a template instance, a function that makes
columns, or a ``Target`` written as code; the harness scores what each piece adds by dropping it.

Every feature enters a term as its within-date z-score, so a feature's units never matter. A term reads
the target only at past dates, through ``frame``: y after the theory's target transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl


@dataclass
class Template:
    kind: str                      # "target" or "term"
    says: str                      # the hypothesis, in general terms
    params: dict[str, str]         # name -> type: feature, series, int lo..hi, float lo..hi, choice a|b
    defaults: dict = field(default_factory=dict)
    lookback: Callable[[dict], int] = lambda p: 2   # past dates a forecast needs


def z(c: str) -> pl.Expr:
    m, s = pl.col(c).mean().over("date"), pl.col(c).std().over("date")
    return ((pl.col(c) - m) / s).fill_nan(0.0).fill_null(0.0)


def past(e: pl.Expr, k: int = 1) -> pl.Expr:
    return e.shift(k).over("asset")


TEMPLATES: dict[str, Template] = {
    # target transforms
    "factor": Template("target", "Common moves hide asset-specific signal: fit y net of common factors (the "
                       "date mean, a group's date mean, or the top principal components of past y).",
                       {"kind": "choice market|group|pca", "group": "feature", "k": "int 1..10"},
                       {"kind": "market", "group": "", "k": 3}),
    "vol": Template("target", "The size of y varies by asset and time: fit y divided by a forecast of its size "
                    "(its trailing absolute value per asset, or a fit of log|y| on a feature).",
                    {"by": "choice past|feature", "feature": "feature", "window": "int 5..250"},
                    {"by": "past", "feature": "", "window": 60}, lambda p: p.get("window", 60) + 2),
    "rank": Template("target", "Only the ordering across assets is predictable: fit the within-date rank of y.",
                     {}),
    # terms
    "linear": Template("term", "y moves with a feature, in proportion.", {"feature": "feature"}),
    "lag": Template("term", "y moves with a feature's value k dates earlier.",
                    {"feature": "feature", "k": "int 1..20"}, {"k": 1}, lambda p: p.get("k", 1) + 2),
    "saturating": Template("term", "y moves with a feature, but the effect levels off beyond a scale.",
                           {"feature": "feature", "scale": "float 0.1..5"}, {"scale": 1.0}),
    "hinge": Template("term", "a feature matters only beyond a threshold (on one side).",
                      {"feature": "feature", "cut": "float -3..3", "side": "choice above|below"},
                      {"cut": 0.0, "side": "above"}),
    "bump": Template("term", "the effect of a feature peaks (or dips) around a value and fades away from it.",
                     {"feature": "feature", "center": "float -3..3", "width": "float 0.1..3"},
                     {"center": 0.0, "width": 1.0}),
    "periodic": Template("term", "the effect of a feature oscillates with its value.",
                         {"feature": "feature", "freq": "float 0.1..10"}, {"freq": 1.0}),
    "interaction": Template("term", "the effect of one feature depends on the level of another.",
                            {"feature": "feature", "other": "feature"}),
    "group": Template("term", "what matters is a feature relative to its group's mean on the date.",
                      {"feature": "feature", "group": "feature"}),
    "market": Template("term", "a date-wide quantity moves every asset: the date mean of a feature, or of past "
                       "y (frame), k dates earlier.",
                       {"series": "series", "k": "int 0..20"}, {"k": 0}, lambda p: p.get("k", 0) + 2),
    "lead": Template("term", "some assets lead others: a series of a source k dates earlier moves every asset "
                     "in a scope. Source: one asset (asset:<id>), the mean of the scope without the asset itself "
                     "(mean), or the asset with the largest value of a feature (top:<feature>); scope: all assets "
                     "(all) or each value of a group feature, the source taken within the group.",
                     {"series": "series", "k": "int 1..20", "source": "source", "scope": "scope"},
                     {"k": 1, "source": "mean", "scope": "all"}, lambda p: p.get("k", 1) + 2),
    "own_lag": Template("term", "y follows its own past: frame k dates earlier (momentum or reversal).",
                        {"k": "int 1..20"}, {"k": 1}, lambda p: p.get("k", 1) + 2),
    "ou": Template("term", "a level reverts to its mean: the OU s-score of the cumulative frame (or of a "
                   "feature's level) over a trailing window.",
                   {"series": "series", "window": "int 10..250"}, {"series": "frame", "window": 60},
                   lambda p: p.get("window", 60) + 2),
    "gate": Template("term", "a feature's effect differs by regime: the feature's effect, separately where "
                     "another feature (or the date's spread of past frame) is above and below a cut.",
                     {"feature": "feature", "by": "series", "cut": "float -3..3"}, {"cut": 0.0}),
}


def check(inst: dict, features: list[str]) -> dict:
    """The instance with defaults filled in; raises ValueError naming what is wrong."""
    name = inst.get("template")
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; templates: {', '.join(TEMPLATES)}")
    t = TEMPLATES[name]
    p = {**t.defaults, **(inst.get("params") or {})}
    for k, typ in t.params.items():
        if k not in p:
            raise ValueError(f"{name}: missing parameter {k} ({typ})")
        v = p[k]
        if typ == "feature" and v not in features and not (v == "" and k in t.defaults):
            raise ValueError(f"{name}.{k}: {v!r} is not a feature")
        if typ == "series" and v != "frame" and v not in features:
            raise ValueError(f"{name}.{k}: {v!r} is neither 'frame' nor a feature")
        if typ == "source" and not (v == "mean" or str(v).startswith("asset:")
                                    or str(v).startswith("top:") and str(v)[4:] in features):
            raise ValueError(f"{name}.{k}: {v!r} is not mean, asset:<id> or top:<feature>")
        if typ == "scope" and v != "all" and v not in features:
            raise ValueError(f"{name}.{k}: {v!r} is neither 'all' nor a feature")
        if typ.startswith(("int", "float")):
            lo, hi = map(float, typ.split()[1].split(".."))
            if not lo <= float(v) <= hi:
                raise ValueError(f"{name}.{k}: {v} not in {lo}..{hi}")
        if typ.startswith("choice") and v not in typ.split()[1].split("|"):
            raise ValueError(f"{name}.{k}: {v!r} not one of {typ.split()[1]}")
    unknown = set(p) - set(t.params)
    if unknown:
        raise ValueError(f"{name}: unknown parameters {sorted(unknown)}")
    if name == "factor" and p["kind"] == "group" and not p["group"]:
        raise ValueError("factor: kind group needs group")
    if name == "vol" and p["by"] == "feature" and not p["feature"]:
        raise ValueError("vol: by feature needs feature")
    return {"template": name, "params": p}


def label(inst) -> str:
    if not isinstance(inst, dict):
        return getattr(inst, "__name__", type(inst).__name__)
    return f"{inst['template']}(" + ", ".join(f"{k}={v}" for k, v in inst["params"].items() if v != "") + ")"


# ---- target transforms: fit state on training rows, apply to rows with y, map a forecast back ----------

@dataclass
class Target:
    """A target transform written as code: ``forward(d, y)`` is the new target made from column ``y`` (a
    polars expression or an array over the rows of ``d``); ``scale(d)``, if given, maps a forecast of it back
    (y = forecast * scale). Without it the theory's final calibration sets the scale."""
    forward: Callable
    scale: Callable | None = None


def column(d: pl.DataFrame, v, names: list[str]) -> pl.DataFrame:
    """``d`` with ``v`` (an expression, a series, or an array with one column per name) as columns ``names``."""
    if isinstance(v, pl.Expr):
        return d.with_columns(v.alias(names[0]))
    a = np.asarray(v.to_numpy() if isinstance(v, pl.Series) else v, dtype=float).reshape(d.height, -1)
    return d.with_columns(pl.Series(n, a[:, i]) for i, n in enumerate(names))


def width(d: pl.DataFrame, v) -> int:
    return 1 if isinstance(v, (pl.Expr, pl.Series)) else int(np.asarray(v).size // max(d.height, 1))


def target_fit(inst, d: pl.DataFrame, y: str) -> dict:
    if isinstance(inst, Target):
        return {}
    p = inst["params"]
    if inst["template"] == "factor" and p["kind"] == "pca":
        wide = d.pivot(on="asset", index="date", values=y).sort("date")
        m = np.nan_to_num(wide.drop("date").to_numpy())
        return {"assets": wide.columns[1:], "v": np.linalg.svd(m, full_matrices=False)[2][:p["k"]].tolist()}
    if inst["template"] == "vol" and p["by"] == "feature":
        g = d.select(z(p["feature"]).alias("f"), pl.col(y).abs().log().alias("l")).filter(
            pl.col("l").is_finite())
        b, a = np.polyfit(g["f"].to_numpy(), g["l"].to_numpy(), 1)
        return {"a": float(a), "b": float(b)}
    return {}


def scale_col(inst: dict, state: dict, y: str) -> pl.Expr:
    """The vol transform's forecast of the size of y, from past values or a feature."""
    p = inst["params"]
    if p["by"] == "feature":
        return (state["a"] + state["b"] * z(p["feature"])).exp()
    s = pl.col(y).abs().shift(1).rolling_mean(p["window"], min_samples=5).over("asset")
    return s.fill_null(pl.col(y).abs().mean()).clip(lower_bound=1e-8)


def target_apply(inst: dict, state: dict, d: pl.DataFrame, y: str, out: str) -> pl.DataFrame:
    """``d`` with column ``out`` = the transform of column ``y`` (null where y is null), and ``_scale_<out>``
    where the transform needs it to map back."""
    if isinstance(inst, Target):
        d = column(d, inst.forward(d, y), [out])
        d = d.with_columns(pl.when(pl.col(y).is_null()).then(None).otherwise(pl.col(out)).alias(out))
        return column(d, inst.scale(d), [f"_scale_{out}"]) if inst.scale else d
    p, t = inst["params"], inst["template"]
    if t == "factor":
        if p["kind"] == "market":
            return d.with_columns((pl.col(y) - pl.col(y).mean().over("date")).alias(out))
        if p["kind"] == "group":
            return d.with_columns((pl.col(y) - pl.col(y).mean().over("date", p["group"])).alias(out))
        v = np.array(state["v"])
        idx = {a: i for i, a in enumerate(state["assets"])}
        parts = []
        for g in d.partition_by("date"):
            vec = np.zeros(len(idx))
            yy, cols = g[y].to_numpy(), [idx.get(str(a), -1) for a in g["asset"].cast(pl.String)]
            ok = np.array([c >= 0 for c in cols]) & np.isfinite(yy.astype(float))
            vec[[c for c, o in zip(cols, ok) if o]] = yy[ok]
            fit = vec @ v.T @ v
            parts.append(g.select("date", "asset").with_columns(
                pl.Series(out, yy - np.array([fit[c] if c >= 0 else 0.0 for c in cols]))))
        return d.join(pl.concat(parts), on=["date", "asset"], how="left")
    if t == "vol":
        d = d.with_columns(scale_col(inst, state, y).alias(f"_scale_{out}"))
        return d.with_columns((pl.col(y) / pl.col(f"_scale_{out}")).alias(out))
    if t == "rank":
        return d.with_columns(((pl.col(y).rank().over("date") - 0.5) / pl.col(y).count().over("date") - 0.5)
                              .alias(out))
    raise ValueError(t)


def target_back(inst, d: pl.DataFrame, f: np.ndarray, out: str) -> np.ndarray:
    """A forecast of the transformed target, mapped to the scale it was transformed from. Factor and rank
    leave the scale to the theory's final calibration; vol multiplies by the forecast size."""
    scaled = inst.scale is not None if isinstance(inst, Target) else inst["template"] == "vol"
    return f * d[f"_scale_{out}"].to_numpy() if scaled else f


# ---- terms -------------------------------------------------------------------------------------------

def s_score(d: pl.DataFrame, col: str, window: int, name: str) -> pl.DataFrame:
    """Per asset and date, the OU s-score of the cumulative sum of ``col`` over the ``window`` dates before
    (a level such as a feature is used as is): X' = a + b X + e on that window, score (X_last - a/(1-b)) /
    sqrt(var(e)/(1-b^2)); 0 where b is not in (0, 1) or there is too little history."""
    parts = []
    for g in d.select("date", "asset", col).partition_by("asset"):
        r = g[col].to_numpy().astype(float)
        out = np.zeros(g.height)
        for t in range(window, g.height):
            w = np.nan_to_num(r[t - window:t])
            x = np.cumsum(w) if col.startswith("_frame") else w
            x0, x1 = x[:-1], x[1:]
            vx = x0.var()
            if vx <= 0:
                continue
            b = ((x0 - x0.mean()) * (x1 - x1.mean())).mean() / vx
            if not 0 < b < 1:
                continue
            a = x1.mean() - b * x0.mean()
            e = x1 - a - b * x0
            sd = np.sqrt(e.var() / (1 - b * b))
            out[t] = (x[-1] - a / (1 - b)) / sd if sd > 0 else 0.0
        parts.append(g.select("date", "asset").with_columns(pl.Series(name, out)))
    return d.join(pl.concat(parts), on=["date", "asset"], how="left")


def series(name: str) -> str:
    return "_frame" if name == "frame" else name


def term(inst: dict, d: pl.DataFrame, name: str) -> tuple[pl.DataFrame, list[str]]:
    """``d`` (sorted by asset, date) with the instance's columns added; their names. ``_frame`` holds the
    transformed target, null on dates being forecast; a function piece gets it as ``frame``."""
    if callable(inst):
        v = inst(d)
        names = [f"{name}_{i}" for i in range(width(d, v))]
        d = column(d, v, names)
        return d.with_columns(pl.col(n).fill_nan(0.0).fill_null(0.0) for n in names), names
    p, t = inst["params"], inst["template"]
    f = p.get("feature")
    if t == "linear":
        e = [z(f)]
    elif t == "lag":   # polars cannot nest the date window inside the asset window: z-score first
        d = d.with_columns(z(f).alias(f"{name}_z"))
        e = [past(pl.col(f"{name}_z"), p["k"])]
    elif t == "saturating":
        e = [(z(f) / p["scale"]).tanh()]
    elif t == "hinge":
        e = [(z(f) - p["cut"]).clip(lower_bound=0) if p["side"] == "above" else (p["cut"] - z(f)).clip(lower_bound=0)]
    elif t == "bump":
        e = [(-((z(f) - p["center"]) / p["width"]) ** 2).exp()]
    elif t == "periodic":
        e = [(z(f) * p["freq"]).sin(), (z(f) * p["freq"]).cos()]
    elif t == "interaction":
        e = [z(f) * z(p["other"])]
    elif t == "group":
        e = [(pl.col(f) - pl.col(f).mean().over("date", p["group"])) / pl.col(f).std().over("date")]
    elif t == "market":
        s = series(p["series"])
        k = p["k"] if s != "_frame" else max(p["k"], 1)
        d = d.with_columns(pl.col(s).mean().over("date").alias(f"{name}_mean"))
        e = [past(pl.col(f"{name}_mean"), k) if k else pl.col(f"{name}_mean")]
    elif t == "lead":   # the source's value on each date (and scope), then that k dates earlier per asset
        s, src, by = series(p["series"]), p["source"], ["date"] if p["scope"] == "all" else ["date", p["scope"]]
        if src == "mean":
            has = pl.col(s).is_not_null().cast(pl.Float64)
            v = (pl.col(s).sum().over(by) - pl.col(s).fill_null(0.0)) / (pl.col(s).count().over(by) - has).clip(1)
        elif src.startswith("asset:"):
            v = pl.col(s).filter(pl.col("asset").cast(pl.String) == src[6:]).first().over(by)
        else:
            v = pl.col(s).sort_by(pl.col(src[4:]), descending=True, nulls_last=True).first().over(by)
        d = d.with_columns(v.alias(f"{name}_src"))
        e = [past(pl.col(f"{name}_src"), p["k"])]
    elif t == "own_lag":
        e = [past(pl.col("_frame"), p["k"])]
    elif t == "ou":
        s = series(p["series"])
        d = s_score(d, s, p["window"], name)
        return d.with_columns(pl.col(name).fill_null(0.0)), [name]
    elif t == "gate":
        by = series(p["by"])
        if by == "_frame":   # the date's spread of past frame, standardised over the rows given
            d = d.with_columns(past(pl.col("_frame")).alias(f"{name}_past"))
            d = d.with_columns(pl.col(f"{name}_past").std().over("date").alias(f"{name}_spread"))
            g = (pl.col(f"{name}_spread") - pl.col(f"{name}_spread").mean()) / pl.col(f"{name}_spread").std()
        else:
            g = z(by)
        e = [z(f) * (g > p["cut"]).cast(pl.Float64), z(f) * (g <= p["cut"]).cast(pl.Float64)]
    else:
        raise ValueError(t)
    names = [f"{name}_{i}" for i in range(len(e))]
    d = d.with_columns(x.alias(n) for x, n in zip(e, names))
    return d.with_columns(pl.col(n).fill_nan(0.0).fill_null(0.0) for n in names), names


# ---- the theory ---------------------------------------------------------------------------------------

class Theory:
    """Target transforms applied in order, then a ridge of the transformed target on the term columns,
    mapped back through the transforms and calibrated to y by least squares on the training rows."""

    def __init__(self, spec: dict, features: list[str], ridge: float = 1.0, lookback: int = 2):
        self.targets = [i if isinstance(i, Target) else check(i, features) for i in spec.get("targets", [])]
        self.terms = [i if callable(i) else check(i, features) for i in spec.get("terms", [])]
        self.ridge = ridge
        self.lookback = max([TEMPLATES[i["template"]].lookback(i["params"]) for i in self.targets + self.terms
                             if isinstance(i, dict)] + [lookback, 2])

    def frame(self, d: pl.DataFrame) -> pl.DataFrame:
        """``d`` with ``_t0`` = y, ``_t1``.. after each transform, ``_frame`` the last."""
        d = d.with_columns(pl.col("y").alias("_t0"))
        for i, (inst, st) in enumerate(zip(self.targets, self.states)):
            d = target_apply(inst, st, d, f"_t{i}", f"_t{i + 1}")
        return d.with_columns(pl.col(f"_t{len(self.targets)}").alias("_frame"))

    def design(self, d: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        cols = []
        d = d.with_columns(pl.col("_frame").alias("frame"))
        for j, inst in enumerate(self.terms):
            d, names = term(inst, d, f"_m{j}")
            cols += names
        return d, cols

    def raw(self, d: pl.DataFrame, cols: list[str]) -> np.ndarray:
        x = (d.select(cols).to_numpy() - self.mu) / self.sd if cols else np.zeros((d.height, 0))
        f = x @ self.beta + self.b0
        for i in reversed(range(len(self.targets))):
            f = target_back(self.targets[i], d, f, f"_t{i + 1}")
        return f

    def fit(self, train: pl.DataFrame) -> "Theory":
        d = train.sort("asset", "date")
        self.states = []
        cur = d.with_columns(pl.col("y").alias("_t0"))
        for i, inst in enumerate(self.targets):   # each transform is fitted on the previous one's output
            self.states.append(target_fit(inst, cur, f"_t{i}"))
            cur = target_apply(inst, self.states[-1], cur, f"_t{i}", f"_t{i + 1}")
        d, cols = self.design(self.frame(d))
        ok = d["_frame"].is_not_null() & d["_frame"].is_finite()
        x, t = d.filter(ok).select(cols).to_numpy(), d.filter(ok)["_frame"].to_numpy()
        self.mu, self.sd = (x.mean(0), x.std(0) + 1e-12) if cols else (0.0, 1.0)
        xs = (x - self.mu) / self.sd if cols else np.zeros((len(t), 0))
        self.b0 = float(t.mean())
        self.beta = np.linalg.solve(xs.T @ xs + self.ridge * np.eye(xs.shape[1]), xs.T @ (t - self.b0)) if cols \
            else np.zeros(0)
        f = self.raw(d.filter(ok), cols)
        yy = d.filter(ok)["y"].to_numpy()
        v = f.var()
        self.c = float(np.cov(f, yy)[0, 1] / v) if v > 0 else 0.0
        self.a = float(yy.mean() - self.c * f.mean())
        self.cols = cols
        return self

    def predict(self, history: pl.DataFrame, today: pl.DataFrame) -> np.ndarray:
        dates = history["date"].unique().sort()
        tail = history.filter(pl.col("date") >= dates[max(0, len(dates) - self.lookback)])
        d = pl.concat([tail, today.with_columns(pl.lit(None, dtype=pl.Float64).alias("y"))], how="diagonal_relaxed")
        d = d.with_row_index("_i").sort("asset", "date")
        d, cols = self.design(self.frame(d))
        last = d.filter(pl.col("date") == today["date"][0]).sort("_i")
        out = self.a + self.c * self.raw(last, cols)
        # rows of ``today`` in its own order
        order = today.with_row_index("_r").join(last.select("asset", pl.int_range(pl.len()).alias("_k")),
                                                on="asset", how="left")["_k"].to_numpy()
        return np.nan_to_num(out[order])

    def columns(self, d: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
        """Training-style rows with each term's columns, keyed by the term's label (for analysis)."""
        d = self.frame(d.sort("asset", "date"))
        by = {}
        for j, inst in enumerate(self.terms):
            d, names = term(inst, d, f"_m{j}")
            by[label(inst)] = names
        return d, by


# ---- a model as a sum of named pieces ------------------------------------------------------------------

def template(name: str, **params) -> dict:
    """A template instance, for use as a piece: template("lag", feature="x01", k=2)."""
    return {"template": name, "params": params}


def help() -> str:
    return "\n".join(f"{n} ({t.kind}): {t.says} Parameters {t.params}, defaults {t.defaults}"
                     for n, t in TEMPLATES.items())


class Pieces:
    """A model as a sum of named pieces: ``{name: (claim, piece)}``, where a piece is a template instance, a
    ``Target``, or a function of the rows ``d`` (sorted by asset, date; y and frame null on the dates being
    forecast) returning one column or several. Targets apply in order, then a ridge of the target on every
    piece's columns, calibrated to y. ``lookback``: dates of history a forecast needs."""

    def __init__(self, pieces: dict, lookback: int = 120, ridge: float = 1.0):
        self.pieces, self.lookback, self.ridge = dict(pieces), lookback, ridge

    @staticmethod
    def is_target(piece) -> bool:
        return isinstance(piece, Target) or isinstance(piece, dict) and \
            TEMPLATES.get(piece.get("template"), Template("term", "", {})).kind == "target"

    def names(self) -> list[list[str]]:
        return [[n, c] for n, (c, _) in self.pieces.items()]

    def spec(self) -> dict:
        ps = [p for _, p in self.pieces.values()]
        return {"targets": [p for p in ps if self.is_target(p)], "terms": [p for p in ps if not self.is_target(p)]}

    def without(self, k: int) -> "Pieces":
        return Pieces({n: v for j, (n, v) in enumerate(self.pieces.items()) if j != k}, self.lookback, self.ridge)

    def fit(self, train: pl.DataFrame) -> Theory:
        feats = [c for c in train.columns if c not in ("date", "asset", "y")]
        return Theory(self.spec(), feats, self.ridge, self.lookback).fit(train)

    @staticmethod
    def predict(state: Theory, history: pl.DataFrame, today: pl.DataFrame) -> np.ndarray:
        return state.predict(history, today)
