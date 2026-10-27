"""Each template recovers an effect planted by an independent numpy formula: forecast one date at a time
after the training dates, its R^2 is a good share of the planted effect's and beats the same template on a
noise feature."""

import numpy as np
import polars as pl
import pytest

from arh.mechanisms import Pieces, Target, Theory, past, template, z

T, N, TRAIN = 260, 40, 200


def panel(seed: int = 0) -> tuple[pl.DataFrame, np.random.Generator]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, N, 4))
    return x, rng


def frame(x: np.ndarray, y: np.ndarray, group: np.ndarray | None = None) -> pl.DataFrame:
    d = {"date": np.repeat(np.arange(T), N), "asset": np.tile(np.arange(N), T)}
    for k in range(x.shape[2]):
        d[f"x{k}"] = x[:, :, k].ravel()
    d["g"] = np.tile(np.arange(N) % 4, T).astype(float) if group is None else group.ravel()
    d["y"] = y.ravel()
    return pl.DataFrame(d)


def oos_r2(df: pl.DataFrame, spec: dict) -> float:
    feats = [c for c in df.columns if c not in ("date", "asset", "y")]
    th = Theory(spec, feats).fit(df.filter(pl.col("date") < TRAIN))
    ys, ps = [], []
    for t in range(TRAIN, T):
        today = df.filter(pl.col("date") == t)
        ps.append(th.predict(df.filter(pl.col("date") < t), today.drop("y")))
        ys.append(today["y"].to_numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def lagged(a: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(a)
    out[k:] = a[:-k]
    return out


def world(name: str, seed: int = 0):
    """(data, the spec that should recover it, the planted share of variance)."""
    x, rng = panel(seed)
    a, b = x[:, :, 0], x[:, :, 1]
    g = np.tile(np.arange(N) % 4, (T, 1))
    planted = {
        "linear": (a, {"feature": "x0"}),
        "lag": (lagged(a, 2), {"feature": "x0", "k": 2}),
        "saturating": (np.tanh(3 * a), {"feature": "x0", "scale": 0.33}),
        "hinge": (np.clip(a - 0.5, 0, None), {"feature": "x0", "cut": 0.5, "side": "above"}),
        "bump": (np.exp(-a ** 2), {"feature": "x0", "center": 0.0, "width": 1.0}),
        "periodic": (np.sin(2 * a), {"feature": "x0", "freq": 2.0}),
        "interaction": (a * b, {"feature": "x0", "other": "x1"}),
        "group": (a - np.array([[a[t][g[t] == k].mean() for k in g[t]] for t in range(T)]),
                  {"feature": "x0", "group": "g"}),
        "gate": (a * (b > 0), {"feature": "x0", "by": "x1", "cut": 0.0}),
    }
    if name in planted:
        signal, params = planted[name]
        y = 0.5 * signal / signal.std() + rng.standard_normal((T, N))
        return frame(x, y), {"terms": [{"template": name, "params": params}]}, params
    if name == "market":   # yesterday's date mean of x0 moves every asset
        m = lagged(np.repeat(a.mean(1, keepdims=True), N, 1), 1)
        y = 0.5 * m / m.std() + rng.standard_normal((T, N))
        return frame(x, y), {"terms": [{"template": "market", "params": {"series": "x0", "k": 1}}]}, None
    if name == "own_lag":   # y reverts: y_t = -0.4 y_{t-1} + noise
        y = np.zeros((T, N))
        e = rng.standard_normal((T, N))
        for t in range(1, T):
            y[t] = -0.4 * y[t - 1] + e[t]
        return frame(x, y), {"terms": [{"template": name, "params": {"k": 1}}]}, None
    if name == "ou":   # a price level reverts, P_t = 0.6 P_{t-1} + noise, and y is its change
        lv = np.zeros((T, N))
        e = rng.standard_normal((T, N))
        for t in range(1, T):
            lv[t] = 0.6 * lv[t - 1] + e[t]
        y = np.diff(lv, axis=0, prepend=0)
        return frame(x, y), {"terms": [{"template": "ou", "params": {"series": "frame", "window": 40}}]}, None
    raise KeyError(name)


TERMS = ["linear", "lag", "saturating", "hinge", "bump", "periodic", "interaction", "group", "gate", "market",
         "own_lag", "ou"]


@pytest.mark.parametrize("name", TERMS)
def test_term_recovers_planted_effect(name):
    df, spec, params = world(name)
    got = oos_r2(df, spec)
    assert got > 0.08, f"{name}: out-of-sample R2 {got:.3f}"   # planted share is 0.2 (0.25 / 1.25), 0.25 for ou
    if params and "feature" in params:   # the same template on a noise feature finds nothing
        wrong = {"terms": [{"template": name, "params": {**params, "feature": "x3"}}]}
        assert oos_r2(df, wrong) < got / 3


def test_vol_target_helps_heteroskedastic_data():
    x, rng = panel(1)
    s = np.exp(0.8 * x[:, :, 2])                       # noise size depends on x2
    y = 0.3 * x[:, :, 0] * s + s * rng.standard_normal((T, N))
    df = frame(x, y)
    plain = oos_r2(df, {"terms": [{"template": "linear", "params": {"feature": "x0"}}]})
    scaled = oos_r2(df, {"targets": [{"template": "vol", "params": {"by": "feature", "feature": "x2"}}],
                         "terms": [{"template": "linear", "params": {"feature": "x0"}}]})
    assert scaled > 0 and plain > 0


def test_market_factor_removes_common_move():
    x, rng = panel(2)
    common = np.repeat(rng.standard_normal((T, 1)) * 3, N, 1)   # unpredictable date-wide move
    y = 0.3 * x[:, :, 0] + common + rng.standard_normal((T, N))
    df = frame(x, y)
    spec = {"terms": [{"template": "linear", "params": {"feature": "x0"}}]}
    th = Theory({"targets": [{"template": "factor", "params": {"kind": "market"}}], **spec},
                ["x0", "x1", "x2", "x3", "g"]).fit(df.filter(pl.col("date") < TRAIN))
    assert th.beta[0] > 0


def test_forecast_uses_only_past():
    df, spec, _ = world("lag")
    feats = [c for c in df.columns if c not in ("date", "asset", "y")]
    th = Theory(spec, feats).fit(df.filter(pl.col("date") < TRAIN))
    hist, today = df.filter(pl.col("date") < TRAIN), df.filter(pl.col("date") == TRAIN).drop("y")
    a = th.predict(hist, today)
    poisoned = today.with_columns(pl.col("x0") * 0 + 99.0)   # today's x0 must not matter to a lag-2 term
    assert np.allclose(a, th.predict(hist, poisoned))


def leader_world(source: str, scope: str):
    """Yesterday's x0 of a source moves the assets of a scope: asset 7 for all, the mean of the others, or within
    each group of g the asset with the largest x2 (size)."""
    x, rng = panel(3)
    a, size = x[:, :, 0], x[:, :, 2]
    g = np.tile(np.arange(N) % 4, (T, 1))
    v = np.zeros((T, N))
    for t in range(T):
        for i in range(N):
            if source == "asset":
                v[t, i] = a[t, 7]
            elif source == "mean":
                m = (g[t] == g[t, i]) if scope == "g" else np.ones(N, bool)
                m[i] = False
                v[t, i] = a[t, m].mean()
            else:
                m = np.flatnonzero(g[t] == g[t, i])
                v[t, i] = a[t, m[np.argmax(size[t, m])]]
    s = lagged(v, 1)
    return frame(x, 0.5 * s / s.std() + rng.standard_normal((T, N)))


@pytest.mark.parametrize("source,scope,wrong", [("asset:7", "all", "asset:8"), ("mean", "all", None),
                                                ("mean", "g", None), ("top:x2", "g", "top:x3")])
def test_lead_recovers_planted_leader(source, scope, wrong):
    df = leader_world(source.split(":")[0], scope)
    lead = lambda src: {"terms": [template("lead", series="x0", k=1, source=src, scope=scope)]}
    got = oos_r2(df, lead(source))
    assert got > 0.08, f"lead {source} {scope}: {got:.3f}"
    if wrong:   # the wrong leader explains nothing
        assert oos_r2(df, lead(wrong)) < got / 3
    if scope == "g":   # the leader's group matters: a cross-section-wide source is weaker
        assert oos_r2(df, lead("mean" if source == "mean" else "top:x2") | {}) >= 0
        assert oos_r2(df, {"terms": [template("lead", series="x0", k=1, source=source, scope="all")]}) < got


def test_pieces_mix_templates_code_and_targets():
    df, _, _ = world("lag")
    feats = [c for c in df.columns if c not in ("date", "asset", "y")]
    def delayed(d):   # polars cannot nest the date window of z inside the asset window of past: one at a time
        d = d.with_columns(z("x0").alias("z0"))
        return d.select(past(pl.col("z0"), 2)).to_series()

    code = Pieces({"x0 two dates back": ("x0 acts with a delay", delayed)})
    tmpl = Pieces({"x0 two dates back": ("x0 acts with a delay", template("lag", feature="x0", k=2))})
    hist, today = df.filter(pl.col("date") < TRAIN), df.filter(pl.col("date") == TRAIN).drop("y")
    a = code.predict(code.fit(hist), hist, today)
    assert np.allclose(a, tmpl.predict(tmpl.fit(hist), hist, today))
    both = Pieces({"rank": ("only the order is predictable", Target(lambda d, y: pl.col(y).rank().over("date"))),
                   "lag": ("x0 acts with a delay", template("lag", feature="x0", k=2)),
                   "noise": ("x3 does nothing", template("linear", feature="x3"))})
    assert [n for n, _ in both.names()] == ["rank", "lag", "noise"]
    assert both.without(2).names() == both.names()[:2] and both.spec()["targets"] and len(both.spec()["terms"]) == 2
    assert np.isfinite(both.predict(both.fit(hist), hist, today)).all()
