from pathlib import Path

import numpy as np
import polars as pl
import pytest

from arh import score
from arh.store import Store, render
from arh.summary import summarize


def panel(n_dates=120, n_assets=10, seed=0) -> pl.DataFrame:
    """y = 0.5 * x00 + noise, so the oracle R2 is about 0.2."""
    rng = np.random.default_rng(seed)
    d, a = np.meshgrid(np.arange(n_dates), np.arange(n_assets), indexing="ij")
    x = rng.normal(size=d.size)
    return pl.DataFrame({"date": d.ravel(), "asset": a.ravel(), "x00": x,
                         "y": 0.5 * x + rng.normal(size=d.size)})


def write_model(tmp: Path, body: str) -> Path:
    (tmp / "m").mkdir(exist_ok=True)
    p = tmp / "m" / "model.py"
    p.write_text("import numpy as np, polars as pl\n" + body)
    return p


LINEAR = """
def fit(train):
    x, y = train["x00"].to_numpy(), train["y"].to_numpy()
    return (x @ y) / (x @ x)
def predict(b, history, today):
    return b * today["x00"].to_numpy()
"""


def test_store_search_open_and_render(tmp_path):
    s = Store(tmp_path / "s.db")
    q = s.add("question", "ideator", "is x03 autocorrelated within asset?")
    i = s.add("idea", "ideator", "momentum in x03 predicts y")
    assert {n["id"] for n in s.search(open_only=True)} == {q, i}
    s.add("finding", "analyst", "x03 lag-1 autocorr 0.9", refs=[q])
    assert [n["id"] for n in s.search(open_only=True)] == [i]
    assert s.search("autocorr")[0]["kind"] in ("finding", "question")
    f = s.add("forecast", "forecaster", "ridge on x03", refs=[i], cv=0.01)
    s.set_holdout(f, 0.123)
    assert "0.123" not in render(s.get(f), full=True)
    assert s.get(i)["cited_by"] == [f]
    with pytest.raises(ValueError):
        s.add("idea", "ideator", "x", refs=[999])


def test_cv_recovers_planted_signal(tmp_path):
    folds = score.cv(write_model(tmp_path, LINEAR), panel())
    assert len(folds) == 5 and 0.1 < np.mean(folds) < 0.3


def test_model_never_sees_today_y_or_future(tmp_path):
    model = write_model(tmp_path, """
import gc
def fit(train):
    return train["date"].max()
def predict(last, history, today):
    d = today["date"][0]
    assert "y" not in today.columns
    assert history["date"].max() == d - 1, (history["date"].max(), d)
    frames = [o for o in gc.get_objects() if isinstance(o, pl.DataFrame)]
    assert all(f.filter(pl.col("date") > d).height == 0 for f in frames if "date" in f.columns)
    return np.zeros(today.height)
""")
    assert len(score.cv(model, panel())) == 5


def test_sandbox_denies_private_reads(tmp_path, monkeypatch):
    secret = tmp_path / "private" / "holdout.txt"
    secret.parent.mkdir()
    secret.write_text("1.0")
    public = tmp_path / "public.txt"
    public.write_text("1.0")
    monkeypatch.setenv("ARH_PRIVATE", str(secret.parent))
    reader = "def fit(train):\n    return float(open({!r}).read())\n" \
             "def predict(s, history, today):\n    return np.full(today.height, s)\n"
    assert len(score.cv(write_model(tmp_path, reader.format(str(public))), panel())) == 5   # positive control
    with pytest.raises(score.ScoreError, match="Operation not permitted|PermissionError"):
        score.cv(write_model(tmp_path, reader.format(str(secret))), panel())


def test_model_error_comes_back(tmp_path):
    bad = write_model(tmp_path, "def fit(train):\n    return 1\ndef predict(s, h, t):\n    return [1.0]\n")
    with pytest.raises(score.ScoreError, match="finite numbers"):
        score.cv(bad, panel())


def test_summary_mentions_every_feature():
    text = summarize(panel())
    assert "x00" in text and "rank_ic" in text


def test_grounded_numbers_match_at_written_precision():
    from arh.evals import grounded, numbers
    assert numbers("x03 ic 0.330, 69% of dates, alpha 1000, R2 -0.012") == ["0.330", "69%", "-0.012"]
    assert grounded(["0.330", "69%"], "ic=0.33012 share 0.6901") == 1.0
    assert grounded(["0.412"], "ic 0.298") == 0.0
    assert grounded([], "anything") is None


def test_accuracy_names_mechanisms_and_decoy():
    from arh.evals import accuracy
    truth = {"mechanisms": [{"id": "a", "features": ["x07", "x11"]}, {"id": "m", "features": ["y"]}],
             "decoys": [{"feature": "x05"}], "noise_features": ["x00"], "oracle_holdout_r2": 0.2}
    out = accuracy({"kind": "idea", "text": "x07 works only when x11 > 0; x05 too"}, truth)
    assert out["mechanisms_hit"] == ["a"] and out["decoy_mentioned"] and out["noise_named"] == 0
    assert accuracy({"kind": "forecast", "text": "ridge", "holdout": 0.1}, truth,
                    code="lag = history['y'].shift(1)")["mechanisms_hit"] == ["m"]
    dropped = 'FEATS = ["x03"]  # x05 dropped: proxy\ndef fit(t):\n    """x05 is unused"""\n    return t[FEATS]'
    assert not accuracy({"kind": "forecast", "text": ""}, truth, code=dropped)["decoy_in_model"]
    assert accuracy({"kind": "forecast", "text": ""}, truth, code='FEATS = ["x03", "x05"]')["decoy_in_model"]


def test_mechanism_form_and_exploration():
    from arh.evals import accuracy, exploration
    truth = {"mechanisms": [{"id": "u", "features": ["x04"], "pattern": r"squar|u-shape"}],
             "decoys": [], "noise_features": []}
    assert accuracy({"text": "x04 predicts y", "kind": "finding"}, truth)["n_mechanisms_hit"] == 0
    assert accuracy({"text": "x04 squared predicts y", "kind": "finding"}, truth)["n_mechanisms_hit"] == 1
    e = exploration('pl.col("x04")**2; df.shift(2); np.corrcoef(Y[:-1].T, Y[1:].T)')
    assert {"nonlinear", "lags", "multi_lag", "cross_asset"} <= set(e["checks"]) and e["n_features_examined"] == 1
