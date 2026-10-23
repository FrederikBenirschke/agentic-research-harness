"""Score a model file: walk-forward CV on the agent's training data (shown to
agents) and the sealed holdout (recorded, never shown).

The model runs in arh.worker under a macOS sandbox that denies reads of the
private roots (default: the home dir, where every world's sealed data lives) except
the model's own directory and the Python runtime. It is fed one date at a
time and sees a date's targets only after it has forecast that date.
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import polars as pl

WORKER = Path(__file__).with_name("worker.py")
CV_FOLDS = 5
CV_SHARE = 0.5          # the last half of the training dates is tested, in CV_FOLDS blocks
ONE_THREAD = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}  # evals run many fits in parallel


class ScoreError(Exception):
    pass


def private_roots() -> list[Path]:
    env = os.environ.get("ARH_PRIVATE")
    return [Path(p) for p in env.split(":")] if env else [Path.home()]


def runtime_paths() -> list[Path]:
    paths = {Path(sys.prefix), Path(sys.base_prefix), *(Path(p) for p in sys.path if "site-packages" in p)}
    return sorted(p.resolve() for p in paths)


def profile(allow: list[Path]) -> str:
    """Seatbelt: allow by default, deny network and the private roots, re-allow
    ``allow`` (later rules win)."""
    q = lambda p: '"' + str(Path(p).resolve()).replace('"', '\\"') + '"'
    deny = " ".join(f"(subpath {q(p)})" for p in private_roots())
    again = " ".join(f"(subpath {q(p)})" for p in allow)
    return (f"(version 1)(allow default)(deny network*)"
            f"(deny file-read* file-write* {deny})(allow file-read* {again})")


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def _ipc(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_ipc(buf)
    return buf.getvalue()


def forecast(model: Path, train: pl.DataFrame, test: pl.DataFrame, timeout_s: float = 300) -> np.ndarray:
    """Fit on ``train``, forecast ``test`` date by date; yhat in ``test``'s row order."""
    if sys.platform != "darwin":
        raise ScoreError("model sandbox needs macOS sandbox-exec")
    model = model.resolve()
    allow = [model.parent, WORKER, *runtime_paths()]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(["sandbox-exec", "-p", profile(allow), sys.executable, "-I", str(WORKER), str(model)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, cwd=model.parent,
                                env={**os.environ, **ONE_THREAD})
        timer = threading.Timer(timeout_s, proc.kill)
        timer.start()
        try:
            return _drive(proc, train, test)
        except (BrokenPipeError, struct.error) as e:
            err.seek(0)
            raise ScoreError(f"worker died ({'timeout' if not timer.is_alive() else e}): "
                             f"{err.read()[-2000:].decode(errors='replace')}") from None
        finally:
            timer.cancel()
            proc.kill()
            proc.wait()


def _drive(proc: subprocess.Popen, train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    def send(kind: bytes, df: pl.DataFrame) -> None:
        payload = _ipc(df)
        proc.stdin.write(kind + struct.pack("<Q", len(payload)) + payload)
        proc.stdin.flush()

    def recv() -> bytes:
        head = proc.stdout.read(9)
        kind, n = head[:1], struct.unpack("<Q", head[1:])[0]
        payload = proc.stdout.read(n)
        if kind == b"E":
            raise ScoreError(payload.decode(errors="replace"))
        return payload

    send(b"F", train)
    recv()
    test = test.with_row_index("_row")
    out = np.empty(test.height)
    for (_,), today in test.group_by("date", maintain_order=True):
        rows = today["_row"].to_numpy()
        today = today.drop("_row")
        send(b"P", today.drop("y"))
        out[rows] = pl.read_ipc(io.BytesIO(recv()))["yhat"].to_numpy()
        send(b"R", today)
    return out


def rank_metrics(dates: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> dict:
    """Mean per-date Spearman IC, and the per-date Sharpe of the top-minus-bottom quintile of the forecast
    (equal weights, not annualised). A date with a constant forecast counts as IC 0 and spread 0."""
    df = pl.DataFrame({"date": dates, "y": y, "f": yhat}).with_columns(
        q=(pl.col("f").rank("ordinal").over("date") - 1) / pl.len().over("date"))
    per = df.group_by("date").agg(
        ic=pl.corr("f", "y", method="spearman"),
        ls=pl.col("y").filter(pl.col("q") >= 0.8).mean() - pl.col("y").filter(pl.col("q") < 0.2).mean(),
        flat=pl.col("f").n_unique() == 1)
    per = per.with_columns(pl.when(pl.col("flat")).then(0.0).otherwise(pl.col(c)).fill_nan(0.0).fill_null(0.0).alias(c)
                           for c in ("ic", "ls"))
    sd = per["ls"].std()
    return {"ic": float(per["ic"].mean()), "ls_sharpe": float(per["ls"].mean() / sd) if sd else 0.0}


def cv_full(model: Path, train: pl.DataFrame, timeout_s: float = 600) -> tuple[list[float], dict]:
    """Expanding-window walk-forward R^2 per fold, and rank metrics over all CV test dates."""
    train = train.sort("date", "asset")
    dates = train["date"].unique().sort()
    start = int(len(dates) * (1 - CV_SHARE))
    blocks = np.array_split(dates[start:].to_numpy(), CV_FOLDS)
    scores, parts = [], []
    for block in blocks:
        fit = train.filter(pl.col("date") < block[0])
        test = train.filter(pl.col("date").is_in(block))
        yhat = forecast(model, fit, test, timeout_s / CV_FOLDS)
        scores.append(r2(test["y"].to_numpy(), yhat))
        parts.append((test["date"].to_numpy(), test["y"].to_numpy(), yhat))
    return scores, rank_metrics(*(np.concatenate(p) for p in zip(*parts)))


def cv(model: Path, train: pl.DataFrame, timeout_s: float = 600) -> list[float]:
    return cv_full(model, train, timeout_s)[0]


def holdout_full(model: Path, world: Path, timeout_s: float = 600) -> dict:
    train = pl.read_parquet(world / "public" / "train.parquet").sort("date", "asset")
    hold = pl.read_parquet(world / "sealed" / "holdout.parquet").sort("date", "asset")
    y, yhat = hold["y"].to_numpy(), forecast(model, train, hold, timeout_s)
    return {"r2": r2(y, yhat), **rank_metrics(hold["date"].to_numpy(), y, yhat)}


def holdout(model: Path, world: Path, timeout_s: float = 600) -> float:
    return holdout_full(model, world, timeout_s)["r2"]
