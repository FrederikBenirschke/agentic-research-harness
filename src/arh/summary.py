"""Deterministic summary of the training panel. It is note #1 of every
research run, is appended to every agent's prompt, and is the only view of
the data the ideator gets."""

from __future__ import annotations

import os

import polars as pl


def summarize(train: pl.DataFrame) -> str:
    feats = [c for c in train.columns if c not in ("date", "asset", "y")]
    head = (f"Panel: {train.height} rows, {train['date'].n_unique()} dates, {train['asset'].n_unique()} assets, "
            f"{len(feats)} features. y = the asset's return over the next period.")
    if os.environ.get("ARH_SUMMARY") == "none":   # eval arm: omit the per-feature statistics
        return head
    lines = [head,
             _y_line(train),
             "Per feature: null share, mean, std, pooled corr(x, y), mean within-date rank corr(x, y), "
             "mean per-asset lag-1 autocorr of x, n distinct values:"]
    ranked = train.with_columns(pl.col([*feats, "y"]).rank().over("date"))
    for f in feats:
        col = train[f]
        pooled = train.select(pl.corr(f, "y")).item()
        constant = train.group_by("date").agg(pl.col(f).n_unique()).select(pl.col(f).max()).item() == 1
        rank_ic = None if constant else ranked.group_by("date").agg(pl.corr(f, "y")).select(pl.col(f).mean()).item()
        ac = train.sort("date").group_by("asset").agg(pl.corr(pl.col(f), pl.col(f).shift(1))).select(
            pl.col(f).mean()).item()
        lines.append(f"  {f}: null {col.null_count() / len(col):.2f}, mean {col.mean():.3g}, std {col.std():.3g}, "
                     f"corr {pooled:+.3f}, " + ("date-constant (same for all assets)" if constant
                                                 else f"rank_ic {rank_ic:+.3f}") +
                     f", ac1 {ac:+.2f}, distinct {col.n_unique()}")
    if os.environ.get("ARH_SUMMARY") == "groups":  # eval arm: also decompose features by each group-code column
        lines += _group_lines(train, feats)
    return "\n".join(lines)


def _group_lines(train: pl.DataFrame, feats: list[str]) -> list[str]:
    """For each column that is constant per asset with few values (a group code), each feature's share of
    within-date variance in its date x group mean, and the rank IC of the deviation from that mean."""
    codes = [g for g in feats if train[g].n_unique() <= 50
             and train.group_by("asset").agg(pl.col(g).n_unique()).select(pl.col(g).max()).item() == 1]
    out = []
    for g in codes:
        xs = [f for f in feats if f not in codes]
        d = train.with_columns(*[(pl.col(f) - pl.col(f).mean().over("date")).alias(f) for f in xs])
        d = d.with_columns(*[(pl.col(f) - pl.col(f).mean().over(["date", g])).alias(f"{f}_w") for f in xs])
        r = d.with_columns(pl.col([f"{f}_w" for f in xs] + ["y"]).rank().over("date"))
        parts = []
        for f in xs:
            v = d[f].var()
            share = 1 - d[f"{f}_w"].var() / v if v else 0.0
            ic = r.group_by("date").agg(pl.corr(f"{f}_w", "y")).select(pl.col(f"{f}_w").mean()).item()
            parts.append(f"{f} {share:.2f}/{ic:+.3f}")
        out.append(f"{g} looks like a group code ({train[g].n_unique()} values, constant per asset). Per feature: "
                   f"share of within-date variance in the date x {g} mean / mean within-date rank corr(y, x minus "
                   f"its date x {g} mean): " + ", ".join(parts))
    return out


def _y_line(train: pl.DataFrame) -> str:
    y = train["y"]
    ac = train.sort("date").group_by("asset").agg(pl.corr("y", pl.col("y").shift(1))).select(pl.col("y").mean()).item()
    share = train.group_by("date").agg(pl.col("y").mean()).select(pl.col("y").var()).item() / y.var()
    return (f"y: mean {y.mean():.3g}, std {y.std():.3g}, mean per-asset lag-1 autocorr {ac:+.3f}, "
            f"share of variance in the cross-date mean {share:.2f}")
