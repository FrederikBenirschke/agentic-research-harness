"""A deterministic summary of the training panel: what the ideator sees
instead of raw data."""

from __future__ import annotations

import os

import polars as pl


def summarize(train: pl.DataFrame) -> str:
    feats = [c for c in train.columns if c not in ("date", "asset", "y")]
    head = (f"Panel: {train.height} rows, {train['date'].n_unique()} dates, {train['asset'].n_unique()} assets, "
            f"{len(feats)} features. y = the asset's return over the next period.")
    if os.environ.get("ARH_SUMMARY") == "none":   # eval ablation: no per-feature screen to anchor on
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
    return "\n".join(lines)


def _y_line(train: pl.DataFrame) -> str:
    y = train["y"]
    ac = train.sort("date").group_by("asset").agg(pl.corr("y", pl.col("y").shift(1))).select(pl.col("y").mean()).item()
    share = train.group_by("date").agg(pl.col("y").mean()).select(pl.col("y").var()).item() / y.var()
    return (f"y: mean {y.mean():.3g}, std {y.std():.3g}, mean per-asset lag-1 autocorr {ac:+.3f}, "
            f"share of variance in the cross-date mean {share:.2f}")
