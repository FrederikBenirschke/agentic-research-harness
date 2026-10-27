"""Screen every feature under a package of transforms against y or a model's residuals.

    python screen.py PACKAGE [PACKAGE ...] [--data train.parquet] [--target TARGET] [--top 30]

Packages: raw, cs (within-date z-score), lag (feature at past dates, change, trailing mean), cslag
(within-date rank of the lagged feature), inter (pairwise products of within-date z-scores), shape
(magnitude, square, sign, tails), cond (the raw relation by time block, by low-cardinality feature and by
market regime), factor (principal components and stepwise peeling of the target, each followed by a study
of the residual it leaves), vol (what predicts the size |target| and the sign of the target, and whether
size clusters in time), ou (per asset AR(1) speed and half-life of the target and every feature, the OU
s-score of the cumulative target over a trailing window, as in statistical arbitrage on factor residuals, and
the screen of each series' deviation from its trailing mean). Each transformed column is scored by its mean per-date rank correlation with the target
(IC), the t-stat of that mean over dates, and the share of 5 time blocks whose IC has the overall sign.
TARGET is ``y``; ``z`` (y z-scored within each date); ``pcaK`` (y minus its projection on the top K
principal components of the date x asset matrix of y, fitted on all dates, so in-sample); ``ols:x01,x02``
(the residual of a per-date cross-sectional regression of y on those features); or a ``resid.parquet``
(date, asset, resid, as the harness writes for a node), scored on its own dates only. The rank IC is the
same for y and for any per-date monotone transform of it, such as its within-date rank.
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import polars as pl
import polars.selectors  # noqa: F401

PACKAGES = ("raw", "cs", "lag", "cslag", "inter", "shape", "cond", "factor", "vol", "ou")
LAGS = (1, 2, 3, 5)


def cs_z(c: str) -> pl.Expr:
    return ((pl.col(c) - pl.col(c).mean().over("date")) / pl.col(c).std().over("date")).fill_nan(0.0)


def columns(d: pl.DataFrame, feats: list[str], package: str) -> tuple[pl.DataFrame, list[str]]:
    """``d`` sorted by asset, date, with the package's transformed columns added; their names."""
    by = lambda e: e.over("asset")   # noqa: E731
    if package == "raw":
        return d, feats
    if package == "cs":
        new = [cs_z(f).alias(f"{f}|cs") for f in feats]
    elif package == "lag":
        new = ([by(pl.col(f).shift(k)).alias(f"{f}|lag{k}") for f in feats for k in LAGS]
               + [by(pl.col(f).diff()).alias(f"{f}|diff") for f in feats]
               + [by(pl.col(f).rolling_mean(5).shift(1)).alias(f"{f}|mean5prev") for f in feats])
    elif package == "cslag":
        lagged = d.with_columns(by(pl.col(f).shift(k)).alias(f"_{f}_{k}") for f in feats for k in LAGS)
        out = lagged.with_columns((pl.col(f"_{f}_{k}").rank().over("date") / pl.len().over("date")).alias(
            f"{f}|cs_rank_lag{k}") for f in feats for k in LAGS)
        return out.drop([f"_{f}_{k}" for f in feats for k in LAGS]), [f"{f}|cs_rank_lag{k}" for f in feats
                                                                       for k in LAGS]
    elif package == "inter":
        z = d.with_columns(cs_z(f).alias(f"_{f}") for f in feats)
        new = [(pl.col(f"_{a}") * pl.col(f"_{b}")).alias(f"{a}*{b}") for a, b in itertools.combinations(feats, 2)]
        z = z.with_columns(new)
        return z.drop([f"_{f}" for f in feats]), [f"{a}*{b}" for a, b in itertools.combinations(feats, 2)]
    elif package == "shape":
        new = ([(pl.col(f) - pl.col(f).median().over("date")).abs().alias(f"{f}|abs") for f in feats]
               + [cs_z(f).pow(2).alias(f"{f}|sq") for f in feats]
               + [cs_z(f).sign().alias(f"{f}|sign") for f in feats]
               + [cs_z(f).clip(-1, 1).alias(f"{f}|clip1") for f in feats]
               + [(cs_z(f).abs() > 1.5).cast(pl.Float64).alias(f"{f}|tail") for f in feats])
    else:
        raise ValueError(package)
    out = d.with_columns(new)
    return out, [e.meta.output_name() for e in new]


def ic_table(d: pl.DataFrame, cols: list[str], target: str) -> pl.DataFrame:
    """Per column: mean per-date rank IC with ``target``, its t-stat over dates, and the share of 5 time
    blocks whose mean IC has the same sign as the overall mean."""
    dates = d["date"].unique().sort()
    block = pl.DataFrame({"date": dates, "block": np.arange(len(dates)) * 5 // len(dates)})
    per = (d.select(["date", target, *cols]).drop_nulls(target)
           .group_by("date").agg(pl.corr(pl.col(c).rank(), pl.col(target).rank()).alias(c) for c in cols)
           .join(block, on="date"))
    rows = []
    for c in cols:
        s = per.select("block", pl.col(c).fill_nan(None)).drop_nulls(c)
        if s.height < 10:
            continue
        m, sd = s[c].mean(), s[c].std()
        blocks = s.group_by("block").agg(pl.col(c).mean())[c].to_numpy()
        rows.append((c, m, m / sd * np.sqrt(s.height) if sd else 0.0, float(np.mean(np.sign(blocks) == np.sign(m)))))
    schema = {"column": pl.String, "ic": pl.Float64, "t": pl.Float64, "stable": pl.Float64}
    return pl.DataFrame(rows, schema=schema, orient="row").sort(pl.col("t").abs(), descending=True)


def cond_table(d: pl.DataFrame, feats: list[str], target: str) -> pl.DataFrame:
    """The raw per-date rank IC of each feature within subsets: 5 time blocks, the values of each feature with
    at most 10 distinct values, and dates after a low or high cross-sectional spread of the target."""
    dates = d["date"].unique().sort()
    d = d.join(pl.DataFrame({"date": dates, "period": np.arange(len(dates)) * 5 // len(dates)}), on="date")
    spread = d.group_by("date").agg(pl.col(target).std().alias("_sd")).sort("date")
    spread = spread.with_columns(pl.col("_sd").shift(1).alias("_sd"))
    d = d.join(spread.with_columns((pl.col("_sd") > pl.col("_sd").median()).cast(pl.Int8).alias("regime")),
               on="date").drop("_sd")
    splits = ["period", "regime"] + [f for f in feats if d[f].n_unique() <= 10]
    rows = []
    for g in splits:
        for v in d[g].drop_nulls().unique().sort():
            sub = d.filter(pl.col(g) == v)
            t = ic_table(sub, [f for f in feats if f != g], target)
            rows += [(f"{g}={v}", *r) for r in t.head(5).iter_rows()]
    return pl.DataFrame(rows, schema=["subset", "column", "ic", "t", "stable"], orient="row")


def pca_resid(d: pl.DataFrame, col: str, k: int) -> tuple[pl.DataFrame, float]:
    """(date, asset, resid) of ``col`` net of the top ``k`` principal components of its date x asset matrix
    (in-sample), and the share of its variance they explain."""
    wide = d.pivot(on="asset", index="date", values=col).sort("date")
    assets = wide.columns[1:]
    m = wide.select(assets).to_numpy()
    filled = np.nan_to_num(m - np.nanmean(m, axis=0))
    v = np.linalg.svd(filled, full_matrices=False)[2][:k]
    res = filled - filled @ v.T @ v
    share = 1 - (res ** 2).sum() / (filled ** 2).sum()
    long = pl.DataFrame(np.where(np.isnan(m), np.nan, res), schema=assets).with_columns(wide["date"]).unpivot(
        index="date", variable_name="asset", value_name="resid").drop_nans()
    return long.with_columns(pl.col("asset").cast(d["asset"].dtype)), float(share)


def ols_resid(d: pl.DataFrame, col: str, xs: list[str]) -> tuple[pl.DataFrame, float]:
    """(date, asset, resid) of a per-date cross-sectional regression of ``col`` on ``xs`` (nulls as 0), and
    the share of the variance of ``col`` it explains."""
    parts, ss, tot = [], 0.0, 0.0
    for g in d.drop_nulls(col).partition_by("date"):
        x = np.column_stack([np.ones(g.height), *(np.nan_to_num(g[c].to_numpy()) for c in xs)])
        yy = g[col].to_numpy()
        r = yy - x @ np.linalg.lstsq(x, yy, rcond=None)[0]
        ss, tot = ss + (r ** 2).sum(), tot + ((yy - yy.mean()) ** 2).sum()
        parts.append(g.select("date", "asset").with_columns(pl.Series("resid", r)))
    return pl.concat(parts), 1 - ss / tot


def target_column(d: pl.DataFrame, spec: str) -> tuple[pl.DataFrame, str]:
    """``d`` with the target named by ``spec`` added as column ``target``."""
    if spec == "y":
        return d.with_columns(pl.col("y").alias("target")), "target"
    if spec == "z":
        return d.with_columns(cs_z("y").alias("target")), "target"
    if spec.startswith("pca"):
        r = pca_resid(d, "y", int(spec[3:]))[0]
    elif spec.startswith("ols:"):
        r = ols_resid(d.with_columns(columns(d, [c for c in d.columns if c.startswith("x")], "lag")[0].select(
            pl.selectors.contains("|"))), "y", spec[4:].split(","))[0]
    else:
        r = pl.read_parquet(spec).select("date", "asset", "resid")
    return d.join(r.rename({"resid": "target"}), on=["date", "asset"], how="left"), "target"


def study(d: pl.DataFrame, feats: list[str], target: str, top: int) -> None:
    """What is left in ``target``: the best columns of the raw, lag, inter and shape packages, and whether
    it is predictable from its own past (per asset, and across assets at the last date)."""
    for pkg in ("raw", "lag", "inter", "shape"):
        dd, cols = columns(d, feats, pkg)
        print(f"   {pkg}:", ", ".join(f"{c} {ic:+.3f} (t {t:+.1f})" for c, ic, t, _ in
                                    ic_table(dd, cols, target).head(top).iter_rows()))
    own = d.with_columns(pl.col(target).shift(1).over("asset").alias("_prev"))
    t = ic_table(own, ["_prev"], target)
    if t.height:
        print(f"   own lag 1 rank IC {t['ic'][0]:+.3f} (t {t['t'][0]:+.1f})")


def factor(d: pl.DataFrame, feats: list[str], target: str, top: int) -> None:
    """Common factor studies of the target, each followed by a study of what it leaves: principal
    components of the date x asset matrix, and stepwise peeling, which regresses out the strongest raw or
    lag-1 feature, rescreens the residual and repeats while something has |t| > 3."""
    for k in (1, 3, 5):
        r, share = pca_resid(d, target, k)
        print(f"-- pca{k}: explains {share:.1%} of the target's variance in-sample; what is left:")
        study(d.join(r, on=["date", "asset"], how="left"), feats, "resid", top)
    lagged, lag_cols = columns(d, feats, "lag")
    cand = feats + [c for c in lag_cols if c.endswith("|lag1")]
    chosen, cur = [], lagged
    for i in range(4):
        t = ic_table(cur, [c for c in cand if c not in chosen], target if not chosen else "resid").head(1)
        if not t.height or abs(t["t"][0]) < 3:
            break
        chosen.append(t["column"][0])
        r, share = ols_resid(lagged, target, chosen)
        print(f"-- peel {i + 1}: add {chosen[-1]} (rank IC {t['ic'][0]:+.3f}, t {t['t'][0]:+.1f}); "
              f"{', '.join(chosen)} explain {share:.1%} of the target's variance; what is left:")
        cur = lagged.join(r, on=["date", "asset"], how="left")
        study(cur, feats, "resid", top)
    if not chosen:
        print("-- peel: no raw or lag-1 feature has |t| > 3")


def vol(d: pl.DataFrame, feats: list[str], target: str, top: int) -> None:
    """The size and the sign of the target, screened separately, and how much of the size is predictable from
    its own past (volatility clustering)."""
    d = d.with_columns(pl.col(target).abs().alias("_size"), pl.col(target).sign().alias("_sign"))
    d = d.with_columns(*(pl.col("_size").shift(k).over("asset").alias(f"|target||lag{k}") for k in (1, 5)),
                       pl.col("_size").shift(1).rolling_mean(20).over("asset").alias("|target||mean20prev"))
    own = ["|target||lag1", "|target||lag5", "|target||mean20prev"]
    for name, col in (("size |target|", "_size"), ("sign", "_sign")):
        print(f"-- {name}:")
        for pkg in ("raw", "lag", "shape"):
            dd, cols = columns(d, feats, pkg)
            cols = cols + own if (pkg == "raw" and col == "_size") else cols
            print(f"   {pkg}:", ", ".join(f"{c} {ic:+.3f} (t {t:+.1f})" for c, ic, t, _ in
                                        ic_table(dd, cols, col).head(top).iter_rows()))


WINDOW = 60


def s_score(d: pl.DataFrame, target: str) -> tuple[pl.DataFrame, float]:
    """The OU s-score of each asset at each date, from the target's past only: the cumulative sum of its last
    WINDOW values is fitted as an AR(1) X' = a + b X + e; the score is (X_last - a/(1-b)) / sqrt(var(e)/(1-b^2)),
    null where b is not in (0, 1). Returns (date, asset, s_score) and the median b."""
    rows, bs = [], []
    for g in d.select("date", "asset", target).sort("date").partition_by("asset"):
        r = np.nan_to_num(g[target].to_numpy())
        for t in range(WINDOW, g.height):
            x = np.cumsum(r[t - WINDOW:t])
            b = np.cov(x[1:], x[:-1])[0, 1] / x[:-1].var(ddof=1)
            bs.append(b)
            if not 0 < b < 1:
                continue
            a = x[1:].mean() - b * x[:-1].mean()
            e = x[1:] - a - b * x[:-1]
            rows.append((g["date"][t], g["asset"][t], (x[-1] - a / (1 - b)) / np.sqrt(e.var() / (1 - b * b))))
    sc = pl.DataFrame(rows, schema={"date": d["date"].dtype, "asset": d["asset"].dtype, "s_score": pl.Float64},
                      orient="row")
    return sc, float(np.median(bs)) if bs else float("nan")


def ou(d: pl.DataFrame, feats: list[str], target: str, top: int) -> None:
    """Per series (the target and each feature): the median over assets of the AR(1) coefficient of its
    deviation from the asset's mean, and the half-life it implies; then the screen of each series' deviation
    from its trailing 20-date mean, scaled by its trailing sd, against the target."""
    rows = []
    for c in [target, *feats]:
        x = pl.col(c) - pl.col(c).mean().over("asset")
        g = d.select("asset", x.alias("v"), x.shift(1).over("asset").alias("p")).drop_nulls()
        phi = g.group_by("asset").agg((pl.cov("v", "p") / pl.col("p").var()).alias("phi"))["phi"].median()
        half = np.log(0.5) / np.log(phi) if phi and 0 < phi < 1 else float("nan")
        rows.append((c if c != target else "target", phi, half))
    print(pl.DataFrame(rows, schema=["series", "ar1", "half_life"], orient="row").fill_nan(None).sort("ar1", descending=True, nulls_last=True))
    prev = lambda c, f: getattr(pl.col(c).shift(1), f)(20).over("asset")   # noqa: E731
    dev = [((pl.col(c) - prev(c, "rolling_mean")) / prev(c, "rolling_std")).alias(f"{c}|dev20") for c in feats]
    dev.append(((pl.col(target).shift(1).over("asset") - prev(target, "rolling_mean").shift(1))
                / prev(target, "rolling_std").shift(1)).alias("target_prev|dev20"))
    dd = d.with_columns(dev)
    sc, b = s_score(d, target)
    dd = dd.join(sc, on=["date", "asset"], how="left")
    t = ic_table(dd, ["s_score"], target)
    if t.height:
        hl = np.log(0.5) / np.log(b) if 0 < b < 1 else float("nan")
        print(f"   OU on the cumulative target over the last {WINDOW} dates (per asset): median AR(1) of the cumulative "
              f"{b:.3f} (half-life {hl:.1f} dates); s-score rank IC {t['ic'][0]:+.3f} (t {t['t'][0]:+.1f})")
    print("   deviation from trailing mean:", ", ".join(f"{c} {ic:+.3f} (t {t:+.1f})" for c, ic, t, _ in
          ic_table(dd, [e.meta.output_name() for e in dev], target).head(top).iter_rows()))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("packages", nargs="+", choices=PACKAGES)
    p.add_argument("--data", default="../data/train.parquet")
    p.add_argument("--target", default="y", help="y, z, pcaK, ols:x01,x02, or a residuals parquet")
    p.add_argument("--top", type=int, default=30)
    a = p.parse_args(argv)
    d = pl.read_parquet(a.data).sort("asset", "date")
    feats = [c for c in d.columns if c.startswith("x")]
    d, target = target_column(d, a.target)
    target_name = a.target
    pl.Config.set_tbl_rows(a.top + 5)
    pl.Config.set_tbl_width_chars(120)
    for pkg in a.packages:
        if pkg == "cond":
            print(f"== cond: top 5 features by |t| within each subset, target {target_name}")
            print(cond_table(d, feats, target))
            continue
        if pkg in ("vol", "ou"):
            print(f"== {pkg}: target {target_name}")
            (vol if pkg == "vol" else ou)(d, feats, target, 5)
            continue
        if pkg == "factor":
            print(f"== factor: target {target_name}, top 5 per package of what each factor model leaves")
            factor(d, feats, target, 5)
            continue
        dd, cols = columns(d, feats, pkg)
        t = ic_table(dd, cols, target)
        print(f"== {pkg}: {len(cols)} columns, target {target_name}, top {a.top} by |t| "
              f"(|t| > 3: {t.filter(pl.col('t').abs() > 3).height})")
        print(t.head(a.top))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
