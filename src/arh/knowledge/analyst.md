# Analyst: measuring what bears on the forecast

Background, not a procedure. These are kinds of measurements that are often
useful and the standards that make one reusable. Which ones matter depends
on the data; choose by what is unknown and would change a model, not by
working down the list.

You run code and write findings. A finding is useful to a forecaster only if
it tells them what to put in a model and what to expect from it. Read
`general.md` first; snippets assume its preamble (`df`, `feats`, `ic_by_date`).

## 1. Order of questions

Each later question is cheap once the earlier one is answered.

1. Where does the variance of y sit (share_date), and is the date component
   forecastable from date-level aggregates or lagged market returns? This
   decides whether per-date transforms are admissible (general.md 2.1).
2. How heavy are the tails of y and of each feature, and do the tails of the
   strong features carry signal beyond a linear fit (general.md 2.3)?
3. Which features have a cross-sectional relation with y (IC with t-stat,
   sign consistency, both halves)? Which of those are proxies for each other?
4. What shape is each relation (quantile sorts), and does it depend on a
   date-level state?
5. What is in `history` that the features do not contain: past returns at
   several horizons, past volatility, past cross-sectional dispersion?
6. What does the best existing model still miss (residual analysis)?

## 2. Recipes

### Quantile sort with per-date spreads

```python
def sort_table(df, col, q):
    b = df.with_columns(bucket=((pl.col(col).rank("ordinal").over("date") - 1) * q
                                // pl.len().over("date")))
    per_date = (b.group_by("date", "bucket").agg(pl.col("y").mean())
                 .pivot(on="bucket", index="date", values="y").sort("date"))
    spread = per_date[str(q - 1)] - per_date["0"]
    table = b.group_by("bucket").agg(pl.col("y").mean(), pl.len()).sort("bucket")
    return table, spread.mean(), spread.mean() / (spread.std() / spread.len() ** 0.5)
```

The number of buckets is a choice; too many for the number of assets per
date gives noisy buckets. Report mean y per bucket, the top-minus-bottom
spread and its t-stat over dates.

### Tail check before anyone clips

```python
def tail_check(df, col, q):
    z = df.with_columns(u=(pl.col(col).rank().over("date") - 0.5) / pl.len().over("date"))
    core = z.filter(pl.col("u").is_between(q, 1 - q))
    line = np.polyfit(core[col].to_numpy(), core["y"].to_numpy(), 1)
    out = {}
    for name, side in (("low", pl.col("u") < q), ("high", pl.col("u") > 1 - q)):
        t = z.filter(side)
        out[name] = (t["y"].mean(), float(np.polyval(line, t[col].to_numpy()).mean()))
    return out
```

Returns, per tail, the realised mean y and the linear extrapolation from the
centre. Two comparisons answer two questions. Against the line in raw units:
a tail near or above it says clipping removes signal, a tail far below it or
of the wrong sign says clipping removes noise. Against the sort table's trend
in bucket number: extreme buckets that jump far beyond the trend say a rank
transform loses signal. Repeat in both halves of the dates.

### Regime split of an IC

```python
def ic_by_regime(df, col, state_col):
    s = (df.group_by("date").agg(pl.col(state_col).mean().alias("s"))
           .with_columns(hi=pl.col("s") > pl.col("s").median()))
    ic = ic_by_date(df, col).join(s, on="date")
    return (ic.group_by("hi").agg(pl.col("ic").mean().alias("mean_ic"), pl.col("ic").std().alias("sd_ic"),
                                  pl.len().alias("n_dates")).sort("hi"))
```

Any date-level variable can be the state: a date-constant feature, the
per-date SD of y from `history`, the per-date mean of a feature, realised
market volatility. A regime finding needs the difference between the two
mean ICs to exceed about two of its standard error, and to appear in both
halves. Splitting one effect by many states is a multiple-testing problem;
count the splits.

### Per-asset heterogeneity of a slope

```python
def slope_spread(df, col):
    g = df.group_by("asset").agg(b=pl.cov(col, "y") / pl.col(col).var(),
                                 se=pl.col("y").std() / (pl.col(col).std() * pl.len().sqrt()))
    return g["b"].var() - (g["se"] ** 2).mean()   # > 0: slopes differ across assets
```

If the variance of per-asset slopes is no larger than what their standard
errors imply, pooling is right and per-asset models only add noise. If it is
larger, partial pooling (shrinking per-asset slopes toward the pooled one)
is a candidate for the forecaster.

### Adversarial validation

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def date_split_auc(df, cols):
    lab = (df["date"] > df["date"].median()).to_numpy()
    clf = GradientBoostingClassifier(max_depth=2, n_estimators=50)
    auc = cross_val_score(clf, df.select(cols).to_numpy(), lab, cv=5, scoring="roc_auc").mean()
    imp = clf.fit(df.select(cols).to_numpy(), lab).feature_importances_
    return auc, dict(zip(cols, imp.round(3)))
```

An AUC near one half means the feature distribution is stationary. A high
AUC names the features whose distribution changes over time (importances);
a model fitted on early dates extrapolates on those, and a per-date or
differenced form is worth comparing with the level.

## 3. What makes a finding solid

- Effect size in the units a forecaster uses: mean IC, spread in units of y,
  slope with standard error, share of var(y). A p-value alone is not reusable.
- n: number of dates, not rows. Persistent features inflate row counts.
- Stability: both halves of the dates, with the sign and size in each.
- Scope: what was held fixed (transform, horizon, sample) and what was not
  tried. A negative is a finding when its scope is written down.
- A null control: shuffle y within date (or across dates) and rerun the same
  screen once, to see what size of effect appears by chance with this many
  features and splits.
- The script as artifact, with the exact definition of every derived
  variable, so a forecaster can compute it inside `fit` and `predict`.
- One claim per finding, then the numbers, then how it was measured, then
  what it does not show.

Common errors: pooled correlations on fat-tailed data without the per-date
rank version alongside; t-stats from row-level OLS on persistent features;
regime findings from one split without a control; sorts with more buckets
than assets per date can support; an "effect" that is a proxy of a feature
already in a finding.

## 4. Creative measurements

Less obvious things an experienced quant measures. Each: what, when, cheap test.

- **Date-level forecastability**: regress m_t on date-level aggregates of the
  features and on lagged m_t, walk-forward. Helps when share_date is large;
  decides the whole transform question.
- **Residuals of the best model**: refit the best forecast's recipe
  walk-forward, then compute the IC of its residuals with every feature,
  with ranks, with pairwise bins and with `history` features. What survives
  is what the model misses.
- **IC time series**: autocorrelation of the per-date IC, and its correlation
  with lagged dispersion, lagged volatility and date-level features. A
  predictable IC supports time-varying signal weights.
- **Signal horizon**: IC of each feature against y summed over the next h
  periods for several h, and against y lagged by several dates. Slow decay
  supports smoother models; a signal present only at one offset is fragile.
- **Pipeline power check**: add a synthetic linear effect of known size to a
  copy of y and run the same screen. The smallest size recovered is the
  minimum detectable effect; report it with negatives.
- **Interaction scan**: for each pair, the slope of y on feature A in the top
  vs bottom half of feature B, compared against the number of pairs
  screened. Only pairs stable across halves count.
- **Shape scan**: slope of y on a feature within each quantile bin of that
  feature. Varying slopes across bins is nonlinearity; the sort table shows
  where.
- **Feature-on-features**: R² of each feature regressed on the others. A
  feature that is nearly a function of the rest is redundant, or a transform
  of something already present.
- **Dispersion as state**: per-date SD of y and of each strong feature over
  time, and whether ICs differ on high- vs low-dispersion dates.
- **Rank-vs-Pearson gap**: features where the pooled correlation and the mean
  IC disagree in size or sign. The gap locates signal in the tails or in the
  date dimension.
- **Per-asset structure**: per-asset mean and SD of y, per-asset beta to the
  per-date mean, and whether any of them is stable from the first half to
  the second. Stable heterogeneity is a feature; unstable is noise.
- **Missingness and discreteness**: is null-ness or a repeated value of a
  feature related to y or to the date?
- **Lead-lag at the date level**: correlation of m_t with lagged date-level
  aggregates of features and with lagged means of asset groups.
- **Nonlinear dependence screen**: mutual information or distance correlation
  of each feature with y, compared with the Spearman IC. A feature that
  scores on the former and not the latter has a non-monotone relation worth
  a sort.
