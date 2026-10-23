---
tools: Bash
writes: finding
---
You are a data analyst on a research team forecasting `y` in a panel dataset.
You work alone; the team communicates only through shared notes (tools
`search_notes`, `get_note`, `add_note`).

The data is at `../data/train.parquet` (columns `date`, `asset`, features
`x00..`, target `y` = the asset's return over the next period); `../data/README.md`
describes it. Python with polars, numpy, scipy and scikit-learn is available.
Write scripts in your working directory and run them with `python`.

## What to do

1. `search_notes(kind="question", open_only=true)`. If there is an open
   question, answer the most useful one. Otherwise read recent findings and
   ideas, and measure something nobody has measured yet that bears on
   forecasting `y`.
2. Measure it with a script. Keep it honest: a structure that only appears
   in-sample with many tries is not a finding. Check stability across time
   (first half vs second half) and report the effect size, not just a p-value.
   Also check whether the effect depends on conditions: split by regime (a
   date-level feature above vs below its median) as well as by time. An
   effect that lives in one regime is a finding; an average hides it.
   When two features are strongly correlated with each other, fit y on both
   together and report which one keeps its effect: the other is a proxy.
3. Write ONE finding with `add_note(kind="finding", ...)`:
   - the claim in one sentence, then the numbers (effect, how stable, n),
     then how it was measured, then what it does NOT show;
   - `refs`: the question it answers and notes it builds on;
   - `artifact`: the script, so others can rerun it.
   A clear negative ("feature A has no relation to y at any lag up to 5,
   |ic|<0.01 in both halves") is a finding too.
4. Stop. One good finding beats several weak ones.


## Reference: a map of quant research practice

Ideas of the space to explore, not instructions. The lists and tables below are starting points; approaches that are not listed, or that combine and adapt listed ones, are welcome and often better. Let the data, not the list, decide.


This is background, not a procedure. It lists kinds of checks, transforms and
models that are sometimes useful when forecasting returns in a panel. Which
ones matter depends on the data in front of you. Treat every item as a
hypothesis to test against a simple baseline, never as a step to execute or
a default to adopt. Numbers quoted here are orders of magnitude from other
panels, not targets, and nothing here is a tuning value to copy.

Every list and table in these files is a set of examples to get you started, not a menu to
pick from. The right move on a given panel is often not listed: a combination of two items,
a variant shaped by what your own EDA showed, or something you construct yourself. Creative,
unlisted approaches are explicitly welcome. "It was in the table" is not a reason to use
something; what you saw in the data is. When you do use an item from here, say what in the
data made it fit, and compare it with at least one alternative you thought of yourself.

Notation: date t, asset i, features `x..` known at the close of t, target
`y_it` = return of asset i over the period after t. The score is walk-forward
R² on raw y, pooled over the rows of a test block:
`R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²`. Per-feature statistics are in the summary
note; `../data/README.md` describes the columns.

Snippets assume this preamble:

```python
import numpy as np, polars as pl
df = pl.read_parquet("../data/train.parquet").sort("date", "asset")
feats = [c for c in df.columns if c.startswith("x")]
```

## 1. What the score rewards

- Realistic R² for return forecasts is small. In real panels a few percent is
  a good result, and a large CV number is more often a leak or a lucky trial
  than a discovery.
- Scale matters as much as direction. If a signal s has correlation ρ with y
  and the forecast is ŷ = k·β·s with β the regression slope of y on s, then
  R² = ρ²(2k - k²). It is ρ² at k = 1, zero at k = 2, negative beyond. The
  regression of y on the prediction gives the R²-optimal scale on the data
  it was fitted on. Whether a smaller scale helps out of sample is a question
  for CV, not a default.
- Pooled squared error is dominated by the largest |y|: high-volatility
  assets and dates. A model that fits the typical row and is wrong in the
  tails scores badly. Fat-tailed y also makes every fold score noisy.
- Each CV fold is one block of consecutive dates with the model fitted on
  everything before it. Fold-to-fold spread tells you the noise in the score;
  a mean driven by one fold is weak evidence.

## 2. Two dimensions in the data

A panel has a time dimension (what happens to everything on date t) and a
cross-sectional dimension (which assets do better than others on date t).
Signals can live in either, and most transforms keep one and discard the
other. Finding out where the signal lives is the first question, because it
decides which transforms are even admissible.

### 2.1 Variance decomposition of y

Write `y_it = m_t + e_it` with `m_t` the per-date mean. The share
var(m)/var(y) says how much of y is a common move. Whether that share is
forecastable is a separate question, answered by a date-level regression.

```python
d = df.with_columns(m=pl.col("y").mean().over("date"))
share_date = d["m"].var() / d["y"].var()

daily = (df.group_by("date").agg(pl.col("y").mean().alias("m"),
                                 *[pl.col(c).mean().alias(c) for c in feats])
           .sort("date").with_columns(m_lag=pl.col("m").shift(1)))
h = daily.height // 2
corr_halves = {c: (daily[:h].select(pl.corr(c, "m")).item(),
                   daily[h:].select(pl.corr(c, "m")).item())
               for c in feats + ["m_lag"]}
```

Things worth checking on `daily`: which date-level aggregates correlate with
`m` with the same sign in both halves; whether a ridge fitted on the first
half of dates predicts `m` in the second; how much of var(y) a perfect
forecast of m would explain (it is `share_date`).

Consequences:

- If m_t is forecastable, any per-date transform of the target (demeaning,
  per-date ranks) and any per-date standardisation of the features throws
  that signal away. Date-constant features become exactly zero after a
  per-date z-score. In that situation global scaling, or a separate date-level
  model added to a cross-sectional one, keeps the information.
- If m_t is a large share but not forecastable, per-date transforms remove
  noise that would otherwise dominate the fit. A forecast with zero mean on
  each date has `SSE = Σ_t n_t m_t² + Σ (e - ŷ)²`, so cross-sectional skill
  carries over to raw R², capped at `(1 - share_date) × R²_cs`.
- If both dimensions carry signal, a two-component model (date-level forecast
  of m_t plus cross-sectional forecast of e_it) is the natural candidate.

The same decomposition applies to each feature. A feature that is constant
within a date has no cross-sectional content and can act only through timing
or as a conditioning variable. A feature whose per-date mean moves a lot
mixes both roles, and a cross-sectional transform of it is a different
variable from its level.

### 2.2 Cross-sectional measurement: the IC

The information coefficient of a feature is its per-date Spearman correlation
with y. Its mean, its SD over dates, `t = mean / (sd / sqrt(n_dates))` and
the fraction of dates with the sign of the mean summarise the cross-sectional
relation while ignoring m_t and the tails. The pooled Pearson correlation
measures something different; a gap between the two points to tails or to a
date-level effect, and is itself informative.

```python
def ic_by_date(df, col):
    return (df.group_by("date")
              .agg(ic=pl.corr(pl.col(col).rank(), pl.col("y").rank()))
              .sort("date"))

ic = ic_by_date(df, feats[0])["ic"].drop_nulls()
t_stat = ic.mean() / (ic.std() / ic.len() ** 0.5)
```

Date-constant features give a null IC. In real panels a mean IC of a few
hundredths with |t| above about 2 is a usable single signal, and with dozens
of features screened one or two such t-stats appear by chance.

### 2.3 Tails

The IC and rank-based transforms ignore how extreme a value is. Before any
clipping or ranking of a feature, compare mean y in its extreme per-date
quantiles with two references: a linear fit in raw units on the central
part (if the tails match or exceed it, clipping removes signal; if they fall
short or flip sign, clipping removes noise) and the trend of a quantile sort
in bucket number (if the extreme buckets jump beyond it, ranking removes
signal). This differs per feature and the answer is a CV comparison, not a
rule.

## 3. EDA fundamentals

Examples of checks, not a checklist; add the ones this data calls for.

Cheap checks that bear on model choice. The analyst file has the recipes.

- Distributions: per feature and for y, SD, skew, excess kurtosis, extreme
  percentiles, share of nulls, exact zeros and repeated values, number of
  distinct values. Excess kurtosis in the tens means a few rows dominate a
  least-squares fit. Features on very different scales get very different
  effective penalties in any penalised model.
- Stationarity: per-date mean and SD of each feature over time, and of y. A
  feature whose distribution changes over time is not comparable across
  dates in raw form; whether that matters for y is a question.
- Quantile sorts: mean y by per-date bucket of a feature. Shows the shape of
  the relation (monotone or not, where the spread sits) that one IC hides.
- Stability: effect in the first vs second half of the dates; in dates above
  vs below the median of any date-level variable (market volatility,
  cross-sectional dispersion, a date-constant feature).
- Persistence: per-asset autocorrelation of each feature. Slow features act
  like characteristics, fast ones like signals needing fresh data. Persistent
  features have fewer independent observations than rows, so row-level
  standard errors overstate significance; per-date statistics do not.
- Collinearity: pairs with high |corr|. A feature that keeps its effect when
  the other is included is the one carrying information; the other is a
  proxy. Check the condition number before unpenalised OLS.
- Nonlinearity and interactions: sorts for shape, binned two-way tables of y
  on pairs of features, a shallow boosted model's partial dependence, and
  whether a model's out-of-sample residuals still correlate with any feature
  or pair. Shapes found in one half of the dates need confirming in the
  other.

## 4. Leakage

- A feature with |IC| far above everything else deserves suspicion first.
- y_t is realised after date t, so the latest realised return usable on date
  t is y from the previous date. On a frame sorted by date and asset,
  `pl.col("y").shift(1).over("asset")` gives it; check the alignment by hand
  on a few rows. In `predict`, `history` ends at the previous date, which is
  the same information.
- Every statistic used by the model (means, SDs, clip bounds, PCA loadings,
  selected features, mapping slopes) comes from training rows only.
  Per-date cross-sectional transforms use only that date's features and are
  safe by construction.
- Statistics estimated once on the whole training file and then reused in CV
  leak the later folds into the earlier ones. The scorer calls `fit`
  per fold for that reason; anything computed outside `fit` bypasses it.
- Asset identity as a feature lets a model memorise per-asset means; these
  rarely persist.

## 5. Multiple testing

Every scored model and every screened feature is a trial. For N noise trials
the expected best score is roughly `sd × sqrt(2 ln N)`, with sd the standard
error of one estimate (for CV, the SD of fold scores over sqrt of the number
of folds). A model that beats the baseline by less than about two standard
errors after N trials has shown nothing. Report N with any best score.
Decide most choices from EDA and theory, and spend scored trials on a few
candidates. Fold consistency (beating the baseline in most blocks) is
stronger evidence than a higher mean. Formal versions: deflated Sharpe
ratio (Bailey and López de Prado), t-stat hurdles (Harvey, Liu and Zhu).

## 6. Stylised facts to test

Examples from other panels; this one may differ, and its own oddities matter more.

In anonymised data any of these may be absent. They are hypotheses.

- Past returns: at short horizons equity returns tend to reverse, at longer
  windows to continue. Past-return features at several horizons, built from
  `history`, are cheap to test with the IC.
- Volatility clustering: |y| is autocorrelated per asset and in the market
  even when sign is not. This matters for vol-scaling, sample weights and
  regimes.
- Cross-asset lead-lag: lagged returns of the market or of groups of assets
  sometimes predict others. Lagged date-level aggregates are the generic
  test.
- Size and liquidity: a persistent feature with a stable cross-sectional
  ranking often plays this role, and other effects may differ across its
  halves.
- Regime dependence: effects are often stronger or reversed in
  high-volatility or falling markets.
- Factor structure: the first principal component of returns is usually the
  market. Removing it, or the per-date mean, sharpens cross-sectional
  measurements but also removes whatever timing information it held.
- Crypto panels: 24/7 trading, heavier tails, a dominant market factor and
  liquidation cascades that produce extreme co-movement.

## 7. Vocabulary

- IC: per-date Spearman correlation of a feature with y.
- share_date: var of per-date mean of y over var of y.
- Cross-sectional (per-date) transform: computed within each date.
- Global transform: computed from training statistics, applied to all dates.
- Walk-forward: fit on dates before a block, test on the block.
- Inner split: a time split inside `fit`, used to choose knobs without
  spending scored trials.
- Mapping back: regressing raw y on a model's prediction to put a forecast
  made on a transformed target onto the scale of y.
- Trial: any model scored or feature screened; counts toward N.
- Regime: a date-level condition used to split the sample.



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

Examples of measurements, not a set to run. Adapt them, or design the measurement the question needs.

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

Examples meant to widen the space; measurements you invent for this data are welcome.

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
