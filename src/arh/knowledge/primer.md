# Quant research primer: return panels

Standard practice for forecasting next-period returns in a panel of assets.
It applies to equity and crypto panels in general. Treat every item as
something to check in the data at hand, not as a fact about it.

Notation: `date` t, `asset` i, features `x..` known at the close of t, target
`y` = return of asset i over the next period. The score is walk-forward R² on
raw `y`, pooled over all rows of a test fold:
`R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²`.

## 1. What the score rewards

- Return forecasts have low R². In real panels 0.5 to 2 percent is good.
  A large CV number on a real-looking panel is more likely a leak or luck
  than a discovery.
- The scale of ŷ matters as much as its direction. If a signal s has
  correlation ρ with y and you predict ŷ = k·β·s, where β = cov(y, s)/var(s),
  then R² = ρ²(2k - k²). At k = 1 you get ρ². At k = 2 you get zero, and
  beyond that it is negative. Overconfident forecasts are punished hard, so
  shrink toward zero when in doubt.
- Pooled squared error is dominated by the largest |y|: high-volatility
  assets, high-volatility dates and tail events. A model that fits the
  typical row well but is wrong on the tails can score poorly.
- `fit` is called once per fold. `predict` also receives `history` (all past
  rows with y), so features built from past returns are available at
  forecast time. Compute them identically in `fit` and `predict`.

## 2. Exploratory analysis

Run these before choosing a model. Each one is cheap.

### 2.1 Variance decomposition of y

Split y into a common (per-date) part and a cross-sectional part:
`y_it = m_t + e_it`, with `m_t` the per-date mean. Report var(m)/var(y). In
equity panels the common part is often 10 to 30 percent of var(y); in crypto
it is usually larger. Do the same for each feature: a feature that is nearly
constant within a date is a market-level variable, has no cross-sectional
information, and acts through time-series timing or as a conditioning
variable.

```python
import polars as pl
df = pl.read_parquet("../data/train.parquet")
d = df.with_columns(m=pl.col("y").mean().over("date"))
share_time = d["m"].var() / d["y"].var()
```

### 2.2 Distributions and scale

For each feature and for y: mean, SD, skew, excess kurtosis, 1st/99th
percentiles, share of missing values, share of exact zeros or repeated values.
Excess kurtosis above about 10 means a few rows will dominate any least-squares
fit. Features on very different scales matter for any penalised model (see 3.1).

### 2.3 Information coefficient (IC)

The IC of a feature is the cross-sectional Spearman correlation of the feature
with y, computed per date. Report its mean, its SD over dates, the t-stat
`mean / (sd / sqrt(n_dates))`, and the fraction of dates with the same sign as
the mean. Per-date IC removes the common component and is robust to tails, so
it is the standard first screen. Also compute the pooled Pearson correlation;
a large gap between the two points to tails or to a time-series effect.

```python
def ic_by_date(df, col):
    return (df.group_by("date")
              .agg(ic=pl.corr(pl.col(col).rank(), pl.col("y").rank()))
              .sort("date"))

ic = ic_by_date(df, "x00")
t = ic["ic"].mean() / (ic["ic"].std() / ic.height ** 0.5)
```

Rules of thumb: a single-feature mean IC of 0.02 to 0.05 is a usable signal in
real data. |t| < 2 is noise. With 30 features screened, expect one or two with
|t| > 2 by chance.

Signal decay: repeat the IC with the feature lagged by k dates (k = 0, 1, 2,
...) and with y summed over the next h periods. A signal that decays slowly
supports a slower, smoother model; one that only exists at k = 0 is fragile.

### 2.4 Quantile sorts

Per date, sort assets into quintiles or deciles of the feature and average y
in each bucket, then average over dates. Check monotonicity and the spread
between top and bottom buckets, with a t-stat from the per-date spreads. A
sort shows the shape of the relation (linear, saturating, non-monotone,
concentrated in one tail) that an IC summarises into one number.

### 2.5 Stability

- Split dates into halves or thirds and recompute IC and sort spreads. An
  effect that flips sign or lives in one subperiod should not be traded at
  full weight.
- Split by regime: dates with high vs low market volatility, up vs down
  market, high vs low cross-sectional dispersion, and above vs below the
  median of any date-level feature. Report the effect in each regime.
- Plot or tabulate the rolling mean IC (for example a 60-date window).

### 2.6 Persistence and turnover

- Per-asset autocorrelation of each feature at lag 1, and the rank
  autocorrelation of the cross-section from one date to the next. Slow
  features (autocorrelation near 1) behave like characteristics; fast ones
  behave like signals that need fresh data each period.
- Persistent features have fewer effective independent observations than
  rows, so their t-stats are overstated. Use per-date ICs (one number per
  date) or Newey-West errors, not row-level OLS errors.

### 2.7 Collinearity and near-duplicates

Correlation matrix of features, pooled and per date. Pairs with |corr| > 0.8
are candidates for near-duplicates or proxies. Fit y on both together (or
compare partial ICs) to see which one keeps its effect. Check the condition
number of the feature matrix before using unpenalised OLS.

### 2.8 Lookahead and leakage

- A feature with |IC| above about 0.2, or far stronger than everything else,
  is suspicious. Check whether it is a function of y or of future data.
- Lagged-return features must use returns realised by date t only. y at date
  t is realised after t, so the latest y usable when forecasting date t is
  the one from date t-1. Use `shift(1).over("asset")` on a date-sorted frame
  and check the alignment by hand on a few rows.
- Any statistic used in the model (means, SDs, clip bounds, bins, PCA loadings,
  feature selection) must be computed from training rows only. Per-date
  cross-sectional transforms use only that date's features and are safe.
- Asset identifiers used as features let a model memorise per-asset means.
  These rarely persist out of sample.

### 2.9 Nonlinearity and interactions

A linear model on raw features is the baseline, not the answer. General checks:
quantile sorts (2.4) for shape; binned 2D sorts of y on pairs of features;
partial dependence from a shallow gradient-boosted model; and whether a
flexible model's out-of-sample residuals still correlate with any feature,
transform or pair. Only keep structure that holds in both halves of the data.

## 3. Preprocessing

### 3.1 Scaling

Ridge and lasso penalise coefficients, and coefficient size depends on the
feature's unit. Unscaled features with SDs from 0.8 to 5 get very different
effective penalties, so the penalty choice is arbitrary. Standardise before
any penalised or distance-based model. Tree models do not need scaling.

- Global z-score: `(x - mean_train) / sd_train`, statistics from `fit` only.
- Cross-sectional z-score: per date, `(x - mean_t) / sd_t`. This removes the
  common level of the feature, makes it comparable across dates and needs no
  stored statistics. It is the default for cross-sectional signals.
- Choose one deliberately. A cross-sectional z-score discards any time-series
  information in the feature level; keep a global version alongside it if
  the level may matter.

```python
def cs_z(df, cols):
    return df.with_columns([
        ((pl.col(c) - pl.col(c).mean().over("date"))
         / pl.col(c).std().over("date")).fill_nan(0).fill_null(0).alias(c)
        for c in cols])
```

### 3.2 Winsorizing

Clip each feature at percentiles computed on the training data (for example
1st/99th, or median ± 5 MAD), store the bounds in the fit state, and apply the
same bounds in `predict`. Per-date winsorizing (clip at that date's
percentiles) also avoids stored statistics. `clip(-3, 3)` on an unscaled
feature is not winsorizing; the bounds mean nothing unless the feature is
already standardised.

```python
def fit_bounds(train, cols, q=0.01):
    return {c: (train[c].quantile(q), train[c].quantile(1 - q)) for c in cols}

def apply_bounds(df, bounds):
    return df.with_columns([pl.col(c).clip(lo, hi) for c, (lo, hi) in bounds.items()])
```

### 3.3 Rank and Gaussian-rank transforms

Per-date rank scaled to (0, 1), optionally mapped through the inverse normal
CDF. This removes outliers and scale in one step and is the most robust
choice for heavy-tailed features. It discards the magnitude of extreme values,
so compare it against the winsorized z-score rather than assuming it is better.

```python
import numpy as np
from scipy.special import ndtri

def gauss_rank(df, cols):
    u = df.select([((pl.col(c).rank().over("date") - 0.5)
                    / pl.len().over("date")).alias(c) for c in cols])
    return df.with_columns([pl.Series(c, ndtri(u[c].to_numpy())) for c in cols])
```

### 3.4 Missing values

Count missing values per feature and per date. Impute with the per-date
cross-sectional median (or 0 after a cross-sectional z-score), and add a
missing indicator if missingness itself correlates with y. Never impute with
statistics from the full sample. `predict` must return a finite number for
every row, so guard against NaN after every transform.

## 4. Target choice

The score is on raw y, but the model can be fitted to a different target.

| target | when it helps | cost |
|---|---|---|
| raw y | small common component, light tails | fit is driven by tail rows and by m_t |
| winsorized y (train bounds) | heavy-tailed y | biased if the tails are predictable |
| per-date demeaned y | large common component | gives up any timing of m_t |
| per-date rank or Gaussian rank of y | heavy tails and a large common component | loses magnitude; needs rescaling |
| vol-scaled y (y / σ_it from past data) | volatility differs a lot across assets or over time | reweights rows away from what raw R² weights |

Fitting on a transformed target requires mapping predictions back to the
scale of y. In `fit`, produce in-sample (or better, an inner time split)
predictions s on the transformed target, regress raw y on s, and store the
slope b. In `predict`, return `b · s`. Shrink b further (for example by half)
if it comes from in-sample predictions, since in-sample s overstates the fit.
For vol-scaled targets, multiply back by σ_it before the slope step.

```python
# inside fit, after training the model on a transformed target:
s = model.predict(X_inner_test)
b = float(np.dot(s, y_inner_test) / np.dot(s, s))   # no intercept
state = {"model": model, "b": max(b, 0.0)}
```

Why per-date demeaned forecasts still score on raw R²: write y = m_t + e_it.
If ŷ has mean zero on each date, the cross term between m_t and (e - ŷ)
vanishes within each date, so
`SSE = Σ n_t m_t² + Σ (e - ŷ)²`. All of the cross-sectional improvement
carries over to raw R². The common part stays unexplained, which caps the
score at about (1 - share_time) × the cross-sectional R². That is usually
fine: m_t has one observation per date, so forecasting it is a time-series
problem with a few hundred data points, and a noisy forecast of m_t adds
error to every row of that date. Add a market-timing term only if it has its
own CV evidence, and keep it small.

## 5. Stylised facts to test

Check these as hypotheses. In anonymised data, a feature may or may not be
one of these, and none is guaranteed to be present.

- **Short-term reversal and momentum.** Past returns over very short
  horizons tend to reverse in equities; past returns over longer windows
  (months, skipping the most recent period) tend to continue. Build lagged
  and summed past returns from `history` at several horizons and run 2.3
  on them.
- **Volatility clustering.** |y| and y² are autocorrelated per asset and in
  the market. This matters for vol-scaling and for regimes even when the sign
  of y is unpredictable.
- **Cross-asset lead-lag.** Lagged returns of the market, of larger or more
  liquid assets, or of related assets can predict the returns of others.
  Test lagged market return and lagged cross-sectional averages as features.
- **Size and liquidity.** Small and illiquid assets have higher volatility
  and different factor exposures. A persistent feature with a stable
  cross-sectional ranking may play this role; check whether other effects
  differ between its top and bottom halves.
- **Regime dependence.** Many effects are stronger or flip sign in
  high-volatility or falling markets (momentum crashes are the standard
  example). See 2.5.
- **Factor structure.** The first principal component of returns is usually
  the market. Removing it (or a per-date mean) before computing ICs and
  correlations sharpens most measurements.

Crypto panels in brief: trading is 24/7, so the period boundary is a
convention and there is no overnight gap. Tails are heavier than in equities,
and liquidation cascades produce extreme cross-asset co-movement. A single
market factor (BTC) often dominates the variance, so the common share in 2.1
is large and per-date demeaning matters more. Short-horizon reversal and
momentum exist but are less stable across periods.

## 6. Modelling and validation

- **Start simple.** Baseline: a cross-sectionally standardised (or
  Gaussian-ranked), winsorized feature set into ridge, with the target
  demeaned per date and the predictions rescaled as in section 4. Every more
  complex model must beat this on CV.
- **Shrinkage.** Prefer ridge to OLS. Pick the penalty from a small grid
  (3 to 5 values spread by factors of 10) on an inner time split inside `fit`,
  not by calling the scorer repeatedly.
- **Few knobs.** Each choice tested on the CV (feature set, transform,
  penalty, model family) is a trial. Decide most of them from EDA and theory,
  and use the scorer to compare a small number of candidates.
- **Walk-forward only.** Never shuffle dates. Inner validation inside `fit`
  should also be a time split, with the last part of the training dates held
  out. Leave a gap of a few dates if features or targets overlap in time.
- **Nonlinear models.** Gradient boosting with strong regularisation (shallow
  trees, depth 2 to 4, learning rate ≤ 0.05, row and column subsampling, early
  stopping on a time split) is the usual next step after a linear model. With
  low signal-to-noise it overfits easily; compare it to the linear baseline
  at matched effort.
- **Ensembling.** Averaging a few simple models (for example ridge on two
  feature transforms, plus a small boosted model) is usually more stable than
  any single one. Equal weights are a strong default; fitted weights add knobs.
- **Multiple testing.** If you tried N models, the best CV is biased upward.
  For noise models the expected maximum is roughly `sd_cv × sqrt(2 ln N)`,
  where sd_cv is the standard error of one CV estimate. Estimate sd_cv from
  the spread of the fold scores: `sd(folds) / sqrt(n_folds)`. A model that
  beats the baseline by less than about two of these, after N trials, has no
  demonstrated edge. Report N with any best score (see the deflated Sharpe
  ratio of Bailey and López de Prado, and Harvey, Liu and Zhu on t-stat
  hurdles, for the formal versions).
- **Fold consistency.** Prefer a model that beats the baseline in most folds
  to one with a higher mean driven by one fold.

## 7. Checklist before submitting a model

1. Variance decomposition done; target choice follows from it and from the
   tails of y.
2. Every feature is scaled, winsorized or ranked, with statistics from `fit`.
3. Penalised model on standardised inputs; penalty chosen on an inner time
   split.
4. Predictions mapped back to the scale of y and shrunk.
5. `predict` returns finite values for every row, including on dates with
   missing values or a single asset.
6. No feature uses information after date t.
7. CV beats the simple baseline in most folds, and the number of trials is
   reported with the score.
