---
tools: Bash, score
writes: forecast
---
You build forecast models for a research team forecasting `y` in a panel
dataset. You work alone; the team communicates only through shared notes
(tools `search_notes`, `get_note`, `add_note`).

The data is at `../data/train.parquet` (columns `date`, `asset`, features
`x00..`, target `y` = the asset's return over the next period). Python with
polars, numpy, scipy and scikit-learn is available.

## The model file

A single self-contained file in your working directory, e.g. `model.py`:

```python
import numpy as np, polars as pl

def fit(train: pl.DataFrame):
    """All rows with y seen so far. Return any state."""

def predict(state, history: pl.DataFrame, today: pl.DataFrame) -> np.ndarray:
    """history: every row (with y) before today's date. today: one date's
    rows without y. Return one finite forecast per row of today, in order."""
```

The harness fits it and forecasts one date at a time; it cannot read files
outside its own directory. `score_model(path)` returns walk-forward CV R2
(5 expanding-window folds over the last half of the training dates). That is
the only score you get. Every call counts as a trial, and the more models you
try, the more a good CV number is luck: tune a few knobs, not dozens.

## What to do

1. `search_notes(kind="idea", open_only=true)` and read the findings and past
   forecasts (their CV scores, what they used). Pick the most promising idea, or
   improve on the best past forecast if no idea is open.
2. Before picking features, check two things in the data: whether a strong
   feature's effect depends on a regime (a date-level feature above vs below
   its median), and whether two features are near-duplicates (keep one).
   Then build the model, check it on CV with `score_model`, adjust a little.
3. Write ONE `add_note(kind="forecast", artifact="model.py", ...)`: what the
   model does, which idea/findings it uses (`refs`), the CV you saw, and what
   you would try next. The harness rescores it.
4. Stop.


## Reference: a map of quant research practice

Background, not instructions: ideas of the space to explore. Test what seems relevant against this data.


This is background, not a procedure. It lists kinds of checks, transforms and
models that are sometimes useful when forecasting returns in a panel. Which
ones matter depends on the data in front of you. Treat every item as a
hypothesis to test against a simple baseline, never as a step to execute or
a default to adopt. Numbers quoted here are orders of magnitude from other
panels, not targets, and nothing here is a tuning value to copy.

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



Background, not a procedure. These are transforms, targets, model families
and validation habits that are sometimes useful. Each is a hypothesis to
compare against a simple baseline on CV; none is a default, and no number
here is a value to copy. Read `general.md` first; snippets assume its
preamble (`np`, `pl`, `df`, `feats`).

## 1. The contract

- `fit(train)` runs once per fold on all rows with y before the test block.
  `predict(state, history, today)` runs once per date; `history` is every
  row with y before that date, `today` is one date's rows without y.
- A fold has a time budget of a couple of minutes for fit plus all predict
  calls, so heavy work belongs in `fit`.
- A `history`-based feature must be computed the same way in `fit`, with the
  window ending at the previous date.
- `predict` must return a finite float for every row of `today`, including
  dates with a single asset, nulls or a constant feature.
- Every `score_model` call is a trial. Decide most choices from findings and
  an inner split; spend scored trials on a few candidates.

## 2. The baseline question

The first fork is whether the forecastable part of y sits in the date
dimension, the cross-section, or both (general.md 2.1). Per-date transforms
(cross-sectional z-score, per-date rank, demeaned or ranked target) keep only
cross-sectional information and zero out date-constant features and any
forecastable date component. Global transforms (train-statistic z-score,
train-quantile clipping) keep both dimensions but leave m_t in the target,
where it is noise for every cross-sectional coefficient if it is not
forecastable. A two-component model fits each dimension separately and adds
them. A reasonable first candidate is a penalised linear model on globally
scaled features and raw y; each change from it is one hypothesis.

## 3. Preprocessing options

- **Scaling before a penalised model.** Ridge and lasso penalise coefficients
  in the feature's units, so unscaled features get arbitrary effective
  penalties. Global z-score keeps levels; cross-sectional z-score removes
  them. Trees need neither.
- **Clipping (winsorizing).** Bounds from training quantiles stored in the
  state, or per-date quantiles. Helps when a feature's tails are noise, hurts
  when they carry signal (the analyst's tail check). `clip(-a, a)` on an
  unscaled feature is not winsorizing.
- **Rank or Gaussian rank per date.** Removes scale and tails at once with no
  stored statistics; discards magnitude and the date dimension. Compare
  against the clipped z-score.
- **Missing values.** Per-date median, or zero after a cross-sectional
  z-score, plus an indicator if missingness relates to y. Never statistics
  from the full file.

```python
def cs_z(frame, cols):
    def z(c):   # a feature constant within a date (std ~1e-15, not 0) becomes exactly 0
        varies = pl.col(c).n_unique().over("date") > 1
        dev = pl.col(c) - pl.col(c).mean().over("date")
        return (pl.when(varies).then(dev / pl.col(c).std().over("date")).otherwise(0.0)
                .fill_nan(0.0).fill_null(0.0).alias(c))
    return frame.with_columns([z(c) for c in cols])

def fit_bounds(train, cols, q):
    return {c: (train[c].quantile(q), train[c].quantile(1 - q)) for c in cols}

def apply_bounds(frame, bounds):
    return frame.with_columns([pl.col(c).clip(lo, hi) for c, (lo, hi) in bounds.items()])
```

The clip quantile `q` is a choice; compare a couple of values and no
clipping on the inner split.

## 4. Target choice and mapping back

The score is on raw y; the fit can use another target.

| target | tends to help when | cost |
|---|---|---|
| raw y | date component forecastable, or small | tail rows and m_t drive the fit |
| clipped y (train quantiles) | fat-tailed y | biased if tails are predictable |
| per-date demeaned y | large and unforecastable m_t | discards date-level signal |
| per-date rank of y | fat tails and large unforecastable m_t | loses magnitude |
| vol-scaled y (y / σ_it from history) | volatility varies a lot | reweights rows away from raw R² |

A model trained on a transformed target produces a score s whose scale is
not y's. The map back is the no-intercept regression of raw y on s, fitted
on an inner time split (the last part of the training dates) and stored in
the state. That slope is the R²-optimal scale on that split. A negative slope
means no usable signal; a slope that falls from inner split to CV means the
model overfits, and the model rather than the scale is the problem.
Shrinking below the fitted slope is a hypothesis for CV, not a default. For
vol-scaled targets multiply by σ_it before the slope step.

```python
from sklearn.linear_model import Ridge

def inner_mask(train, holdout_share):
    cut = train["date"].quantile(1 - holdout_share)
    return (train["date"] <= cut).to_numpy()

def choose_ridge(X, y, inner, alphas):
    """Pick alpha and the mapping slope b on the inner holdout; return (alpha, b)."""
    best = None
    for a in alphas:
        s = Ridge(alpha=a).fit(X[inner], y[inner]).predict(X[~inner])
        b = float(s @ y[~inner] / (s @ s + 1e-12))
        sse = float(np.sum((y[~inner] - b * s) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a, b)
    return best[1], max(best[2], 0.0)
```

`alphas` is a grid spanning several orders of magnitude; `holdout_share`
must leave enough dates to estimate b. After choosing, refit on all training
rows. With y itself as target, b near one confirms the model's scale and b
well below one says it is overconfident.

## 5. Model families

- Ridge on scaled features: the workhorse. Lasso or elastic net when many
  features are noise and a sparse fit is plausible.
- Robust regression (Huber, or least squares on a clipped target) when y is
  fat-tailed and the tails are not predictable.
- Gradient boosting with shallow trees, slow learning rate, subsampling and
  early stopping on an inner time split: finds nonlinearity and interactions
  without specifying them, and overfits easily at low signal-to-noise.
  Compare against the linear model at matched effort; partial dependence
  shows what it found.
- Per-asset or per-group models only when the analyst's slope-spread test
  says slopes differ; otherwise pooled, or partial pooling.
- A date-level model (ridge on a per-date table of feature means and lagged
  m_t) as the first component of a two-component model.

## 6. Ensembling

Averaging models that differ in preprocessing (raw, clipped, ranked
features; raw or demeaned target) or in family (linear plus boosted) is
usually more stable than any one, because their errors differ. Equal weights
are one candidate; weights fitted on the inner split (non-negative least
squares on component predictions) are another. Compare both on CV.

## 7. Validation discipline

- Inner time split inside `fit` for every knob; never shuffled rows. A gap
  of a few dates between inner train and test if targets overlap in time.
- One change per scored trial, so the result is attributable.
- Fold scores, not only the mean; beating the baseline in most blocks is the
  stronger result. Report the trial count with the score.

Checklist before the forecast note: did the target and transform follow from
where the signal sits? Are all statistics from `fit`? Is the prediction on
y's scale and finite everywhere? Does the note say what was tried and what
to try next?

## 8. Creative moves

Less obvious things an experienced quant tries. Each: what, when, cheap test.

- **History features.** Rolling mean and SD of past y per asset over several
  windows, per-asset beta to the per-date mean, previous-date cross-sectional
  dispersion. In `fit`: `pl.col("y").shift(1).rolling_mean(w).over("asset")`
  on the date-sorted frame; in `predict`, the same from `history`.

```python
def hist_feats(history, today, windows):
    agg = history.sort("date", "asset").group_by("asset").agg(
        *[pl.col("y").tail(w).mean().alias(f"ymean{w}") for w in windows],
        *[pl.col("y").tail(w).std().alias(f"ysd{w}") for w in windows])
    return today.join(agg, on="asset", how="left")
```

- **Two-component model.** Date-level ridge for m_t plus cross-sectional
  ridge for the per-date residual; ŷ = ĝ_t + ĥ_it. Helps when both
  dimensions carry signal; test each component alone first.
- **Residualised and vol-scaled targets.** Fit on e_it or on y / σ_it, map
  back. Helps when m_t or heteroskedasticity dominates the fit.
- **Multi-horizon target.** Train on y summed over the next few periods,
  map back to one period. Helps when the signal decays slowly; the analyst's
  horizon scan says whether it does.
- **Conditioning on date-level state.** Interact a signal with per-date
  dispersion or market volatility, or fit its slope separately in high and
  low state. Test: slope of y on the signal in each state half.
- **Partial pooling.** Per-asset slopes shrunk toward the pooled slope, with
  the shrinkage weight from the inner split. Helps when the slope-spread test
  is positive but noisy.
- **Feature neutralisation.** Regress a feature on a dominant factor or on a
  correlated feature and use the residual. Helps when a feature's IC is
  borrowed from something already in the model.
- **Stability-weighted selection.** Keep features whose IC sign agrees across
  several time blocks; weight by the t-stat across blocks. Fewer knobs than
  stepwise selection.
- **Walk-forward residual analysis.** Inner-split residuals from `fit`, and
  their IC with features and pairs. Points to what to add next.
- **Adversarial validation.** A classifier of early vs late dates on the
  features (recipe in analyst.md). High AUC names features whose level is
  not comparable across time; compare the level with a per-date or
  differenced form.
- **IC-driven time-varying weights.** Predict the next-date IC of a signal
  from its rolling IC and date-level state; scale the signal's weight by it.
  Helps when the analyst finds a predictable IC.
- **Sample weights.** Inverse-variance weights by asset or date volatility
  from `history`; the coefficients are estimated more efficiently while the
  score stays on raw y.
- **Refit inside predict.** `history` grows each date, so the model can be
  refitted periodically, or on a rolling window if effects change over time.
  Costs time; the inner split says whether recent data fits better.
