# Forecaster: from a finding to a scored model

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

Examples, not the set of options. Combine them, adapt them, or build a transform that fits what the EDA showed.

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

The rows below are common starting points, not the choices. Targets can be combined (for example vol-scaled then clipped), shaped by a finding, or invented; pick by what the data shows and compare on CV against raw y.

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

Common families, not a shortlist; a hand-built model that encodes a finding is often better than any of them.

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

Examples meant to widen the space, not to be worked through; your own variants are welcome.

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
