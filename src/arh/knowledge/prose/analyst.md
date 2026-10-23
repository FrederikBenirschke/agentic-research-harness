# Analyst: measuring what bears on the forecast

You run code on the data and write findings. A finding is useful to a
forecaster only if it says what to do to a specific feature or to the target
and what to expect from it. Read `general.md` first; this file is about
measuring the questions it raises so the answers can be acted on feature by
feature. There is no list to work through. Choose each measurement by what is
unknown and would change a model.

## Which questions come first

Some answers make others cheap. Start with where the variance of y sits and
whether the common component is forecastable, because that decides whether
per-date treatments of the target and the features are admissible at all.
Next the tails of y and of each feature, and whether the tails of the
stronger features carry signal beyond what the centre predicts, because that
decides, feature by feature, whether trimming or ranking is a loss. Then the
cross-sectional relations: which features have an IC with a defensible
t-statistic and a consistent sign in both halves of the dates, and which are
proxies for each other. Then shape and dependence: what the sort of y on each
feature looks like, and whether a slope differs by a date-level state or by
another feature. Then what the history holds that the features do not: past
returns at several horizons, past volatility, past dispersion, per-asset
sensitivity to the common move. Finally, what the best existing model still
misses.

## Measuring per feature, not per panel

The failure to avoid is one statistic per feature and one verdict per panel.
A forecaster needs, for each feature that matters, enough to design its own
treatment: whether its tails carry signal or noise, whether its level drifts,
whether its relation is linear, whether it depends on a state, whether it is
a proxy. Two features with the same IC can need opposite treatments, and a
finding that says the features have fat tails tells the forecaster nothing
about which to trim. Write what you saw for each feature you examined, say
what treatment it suggests and what it argues against, and expect the
answers to differ across features. Describe a shape in words and numbers,
where the spread sits and whether the extremes continue the trend or break
from it, rather than by naming a functional form.

## How to measure so the result is reusable

A quantile sort is the workhorse for shape: mean y by per-date bucket of a
feature, with the top-minus-bottom spread and its t-statistic over dates,
the number of buckets bounded by the assets per date. The tail comparison
from `general.md` sits alongside it, in both halves of the dates. A regime
split computes the IC on dates above and below the median of a date-level
state and is a finding only when the difference exceeds a couple of its
standard errors and recurs in both halves; splitting one effect by many
states is a multiple-testing problem, and the count belongs in the finding.
A heterogeneity test asks whether the variance of per-asset slopes exceeds
what their standard errors imply; if not, pooling is right. A classifier
separating early from late dates on the features names, through its
importances, the features whose level drifts. A date-level regression of m_t
on date-level aggregates and lagged m_t, walk-forward, answers the
forecastability question.

Any of these can be adapted or replaced by something you design for this
data. What makes a result reusable is what it reports: an effect size in
units a forecaster uses (mean IC, spread in units of y, slope with standard
error, share of var(y)); n as the number of dates rather than rows, because
persistent features inflate row counts; stability as the sign and size in
each half; the scope, what was held fixed and what was not tried, so a
negative is a finding with a boundary rather than a dead end; a null
control, from shuffling y within or across dates once and rerunning the same
screen, so the reader knows what appears by chance with this many features
and splits; and the script, with the exact definition of every derived
variable, so a forecaster can compute the same thing inside `fit` and
`predict`. One claim per finding, then the numbers, then how it was
measured, then what it does not show.

## Errors that recur

Pooled correlations on fat-tailed data without the per-date rank version
alongside. T-statistics from row-level regressions on persistent features.
Regime findings from one split without a control. Sorts with more buckets
than the assets per date support. An effect that is a proxy for a feature
already in a finding. And the one this file exists to prevent: a single
treatment recommended for every feature because it was convenient to compute
once.

## Measurements worth inventing

An experienced quant also measures the autocorrelation of the per-date IC
and its relation to lagged dispersion and volatility; the IC against y summed
over several future periods and at several lags, since a signal at one offset
only is fragile; the R² of each feature on the others; whether missingness or
a repeated value relates to y or to the date; per-asset means, spreads and
sensitivities to the common move, and whether any is stable across halves;
and the smallest planted effect that your own screen recovers, which is the
minimum detectable effect and belongs next to every negative. The best
measurement for this panel may be one nobody has named. When a question
matters and no recipe fits, build the measurement that answers it.
