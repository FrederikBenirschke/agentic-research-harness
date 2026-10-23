# Return-panel research: shared background

Background for forecasting the next-period return of each asset in a panel
of dates and assets, from features known at the close of each date. It is
organised around questions the data can pose, because the data decides
everything that matters. Nothing here is a procedure, a default or a
template. There is no such thing as "the" transform for the features of a
panel: each feature may need its own treatment, and the right one for a
feature is often wrong for its neighbour. Look at each feature, say what you
saw, and build the treatment that fits what you saw, inventing it if nothing
you know fits. Remedies appear below in passing as possibilities; none is
the answer, and a model should be the consequence of the EDA, never a
template the EDA was fitted around.

Notation: date t, asset i, features known at the close of t, target y_it the
return of asset i over the period after t. Per-feature statistics are in the
summary note; the data README describes the columns.

## What the score rewards

The score is walk-forward R² on raw y, pooled over the rows of a test block:
one minus the sum of squared errors over the sum of squared deviations of y
from its mean. Realistic values are small; in real return panels a few
percent is good, and a large number is more often a leak or a lucky trial
than a discovery. Scale matters as much as direction: if a signal has
correlation ρ with y and the forecast is k times the regression fit on it,
R² is ρ²(2k − k²), so it is ρ² at k = 1, zero at k = 2 and negative beyond.
Because the error is squared and pooled, it is dominated by the largest |y|,
the high-volatility assets and dates; a model that fits the typical row and
is wrong in the tails scores badly, and fat-tailed y makes every fold noisy.
Each fold is one block of consecutive dates with the model fitted on
everything before it; the spread across folds is the noise in the score.

## Questions the data can pose

The questions are generic. The answers are not, and they differ from feature
to feature within one panel.

**Where does the variance of y sit, and is the common part forecastable?**
Write y_it as a per-date mean m_t plus a deviation e_it. The share of var(y)
in m_t says how much of y is a common move; whether that share is
forecastable is a separate question, answered by a per-date table of the
mean of y and date-level aggregates of the features, including lagged m_t,
and by asking what predicts m_t with the same sign in both halves of the
dates, walk-forward. If m_t is forecastable, any per-date operation on the
target and any per-date standardisation of a feature throws that signal
away, and a feature constant within each date becomes exactly zero. If m_t
is large but not forecastable, it is noise in every cross-sectional
coefficient, and removing it from the target can help; a forecast with zero
mean on each date carries its cross-sectional skill over to raw R², capped
at the non-common share of variance. If both dimensions carry signal, each
can be forecast on its own and the two added. The same decomposition applies
to each feature: one whose per-date mean moves a lot mixes a timing role
with a ranking role, and its cross-sectional version is a different variable
from its level. Decide per feature which you want, and why.

**What does the cross-section say, and does the pooled correlation agree?**
The information coefficient of a feature is its per-date rank correlation
with y; its mean over dates, its spread over dates, the t-statistic of the
mean and the fraction of dates with the majority sign summarise the ranking
relation while ignoring m_t and the tails. The pooled Pearson correlation
includes the date dimension and is pulled by extreme rows, so it measures
something else. When the two disagree in size or sign, the gap is itself a
finding: the relation lives in the tails or in the date dimension, and a
rank-based treatment would change or lose it.

**Do a feature's extreme values carry no information, or most of it?** The
question most often answered by habit. For each feature, take the rows in
its extreme per-date quantiles and compare mean y there with what a linear
fit on the centre extrapolates in raw units, and with what the trend of a
quantile sort in bucket number predicts. Tails that match or exceed the
extrapolation mean trimming removes signal; tails that fall short or flip
sign mean trimming removes noise; extreme buckets that jump beyond the sort's
trend mean ranking loses signal. Different features in one panel give
different answers, which is why one trimming rule for all of them is a
mistake. Winsorising at a training quantile, ranking within a date, a smooth
compression of large values, or leaving the feature raw are all
possibilities; which, and at what point, is a per-feature decision from this
comparison, confirmed on CV.

**Do features differ wildly in scale, and does a feature's scale change over
time?** Standard deviations spanning orders of magnitude mean a penalised
model on raw units penalises features arbitrarily. The per-date mean and
spread of each feature over time show which features drift, and a drifting
feature is not comparable across dates in raw form. A scaling from training
statistics keeps levels, a per-date scaling removes them, a differenced form
keeps the change and drops the level: different variables, and the one that
suits a drifting feature may be wrong for a stable one in the same model.

**Is a feature's effect linear?** A quantile sort, mean y by per-date bucket,
shows the shape one correlation hides: monotone or not, where the spread
sits, whether the middle is flat. The slope within each bucket says it more
sharply, and a nonlinear dependence measure compared with the rank IC points
at relations that are not monotone. Shapes found in one half of the dates
need confirming in the other. A confirmed shape calls for a transform of
that feature that follows what you saw, not a switch of the whole model to a
family that might find it.

**Does an effect depend on another variable?** The slope of y on a feature
may differ between dates with high and low volatility or dispersion, between
levels of a date-constant feature, or between the halves of another
cross-sectional feature. A binned two-way table of y on the pair, or the IC
split by a date-level state, shows it. The difference must exceed a couple of
its standard errors, recur in both halves, and be read against the number of
pairs and states screened. When it holds, that one feature needs the state
built into its treatment; it is not a reason to fit everything separately.

**Are two features the same thing?** High pairwise correlation marks proxies;
the one that keeps its effect when the other is included carries the
information. A feature with high per-asset autocorrelation has far fewer
independent observations than rows, so row-level standard errors overstate
significance while per-date statistics do not.

## Leakage and walk-forward discipline

A feature with an IC far above everything else deserves suspicion before
admiration. The latest realised return usable on date t is y from the
previous date, since y_t is realised after t; check the alignment by hand on
a few rows. Every statistic the model uses (means, spreads, trimming bounds,
loadings, selected features, mapping slopes) comes from training rows only
and is computed inside fit so it is redone per fold; anything computed once
on the whole file leaks later folds into earlier ones. Per-date transforms
use only that date's features and are safe by construction. Asset identity
as a feature memorises per-asset means, which rarely persist.

## Multiple testing

Every scored model and every screened feature is a trial. Among N noise
trials the expected best is roughly the standard error times the square root
of 2 ln N, so beating the baseline by less than a couple of standard errors
after many trials shows nothing. Report N with any best score, make most
decisions from EDA and reasoning, and prefer beating the baseline in most
folds over a higher mean.

## Mapping a transformed target back

The score is on raw y, but a model may be fitted on a transformed target,
whose output is then on the wrong scale. Regress raw y on the output over an
inner time split of the training dates, store the slope, apply it in
prediction. A negative slope means no usable signal; a slope that shrinks
from inner split to CV means the model overfits, and the model rather than
the scale is the problem. For a volatility-scaled target, undo the scaling
first. An inner split, throughout, means a time split inside fit used to
choose knobs without spending scored trials.
