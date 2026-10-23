---
tools: Bash, score
writes: forecast
task: Run a full research session on this forecasting problem and stop only when the protocol says so.
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
the only score you get. A sealed holdout the team never sees guards against
selecting on CV luck, so do not ration your trials: comparing many genuinely
different ideas is the job. Ration only knob-twiddling of one idea.

## What to do

You have a long budget (dozens of turns). Use it. A run that stops after one
model has failed, however good that model's CV.

0. Before anything else, write `own_plan.md` in your working directory: how you,
   as a senior quant researcher, would attack a panel like this from your own
   knowledge. List at least eight structures that panels of asset returns
   commonly have, and for each the check in the data that would show it. Do not
   take them from the reference below; the reference is one colleague's partial
   notes, silent on much of what you know. Run every check in your plan during
   step 2, and say in the note which ones showed something.
1. `search_notes(kind="idea", open_only=true)` and read the findings and past
   forecasts. They are starting points, not the answer.
2. Look at the data before modelling: each feature's distribution, outliers,
   per-date scale, its relation to `y` (binned means, not only correlation),
   pairs of features, near-duplicates, regimes.
3. Before your first model, write down at least four different ways to pose
   the problem (what is predicted, with what loss, from what), each with the
   check in the data that would favour it, and run those checks. Your first
   trials should test the most different of these, not one of them refined.
   If an idea needs a library you do not have, call `request` with what you
   wanted, then go on with your fallback.
4. Iterate. Each round: state a hypothesis about the data in one sentence,
   build the model that tests it, `score_model` it with `relation`, `builds_on` and `why`:
   either a follow-up (which earlier trial or note it refines, and what the
   earlier result told you that makes this refinement worth it) or new (which
   existing ideas it differs from and in what information or structure, and
   what in the data suggested it).
   Write `why` and `learned` as real prose, a paragraph each, not a label: reason
   it through in your own words, as you would in a lab notebook. `learned` says
   what the last result meant, including a surprise or a failure, before you
   decide what comes next. If you cannot say why, do not score it.
   Keep a running table
   of every trial (hypothesis, CV). When a round improves on the best, ask why
   and push that direction; when it does not, ask what the failure says about
   the data and try something structurally different (other information,
   preprocessing, target or model form), not a different knob value.
5. At least once, try an idea that is in none of the notes and none of the
   reference below, built from something you saw in this data.
6. Your run ends hard at 60 turns, and a run without a note scores nothing. After about 40 turns,
   stop exploring and write the note.
7. Stop only when three structurally different ideas in a row have failed to
   improve the best CV, or you are running out of turns. Then write ONE
   `add_note(kind="forecast", artifact="model.py", ...)` for the best model:
   what it does, `refs`, its CV, the full trial table, and what you would try
   next. The harness rescores it.


## Reference: a map of quant research practice

One colleague's partial notes, not a complete method and not instructions. Where they are silent, your own knowledge is the reference; your `own_plan.md` comes first.


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



Read `general.md` first. This file is about turning what the analysts saw
into a model that scores on walk-forward R² on raw y, and about the
discipline that keeps the score honest. There is no default model here and
no first candidate. The model is whatever the findings imply, built one
feature at a time.

## The contract

`fit(train)` runs once per fold on all rows with y before the test block.
`predict(state, history, today)` runs once per date; `history` is every row
with y before that date, `today` is one date's rows without y. A fold has a
time budget of a couple of minutes, so heavy work belongs in `fit`. A feature
built from `history` must be computed the same way in `fit`, with the window
ending at the previous date. `predict` must return a finite float for every
row of `today`, including dates with a single asset, nulls or a constant
feature. Every scored model is a trial that counts toward N.

## Starting from the findings, not from a template

First ask what the findings say about where the forecastable part of y sits.
If the common component is forecastable, per-date operations on the target
or on features destroy signal and zero out date-constant features; if it is
large and unforecastable, it is noise that a per-date treatment of the target
removes; if both dimensions carry signal, the two can be forecast separately
and added. This decides which treatments are admissible, and it is a question
about y, not about the features.

Then go feature by feature. For each feature the model will use, the
findings should have said something about its tails, its drift, the shape of
its relation and whether its effect depends on a state; design that
feature's treatment from that. A feature whose tails carry signal keeps
them, perhaps in raw units; one whose tails are noise has them trimmed or
ranked; one whose level drifts enters as a per-date or differenced quantity
while a stable one enters as a level; one with a confirmed non-linear shape
enters through a transform that follows the shape you saw; one whose slope
depends on a state enters together with that state. Two features in the same
model can and often should receive different treatments, and applying one
transform to every column because it was easy is the error this file exists
to prevent. A transform you build to match a finding beats a standard one
that approximately fits; combining treatments and inventing new ones is
welcome. A penalised model penalises in each feature's units, so features on
different scales get different effective penalties unless you make them
comparable; tree models do not need that but overfit easily at low
signal-to-noise.

The model family is also a consequence of the findings. A penalised linear
model suits relations that are mostly monotone and already shaped per
feature; a robust loss or a trimmed target suits fat-tailed y whose tails are
not predictable; a boosted model with early stopping on an inner time split
suits non-linear or interactive findings you could not express by hand, and
must be compared against a linear model at matched effort; models per asset
or group are justified only when the heterogeneity test says slopes differ.
Averaging models that differ in treatment or family is often more stable
than any one, because their errors differ.

## Target and mapping back

The score is on raw y; the fit may use another target. Removing the per-date
mean, ranking within a date, trimming at training quantiles or dividing by a
volatility estimate from history each change what the model learns and
ignores; each is a hypothesis to compare against raw y on CV, chosen from
what the findings said about m_t and the tails of y. A model trained on a
transformed target produces a score on the wrong scale. The map back is the
no-intercept regression of raw y on that score, fitted on an inner time split
at the end of the training dates and stored in the state; that slope is the
R²-optimal scale on that split. A negative slope means no usable signal; a
slope that falls from inner split to CV means the model overfits, and the
remedy is in the model, not the scale. For a volatility-scaled target,
multiply by the volatility before the slope step. With raw y as target, a
slope well below one says the model is overconfident.

## The question you ask of the data

Regression on y is one way to pose the problem, not the only one, and the
data often says another way is easier. Ask which part of y is predictable.
Sometimes it is the order of assets within a date and not the size of the
gaps: the rank correlation of a feature with y is steady while its R² is
tiny and its slope swings with the market's volatility. Then a model of the
rank (a regression on per-date ranks of y, a pairwise or listwise ranking
objective, a model of the probability that one asset beats another) learns
the stable part and leaves the unstable part to the map back. Sometimes it
is the sign: binned means of y by a feature stay flat while the share of
positive y moves, because the size of a move is fat-tailed noise and only
its direction responds. A classifier of y > 0, or of y above the date's
median, then sees a signal that squared error drowns, and the forecast is the
predicted probability, centred and scaled back to y. Sometimes it is a tail:
the feature moves the 10th or 90th percentile of y and not its centre, which
quantile regression shows and least squares averages away. Sometimes it is a
rare event: most rows carry nothing and a few carry a large, predictable
move, so a model of whether the event happens and one of its size beat one
model of both. Sometimes it differs by asset or by group, and a pooled model
blurs assets whose relations have opposite sign. The test in every case is
the same: look at the data through that lens first (rank correlations per
date, the share of positive y in bins, quantiles in bins, the distribution of
y's size), and pose the problem the way the data is most regular. The
harness scores every trial on R² and also reports the rank IC and the
per-date Sharpe of a top-minus-bottom quintile spread; a model of order or
sign usually shows its worth there first, and needs the map back to show it
in R².

## Validation discipline

Every knob, whether penalty strength, trimming point, window length, mapping
slope or ensemble weight, is chosen on an inner time split inside `fit`,
never on shuffled rows and never on the scored folds. Change one thing per
scored trial so the result is attributable. Report fold scores, not only the
mean; beating the baseline in most blocks is the stronger result, and the
trial count belongs next to the score. Before writing the forecast note, ask
whether the target and each feature's treatment followed from a finding,
whether every statistic is computed inside `fit`, whether the prediction is
on y's scale and finite everywhere, and whether the note says what was tried
and what to try next.

## Moves worth knowing about

Experienced forecasters also build features from `history`, such as past
returns and their spread per asset or the previous date's dispersion;
interact a signal with a date-level state; residualise a feature against a
correlated one when its IC is borrowed; train on a multi-period target and
map back when the signal decays slowly; use sample weights so coefficients
are estimated efficiently while the score stays on raw y; and check what the
inner-split residuals still correlate with. None of these is a step. Each is
a possibility a specific finding might make worth testing, and the version
that fits this data is likely a variant you build yourself.


