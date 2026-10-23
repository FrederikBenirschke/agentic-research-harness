# Forecaster: from findings to a scored model

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
