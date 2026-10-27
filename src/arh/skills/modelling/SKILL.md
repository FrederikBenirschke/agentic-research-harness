---
name: modelling
description: How to build and score models honestly. Read it when you are about to write or score a model.
---
# Building models

- Start simple, so that you can tell what each later change bought.
- Before you score a model, decide what result would mean what. A score you
  had no expectation for teaches little.
- Compare against the simplest reasonable alternative, not only against
  your previous model.
- Use only the past to predict the future: anything computed over the whole
  panel (a scale, a rank, a mean) leaks the future unless it is computed per
  date or from history.
- Every scored model is a trial. Many trials of small variations make the
  best CV look better than it is; a few trials of different ideas tell you
  more.
- When a model stops improving, more tuning rarely helps. Find out where it
  is wrong and why, then go back to the data.
- The first model that works is a starting point. If you have turns left,
  ask what it is still missing.
