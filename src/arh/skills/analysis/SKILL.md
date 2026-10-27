---
name: analysis
description: How to explore data so that what you find is real. Read it when you are about to study the data, or think you have found something.
---
# Exploring the data

- Start from a question, not a checklist: what would have to be true about
  this data for `y` to be predictable, and what would you see if it were?
- A summary number averages over all the data. A small number can hide an
  effect that is strong in part of the data or in a form the number does
  not measure. When a number is small, ask what it averages over before
  concluding there is nothing.
- Look at the data directly as well as through statistics: binned means,
  tables of counts, a few rows.
- The best guide to what to look at next is what your current model gets
  wrong. Its errors are data too: compute them out of sample and explore
  them the way you explored `y`.
- Make a finding earn its place: report the effect size, check it in both
  halves of the time range, and remember that the more things you try, the
  more of them look good by chance.
- Write down what you measured with the script that measured it, so the
  result can be rerun.
