#!/bin/bash
# Model x prompt version (v1, cur) x world (planted1b held out, planted2 control), Opus judge. Calls models (~$30).
set -u
cd "$(dirname "$0")/.."
OUT=runs/evals/ablation; mkdir -p $OUT
one() {  # model n parallel
  for w in planted1b planted2; do for v in v1 cur; do
    .venv/bin/arh eval all --spec evals/$w.json --model $1 --n $2 --parallel $3 --max-turns 15 \
      --prompts evals/prompts/$v --out $OUT/$1-$v-$w.jsonl > $OUT/$1-$v-$w.log 2>&1
  done; done
}
for m in claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5 claude-fable-5-1; do one $m 5 3 & done
wait
one gpt-oss-32k 3 4   # local, via Ollama
