#!/usr/bin/env bash
set -e

python3 run_full_eval.py \
    --config config/config.json \
    --prompts data/prompts_200.jsonl \
    --alpha 0.5 \
    --bestof 8 \
    --seqsteps 5