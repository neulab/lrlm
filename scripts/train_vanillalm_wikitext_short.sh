#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp vanillalm-wikitext-short \
    --script $0 \
    --lr 1e-3 \
    --lr-scaler 0.9 \
    --batch-size 30 \
    --update-batch-size 60 \
    --bptt-size 150 \
    --num-epochs 50 \
    \
    --num-layers 2 \
    --embed-size 400 \
    --hidden-size 1000 \
    --vocab-mlp-hidden-dim 1000 \
    --vocab-mlp-dropout 0.5 \
    --dropout 0.5 \
    \
    --adaptive-embed \
    \
    --use-only-first-section \
    --path data/wikitext \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --tbdir ~/tb/ \
    --model VanillaLM \
    --logging-level DEBUG \
    --vocab-size None \
    --min-freq 3 \
    --vocab-dir data/vocab/vanillalm-wikitext-short \
    --use-unk-probs \
    \
    $@
