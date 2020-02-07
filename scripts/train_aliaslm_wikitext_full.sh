#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp aliaslm-wikitext-full \
    --script $0 \
    --lr 1e-3 \
    --lr-scaler 0.9 \
    --batch-size 30 \
    --update-batch-size 60 \
    --bptt-size 150 \
    --num-epochs 50 \
    \
    --num-layers 4 \
    --embed-size 512 \
    --hidden-size 1024 \
    --dropout 0.1 \
    --kb-embed-dim 100 \
    --vocab-mlp-dropout 0.1 \
    --vocab-mlp-hidden-dim 500 \
    \
    --adaptive-embed \
    --rnn-dropout-pos both \
    \
    --path data/wikitext \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --model AliasLM \
    --logging-level DEBUG \
    --vocab-size None \
    --min-freq 3 \
    --vocab-dir data/vocab/aliaslm-wikitext-full/ \
    --tbdir ~/tb/ \
    --use-unk-probs \
    \
