#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp aliaslm-wikifacts \
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
    --rnn-dropout-pos both \
    \
    --kb-embed-dim 50 \
    \
    --path data/wikifacts/ \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --tbdir ~/tb/ \
    --model AliasLM \
    --logging-level DEBUG \
    --vocab-size 40000 \
    --vocab-dir data/vocab/aliaslm-wikifacts \
    --use-unk-probs \
    \
    $@
