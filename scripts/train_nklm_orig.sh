#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp nklm-orig-wikifacts \
    --script $0 \
    --lr 0.5 \
    --lr-decay 0.02 \
    --lr-scaler 1.0 \
    --optimizer sgd \
    --optimizer-strategy none \
    --clip 5.0 \
    --batch-size 50 \
    --bptt-size 30 \
    --num-epochs 50 \
    \
    --num-layers 2 \
    --embed-size 400 \
    --hidden-size 1000 \
    --dropout 0.5 \
    --pos-embed-dim 40 \
    --pos-embed-count 20 \
    --kb-embed-dim 50 \
    \
    --entity-disamb-strategy entvec \
    --alias-disamb-strategy oracle \
    --vocab-mlp-activation relu \
    --rnn-dropout-pos both \
    \
    --use-anchor \
    --path data/wikifacts_orig \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --model NKLM \
    --logging-level DEBUG \
    --vocab-size 40000 \
    --vocab-dir data/vocab/nklm-orig-wikifacts \
    --tbdir ~/tb/ \
    \
    $@
