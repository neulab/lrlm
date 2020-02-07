#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp nklm-wikifacts \
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
    --dropout 0.5 \
    --pos-embed-dim 40 \
    --pos-embed-count 20 \
    --kb-embed-dim 50 \
    \
    --vocab-mlp-activation none \
    --rnn-dropout-pos both \
    \
    --path data/wikifacts \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --model NKLM \
    --logging-level DEBUG \
    --vocab-size 40000 \
    --vocab-dir data/vocab/nklm-wikifacts \
    --tbdir ~/tb/ \
    --use-unk-probs \
    \
    $@
