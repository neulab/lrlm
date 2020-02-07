#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp vanillalm-wikifacts \
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
    \
    --path data/wikifacts/ \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --tbdir ~/tb/ \
    --model VanillaLM \
    --logging-level DEBUG \
    --vocab-size 40000 \
    --vocab-dir data/vocab/vanillalm-wikifacts \
    --use-unk-probs \
    \
    $@
