#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp lrlm-transformer-wikifacts \
    --script $0 \
    --lr 0.00025 \
    --lr-scaler 0.9 \
    --batch-size 15 \
    --update-batch-size 60 \
    --bptt-size 150 \
    --memory-size 150 \
    --num-epochs 20 \
    --clip 0.25 \
    --optimizer-strategy none \
    --lr-scheduler cosine \
    --warm-up-steps 6000 \
    \
    --base-rnn transformer \
    --num-layers 16 \
    --embed-size 410 \
    --hidden-size 410 \
    --num-heads 10 \
    --head-dim 41 \
    --ffn-inner-dim 2100 \
    --attention-dropout 0.0 \
    --dropout 0.1 \
    --adaptive-embed \
    --tie-embed-weights \
    --rnn-dropout-pos both \
    --no-pre-lnorm \
    --alias-disamb-strategy oracle \
    \
    --kb-embed-dim 50 \
    --use-rel-mlp True \
    --rel-mlp-hidden-dim 500 \
    --rel-mlp-dropout 0.1 \
    --rel-mlp-activation none \
    \
    --path data/wikifacts \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --tbdir ~/tb/ \
    --model LRLM \
    --logging-level DEBUG \
    --vocab-size 40000 \
    --min-freq 0 \
    --vocab-dir data/vocab/lrlm-transformer-wikifacts \
    --use-unk-probs \
    \
    $@
