#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp lrlm-wikitext-full \
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
    --alias-disamb-strategy fasttext \
    \
    --vocab-mlp-hidden-dim 500 \
    --vocab-mlp-dropout 0.1 \
    --vocab-mlp-activation none \
    --use-rel-mlp True \
    --rel-mlp-hidden-dim 500 \
    --rel-mlp-dropout 0.1 \
    --rel-mlp-activation none \
    --rnn-dropout-pos both \
    --adaptive-embed \
    --fasttext-model-path data/fasttext/wt-full.bin \
    \
    --path data/wikitext \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --model LRLM \
    --logging-level DEBUG \
    --vocab-size None \
    --min-freq 3 \
    --vocab-dir data/vocab/lrlm-wikitext-full/ \
    --tbdir ~/tb/ \
    --use-unk-probs \
    \
    $@
