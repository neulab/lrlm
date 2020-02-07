#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp lrlm-wikitext-short \
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
    --kb-embed-dim 100 \
    --alias-disamb-strategy fasttext \
    \
    --vocab-mlp-hidden-dim 800 \
    --vocab-mlp-dropout 0.5 \
    --vocab-mlp-activation none \
    --use-rel-mlp True \
    --rel-mlp-hidden-dim 800 \
    --rel-mlp-dropout 0.5 \
    --rel-mlp-activation none \
    --rnn-dropout-pos both \
    --adaptive-embed \
    --fasttext-model-path data/fasttext/wt-short.bin \
    \
    --use-only-first-section \
    --path data/wikitext \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --model LRLM \
    --logging-level DEBUG \
    --vocab-size None \
    --min-freq 3 \
    --vocab-dir data/vocab/lrlm-wikitext-short/ \
    --tbdir ~/tb/ \
    --use-unk-probs \
    \
    $@
