#!/usr/bin/env bash

SCRIPT_NAME=`basename "$0"`

python run.py \
    --exp nklm-wikitext-short \
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
    --kb-embed-dim 100 \
    --adaptive-embed \
    \
    --vocab-mlp-activation none \
    --rnn-dropout-pos both \
    --alias-disamb-strategy fasttext \
    --fasttext-model-path data/fasttext/wt-short.bin \
    \
    --use-only-first-section \
    --path data/wikitext \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --model NKLM \
    --logging-level DEBUG \
    --vocab-size none \
    --min-freq 3 \
    --vocab-dir data/vocab/nklm-wikitext-short \
    --tbdir ~/tb/ \
    --use-unk-probs \
    \
    $@
