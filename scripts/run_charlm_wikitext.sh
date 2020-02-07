#!/usr/bin/env bash
# This script trains a character model AND evaluate the whole dataset to dump word probabilities.
# Exactly the same hyperparameters for both WT and WF.
# Move the <exp_dir>/unk_probs.txt to the data location after this script is finished.

python run_charlm.py \
    --exp charlm-wikitext \
    --script $0 \
    --lr 5e-4 \
    --batch-size 30 \
    --update-batch-size 60 \
    --bptt-size 140 \
    --num-epochs 20 \
    \
    --num-layers 2 \
    --embed-size 256 \
    --hidden-size 512 \
    --dropout 0.1 \
    --vocab-mlp-activation none \
    --vocab-mlp-hidden-dim 256 \
    --vocab-mlp-dropout 0.1 \
    \
    --path data/wikitext \
    --mode train \
    --cuda \
    --checkpoint-interval -1 \
    --model VanillaLM \
    --vocab-dir data/vocab/charlm-wikitext/ \
    --logging-level DEBUG \
    --no-use-unk-probs \
    --tbdir ~/tb/ \
    --dump-probs \
    \
    $@
