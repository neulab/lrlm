## Preprocess data to obtain required data for the character model

This directory contains the following file:

- `prepare_charlm_data.py`  is the script to extract unique vocabulary over the given dataset and form
another dataset for the character model to train. Use like below:

```sh
python prepare_charlm_data.py \
    --data-dir data/lm \
    --split 60 20 20 \  # Proportion of train/valid/test splits
    --output-dir data/char
```
