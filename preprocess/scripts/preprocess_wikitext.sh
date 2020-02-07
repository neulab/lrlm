#!/usr/bin/env bash
set -euo pipefail

############################ MODIFY THESE ACCORDINGLY #################################
WIKITEXT_DIR=                 # Path to the unzipped wikitext-raw dataset.
WIKIDATA_DUMP_DIR=            # Path to the directory containing `latest-truthy.nt` file.
ALIGNED_DATA_DIR='./dataset'  # Target path to dump the aligned results between Wikidata and WikiText

# Additionally, don't forget to modify the path in `preprocess/process_wikitext.py`.
#######################################################################################

# Split wikitext into articles.
python preprocess/wikitext/split_wikitext.py \
    --wikitext-dir $WIKITEXT_DIR \
    --ids-dir data/found_ids \
    --output-dir $WIKITEXT_DIR

echo "Articles are splitted into $WIKITEXT_DIR" >&2

# Process raw Wikidata dump into usable format.
python preprocess/wikitext/extract_wikidata.py \
    --source-path $WIKIDATA_DUMP_DIR/latest-truthy.nt \
    --target-dir $WIKIDATA_DUMP_DIR \
    --lang en \

echo "Wikdata KG is extracted into $WIKIDATA_DUMP_DIR" >&2

# Attempt surface matching between Wikidata and the splitted articles.
python preprocess/wikitext/match_wikidata.py \
    --wikitext-dir $WIKITEXT_DIR \
    --canonical-forms-dir data/canonical_forms/ \
    --wikidata-dir $WIKIDATA_DUMP_DIR \
    --output-dir $ALIGNED_DATA_DIR \
    --lower \

echo "Wikdata and Wikitext articles are aligned and stored at $ALIGNED_DATA_DIR" >&2

# Preprocess the data
# NOTE: Check the following paths in the file!
#     OPENKE_DIR        ... Directory to the unzipped OpenKE Wikidata archive.
#     WIKIDATA_DUMP_DIR ... same as $WIKIDATA_DUMP_DIR in this file.
#     WIKITEXT_DIR      ... same as $WIKITEXT_DIR in this file.
#     ALIGNED_DATA_DIR  ... same as $ALIGNED_DATA_DIR in this file.
#     SAVE_DIR          ... Directory to save the processed Wikitext-S,F data.
python preprocess/process_wikitext.py

echo "Dataset is created and ready for training." >&2
