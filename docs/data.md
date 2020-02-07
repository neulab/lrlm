## Dataset

### Dataset Format

A dataset consists of the following files:

- `{train,valid,test}.pkl`: Examples in each data split, in pickled format. These files contain the sentences, topic
  entity, and relations for each article.
- `{train,valid,test}.span.pkl`: Span annotations for each example in data split, in pickled format. These files contain
  spans with their matching relations for each article.
- `entity_vec.pt` and `relation_vec.pt`: Processed knowledge base embeddings for entities and relations, in PyTorch
  weights format.
- `unk_probs.txt`: Word probabilities precomputed by the character-level LM.

The detailed format for each file are:

- `{train,valid,test}.pkl`: A list where each element corresponds to an article. Each article is an tuple consisting of
  the following, in order:

  - `sentence` (`List[str]`): A list of words representing the tokenized article.
  - `topic_id`: A unique ID of the topic entity. This field is ignored.
  - `relations`: A list of relations of the topic entity. Each relation is a tuple consisting of three fields:

    - `rel_typ` (`int`): The ID of the type of the relation.
    - `obj_id` (`int`): The ID of the object entity of the relation.
    - `obj_alias` (`List[str]`): A list of surface forms (aliases) of the object entity.
  
  For NKLM-compatible datasets, the tuple should also contain the following fields:
  
  - `rel_ids` (`List[int]`): The index of the relation (index into the `relations` list) for each word of the article.
    For words which are not a part of a entity mention, use `-1` as the index.
  - `copy_pos` (`List[int]`): The copying position of the word (please refer to Appendix D of the paper). For words
    which are not a part of a entity mention, use `-1` as the position. 
  - `surface_indices` (`List[int]`): The index of the surface form (index into the `obj_alias` list of the relation
    specified by `rel_ids`) for each word of the article. For words which are not a part of a entity mention, use `-1`
    as the index.
- `{train,valid,test}.span.pkl`: A list where each element corresponds to an article, in the same order as
  `{train,valid,test}.pkl`. Each article is an list of matched spans, where each matched span is a tuple consisting of
  5 fields:
  
  - `start` (`int`): The start index of the span.
  - `end` (`int`): The end index of the span. The span is left-closed and right-open (`[start, end)`), e.g. the words
    in the span are `sentence[start, start+1, ..., end-1]`.
  - `rel_typ` (`int`): The ID of the type of the relation.
  - `rel_idx` (`int`): The index of the relation (index into the `relations` list).
  - `surface_idx` (`int`): The index of the surface form (index into the `obj_alias` list of the relation specified
    by `rel_idx`).
- `entity_vec.pt` and `relation_vec.pt`: A PyTorch tensor of shape `[n_vecs, vec_dim]`. Tensors should be ordered
  according to their ID, e.g. the vector for the entity with ID `x` should be `relation_vec[x]`.
- `unk_probs.txt`: A tab-separated file with two columns, corresponding to words and their log-probabilities.  

**Notes on relation type IDs:** IDs for relation types with corresponding pre-trained embeddings are numbered from 0.
There are also negative IDs that are used for specific purposes:

- ID `-1` is used to denote that the span is not a relation (also called "not a fact").
- ID `-2` indicates a hyperlink to another Wikipedia article (also called "anchor" relations). This ID only exists for
  WikiFacts. Please see paper Section 4.1 and Appendix C for details.
- ID `-3` indicates a mention of the article title (also called "title" relations).
- IDs `-4` and below indicates relations that do not have pre-trained embeddings. We train embeddings for these
  relations.

**Notes on entity IDs:** Similar to relation type IDs, entities with corresponding pre-trained embeddings are numbered
from 0. A special ID `-1` is used to represent entities without pre-trained embeddings (also called unknown entities).

### Creating the WikiFacts Dataset

To create the WikiFacts dataset from scratch:

1. Download pre-trained Freebase embeddings from
   [OpenKE](http://openke.thunlp.org/index/toolkits#pretrained-embeddings). This might take a long time.
2. Download the original WikiFacts dataset:
   <https://bitbucket.org/skaasj/wikifact_filmactor/src/master/>.
3. Install `sacremoses==0.0.33`.
4. Manually modify the path variables in `preprocess/process_wikifacts.py`:

   - `DATASET_PATH` should point to the Tar archive downloaded from the WikiFacts repository.
   - `ENTITY_VEC_PATH` and related paths should point to appropriate files downloaded from OpenKE.
   - `SAVE_DIR` should point to the directory where the generated dataset will be stored.
5. Run `preprocess/process_wikifacts.py`.
6. Train a character-level language model to produce probabilities for unknown words:
   ```bash
   # Replace SAVE_DIR with the path specified above
   ./scripts/run_charlm_wikitext.sh --data SAVE_DIR
   ```

### Creating the WikiText Dataset

To create the WikiText-S,F dataset from scratch:

1. Download pre-trained Wikidata embeddings from
   [OpenKE](http://openke.thunlp.org/index/toolkits#pretrained-embeddings). This might take a long time.
2. Download the original WikiText-103 character-level dataset and unzip the archive:
   <https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip>.
3. Download the Wikidata dump triples and unzip the archive. This takes significant space (374GB after unzipped as of 9/21/2018).
   <https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz>.
4. Install the requirements: `requests`, `tqdm`, and `nltk`.
5. Manually modify the path variables for the two files:

   - `preprocess/scripts/preprocess_wikitext.sh`:
      - `WIKITEXT_DIR`: Path to the unzipped directory from Step 2.
      - `WIKIDATA_DUMP_DIR`: Path to the directory containing the unzipped Wikidata dump from Step 3.
      - `ALIGNED_DATA_DIR`: Directory to save the aligned data between Wikidata & WikiText.

   - `preprocess/process_wikitext.py`:
      - `OPENKE_PATH`: Path to the unzipped archive from Step 1.
      - `WIKITEXT_DIR`: Path to the unzipped directory from Step 2.
      - `WIKIDATA_DUMP_DIR`: Path to the directory containing the unzipped Wikidata dump from Step 3.
      - `ALIGNED_DATA_DIR`: Directory to save the aligned data between Wikidata & WikiText.
      - `SAVE_DIR` should point to the directory where the generated dataset will be stored.

6. Run `preprocess/scripts/preprocess_wikitext.sh` from the root of the directory.
7. Train fastText embeddings by running:
   ```bash
   # Embeddings for WT-Full
   python preprocess/train_fasttext.py \
       --fasttext-path /path/to/fasttext/binary \
       --dataset-dir data/wikitext/ \
       --output-dir data/fasttext/wt-full
   # Embeddings for WT-Short
   python preprocess/train_fasttext.py \
       --fasttext-path /path/to/fasttext/binary \
       --dataset-dir data/wikitext/ \
       --use-only-first-section \
       --output-dir data/fasttext/wt-short
   ```
6. Train a character-level language model to produce probabilities for unknown words:
   ```bash
   # Replace SAVE_DIR with the path specified above
   ./scripts/run_charlm_wikitext.sh --data SAVE_DIR
   ```
    
