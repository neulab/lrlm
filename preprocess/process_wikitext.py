import json
import logging
import pickle
import pprint
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, NewType, Optional, Set, Tuple, TypeVar

import numpy as np
import torch
from torch import Tensor

from nnlib import utils
from nnlib.utils import Logging

LOGGER = logging.getLogger(__name__)


####################### MODIFY THESE ACCORDING TO YOUR PATHS ##########################
# For the step numbers, please follow docs/data.md

# Downloaded and unzipped OpenKE *Wikidata* directory from Step 1
OPENKE_DIR = Path('./openke/Wikidata')
# Result of Step 3
WIKIDATA_DUMP_DIR = Path('./wikidata/')
# The same path as specified in `preprocess/scripts/preprocess_wikitext.sh`
ALIGNED_DATA_DIR = Path('./dataset')
# Path to the unzipped WikiText-103-raw from Step 2
WIKITEXT_DIR = Path('./wikitext-raw')

# Target directory
SAVE_DIR = Path('./processed/')
#######################################################################################


VEC_DIM = 100
ENTITY_VEC_PATH = OPENKE_DIR / '/embeddings/dimension_100/transe/entity2vec.bin'
RELATION_VEC_PATH = OPENKE_DIR / '/embeddings/dimension_100/transe/relation2vec.bin'
ENTITY_MAPPING_PATH = OPENKE_DIR / '/Wikidata/knowledge graphs/entity2id.txt'
RELATION_MAPPING_PATH = OPENKE_DIR / '/knowledge graphs/relation2id.txt'

TOPIC_JSON_PATH = lambda split: (WIKITEXT_DIR / f'{split}.json')

# Type definitions
K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')

WikidataID = NewType('WikidataID', str)
Span = Tuple[int, int]
Sentence = List[str]


class EntityMention(NamedTuple):
    span: Span
    surface: str
    rel_idx: int  # index to the unique relations of a topic


class RelationWikiID(NamedTuple):
    rel_typ: WikidataID
    obj_id: WikidataID
    obj_alias: List[str]  # canonical form is the first element


class RawExampleWikiID(NamedTuple):
    topic_id: WikidataID
    sentence: Sentence
    relations: List[RelationWikiID]
    mentions: List[EntityMention]


RawDump = Tuple[
    Sentence,          # sentence
    List[              # List of matched spans
        Tuple[
            str,       # Wikidata ID for the object entity
            str,       # Wikidata ID for the subject entity (topic entity)
            Span,      # span
            Tuple[     # List of relations that connects subject and object
                List[str],  # Relations in the form of WikidataID
                List[str]   # Relations in the form of text
            ],
            str,       # matched surface form
            List[str]  # List of aliases
        ]
    ]
]


# do not use a named tuple for objects to dump
Example = Tuple[
    Sentence,  # sentence
    int,  # topic_id
    List[Tuple[int, int, List[str]]],  # relations: all relations of the topic, in format like RelationWikiID
    List[int],  # rel_ids: relation index of each word
    List[int],  # copy_pos: copy position of the object for each word, -1 for NaF
    List[int],  # surface_indices: surface form index of the object for each word, -1 for NaF
]


MatchedSpan = Tuple[
    int,  # start of span
    int,  # end of span (+1)
    int,  # rel-type ID
    int,  # index into `relations` array (relID for example)
    int,  # surface form's index into `alias` array
]

# Constant definitions
NAF = WikidataID('__NaF__')
ANCHOR = WikidataID('__anchor__')
TOPIC_ITSELF = WikidataID('__topic_itself__')
UNK_ENTITY = WikidataID('__none__')


def preprocess(t: str) -> str:
    t = re.sub(r"\s@-@\s", "-", t)
    t = re.sub(r"\(\s(.+)\s\)", r"(\1)", t)
    t = re.sub(r"\s\'(\w+)", r"'\1", t)
    t = t.replace("What (ever)", "What(ever)")
    t = t.replace(" ; ", ";")
    t = t.replace(" – ", "–")  # something called "em dash" present in some titles
    t = t.replace("... ", "...")
    t = t.replace(" ...", "...")
    t = t.replace(" , ", ", ")
    t = t.replace(" ' ", "' ")
    t = re.sub(r"\s?'\s?([a-z]{,2})\s", r"'\1 ", t)
    t = t.replace(" !", "!")
    t = t.replace(" ?", "?")
    t = t.replace(" :", ":")
    # Only reduce spaces when single characters are ampersand'ed.
    t = re.sub(r"(^|\s)(\w)\s&\s(\w)(\s|$)", r"\1\2&\3\4", t)
    return t


def load_id2str(id2str_path: Path, keys: Optional[Set[WikidataID]] = None, skip_header=False) \
        -> Dict[WikidataID, str]:
    id2str: Dict[WikidataID, str] = {}
    with id2str_path.open() as f:
        if skip_header:
            _ = f.readline()
        for line in f:
            if line.strip() == "":
                continue
            wiki_id, *rel_names = line.strip().split('\t')
            rel_name = rel_names[0]  # get rid of AKAs
            wikidata_id = WikidataID(wiki_id)
            if keys is not None and wikidata_id not in keys:
                continue  # save memory
            id2str[wikidata_id] = rel_name
    return id2str


def load_id2vec(id2vec_path: Path, keys: Optional[Set[WikidataID]] = None) -> Dict[WikidataID, int]:
    id2vec: Dict[WikidataID, int] = {}
    with id2vec_path.open() as f:
        _ = f.readline()  # always skip first line
        for line in f:
            if line.strip() == "":
                continue
            wiki_id, vec_id = line.split()
            wikidata_id = WikidataID(wiki_id)
            if keys is not None and wikidata_id not in keys:
                continue  # save memory
            id2vec[wikidata_id] = int(vec_id)
    return id2vec


def _load_vecs(all_ids: Set[WikidataID], id2vec_path: Path, vec_path: Path,
               vec_dim: int) -> Tuple[Dict[WikidataID, int], Tensor]:
    id2vec = load_id2vec(id2vec_path, keys=all_ids)

    # check coverage
    if len(all_ids) != len(id2vec):
        missing = all_ids - set(id2vec.keys())
        print(f"Missing id2vec items ({len(missing)})")

    all_vecs = np.memmap(vec_path, dtype=np.float32, mode='r')
    extracted_vecs = [all_vecs[(idx * vec_dim):((idx + 1) * vec_dim)]
                      for idx in sorted(id2vec.values())]
    vecs = torch.from_numpy(np.stack(extracted_vecs))
    # map to vec idx
    final_map = {r: idx for idx, (r, _) in enumerate(sorted(id2vec.items(), key=lambda x: x[1]))}
    return final_map, vecs


def load_entities(all_entities: Set[WikidataID]) -> Dict[WikidataID, int]:
    entity_vec_path = SAVE_DIR / 'entity_vec.pt'
    entity_map, entity_vec = _load_vecs(
        all_entities,
        id2vec_path=ENTITY_MAPPING_PATH,
        vec_path=ENTITY_VEC_PATH,
        vec_dim=VEC_DIM
    )
    torch.save(entity_vec, entity_vec_path)

    return entity_map


def load_relations(all_rels: Set[WikidataID]) -> Dict[WikidataID, int]:
    rel_vec_path = SAVE_DIR / 'relation_vec.pt'
    rel_map, rel_vec = _load_vecs(
        all_rels,
        id2vec_path=RELATION_MAPPING_PATH,
        vec_path=RELATION_VEC_PATH,
        vec_dim=VEC_DIM
    )
    torch.save(rel_vec, rel_vec_path)

    return rel_map


def numericalize_rel(data: List[RawExampleWikiID],
                     rel_map: Dict[WikidataID, int], entity_map: Dict[WikidataID, int]) \
        -> Tuple[List[Example], List[List[MatchedSpan]]]:
    processed_data = []
    processed_spans = []
    for example in data:
        # We store the surface forms for mentions because there might be subtle differences with sentence tokens.
        # We also rely on the model to compute the probabilities (either on-the-fly or by per-batch preprocessing).
        sent = example.sentence
        topic_id = entity_map.get(example.topic_id, -1)
        rel_ids = [-1] * len(sent)  # NaF
        copy_pos = [-1] * len(sent)
        surface_indices = [-1] * len(sent)
        matched_spans = []

        # noinspection PyShadowingNames
        def sort_key(mention: EntityMention):
            l, r = mention.span
            is_canonical = (example.relations[mention.rel_idx].obj_alias[0] == mention.surface)
            return (
                -(r - l),  # span length (desc)
                l,  # span position (incr)
                not is_canonical,  # canonical mentions first
                mention.rel_idx,  # break ties by relation ID
                mention.surface,  # break further ties by surface form
            )

        # Sort mentions in reverse order of span length.
        # In case of tie, put earlier spans in the front. Break further ties by relation ID to keep consistency.
        # E.g. sentence = [a, b, c, d, e]. Prefer [a, b, c, d] to [a, b, c]; prefer [a, b, c] to [b, c, d].
        for mention in sorted(example.mentions, key=sort_key):
            l, r = mention.span
            # simple overwrite the spans if they overlap
            if l == -1 or r == -1:
                continue  # non-existing
            l = max(l, 0)
            r = min(r, len(sent))
            if l >= r:
                continue

            rel_idx = mention.rel_idx
            rel = example.relations[rel_idx]
            try:
                surface_idx = next(i for i, x in enumerate(rel.obj_alias) if x == mention.surface)  # must exist
            except StopIteration:
                continue  # skip this mention
            # spans information is only used by LRLM, so we keep these anyway
            matched_spans.append((l, r, rel_map[rel.rel_typ], rel_idx, surface_idx))

            if any(x != -1 for x in rel_ids[l:r]):
                # if any token in span is already annotated, skip it to avoid inconsistent annotations
                continue

            rel_ids[l:r] = [mention.rel_idx] * (r - l)
            copy_pos[l:r] = range(r - l)
            surface_indices[l:r] = [surface_idx] * (r - l)

        processed_spans.append(matched_spans)
        processed_data.append((
            sent,
            topic_id,
            [(rel_map[rel.rel_typ], entity_map.get(rel.obj_id, -1), rel.obj_alias)
             for rel in example.relations],
            rel_ids,
            copy_pos,
            surface_indices,
        ))
    return processed_data, processed_spans


def read_data(path: Path) -> Dict[str, List[RawExampleWikiID]]:
    bad_examples: List[Tuple[str, int, str]] = []
    data = {}
    for split in ['train', 'valid', 'test']:
        with (path / f'{split}.pkl').open('rb') as f:
            # relation tuple: (span, rel_type_desc, name, canonical_name)
            with utils.work_in_progress(f"Loading {split} set"):
                dump: List[RawDump] = pickle.load(f)

            examples = []
            for idx, (sent, rels) in enumerate(utils.progress(dump, desc='Reading data')):
                # map (rel_typ, canonical) to list of aliases, since lists aren't hashable
                rel_to_alias: Dict[Tuple[str, str], List[str]] = \
                    {(rel[0][0], obj_id): alias for obj_id, _, _, rel, _, alias in rels}

                # sort it so the order is consistent
                relations: List[RelationWikiID] = sorted([
                    RelationWikiID(WikidataID(rel_id), WikidataID(obj_id), obj_alias)
                    for (rel_id, obj_id), obj_alias in rel_to_alias.items()
                ])
                rel_to_id: Dict[Tuple[str, str], int] = {
                    (rel_id, obj_id): idx
                    for idx, (rel_id, obj_id, obj_alias) in enumerate(relations)
                }
                # dedup to remove duplicate (-1, -1)
                mentions: List[EntityMention] = list(set(
                    EntityMention(span, surface, rel_to_id[(rel_info[0][0], obj_id)])
                    for obj_id, head_id, span, rel_info, surface, _ in rels
                ))
                try:
                    # must exist - head id with the relation: @TITLE@ is the topic WikidataID
                    topic_id = next(head_id for _, head_id, _, rel_info, surface, alias in rels if rel_info[0][0] == "@TITLE@")
                except StopIteration:
                    bad_examples.append((split, idx, ' '.join(sent)[:100]))
                    continue

                converted_relations = []
                for r in relations:
                    converted_relations.append(
                        RelationWikiID(
                            TOPIC_ITSELF if r.rel_typ == "@TITLE@" else r.rel_typ,
                            r.obj,
                            r.obj_alias
                        )
                    )

                example = RawExampleWikiID(WikidataID(topic_id), sent, converted_relations, mentions)
                examples.append(example)
            data[split] = examples

    if len(bad_examples) > 0:
        Logging.warn(f"{len(bad_examples)} bad examples:\n"
                     f"{pprint.pformat(bad_examples)}")
    else:
        Logging.verbose("All examples are good")

    return data


def main():
    Logging.verbosity_level = Logging.VERBOSE

    Logging.warn("This program requires lots of memory (preferably >= 30GB).")

    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True)

    # Read the Wikimedia IDs for each article, and filter the relations
    topic_ids: Set[WikidataID] = set()
    split_title_id: Dict[str, List[Tuple[str, WikidataID]]] = {}
    for split in ['train', 'valid', 'test']:
        with utils.work_in_progress(f"Loading {split} set titles"), \
             open(TOPIC_JSON_PATH(split=split)) as f:
            j = json.load(f)
        split_title_id[split] = [(article['title'], WikidataID(article['id'])) for article in j]
        topic_ids.update([wid for _, wid in split_title_id[split]])
        del j

    with utils.work_in_progress("Loading Wikidata ID mapping"):
        id2rel = load_id2str(WIKIDATA_DUMP_DIR / 'properties.txt')

    # Match the relations
    matched_dataset = read_data(ALIGNED_DATA_DIR)

    # Gather entities & relation vectors
    found_entities = set()
    found_rels = set()
    for split in matched_dataset:
        for example in matched_dataset[split]:
            found_entities.add(example.topic_id)
            for rel in example.relations:
                found_entities.add(rel.obj_id)
                found_rels.add(rel.rel_typ)
    found_entities -= {UNK_ENTITY}
    found_rels -= {NAF, ANCHOR, TOPIC_ITSELF}
    with utils.work_in_progress("Building rel vecs"):
        rel_map = load_relations(found_rels)
        rel_map.update({NAF: -1, ANCHOR: -2, TOPIC_ITSELF: -3})
        unk_rels = found_rels.difference(rel_map)
        # NOTE: unk_rels is a set, its order is undetermined, so we sort it to make sure it's consistent between runs
        for idx, rel in enumerate(sorted(unk_rels)):
            rel_map[rel] = -idx - 4  # starting from -4, going towards -inf
    with utils.work_in_progress("Building entity vecs"):
        entity_map = load_entities(found_entities)
        entity_map.update({UNK_ENTITY: -1})
        print(f"Topic ID coverage: {len(topic_ids.intersection(entity_map))}/{len(topic_ids)}")

    # save relation type names for use during generation
    id_to_rel_name = dict(id2rel)
    id_to_rel_name.update({NAF: 'Not-A-Fact', ANCHOR: 'ANCHOR', TOPIC_ITSELF: 'TITLE'})
    rel_names: Dict[int, str] = {}
    for r_rel, rel_id in rel_map.items():
        rel_names[rel_id] = id_to_rel_name[r_rel]
    with (SAVE_DIR / 'rel_names.pkl').open('wb') as f:
        pickle.dump(rel_names, f)
        print(f"Relation names saved to {(SAVE_DIR / 'rel_names.pkl')}")

    # Convert into numbers to create the final dataset
    for split in matched_dataset:
        with utils.work_in_progress(f"Converting {split} set"):
            dataset, matched_spans = numericalize_rel(matched_dataset[split], rel_map, entity_map)

        path = SAVE_DIR / f'{split}.pkl'
        with path.open('wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset split '{split}' saved to {path}, {len(dataset)} examples")

        path = SAVE_DIR / f'{split}.span.pkl'
        with path.open('wb') as f:
            pickle.dump(matched_spans, f)
        print(f"Matched spans split '{split}' saved to {path}")


if __name__ == '__main__':
    from IPython.core import ultratb
    import sys

    sys.excepthook = ultratb.FormattedTB(mode="Context", color_scheme="Linux", call_pdb=1)

    main()
