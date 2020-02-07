import pickle
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NewType, Set, Tuple, TypeVar

import numpy as np
import torch
import sacremoses

from nnlib import utils


####################### MODIFY THESE ACCORDING TO YOUR PATHS ##########################
DATASET_PATH = Path('../wikifact_filmactor/tagged_film_actor_v0.1.tar.gz')

# Downloaded and unzipped OpenKE *Freebase* directory from Step 1
OPENKE_DIR = Path('./openke/Freebase')

SAVE_DIR = Path('./processed_wikifacts/')
#######################################################################################

VEC_DIM = 50
ENTITY_VEC_PATH = OPENKE_DIR / '/embeddings/dimension_50/transe/entity2vec.bin'
RELATION_VEC_PATH = OPENKE_DIR / 'embeddings/dimension_50/transe/relation2vec.bin'
ENTITY_MAPPING_PATH = OPENKE_DIR / 'knowledge graphs/entity2id.txt'
RELATION_MAPPING_PATH = OPENKE_DIR / 'knowledge graphs/relation2id.txt'

SAVE_DATASET_PATH = lambda split: SAVE_DIR / f'{split}.pkl'
SAVE_MATCHED_SPANS_PATH = lambda split: SAVE_DIR / f'{split}.span.pkl'
SAVE_ENTITY_VEC_PATH = SAVE_DIR / 'entity_vec.pt'
SAVE_RELATION_VEC_PATH = SAVE_DIR / 'relation_vec.pt'

T = TypeVar('T')

# Type definitions
FreebaseID = NewType('FreebaseID', str)
Relation = Tuple[FreebaseID, FreebaseID]
Sentence = List[str]

# There are no aliases for WikiFacts, but we add surface form related stuff for consistency
Example = Tuple[
    Sentence,  # sentence
    int,  # topic_id
    # List[Tuple[int, int]],  # relations: all relations of the topic
    List[Tuple[int, int, List[str]]],  # relations: all relations of the topic
    List[int],  # rel_ids: relation index of each word
    List[int],  # copy_pos: copy position of the object for each word, -1 for NaF
    List[int],  # surface_idx: surface form index of the object for each word, -1 for NaF
]
MatchedSpan = Tuple[
    int,  # start of span
    int,  # end of span (+1)
    int,  # rel-type ID
    int,  # index into `relations` array (relID for example)
    int,  # surface form's index into `alias` array
]

# Constant definitions
NAF = -1
ANCHOR = -2
TOPIC_ITSELF = -3
UNK_ENTITY = -1


def extract_id(path: tarfile.TarInfo) -> FreebaseID:
    # format: [topic_id].[freebase_id...].[extension]
    # e.g.: 4693.m.0k3jp_.sm
    return FreebaseID('.'.join(path.name.split('/')[-1].split('.')[1:-1]))


def remove_prefix(s: str) -> FreebaseID:
    # remove the '/ns/' prefix
    return FreebaseID(s[4:])


def extract_vector(vec: np.ndarray, idx: int) -> np.ndarray:
    return vec[(idx * VEC_DIM):((idx + 1) * VEC_DIM)]


class Vocabulary(defaultdict):
    def __init__(self, default_factory=None):
        if default_factory is None:
            default_factory = self.__len__
        super().__init__(default_factory=default_factory)


def load_wikifacts():
    """ Load the WikiFacts dataset """
    id_name: Dict[FreebaseID, str] = {}
    id_summary: Dict[FreebaseID, str] = {}
    id_relations: Dict[FreebaseID, List[Relation]] = {}
    relation_types: Set[FreebaseID] = set()

    def add_or_check(fid: FreebaseID, name: str):
        name = name.lower()
        if fid in id_name:
            assert name == id_name[fid]
        else:
            id_name[fid] = name

    tar = tarfile.open(DATASET_PATH)

    id_fb_raw: Dict[FreebaseID, str] = {}
    id_en_raw: Dict[FreebaseID, str] = {}

    # Annotated Wikipedia summary (.sm)
    for summary_file in utils.progress(tar.getmembers(), desc='Reading summary (.sm)'):
        if not summary_file.name.endswith('.sm'):
            continue
        freebase_id = extract_id(summary_file)
        with tar.extractfile(summary_file) as f:
            id_summary[freebase_id] = f.read().decode('utf-8')

    # Cache Freebase Topic files (.fb, .en)
    for file in utils.progress(tar.getmembers(), desc='Caching topic files'):
        freebase_id = extract_id(file)
        if file.name.endswith('.fb'):
            with tar.extractfile(file) as f:
                id_fb_raw[freebase_id] = f.read().decode('utf-8')
        elif file.name.endswith('en'):
            with tar.extractfile(file) as f:
                id_en_raw[freebase_id] = f.read().decode('utf-8')

    # Freebase Topic (.fb, .en)
    for freebase_id in utils.progress(id_fb_raw, desc='Extracting relations'):
        relations = []
        f_fb = id_fb_raw[freebase_id].split('\n')
        f_en = id_en_raw[freebase_id].split('\n')
        for rel_line, name_line in zip(f_fb, f_en):
            parts = rel_line.split()
            if len(parts) == 0:  # empty line
                continue
            rel_name = name_line.strip().split(' ')  # do not split on non-breaking or unicode spaces
            assert len(parts) == len(rel_name)
            if len(parts) == 5:  # composite value type
                if parts[0].startswith('['):  # subject is CVT
                    continue
                parts = [parts[0], parts[3], parts[4].rstrip(']')]
                if parts[0] == parts[-1]:  # don't include a relation with itself
                    continue
                # remove the final ']', but do not use `.rstrip` because the name could contain ']'
                rel_name = [rel_name[0], rel_name[3], rel_name[4][:-1]]
            elif len(parts) == 3:  # simple relation
                pass
            else:
                raise ValueError  # malformed data
            rel = [remove_prefix(r) for r in parts]
            r_sub, r_rel, r_obj = rel
            if r_sub != freebase_id:  # only keep relations whose subject matches topic
                continue
            add_or_check(r_sub, rel_name[0])
            add_or_check(r_obj, rel_name[2])
            relations.append((r_rel, r_obj))
            relation_types.add(r_rel)
        id_relations[freebase_id] = relations

    # # Freebase to Wikidata mappings
    # freebase_wikidata_map = {}
    # with utils.progress(open(FREEBASE_MAPPING_PATH), desc='Reading fb2w') as f:
    #     for line in f:
    #         if line.startswith('#') or line.strip() == '':
    #             continue
    #         parts = line.split()
    #         freebase_id = parts[0].split('/')[-1][:-1]
    #         wikidata_id = parts[2].split('/')[-1][:-1]
    #         freebase_wikidata_map[freebase_id] = wikidata_id
    #
    # # Check coverage
    # print(f"id_name coverage in fb2w: "
    #       f"{sum(int(fid in freebase_wikidata_map) for fid in id_name)}/{len(id_name)}")
    # print(f"id_summary coverage in fb2w: "
    #       f"{sum(int(fid in freebase_wikidata_map) for fid in id_summary)}/{len(id_summary)}")

    return id_name, id_summary, id_relations, relation_types


def load_openke_embeddings():
    """ Load OpenKE pretrained embeddings """

    entity_index: Dict[FreebaseID, int] = {}
    relation_index: Dict[FreebaseID, int] = {}

    # Entity to OpenKE ID mapping
    with utils.work_in_progress("Reading entity2id"), ENTITY_MAPPING_PATH.open() as f:
        entity_count = int(f.readline())
        for line in f:
            parts = line.split()
            freebase_id = FreebaseID(parts[0])
            entity_index[freebase_id] = int(parts[1])
        assert len(entity_index) == entity_count

    # Relation to OpenKE ID mapping
    with utils.work_in_progress("Reading relation2id"), RELATION_MAPPING_PATH.open() as f:
        relation_count = int(f.readline())
        for line in f:
            parts = line.split()
            freebase_id = FreebaseID(parts[0])
            relation_index[freebase_id] = int(parts[1])
        assert len(relation_index) == relation_count

    # Load binary vectors
    entity_vec = np.memmap(ENTITY_VEC_PATH, dtype=np.float32, mode='r')
    relation_vec = np.memmap(RELATION_VEC_PATH, dtype=np.float32, mode='r')

    return entity_index, relation_index, entity_vec, relation_vec


def main():
    replace_canonical = any(arg.startswith('--replace') for arg in sys.argv)
    if not replace_canonical:
        # global SAVE_DIR
        # SAVE_DIR = Path('./data/wikifacts_orig/')
        pass
    else:
        print("Arguments: Will replace the canonical forms.")
    print(f"Output directory: {SAVE_DIR}")

    skip_embeddings = any(arg.startswith('--skip') for arg in sys.argv)
    if skip_embeddings:
        print("Arguments: Will skip embedding generation.")

    id_name, id_summary, id_relations, relation_types = load_wikifacts()

    entity_index, relation_index, entity_vec, relation_vec = load_openke_embeddings()

    # Check OpenKE coverage
    print(f"Entity coverage in OpenKE: "
          f"{sum(int(fid in entity_index) for fid in id_name)}/{len(id_name)}")
    print(f"Relation coverage in OpenKE: "
          f"{sum(int(fid in relation_index) for fid in relation_types)}/{len(relation_types)}")

    """ Match entity positions and generate pickled dataset """

    # Remap entities and rel-types and store the mapping
    entity_map: Dict[FreebaseID, int] = {}
    relation_map: Dict[FreebaseID, int] = {}
    mapped_entity_vecs: List[np.ndarray] = []
    mapped_relation_vecs: List[np.ndarray] = []

    # noinspection PyShadowingNames
    def get_relation_id(rel: FreebaseID) -> int:
        rel_id = relation_map.get(rel, None)
        if rel_id is None:
            rel_id = relation_map[rel] = len(relation_map)
            mapped_relation_vecs.append(extract_vector(relation_vec, relation_index[rel]))
        return rel_id

    # noinspection PyShadowingNames
    def get_entity_id(entity: FreebaseID) -> int:
        ent_id = entity_map.get(entity, None)
        if ent_id is None:
            if entity not in entity_index:  # not all covered
                return UNK_ENTITY
            ent_id = entity_map[entity] = len(entity_map)
            mapped_entity_vecs.append(extract_vector(entity_vec, entity_index[entity]))
        return ent_id

    # Create the dataset
    dataset: List[Example] = []  # mapping of topic ID to data example
    dataset_matched_spans: List[List[MatchedSpan]] = []  # mapping of topic ID to matched spans

    # noinspection PyShadowingNames
    def find_relation(rels: List[Tuple[FreebaseID, FreebaseID]], obj: FreebaseID) -> List[FreebaseID]:
        # when there are multiple matches, just find the first one
        matched_rels = []
        for r_rel, r_obj in rels:
            if r_obj == obj:
                matched_rels.append(r_rel)
        return matched_rels

    # noinspection PyShadowingNames
    def match_positions(tokens: List[str], name: List[str]) -> List[int]:
        matched = [False] * len(name)
        positions = []
        for idx, token in enumerate(tokens):
            for match_idx, match_token in enumerate(name):
                if matched[match_idx] or match_token != token:
                    continue
                positions.append(match_idx)
                matched[match_idx] = True
                break
            else:
                return []
        return positions

    tokenizer = sacremoses.MosesTokenizer(lang='en')
    position_stats = defaultdict(int)
    for freebase_id in utils.progress(id_summary, desc='Creating dataset'):
        topic_id = get_entity_id(freebase_id)
        summary = id_summary[freebase_id].strip().split(' ')
        raw_relations = id_relations[freebase_id]
        relations = defaultdict(lambda: len(relations))
        rel_obj_names: Dict[int, str] = {}
        for r_rel, r_obj in raw_relations:
            rel_obj_names[relations[(get_relation_id(r_rel), get_entity_id(r_obj))]] = id_name[r_obj]
        topic_name = summary[0][2:summary[0].index('/')] if '/' in summary[0] else "<unknown topic name>"
        rel_obj_names[relations[(TOPIC_ITSELF, topic_id)]] = topic_name  # topic_itself
        sentence, rel_ids, copy_pos, surface_indices = [], [], [], []
        matched_spans: List[MatchedSpan] = []

        def add_words(s: str):
            tokens = tokenizer.tokenize(s, escape=False)
            sentence.extend(tokens)
            rel_ids.extend([NAF] * len(tokens))
            copy_pos.extend([-1] * len(tokens))
            surface_indices.extend([-1] * len(tokens))

        for word in summary:
            if '@@' in word:
                start_pos = word.find('@@')
                end_pos = word.find('@@', start_pos + 2)

                if start_pos > 0:
                    add_words(word[:start_pos])  # leading stuff

                entity_desc = word[(start_pos + 2):end_pos].split('/')  # there could be punctuation following '@@'
                assert len(entity_desc) >= 4  # entity name could contain '/'
                trailing = word[(end_pos + 2):]  # trailing stuff
                entity_name = '/'.join(entity_desc[:-3]).split('_')
                r_obj = FreebaseID(entity_desc[-1])
                obj_id = get_entity_id(r_obj)
                if entity_desc[-3] == 'f':
                    if r_obj == freebase_id:  # topic_itself
                        rel_id = TOPIC_ITSELF
                        matched_rels = [TOPIC_ITSELF]
                    else:
                        matched_rels = [get_relation_id(r_rel) for r_rel in find_relation(raw_relations, r_obj)]
                        # the relation might not exist
                        rel_id = None if len(matched_rels) == 0 else matched_rels[0]
                else:
                    rel_id = ANCHOR
                    matched_rels = [ANCHOR]  # include anchors anyway and filter them out if we don't use them
                    # matched_rels = []  # don't include anchors because we're only using them for our model

                if rel_id is None:  # not anymore, we allow unknown stuff: if obj_id is None or rel_id is None:
                    # no embedding for word, convert to normal token
                    add_words(' '.join(entity_name) + trailing)
                else:
                    position_stats['all'] += 1

                    if replace_canonical and r_obj in id_name:  # basically, everything except anchors
                        # just replace the entity_name, position will be in order
                        canonical_name = id_name[r_obj].split('_')
                        if canonical_name != entity_name:
                            position_stats['not_canonical'] += 1
                        entity_name = canonical_name

                    if entity_desc[-3] == 'f' and r_obj != freebase_id:
                        positions = match_positions(entity_name, id_name[r_obj].split('_'))
                        if len(positions) == 0:  # cannot match, resort to replacing
                            entity_name = id_name[r_obj].split('_')
                            positions = list(range(len(entity_name)))
                            position_stats['order'] += 1
                        else:
                            position_stats['match'] += 1
                            if positions == list(range(len(entity_name))):
                                position_stats['order'] += 1
                                position_stats['match_order'] += 1
                            elif positions == list(range(len(positions))):
                                position_stats['match_prefix'] += 1
                            elif positions == list(range(positions[0], positions[-1] + 1)):
                                position_stats['match_sub'] += 1
                    else:
                        # we don't replace canonical names for anchors (because we don't have them)
                        # anyway, these can't be used by our model
                        if entity_desc[-3] == 'a':
                            position_stats['anchor'] += 1
                            # add the anchor relation
                            rel_obj_names[relations[(ANCHOR, obj_id)]] = ' '.join(entity_name)
                        positions = list(range(len(entity_name)))  # these may not be in `id_name`, use as is
                        position_stats['order'] += 1

                    assert (rel_id, obj_id) in relations
                    rel_idx = relations[(rel_id, obj_id)]  # must exist
                    for rel_typ in matched_rels:
                        matched_spans.append((len(sentence), len(sentence) + len(entity_name), rel_typ, rel_idx, 0))
                    sentence.extend(entity_name)
                    rel_ids.extend([rel_idx] * len(entity_name))
                    copy_pos.extend(positions)
                    surface_indices.extend([0] * len(entity_name))

                    add_words(trailing)
            else:
                add_words(word)

        # we assume everything's canonical, so just add a pseudo canonical form
        rel_rev_map = {idx: rel for rel, idx in relations.items()}
        rel_list = []  # just in case `list(relations)` doesn't do as we expect
        for idx in range(len(rel_rev_map)):
            rel_id, obj_id = rel_rev_map[idx]
            rel_list.append((rel_id, obj_id, [rel_obj_names[idx].replace('_', ' ')]))
        example = (sentence, topic_id, rel_list, rel_ids, copy_pos, surface_indices)
        dataset.append(example)
        dataset_matched_spans.append(matched_spans)
    print(f"Position stats: {position_stats}")

    if replace_canonical:
        assert position_stats['all'] == position_stats['order']

    # Save them
    directory: Path = SAVE_DATASET_PATH('train').parent
    if not directory.exists():
        directory.mkdir(parents=True)

    # noinspection PyShadowingNames
    def split_dataset(dataset: List[T], splits: List[Tuple[int, int]]) -> List[List[T]]:
        dataset_size = len(dataset)
        dataset_splits = []
        for l, r in splits:
            start = int(dataset_size * (l / 100))
            end = int(dataset_size * (r / 100))
            dataset_splits.append(dataset[start:end])
        return dataset_splits

    splits = [(0, 80), (80, 90), (90, 100)]
    dataset_splits = split_dataset(dataset, splits)
    matched_spans_splits = split_dataset(dataset_matched_spans, splits)

    print(f"{len(dataset)} examples")
    for split, data, spans in zip(['train', 'valid', 'test'], dataset_splits, matched_spans_splits):
        path = SAVE_DATASET_PATH(split)
        with path.open('wb') as f:
            pickle.dump(data, f)
        print(f"Dataset split '{split}' saved to {path}, {len(data)} examples")

        path = SAVE_MATCHED_SPANS_PATH(split)
        with path.open('wb') as f:
            pickle.dump(spans, f)
        print(f"Matched spans split '{split}' saved to {path}")

    # save relation type names for use during generation
    rel_names: Dict[int, str] = {NAF: 'Not-A-Fact', ANCHOR: 'ANCHOR', TOPIC_ITSELF: 'TITLE'}
    for r_rel, rel_id in relation_map.items():
        rel_names[rel_id] = r_rel
    with (SAVE_DIR / 'rel_names.pkl').open('wb') as f:
        pickle.dump(rel_names, f)
    print("Relation names saved.")

    if not skip_embeddings:
        with utils.work_in_progress("Saving entity vecs"):
            stacked_entity_vecs = torch.from_numpy(np.stack(mapped_entity_vecs))
            torch.save(stacked_entity_vecs, SAVE_ENTITY_VEC_PATH)
        print(f"Entity vecs saved to {SAVE_ENTITY_VEC_PATH}, {len(stacked_entity_vecs)} vectors in total")

        with utils.work_in_progress("Saving relation vecs"):
            stacked_relation_vecs = torch.from_numpy(np.stack(mapped_relation_vecs))
            torch.save(stacked_relation_vecs, SAVE_RELATION_VEC_PATH)
        print(f"Relation vecs saved to {SAVE_RELATION_VEC_PATH}, {len(stacked_relation_vecs)} vectors in total")
    else:
        print("Embedding updates skipped.")

    print("Processing done.")


if __name__ == '__main__':
    from IPython.core import ultratb
    import sys

    sys.excepthook = ultratb.FormattedTB(mode="Context", color_scheme="Linux", call_pdb=1)

    main()
