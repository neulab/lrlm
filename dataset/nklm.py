import copy
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Counter as CounterType, Dict, List, NamedTuple, Tuple

from dataset.base import KBLMDataset
from dataset.utils import BatchSequence, Relation, Sentence
from nnlib import utils

__all__ = [
    'NKLMExample',
    'NKLMDataset',
]

LOGGER = logging.getLogger(__name__)


class NKLMExample(NamedTuple):
    sentence: Sentence  # sentence
    topic_id: int  # topic_id
    relations: List[Relation]  # relations: all relations of the topic
    rel_ids: List[int]  # rel_ids: relation index of each word
    copy_pos: List[int]  # copy_pos: copy position of the object for each word, -1 for NaF
    surface_indices: List[int]  # surface_indices: surface form index of the object for each word, -1 for NaF

    def __len__(self) -> int:
        return len(self.sentence)


class NKLMDataset(KBLMDataset[NKLMExample]):
    """
    Relations are represented as a sequence the length of the sentence.
    Each time step is a tuple: (rel_id, obj_id, copy_pos).
    For NaF, rel_id = -1. For anchor, rel_id = -2. For, rel_id = -3.
    Additionally, topic IDs are included as another list.
    """

    def __init__(self, path, batch_size, vocab_dir, *args, unk_rels_strategy='unk', **kwargs):
        kwargs['use_entity_vecs'] = True
        self._unk_rels_strategy = unk_rels_strategy
        super().__init__(path, batch_size, vocab_dir, *args, **kwargs)

    def read_data(self, path: Path) -> Dict[str, List[NKLMExample]]:
        data: Dict[str, List[NKLMExample]] = {}
        for split in (['train'] if self._include_train else []) + ['valid', 'test']:
            with (path / f'{split}.pkl').open('rb') as data_file:
                dataset = []
                for ex_idx, tup in enumerate(pickle.load(data_file)):
                    assert len(tup) == 6
                    # type-annotated unpacking
                    sentence: Sentence = ['<s>'] + tup[0]
                    topic_id: int = tup[1]
                    # WikiFacts does not contain aliases, so we just treat it as canonical
                    relations: List[Relation] = [Relation(*rel) for rel in tup[2]]
                    rel_ids: List[int] = [-1] + tup[3]
                    copy_pos: List[int] = [-1] + tup[4]
                    surface_indices: List[int] = [-1] + tup[5]

                    if self._use_only_first_section:
                        end_index = self.find_first_section(sentence)
                        sentence = sentence[:end_index]
                        rel_ids = rel_ids[:end_index]
                        copy_pos = copy_pos[:end_index]
                        surface_indices = surface_indices[:end_index]

                    # Add suffix only after finalizing the article.
                    sentence = sentence + ['</s>']
                    rel_ids = rel_ids + [-1]
                    copy_pos = copy_pos + [-1]
                    surface_indices = surface_indices + [-1]

                    if (not self._use_anchor) or (self._unk_rels_strategy in ['unk', 'remove']):
                        # remap indices
                        relations, new_indices = self.remap_rels(relations, self._use_anchor, self._unk_rels_strategy)
                        for idx in range(len(rel_ids)):
                            if rel_ids[idx] != -1:
                                new_idx = new_indices[rel_ids[idx]]
                                if new_idx == -1:
                                    copy_pos[idx] = rel_ids[idx] = -1
                                else:
                                    rel_ids[idx] = new_idx
                    ex = NKLMExample(sentence, topic_id, relations, rel_ids, copy_pos, surface_indices)
                    dataset.append(ex)
                data[split] = dataset
        return data

    def gather_entity_stats(self, dataset: List[NKLMExample]) -> Dict[int, CounterType[int]]:
        entity_count_per_type: Dict[int, CounterType[int]] = defaultdict(Counter)
        for ex in self.data['train']:
            rel_count = Counter(rel_id for rel_id, pos in zip(ex.rel_ids, ex.copy_pos) if pos == 0)
            for rel_id, count in rel_count.items():
                rel = ex.relations[rel_id]
                if rel.rel_typ not in [-1, -2]:  # not TITLE or ANCHOR
                    entity_count_per_type[rel.rel_typ].update({rel.obj_id: count})
        return entity_count_per_type

    def create_one_batch(self, examples: List[NKLMExample], bptt_size: int) \
            -> Tuple[List[List[Relation]], List[BatchSequence]]:
        sentences = [ex.sentence for ex in examples]
        rels = [ex.relations for ex in examples]
        rel_ids = [ex.rel_ids for ex in examples]
        copy_pos = [ex.copy_pos for ex in examples]
        surface_indices = [ex.surface_indices for ex in examples]

        max_length = max(len(s) for s in sentences)
        n_splits = utils.ceil_div(max_length - 1, bptt_size)  # minus 1 because there's always one extra

        batches = []
        for split_idx in range(n_splits):
            interval = slice(split_idx * bptt_size, (split_idx + 1) * bptt_size + 1)
            sequence = [s[interval] for s in sentences]
            batch = BatchSequence(self.word_vocab, sequence,
                                  rel_ids=([s[interval] for s in rel_ids], -1),
                                  copy_pos=([s[interval] for s in copy_pos], -1),
                                  surface_indices=([s[interval] for s in surface_indices], -1))
            batches.append(batch)
        return rels, batches

    def remove_ambiguity(self, ex: NKLMExample, entity=False, alias=False) -> NKLMExample:
        """Takes relations and remove some of them according to the specified condition."""

        if not entity and not alias:
            return ex

        relations = copy.copy(ex.relations)
        rel_ids = copy.copy(ex.rel_ids)
        copy_pos = copy.copy(ex.copy_pos)
        surface_indices = copy.copy(ex.surface_indices)
        seq_len = len(rel_ids)

        if entity:
            new_indices, relations = self.remap_entity_ambiguous_rels(relations)
            if new_indices is not None:
                for idx in range(seq_len):
                    rel_id = rel_ids[idx]
                    if rel_id == -1:
                        continue
                    rel_ids[idx] = new_indices[rel_id]
                    if new_indices[rel_id] == -1:
                        copy_pos[idx] = surface_indices[idx] = -1

        if alias:
            # remove non-canonical indices
            for idx in range(seq_len):
                if surface_indices[idx] > 0:
                    rel_ids[idx] = copy_pos[idx] = surface_indices[idx] = -1

        return ex._replace(relations=relations, rel_ids=rel_ids, copy_pos=copy_pos, surface_indices=surface_indices)
