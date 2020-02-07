import copy
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Counter as CounterType, Dict, List, NamedTuple, Tuple

from dataset.base import KBLMDataset
from dataset.utils import BatchSequence, MatchedSpan, Relation, Sentence
from nnlib import utils

__all__ = [
    'LRLMExample',
    'LRLMDataset',
]

LOGGER = logging.getLogger(__name__)


class LRLMExample(NamedTuple):
    sentence: Sentence  # sentence
    spans: List[MatchedSpan]  # spans: matched entity mentions
    relations: List[Relation]  # relations: all relations of the topic

    def __len__(self) -> int:
        return len(self.sentence)


class LRLMDataset(KBLMDataset[LRLMExample]):
    def read_data(self, path: Path) -> Dict[str, List[LRLMExample]]:
        data: Dict[str, List[LRLMExample]] = {}
        for split in (['train'] if self._include_train else []) + ['valid', 'test']:
            with (path / f'{split}.pkl').open('rb') as data_file:
                data_pkl = pickle.load(data_file)
            with (path / f'{split}.span.pkl').open('rb') as span_file:
                spans_pkl = pickle.load(span_file)

            dataset = []
            for ex_idx, (ex, raw_matched_spans) in enumerate(zip(data_pkl, spans_pkl)):
                sentence = ex[0]
                if self._use_only_first_section:
                    end_index = self.find_first_section(ex[0])
                    sentence = sentence[:end_index]
                    raw_matched_spans = [s for s in raw_matched_spans if s[1] < end_index]
                tokens = ['<s>'] + sentence + ['</s>']  # add <sos> token, spans are shifted during `make_batch`

                relations: List[Relation] = [Relation(*rel) for rel in ex[2]]
                # `unk_rels_strategy` defaults to 'params', because in LRLM we train our own embeddings anyway
                relations, new_indices = self.remap_rels(relations, self._use_anchor, unk_rels_strategy='params')

                matched_spans: List[MatchedSpan] = [MatchedSpan(*s) for s in raw_matched_spans]
                spans = []  # shouldn't have duplicates
                for span in matched_spans:
                    new_rel_idx = new_indices[span.rel_idx]
                    if new_rel_idx != -1:
                        spans.append(span._replace(rel_idx=new_rel_idx))
                dataset.append(LRLMExample(tokens, spans, relations))
            data[split] = dataset
        return data

    def gather_entity_stats(self, dataset: List[LRLMExample]) -> Dict[int, CounterType[int]]:
        entity_count_per_type: Dict[int, CounterType[int]] = defaultdict(Counter)
        for ex in dataset:
            rel_count = Counter(span.rel_idx for span in ex.spans)
            for rel_id, count in rel_count.items():
                rel = ex.relations[rel_id]
                if rel.rel_typ not in [-1, -2, -3]:  # not TITLE or ANCHOR
                    entity_count_per_type[rel.rel_typ].update({rel.obj_id: count})
        return entity_count_per_type

    def create_one_batch(self, examples: List[LRLMExample], bptt_size: int) \
            -> Tuple[List[List[Relation]], List[BatchSequence]]:
        sentences = [x.sentence for x in examples]  # <s> accounted in construct_example method
        max_length = max(len(s) for s in sentences)
        n_splits = utils.ceil_div(max_length - 1, bptt_size)
        init_batch = [ex.relations for ex in examples]

        split_spans: List[List[List[MatchedSpan]]] = [[[] for __ in examples] for _ in range(n_splits)]
        for b_idx, ex in enumerate(examples):
            for span in ex.spans:
                # Begin index of the span should be the index before the expression begin.
                # End index of the span points to the index of the last word in the entity
                start = span.start % bptt_size
                end = (span.end - 1) % bptt_size
                # if start > end:
                #     continue
                split_spans[span.start // bptt_size][b_idx].append(span._replace(start=start, end=end))
        batches = []
        for split_idx in range(n_splits):
            interval = slice(split_idx * bptt_size, (split_idx + 1) * bptt_size + 1)
            rels_interval = split_spans[split_idx]
            sentence_interval = [sent[interval] for sent in sentences]
            batch = BatchSequence(self.word_vocab, sentence_interval, rels_interval)
            batches.append(batch)
        return init_batch, batches

    def remove_ambiguity(self, ex: LRLMExample, entity=False, alias=False) -> LRLMExample:
        """Takes relations and remove some of them according to the specified condition."""

        if not entity and not alias:
            return ex

        relations = copy.copy(ex.relations)
        spans = copy.copy(ex.spans)

        if entity:
            new_indices, relations = self.remap_entity_ambiguous_rels(relations)
            if new_indices is not None:
                new_spans = []
                for span in spans:
                    rel_id = span.rel_idx
                    if new_indices[rel_id] != -1:
                        new_spans.append(span._replace(rel_idx=new_indices[rel_id]))
                spans = new_spans

        if alias:
            # Keep only the cases where the surface matches the canonical form
            spans = [span for span in spans if span.surface_idx == 0]

        return ex._replace(spans=spans, relations=relations)
