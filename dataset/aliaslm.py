import logging
from typing import List, Tuple

import torch
from torch import Tensor

from dataset.lrlm import LRLMDataset, LRLMExample
from dataset.utils import BatchSequence, pad
from nnlib import utils

__all__ = [
    'AliasLMDataset',
]

LOGGER = logging.getLogger(__name__)


class AliasLMDataset(LRLMDataset):

    def create_one_batch(self, examples: List[LRLMExample], bptt_size: int) -> Tuple[Tensor, List[BatchSequence]]:
        sentences = [x.sentence for x in examples]  # <s> accounted in construct_example method
        max_length = max(len(s) for s in sentences)
        n_splits = utils.ceil_div(max_length - 1, bptt_size)
        init_batch = ([(ex.spans, ex.relations) for ex in examples], self.vocab)

        mentioned_relation_ids = [
            (ex.relations, list(set(span.rel_idx for span in ex.spans)))
            for ex in examples
        ]

        # linearized alias surface forms whose relation actually appear in the article
        linearized_aliases: List[List[int]] = [
            [
                self.word_vocab.w2i.get(word, 0)
                for rel_idx, relation in enumerate(relations) if rel_idx in mentioned
                for words in relation.obj_alias
                for word in words.split(" ")
            ]
            for relations, mentioned in mentioned_relation_ids
        ]

        for i, aliases in enumerate(linearized_aliases):
            if len(aliases) == 0:
                linearized_aliases[i].append(0)  # Add an UNK as a dummy, if the seq_len is zero.

        padded_aliases = torch.tensor(pad(linearized_aliases, pad_symbol=1, append=False), dtype=torch.long)

        batches = []
        for split_idx in range(n_splits):
            interval = slice(split_idx * bptt_size, (split_idx + 1) * bptt_size + 1)
            sentence_interval = [sent[interval] for sent in sentences]
            batch = BatchSequence(self.word_vocab, sentence_interval)
            batches.append(batch)

        return padded_aliases, batches
