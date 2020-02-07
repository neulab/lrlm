import copy
from collections import defaultdict, deque
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import torch
from torch import LongTensor

from dataset.vocab import Vocab

__all__ = [
    'BatchSequence', 'pad',
    'Sentence', 'Span', 'Relation', 'MatchedSpan',
    'flip_batches', 'search_paths',
]

T = TypeVar("T")

Span = Tuple[int, int]
Sentence = List[str]


class Relation(NamedTuple):
    rel_typ: int
    obj_id: int
    # `obj_alias` stores the surface forms of each alias
    # but in case fastText vectors are used, it changes to the vector IDs of each alias
    obj_alias: List[int]


# The raw data loaded uses [l, r), but in our batches we use [l, r]
class MatchedSpan(NamedTuple):
    start: int  # start of span
    end: int  # end of span (+1)
    rel_typ: int  # rel-type ID
    rel_idx: int  # index into `relations` array (relID for example)
    surface_idx: int  # surface form's index into `alias` array


class BatchSequence:
    """Template for the batch of sequences with variable lengths. This class
    handles numericalizing at the same time if needed, hence requires vocab.
    """

    def __init__(self, vocab: Vocab, sequence, spans: Optional[List[List[MatchedSpan]]] = None,
                 **other_seqs: Tuple[List[List[T]], T]):
        """Instantiate a batch of sequence."""
        self.raw_sequence = sequence
        self.spans = spans
        self.unk_probs = None
        self.batch_size = len(self.raw_sequence)

        indices = [vocab.numericalize(s) for s in self.raw_sequence]  # TODO: Improve performance (47.582s)
        sequence = pad_simple(indices, pad_symbol=1)
        self.sequence = torch.from_numpy(sequence[:, :-1])
        self.target = torch.from_numpy(sequence[:, 1:])

        self.seqs: Dict[str, LongTensor] = {}
        self.seqs_pad_symbols: Dict[str, T] = {}
        for name, (seq, pad_symbol) in other_seqs.items():
            self.seqs[name] = torch.from_numpy(pad_simple(seq, pad_symbol=pad_symbol))
            self.seqs_pad_symbols[name] = pad_symbol

        # don't make length -1 for empty sequences
        self.lengths = torch.LongTensor([max(0, len(s) - 1) for s in self.raw_sequence])

        # For loss calculation
        self.ntokens = torch.sum(self.lengths).item()

        self.has_article_end = vocab.w2i['</s>'] in [s[-1] for s in indices if len(s) > 0]

    def __len__(self):
        return self.batch_size

    def to(self, device, persistent=False):
        if persistent:
            obj = self
        else:
            obj = copy.copy(self)
        obj.sequence = self.sequence.to(device)
        obj.target = self.target.to(device)
        obj.lengths = self.lengths.to(device)
        if self.unk_probs is not None:
            obj.unk_probs = self.unk_probs.to(device)
        obj.seqs = {name: seq.to(device) for name, seq in self.seqs.items()}
        return obj

    def chunk(self, n_chunks: int) -> List['BatchSequence']:
        ret = []
        batch_size = self.batch_size // n_chunks
        for start_batch in range(0, self.batch_size, batch_size):
            interval = slice(start_batch, start_batch + batch_size)
            batch = copy.copy(self)
            batch.batch_size = batch_size
            batch.sequence = self.sequence[interval]
            batch.target = self.target[interval]
            batch.lengths = self.lengths[interval]
            batch.ntokens = torch.sum(batch.lengths).item()
            batch.has_article_end = '</s>' in [s[-1] for s in self.raw_sequence[interval] if len(s) > 0]
            if self.unk_probs is not None:
                batch.unk_probs = self.unk_probs[interval]
            batch.seqs = {name: seq[interval] for name, seq in self.seqs.items()}
            ret.append(batch)
        return ret

    def add_rels(self, value):
        self.spans = value

    def add_unk_probs(self, unk_probs, full_vocab):
        # Get the label for targets
        self.unk_probs = torch.zeros_like(self.target, dtype=torch.float, device=self.target.device)
        for i, j in (self.target == 0).nonzero():
            self.unk_probs[i, j] = unk_probs[full_vocab[self.raw_sequence[i][j + 1]]]

    def remove_last_token(self):
        # Pad out all the EOS tokens
        obj = copy.deepcopy(self)
        for idx, raw_s in enumerate(obj.raw_sequence):
            if len(raw_s) > 1 and raw_s[-1] == '</s>':
                end_pos = obj.lengths[idx] - 1
                obj.sequence[idx, end_pos] = 1  # token pad_symbol
                obj.target[idx, end_pos] = 1
                for name in obj.seqs:
                    pad_symbol = obj.seqs_pad_symbols[name]
                    obj.seqs[name][idx, end_pos] = pad_symbol
                obj.ntokens -= 1
        return obj

    @staticmethod
    def _padded_flip(tensor: LongTensor, pad_index: int = 1) -> LongTensor:
        """ Return a flipped and pad-aligned tensor along the second dimension.
        e.g. if PAD = _,
            [[1, 2, 3, 4],            [[4, 3, 2, 1],
             [5, 6, 4, _],      =>     [4, 6, 5, _],
             [2, _, _, _],             [2, _, _, _],
             [2, 3, _, _]]             [3, 2, _, _]]
        """
        flipped_values = [[v for v in seq if v != pad_index]
                          for seq in torch.flip(tensor, 1).tolist()]
        return torch.LongTensor(pad(flipped_values, pad_symbol=1)).to(tensor.device)

    def _flip_spans(self, rels: List[List[MatchedSpan]], lengths: LongTensor):

        flipped: List[List[MatchedSpan]] = [[] for _ in range(self.batch_size)]
        for batch, span in [(b, r) for b, rel in enumerate(rels) for r in rel]:
            actual_len = lengths[batch]
            # calculate flipped start and end
            f_start, f_end = actual_len - span.end, actual_len - span.start
            span = MatchedSpan(f_start, f_end, span.rel_typ, span.rel_idx, span.surface_idx)
            flipped[batch].append(span)
        return flipped

    def flip(self):
        obj = copy.copy(self)
        trg = obj._padded_flip(obj.sequence)
        obj.sequence = obj._padded_flip(obj.target)
        obj.target = trg
        obj.spans = obj._flip_spans(obj.spans, obj.lengths)
        return obj


def flip_batches(batches: List[BatchSequence]):
    """flips the order of batches in an article and the sequence in each batch."""
    new_batches = []
    for b in batches:
        new_batches.append(b.flip())
    return new_batches[::-1]


def search_paths(spans: List[MatchedSpan], begin: int, end: int):
    """search all the possible paths spanning from begin to end, given the list of matched
    relation spans.
    :param spans: List of matched spans.
    :param begin: Begin index of the range to search over.
    :param end: End index of the range to search over.
    """
    # Add word transition
    spans = spans + [MatchedSpan(i, i, -100, -100, -100) for i in range(begin, end + 1)]
    begin2id: Dict[int, List[int]] = defaultdict(list)
    for idx, s in enumerate(spans):
        begin2id[s.start].append(idx)

    completed = []
    queue: Deque[List[MatchedSpan]] = deque()
    for idx in begin2id[begin]:
        queue.appendleft([spans[idx]])
    # queue = [[spans[i]] for i in begin2id[begin]] + [[MatchedSpan(begin, begin, -100, -100, -100)]]
    for path in queue:
        if path[-1].end == end:
            completed.append(path)

    while len(queue) > 0:
        path = queue.pop()
        current_node = path[-1]
        branches = [spans[i] for i in begin2id[current_node.end + 1]]
        # branches.append(MatchedSpan(current_node.end+1, current_node.end+1, -100, -100, -100))  # Add word transition
        for node in branches:
            new_path = path + [node]
            if node.end == end:
                completed.append(new_path)
            elif node.end < end:
                queue.appendleft(new_path)
    return completed


def pad_simple(data: List[List[T]], *, pad_symbol: T) -> np.ndarray:
    """Pad sequences with a padding index.
    :param data: Data to be padded. This can be either text-based, or numericalized form.
    :param pad_symbol: Padding idx or padding symbol.
    """

    batch_size = len(data)
    max_len = max(len(x) for x in data)

    padded = np.full((batch_size, max_len), pad_symbol)
    for idx, seq in enumerate(data):
        padded[idx, :len(seq)] = seq

    return padded


def pad(data: List[List[T]], max_len: Optional[int] = None, *, pad_symbol: T,
        bos: Optional[T] = None, eos: Optional[T] = None, append: bool = True) -> np.ndarray:
    """Pad sequences with a padding index.
    :param data: Data to be padded. This can be either text-based, or numericalized form.
    :param max_len: Maximum length of sequence. Defaults to the maximum length found in data.
    :param pad_symbol: Padding idx or padding symbol.
    :param bos: BOS idx or BOS symbol.
    :param eos: EOS idx or EOS symbol.
    :param append: Whether to append the pads or prepend pads.
    """

    batch_size = len(data)
    if not max_len:
        max_len = max(len(x) for x in data) + int(bos is not None) + int(eos is not None)
    else:
        actual_len = max_len - int(bos is not None) - int(eos is not None)
        if actual_len <= 0:
            raise ValueError(f"'max_len' too small (value = {max_len}, after considering bos/eos = {actual_len})")
        data = [seq[:actual_len] for seq in data]

    padded = np.full((batch_size, max_len), pad_symbol)
    if append:
        if bos is not None:
            padded[:, 0] = bos
            offset = 1
        else:
            offset = 0
        for idx, seq in enumerate(data):
            padded[idx, offset:(offset + len(seq))] = seq
        if eos is not None:
            for idx, seq in enumerate(data):
                padded[idx, offset + len(seq)] = eos
    else:
        if eos is not None:
            padded[:, -1] = eos
            offset = 1
            for idx, seq in enumerate(data):
                padded[idx, -(len(seq) + 1):-1] = seq
        else:
            offset = 0
            for idx, seq in enumerate(data):
                padded[idx, -len(seq):] = seq
        if bos is not None:
            for idx, seq in enumerate(data):
                padded[idx, -(len(seq) + offset)] = bos

    return padded
