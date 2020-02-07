import copy
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from utils import loadpkl, savepkl

LOGGER = logging.getLogger(__name__)

__all__ = [
    'Vocab',
    'prepare_unkprob',
]


class Vocab:
    """Simple vocabulary class."""

    def __init__(self, symbols):

        self.w2i = {}
        self.i2w = []
        self.symbols = symbols

        for sym in symbols:
            self.i2w.append(sym)
            self.w2i[sym] = len(self.i2w) - 1

        self.min_freq = None
        self.max_vocab = None
        self.freq = None

    @classmethod
    def from_corpus(cls, corpus: List[str], min_freq: Optional[int] = None, max_vocab: Optional[int] = None,
                    unk: str = '<unk>', lowercase=True,
                    alphabetical=True, prioritized_vocab=None):
        """read corpus (flattened list) and return vocabulary object.
        Parameters:
            corpus (List[str]): Flattened tokens.
            min_freq (int): Minimum frequency of a word.
            max_vocab (int): Maximum number of vocabulary.
            unk (str): token representation of the unknown word.
            lowercase (bool): Default - True.
            alphabetical (bool): Whether sort the vocabulary in alphabetical
                order when the frequencies are equal.
            prioritized_vocab (List[str]): Vocabulary list to rank higher than
                others. This may be critical when word counts tied.
        """
        # Register special symbols beforehand
        vocab = Vocab(symbols=[unk, "<pad>", "<s>", "</s>"])

        min_freq = min_freq or 0
        if lowercase is True:
            corpus = [w.lower() for w in corpus]

        # Always store vocabularies with higher counts
        counts = Counter(corpus)
        frequencies = counts.most_common()  # list of (word, count)

        # Alphabetical, descending order of frequency
        if alphabetical:
            frequencies = sorted(frequencies, key=lambda x: x[0])

        if prioritized_vocab is not None:
            # Words in this list will appear higher
            freq_set = {i[0] for i in frequencies}
            w_in_freq = list(freq_set.intersection(set(prioritized_vocab)))
            w_out_freq = list(freq_set - set(w_in_freq))
            # Reorder by the ones in the given vocab, followed by the ones not
            # in the given vocab
            frequencies = [(w, counts[w]) for w in w_in_freq + w_out_freq]

        freq_vocab = sorted(frequencies, key=lambda x: x[1], reverse=True)

        for word, freq in tqdm(freq_vocab, desc="Constructing vocab..", ncols=80, ascii=True):
            if (word not in vocab.w2i) and (freq >= min_freq) \
                    and ((max_vocab is None) or (len(vocab.i2w) < max_vocab + len(vocab.symbols))):
                vocab.i2w.append(word)
                vocab.w2i[word] = len(vocab.i2w) - 1

        return vocab

    @classmethod
    def from_dict(cls, path: Union[str, Path], gz=False, mode=None):
        """Load and construct dictionary into a file. No restriction about the
        size or freq, just loads given file and set as the dictionary.
        Parameters:
            path (str): Path to the file.
            gz (bool): Whether the file is gz or not.
            mode (str or None): If 'i2w' or 'w2i', the loaded dict is used as the actual mapping of given type, and
                the other mapping is automatically created.
        """
        valid_modes = ['i2w', 'w2i']
        dic: dict = loadpkl(path, gz=gz)

        vocab = Vocab(symbols=[])
        if mode is None:
            vocab.w2i = dic['w2i']
            vocab.i2w = dic['i2w']
        elif mode in valid_modes:
            setattr(vocab, mode, dic)
            setattr(vocab, next(m for m in valid_modes if mode != m), {v: k for k, v in dic.items()})
        elif mode == 'w2i':
            vocab.w2i = dic
        else:
            raise ValueError(f"Invalid mode '{mode}'")
        return vocab

    def numericalize(self, sequence):
        """convert tokens to indices.
        Parameters:
            sequence (List[str]): Token sequence.
        Returns:
            List[int]
        """
        return [self.w2i.get(s, 0) for s in sequence]

    def denumericalize(self, sequence):
        """convert indices to tokens.
        Parameters:
            sequence (List[int]): Numericalized sequence.
        Returns:
            List[str]
        """
        return [self.i2w[s] for s in sequence]

    def save_dict(self, path, gz=False):
        """Save dictionary into a file.
        Parameters:
            path (str): Path to the file.
            gz (bool): Whether to gz or not.
        """
        dic = {'w2i': self.w2i, 'i2w': self.i2w}
        savepkl(dic, path, gz=gz)

    def __len__(self):
        return len(self.i2w)

    def __repr__(self):
        return f"Vocabulary(size={len(self.i2w)}, min_freq={self.min_freq})"


def prepare_unkprob(path: str, vocab: Vocab, uniform_unk: bool = False):
    res = [0.0] * len(vocab)
    with open(path) as f:
        word2prob = {
            l.split('\t')[0]: float(l.split('\t')[1]) for l in f.read().strip().split('\n')
        }

    missed = 0
    for word, idx in vocab.w2i.items():
        if word not in word2prob:
            LOGGER.debug(f"'{word}' was not in the dumped word probs.")
            missed += 1
        res[idx] = word2prob.get(word, 0.0)

    unked_vocabs = list(set(word2prob.keys()) - set(vocab.w2i.keys()))
    if uniform_unk:
        uniform_backoff = -np.log(len(unked_vocabs))
        res += [uniform_backoff for uv in unked_vocabs]
    else:
        res += [word2prob[uv] for uv in unked_vocabs]

    total_w2i = copy.copy(vocab.w2i)  # make a copy!
    total_w2i.update({v: k + len(vocab.w2i) for k, v in enumerate(unked_vocabs)})

    LOGGER.info(f"{missed / len(vocab.w2i)}% vocabulary is covered by char model.")
    return res, total_w2i
