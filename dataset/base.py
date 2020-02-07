import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Counter as CounterType, Dict, Generic, List, NamedTuple, Optional, Set, Sized, Tuple, TypeVar

import numpy as np
import torch
from torch import Tensor

from dataset.utils import BatchSequence, Relation, Sentence
from dataset.vocab import Vocab, prepare_unkprob
from nnlib import utils
from nnlib.utils import Logging
from utils import loadpkl, savepkl

__all__ = [
    'Dataset',
    'LMDataset',
    'KBLMDataset',
    'LMExample'
]

LOGGER = logging.getLogger(__name__)

Example = TypeVar('Example', bound=Sized)


class Dataset(Generic[Example]):
    batches: Dict[str, List[Tuple[Any, List[BatchSequence]]]]
    ntokens: Dict[str, int]
    data: Dict[str, List[Example]]

    def __init__(self, path: str, batch_size: int, vocab_dir: str, bptt_size: int,
                 vocab_size: Optional[int], min_freq: Optional[int] = None,
                 include_train: bool = True, create_batches: bool = True,
                 use_only_first_section: bool = False,
                 unk_probs_path: Optional[str] = None, use_upp: bool = False,
                 cache_batches: bool = True, **_kwargs):
        self.batch_size = batch_size
        self.unk_probs = None
        self.bptt_size = bptt_size
        self._unk_probs_path = unk_probs_path
        self._include_train = include_train
        self._use_only_first_section = use_only_first_section
        self.vocab_size = vocab_size
        self.min_freq = min_freq

        self._path = path = Path(path)
        vocab_dir = Path(vocab_dir)

        if not vocab_dir.exists():
            vocab_dir.mkdir(parents=True)
        vocab_file_name = 'vocab'
        if vocab_size is not None:
            vocab_file_name += f'.{vocab_size}'
        if min_freq is not None:
            vocab_file_name += f'.freq{min_freq}'
        vocab_file_name += '.pt'
        self.vocab_path = vocab_path = vocab_dir / vocab_file_name

        loaded_batches = False
        if cache_batches and create_batches and vocab_path.exists() and include_train:
            # Try to load a cached version of the batches if possible; and do not load data
            loaded_batches = self._try_load_cache(path)
        if not loaded_batches:
            # Failed to load cached batches
            self.data = self.read_data(path)

        if not include_train or vocab_path.exists():
            self.word_vocab = torch.load(vocab_path)
            LOGGER.info(f"Word Vocabulary of size {len(self.word_vocab)} loaded from {vocab_path}.")
        else:
            train_corpus = [w for ex in self.data['train'] for w in ex.sentence]  # type: ignore
            self.word_vocab = Vocab.from_corpus(train_corpus,
                                                min_freq=min_freq, max_vocab=vocab_size, lowercase=False)
            torch.save(self.word_vocab, vocab_path)
            LOGGER.info(f"Vocabulary of size {len(self.word_vocab)} constructed, saved to {vocab_path}.")
        self.vocab = self.word_vocab

        if unk_probs_path is not None:
            unk_probs, total_w2i = prepare_unkprob(unk_probs_path, self.word_vocab, uniform_unk=use_upp)
            unk_probs = torch.tensor(unk_probs, dtype=torch.float, requires_grad=False)
            self.unk_probs = unk_probs
            unk_probs[1] = 0  # <pad>
            self.total_w2i = total_w2i

        self._extra_init(loaded_batches)

        if not loaded_batches and create_batches:
            self.create_batches(batch_size, bptt_size)
            if cache_batches and include_train:
                self._save_cache(path)

    def _try_load_cache(self, path: Path) -> bool:
        r"""Try loading a cached dataset from the data directory.

        :param path: The path to data directory.
        :return: Whether loading was successful.
        """
        cache_dir = path / '_cached'
        if not cache_dir.exists():
            return False
        params_index_path = cache_dir / 'params_index.pkl'
        params_index: List[Dict[str, Any]] = loadpkl(params_index_path)
        params = self._get_params()
        index = next((idx for idx, p in enumerate(params_index)
                      if (cache_dir / f'{idx}.pkl').exists() and self._compare_params(p, params)), -1)
        if index != -1:
            load_path = cache_dir / f'{index}.pkl'
            self.batches = loadpkl(load_path)
            self.ntokens = {
                split: sum(batch.ntokens for _, batches in dataset for batch in batches)
                for split, dataset in self.batches.items()
            }
            LOGGER.info(f"Cached dataset loaded from {load_path}, with settings: {params}")
            # check for excluded keys and warn in case of mismatch
            load_params = params_index[index]
            for key in self.EXCLUDE_KEYS:
                if key in params or key in load_params:
                    current = params.get(key, "<does not exist>")
                    loaded = load_params.get(key, "<does not exist>")
                    if current != loaded:
                        LOGGER.info(Logging.color(
                            'red', f"Ignored data param '{key}' mismatch "
                                   f"(current: {current}, loaded: {loaded})"))
            return True
        return False

    def _save_cache(self, path: Path):
        r"""Save the dataset to cache in the data directory.

        :param path: The path to data directory.
        """
        cache_dir = path / '_cached'
        params_index_path = cache_dir / 'params_index.pkl'
        params = self._get_params()
        if not cache_dir.exists():
            cache_dir.mkdir()
            index = 0
            params_index = [params]
        else:
            params_index = loadpkl(params_index_path)
            index = next((idx for idx in range(len(params_index))
                          if not (cache_dir / f'{idx}.pkl').exists()), len(params_index))
            params_index[index:(index + 1)] = [params]  # replace or append
        savepkl(params_index, params_index_path)
        load_path = cache_dir / f'{index}.pkl'
        savepkl(self.batches, load_path)
        LOGGER.info(f"Dataset cached to {load_path}, with settings: {params}")

    def _get_params(self) -> Dict[str, Any]:
        r"""Return a dictionary of data loader parameters that characterize the created batches. Used as lookup
        key when loading from cache.

        :return: The dictionary of parameters.
        """
        params = dict(batch_size=self.batch_size, bptt_size=self.bptt_size, path=str(self._path),
                      vocab_path=str(self.vocab_path), vocab_size=self.vocab_size, min_freq=self.min_freq,
                      klass=self.__class__.__name__)
        # add the rest of boolean flags
        for attr in dir(self):
            if attr.startswith('_'):
                val = getattr(self, attr)
                if isinstance(val, bool):
                    params[attr.lstrip('_')] = val  # remove the leading '_'
        return params

    EXCLUDE_KEYS = {'vocab_path', 'path'}  # ignore 'path' because the cache dir is stored under the path

    def _compare_params(self, saved: Dict[str, Any], current: Dict[str, Any]) -> bool:
        r"""Compare two sets of parameters. All keys except those in `EXCLUDED_KEYS` are expected to match. It is
        allowed for saved parameters to contain

        :return: Whether the two sets of parameters match.
        """
        for key in set(current.keys()) - self.EXCLUDE_KEYS:
            if key not in saved or current[key] != saved[key]:
                return False
        return True

    def _extra_init(self, loaded_batches: bool):
        r"""Extra initialization routine performed before creation of batches.

        :param loaded_batches: Whether batches are already loaded from cache. Subclass implementations should note that
            when this is `True`, `self.data` is not accessible.
        """
        pass

    def read_data(self, path: Path) -> Dict[str, List[Example]]:
        r"""Read data in raw form.

        :return: A dictionary mapping data splits to a list of data examples.
        """
        raise NotImplementedError

    def create_batches(self, batch_size: int, bptt_size: int):
        r"""A general routine to create batches of specified batch size and BPTT length.

        :param batch_size: The number of examples in one batch.
        :param bptt_size: The length for truncated-backprop, i.e. the maximum length of sentences in one batch.
        """
        self.batches = {}
        self.ntokens = {}
        for split, raw_dataset in self.data.items():
            ntokens = 0
            # sort the data by document length
            parts = sorted(raw_dataset, key=len)
            num_batches = utils.ceil_div(len(parts), batch_size)
            all_batches = []
            for batch_idx in utils.progress(num_batches, desc="Creating batches", ascii=True, ncols=80):
                part = parts[(batch_idx * batch_size): ((batch_idx + 1) * batch_size)]
                init_batch, batches = self.create_one_batch(part, bptt_size)
                ntokens += sum(batch.ntokens for batch in batches)
                all_batches.append((init_batch, batches))
            self.batches[split] = all_batches
            self.ntokens[split] = ntokens

        unk_probs = self.unk_probs
        if unk_probs is not None:
            total_w2i = self.total_w2i
            for split, dataset in self.batches.items():
                dataset = utils.progress(dataset, ncols=80, desc=f"Adding unk vocab for {split} set", ascii=True)
                for _, batches in dataset:
                    for batch in batches:
                        batch.add_unk_probs(unk_probs, total_w2i)

    def create_one_batch(self, examples: List[Example], bptt_size: int) -> Tuple[Any, List[BatchSequence]]:
        r"""Create one batch given a list of examples. Note that you may return multiple batches if the sentences are
        too long.

        :param examples: The examples to combine into one batch.
        :param bptt_size: The length for truncated-backprop, i.e. the maximum length of sentences in one batch.
        :return: A tuple of (init_batch, batches):
            - Stuff that's the same for the entire batch, or None.
            - A list containing: the batch cut into sequences no longer than `bptt_size`.
        """
        raise NotImplementedError

    def get_batches(self, split: str, shuffle: bool = True) \
            -> Optional[List[Tuple[Any, List[BatchSequence]]]]:
        r"""Return all batches in the specified data split.

        :param split: The data split.
        :param shuffle: Whether to shuffle the batches. Note that we perform shuffle on the stored `batches`, so the
            result of shuffle is carried.
        :return: A list of batches, or `None` if the split does not exist.
        """
        if split not in self.batches:
            return None
        # handle shuffling here
        if shuffle:
            random.shuffle(self.batches[split])
        return self.batches[split]

    BOUNDARY = "="

    @classmethod
    def find_first_section(cls, tokens) -> int:
        r"""Find the end of first section given a list of tokens. Used for WikiText dataset with option
        `use_only_first_section`.

        :param tokens: The list of tokens.
        :return: The position to the end of first section.
        """
        pointer = 0
        while pointer < len(tokens):
            try:
                i = tokens[pointer:].index(cls.BOUNDARY)
            except ValueError:  # If section headers couldn't be found anymore, use full tokens.
                pointer = len(tokens)
                break
            pointer += i
            if tokens[pointer + 1] == cls.BOUNDARY:
                break
            else:
                pointer += 1

        return pointer


class LMExample(NamedTuple):
    sentence: Sentence

    def __len__(self) -> int:
        return len(self.sentence)


class LMDataset(Dataset[LMExample]):
    def read_data(self, path: Path) -> Dict[str, List[LMExample]]:
        data = {}
        for split in (['train'] if self._include_train else []) + ['valid', 'test']:
            with (path / f'{split}.pkl').open('rb') as data_file:
                dump = pickle.load(data_file)
                examples = [ex[0] for ex in dump]
                if self._use_only_first_section:
                    examples = [ex[:self.find_first_section(ex)] for ex in examples]

                data[split] = [LMExample(['<s>'] + ex + ['</s>']) for ex in examples]
        return data

    def create_one_batch(self, examples: List[LMExample], bptt_size: int) -> Tuple[None, List[BatchSequence]]:
        sentences = [x.sentence for x in examples]
        max_length = max(len(s) for s in sentences)
        n_splits = utils.ceil_div(max_length - 1, bptt_size)

        batches = []
        for split_idx in range(n_splits):
            interval = slice(split_idx * bptt_size, (split_idx + 1) * bptt_size + 1)
            sentence_interval = [s[interval] for s in sentences]
            batch = BatchSequence(self.word_vocab, sentence_interval)
            batches.append(batch)

        return None, batches


class KBLMDataset(Dataset[Example]):
    rel_vocab: Vocab
    vocab: Dict[str, Vocab]
    max_unkrel: int
    entity_count_per_type: Optional[Dict[int, CounterType[int]]]
    alias_vectors: Optional[Tensor]

    def __init__(self, path: str, batch_size: int, vocab_dir: str, bptt_size: int, *args,
                 use_anchor: bool = False, fasttext_model_path: Optional[str] = None,
                 exclude_entity_disamb: bool = False, exclude_alias_disamb: bool = False,
                 **kwargs):
        if exclude_entity_disamb and not kwargs.get('include_train', False):
            raise ValueError("`include_train` must be True, because `exclude_entity_disamb` is set to True")

        self._use_anchor = use_anchor
        self._exclude_entity_disamb = exclude_entity_disamb
        self._exclude_alias_disamb = exclude_alias_disamb
        self._fasttext_model_path = fasttext_model_path
        self._use_fasttext = (fasttext_model_path is not None)

        self.entity_count_per_type = None

        super().__init__(path, batch_size, vocab_dir, bptt_size, *args, **kwargs)

    def _extra_init(self, loaded_batches: bool):
        self.rel_vocab = Vocab.from_dict(self._path / 'rel_names.pkl', mode='i2w')
        self.vocab: Dict[str, Vocab] = {"word": self.word_vocab, "rel": self.rel_vocab}

        self.max_unkrel = max((-rel_typ - 3 for rel_typ in self.rel_vocab.i2w if rel_typ < -3), default=0)

        if self._use_fasttext:
            def _alias_path(name):
                path = Path(self._fasttext_model_path)
                return path.parent / (path.name + f'.{name}')

            # gather all entity aliases and compute fastText embeddings
            alias_dict_path = _alias_path('alias_dict.pkl')
            if alias_dict_path.exists():
                alias_dict: Dict[str, int] = loadpkl(alias_dict_path)
                loaded = True
            else:
                alias_dict = defaultdict(lambda: len(alias_dict))
                loaded = False
            if not loaded_batches:
                for dataset in self.data.values():
                    for example in dataset:
                        for idx, rel in enumerate(example.relations):  # type: ignore
                            example.relations[idx] = rel._replace(  # type: ignore
                                obj_alias=[alias_dict[s] for s in rel.obj_alias])
            if not alias_dict_path.exists():
                alias_dict = dict(alias_dict)
                savepkl(alias_dict, alias_dict_path)

            alias_vectors_path = _alias_path('alias_vectors.pt')
            if not alias_vectors_path.exists() or not loaded:
                import fastText
                ft_model = fastText.load_model(self._fasttext_model_path)
                alias_vectors = []
                alias_list = utils.reverse_map(alias_dict)
                for alias in utils.progress(alias_list, desc="Building fastText vectors", ascii=True, ncols=80):
                    vectors = [ft_model.get_word_vector(w) for w in alias.split()]
                    vectors = np.sum(vectors, axis=0).tolist()
                    alias_vectors.append(vectors)
                alias_vectors = torch.tensor(alias_vectors)
                torch.save(alias_vectors, alias_vectors_path)

        if not loaded_batches and (self._exclude_entity_disamb or self._exclude_alias_disamb):
            # no need to do this if batches are loaded
            if self._exclude_entity_disamb:
                # gather training set stats
                self.entity_count_per_type = self.gather_entity_stats(self.data['train'])

            for dataset in self.data.values():
                for idx in range(len(dataset)):
                    dataset[idx] = self.remove_ambiguity(
                        dataset[idx], self._exclude_entity_disamb, self._exclude_alias_disamb)

    def read_data(self, path: Path) -> Dict[str, List[Example]]:
        raise NotImplementedError

    def gather_entity_stats(self, dataset: List[Example]) -> Dict[int, CounterType[int]]:
        raise NotImplementedError

    def remove_ambiguity(self, ex: Example, entity=False, alias=False) -> Example:
        raise NotImplementedError

    def create_one_batch(self, examples: List[Example], bptt_size: int) \
            -> Tuple[List[List[Relation]], List[BatchSequence]]:
        raise NotImplementedError

    @classmethod
    def remap_rels(cls, relations: List[Relation], use_anchor=True, unk_rels_strategy='params') \
            -> Tuple[List[Relation], List[int]]:
        count = 0
        new_indices = []
        new_rels = []
        for rel in relations:
            if rel.rel_typ == -2 and not use_anchor:
                new_indices.append(-1)
            elif rel.rel_typ < -3 and unk_rels_strategy == 'remove':
                new_indices.append(-1)
            elif rel.rel_typ < -3 and unk_rels_strategy == 'unk':
                new_indices.append(count)
                count += 1
                new_rels.append(Relation(-4, rel.obj_id, rel.obj_alias))
            else:
                new_indices.append(count)
                count += 1
                new_rels.append(rel)
        return new_rels, new_indices

    def remap_entity_ambiguous_rels(self, relations: List[Relation]) -> Tuple[Optional[List[int]], List[Relation]]:
        assert self.entity_count_per_type is not None

        rels_by_typ: Dict[int, List[int]] = defaultdict(list)
        for idx, rel in enumerate(relations):
            rels_by_typ[rel.rel_typ].append(idx)
        remove_ids: Set[int] = set()
        for rel_typ, rels in rels_by_typ.items():
            if len(rels) == 1:
                continue
            counts = [self.entity_count_per_type[rel_typ].get(relations[idx].obj_id, 0) for idx in rels]
            keep_rel_id = rels[int(np.argmax(counts))]
            remove_ids.update(rel_id for rel_id in rels if rel_id != keep_rel_id)

        if len(remove_ids) == 0:
            return None, relations

        # remap relation indices
        count = 0
        new_indices = []
        new_rels = []
        for idx, rel in enumerate(relations):
            if idx in remove_ids:
                new_indices.append(-1)
            else:
                new_indices.append(count)
                count += 1
                new_rels.append(rel)
        return new_indices, new_rels
