import argparse
import math
import os
import random
import re
import sys
from typing import List, Tuple, Dict

import numpy as np

from dataset import LRLMDataset
from dataset.utils import BatchSequence, Relation
from nnlib.utils import Logging


class Average:
    def __init__(self):
        self._value = 0
        self._count = 0

    def add(self, val, cnt=1):
        self._value += val * cnt
        self._count += cnt

    def value(self):
        return self._value / self._count if self._count > 0 else 0.0

    def clear(self):
        self._value = 0
        self._count = 0


def create_bins(counts, max_value, n_bins=10):
    bin_idx = []
    bins = []
    total_counts = sum(counts)
    n_bin = 0
    cumsum = 0
    prev_idx = 0
    for idx, cnt in enumerate(counts):
        cumsum += cnt
        bin_idx.append(n_bin)
        if cumsum / total_counts > (n_bin + 1) / n_bins:
            bins.append((prev_idx, idx + 1))
            prev_idx = idx + 1
            n_bin += 1
    if cumsum > 0:
        n_bin += 1
        bins.append((prev_idx, max_value))

    return bins, bin_idx


def get_diff(results):
    (k1, m1), (k2, m2) = results.items()
    if 'lrlm' in results:
        if k1 == 'lrlm': diff = m1 - m2
        else: diff = m2 - m1
    elif 'nklm' in results:
        if k1 == 'nklm': diff = m1 - m2
        else: diff = m2 - m1
    else:
        diff = m1 - m2

    return diff


def dump_bin_probs(examples_per_criterion, model_probs_per_criterion, bins, n_articles):
    for seg_idx, bin_range in enumerate(bins):
        results = {model: probs[seg_idx].value() for model, probs in model_probs_per_criterion.items()}
        best_result = max(results.values())
        results_str = {model: f'{val:8.4f}' for model, val in results.items()}
        diff = get_diff(results)
        if best_result == 0.0:
            continue
        print(f"{seg_idx:3d} ({bin_range[0]:4d},{bin_range[1]:4d}):"
              f"  examples = {sum(examples_per_criterion[slice(*bin_range)]) / n_articles:.4f}",
              '  '.join(f"{model} = {Logging.color('red', val) if results[model] == best_result else val}"
                        for model, val in results_str.items()),
              f"  diff = {diff:8.4f}")


def aggregate_model_loss(criterion, all_batches, model_probs, n_bins, criterion2bin_idx):
    def get_criterion(criterion):
        if criterion == "nbatches":
            return seg_idx
        elif criterion == "nrels":
            return len(init_batch[ex_offset])
        elif criterion == "nspans":
            return sum(len(batch.spans[ex_offset]) for batch in batches)
        else:
            raise ValueError(f"{criterion} not known.")

    model_prob_per_criterion = {model: [Average() for _ in range(n_bins)] for model in model_probs}
    for model, probs in model_probs.items():
        start_idx = 0
        for init_batch, batches in all_batches:
            for ex_offset in range(len(init_batch)):
                cur_loss = 0.0
                ntokens = 0
                for seg_idx, batch in enumerate(batches):
                    if (batch.lengths[ex_offset].item() == 0) or (len(probs[start_idx + ex_offset]) <= seg_idx):
                        break
                    cur_loss += probs[start_idx + ex_offset][seg_idx] * batch.lengths[ex_offset].item()
                    ntokens += batch.lengths[ex_offset].item()
                else:
                    seg_idx = len(batches)

                cur_loss /= ntokens
                crit_val = get_criterion(criterion)
                model_prob_per_criterion[model][criterion2bin_idx[crit_val]].add(cur_loss)
            start_idx += len(init_batch)

    return model_prob_per_criterion


def bin_probs_by_nsegments(all_batches, model_probs, n_bins=10):
    """bin by the number of segments (i.e. batches/article) in each article."""
    n_batches = sum(len(init_batch) for init_batch, _ in all_batches)
    max_nbatches = max(len(batches) for _, batches in all_batches)
    print(n_batches, max_nbatches)

    examples_per_length = [0 for _ in range(max_nbatches + 1)]
    rel_per_length = [Average() for _ in range(max_nbatches + 1)]
    for init_batch, batches in all_batches:
        for b_idx in range(len(init_batch)):
            seg_idx = next((idx for idx, batch in enumerate(batches) if batch.lengths[b_idx].item() == 0), len(batches))
            examples_per_length[seg_idx] += 1
            rel_per_length[seg_idx].add(len(init_batch[b_idx]))
    rel_per_length = [avg.value() for avg in rel_per_length]

    bins, len2bin_idx = create_bins(examples_per_length, max_nbatches, n_bins)

    model_prob_per_length = aggregate_model_loss("nbatches", all_batches, model_probs, len(bins), len2bin_idx)
    dump_bin_probs(examples_per_length, model_prob_per_length, bins, n_batches)
    # dump_bin_probs(rel_per_length, model_prob_per_length, bins, n_batches)


def bin_probs_by_n_rels(all_batches, model_probs, n_bins=10):
    """bin by the number of relations in each article."""
    n_batches = sum(len(init_batch) for init_batch, _ in all_batches)
    max_nrels = max(len(rels) for init_batch, _ in all_batches for rels in init_batch)
    print(n_batches, max_nrels)

    examples_per_nrels = [0 for _ in range(max_nrels + 1)]
    for init_batch, batches in all_batches:
        for b_idx in range(len(init_batch)):
            nrels = len(init_batch[b_idx])
            examples_per_nrels[nrels] += 1

    bins, val2bin_idx = create_bins(examples_per_nrels, max_nrels, n_bins)

    model_prob_per_nrels = aggregate_model_loss("nrels", all_batches, model_probs, len(bins), val2bin_idx)
    dump_bin_probs(examples_per_nrels, model_prob_per_nrels, bins, n_batches)


def bin_probs_by_n_spans(all_batches, model_probs, bins=10):
    """bin by the number of spans in each article."""
    n_batches = sum(len(init_batch) for init_batch, _ in all_batches)
    max_nspans = max(sum(len(batch.spans[b_idx]) for batch in batches)
                     for init_batch, batches in all_batches for b_idx in range(len(init_batch)))
    print(n_batches, max_nspans)

    examples_per_nspans = [0 for _ in range(max_nspans + 1)]
    for init_batch, batches in all_batches:
        for b_idx in range(len(init_batch)):
            nspans = sum(len(batch.spans[b_idx]) for batch in batches)
            examples_per_nspans[nspans] += 1

    bins, span2bin_idx = create_bins(examples_per_nspans, max_nspans, n_bins)

    model_prob_per_nspans = aggregate_model_loss("nspans", all_batches, model_probs, len(bins), span2bin_idx)
    dump_bin_probs(examples_per_nspans, model_prob_per_nspans, bins, n_batches)


def main():
    random.seed(4731)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--stat-test", action="store_true")
    parser.add_argument("--bin-probs", type=str, choices=["nrels", "nsegments", "nspans"],
                        default="nrels")
    parser.add_argument("--num-bins", type=int, default=10)
    args = parser.parse_args()

    paths = args.exp
    print(paths)

    if "wikitext-short" in paths[0]:
        dataset = LRLMDataset(path='data/lm', batch_size=15, bptt_size=150, vocab_size=None,
                              min_freq=3, vocab_dir='data/vocab/analyze-wt-short',
                              use_only_first_section=True)

    elif "wikitext" in paths[0]:  # wt-full
        dataset = LRLMDataset(path='data/lm', batch_size=30, bptt_size=150, vocab_size=None,
                              min_freq=3, vocab_dir='data/vocab/analyze-wt-full')

    elif "wikifacts" in paths[0]:
        dataset = LRLMDataset(path='data/wikifacts_orig', batch_size=30, bptt_size=150,
                              vocab_size=40000, vocab_dir='data/vocab/analyze-wf')

    paths = {re.search(r"EVAL-(.*?)-", p).group(1): p for p in paths}
    # sanity check
    for k, v in paths.items():
        assert k.lower() in v.lower(), f"{k} not in {v}"

    split_model_probs = {}
    for split in ['train', 'train_sample', 'test', 'valid']:
        model_probs = {}
        for model, path in paths.items():
            path = os.path.join(path, f'prob_dump_{split}.txt')
            if not os.path.exists(path):
                break
            cur_probs = []
            with open(path, 'r') as f:
                for line in f:
                    cur_probs.append([float(x) for x in line.split()])
            model_probs[model] = cur_probs
            print(f"{model} {split}: {len(cur_probs)} articles")
        if len(model_probs) > 0:
            split_model_probs[split] = model_probs

    # do it twice to simulate effect of add unk probs
    all_splits: Dict[str, List[Tuple[List[List[Relation]], List[BatchSequence]]]] = {}
    for _ in range(2):
        all_splits = {}
        all_splits['train'] = dataset.get_batches("train", shuffle=False)
        all_splits['train_sample'] = all_splits['train'][:5]
        all_splits['valid'] = dataset.get_batches("valid", shuffle=False)
        all_splits['test'] = dataset.get_batches("test", shuffle=False)

    split_probs_per_example = {}
    # sanity check: compute valid and test PPL
    for split in ['valid', 'test']:
        split_probs_per_example[split] = {}
        if split not in split_model_probs:
            continue
        for model, probs in split_model_probs[split].items():
            split_probs_per_example[split][model] = []
            start_idx = 0
            total_loss = 0.0
            ntokens = 0
            for init_batch, batches in all_splits[split]:
                for ex_offset in range(len(init_batch)):
                    cur_loss = 0.0
                    cur_ntokens = 0
                    for seg_idx, batch in enumerate(batches):
                        if batch.lengths[ex_offset] == 0:
                            break
                        cur_loss += probs[start_idx + ex_offset][seg_idx] * batch.lengths[ex_offset].item()
                        cur_ntokens += batch.lengths[ex_offset].item()
                    split_probs_per_example[split][model].append(cur_loss / cur_ntokens)
                    total_loss += cur_loss
                    ntokens += cur_ntokens
                start_idx += len(init_batch)
            split_probs_per_example[split][model] = np.asarray(split_probs_per_example[split][model])
            total_loss /= ntokens
            print(f"{model} on {split} set: loss = {total_loss:.4f}, PPL = {math.exp(-total_loss):.4f}")

    if args.stat_test:
        try:
            # Use Wilcoxon signed-rank test following: https://arxiv.org/pdf/1809.01448.pdf
            from scipy.stats import wilcoxon
            for split in ["valid", "test"]:
                model_probs_per_example = split_probs_per_example[split]
                assert len(model_probs_per_example) == 2, "Can only compare exactly two models."

                diff_array = get_diff(model_probs_per_example)
                print(f"p-value for {split}: {wilcoxon(diff_array).pvalue}")
            print()
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install scipy.stats to run statistical testing.")

    for split in split_model_probs:
        print(split)
        if args.bin_probs == "nrels":
            bin_probs_by_n_rels(all_splits[split], split_model_probs[split])
        elif args.bin_probs == "nsegments":
            bin_probs_by_nsegments(all_splits[split], split_model_probs[split])
        elif args.bin_probs == "nspans":
            bin_probs_by_n_spans(all_splits[split], split_model_probs[split])


if __name__ == '__main__':
    if '--ipdb' in sys.argv:
        from IPython.core import ultratb

        sys.excepthook = ultratb.FormattedTB(mode="Context", color_scheme="Linux", call_pdb=1)
        sys.argv.remove('--ipdb')

    main()
