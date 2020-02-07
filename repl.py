import functools
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from dataset import LRLMExample
from dataset.base import KBLMDataset
from dataset.utils import BatchSequence, MatchedSpan, Relation
from models.base import BaseLM
from models.sample_utils import SampledOutput
from nnlib import utils
from nnlib.utils import Logging


def repl(dataset: KBLMDataset, model: BaseLM):
    import re
    from run import compute_batch_loss  # avoid circular import

    print(Logging.color(
        s="Execute `sample(name=\"Barack Obama\")` to generate samples for given entity.\n"
          "Execute `sample(split='test', index=1)` to generate samples for specific data entry.\n"
          "For more configurable settings, please refer to method `models.lrlm.LRLM.sampling_decode`.\n",
        col='green'))

    def get_example(func):
        def find_name(name: str) -> Tuple[str, int]:
            for split in dataset.data:
                try:
                    index = next(idx for idx, ex in enumerate(dataset.data[split])
                                 if name in ' '.join(ex.sentence[:10]))
                    return split, index
                except StopIteration:
                    continue
            else:
                raise ValueError("Name not found!")

        @functools.wraps(func)
        def wrapped(name: Optional[str] = None, split: str = 'test', index: int = 1,
                    start_symbol: str = '<s>', **kwargs):
            if name is not None:
                try:
                    split, index = find_name(f"= {name} =")
                except ValueError:
                    split, index = find_name(name)
            if start_symbol is None:
                start_symbol = '<s>'
            start_symbol = dataset.word_vocab.w2i[start_symbol] if isinstance(start_symbol, str) else start_symbol
            example = dataset.data[split][index]
            try:
                name = ' '.join(example.sentence[2:example.sentence.index('=', 2)])
            except ValueError:
                # probably WikiFacts
                pos = min((example.sentence.index(token) for token in ['is', 'was', '(']
                           if token in example.sentence), default=3)
                name = ' '.join(example.sentence[1:pos])
            if "file" not in kwargs:
                file = sys.stdout
            else:
                file = kwargs["file"]
            print(f"Data split {split}, index {index}, name \"{name}\"", file=file)
            func(example, start_symbol, **kwargs)

        return wrapped

    end_symbol = dataset.word_vocab.w2i['</s>']

    @get_example
    def sample(example, start_symbol, max_length=500, n_tries=1, bptt_size=150, **kwargs):
        (init_batch, batches) = dataset.create_one_batch([example], bptt_size)
        if kwargs.get('generate_unk', False):
            fulli2w = [w for w, i in sorted(dataset.total_w2i.items(), key=lambda x: x[1])]
            unkinfo = (dataset.unk_probs, fulli2w)
            del kwargs["generate_unk"]
        else:
            unkinfo = None

        best_output = None
        for _ in range(n_tries):
            print_info = kwargs.get('print_info', n_tries == 1)
            output: SampledOutput = model.sampling_decode(
                dataset.vocab,
                example,
                begin_symbol=start_symbol,  # this is the actual start symbol in all articles
                end_symbol=end_symbol,
                max_length=max_length,
                color_outputs=True,  # generate colored outputs in terminal
                print_info=print_info,
                unkinfo=unkinfo,
                init_batch=init_batch,
                **kwargs)
            if best_output is None or output.sample_loss < best_output.sample_loss:
                best_output = output

        # format title & subtitles
        if n_tries > 1:
            print(f"Sample loss: {best_output.sample_loss:.3f}, PPL: {math.exp(best_output.sample_loss):.3f}")
            print(f"Complete / incomplete entities: {best_output.complete_copies} / {best_output.incomplete_copies}")
        sentence = re.sub(r'= = (.*?) = = ', r'\n\n== \1 ==\n    ', ' '.join(best_output.sentence))
        sentence = re.sub(r'= (.*?) = ', r'= \1 =\n    ', sentence)
        print(sentence)

    @get_example
    def average_copies(example, start_symbol, max_length=200, n_tries=10, progress=False,
                       show_samples=False, **kwargs):
        complete_copies = utils.SimpleAverage()
        incomplete_copies = utils.SimpleAverage()

        if show_samples:
            sample_kwargs = dict(print_info=True, color_outputs=True, color_incomplete=False, **kwargs)
        else:
            sample_kwargs = dict(print_info=False, color_outputs=False, **kwargs)

        for _ in utils.progress(n_tries, verbose=progress):
            output: SampledOutput = model.sampling_decode(
                dataset.vocab, example, begin_symbol=start_symbol, end_symbol=end_symbol,
                max_length=max_length, **sample_kwargs)
            complete_copies.add(output.complete_copies)
            incomplete_copies.add(output.incomplete_copies)
            if show_samples:
                print(' '.join(output.sentence))
        if isinstance(example, LRLMExample):
            n_gold_rels = sum(int(span.end < max_length) for span in example.spans)
            print(f"Complete / Gold entities: {complete_copies.value()} / {n_gold_rels}")
        else:
            print(f"Complete / Incomplete entities: {complete_copies.value()} / {incomplete_copies.value()}")

    @get_example
    def logprob_article(example, _start_symbol, bptt_size=150, **_kwargs):
        (init_batch, batches) = dataset.create_one_batch([example], bptt_size)

        # log-prob calculation
        def callback(loss: Tensor, batch: BatchSequence, *_extra):
            if batch.spans is None:
                # NKLM
                rel_ids = np.concatenate([[-1], batch.seqs['rel_ids'][0].cpu().numpy(), [-1]])
                diff = np.ediff1d(rel_ids)
                n_rels = np.count_nonzero(np.cumsum(diff[diff != 0]))
            else:
                n_rels = len(batch.spans[0])
            print(f"{n_rels:2d}\t{math.exp(loss.item()):2.3f}")

        print("#rels\tPPL")
        with torch.no_grad():
            model.eval()
            total_loss = compute_batch_loss(model, init_batch, batches, use_unk_probs=True,
                                            callback=callback, evaluate=True)
        total_loss /= sum(batch.ntokens for batch in batches)
        print(f"Total PPL: {math.exp(total_loss)}")

    @get_example
    def posterior_log_probs(example, _, bptt_size=140, n_context=5, max_segments=-1,
                            ignore_single_relation=False, file=sys.stdout):
        from collections import defaultdict
        from dataset.utils import flip_batches, search_paths

        (init_batch, batches) = dataset.create_one_batch([example], bptt_size)
        flipped_batches = flip_batches(batches)

        word_prob: Dict[str, List[np.ndarray]] = {k: [] for k in ["forward", "backward"]}
        marginal_prob: Dict[str, List[np.ndarray]] = {k: [] for k in ["forward", "backward"]}
        posterior_probs: List[Dict[MatchedSpan, Tuple[float, float]]] = []  # list(n_batches) of list(n_spans)
        seq_loss: List[float] = []

        def callback(loss: Tensor, _batch: BatchSequence):
            posterior_probs.append(model.model_cache['posterior_log_probs'][0])
            word_prob['forward'].append(model.model_cache['target_cond_log_probs'][0])
            marginal_prob['forward'].append(model.model_cache['stacked_log_probs'][0])
            seq_loss.append(loss.item())

        def callback_flip(_loss: Tensor, _batch: BatchSequence):
            marginal_prob['backward'].append(model.model_cache['stacked_log_probs'][0])

        with torch.no_grad():
            model.eval()
            compute_batch_loss(model, init_batch, batches, use_unk_probs=True,
                               callback=callback, evaluate=True,
                               calc_loss_kwargs={'dump_posterior_probs': True})

            compute_batch_loss(model, init_batch, flipped_batches, use_unk_probs=True,
                               callback=callback_flip, evaluate=True,
                               calc_loss_kwargs={'dump_posterior_probs': True})

        # credit: http://bayesjumping.net/log-sum-exp-trick/ for without using scipy
        def log_sum_exp(ns: List[int]):
            max_ = np.max(ns)
            sum_exp = np.exp(ns - max_).sum()
            return max_ + np.log(sum_exp)

        n_words = 0
        for seq_idx, (batch, probs_dict) in enumerate(zip(batches, posterior_probs)):
            if max_segments != -1 and seq_idx >= max_segments:
                break

            tokens = batch.raw_sequence[0][1:]
            spans = batch.spans[0]
            if len(spans) == 0:
                continue
            overlap_group = defaultdict(list)
            sorted_spans = sorted(spans, key=lambda x: (x.start, x.end))
            latest = (sorted_spans[0].start, sorted_spans[0].end)  # The most recent span group
            overlap_group[latest] = [sorted_spans[0]]
            for sp in sorted_spans[1:]:
                if sp.start > sp.end or sp.end >= batch.ntokens:
                    continue
                if sp.start <= latest[1]:
                    grp = overlap_group[latest]
                    del overlap_group[latest]
                    latest = (latest[0], max(latest[1], sp.end))
                    overlap_group[latest] = grp + [sp]
                else:
                    latest = (sp.start, sp.end)
                    overlap_group[latest] = [sp]

            if np.any([len(g) > 1 for g in overlap_group.values()]) or not ignore_single_relation:
                print(Logging.color('green',
                                    f"Segment #{seq_idx}: "
                                    f"words {n_words} - {n_words + batch.ntokens}, "
                                    f"ppl = {math.exp(seq_loss[seq_idx]):.4f}"),
                      file=file)
                n_words += batch.ntokens

            for span, group in overlap_group.items():
                if ignore_single_relation and len(group) == 1:
                    continue

                alpha = marginal_prob["forward"][seq_idx][span[0] - 1]
                beta = marginal_prob["backward"][::-1][seq_idx][batch.lengths[0] - (span[1] + 1) - 1]
                # Enumerate all the paths
                paths = search_paths(group, span[0], span[1])
                log_probs = []
                annotations = []
                delimiters = []
                for path in paths:
                    path_anno = []
                    path_delims = []
                    logprob = alpha + beta
                    for hop in path:
                        if hop.rel_typ == -100:  # dummy relation - word transition
                            logprob += word_prob["forward"][seq_idx][span[0]]
                            path_anno.append("word")
                        else:
                            logprob += probs_dict[hop][0]
                            path_anno.append(dataset.rel_vocab.i2w[hop.rel_typ])
                        path_delims += ["   "] * (hop.end - hop.start)
                        if hop.end < span[1]:
                            path_delims.append(" | ")

                    delimiters.append(path_delims)
                    log_probs.append(logprob)
                    annotations.append(path_anno)
                log_denom = log_sum_exp(log_probs)
                normalized_probs = [np.exp(log_prob - log_denom) for log_prob in log_probs]

                l = max(0, span[0] - n_context)
                r = min(batch.ntokens, span[1] + n_context)
                token_string = " ".join([
                    '    ',
                    '... ' if l > 0 else '',
                    ' '.join(tokens[idx] for idx in range(l, span[0])),
                    '|',
                    '   '.join(Logging.color('red', tokens[idx]) for idx in range(span[0], span[1] + 1)),
                    '|',
                    ' '.join(tokens[idx] for idx in range(span[1] + 1, r)),
                    ' ...' if r < batch.ntokens else ''])
                print(token_string, file=file)

                annotation_strings = [" => ".join(a) for a in annotations]
                max_anno_len = max([len(a) for a in annotation_strings])
                max_score_idx = np.argmax(normalized_probs)

                for idx, (delim, anno, prob) in enumerate(zip(delimiters, annotation_strings, normalized_probs)):
                    matched_span_tokens = (" " * token_string.index(" | ") + " | " +
                                           ' _ '.join(tokens[idx] for idx in range(span[0], span[1] + 1)) + " | ")
                    delim_positions = re.finditer(r" _ ", matched_span_tokens)
                    for d, match in zip(delim, delim_positions):
                        pos = match.start(0)
                        matched_span_tokens = matched_span_tokens[:pos] + d + matched_span_tokens[(pos + 3):]
                    score = Logging.color("green", f"  {prob:1.4f}") if idx == max_score_idx else f"  {prob:1.4f}"
                    matched_span_tokens += "  " + f"{anno}{' ' * (max_anno_len - len(anno))}" + score
                    print(matched_span_tokens, file=file)

                print(file=file)

    @get_example
    def span_log_probs(example, _, bptt_size=140, ppl_threshold=200.0,
                       split_len=20, n_context=5, max_segments=-1):
        (init_batch, batches) = dataset.create_one_batch([example], bptt_size)
        rels: List[Relation] = init_batch[0]

        posterior_probs: List[List[Optional[Tuple[float, float]]]] = []  # list(n_batches) of list(n_spans)
        seq_loss: List[float] = []

        # noinspection PyShadowingNames
        def callback(loss: Tensor, batch: BatchSequence) -> None:
            assert batch.spans is not None
            probs_dict: Dict[MatchedSpan, Tuple[float, float]] = model.model_cache['posterior_log_probs'][0]
            probs = [probs_dict.get(span, None) for span in batch.spans[0]]
            posterior_probs.append(probs)
            seq_loss.append(loss.item())

        def color_if_less(val: float, threshold: float, format_str: str = '{:.4f}', color: str = 'yellow'):
            s = format_str.format(val)
            return Logging.color(color, s) if val < threshold else s

        with torch.no_grad():
            model.eval()
            compute_batch_loss(model, init_batch, batches, use_unk_probs=True,
                               callback=callback, evaluate=True, calc_loss_kwargs={'dump_posterior_probs': True})

        n_words = 0
        for seq_idx, (batch, probs) in enumerate(zip(batches, posterior_probs)):
            if max_segments != -1 and seq_idx >= max_segments:
                break
            print(Logging.color('green',
                                f"Segment #{seq_idx}: "
                                f"words {n_words} - {n_words + batch.ntokens}, "
                                f"ppl = {math.exp(seq_loss[seq_idx]):.4f}"))
            n_words += batch.ntokens

            tokens = batch.raw_sequence[0][1:]
            spans = batch.spans[0]
            is_in_span = [False] * batch.ntokens
            for span in spans:
                if span.start > span.end or span.end >= batch.ntokens:
                    continue
                is_in_span[span.start:(span.end + 1)] = [True] * (span.end - span.start + 1)
            for idx in range(0, batch.ntokens, split_len):
                print(f'{idx:3d}:', ' '.join(
                    Logging.color('red', w) if in_span else w
                    for w, in_span in zip(tokens[idx:(idx + split_len)], is_in_span[idx:(idx + split_len)])))
            print()

            for span, prob in sorted(zip(spans, probs)):
                if prob is None:
                    continue
                rel_prob, word_prob = prob
                l = max(0, span.start - n_context)
                r = min(batch.ntokens, span.end + 1 + n_context)
                print(f"[{span.start}, {span.end}]"
                      f" <{dataset.rel_vocab.i2w[span.rel_typ]}>"
                      f" {Logging.color('red', rels[span.rel_idx].obj_alias[span.surface_idx])}"
                      f"{' (alias)' if span.surface_idx > 0 else ''}"
                      f": rel = {color_if_less(math.exp(-rel_prob), ppl_threshold)}"
                      f", word = {color_if_less(math.exp(-word_prob), ppl_threshold)}")
                print('    ',
                      '... ' if l > 0 else '',
                      ' '.join(Logging.color('red', tokens[idx])
                               if span.start <= idx <= span.end else tokens[idx]
                               for idx in range(l, r)),
                      ' ...' if r < batch.ntokens else '')
            print()

    from IPython import embed
    embed()
