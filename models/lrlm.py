import functools
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Optional, Tuple

from mypy_extensions import TypedDict
import torch
from torch import nn
from torch import LongTensor, Tensor
from torch.nn import functional as F

from arguments import AliasDisamb, LMArguments
from dataset import BatchSequence, LRLMExample, MatchedSpan, Relation, Vocab
from models import sample_utils, utils
from models.base import BaseLM, HiddenState
from models.sample_utils import SampledOutput
from nnlib.utils import Logging, reverse_map
from utils import loadpkl

__all__ = [
    'LRLM',
]

SpanInfo = Tuple[int, MatchedSpan]  # (batch, span)


class LRLM(BaseLM):
    class CacheDict(TypedDict):
        num_facts: LongTensor
        batch_rels: List[List[Relation]]
        fact_embeds: Tensor
        knowledge_embed: Optional[Tensor]

    _cache: CacheDict

    def __init__(self, args: LMArguments, *, vocab_size: int, rel_vocab_size: int, max_unkrel: int):
        self._rel_vocab_size = rel_vocab_size
        self._max_unkrel = max_unkrel
        self._kb_embed_dim = args.kb_embed_dim
        self._fact_embed_dim = args.kb_embed_dim * 2

        pred_input_dim = args.hidden_size + (self._fact_embed_dim if args.use_knowledge_embed else 0)
        super().__init__(args, vocab_size, pred_input_dim=pred_input_dim)

        self._num_layers = args.num_layers
        self._use_knowledge_embed = args.use_knowledge_embed
        self._train_relation_vec = args.train_relation_vec
        self._alias_disamb = args.alias_disamb_strategy

        self.selector = nn.Linear(self._pred_input_dim, 2)

        if args.alias_disamb_strategy is AliasDisamb.FastText:
            # Alias disambiguation using FastText.
            def _alias_path(name):
                path = Path(args.fasttext_model_path)
                return path.parent / (path.name + f'.{name}')

            self.alias_vec = torch.load(_alias_path('alias_vectors.pt')).to(self.device)
            if args.normalize_fasttext_embeds:
                self.alias_vec = self.alias_vec / torch.norm(self.alias_vec, dim=1).unsqueeze(0)
            self.alias_list: List[str] = reverse_map(loadpkl(_alias_path('alias_dict.pkl')))
            self._alias_vec_dim = self.alias_vec.size(1)
            self.hid_to_alias = nn.Linear(self._pred_input_dim, self._alias_vec_dim)

        # Entity disambiguation using entity embeddings.
        self.entity_vec, relation_vec = utils.load_kb_embed(args.path, self.device)
        if self._train_relation_vec:
            rel_vocab_size, rel_embed_dim = relation_vec.size()
            self.relation_embed = nn.Embedding(rel_vocab_size, rel_embed_dim)
        else:
            self.relation_vec = relation_vec

        # (added 1) -2: anchor, -3: topic_itself, < -4: UNKs
        self.special_rel_vecs = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self._kb_embed_dim)) for _ in range(max_unkrel + 2)])
        self.unk_entity_vec = nn.Parameter(torch.Tensor(self._kb_embed_dim))  # -1

        def _(a, b):
            return a if a != -1 else b

        # Entity prediction
        if args.use_rel_mlp:
            self.hid_to_fact: nn.Module = utils.MLP(self._pred_input_dim,
                                                    _(args.fact_key_mlp_hidden_dim, self._fact_embed_dim * 2),
                                                    self._fact_embed_dim, dropout=args.dropout)
        else:
            self.hid_to_fact = nn.Linear(self._pred_input_dim, self._fact_embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for param in [self.unk_entity_vec] + list(self.special_rel_vecs.parameters()):
            param.data.normal_(0.0, 0.02)

    def _get_relation_vec(self, rel_typ: int) -> Tensor:
        if self._train_relation_vec:
            return self.relation_embed.weight[rel_typ]
        return self.relation_vec[rel_typ]

    @staticmethod
    def _gather_spans(rels: List[List[MatchedSpan]], seq_len: int) -> List[List[SpanInfo]]:
        r"""
        Return a list of list of spans which start at each position.
        """
        result: List[List[SpanInfo]] = [[] for _ in range(seq_len)]
        for batch, span in [(b, r) for b, rel in enumerate(rels) for r in rel]:
            # relation exists at the very beginning, can't capture due to seq_len splits
            if span.start < 0 or span.end > seq_len - 1:
                continue
            # A relation that span beyond batches
            elif span.start > span.end:
                continue
            # relation: (bidx, relation)
            result[span.start].append((batch, span))
        return result

    class ComputedLogProbs(NamedTuple):
        selector: Tensor
        word: Tensor
        rel: Tensor
        alias_logits: Optional[Tensor]

    def _compute_log_probs(self, inputs: LongTensor, hidden: Optional[HiddenState] = None,
                           target: Optional[LongTensor] = None) -> Tuple[ComputedLogProbs, HiddenState]:
        batch_size, seq_len = inputs.size()

        # word_embed: (batch_size, seq_len, word_embed_dim)
        word_embed = self.word_embed(inputs)
        # states: (batch_size, seq_len, hidden_dim)
        states, new_hidden = self.rnn(word_embed, hidden)

        if self._use_knowledge_embed:
            knowledge_embed = self._cache['knowledge_embed']
            assert knowledge_embed is not None
            # predictor_input: (batch_size, seq_len, hidden_dim + fact_embed_dim)
            predictor_input = torch.cat([
                states, knowledge_embed.unsqueeze(1).expand(batch_size, seq_len, -1)
            ], dim=2)
        else:
            # predictor_input: (batch_size, seq_len, hidden_dim)
            predictor_input = states

        selector_logits = self.selector(predictor_input)
        # selector_log_probs: (batch_size, seq_len, 2)
        selector_log_probs = F.log_softmax(selector_logits, dim=2)

        if target is not None:
            # target_log_probs: (batch_size, seq_len)
            word_log_probs = -self.word_predictor(predictor_input, target)
        else:
            # word_log_probs: (batch_size, seq_len, vocab_size)
            word_log_probs = self.word_predictor.log_probs(predictor_input)

        fact_embeds = self._cache['fact_embeds']
        num_facts = self._cache['num_facts']
        # fact_key: (batch_size, seq_len, fact_embed_dim)
        fact_key = self.hid_to_fact.forward(predictor_input)
        # rel_logits: (batch_size, seq_len, max_facts)
        rel_logits = torch.bmm(fact_key, fact_embeds.transpose(1, 2))
        # rel_logits_mask: (batch_size, max_facts)
        rel_logits_mask = utils.sequence_mask(num_facts, mask_val=0.0, default_val=-math.inf, device=self.device)
        # rel_log_prob: (batch_size, seq_len, max_facts)
        rel_log_probs = F.log_softmax(rel_logits + rel_logits_mask.unsqueeze(1), dim=-1)

        alias_logits = None
        if self._alias_disamb is AliasDisamb.FastText:
            # alias_logits: (batch_size, seq_len, alias_vec_dim)
            alias_logits = self.hid_to_alias.forward(predictor_input)

        computed_log_probs = self.ComputedLogProbs(
            selector=selector_log_probs, word=word_log_probs, rel=rel_log_probs, alias_logits=alias_logits)
        return computed_log_probs, new_hidden

    def forward(self,  # type: ignore
                inputs: LongTensor, targets: LongTensor, lengths: LongTensor,
                spans: List[List[MatchedSpan]],
                hidden: Optional[HiddenState] = None,
                unk_probs: Optional[Tensor] = None,
                dump_posterior_probs=False) -> Tuple[Tensor, HiddenState]:
        r"""
        :return (per-batch loss, new hidden state, log-probs for each matched span)
        """
        batch_size, seq_len = inputs.size()

        gathered_spans = self._gather_spans(spans, seq_len)

        computed_log_probs, new_hidden = self._compute_log_probs(inputs, hidden, targets)

        target_log_probs = computed_log_probs.word
        if unk_probs is not None:
            target_log_probs = target_log_probs + unk_probs

        word_selector = computed_log_probs.selector[:, :, 0]
        rel_selector = computed_log_probs.selector[:, :, 1]

        # target_cond_log_probs: (batch_size, seq_len)
        target_cond_log_probs = target_log_probs + word_selector

        log_probs_list: List[List[Tensor]] = [[] for _ in range(seq_len)]
        # log_probs: list(seq_len) of (batch_size)
        log_probs: List[Tensor] = []

        # posterior log-probs for each span
        posterior_log_probs: List[Dict[MatchedSpan, Tuple[float, float]]] = [{} for _ in range(batch_size)]

        batch_rels: List[List[Relation]] = self._cache['batch_rels']
        # Slicing the input by a batch of words at each time step
        for idx in range(seq_len):
            if idx == 0:
                prev_prob = torch.zeros(batch_size, dtype=torch.float, device=self.device)
            else:
                if len(log_probs_list[idx - 1]) == 1:
                    prev_prob = log_probs_list[idx - 1][0]
                else:
                    prev_prob = torch.logsumexp(torch.stack(log_probs_list[idx - 1]), dim=0)
                log_probs.append(prev_prob)

            # Update the base selector's probability
            # p(path_with_word) = p(ctxt) * p(selector=word|ctxt) * p(w|ctxt)
            new_prob = prev_prob + target_cond_log_probs[:, idx]

            # Duplicate the new_prob vector and mask it, update the logprob
            # matrix' particular column
            log_probs_list[idx].append(new_prob)

            # p(path_with_rel) = p(ctxt) * p(selector=rel|ctxt) * p(rel=r|ctxt)
            # cur_rel_log_probs: (batch_size, rel_vocab_size)
            cur_rel_log_probs = (prev_prob.view(-1, 1) + rel_selector[:, idx].view(-1, 1) +
                                 computed_log_probs.rel[:, idx, :])

            for batch_idx, span in gathered_spans[idx]:
                rel = batch_rels[batch_idx][span.rel_idx]

                if self._alias_disamb is not AliasDisamb.FastText:
                    alias_log_prob = 0.0
                else:
                    assert computed_log_probs.alias_logits is not None
                    logits = computed_log_probs.alias_logits[batch_idx, idx]
                    aliases = torch.tensor(rel.obj_alias, device=self.device, dtype=torch.long)
                    alias_vecs = torch.index_select(self.alias_vec, dim=0, index=aliases)
                    alias_log_prob = F.log_softmax(torch.mv(alias_vecs, logits), dim=0)[span.surface_idx]

                rel_log_prob = cur_rel_log_probs[batch_idx][span.rel_idx] + alias_log_prob

                adder = torch.full((batch_size,), -math.inf, device=self.device)
                adder[batch_idx] = rel_log_prob

                if dump_posterior_probs:
                    rel_typ = span.rel_idx
                    rel_posterior_log_prob = (alias_log_prob + rel_selector[batch_idx, idx] +
                                              computed_log_probs.rel[batch_idx][idx, rel_typ])
                    word_posterior_log_prob = torch.sum(target_cond_log_probs[batch_idx, idx:(span.end + 1)])
                    posterior_log_probs[batch_idx][span] = (rel_posterior_log_prob.item(),
                                                            word_posterior_log_prob.item())

                log_probs_list[span.end].append(adder)
        log_probs.append(torch.logsumexp(torch.stack(log_probs_list[-1]), dim=0))
        assert len(log_probs) == seq_len

        # stacked_log_probs: (batch_size, seq_len)
        stacked_log_probs = torch.stack(log_probs, dim=1)

        # nonzero_indices = torch.arange(batch_size)[lengths > 0]
        nonzero_lengths = torch.max(torch.zeros(batch_size, dtype=torch.long, device=self.device), lengths - 1)

        loss = -stacked_log_probs[torch.arange(batch_size), nonzero_lengths]

        if dump_posterior_probs:
            self.model_cache.update(
                posterior_log_probs=posterior_log_probs,
                target_cond_log_probs=target_cond_log_probs.cpu().numpy(),
                stacked_log_probs=stacked_log_probs.cpu().numpy(),
            )

        return loss, new_hidden

    def calc_loss(self, batch: BatchSequence, hidden: Optional[HiddenState] = None,
                  use_unk_probs: bool = False, dump_probs: bool = False, dump_posterior_probs: bool = False) \
            -> Tuple[Tensor, HiddenState]:
        assert batch.spans is not None
        batch_loss, new_hidden = self.forward(
            batch.sequence,
            batch.target,
            batch.lengths,
            batch.spans,
            hidden=hidden,
            unk_probs=(None if not use_unk_probs else batch.unk_probs),
            dump_posterior_probs=dump_posterior_probs)

        # Returns the per-token loss
        loss = torch.sum(batch_loss) / batch.ntokens

        if dump_probs:
            log_probs = [-prob / length if length > 0 else 0.0
                         for prob, length in zip(batch_loss.tolist(), batch.lengths.tolist())]
            self.model_cache.update(log_probs=log_probs)

        return loss, new_hidden

    def init_hidden(self, batch_size, init_batch: List[List[Relation]]):
        batch_rels = init_batch

        max_facts = max([len(r) for r in batch_rels])
        num_facts = torch.tensor([len(r) for r in batch_rels])

        # fact_embeds: (batch_size, max_facts, fact_embed_dim)
        fact_embeds = torch.stack([
            torch.cat([torch.cat([
                torch.stack([
                    self._get_relation_vec(rel.rel_typ) if rel.rel_typ >= 0 else self.special_rel_vecs[rel.rel_typ + 1]
                    for rel in relations
                ]),
                torch.stack([
                    self.entity_vec[rel.obj_id] if rel.obj_id != -1 else self.unk_entity_vec
                    for rel in relations
                ])  # batches with fewer relations are padded to enable vectorized operations
            ], dim=1), torch.zeros(max_facts - num_facts[idx], self._fact_embed_dim, device=self.device)])
            for idx, relations in enumerate(batch_rels)
        ])
        knowledge_embed = None
        if self._use_knowledge_embed:
            # knowledge_embed: (batch_size, fact_embed_dim)
            knowledge_embed = torch.stack([torch.mean(e, dim=0) for e in fact_embeds])

        self._cache = {
            'num_facts': num_facts,
            'batch_rels': batch_rels,
            'fact_embeds': fact_embeds,
            'knowledge_embed': knowledge_embed,
        }

        return self.rnn.init_hidden(batch_size)

    @torch.no_grad()
    def sampling_decode(self, vocab: Dict[str, Vocab], example: LRLMExample,
                        begin_symbol: int = 2, end_symbol: int = 5,
                        initial_hidden: Optional[HiddenState] = None, warm_up: Optional[int] = None,
                        max_length: int = 200, greedy: bool = False, topk: Optional[int] = None,
                        print_info: bool = True, color_outputs: bool = False, show_rel_type: bool = True,
                        sanity_check: bool = False, unkinfo: Optional[Tuple[Tensor, List[str]]] = None, **kwargs) \
            -> SampledOutput:
        r"""
        Sampling for LRLM.

        Output format:
        - Red words:       Copied from canonical form of entity.
        - Green words:     Copied from alias form of entity.
        - Yellow words:    Warm-up context.
        - word_[type]:     "word" is an entity of type "type".
        - @-@:             A dash in the original text without spaces around, e.g. M @-@ 82 => M-82.

        :param vocab: Vocabulary containing id2word mapping.
        :param example: The :class:`Example` object of the current topic.
        :param begin_symbol: Start of sentence symbol.
        :param end_symbol: End of sentence symbol. Sampling stops when this symbol is generated.
        :param initial_hidden: If not specified, default hidden states returned by :meth:`init_hidden` is used.
        :param warm_up: Number of tokens to provide as context before performing sampling.
        :param max_length: If generated sentence exceeds specified length, sampling is force terminated.
        :param greedy: If ``True``, use greedy decoding instead of sampling.
        :param topk: If not ``None``, only sample from indices with top-k probabilites.

        :param print_info: If ``True``, print information about sampled result.
        :param color_outputs: If ``True``, include annotations for each output token. Tokens from entities will be
            colored red.
        :param show_rel_type: If ``True``, show relation types for copied entities.

        :param sanity_check: If ``True``, perform sanity check on generated sample.

        :param unkinfo: Precomputed unkprobs and the index-to-vocabulary mapping.

        :return: A tuple of (loss_value, formatted list of words).
        """
        if unkinfo is not None:
            unkprob, unki2w = unkinfo
            unkprob = unkprob[self._vocab_size:]
            unki2w = unki2w[self._vocab_size:]
            normalized_unkprob = F.log_softmax(unkprob, dim=0)

        # noinspection PyPep8Naming
        UNK, INVALID, CANONICAL_IDX, WORD_PREDICTOR, REL_PREDICTOR, EPS = -100, -1, 0, 0, 1, 1e-4

        self.eval()
        self.init_hidden(1, [example.relations])

        word_vocab, rel_vocab = vocab['word'], vocab['rel']

        tensor = functools.partial(sample_utils.tensor, device=self.device)
        sample = functools.partial(sample_utils.sample, greedy=greedy, topk=topk)
        np_sample = functools.partial(sample_utils.np_sample, greedy=greedy, topk=topk)

        # noinspection PyShadowingNames
        def compute_loss(inputs: List[int], spans: List[MatchedSpan],
                         hidden: Optional[HiddenState] = None) -> Tuple[float, HiddenState]:
            batch = SimpleNamespace(
                sequence=tensor(inputs[:-1]),
                target=tensor(inputs[1:]),
                spans=[spans],
                unkprob=None,
                lengths=torch.tensor([len(inputs) - 1], device=self.device),
                ntokens=len(inputs) - 1,
            )
            loss, next_hidden = self.calc_loss(batch, hidden=hidden)  # type: ignore
            return loss.item(), next_hidden

        if warm_up is None:
            inputs = [begin_symbol]
            rel_ids = [INVALID]
            surface_indices = [INVALID]
            spans: List[MatchedSpan] = []
            total_log_prob = 0.0
            marginal_log_prob = 0.0
            hidden = initial_hidden
        else:
            inputs = list(word_vocab.numericalize(example.sentence[:warm_up]))
            rel_ids = [INVALID] * len(inputs)  # assume everything is generated from vocabulary
            surface_indices = [INVALID] * len(inputs)
            spans = [span for span in example.spans if span.end < warm_up]
            loss, hidden = compute_loss(inputs, spans, initial_hidden)
            total_log_prob = -loss * (len(inputs) - 1)
            marginal_log_prob = -loss * (len(inputs) - 1)

        while len(inputs) < max_length and inputs[-1] != end_symbol:
            computed_log_probs, new_hidden = self._compute_log_probs(tensor(inputs[-1]), hidden)
            predictor, selector_loss = sample(computed_log_probs.selector)

            if predictor == REL_PREDICTOR:
                rel_id, rel_loss = sample(computed_log_probs.rel[0])

                if self._alias_disamb is AliasDisamb.FastText:
                    assert computed_log_probs.alias_logits is not None
                    aliases = example.relations[rel_id].obj_alias
                    alias_vecs = self.alias_vec[aliases]
                    surface_log_prob = F.log_softmax(
                        torch.mv(alias_vecs, computed_log_probs.alias_logits.flatten()), dim=0)
                    surface_idx, alias_loss = sample(surface_log_prob)
                    alias = self.alias_list[aliases[surface_idx]]
                else:
                    # can't tell which one under oracle, use the canonical (first) alias
                    surface_idx = 0
                    alias_loss = 0.0
                    alias = example.relations[rel_id].obj_alias[0]  # type: ignore

                # forward the hidden state according to the generated in-vocab tokens
                raw_tokens: List[str] = alias.split()
                token_ids: List[int] = word_vocab.numericalize(raw_tokens)
                if len(raw_tokens) > 1:
                    _, new_hidden = self._compute_log_probs(tensor(token_ids[:-1]), new_hidden)

                # compute marginal probability for current span
                span_inputs = tensor([inputs[-1]] + token_ids[:-1])
                span_computed_log_probs, _ = self._compute_log_probs(span_inputs, hidden)
                word_gen_loss = torch.sum(
                    span_computed_log_probs.selector[0, :, WORD_PREDICTOR] +
                    torch.gather(span_computed_log_probs.word, index=tensor(token_ids).unsqueeze(-1), dim=2).flatten()
                ).item()
                marginal_log_prob += torch.logsumexp(
                    tensor([selector_loss + rel_loss + alias_loss, word_gen_loss]), dim=1
                ).item()

                spans.append(MatchedSpan(len(inputs) - 1, len(inputs) + len(token_ids) - 1,
                                         example.relations[rel_id].rel_typ, rel_id, surface_idx))
                inputs.extend(token_ids)
                rel_ids.extend([rel_id] + [INVALID] * (len(token_ids) - 1))
                surface_indices.extend([surface_idx] + [INVALID] * (len(token_ids) - 1))

                total_log_prob += selector_loss + rel_loss + alias_loss
            elif predictor == WORD_PREDICTOR:
                word, word_loss = sample(computed_log_probs.word)
                total_log_prob += selector_loss + word_loss
                marginal_log_prob += selector_loss + word_loss

                if word == 0 and unkinfo is not None:  # unk
                    unk_idx, unk_loss = np_sample(normalized_unkprob)
                    total_log_prob += unk_loss
                    marginal_log_prob += unk_loss
                    # Ugly multi-purpose use of variables.
                    surface_indices.append(unk_idx)  # Record unk word index in surface_indices.
                    rel_ids.append(UNK)  # Record UNK in rel_ids.
                else:
                    rel_ids.append(INVALID)
                    surface_indices.append(INVALID)

                inputs.append(word)
            else:
                raise ValueError

            hidden = new_hidden

        sample_loss = -total_log_prob / (len(inputs) - 1)
        marginal_loss = -marginal_log_prob / (len(inputs) - 1)
        if print_info:
            print(f"Sample loss: {sample_loss:.3f}, PPL: {math.exp(sample_loss):.3f}")
            print(f"Marginal sample loss: {marginal_loss:.3f}, PPL: {math.exp(marginal_loss):.3f}")
        # Sanity checks
        if sanity_check:
            # noinspection PyTypeChecker
            loss_val, gold_hidden = compute_loss(inputs, spans, initial_hidden)
            assert hidden is not None
            hidden_state_diff = max(torch.max(torch.abs(g - h)).item() for g, h in zip(gold_hidden, hidden))
            if hidden_state_diff > EPS:
                Logging.warn(f"Hidden states do not match. Difference: {hidden_state_diff}")
            if abs(marginal_loss - loss_val) > EPS:
                Logging.warn(f"Marginal loss values do not match. "
                             f"Forward loss: {loss_val}, difference: {abs(marginal_loss - loss_val)}")

        num_rels_generated = sum(int(rel_id != INVALID) for rel_id in rel_ids)
        if print_info:
            print(f"Relations [Generated / Annotated]: "
                  f"[{num_rels_generated} / {len([s for s in example.spans if s.end < max_length])}]")

        words = []
        idx = 0
        copy_count = 0
        while idx < len(inputs):
            is_warm_up = (warm_up is not None and idx < warm_up)
            token_id, rel_id, surface_idx = inputs[idx], rel_ids[idx], surface_indices[idx]
            if rel_id == INVALID:
                token = word_vocab.i2w[token_id]
                idx += 1
            elif rel_id == UNK:
                token = Logging.color('blue', unki2w[surface_idx])
                idx += 1
            else:
                copy_count += 1
                word_id = example.relations[rel_id].obj_alias[surface_idx]  # multiple words
                token = self.alias_list[word_id]
                idx += len(token.split())
                if show_rel_type:
                    token = f"{token}_[{rel_vocab.i2w[example.relations[rel_id].rel_typ]}]"
                if color_outputs and not is_warm_up:
                    token = Logging.color('red' if surface_idx == CANONICAL_IDX else 'green', token)
            if color_outputs and is_warm_up:
                token = Logging.color('yellow', token)
            words.append(token)

        if print_info:
            print(f"# of copied entities: {copy_count}")

        output = SampledOutput(sentence=words, sample_loss=sample_loss,
                               complete_copies=copy_count, incomplete_copies=0)
        return output
