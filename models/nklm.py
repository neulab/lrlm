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
from dataset import BatchSequence, NKLMExample, Relation, Vocab
from models import sample_utils, utils
from models.base import BaseLM, HiddenState
from models.sample_utils import SampledOutput
from nnlib.utils import Logging, reverse_map
from utils import loadpkl

__all__ = [
    'NKLM',
]


class NKLM(BaseLM):
    class CacheDict(TypedDict):
        num_facts: Tensor
        fact_embeds: Tensor
        knowledge_embed: Optional[Tensor]
        alias_word_cnt: Optional[List[List[List[int]]]]
        alias_vecs: Optional[List[Tensor]]
        alias_masks: Optional[List[Tensor]]

    _cache: CacheDict

    def __init__(self, args: LMArguments, *, vocab_size: int, rel_vocab_size: int, max_unkrel: int):
        self._word_embed_dim: int = args.embed_size
        self._hidden_dim: int = args.hidden_size
        self._rel_vocab_size = rel_vocab_size
        self._position_embed_dim: int = args.pos_embed_dim
        self._position_size: int = args.pos_embed_count
        self._kb_embed_dim: int = args.kb_embed_dim
        self._num_layers: int = args.num_layers
        self._dropout: float = args.dropout
        self._use_anchor_rels: bool = args.use_anchor
        self._alias_disamb = args.alias_disamb_strategy

        self._fact_embed_dim: int = self._kb_embed_dim * 2  # concat of TransE embeds for relation & object

        input_dim = self._fact_embed_dim + self._word_embed_dim + self._position_embed_dim
        pred_input_dim = self._hidden_dim + self._fact_embed_dim
        super().__init__(args, vocab_size, input_dim=input_dim, pred_input_dim=pred_input_dim,
                         embed_dim=self._word_embed_dim)

        self._fact_sel_strategy = args.fact_sel_strategy
        self._mask_invalid_pos = args.mask_invalid_pos
        self._use_knowledge_embed = args.use_knowledge_embed

        def _(a, b):
            return a if a != -1 else b

        # All MLP hidden dim specs are copied from the original code
        # Sec 3.2.2 Fact Extraction
        self.fact_key_mlp = utils.MLP(self._hidden_dim + (self._fact_embed_dim if self._use_knowledge_embed else 0),
                                      _(args.fact_key_mlp_hidden_dim, self._fact_embed_dim * 2),
                                      self._fact_embed_dim, dropout=self._dropout)
        # Sec 3.2.3 Selecting Word Generation Source
        self.copy_predictor = utils.MLP(self._pred_input_dim,
                                        _(args.copy_mlp_hidden_dim, self._hidden_dim),
                                        1, dropout=self._dropout)
        self.pos_predictor = utils.MLP(self._pred_input_dim,
                                       _(args.pos_mlp_hidden_dim, self._position_embed_dim * 2),
                                       self._position_size, dropout=self._dropout)

        if args.alias_disamb_strategy is AliasDisamb.FastText:
            def _alias_path(name):
                path = Path(args.fasttext_model_path)
                return path.parent / (path.name + f'.{name}')

            self.alias_vec = torch.load(_alias_path('alias_vectors.pt')).to(self.device)
            if args.normalize_fasttext_embeds:
                self.alias_vec = self.alias_vec / torch.norm(self.alias_vec, dim=1).unsqueeze(0)
            self.alias_list: List[str] = reverse_map(loadpkl(_alias_path('alias_dict.pkl')))
            self._alias_vec_dim = self.alias_vec.size(1)
            self.hid_to_alias = nn.Linear(self._pred_input_dim, self._alias_vec_dim)

        # Embeddings
        self.position_embed = nn.Embedding(self._position_size, self._position_embed_dim)

        # KB related
        self.entity_vec, self.relation_vec = utils.load_kb_embed(args.path, self.device)
        self.naf_vec = nn.Parameter(torch.Tensor(self._fact_embed_dim))  # -1
        # (added 1) -2: anchor, -3: topic_itself, < -4: UNKs
        self.special_rel_vecs = nn.ParameterList([nn.Parameter(torch.Tensor(self._kb_embed_dim))
                                                  for _ in range(max_unkrel + 2)])
        self.unk_entity_vec = nn.Parameter(torch.Tensor(self._kb_embed_dim))  # -1

        self.reset_parameters()

    def reset_parameters(self):
        for param in [self.naf_vec, self.unk_entity_vec] + list(self.special_rel_vecs.parameters()):
            param.data.normal_(0.0, 0.02)

    @staticmethod
    def _pick_embed(embedding: nn.Embedding, indices: LongTensor, mask: torch.ByteTensor):
        indices = indices.clone()
        indices[~mask] = 0
        embed = embedding.forward(indices)
        embed *= mask.to(dtype=torch.float).unsqueeze(-1).expand_as(embed)
        return embed

    def _get_fact_embeds(self, rel_ids: LongTensor) -> Tensor:
        fact_embeds = self._cache['fact_embeds']
        gold_fact_embed = torch.gather(fact_embeds, dim=1,
                                       index=(rel_ids + 1).unsqueeze(-1).expand(*rel_ids.size(), fact_embeds.size(-1)))
        return gold_fact_embed

    def _compute_fact_log_probs(self, inputs: LongTensor, rel_ids: LongTensor, copy_pos: LongTensor,
                                hidden: Optional[HiddenState]) \
            -> Tuple[List[Tensor], Tensor, Tensor, HiddenState]:
        """
        Compute the log probability of selecting each fact.

        .. note::
            Different to the :meth:`forward` method, the final element should be removed from tensors ``rel_ids`` and
            ``copy_pos``. As a result, both tensors should have the same size as ``inputs``.

        :return: A tuple of 4 elements:
            - fact_log_probs  : list(batch_size) of (seq_len, num_facts + 1), log probabilities of selecting each fact.
            - output          : (batch_size, seq_len, hidden_dim), output of LSTM/Transformer.
            - gold_fact_embed : Embeddings of the gold facts selected at each time step. Can be reused later.
            - next_hidden     : The next hidden state.
        """
        batch_size, seq_len = inputs.size()

        """ Fact Embeddings """
        # Fact embeddings are calculated in `init_hidden`
        fact_embeds = self._cache['fact_embeds']

        """ Input """
        # word_embed: (batch_size, seq_len, word_embed_dim)
        # pos_embed: (batch_size, seq_len, pos_embed_dim)
        word_embed: Tensor = self._pick_embed(self.word_embed, inputs, mask=(rel_ids == -1))
        pos_embed: Tensor = self._pick_embed(self.position_embed, copy_pos, mask=(copy_pos != -1))
        # gold_fact_embed: (batch_size, seq_len, fact_embed_dim)
        gold_fact_embed: Tensor = self._get_fact_embeds(rel_ids)
        # input_embed: (batch_size, seq_len, input_dim)
        input_embed = torch.cat([gold_fact_embed, word_embed, pos_embed], dim=2)

        """ LSTM """
        # output: (batch_size, seq_len, hidden_dim)
        # next_hidden: tuple[2] of (num_layers, batch_size, hidden_dim)
        output, next_hidden = self.rnn.forward(input_embed, hidden)

        """ Fact Extraction """
        if self._use_knowledge_embed:
            knowledge_embed = self._cache['knowledge_embed']
            assert knowledge_embed is not None
            # predictor_input: (batch_size, seq_len, hidden_dim + fact_embed_dim)
            predictor_input = torch.cat([
                knowledge_embed.unsqueeze(1).expand(batch_size, seq_len, -1), output
            ], dim=2)
        else:
            # predictor_input: (batch_size, seq_len, hidden_dim)
            predictor_input = output
        # fact_queries: (batch_size, seq_len, fact_embed_dim)
        fact_queries = self.fact_key_mlp.forward(predictor_input)
        # Calculate log-prob per batch example, because there could be different number of facts
        fact_log_probs = []
        num_facts = self._cache['num_facts']
        all_fact_logits = torch.bmm(fact_queries, fact_embeds.transpose(1, 2))
        for batch in range(batch_size):
            # cur_fact_log_probs: (seq_len, num_facts + 1)
            cur_fact_log_probs = F.log_softmax(all_fact_logits[batch, :, :(num_facts[batch] + 1)], dim=1)
            fact_log_probs.append(cur_fact_log_probs)

        return fact_log_probs, output, gold_fact_embed, next_hidden

    def _compute_generate_log_probs(self, lstm_output: Tensor, next_fact_embed: Tensor, rel_ids: LongTensor,
                                    target: Optional[LongTensor] = None) \
            -> Tuple[Tensor, Optional[List[Tensor]], Tensor, Tensor]:
        """
        Compute the log probability of generating each token.

        :return: A tuple of 4 elements, each representing different log-probabilities:
            - copy_indicator  : (batch_size, seq_len)
            - alias_log_probs : list(batch_size) of (seq_len, max_facts + 1), None when alias disamb. is not FastText.
            - pos_log_probs   : (batch_size, seq_len, position_size)
            - vocab_log_probs : (batch_size, seq_len, vocab_size)
          Or, if `target` is specified:
            - vocab_loss      : (batch_size, seq_len)
        """
        # queries_input: (batch_size, seq_len, hidden_dim + fact_embed_dim)
        queries_input = torch.cat([lstm_output, next_fact_embed], dim=2)

        # copy_indicator: (batch_size, seq_len)
        copy_indicator = torch.sigmoid(self.copy_predictor.forward(queries_input).squeeze(-1))

        alias_log_probs = None
        if self._alias_disamb is AliasDisamb.FastText:
            batch_size, seq_len, _ = lstm_output.size()
            alias_vecs = self._cache['alias_vecs']
            alias_masks = self._cache['alias_masks']
            assert alias_vecs is not None and alias_masks is not None
            # alias_input: (batch_size, seq_len, alias_vec_dim)
            alias_input = self.hid_to_alias.forward(queries_input)
            # alias_log_probs: list(batch_size) of (seq_len, max_facts + 1)
            # note that `alias_log_probs` will be masked where `rel_ids == -1`, so we can fill in garbage values
            alias_log_probs = []
            for b in range(batch_size):
                num_aliases = alias_masks[b].size(1)
                # cur_rel_ids: (seq_len)
                cur_rel_ids = rel_ids[b, 1:]
                # nonzero_*: (valid_seq_len)
                valid_indices = (cur_rel_ids != -1).nonzero()
                if valid_indices.size(0) == 0:
                    alias_log_probs.append(torch.empty(seq_len, num_aliases, device=self.device))
                    continue
                else:
                    valid_indices = valid_indices.squeeze(1)
                valid_rel_ids = cur_rel_ids[valid_indices]
                # cur_alias_vecs: (valid_seq_len, max_aliases + 1, alias_vec_dim)
                cur_alias_vecs = torch.index_select(alias_vecs[b], dim=0, index=valid_rel_ids)
                # cur_alias_logits: (valid_seq_len, max_aliases + 1)
                cur_alias_logits = torch.bmm(cur_alias_vecs, alias_input[b, valid_indices].unsqueeze(2)).squeeze(2)
                # valid_alias_log_probs: (valid_seq_len, max_aliases + 1)
                valid_alias_log_probs = F.log_softmax(cur_alias_logits + alias_masks[b][valid_rel_ids], dim=1)
                # cur_alias_log_probs: (valid_seq_len, max_aliases + 1)
                cur_alias_log_probs = torch.empty(seq_len, num_aliases, device=self.device)
                cur_alias_log_probs.index_copy_(dim=0, index=valid_indices, source=valid_alias_log_probs)
                alias_log_probs.append(cur_alias_log_probs)

        # pos_log_probs: (batch_size, seq_len, position_size)
        pos_log_probs = F.log_softmax(self.pos_predictor.forward(queries_input), dim=2)
        if target is not None:
            # vocab_loss: (batch_size, seq_len)
            vocab_loss = self.word_predictor.forward(queries_input, target)
            return copy_indicator, alias_log_probs, pos_log_probs, vocab_loss
        else:
            # vocab_log_probs: (batch_size, seq_len, vocab_size)
            vocab_log_probs = self.word_predictor.log_probs(queries_input)
            return copy_indicator, alias_log_probs, pos_log_probs, vocab_log_probs

    class ComputedLogProbs(NamedTuple):
        fact: List[Tensor]
        copy_indicator: Tensor
        alias: Optional[List[Tensor]]
        vocab: Tensor
        pos: Tensor

    def _compute_log_probs(self, inputs: LongTensor, rel_ids: LongTensor, copy_pos: LongTensor,
                           hidden: Optional[HiddenState], target: Optional[LongTensor] = None,
                           use_argmax_facts=False) -> Tuple[ComputedLogProbs, HiddenState]:
        """
        Compute the log probability for each loss component.
        """

        fact_log_probs, output, gold_fact_embed, next_hidden = \
            self._compute_fact_log_probs(inputs, rel_ids[:, :-1], copy_pos[:, :-1], hidden)

        # next_fact_embed: (batch_size, seq_len, fact_embed_dim)
        if use_argmax_facts:
            argmax_facts = torch.tensor([torch.argmax(log_probs, dim=1) - 1 for log_probs in fact_log_probs],
                                        device=self.device)
            next_fact_embed = self._get_fact_embeds(argmax_facts)
        else:
            # final_gold_fact_embed: (batch_size, fact_embed_dim)
            final_gold_fact_embed: Tensor = self._get_fact_embeds(rel_ids[:, -1:])
            next_fact_embed = torch.cat([gold_fact_embed[:, 1:], final_gold_fact_embed], dim=1)

        copy_indicator, alias_log_probs, pos_log_probs, vocab_log_probs = \
            self._compute_generate_log_probs(output, next_fact_embed, rel_ids, target)

        computed_log_probs = self.ComputedLogProbs(
            fact=fact_log_probs, copy_indicator=copy_indicator, alias=alias_log_probs,
            vocab=vocab_log_probs, pos=pos_log_probs)
        return computed_log_probs, next_hidden

    def forward(self,  # type: ignore
                inputs: LongTensor, targets: LongTensor, masks: Tensor,
                rel_ids: LongTensor, copy_pos: LongTensor, surface_indices: LongTensor,
                hidden: Optional[HiddenState],
                pos_mask: Optional[Tensor] = None,
                unk_probs: Optional[Tensor] = None) -> Tuple[Tensor, HiddenState]:
        """
        :param inputs: (batch_size, seq_len), includes </s>.
        :param targets: (batch_size, seq_len), includes </s>.
        :param masks: (batch_size, seq_len), sequence masks.

        :param rel_ids: (batch_size, seq_len + 1), relation index (of `rels` list) at each time step. -1 for NaF.
        :param copy_pos: (batch_size, seq_len + 1), copy position at each time step, -1 for NaF.
        :param surface_indices: (batch_size, seq_len + 1), index of surface form for each target token.

        :param hidden: Initial hidden state.

        :param pos_mask: (batch_size, seq_len, position_size), used for masking invalid copy positions.

        :param unk_probs: (batch_size, seq_len), UNK-probs for each target token (to be added to prob of choosing UNK).

        :return (per-batch loss, new hidden state, log-probs for each element)
        """
        batch_size, seq_len = inputs.size()

        computed_log_probs, state = self._compute_log_probs(
            inputs, rel_ids, copy_pos, hidden, target=targets,
            use_argmax_facts=(not self.training and self._fact_sel_strategy == 'argmax'))

        # fact_loss: (batch_size, seq_len)
        fact_loss = torch.stack([
            F.nll_loss(log_probs, rel_ids[batch, 1:] + 1, reduction='none')  # add one to rel_ids
            for batch, log_probs in enumerate(computed_log_probs.fact)
        ])

        # indicator_loss: (batch_size, seq_len), 1 for copy, 0 for not
        indicator_loss = F.binary_cross_entropy(
            computed_log_probs.copy_indicator,
            (rel_ids[:, 1:] != -1).to(dtype=torch.float), reduction='none')

        surface_loss = 0
        if self._alias_disamb is AliasDisamb.FastText:
            assert computed_log_probs.alias is not None
            surface_loss = torch.stack([
                F.nll_loss(log_probs, surface_indices[b, 1:], ignore_index=-1, reduction='none')
                for b, log_probs in enumerate(computed_log_probs.alias)
            ])

        vocab_loss = computed_log_probs.vocab
        if unk_probs is not None:
            vocab_loss = vocab_loss - unk_probs

        pos_log_probs = computed_log_probs.pos
        if self._mask_invalid_pos:
            assert pos_mask is not None
            # re-normalize the log-probs
            pos_log_probs = pos_log_probs * pos_mask + (-math.inf) * (1.0 - pos_mask)
            pos_log_probs_sum = torch.logsumexp(pos_log_probs, dim=2, keepdim=True)
            # we mask `pos_log_probs` again, because there could be invalid copies in data
            # (e.g. WikiFacts has no alias info while mentions are annotated for aliases)
            pos_log_probs = (pos_log_probs - pos_log_probs_sum) * pos_mask
        # pos_loss: (batch_size, seq_len)
        pos_loss = F.nll_loss(
            pos_log_probs.view(batch_size * seq_len, -1),
            copy_pos[:, 1:].contiguous().view(-1), ignore_index=-1, reduction='none'
        ).view(batch_size, seq_len)

        """ Final Loss """
        # *loss: (batch_size, seq_len)
        generate_loss = torch.where(rel_ids[:, 1:] == -1, vocab_loss,
                                    pos_loss + surface_loss)  # surface_loss only added for relations
        if not self.training and self._fact_sel_strategy == 'argmax':
            total_loss = generate_loss
        else:
            total_loss = fact_loss + indicator_loss + generate_loss
        total_loss = total_loss * masks
        loss = torch.sum(total_loss, dim=1)

        return loss, state

    def calc_loss(self, batch: BatchSequence, hidden: Optional[HiddenState] = None,
                  use_unk_probs=False, dump_probs=False) -> Tuple[Tensor, HiddenState]:
        sequence, targets = batch.sequence, batch.target

        masks = (targets != 1).to(dtype=torch.float)  # pad symbol is 1

        rel_ids, copy_pos, surface_indices = \
            [batch.seqs[name] for name in ['rel_ids', 'copy_pos', 'surface_indices']]

        pos_mask = None
        if self._mask_invalid_pos:
            alias_word_cnt = self._cache['alias_word_cnt']
            assert alias_word_cnt is not None
            pos_mask = torch.zeros(*targets.size(), self._position_size, dtype=torch.float, device=self.device)
            batch_size = targets.size(0)
            for b in range(batch_size):
                for idx, (rel_id, surface_idx) in enumerate(zip(rel_ids[b, 1:], surface_indices[b, 1:])):
                    if rel_id != -1:
                        pos_mask[b, idx, :alias_word_cnt[b][rel_id][surface_idx]] = 1.0

        assert (sequence.shape == targets.shape and rel_ids.shape == copy_pos.shape and
                sequence.shape[0] == rel_ids.shape[0] and sequence.shape[1] + 1 == rel_ids.shape[1])

        # def if_then(condition: torch.ByteTensor, statement: torch.ByteTensor) -> bool:
        #     return (~condition | statement).all().item()  # P→Q => ¬P ∨ Q
        #
        # assert if_then(rel_ids == -1, copy_pos == -1) and if_then(copy_pos != -1, rel_ids != -1)

        assert torch.sum(masks) == batch.ntokens

        batch_loss, next_state = self.forward(sequence, targets, masks,
                                              rel_ids, copy_pos, surface_indices,
                                              hidden,
                                              pos_mask=pos_mask,
                                              unk_probs=(None if not use_unk_probs else batch.unk_probs))
        # Returns the per-token loss
        loss = torch.sum(batch_loss) / batch.ntokens

        if dump_probs:
            log_probs = [-prob / length if length > 0 else 0.0
                         for prob, length in zip(batch_loss.tolist(), batch.lengths.tolist())]
            self.model_cache.update(log_probs=log_probs)

        return loss, next_state

    def init_hidden(self, batch_size: int, init_batch: List[List[Relation]]) -> HiddenState:
        """
        Initialize stuff shared within whole batch, and return initial hidden states.

        topics: (batch_size), topic ID of each example.
        batch_rels: facts for each topic. Each a list of tuples (rel_id, obj_id, alias).
            - rel_id: {-1: NaF, -2: anchor, -3: topic_itself}
            - obj_id: -1: NaF / UNK
            - obj_alias: list of aliases for the relation.
        """
        batch_rels = init_batch

        """ Fact Embeddings """
        max_facts = max([len(r) for r in batch_rels])
        num_facts = torch.tensor([len(r) for r in batch_rels])

        # insert NaF as the first vector, and add 1 to all rel_ids
        # this is because we need to compute the CE loss including NaF, and -1 can't be used as target
        # fact_embeds: (batch_size, max_facts + 1, fact_embed_dim)
        fact_embeds = torch.stack([
            torch.cat([torch.cat([
                torch.stack([self.naf_vec[:self._kb_embed_dim]] + [
                    self.relation_vec[rel.rel_typ] if rel.rel_typ >= 0 else self.special_rel_vecs[rel.rel_typ + 1]
                    for rel in relations
                ]),
                torch.stack([self.naf_vec[self._kb_embed_dim:]] + [
                    self.entity_vec[rel.obj_id] if rel.obj_id != -1 else self.unk_entity_vec
                    for rel in relations
                ])  # batches with fewer relations are padded to enable vectorized operations
            ], dim=1), torch.zeros(max_facts + 1 - num_facts[idx], self._fact_embed_dim, device=self.device)])
            for idx, relations in enumerate(batch_rels)
        ])

        knowledge_embed = None
        if self._use_knowledge_embed:
            # knowledge_embed: (batch_size, fact_embed_dim)
            knowledge_embed = torch.stack([torch.mean(e, dim=0) for e in fact_embeds])

        alias_word_cnt = None
        if self._mask_invalid_pos:
            # alias_word_cnt: list(batch_size) of list(num_facts) of n_alias
            alias_word_cnt = [[
                [len(self.alias_list[alias].split()) for alias in rel.obj_alias]
                for rel in relations
            ] for relations in batch_rels]

        alias_vecs = alias_masks = None
        if self._alias_disamb is AliasDisamb.FastText:
            max_aliases = [max(len(rel.obj_alias) for rel in relations) for relations in batch_rels]
            # alias_vecs: list(batch_size) of (max_facts + 1, max_aliases, alias_vec_dim)
            # alias_masks: list(batch_size) of (max_facts + 1, max_aliases)
            # the masks set invalid positions to -inf, used during log_softmax
            alias_vecs = []
            alias_masks = []
            for b, relations in enumerate(batch_rels):
                aliases = torch.zeros(len(relations), max_aliases[b] + 1, device=self.device, dtype=torch.long)
                mask = torch.full((len(relations), max_aliases[b] + 1), -math.inf, device=self.device)
                for idx, rel in enumerate(relations):
                    aliases[idx, :len(rel.obj_alias)] = torch.tensor(rel.obj_alias, device=self.device)
                    mask[idx, :len(rel.obj_alias)] = 0
                vectors = F.embedding(aliases, self.alias_vec)
                alias_vecs.append(vectors)
                alias_masks.append(mask)

        self._cache = {
            'num_facts': num_facts,
            'fact_embeds': fact_embeds,
            'knowledge_embed': knowledge_embed,
            'alias_word_cnt': alias_word_cnt,
            'alias_vecs': alias_vecs,
            'alias_masks': alias_masks,
        }

        return self.rnn.init_hidden(batch_size)

    @torch.no_grad()
    def sampling_decode(self, vocab: Dict[str, Vocab], example: NKLMExample,
                        begin_symbol: int = 2, end_symbol: int = 5,
                        initial_hidden: Optional[HiddenState] = None, warm_up: Optional[int] = None,
                        max_length: int = 200, greedy=False, topk=None,
                        fill_incomplete=False, allow_invalid_pos=False,
                        print_info=True, color_outputs=False, color_incomplete=True,
                        show_ellipses=True, show_rel_type=True, show_copy_pos=False,
                        sanity_check=False, unkinfo: Optional[Tuple[Tensor, List[str]]] = None, **kwargs) \
            -> SampledOutput:
        """
        Sampling for NKLM.

        Output format:
        - Red words:       Copied from canonical form of entity.
        - Green words:     Copied from alias form of entity.
        - Yellow words:    Warm-up context.
        - word_[type]:     "word" is an entity of type "type".
        - word...(a_b_c):  "word" is a partially copied entity with remaining suffix "a b c".
        - (a_b_c)...word:  "word" is a partially copied entity with remaining prefix "a b c".
        - @-@:             A dash in the original text without spaces around, e.g. M @-@ 82 => M-82.
        - <X>:             A token copied from an invalid position of an entity.

        :param vocab: Vocabulary containing id2word mapping.
        :param example: The :class:`Example` object of the current topic.
        :param begin_symbol: Start of sentence symbol.
        :param end_symbol: End of sentence symbol. Sampling stops when this symbol is generated.
        :param initial_hidden: If not specified, default hidden states returned by :meth:`init_hidden` is used.
        :param warm_up: Number of tokens to provide as context before performing sampling.
        :param max_length: If generated sentence exceeds specified length, sampling is force terminated.
        :param greedy: If ``True``, use greedy decoding instead of sampling.
        :param topk: If not ``None``, only sample from indices with top-k probabilites.
        :param fill_incomplete: If ``True``, entities that are partially copied will be completed.
        :param allow_invalid_pos: If ``True``, allowing copying from invalid positions, and use <unk> as input.

        :param print_info: If ``True``, print information about sampled result.
        :param color_outputs: If ``True``, include annotations for each output token. Tokens from entities will be
            colored red.
        :param color_incomplete: If ``True`` and ``color_outputs`` is ``True``, also color partially copied entities.
        :param show_ellipses: If ``True``, show ellipses at beginning or end of partially copied entities.
        :param show_rel_type: If ``True``, show relation types for copied entities.
        :param show_copy_pos: If ``True``, show the position from which the entity tokens are copied.

        :param sanity_check: If ``True``, perform sanity check on generated sample.

        :return: A tuple of (loss_value, formatted list of words).
        """

        if unkinfo is not None:
            unkprob, unki2w = unkinfo
            unkprob = unkprob[self._vocab_size:]
            unki2w = unki2w[self._vocab_size:]
            normalized_unkprob = F.log_softmax(unkprob, dim=0)

        # noinspection PyPep8Naming
        UNK, INVALID, UNK_TOKEN, CANONICAL_IDX, EPS = -100, -1, 0, 0, 1e-4

        self.eval()
        self.init_hidden(1, [example.relations])

        word_vocab, rel_vocab = vocab['word'], vocab['rel']

        tensor = functools.partial(sample_utils.tensor, device=self.device)
        randint = sample_utils.randint
        sample = functools.partial(sample_utils.sample, greedy=greedy, topk=topk)
        np_sample = functools.partial(sample_utils.np_sample, greedy=greedy, topk=topk)

        # noinspection PyShadowingNames
        def compute_loss(inputs: List[int], rel_ids: List[int], copy_pos: List[int], surface_indices: List[int],
                         hidden: Optional[HiddenState] = None) -> Tuple[float, HiddenState]:
            batch = SimpleNamespace(
                sequence=tensor(inputs[:-1]),
                target=tensor(inputs[1:]),
                unkprob=None,
                seqs={'rel_ids': tensor(rel_ids),
                      'copy_pos': tensor(copy_pos),
                      'surface_indices': tensor(surface_indices)},
                ntokens=len(inputs) - 1,
            )
            loss, next_hidden = self.calc_loss(batch, hidden=hidden)  # type: ignore
            return loss.item(), next_hidden

        # Initialization
        if warm_up is None:
            inputs = [begin_symbol]
            rel_ids = [INVALID]
            copy_pos = [INVALID]
            surface_indices = [INVALID]
            total_log_prob = 0.0
            hidden = initial_hidden
        else:
            inputs = list(word_vocab.numericalize(example.sentence[:warm_up]))
            rel_ids = list(example.rel_ids[:warm_up])
            copy_pos = list(example.copy_pos[:warm_up])
            surface_indices = list(example.surface_indices[:warm_up])
            total_log_prob, hidden = compute_loss(inputs, rel_ids, copy_pos, surface_indices, initial_hidden)
            total_log_prob = -total_log_prob * (len(inputs) - 1)

        # Sampling procedure
        while len(inputs) < max_length and inputs[-1] != end_symbol:
            fact_log_probs, output, _, next_hidden = \
                self._compute_fact_log_probs(tensor(inputs[-1]), tensor(rel_ids[-1]), tensor(copy_pos[-1]), hidden)
            rel_id, fact_loss = sample(fact_log_probs[0])
            rel_id -= 1
            total_log_prob += fact_loss

            # next_fact_embed: (1, 1, fact_embed_dim)
            next_fact_embed = self._get_fact_embeds(tensor(rel_id))
            copy_indicator, alias_log_probs, pos_log_probs, vocab_log_probs = \
                self._compute_generate_log_probs(output, next_fact_embed, tensor([rel_ids[-1], rel_id]))
            if torch.bernoulli(copy_indicator).item():
                total_log_prob += torch.log(copy_indicator).item()
                assert rel_id != -1
                # copy entity
                aliases = example.relations[rel_id].obj_alias
                if self._alias_disamb is AliasDisamb.FastText:
                    assert alias_log_probs is not None
                    surface_idx, surface_loss = sample(alias_log_probs[0])
                else:
                    surface_idx, surface_loss = 0, 0.0
                alias = self.alias_list[aliases[surface_idx]]

                entity: List[str] = alias.split()
                # normalization not required
                # TODO: keep consistent with _mask_invalid_pos setting
                pos, pos_loss = sample(pos_log_probs if allow_invalid_pos else pos_log_probs.squeeze()[:len(entity)])
                if self._mask_invalid_pos:
                    pos_loss -= torch.logsumexp(pos_log_probs.squeeze()[:len(entity)], dim=0).item()
                total_log_prob += surface_loss + pos_loss
                token = UNK_TOKEN if pos >= len(entity) else word_vocab.w2i.get(entity[pos], UNK_TOKEN)
            else:
                total_log_prob += torch.log(1.0 - copy_indicator).item()
                assert rel_id == -1
                # generate word
                token, token_loss = sample(vocab_log_probs)
                total_log_prob += token_loss
                pos = INVALID
                surface_idx = INVALID

                if token == 0 and unkinfo is not None:  # unk
                    unk_idx, unk_loss = np_sample(normalized_unkprob)
                    total_log_prob += unk_loss
                    # Ugly multi-purpose use of variables.
                    surface_idx = unk_idx  # Record unk word index in surface_indices.
                    rel_id = UNK  # Record UNK in rel_ids.

            inputs.append(token)
            rel_ids.append(rel_id)
            copy_pos.append(pos)
            surface_indices.append(surface_idx)
            hidden = next_hidden

        sample_loss = -total_log_prob / (len(inputs) - 1)
        if print_info:
            print(f"Sample loss: {sample_loss:.3f}, PPL: {math.exp(sample_loss):.3f}")
        # Sanity checks
        if sanity_check:
            loss_val, gold_hidden = compute_loss(inputs, rel_ids, copy_pos, surface_indices, initial_hidden)
            assert hidden is not None
            hidden_state_diff = max(torch.max(torch.abs(g - h)).item() for g, h in zip(gold_hidden, hidden))
            if hidden_state_diff > EPS:
                Logging.warn(f"Hidden states do not match. Difference: {hidden_state_diff}")
            if abs(sample_loss - loss_val) > EPS:
                Logging.warn(f"Loss values do not match. "
                             f"Forward loss: {loss_val}, difference: {abs(sample_loss - loss_val)}")

        # Format the output
        sentence = list(zip(inputs, rel_ids, copy_pos, surface_indices))
        words = []
        copy_count = 0
        complete_count = 0
        last_entity = None
        entity_continuing = False
        for idx, (token, rel_id, pos, surface_idx) in enumerate(sentence):
            is_warm_up = (warm_up is not None and idx < warm_up)
            if rel_id == INVALID:
                word = word_vocab.i2w[token]
            elif rel_id == UNK:
                word = Logging.color('blue', unki2w[surface_idx])
            else:
                copy_count += 1
                entity_id = example.relations[rel_id].obj_alias[surface_idx]
                entity = self.alias_list[entity_id].split()
                if pos >= len(entity):
                    word = "<X>"
                else:
                    word = entity[pos]
                if show_copy_pos:
                    word = f"{pos}_{rel_id}_{surface_idx}_{word}"
                is_last_word_in_entity = (idx == len(sentence) - 1 or
                                          sentence[idx + 1][1:] != (rel_id, pos + 1, surface_idx))
                is_first_word_in_entity = (idx == 0 or sentence[idx - 1][1:] != (rel_id, pos - 1, surface_idx))
                # add entity tag after the last word
                if show_rel_type and is_last_word_in_entity:
                    word = f"{word}_[{rel_vocab.i2w[example.relations[rel_id].rel_typ]}]"

                # check whether fully copied
                if show_ellipses:
                    if pos < len(entity) - 1 and is_last_word_in_entity:
                        word = word + '...' + (f"({'_'.join(entity[(pos + 1):])})" if fill_incomplete else "")
                    if pos > 0 and is_first_word_in_entity:
                        word = (f"({'_'.join(entity[:pos])})" if fill_incomplete else "") + '...' + word

                if entity_continuing:
                    if last_entity == (rel_id, surface_idx, pos - 1):  # Continuing
                        last_entity = (rel_id, surface_idx, pos)
                    else:
                        entity_continuing = False
                        last_entity = None

                if pos == 0 and not entity_continuing:  # reset
                    entity_continuing = True
                    last_entity = (rel_id, surface_idx, 0)

                if color_outputs and not is_warm_up and (color_incomplete or entity_continuing):
                    word = Logging.color('red' if surface_idx == 0 else 'green', word)

                if pos == len(entity) - 1 and entity_continuing:  # commit
                    entity_continuing = False
                    complete_count += 1

            if color_outputs and is_warm_up:
                word = Logging.color('yellow', word)
            words.append(word)

        if print_info:
            print(f"Copied, Completed: {copy_count}, {complete_count}")
        sampled_output = SampledOutput(sentence=words, sample_loss=sample_loss,
                                       complete_copies=complete_count, incomplete_copies=copy_count)
        return sampled_output
