from abc import ABC
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from models.sample_utils import SampledOutput
from torch import nn
from torch import Tensor

from arguments import LMArguments
from dataset.utils import BatchSequence
from models.rnns import BaseRNN, LSTM, LSTMState, TransformerXL, TransformerState
from models.utils import AdaptiveEmbedding, MLPSoftmax, ProjectedAdaptiveLogSoftmax, Softmax

__all__ = [
    'HiddenState',
    'BaseLM',
]

HiddenState = Union[LSTMState, TransformerState]


class BaseLM(nn.Module, ABC):
    model_cache: Dict[str, Any] = {}

    word_embed: nn.Embedding
    word_predictor: Softmax
    rnn: BaseRNN

    def __init__(self, args: LMArguments, vocab_size: int,
                 input_dim: Optional[int] = None, pred_input_dim: Optional[int] = None,
                 embed_dim: Optional[int] = None):
        r"""Initializer for the LM base class.

        ..note::
            Please pay attention to the dimension differences between LSTM and Transformer. Transformers expect input
            dimension to be the same as hidden size, while LSTMs can handle inputs of arbitrary dimension because it
            does a projection of inputs anyway.

            When using Transformers and `input_dim != args.embed_size`, if adaptive embeddings are used, then we add
            projections in `AdaptiveEmbedding` by setting `proj_dim` to `input_dim`; if normal embeddings are used,
            we add projections in `TransformerXL` by setting `embed_dim` to `input_dim`.

        :param args: General model arguments.
        :param vocab_size: Vocabulary size.
        :param input_dim: Dimension of input to the recurrent network. If not specified, `args.embed_size` is used.
        :param pred_input_dim: Dimension of input to the predictors. If not specified, `args.hidden_size` is used.
        :param embed_dim: Dimension of input embeddings. If not specified, `args.embed_size` is used. This **must** be
            specified if the RNN input is the concatenation of embeddings and other tensors.
        """
        super().__init__()

        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self._vocab_size = vocab_size
        self._embed_dim = embed_dim or args.embed_size
        self._hidden_dim = args.hidden_size
        self._input_dim = input_dim or args.embed_size
        self._pred_input_dim = pred_input_dim or args.hidden_size

        if args.adaptive_embed:
            cutoffs = [20000, 40000, 200000]
            cutoffs = [x for x in cutoffs if x < vocab_size]
            div_val = 1
            proj_dim = self._input_dim if args.base_rnn == 'transformer' and embed_dim is None else self._embed_dim
            self.word_embed = AdaptiveEmbedding(
                vocab_size, self._embed_dim, proj_dim, cutoffs, div_value=div_val)
            # use adaptive softmax (including standard softmax)
            self.word_predictor = ProjectedAdaptiveLogSoftmax(
                vocab_size, args.vocab_mlp_hidden_dim, self._pred_input_dim, cutoffs, div_value=div_val)
            if args.tie_embed_weights:
                assert args.vocab_mlp_hidden_dim == self._embed_dim
                # assign adaptive softmax weight to embedding,
                # because its weight matrix is larger (head_size + n_cluster)
                for i in range(len(self.word_predictor.out_layers)):
                    self.word_predictor.out_layers[i].weight = self.word_embed.embed_layers[i].weight
                if self._pred_input_dim == self._hidden_dim and div_val != 1:
                    for i in range(len(cutoffs)):
                        self.word_predictor.out_projs[i + 1] = self.word_embed.embed_projs[i + 1]
        else:
            self.word_embed = nn.Embedding(vocab_size, self._embed_dim, padding_idx=1)
            self.word_predictor = MLPSoftmax(self._pred_input_dim,
                                             args.vocab_mlp_hidden_dim,
                                             vocab_size, dropout=args.vocab_mlp_dropout,
                                             activation=args.vocab_mlp_activation)
            if args.tie_embed_weights:
                assert args.vocab_mlp_hidden_dim == self._embed_dim
                self.word_predictor.layer2.weight = self.word_embed.weight.weight

        if args.base_rnn == 'transformer':
            tfm_embed_dim = self._embed_dim if args.adaptive_embed and embed_dim is None else self._input_dim
            self.rnn = TransformerXL(
                args.num_layers, args.num_heads, self._hidden_dim, args.head_dim,
                args.ffn_inner_dim, args.dropout, args.attention_dropout, dropout_pos=args.rnn_dropout_pos,
                embed_dim=tfm_embed_dim, pre_lnorm=args.pre_lnorm,
                ext_len=0, mem_len=args.memory_size)
        else:
            self.rnn = LSTM(self._input_dim, self._hidden_dim, args.num_layers,
                            dropout=args.dropout, dropout_pos=args.rnn_dropout_pos)

    def calc_loss(self, batch: BatchSequence, hidden: Optional[HiddenState] = None,
                  use_unk_probs: bool = False, dump_probs: bool = False) -> Tuple[Tensor, HiddenState]:
        raise NotImplementedError

    def init_hidden(self, batch_size: int, init_batch):
        raise NotImplementedError

    def sampling_decode(self, vocab, example,
                        begin_symbol: int = 2, end_symbol: int = 5,
                        initial_hidden: Optional[HiddenState] = None, warm_up: Optional[int] = None,
                        max_length: int = 200, greedy: bool = False, topk: Optional[int] = None,
                        print_info=True, color_outputs=False, **_kwargs) -> SampledOutput:
        raise NotImplementedError
