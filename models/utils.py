from abc import ABC
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch import LongTensor, Tensor
from torch.nn import functional as F

__all__ = [
    'load_kb_embed',
    'sequence_mask',
    'linear_weight_init',
    'MLP',
    'Softmax',
    'MLPSoftmax',
    'ProjectedAdaptiveLogSoftmax',
    'AdaptiveEmbedding',
]


def load_kb_embed(path: str, device: Optional[torch.device]) -> Tuple[Tensor, Tensor]:
    path = Path(path)
    entity_embed = torch.load(path / 'entity_vec.pt')
    rel_embed = torch.load(path / 'relation_vec.pt')
    if device is not None:
        entity_embed = entity_embed.to(device=device)
        rel_embed = rel_embed.to(device=device)
    return entity_embed, rel_embed


def sequence_mask(lengths: List[int], max_len: Optional[int] = None, mask_val=1.0, default_val=0.0,
                  dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Tensor:
    batch_size = len(lengths)
    max_len = max_len or max(lengths)
    mask = torch.full((batch_size, max_len), mask_val, dtype=dtype, device=device)
    for b, length in enumerate(lengths):
        mask[b, length:] = default_val
    return mask


def linear_weight_init(weight: nn.Parameter, bias: Optional[nn.Parameter] = None, dim: Optional[int] = None):
    # stdv = 1. / math.sqrt(dim or weight.size(1))
    # weight.data.uniform_(-stdv, stdv)
    weight.data.normal_(0.0, 0.02)
    if bias is not None:
        # bias.data.uniform_(-stdv, stdv)
        bias.data.zero_()


class MLP(nn.Module):
    r"""Wrapper for a two-layer MLP."""

    _ACTIVATION: Dict[str, Callable[[Tensor], Tensor]] = {
        'relu': torch.relu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'none': lambda x: x,
    }

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = 'relu',
                 dropout: Optional[float] = None):
        super().__init__()
        if hidden_dim == -1:
            hidden_dim = output_dim
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = self._ACTIVATION[activation]
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        hidden = self.layer1.forward(input)
        hidden = self.activation(hidden)
        if self.dropout is not None:
            hidden = self.dropout.forward(hidden)
        hidden = self.layer2.forward(hidden)
        return hidden


class Softmax(nn.Module, ABC):
    def forward(self, input: Tensor, target: LongTensor) -> Tensor:  # type: ignore
        raise NotImplementedError

    def log_probs(self, input: Tensor) -> Tensor:
        raise NotImplementedError


class MLPSoftmax(MLP, Softmax):
    def forward(self, input: Tensor, target: LongTensor) -> Tensor:  # type: ignore
        input_shape = input.size()
        logits = super().forward(input)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='none')
        return nll.view(input_shape[:-1])

    def log_probs(self, input: Tensor) -> Tensor:
        return F.log_softmax(super().forward(input), dim=-1)


class ProjectedAdaptiveLogSoftmax(Softmax):
    def __init__(self, vocab_size: int, embed_dim: int, proj_dim: int,
                 cutoffs: List[int], div_value: int = 1):
        super().__init__()

        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_value = div_value
        self.vocab_size = vocab_size

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, embed_dim))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers: 'nn.ModuleList[nn.Linear]' = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_value == 1:
            if proj_dim != embed_dim:
                for i in range(len(self.cutoffs)):
                    self.out_projs.append(nn.Parameter(torch.Tensor(proj_dim, embed_dim)))
            self.out_layers.append(nn.Linear(embed_dim, vocab_size))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                part_embed_dim = embed_dim // (div_value ** i)
                self.out_projs.append(nn.Parameter(torch.Tensor(proj_dim, part_embed_dim)))
                self.out_layers.append(nn.Linear(part_embed_dim, r_idx - l_idx))

        self.reset_parameters()

    def reset_parameters(self):
        if self.n_clusters > 0:
            nn.init.normal_(self.cluster_weight, 0.0, 0.02)
            nn.init.zeros_(self.cluster_bias)
        for param in self.out_projs:
            # nn.init.normal_(param, 0.0, 0.01)
            linear_weight_init(param)

    @staticmethod
    def _compute_logits(hidden: Tensor, weight: nn.Parameter,
                        bias: nn.Parameter, proj: Optional[nn.Parameter]) -> Tensor:
        if proj is None:
            logits = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logits = F.linear(proj_hid, weight, bias=bias)

        return logits

    def _construct_weights(self) -> Tuple[List[Tensor], List[Tensor]]:
        # construct weights and biases
        weights, biases = [], []
        for i in range(len(self.cutoffs)):
            if self.div_value == 1:
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                weight_i = self.out_layers[0].weight[l_idx:r_idx]
                bias_i = self.out_layers[0].bias[l_idx:r_idx]
            else:
                weight_i = self.out_layers[i].weight
                bias_i = self.out_layers[i].bias

            if i == 0:
                weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

            weights.append(weight_i)
            biases.append(bias_i)
        return weights, biases

    def forward(self, input: Tensor, target: LongTensor) -> Tensor:  # type: ignore
        """
        hidden :: [len*bsz x d_proj]
        target :: [len*bsz]
        """
        input_shape = input.size()
        input = input.contiguous().view(-1, input_shape[-1])
        target = target.contiguous().view(-1)

        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logits = self._compute_logits(input, self.out_layers[0].weight,
                                          self.out_layers[0].bias, self.out_projs[0])
            nll = F.nll_loss(logits, target, reduction='none')
        else:
            weights, biases = self._construct_weights()

            head_weight, head_bias = weights[0], biases[0]
            head_proj = self.out_projs[0] if len(self.out_projs) > 0 else None

            head_logits = self._compute_logits(input, head_weight, head_bias, head_proj)
            head_log_probs = F.log_softmax(head_logits, dim=1)

            nonzero_indices: List[torch.ByteTensor] = [
                ((target >= l) & (target < r)).nonzero().squeeze()
                for l, r in zip(self.cutoffs[:-1], self.cutoffs[1:])
            ]
            head_indices: LongTensor = target.clone()
            for idx, indices in enumerate(nonzero_indices):
                if indices.numel() == 0:
                    continue
                index = self.shortlist_size + self.n_clusters - 1 - idx
                head_indices.index_fill_(0, indices, index)

            head_nll = F.nll_loss(head_log_probs, head_indices, reduction='none')

            for idx, indices in enumerate(nonzero_indices):
                if indices.numel() == 0:
                    continue

                weight_i, bias_i = weights[idx + 1], biases[idx + 1]
                proj_i = self.out_projs[idx + 1] if len(self.out_projs) > idx + 1 else None

                cluster_hidden = input.index_select(0, indices)
                cluster_target = target.index_select(0, indices) - self.cutoffs[idx]

                cluster_logits = self._compute_logits(cluster_hidden, weight_i, bias_i, proj_i)
                cluster_nll = F.cross_entropy(cluster_logits, cluster_target, reduction='none')

                tail_nll = torch.zeros_like(head_nll)
                tail_nll.index_copy_(0, indices, cluster_nll)
                head_nll = head_nll + tail_nll

            nll = head_nll

        nll = nll.view(input_shape[:-1])
        return nll

    def log_probs(self, input: Tensor) -> Tensor:
        input_shape = input.size()
        input = input.contiguous().view(-1, input_shape[-1])

        if self.n_clusters == 0:
            logits = self._compute_logits(input, self.out_layers[0].weight,
                                          self.out_layers[0].bias, self.out_projs[0])
            log_probs = F.log_softmax(logits, dim=-1)
        else:
            weights, biases = self._construct_weights()

            head_weight, head_bias = weights[0], biases[0]
            head_proj = self.out_projs[0] if len(self.out_projs) > 0 else None

            head_logits = self._compute_logits(input, head_weight, head_bias, head_proj)
            head_log_probs: Tensor = F.log_softmax(head_logits, dim=-1)
            all_log_probs = [head_log_probs[:, :self.shortlist_size]]

            for idx, (l, r) in enumerate(zip(self.cutoffs[:-1], self.cutoffs[1:])):
                weight_i, bias_i = weights[idx + 1], biases[idx + 1]
                proj_i = self.out_projs[idx + 1] if len(self.out_projs) > idx + 1 else None

                tail_logits = self._compute_logits(input, weight_i, bias_i, proj_i)
                tail_log_probs = F.log_softmax(tail_logits, dim=-1)
                assert tail_log_probs.size(-1) == r - l
                index = self.shortlist_size + self.n_clusters - 1 - idx
                cluster_log_probs = head_log_probs[:, index].unsqueeze(-1) + tail_log_probs
                all_log_probs.append(cluster_log_probs)

            log_probs = torch.cat(all_log_probs, dim=-1)

        log_probs = log_probs.view(*input_shape[:-1], self.vocab_size)
        return log_probs


class AdaptiveEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, embed_dim: int, proj_dim: int, cutoffs: List[int], div_value: int = 1):
        nn.Module.__init__(self)  # skip initialization for nn.Embedding

        self.vocab_size = vocab_size
        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_value = div_value

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.embed_scale = proj_dim ** 0.5

        self.embed_layers: 'nn.ModuleList[nn.Embedding]' = nn.ModuleList()
        self.embed_projs = nn.ParameterList()
        if div_value == 1:
            self.embed_layers.append(nn.Embedding(vocab_size, embed_dim))
            if proj_dim != embed_dim:
                self.embed_projs.append(nn.Parameter(torch.Tensor(proj_dim, embed_dim)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = embed_dim // (div_value ** i)
                self.embed_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.embed_projs.append(nn.Parameter(torch.Tensor(proj_dim, d_emb_i)))

        self.reset_parameters()

    def extra_repr(self) -> str:
        return f''

    def reset_parameters(self):
        for param in self.embed_projs:
            # nn.init.normal_(param, 0.0, 0.01)
            linear_weight_init(param)

    def forward(self, input: LongTensor) -> Tensor:  # type: ignore
        if self.div_value == 1:
            embed = self.embed_layers[0](input)
            if len(self.embed_projs) > 0:
                embed = F.linear(embed, self.embed_projs[0])
        else:
            param = next(self.parameters())
            input_flat = input.view(-1)
            embed_flat = param.new_zeros((input_flat.size(0), self.proj_dim))
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i: torch.ByteTensor = (input_flat >= l_idx) & (input_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                input_i = input_flat.index_select(0, indices_i) - l_idx
                embed_i = self.embed_layers[i](input_i)
                embed_i = F.linear(embed_i, self.embed_projs[i])

                embed_flat.index_copy_(0, indices_i, embed_i)

            embed = embed_flat.view(*input.size(), self.proj_dim)

        embed.mul_(self.embed_scale)
        return embed
