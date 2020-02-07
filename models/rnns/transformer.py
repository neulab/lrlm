# Credit: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
#   Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov
#   Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from arguments import RNNDropoutPos
from models.rnns.base import BaseRNN

__all__ = [
    'TransformerXL',
    'TransformerState',
]

Memory = List[Tensor]
TransformerState = Memory


class PositionalEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, embed_dim: int):
        super().__init__()

        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: Tensor, batch_size: Optional[int] = None) -> Tensor:  # type: ignore
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        pos_embed = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)[:, None, :]

        if batch_size is not None:
            pos_embed = pos_embed.expand(-1, batch_size, -1)
        return pos_embed


class PositionWiseFF(nn.Module):
    def __init__(self, hidden_dim: int, ffn_inner_dim: int, dropout: float, pre_lnorm=False):
        super().__init__()

        self.core_net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_inner_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pre_lnorm = pre_lnorm

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.layer_norm.weight, 1.0, 0.02)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        orig_input = input
        if self.pre_lnorm:
            input = self.layer_norm(input)

        # position-wise feed-forward
        output = self.core_net(input)
        # residual connection
        output = orig_input + output

        if not self.pre_lnorm:
            output = self.layer_norm(output)
        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, head_dim: int, dropout: float,
                 dropout_attn: float = 0.0, pre_lnorm=False):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv_net = nn.Linear(hidden_dim, 3 * num_heads * head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.o_net = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pre_lnorm = pre_lnorm

        self.r_net = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)

        self.scale = 1 / (head_dim ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.layer_norm.weight, 1.0, 0.02)

    @staticmethod
    def _rel_shift(x: Tensor, zero_triu=False) -> Tensor:
        zero_pad = x.new_zeros(x.size(0), 1, *x.size()[2:])
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones(x.size(0), x.size(1))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x

    def forward(self,  # type: ignore
                input: Tensor, pos_embed: Tensor, r_w_bias: Tensor, r_r_bias: Tensor,
                attn_mask: Optional[Tensor] = None, memory: Optional[Tensor] = None) -> Tensor:
        seq_len, rlen, batch_size = input.size(0), pos_embed.size(0), input.size(1)

        orig_input = input
        if memory is not None:
            input = torch.cat([memory, input], dim=0)
        if self.pre_lnorm:
            input = self.layer_norm(input)
        w_heads = self.qkv_net(input)
        r_head_k = self.r_net(pos_embed)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-seq_len:]

        tot_len = w_head_k.size(0)

        # w_head_*: (seq_len, batch_size, num_heads, head_dim)
        w_head_q = w_head_q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        w_head_k = w_head_k.view(tot_len, batch_size, self.num_heads, self.head_dim)
        w_head_v = w_head_v.view(tot_len, batch_size, self.num_heads, self.head_dim)

        r_head_k = r_head_k.view(rlen, self.num_heads, self.head_dim)  # seq_len x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias
        # attn_ac = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # seq_len x tot_len x batch_size x n_head
        dim_i, dim_b, dim_n, dim_d = rw_head_q.size()
        dim_j = w_head_k.size(0)
        attn_ac = torch.bmm(
            rw_head_q.contiguous().view(dim_i, dim_b * dim_n, dim_d).transpose(1, 0),
            w_head_k.contiguous().view(dim_j, dim_b * dim_n, dim_d).permute(1, 2, 0)
        ).permute(1, 2, 0).view(dim_i, dim_j, dim_b, dim_n)

        rr_head_q = w_head_q + r_r_bias
        # attn_bd = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # seq_len x tot_len x batch_size x n_head
        dim_i, dim_b, dim_n, dim_d = rr_head_q.size()
        dim_j = r_head_k.size(0)
        attn_bd = torch.bmm(
            rr_head_q.contiguous().view(dim_i * dim_b, dim_n, dim_d).transpose(1, 0),
            r_head_k.contiguous().permute(1, 2, 0)
        ).view(dim_n, dim_i, dim_b, dim_j).permute(1, 3, 2, 0)
        attn_bd = self._rel_shift(attn_bd)

        # [seq_len x tot_len x batch_size x n_head]
        attn_score = attn_ac + attn_bd
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, :, :, None]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, :, :, None]
            attn_score = attn_score.float().masked_fill(attn_mask, -math.inf).type_as(attn_score)

        # [seq_len x tot_len x batch_size x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout_attn(attn_prob)

        # compute attention vector
        # attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        dim_i, dim_j, dim_b, dim_n = attn_prob.size()
        dim_d = w_head_v.size(3)
        attn_vec = torch.bmm(
            attn_prob.contiguous().view(dim_i, dim_j, dim_b * dim_n).permute(2, 0, 1),
            w_head_v.contiguous().view(dim_j, dim_b * dim_n, dim_d).permute(1, 0, 2)
        ).view(dim_b, dim_n, dim_i, dim_d).permute(2, 0, 1, 3)

        # [seq_len x batch_size x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.num_heads * self.head_dim)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

        # residual connection
        output = orig_input + attn_out
        if not self.pre_lnorm:
            output = self.layer_norm(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, head_dim: int, ffn_inner_dim: int,
                 dropout: float, dropout_attn: float, pre_lnorm=False):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            num_heads, hidden_dim, head_dim, dropout, dropout_attn=dropout_attn, pre_lnorm=pre_lnorm)
        self.pos_ff = PositionWiseFF(hidden_dim, ffn_inner_dim, dropout, pre_lnorm=pre_lnorm)

    def forward(self,  # type: ignore
                input: Tensor, pos_embed: Tensor, r_w_bias: Tensor, r_r_bias: Tensor,
                dec_attn_mask: Optional[Tensor] = None, memory: Optional[Memory] = None) -> Tensor:
        output = self.dec_attn(input, pos_embed, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, memory=memory)
        output = self.pos_ff(output)
        return output


class TransformerXL(BaseRNN[Memory]):
    def __init__(self, num_layers: int, num_heads: int, hidden_dim: int, head_dim: int, ffn_inner_dim: int,
                 dropout: float, dropout_attn: float, dropout_pos: RNNDropoutPos,
                 embed_dim: Optional[int] = None, pre_lnorm=False,
                 ext_len: int = 0, mem_len: int = 0):
        super().__init__(dropout, dropout_pos)

        self.num_layers = num_layers

        self.mem_len = mem_len
        self.ext_len = ext_len

        embed_dim = hidden_dim if embed_dim is None else embed_dim
        self.proj_input = None
        if embed_dim != hidden_dim:
            self.proj_input = nn.Linear(embed_dim, hidden_dim)

        self.layers: 'nn.ModuleList[RelPartialLearnableDecoderLayer]' = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RelPartialLearnableDecoderLayer(
                num_heads, hidden_dim, head_dim, ffn_inner_dim, dropout, dropout_attn=dropout_attn,
                pre_lnorm=pre_lnorm))

        self.pos_embed = PositionalEmbedding(hidden_dim)
        self.r_w_bias = nn.Parameter(torch.Tensor(num_heads, head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(num_heads, head_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.r_w_bias, 0.0, 0.02)
        nn.init.normal_(self.r_r_bias, 0.0, 0.02)

    def set_lengths(self, ext_len: int, mem_len: int):
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_memory(self) -> Optional[Memory]:
        if self.mem_len == 0:
            return None

        param: nn.Parameter = next(self.parameters())
        memory = [param.new_empty(0) for _ in range(self.num_layers + 1)]
        return memory

    def _update_memory(self, states: List[Tensor], memory: Optional[Memory],
                       seq_len: int, mem_len: int) -> Optional[Memory]:
        # does not deal with None
        if memory is None:
            return None

        assert len(states) == len(memory)

        # There are `seq_len + mem_len` steps that can be cached into memories.
        # For the next step, the last `ext_len` of the `seq_len` tokens will
        # be used as the extended context. Hence, we only cache the tokens
        # from `mem_len + seq_len - self.ext_len - self.mem_len`
        # to `mem_len + seq_len - self.ext_len`.
        with torch.no_grad():
            new_memory = []
            end = mem_len + max(0, seq_len - self.ext_len)
            start = max(0, end - self.mem_len)
            for layer_idx in range(len(states)):
                if start >= self.mem_len:
                    total_steps = states[layer_idx]
                    start -= self.mem_len
                    end -= self.mem_len
                else:
                    total_steps = torch.cat([memory[layer_idx], states[layer_idx]], dim=0)
                new_memory.append(total_steps[start:end].detach())

        return new_memory

    def _forward(self, word_embed: Tensor, memory: Optional[Memory] = None) -> Tuple[Tensor, Optional[Memory]]:
        word_embed = word_embed.transpose(0, 1)  # word_embed is batch first
        seq_len, batch_size = word_embed.size(0), word_embed.size(1)

        if self.proj_input is not None:
            word_embed = self.proj_input(word_embed)

        mem_len = memory[0].size(0) if memory is not None else 0
        tot_len = mem_len + seq_len
        dec_attn_mask = torch.triu(word_embed.new_ones(seq_len, tot_len),
                                   diagonal=1 + mem_len).byte()[:, :, None]

        states = []
        output = word_embed

        pos_seq = torch.arange(tot_len - 1, -1, -1.0, device=word_embed.device, dtype=word_embed.dtype)
        pos_embed = self.pos_embed(pos_seq)

        if self.dropout_pos in [RNNDropoutPos.Early, RNNDropoutPos.Both]:
            output = self.dropout(output)
            pos_embed = self.dropout(pos_embed)

        states.append(output)
        for idx, layer in enumerate(self.layers):
            layer_mem = None if memory is None else memory[idx]
            output = layer(output, pos_embed, self.r_w_bias, self.r_r_bias,
                           dec_attn_mask=dec_attn_mask, memory=layer_mem)
            states.append(output)

        if self.dropout_pos in [RNNDropoutPos.Late, RNNDropoutPos.Both]:
            output = self.dropout(output)
        output = output.transpose(0, 1).contiguous()  # convert back to batch-first

        new_memory = self._update_memory(states, memory, mem_len, seq_len)
        if new_memory is not None:
            new_memory = [m.transpose(0, 1) for m in new_memory if m.dim() >= 2]

        return output, new_memory

    def forward(self,  # type: ignore
                input: Tensor, memory: Optional[Memory]) -> Tuple[Tensor, Optional[Memory]]:
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) memories inside the model forward.
        # Moreover, have to return new_memory to allow nn.DataParallel to piece
        # them together.
        if memory is None:
            memory = self.init_memory()
        else:
            # switch from batch first to seq_len first
            memory = [m.transpose(0, 1) for m in memory if m.dim() >= 2]

        output, new_memory = self._forward(input, memory=memory)
        return output, new_memory

    def init_hidden(self, batch_size: int):
        return None  # self.init_memory()
