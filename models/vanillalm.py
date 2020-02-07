import functools
import math
from typing import Optional, Tuple

import torch
from torch import LongTensor, Tensor

from arguments import LMArguments
from dataset import BatchSequence, LMExample, Vocab
from models import sample_utils
from models.base import BaseLM, HiddenState
from models.sample_utils import SampledOutput
from nnlib.utils import Logging

__all__ = [
    'VanillaLM',
]


class VanillaLM(BaseLM):
    def __init__(self, args: LMArguments, vocab_size: int, **_kwargs):
        r"""A normal encoder."""
        super().__init__(args, vocab_size)

    def forward(self,  # type: ignore
                inputs: LongTensor, hidden: Optional[HiddenState] = None,
                target: Optional[LongTensor] = None) -> Tuple[Tensor, HiddenState]:
        r"""A RNN module wrapper.
        Parameters:
            inputs: (B x N x D) (sorted) batch of tensor.
            hidden: hidden state.
            target: (B x N)
        Returns:
        """
        emb = self.word_embed(inputs)
        outputs, hidden = self.rnn(emb, hidden)

        if target is not None:
            # batch_size x seq_len
            loss = self.word_predictor(outputs, target)
            return loss, hidden
        else:
            # batch_size x seq_len x vocab_size
            log_probs = self.word_predictor.log_probs(outputs)
            return log_probs, hidden

    def calc_loss(self, batch: BatchSequence, hidden: Optional[HiddenState] = None,
                  use_unk_probs: bool = False, dump_probs: bool = False) -> Tuple[Tensor, HiddenState]:
        sequence = batch.sequence
        target = batch.target
        unk_probs = batch.unk_probs

        masks = (target != 1).to(dtype=torch.float)  # pad symbol is 1
        loss, new_hidden = self.forward(sequence, hidden, target)

        if use_unk_probs and unk_probs is not None:
            loss = loss - unk_probs

        # Sum of all the real elements
        batch_loss = torch.sum(loss * masks, dim=1)
        loss = torch.sum(batch_loss) / batch.ntokens

        assert batch.ntokens == torch.sum(masks)

        if dump_probs:
            log_probs = [-prob / length if length > 0 else 0.0
                         for prob, length in zip(batch_loss.tolist(), batch.lengths.tolist())]
            self.model_cache.update(log_probs=log_probs)

        return loss, new_hidden

    def init_hidden(self, batch_size: int, init_batch) -> HiddenState:
        return self.rnn.init_hidden(batch_size)

    @torch.no_grad()
    def sampling_decode(self, vocab: Vocab, example: LMExample,
                        begin_symbol: int = 2, end_symbol: int = 5,
                        initial_hidden: Optional[HiddenState] = None, warm_up: Optional[int] = None,
                        max_length: int = 200, greedy: bool = False, topk: Optional[int] = None,
                        print_info: bool = True, color_outputs: bool = False, **_kwargs) \
            -> SampledOutput:
        tensor = functools.partial(sample_utils.tensor, device=self.device)
        sample = functools.partial(sample_utils.sample, greedy=greedy, topk=topk)

        self.eval()
        self.init_hidden(1, None)

        if warm_up is None:
            inputs = [begin_symbol]
            hidden = initial_hidden
            total_log_prob = 0.0
        else:
            inputs = list(vocab.numericalize(example.sentence[:warm_up]))
            total_log_prob, hidden = self.forward(tensor(inputs[:-1]), target=tensor(inputs[1:]))
            total_log_prob = -torch.sum(total_log_prob).item() * (len(inputs) - 1)

        while len(inputs) < max_length and inputs[-1] != end_symbol:
            # full copy of the forward pass, including dropouts. But they won't be applied due to .eval function.
            # Run LSTM over the word
            word_log_probs, new_hidden = self.forward(tensor(inputs[-1]), hidden)
            word_id, word_log_prob = sample(word_log_probs)
            inputs.append(word_id)
            hidden = new_hidden
            total_log_prob += word_log_prob

        sample_loss = -total_log_prob / (len(inputs) - 1)
        if print_info:
            print(f"Sample loss: {sample_loss:.3f}, PPL: {math.exp(sample_loss):.3f}")

        # Format the output
        words = [vocab.i2w[token] for token in inputs]
        if color_outputs and warm_up is not None:
            words[:warm_up] = [Logging.color('yellow', w) for w in words[:warm_up]]

        output = SampledOutput(sentence=words, sample_loss=sample_loss, complete_copies=0, incomplete_copies=0)
        return output
