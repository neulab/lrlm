import functools
import math
from typing import Optional, Dict

import torch
from torch import Tensor

from arguments import LMArguments
from dataset import LRLMExample, Vocab
from models import sample_utils
from models.base import HiddenState
from models.sample_utils import SampledOutput
from models.vanillalm import VanillaLM
from nnlib.utils import Logging
from utils import repackage_hidden

__all__ = [
    'AliasLM',
]


class AliasLM(VanillaLM):
    def __init__(self, args: LMArguments, vocab_size: int, **_kwargs):
        r"""A normal encoder."""
        super().__init__(args, vocab_size)
        self.bptt_size = args.bptt_size

    def init_hidden(self, batch_size: int, init_batch: Tensor) -> HiddenState:
        padded_aliases = init_batch
        batched_aliases = []
        max_len_aliases = padded_aliases.size(1)
        for i in range((max_len_aliases - 1) // self.bptt_size + 1):
            batched_aliases.append(padded_aliases[:, i * self.bptt_size:(i + 1) * self.bptt_size])

        hidden = self.rnn.init_hidden(batch_size)
        for batched_alias in batched_aliases:
            batched_alias = batched_alias.to(self.device)
            if hidden is not None:
                hidden = repackage_hidden(hidden)

            alias_embeds = self.word_embed(batched_alias)
            _, hidden = self.rnn.forward(alias_embeds, hidden)

        return hidden

    @torch.no_grad()
    def sampling_decode(self, vocab: Dict[str, Vocab], example: LRLMExample,
                        begin_symbol: int = 2, end_symbol: int = 5,
                        initial_hidden: Optional[HiddenState] = None, warm_up: Optional[int] = None,
                        max_length: int = 200, greedy: bool = False, topk: Optional[int] = None,
                        print_info: bool = True, color_outputs: bool = False, init_batch=None, **_kwargs) \
            -> SampledOutput:
        tensor = functools.partial(sample_utils.tensor, device=self.device)
        sample = functools.partial(sample_utils.sample, greedy=greedy, topk=topk)

        self.eval()
        self.init_hidden(1, init_batch)

        if warm_up is None:
            inputs = [begin_symbol]
            hidden = initial_hidden
            total_log_prob = 0.0
        else:
            inputs = list(vocab["word"].numericalize(example.sentence[:warm_up]))
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
        words = [vocab["word"].i2w[token] for token in inputs]
        if color_outputs and warm_up is not None:
            words[:warm_up] = [Logging.color('yellow', w) for w in words[:warm_up]]

        output = SampledOutput(sentence=words, sample_loss=sample_loss, complete_copies=0, incomplete_copies=0)
        return output
