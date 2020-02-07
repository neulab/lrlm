import random
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
from torch import LongTensor, Tensor

__all__ = [
    'SampledOutput',
    'tensor',
    'randint',
    'sample',
    'np_sample',
]


class SampledOutput(NamedTuple):
    sentence: List[str]
    sample_loss: float
    complete_copies: int
    incomplete_copies: int


def tensor(x: Union[int, float, List[int]], device: Optional[torch.device] = None) -> LongTensor:
    if not isinstance(x, list):
        x = [x]  # type: ignore
    return torch.tensor(x, device=device).unsqueeze(0)  # (batch, seq_len)


def randint(low: int, high: int) -> int:
    return random.randint(low, high - 1)  # do not use torch.randint so as to preserve random state?


def sample(t: Tensor, greedy: bool = False, topk: Optional[int] = None) -> Tuple[int, float]:
    """ Return one sample based on log probs, and the corresponding log prob. """
    t = t.flatten()
    if greedy:
        i = torch.argmax(t).item()
    elif topk is not None:
        values, indices = torch.topk(t, min(topk, t.numel()))
        x = torch.multinomial(torch.exp(values), 1).item()
        i = indices[x]
    else:
        i = torch.multinomial(torch.exp(t), 1).item()
    return i, t[i].item()


def np_sample(t: Tensor, greedy: bool = False, topk: Optional[int] = None) -> Tuple[int, float]:
    import numpy as np
    t = t.flatten()
    if greedy:
        i = torch.argmax(t).item()
    elif topk is not None:
        values, indices = torch.topk(t, min(topk, t.numel()))
        prob = torch.exp(values).cpu().numpy()
        x = np.random.choice(len(prob), 1, p=prob)[0]
        i = indices[x]
    else:
        prob = torch.exp(t).cpu().numpy()
        i = np.random.choice(len(prob), 1, p=prob)[0]
    return i, t[i].item()
