from abc import ABC
from typing import Generic, TypeVar

from torch import nn

from arguments import RNNDropoutPos

__all__ = [
    'BaseRNN',
]

HiddenState = TypeVar('HiddenState')


class BaseRNN(nn.Module, Generic[HiddenState], ABC):
    def __init__(self, dropout: float, dropout_pos: RNNDropoutPos):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout_pos = dropout_pos

    def init_hidden(self, batch_size: int) -> HiddenState:
        raise NotImplementedError
