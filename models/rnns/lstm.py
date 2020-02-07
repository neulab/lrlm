from typing import Optional, Tuple

from torch import nn
from torch import Tensor

from arguments import RNNDropoutPos
from models.rnns.base import BaseRNN

__all__ = [
    'LSTM',
    'LSTMState',
]

LSTMState = Tuple[Tensor, Tensor]


class LSTM(BaseRNN[LSTMState]):
    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int,
                 *, dropout: float, dropout_pos: RNNDropoutPos):
        super().__init__(dropout, dropout_pos)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

    def forward(self,  # type: ignore
                input: Tensor, hidden: Optional[LSTMState] = None) -> Tuple[Tensor, LSTMState]:
        if self.dropout_pos in [RNNDropoutPos.Early, RNNDropoutPos.Both]:
            input = self.dropout(input)
        output, hidden = self.lstm.forward(input, hidden)
        if self.dropout_pos in [RNNDropoutPos.Late, RNNDropoutPos.Both]:
            output = self.dropout(output)
        return output, hidden

    def init_hidden(self, batch_size) -> LSTMState:
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
            weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
        )
