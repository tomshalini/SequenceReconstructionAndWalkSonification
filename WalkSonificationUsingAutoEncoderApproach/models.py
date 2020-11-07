import sys
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

class Encoder(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim):
        super(Encoder, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim, self.hidden_dim = embedding_dim , 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            bidirectional=True
        )

    def forward(self, x):

        x, (hidden_n, _) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(hidden_n)

        return x, (hidden_n, cell_n)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dimension, sequence_length, output_dim=1):
        super(Decoder, self).__init__()

        self.sequence_length, self.embedding_dim = sequence_length, embedding_dim
        self.hidden_dimension, self.output_dim = hidden_dimension, output_dim
        self.num_classes = num_classes

        self.rnn1 = nn.LSTM(
            input_size=self.embedding_dim * 2,
            hidden_size=self.hidden_dimension,
            num_layers=1,
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dimension,
            hidden_size=self.num_classes,
            num_layers=1,
        )

    def forward(self, x):
        x = x.view(1, x.shape[1], -1)
        x = x.repeat(self.sequence_length, 1, 1)

        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)

        return x.permute(2,1,0)