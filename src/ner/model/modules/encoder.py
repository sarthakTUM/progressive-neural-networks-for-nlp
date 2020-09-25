from torch import nn
import torch


class RNNSequenceEncoder(nn.Module):
    def __init__(self,
                 type: str,
                 in_dim,
                 out_dim,
                 dropout,
                 ) -> torch.Tensor:
        super(RNNSequenceEncoder, self).__init__()
        self.type = type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        if self.type == 'LSTM':
            self.encoder = nn.LSTM(
                                input_size=self.in_dim,
                                hidden_size=self.out_dim,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True,
                                )
        if self.type == 'gru':
            pass
        if self.type == 'rnn':
            pass

    def forward(self, input, sequence_lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(input, sequence_lengths, batch_first=True)
        (out, _) = self.rnn(packed)
        (unpacked_out, _) = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unpacked_out = unpacked_out.contiguous()
        return nn.Dropout(unpacked_out, self.dropout)


class CNNSequenceEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 ):
        super(CNNSequenceEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.char_cnn = nn.Conv1d(in_channels=self.char_embedding_dim,
                                  out_channels=params.char_cnn_filters,
                                  kernel_size=params.char_cnn_kernel_size,
                                  padding=((params.char_cnn_kernel_size - 1) / 2))
    def forward(self, input):
        pass


