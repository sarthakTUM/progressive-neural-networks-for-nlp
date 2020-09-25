"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ner.encoder import CharEncoder, EntityEncoder, WordEncoder
import math
from src.booster.progNN.decoder import CRFDecoder

np.random.seed(0)


class LSTMCRF(nn.Module):

    def __init__(self,
                 params,
                 char_vocab_length: int,
                 num_tags: int,
                 pretrained_word_vecs: torch.Tensor,
                 hidden_dim: int = 100,
                 layers: int = 1,
                 dropout: float = 0.5,
                 decoder_type='fc',
                 bidirectional: bool = True,
                 freeze_embeddings: bool = True):
        super(LSTMCRF, self).__init__()
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)
        np.random.seed(0)

        self.decoder_type = decoder_type

        self.dropout = nn.Dropout(p=dropout) if dropout else None

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        char_matrix = np.random.uniform(low=-(math.sqrt(3/params.char_embedding_dim)),
                                        high=math.sqrt(3/params.char_embedding_dim),
                                        size=(char_vocab_length, params.char_embedding_dim))

        self.char_embedding = nn.Embedding.from_pretrained(torch.from_numpy(char_matrix), freeze=False)

        self.char_cnn = nn.Conv1d(in_channels=params.char_embedding_dim,
                                  out_channels=params.char_cnn_filters,
                                  kernel_size=params.char_cnn_kernel_size,
                                  padding=2)

        self.char_cnn_output_size = params.char_cnn_filters

        self.word_vec_dim = pretrained_word_vecs.size()[1]
        self._freeze_embeddings = freeze_embeddings
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_vecs,
                                                           freeze=freeze_embeddings)

        rnn_input_size = self.word_vec_dim + self.char_cnn_output_size

        self.rnn_1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.rnn_2 = nn.LSTM(
            input_size=hidden_dim*2,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=bidirectional,
            dropout=0,
            batch_first=True,
        )

        self.rnn_output_size = hidden_dim
        if bidirectional:
            self.rnn_output_size *= 2

        self.fc = nn.Linear(self.rnn_output_size, num_tags)

        self.decoder = CRFDecoder(num_tags, num_tags)

    def reset_layers(self, num_output_tags):
        self.fc = nn.Linear(self.rnn_output_size, num_output_tags)
        self.decoder = CRFDecoder(num_output_tags, num_output_tags)

    def activate(self, input):
        """
        Forwarding without PNN activations
        :param input:
        :return:
        """
        char_embs = self.dropout(self.char_embedding(input[CharEncoder.FEATURE_NAME]))
        char_cnn = self.char_cnn(char_embs.float().view(-1,
                                                        char_embs.size()[2],
                                                        char_embs.size()[3]).permute(0, 2, 1))
        char_feats, _ = torch.max(char_cnn, 2)

        word_embs = self.word_embedding(input[WordEncoder.FEATURE_NAME])

        word_feats = torch.cat([char_feats.float(), word_embs.view(-1, word_embs.size()[-1]).float()], dim=-1)

        word_feats = self.dropout(word_feats.view(word_embs.size()[0], word_embs.size()[1], -1))

        packed = torch.nn.utils.rnn.pack_padded_sequence(word_feats, input['timesteps'].tolist(), batch_first=True)
        (out_rnn_1, _) = self.rnn_1(packed)
        (out_rnn_2, _) = self.rnn_2(out_rnn_1)
        (unpacked_out, _) = torch.nn.utils.rnn.pad_packed_sequence(out_rnn_2, batch_first=True)
        unpacked_out = unpacked_out.contiguous()
        unpacked_out = self.dropout(unpacked_out)

        feats = self.fc(unpacked_out.view(-1, self.rnn_output_size)).view(word_embs.size()[0], word_embs.size()[1], -1)
        return [torch.nn.utils.rnn.pad_packed_sequence(out_rnn_1, batch_first=True)[0], unpacked_out, feats]

    def loss(self, input, labels, label_pad_idx):
        """
        Forwarding and getting loss
        :param input:
        :param labels:
        :param label_pad_idx:
        :return:
        """
        feats = self.activate(input)
        # LOSS calculation
        entities = labels[EntityEncoder.FEATURE_NAME]
        mask = (entities != label_pad_idx).float()
        NLL = self.decoder(feats[-1], entities, mask=mask)
        return NLL

    def predict(self, X):
        logits = self.activate(X)
        mask = torch.zeros(X['timesteps'].size(0), torch.max(X['timesteps']))
        for row_idx, row in enumerate(mask):
            mask[row_idx] = torch.tensor(
                [1] * X['timesteps'][row_idx].item() + [0] * (torch.max(X['timesteps']) - X['timesteps'][row_idx]).item())
        preds = self.decoder.predict(logits[-1], mask)
        return preds

    # forward for PNN
    def activate_pnn(self, input, activations):
        """
        Forwarding with PNN activations
        :param input:
        :param activations:
        :return:
        """

        # GET FEATURES
        char_embs = self.dropout(self.char_embedding(input[CharEncoder.FEATURE_NAME]))
        char_cnn = self.char_cnn(char_embs.float().view(-1,
                                                        char_embs.size()[2],
                                                        char_embs.size()[3]).permute(0, 2, 1))
        char_feats, _ = torch.max(char_cnn, 2)

        word_embs = self.word_embedding(input[WordEncoder.FEATURE_NAME])

        word_feats = torch.cat([char_feats.float(), word_embs.view(-1, word_embs.size()[-1]).float()], dim=-1)

        word_feats = self.dropout(word_feats.view(word_embs.size()[0], word_embs.size()[1], -1))

        packed = torch.nn.utils.rnn.pack_padded_sequence(word_feats, input['timesteps'].tolist(), batch_first=True)
        (out, _) = self.rnn_1(packed)
        (out, _) = self.rnn_2(out)
        (unpacked_out, _) = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unpacked_out = unpacked_out.contiguous()
        unpacked_out = F.tanh(torch.add(unpacked_out, activations[0]))
        unpacked_out = self.dropout(unpacked_out)

        feats = self.fc(unpacked_out.view(-1, self.rnn_output_size)).view(word_embs.size()[0], word_embs.size()[1], -1)
        feats = torch.add(feats, activations[1])

        return feats

    def loss_PNN(self, input, activations, labels, label_pad_idx):
        """
        Forwarding and getting loss
        :param input:
        :param activations:
        :param labels:
        :param label_pad_idx:
        :return:
        """
        feats = self.activate_pnn(input, activations)
        # LOSS calculation
        entities = labels[EntityEncoder.FEATURE_NAME]
        mask = (entities != label_pad_idx).float()
        NLL = self.decoder(feats, entities, mask=mask)
        return NLL

    def predict_PNN(self, X, activations):
        logits = self.activate_pnn(X, activations)
        mask = torch.zeros(X['timesteps'].size(0), torch.max(X['timesteps']))
        for row_idx, row in enumerate(mask):
            mask[row_idx] = torch.tensor(
                [1] * X['timesteps'][row_idx].item() + [0] * (torch.max(X['timesteps']) - X['timesteps'][row_idx]).item())
        preds = self.decoder.predict(logits, mask)
        return preds


