"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ner.encoder import WordEncoder, ClassEncoder
from torch.nn import CrossEntropyLoss


np.random.seed(0)


class CNNTC(nn.Module):
    def __init__(self,
                 num_tags: int,
                 pretrained_word_vecs: torch.Tensor,
                 dropout: float = 0.5,
                 freeze_embeddings: bool = True):
        super(CNNTC, self).__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.loss_fn = CrossEntropyLoss()
        self._freeze_embeddings = freeze_embeddings
        self.word_vec_dim = pretrained_word_vecs.size()[1]

        self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_vecs,
                                                           freeze=freeze_embeddings)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=200,
                               kernel_size=(3, 100))
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=200,
                               kernel_size=(4, 100))
        self.conv3 = nn.Conv2d(in_channels=1,
                               out_channels=200,
                               kernel_size=(5, 100))

        self.fc_1 = nn.Linear(600, 128)
        self.fc_2 = nn.Linear(128, num_tags)

    def _conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def reset_layers(self, num_output_tags):
        self.fc_1 = nn.Linear(600, 128)
        self.fc_2 = nn.Linear(128, num_output_tags)

    def activate(self, input):
        """
        Forwarding without PNN activations
        :param input:
        :return:
        """
        word_embs = self.word_embedding(input[WordEncoder.FEATURE_NAME])
        word_embs = word_embs.unsqueeze(1)
        conv_1 = self._conv_block(word_embs, self.conv1)
        conv_2 = self._conv_block(word_embs, self.conv2)
        conv_3 = self._conv_block(word_embs, self.conv3)
        sent_feat = self.dropout(torch.cat([conv_1, conv_2, conv_3], 1))
        logits_1 = self.fc_1(sent_feat)
        logits_1 = self.dropout(F.relu(logits_1))
        logits_2 = self.fc_2(logits_1)
        return [sent_feat, logits_1, logits_2]

    def loss(self, input, labels):
        """
        Forwarding and getting loss
        :param input:
        :param labels:
        :param label_pad_idx:
        :return:
        """
        feats = self.activate(input)
        # LOSS calculation
        loss = self.loss_fn(feats, labels[ClassEncoder.FEATURE_NAME].squeeze())
        return loss

    def predict(self, X):
        preds = []
        logits = self.activate(X)
        mask = torch.zeros(X['timesteps'].size(0), torch.max(X['timesteps']))
        for row_idx, row in enumerate(mask):
            mask[row_idx] = torch.tensor(
                [1] * X['timesteps'][row_idx].item() + [0] * (
                            torch.max(X['timesteps']) - X['timesteps'][row_idx]).item())
        for logit, mask in zip(logits, mask):
            sequence_length = torch.tensor(torch.sum(mask), dtype=torch.int)
            logit = logit[:sequence_length]
            conf, idx = torch.max(logit, 0)
            preds.append(idx.item())
        return preds

    # forward for PNN
    def activate_pnn(self, input, activations):
        """
        Forwarding with PNN activations
        :param input:
        :param activations:
        :return:
        """

        word_embs = self.word_embedding(input[WordEncoder.FEATURE_NAME])
        word_embs = word_embs.unsqueeze(1)
        conv_1 = self._conv_block(word_embs, self.conv1)
        conv_2 = self._conv_block(word_embs, self.conv2)
        conv_3 = self._conv_block(word_embs, self.conv3)
        sent_feat = self.dropout(torch.cat([conv_1, conv_2, conv_3], 1))
        logits_1 = self.fc_1(sent_feat)
        logits_1 = torch.add(logits_1, activations[0])
        logits_1 = self.dropout(F.relu(logits_1))
        logits_2 = self.fc_2(logits_1)
        logits_2 = torch.add(logits_2, activations[1])
        return logits_2

    def loss_PNN(self, input, activations, labels):
        """
        Forwarding and getting loss
        :param input:
        :param labels:
        :param label_pad_idx:
        :return:
        """
        feats = self.activate_pnn(input, activations)
        # LOSS calculation
        loss = self.loss_fn(feats, labels[ClassEncoder.FEATURE_NAME].squeeze())
        return loss

    def predict_PNN(self, X, activations):
        preds = []
        logits = self.activate_pnn(X, activations)
        mask = torch.zeros(X['timesteps'].size(0), torch.max(X['timesteps']))
        for row_idx, row in enumerate(mask):
            mask[row_idx] = torch.tensor(
                [1] * X['timesteps'][row_idx].item() + [0] * (
                        torch.max(X['timesteps']) - X['timesteps'][row_idx]).item())
        for logit, mask in zip(logits, mask):
            sequence_length = torch.tensor(torch.sum(mask), dtype=torch.int)
            logit = logit[:sequence_length]
            conf, idx = torch.max(logit, 0)
            preds.append(idx.item())
        return preds


