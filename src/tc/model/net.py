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
                               out_channels=100,
                               kernel_size=(3, 100))
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=100,
                               kernel_size=(4, 100))
        self.conv3 = nn.Conv2d(in_channels=1,
                               out_channels=100,
                               kernel_size=(5, 100))

        self.fc_1 = nn.Linear(300, num_tags)
        # self.fc_2 = nn.Linear(128, num_tags)

    def reset_layers(self, num_output_tags):
        self.fc_1 = nn.Linear(300, num_output_tags)
        # self.fc_2 = nn.Linear(128, num_output_tags)

    def _conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    # FORWARD for TC
    def forward(self, input, labels, label_pad_idx):
        word_embs = self.word_embedding(input[WordEncoder.FEATURE_NAME])
        word_embs = word_embs.unsqueeze(1)
        conv_1 = self._conv_block(word_embs, self.conv1)
        conv_2 = self._conv_block(word_embs, self.conv2)
        conv_3 = self._conv_block(word_embs, self.conv3)
        sent_feat = self.dropout(torch.cat([conv_1, conv_2, conv_3], 1))
        # logits = self.fc_1(sent_feat)
        # logits = self.dropout(F.relu(logits))
        logits = self.fc_1(sent_feat)
        loss = self.loss_fn(logits, labels[ClassEncoder.FEATURE_NAME].squeeze())
        return loss, logits

    def predict(self, logits, mask):
        preds = []
        logits, mask = logits.data, mask.data
        for logit, mask in zip(logits, mask):
            sequence_length = torch.tensor(torch.sum(mask), dtype=torch.int)
            logit = logit[:sequence_length]
            conf, idx = torch.max(logit, 0)
            preds.append(idx.item())
        return preds


