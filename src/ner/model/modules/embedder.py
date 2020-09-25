from torch import nn
import torch
import numpy as np
np.random.seed(0)
import math


class Embedder(nn.Module):
    def __init__(self,
                 features: dict,
                 mode: str,
                 ) -> torch.Tensor:
        super(Embedder, self).__init__()
        self.features = features
        self.mode = mode
        for feature in self.features:
            if feature == 'WORD':
                if feature['WORD']['pretrained']:
                    self.word_embedding = nn.Embedding.from_pretrained(feature['WORD']['pretrained'],
                                                                       freeze=feature['WORD']['freeze_embeddings'])
                else:
                    # TODO complete word embedding not pretrained
                    pass
            if feature == 'CHAR':
                char_matrix = np.random.uniform(low=-(math.sqrt(3 / feature['CHAR']['emb_dim'])),
                                                high=math.sqrt(3 / feature['CHAR']['emb_dim']),
                                                size=(feature['CHAR']['num_chars'], feature['CHAR']['emb_dim']))
                self.char_embedding = nn.Embedding.from_pretrained(torch.from_numpy(char_matrix), freeze=False)
                if feature['CHAR']['type'] == 'CNN':
                    self.char_cnn = nn.Conv1d(in_channels=feature['CHAR']['emb_dim'],
                                              out_channels=feature['CHAR']['num_filters'],
                                              kernel_size=feature['CHAR']['kernel_size'],
                                              padding=((feature['CHAR']['kernel_size'] - 1) / 2))

    def forward(self,
                input: dict(str)):
        """

        :param input: shape (NxD) where N = number of instances, D = number of dimension of each instance.
        :return: shape: (N x D x E) where E = embedding dimension
        """
        embeddings = []
        if 'WORD' in input:
            word_embs = self.word_embedding(input['WORD']).view(-1, self.features['WORD']['emb_dim']).float()
            embeddings.append(word_embs)
        if 'CHAR' in input:
            char_embs = nn.Dropout(self.char_embedding(input['CHAR']), 0.5)
            char_embs = self.char_cnn(char_embs.float().view(-1,
                                                            char_embs.size()[2],
                                                            char_embs.size()[3]).permute(0, 2, 1))
            char_embs, _ = torch.max(char_embs, 2)
            embeddings.append(char_embs)

        embedded = None
        if self.mode == 'CONCAT':
            embedded = torch.cat(embeddings, dim=-1)

        # TODO find a better way to get the shape
        return embedded.view(word_embs.size()[0], word_embs.size()[1], -1)
