import torch.nn as nn
from src.ner.model.modules.crf import ConditionalRandomField


class CRFDecoder(nn.Module):
    def __init__(self,
                 in_size,
                 out_size):
        super(CRFDecoder, self).__init__()
        self.decoder = ConditionalRandomField(in_size, out_size)

    def forward(self, logits, labels, mask):
        log_lik = self.decoder(logits, labels, mask=mask)
        return -1 * log_lik

    def predict(self, features, mask):
        preds = self.decoder.viterbi_tags(features, mask)
        return [pred[0] for pred in preds]
