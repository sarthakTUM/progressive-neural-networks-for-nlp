import random
import os
import torch
from src.ner import utils
from src.booster.progressive_data import SLIterator
from collections import OrderedDict
from collections import defaultdict


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """
    def __init__(self, params, data_dir, data_encoder, label_encoder=None, device=torch.device('cpu')):

        # loading dataset_params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.data_encoder = data_encoder
        self.label_encoder = label_encoder
        self.data_dir = data_dir
        params.update(json_path)
        self.dataset_params = utils.Params(json_path)
        self.id_to_idx = None
        self.idx_to_id = None
        self.device = device

    def load_sentences_labels_features(self, data_iterator, d):

        sentences = []
        labels = []

        sentence_gen = data_iterator.get_next_X()
        label_gen = data_iterator.get_next_Y()
        id_gen = data_iterator.get_next_ID()

        for id, sentence in zip(id_gen, sentence_gen):
            s = OrderedDict()
            for feature_name, encoder in self.data_encoder.items():
                s[feature_name] = encoder.encode(sentence)
            s['timesteps'] = len(sentence)
            s['word_length'] = [len(list(w)) for w in sentence]
            s['ID'] = id
            sentences.append(s)

        for label in label_gen:
            l = OrderedDict()
            for feature_name, encoder in self.label_encoder.items():
                l[feature_name] = encoder.encode(label)
            l['timesteps'] = len(label)
            labels.append(l)

        assert len(labels) == len(sentences)
        # TODO write assertion for num_labels and num_timesteps

        # storing sentences and labels in dict d
        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_sentences_features(self, data_iterator, d):

        sentences = []
        ids = []

        sentence_gen = data_iterator.get_next_X()
        id_gen = data_iterator.get_next_ID()

        for id, sentence in zip(id_gen, sentence_gen):
            s = OrderedDict()
            for feature_name, encoder in self.data_encoder.items():
                s[feature_name] = encoder.encode(sentence)
            s['timesteps'] = len(sentence)
            s['word_length'] = [len(list(w)) for w in sentence]
            s['ID'] = id
            sentences.append(s)
            ids.append(id)

        # storing sentences and labels in dict d
        d['data'] = sentences
        d['labels'] = None
        d['size'] = len(sentences)
        d['id_to_idx'] = {id: idx for idx, id in enumerate(ids)}
        d['idx_to_id'] = {idx: id for idx, id in enumerate(ids)}

    def _map_to_id(self, data):
        ids_to_idx = {}
        for sentence_idx, sentence in enumerate(data):
            ids_to_idx[sentence['ID']] = sentence_idx
        idx_to_ids = {v:k for k,v in ids_to_idx.items()}
        return ids_to_idx, idx_to_ids

    def load_data(self, types):
        data = {}

        for split_dir in types:
            split = split_dir.split('/')[0]
            iterator_type = self.dataset_params.dict['data_iterators'][split]

            # get the type of data_iterator
            data_iterator = globals()[iterator_type](os.path.join(self.data_dir, split_dir))
            data[split] = {}
            if self.label_encoder:
                self.load_sentences_labels_features(data_iterator, data[split])
            else:
                self.load_sentences_features(data_iterator, data[split])
        return data

    def get_batch_X(self, data, batch_ID):

        # get the instances with the ID
        batch_sentences = [data['data'][idx] for idx in [data['id_to_idx'][ID] for ID in batch_ID]]

        # compute length of longest sentence in batch
        batch_max_len = max([s['timesteps'] for s in batch_sentences])

        # PAD
        for s in batch_sentences:
            s['word_length'].extend([-1] * (batch_max_len - len(s['word_length'])))
            for feature_name, encoded_value in s.items():
                for encoder_name, encoder in self.data_encoder.items():
                    if encoder_name == feature_name:
                        encoder.pad(encoded_value, batch_max_len)
                        break

        # create dictionary of the form: dict(feature_name: list(encoded_sentence))
        super_data_dict = defaultdict(list)
        for s in batch_sentences:
            for k, v in s.items():
                super_data_dict[k].append(v)

        # convert the lists to torch.tensor
        for encoder_name, encoded_value in super_data_dict.items():
            super_data_dict[encoder_name] = torch.tensor(encoded_value).to(self.device) if encoder_name != 'ID' else encoded_value

        return super_data_dict

    def batch_iterator(self, data, batch_size, shuffle=False):

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # one pass over data
        for i in range((data['size'] + 1) // batch_size):
            # fetch sentences and tags
            batch_sentences = [data['data'][idx] for idx in order[i * batch_size:(i + 1) * batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i * batch_size:(i + 1) * batch_size]]

            # sort the length in descending order of sentence_length
            sorted_data = [b for b in sorted(enumerate(batch_sentences),
                                             key=lambda k: k[1]['timesteps'],
                                             reverse=True)]
            sorted_index = [i[0] for i in sorted_data]
            batch_sentences = [i[1] for i in sorted_data]
            sorted_labels = [batch_tags[i] for i in sorted_index]
            batch_tags = sorted_labels

            # compute length of longest sentence in batch
            batch_max_len = max([s['timesteps'] for s in batch_sentences])

            # PAD
            for s in batch_sentences:
                s['word_length'].extend([-1]*(batch_max_len-len(s['word_length'])))
                for feature_name, encoded_value in s.items():
                    for encoder_name, encoder in self.data_encoder.items():
                        if encoder_name == feature_name:
                            encoder.pad(encoded_value, batch_max_len)
                            break
            for l in batch_tags:
                for feature_name, encoded_value in l.items():
                    for encoder_name, encoder in self.label_encoder.items():
                        if encoder_name == feature_name:
                            encoder.pad(encoded_value, batch_max_len)
                            break

            # create dictionary of the form: dict(feature_name: list(encoded_sentence))
            super_data_dict = defaultdict(list)
            for s in batch_sentences:
                for k, v in s.items():
                    super_data_dict[k].append(v)
            super_label_dict = defaultdict(list)
            for s in batch_tags:
                for k, v in s.items():
                    super_label_dict[k].append(v)

            # convert the lists to torch.tensor
            for encoder_name, encoded_value in super_data_dict.items():
                super_data_dict[encoder_name] = torch.tensor(encoded_value).to(self.device) if encoder_name != 'ID' else encoded_value
            for encoder_name, encoded_value in super_label_dict.items():
                super_label_dict[encoder_name] = torch.tensor(encoded_value).to(self.device) if encoder_name != 'ID' else encoded_value
            yield super_data_dict, super_label_dict

