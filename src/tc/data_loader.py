import random
import os
import torch
from src.ner import utils
from src.ner.data import SLIterator
from collections import OrderedDict
from collections import defaultdict


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """
    def __init__(self, params, data_dir, data_encoder, label_encoder):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have run `build_vocab.py` on data_dir before using this
        class.

        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params (such as vocab size, num_of_tags etc.) to params.
        """

        # loading dataset_params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.data_encoder = data_encoder
        self.label_encoder = label_encoder
        self.data_dir = data_dir
        params.update(json_path)
        self.dataset_params = utils.Params(json_path)
        torch.manual_seed(230)
        if params.cuda: torch.cuda.manual_seed(230)

    def load_sentences_labels_features(self, data_iterator, d):
        """
        Loads sentences and labels from their corresponding files. Maps tokens and tags to their indices and stores
        them in the provided dict d.

        Args:
            sentences_file: (string) file with sentences with tokens space-separated
            labels_file: (string) file with NER tags for the sentences in labels_file
            d: (dict) a dictionary in which the loaded data is stored
        """
        sentences = []
        labels = []

        sentence_gen = data_iterator.get_next_X()
        label_gen = data_iterator.get_next_Y()

        for sentence in sentence_gen:
            s = OrderedDict()
            for feature_name, encoder in self.data_encoder.items():
                s[feature_name] = encoder.encode(sentence)
            s['timesteps'] = len(sentence)
            s['word_length'] = [len(list(w)) for w in sentence]
            sentences.append(s)

        for label in label_gen:
            l = OrderedDict()
            for feature_name, encoder in self.label_encoder.items():
                l[feature_name] = encoder.encode(label)
            l['timesteps'] = len(label)
            labels.append(l)

        assert len(labels) == len(sentences)

        # storing sentences and labels in dict d
        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types

        """
        data = {}
        for split in ['train', 'val', 'test']:
            if split in types:
                iterator_type = self.dataset_params.dict['data_iterators'][split]

                # get the type of data_iterator
                data_iterator = globals()[iterator_type](os.path.join(data_dir, split))
                data[split] = {}
                self.load_sentences_labels_features(data_iterator, data[split])

        return data

    def batch_iterator(self, data, params, shuffle=False):
        # TODO convert below utilities into functions and put in utils
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.

        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels

        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # one pass over data
        for i in range((data['size'] + 1) // params.batch_size):
            # 1. fetch sentences and tags
            batch_sentences = [data['data'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]]

            # 2. sort the length in descending order of sentence_length
            sorted_data = [b for b in sorted(enumerate(batch_sentences),
                                             key=lambda k: k[1]['timesteps'],
                                             reverse=True)]
            sorted_index = [i[0] for i in sorted_data]
            batch_sentences = [i[1] for i in sorted_data]
            sorted_labels = [batch_tags[i] for i in sorted_index]
            batch_tags = sorted_labels

            # 3. compute length of longest sentence in batch
            batch_max_len = max([s['timesteps'] for s in batch_sentences])

            # 4. PAD
            for s in batch_sentences:
                s['word_length'].extend([-1]*(batch_max_len-len(s['word_length'])))
                for feature_name, encoded_value in s.items():
                    for encoder_name, encoder in self.data_encoder.items():
                        if encoder_name == feature_name:
                            encoder.pad(encoded_value, batch_max_len, params)
                            break
            for l in batch_tags:
                for feature_name, encoded_value in l.items():
                    for encoder_name, encoder in self.label_encoder.items():
                        if encoder_name == feature_name:
                            encoder.pad(encoded_value, batch_max_len, params)
                            break

            # 5. create dictionary of the form: dict(feature_name: list(encoded_sentence))
            super_data_dict = defaultdict(list)
            for s in batch_sentences:
                for k, v in s.items():
                    super_data_dict[k].append(v)
            super_label_dict = defaultdict(list)
            for s in batch_tags:
                for k, v in s.items():
                    super_label_dict[k].append(v)

            # 6. convert the lists to torch.tensor
            for encoder_name, encoded_value in super_data_dict.items():
                super_data_dict[encoder_name] = torch.tensor(encoded_value).cuda() if params.cuda else torch.tensor(encoded_value)
            for encoder_name, encoded_value in super_label_dict.items():
                super_label_dict[encoder_name] = torch.tensor(encoded_value).cuda() if params.cuda else torch.tensor(encoded_value)

            # 7. Let's go!
            yield super_data_dict, super_label_dict

