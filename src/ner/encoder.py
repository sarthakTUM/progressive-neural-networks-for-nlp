import os
import numpy as np
from collections import Counter
from src.ner import utils


np.random.seed(0)


class Encoder:
    def encode(self, sentence):
        pass


class ClassEncoder(Encoder):

    FEATURE_NAME = 'CLASS'

    def __init__(self, path=None):
        self.class2idx, self.idx2class = None, None
        if path:
            self.class2idx = utils.json_to_dict(os.path.join(path, 'class2idx.json'))
            self.idx2class = utils.json_to_dict(os.path.join(path, 'idx2class.json'), int_keys=True)
        self.PAD = None
        self.UNK = None

    def create_map(self, data_iterators):
        vocab = Counter()
        for iterator in data_iterators:
            gen = iterator.get_next_Y()
            for label in gen:
                vocab.update(label)
        print('Class Distribution: ', vocab)
        self.class2idx = {k: v for v, k in enumerate(list(vocab.keys()))}
        self.idx2class = {int(v): k for k, v in self.class2idx.items()}

    def encode(self, label):
        """# one-hot encode
                labels = [0]*len(self.class2idx)
                for l in label:
                    labels[self.class2idx[l]] = 1
                return labels"""
        return [self.class2idx[l] for l in label]

    def decode(self, encoded_seq):
        return [self.idx2class[e] for e in encoded_seq]

    def pad(self, label, max_len, params):
        return label

    @property
    def maps(self):
        return {'class2idx': self.class2idx, 'idx2class': self.idx2class}

    @property
    def num_tags(self):
        return len(self.class2idx)

    @property
    def pad_idx(self):
        return None


class EntityEncoder(Encoder):

    FEATURE_NAME = 'ENTITY'

    def __init__(self, path=None):
        self.entity2idx, self.idx2entity = None, None
        if path:
            self.entity2idx = utils.json_to_dict(os.path.join(path, 'entity2idx.json'))
            self.idx2entity = utils.json_to_dict(os.path.join(path, 'idx2entity.json'), int_keys=True)
        self.PAD = '__PAD__'
        self.UNK = None

    def create_map(self, data_iterators):
        __PAD__ = self.PAD
        vocab = Counter()
        for iterator in data_iterators:
            gen = iterator.get_next_Y()
            for label in gen:
                vocab.update(label)
        vocab.update([__PAD__])
        print('Entity distribution: ', vocab)
        self.entity2idx = {k: v for v, k in enumerate(list(vocab.keys()))}
        self.idx2entity = {int(v): k for k, v in self.entity2idx.items()}

    def encode(self, label):
        return [self.entity2idx[l] for l in label]

    def decode(self, encoded_seq):
        return [self.idx2entity[e] for e in encoded_seq]

    def pad(self, entities, max_len, params):
        assert params.__dict__['special_tokens']['pad_' + self.FEATURE_NAME] == self.PAD
        entities.extend([self.entity2idx[self.PAD]] * (max_len - len(entities)))
        assert len(entities) == max_len
        return entities

    @property
    def maps(self):
        return {'entity2idx': self.entity2idx, 'idx2entity': self.idx2entity}

    @property
    def num_tags(self):
        return len(self.entity2idx)

    @property
    def pad_idx(self):
        return self.entity2idx[self.PAD]


class CharEncoder(Encoder):

    FEATURE_NAME = 'CHAR'

    def __init__(self, path=None):
        self.char2idx, self.idx2char = None, None
        self.UNK = '__UNK__'
        self.PAD = '__PAD__'
        if path:
            self.char2idx = utils.json_to_dict(os.path.join(path, 'char2idx.json'))
            self.idx2char = utils.json_to_dict(os.path.join(path, 'idx2char.json'), int_keys=True)

    def create_map(self, data_iterators):
        """
        Create a dictionary and mapping of characters, sorted by frequency.
        """
        # TODO: remove this hardcoding of __PAD__ and __UNK__
        __UNK__ = self.UNK
        __PAD__ = self.PAD

        vocab = Counter()
        vocab.update([__PAD__])
        vocab.update([__UNK__])
        for iterator in data_iterators:
            gen = iterator.get_next_X()
            for sentence in gen:
                list_of_chars = [list(word) for word in sentence]
                flat_list_of_chars = [item for sublist in list_of_chars for item in sublist]
                vocab.update(flat_list_of_chars)
        print('Char distribution: ', vocab)
        self.char2idx = {k: v for v, k in enumerate(list(vocab.keys()))}
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}

    def encode(self, sentence):
        chars = []
        for word in sentence:
            chars.append([self.char2idx[c] if c in self.char2idx else self.char2idx[self.UNK] for c in list(word)])
        return chars

    def pad(self, sentence, max_len, params):
        assert params.__dict__['special_tokens']['pad_' + self.FEATURE_NAME] == self.PAD
        max_word_length = params.max_word_length
        for word_idx, word in enumerate(sentence):

            # cut word if greater than max_word_length
            if len(word) >= max_word_length:
                word = word[:max_word_length]
                sentence[word_idx] = word
            # add extra chars if word smaller than max_word_length
            else:
                word.extend([self.char2idx[self.PAD]] * (max_word_length - len(word)))

        # add words if num_words less than max_sentence_length
        sentence.extend([[self.char2idx[self.PAD]] * max_word_length] * (max_len - len(sentence)))
        return sentence

    @property
    def maps(self):
        return {'char2idx': self.char2idx, 'idx2char': self.idx2char}

    @property
    def vocab_length(self):
        return len(self.char2idx)

    @property
    def pad_idx(self):
        return self.char2idx[self.PAD]


class WordEncoder(Encoder):

    FEATURE_NAME = 'WORD'

    def __init__(self, vocab_path, dim, type='glove', word_to_vec_path=None):
        self.vocab_path = vocab_path
        self.type = type
        if not os.path.exists(vocab_path):
            print('directory doesn\'t exist')
        self.UNK = '__UNK__'
        self.PAD = '__PAD__'
        if word_to_vec_path:
            self.word2vec = np.load(word_to_vec_path)
        if os.path.isdir(vocab_path):
            if not self.word2vec:
                self.word2vec = np.load(os.path.join(vocab_path, 'word2vec.npy'))
            self.word2idx = utils.json_to_dict(os.path.join(vocab_path, 'word2idx.json'))
            self.idx2word = utils.json_to_dict(os.path.join(vocab_path, 'idx2word.json'), int_keys=True)
            # assert self.word2vec.shape[1] == dim
        else:
            self.word2vec, self.word2idx, self.idx2word = None, None, None
        self.dim = dim

    def create_map(self, data_iterators):
        if self.type == 'glove':
            # create word to vector map reading from the vocab path
            vectors = []
            self.word2idx = dict()

            __UNK__ = self.UNK
            __PAD__ = self.PAD
            self.word2idx[__UNK__] = 1
            self.word2idx[__PAD__] = 0

            # append a zero vector for PAD words
            vectors.append(np.zeros(self.dim))

            # append a random vector for UNK words
            vectors.append(np.random.uniform(-0.25, 0.25, self.dim))

            # idx 0 and 1 are occupied by __PAD__ and __UNK__ respectively
            idx = 2
            with open(self.vocab_path, 'rb') as f:
                for l in f:
                    line = l.decode().split()

                    # add all words in lower case form
                    word = line[0].lower()

                    self.word2idx[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    assert len(vect) == self.dim
                    vectors.append(vect)
            self.word2vec = np.array(vectors)
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}
            assert len(self.word2idx) == len(self.idx2word)
            assert len(self.word2idx) == len(self.word2vec)
            assert self.word2vec.shape[1] == self.dim

        elif self.type == 'word2vec':
            import gensim

            __UNK__ = self.UNK
            __PAD__ = self.PAD

            # 1. load the model
            print('loading model..')
            model = gensim.models.KeyedVectors.load_word2vec_format(self.vocab_path, binary=True)

            # 2. get word2vec, word2idx and idx2word
            print('creating vectors..')
            self.word2vec = np.zeros((model.vectors.shape[0]+2, model.vectors.shape[1]), dtype=np.float32)
            # self.word2vec[0] = np.zeros(self.dim) -> PAD token
            self.word2vec[1] = np.random.uniform(-0.25, 0.25, self.dim) # UNK token
            self.word2vec[2:] = model.vectors
            print('creating word2idx')
            self.word2idx = {k: v+2 for v, k in enumerate(model.vocab)} # 0 and 1 position occupied by PAD and UNK
            self.word2idx[__PAD__] = 0
            self.word2idx[__UNK__] = 1
            print('creating idx2word')
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}
            assert len(self.word2idx) == len(self.idx2word)
            assert len(self.word2idx) == len(self.word2vec)
            assert self.word2vec.shape[1] == self.dim

    def pad(self, sentence, max_len, params):
        assert params.__dict__['special_tokens']['pad_' + self.FEATURE_NAME] == self.PAD
        sentence.extend([self.word2idx[self.PAD]] * (max_len - len(sentence)))
        assert len(sentence) == max_len
        return sentence

    @property
    def maps(self):
        return {'word2idx': self.word2idx, 'idx2word': self.idx2word, 'word2vec': self.word2vec}

    def encode(self, sentence):
        encoded = []
        for word in sentence:
            if word in self.word2idx:
                encoded.append(self.word2idx[word])
            elif word.lower() in self.word2idx:
                encoded.append(self.word2idx[word.lower()])
            else:
                encoded.append(self.word2idx[self.UNK])
        return encoded

    @property
    def vocab_length(self):
        return len(self.word2idx)

    @property
    def pad_idx(self):
        return self.word2idx[self.PAD]

    @property
    def emb_dim(self):
        return self.word2vec.shape[1]

    @property
    def vectors(self):
        return self.word2vec