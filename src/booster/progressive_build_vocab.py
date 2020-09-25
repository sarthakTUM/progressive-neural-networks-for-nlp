"""Build vocabularies of words and tags from datasets"""

import argparse
from collections import Counter
import json
import os
from src.booster.progressive_encoder import WordEncoder, CharEncoder, EntityEncoder
from src.ner.data import SLIterator
from src.ner import utils

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for tags in the dataset", type=int)
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")


def save_vocab_to_txt_file(vocab, txt_path, encoding='utf-8'):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w", encoding=encoding) as f:
        for token in vocab:
            f.write(token + '\n')
            

def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


def get_stats(data_iterator, words, tags):
    sentence_gen = data_iterator.get_next_X()
    label_gen = data_iterator.get_next_Y()
    num_sentences = 0
    num_labels = 0
    for sent, label in zip(sentence_gen, label_gen):
        assert len(sent) == len(label)
        words.update(sent)
        tags.update(label)
        if sent:
            num_sentences += 1
        if label:
            num_labels += 1

    assert num_sentences == num_labels
    return num_sentences


if __name__ == '__main__':

    # based on the type of data, load iterator
    data_folder = '/media/sarthak/HDD/TUM/Thesis/thesis-sarthak/src/ner/datasets/ncbi'
    feats_folder = os.path.join(data_folder, 'feats')
    embeddings_dim = 100
    embeddings_type = 'glove'
    embeddings_folder = '/media/sarthak/HDD/TUM/Thesis/data/embeddings/glove.6B.100d.txt'

    # feats_folder = os.path.join(data_folder, 'w2vfeats')
    if not os.path.exists(feats_folder):
        os.makedirs(feats_folder)
    print('loading train')
    train_data_iterator = SLIterator(os.path.join(data_folder, 'train'))
    print('loading val')
    val_data_iterator = SLIterator(os.path.join(data_folder, 'val'))
    print('loading test')
    test_data_iterator = SLIterator(os.path.join(data_folder, 'test'))
    # data_iterators = [train_data_iterator, val_data_iterator, test_data_iterator]
    data_iterators = [train_data_iterator, val_data_iterator]

    words = Counter()
    tags = Counter()
    print('getting stats')
    num_sentences_train = get_stats(train_data_iterator, words, tags)
    num_sentences_val = get_stats(val_data_iterator, words, tags)
    num_sentences_test = get_stats(test_data_iterator, words, tags)

    # for all types of features for which vocab has to be created, load encoders
    print('encoding')
    from collections import OrderedDict
    data_encoders = OrderedDict()
    label_encoders = OrderedDict()
    data_encoders[WordEncoder.FEATURE_NAME] = WordEncoder(vocab_path=embeddings_folder,
                                                          dim=embeddings_dim,
                                                          type=embeddings_type)
    """data_encoders[WordEncoder.FEATURE_NAME] = WordEncoder('../ner/embeddings/glove.6B.100d.txt',
                                                          dim=100,
                                                          type='glove')"""
    """data_encoders[WordEncoder.FEATURE_NAME] = WordEncoder('D:/wikipedia-pubmed-and-PMC-w2v.bin',
                                                          dim=200,
                                                          type='word2vec')"""
    data_encoders[CharEncoder.FEATURE_NAME] = CharEncoder()
    label_encoders[EntityEncoder.FEATURE_NAME] = EntityEncoder()

    print('creating and saving maps..')
    # create vocabs with iterators and encoders.
    for feature_name, encoder in data_encoders.items():
        print('creating map: {}'.format(feature_name))
        encoder.create_map(data_iterators)
        # save maps in the data folder.
        for map_name, map in encoder.maps.items():
            utils.save_map(map, map_name, feats_folder)

    for feature_name, encoder in label_encoders.items():
        print('creating map: {}'.format(feature_name))
        encoder.create_map(data_iterators)
        # save maps in the data folder.
        for map_name, map in encoder.maps.items():
            utils.save_map(map, map_name, feats_folder)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    # save_vocab_to_txt_file(words, os.path.join(data_folder, 'words.txt'), encoding='ISO-8859-1')
    # save_vocab_to_txt_file(tags, os.path.join(data_folder, 'tags.txt'), encoding='ISO-8859-1')
    save_vocab_to_txt_file(words, os.path.join(data_folder, 'words.txt'))
    save_vocab_to_txt_file(tags, os.path.join(data_folder, 'tags.txt'))
    print("- done.")

    # Save datasets properties in json file
    print('saving dataset stats')
    encoder_params = dict()
    encoders = [data_encoders, label_encoders]
    for encoder in encoders:
        for encoder_name, enc in encoder.items():
            encoder_params['pad_'+encoder_name] = enc.PAD
            encoder_params['unk_'+encoder_name] = enc.UNK

    sizes = {
        'train_size': num_sentences_train,
        'val_size': num_sentences_val,
        'test_size': num_sentences_test,
        'vocab_size': len(words),
        'number_of_tags': len(tags),
        'special_tokens': encoder_params,
        'data_iterators': {'train': train_data_iterator.__class__.__name__,
                           'val': val_data_iterator.__class__.__name__,
                           'test': test_data_iterator.__class__.__name__},
        'unknown_words': WordEncoder.unknown_words
    }
    save_dict_to_json(sizes, os.path.join(data_folder, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
