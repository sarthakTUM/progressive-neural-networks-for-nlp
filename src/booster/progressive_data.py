import os
import logging


class SLIterator:
    def __init__(self, data_path):
        # TODO: there could be S and L for each document
        # TODO: see if multiple type of XY iterators can be created

        self.data_path = data_path
        self.sentences_path = os.path.join(data_path, 'sentences.txt')
        self.labels_path = os.path.join(data_path, 'labels.txt')
        self.id_path = os.path.join(data_path, 'ID.txt')
        if not os.path.exists(self.sentences_path):
            raise Exception('Data Folder doesn\'t exist')
        if not os.path.exists(self.labels_path):
            logging.log('warn', 'scoring mode for {}'.format(self.data_path))

    def get_next_X(self, mode='one_word_split'):
        if mode == 'one_word_split':
            with open(self.sentences_path, 'r', encoding='utf-8') as sentences_file:
                while True:
                    sentence = sentences_file.readline().rstrip()
                    if not sentence:
                        break
                    sentence = sentence.split(' ')
                    assert len(sentence) > 0
                    yield sentence

    def get_next_ID(self):
        with open(self.id_path, 'r') as id_file:
            while True:
                id = id_file.readline().rstrip()
                if not id:
                    break
                yield id

    def get_next_Y(self, mode='token2tag'):
        if mode == 'token2tag':
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as tags_file:
                    while True:
                        tag = tags_file.readline().rstrip()
                        if not tag:
                            break
                        tag = tag.split(' ')
                        assert len(tag) > 0
                        yield tag
            else:
                raise Exception('file doesn\'t exist')
        if mode == 'sentence2tag':
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as tags_file:
                    while True:
                        tag = tags_file.readline().rstrip()
                        if not tag:
                            break
                        assert len(tag) == 1
                        yield tag
            else:
                raise Exception('file doesn\'t exist')
    # TODO implement methods like __get_item__ and __len__, etc.


def iob_iobes(tags):
    """
    IOB -> IOBES
    implementation taken from: https://github.com/zalandoresearch/flair/blob/master/flair/data.py
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.value.replace('B-', 'S-'))
        elif tag.value.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].value.split('-')[0] == 'I':
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags