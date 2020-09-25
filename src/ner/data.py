import os
import logging
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer as twt


class StringIterator:
    def __init__(self, string):
        self.string = string
        self.list_of_sentences = self._split_sentences(string)
        self.list_of_sentences_tokenized, self.spans, _ = self._split_words(self.list_of_sentences)

    def _split_sentences(self, string):
        return sent_tokenize(string)

    def _split_words(self, list_of_sentences):
        sentence_split_words = []
        sentence_spans = []
        sentences_strings = []
        for sentence in list_of_sentences:
            sentences_strings.append(sentence)
            spans = list(twt().span_tokenize(sentence))
            sentence_spans.append(spans)
            sentence_split_words.append([sentence[t[0]:t[1]] for t in spans])
        return sentence_split_words, sentence_spans, sentences_strings

    def get_next_X(self):
        for sentence in self.list_of_sentences_tokenized:
            assert len(sentence) > 0
            yield sentence


class SLIterator:
    def __init__(self, data_path):

        self.data_path = data_path
        self.sentences_path = os.path.join(data_path, 'sentences.txt')
        self.labels_path = os.path.join(data_path, 'labels.txt')
        if not os.path.exists(self.sentences_path):
            raise Exception('Data Folder doesn\'t exist: {}'.format(data_path))
        if not os.path.exists(self.labels_path):
            logging.log('warn', 'scoring mode for {}'.format(self.data_path))

    def get_next_X(self, mode='one_word_split', encoding='utf-8'):
        if mode == 'one_word_split':
            with open(self.sentences_path, 'r', encoding=encoding) as sentences_file:
                while True:
                    sentence = sentences_file.readline()
                    if not sentence:
                        break
                    sentence = sentence.split(" ")
                    assert len(sentence) > 0
                    yield sentence

    def get_next_Y(self, mode='token2tag'):
        if mode == 'token2tag':
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as tags_file:
                    while True:
                        tag = tags_file.readline().rstrip()
                        if not tag:
                            break
                        tag = tag.split(" ")
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
