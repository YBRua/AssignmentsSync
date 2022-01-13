# coding=utf8
import os
import json

import pypinyin
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'


def pinyin_and_word_vocab_from_file(
        filepath: str,
        padding=False,
        unk=False,
        min_freq=1):
    word_vocab = Vocab(padding=padding, unk=unk, min_freq=min_freq)
    pinyin_vocab = PinyinVocab(padding=padding, unk=unk, min_freq=min_freq)
    with open(filepath, 'r', encoding='utf-8') as f:
        trains = json.load(f)

    word_freq = {}
    pinyin_freq = {}
    for data in trains:
        for utt in data:
            text = utt['asr_1best']
            for char in text:
                pinyins = pypinyin.pinyin(char, style=pypinyin.NORMAL)
                assert len(pinyins) == 1, "BUG"
                pinyin = pinyins[0][0]
                word_freq[char] = word_freq.get(char, 0) + 1
                pinyin_freq[pinyin] = pinyin_freq.get(pinyin, 0) + 1
    for word in word_freq:
        if word_freq[word] >= min_freq:
            idx = len(word_vocab.word2id)
            word_vocab.word2id[word], word_vocab.id2word[idx] = idx, word
    for word in pinyin_freq:
        if pinyin_freq[word] >= min_freq:
            idx = len(pinyin_vocab.word2id)
            pinyin_vocab.word2id[word], pinyin_vocab.id2word[idx] = idx, word

    return word_vocab, pinyin_vocab


class PinyinVocab():
    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(PinyinVocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath, 'r', encoding='utf-8') as f:
            trains = json.load(f)
        word_freq = {}
        for data in trains:
            for utt in data:
                text = utt['asr_1best']
                for char in text:
                    pinyin = pypinyin.pinyin(char)
                    word_freq[pinyin] = word_freq.get(pinyin, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])


class Vocab():
    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath, 'r', encoding='utf-8') as f:
            trains = json.load(f)
        word_freq = {}
        for data in trains:
            for utt in data:
                text = utt['asr_1best']
                for char in text:
                    word_freq[char] = word_freq.get(char, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])


class LabelVocab():

    def __init__(self, root):
        self.tag2idx, self.idx2tag = {}, {}

        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(
            open(os.path.join(root, 'ontology.json'), 'r', encoding='utf-8'))
        acts = ontology['acts']
        slots = ontology['slots']

        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)
