import json
import pypinyin

from utils.vocab import PinyinVocab, Vocab, LabelVocab
from utils.vocab import pinyin_and_word_vocab_from_file
from utils.word2vec import Pinyin2VecUtils, Word2vecUtils
from utils.evaluator import Evaluator
from typing import List


class Example():

    @classmethod
    def configuration(
            cls,
            root,
            train_path=None,
            word2vec_path=None,
            with_pinyin=False):
        cls.evaluator = Evaluator()
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)
        if with_pinyin:
            cls.word_vocab, cls.pinyin_vocab = pinyin_and_word_vocab_from_file(
                filepath=train_path,
                padding=True,
                unk=True)
            cls.pinyin2vec = Pinyin2VecUtils(word2vec_path)
        else:
            cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
            cls.pinyin_vocab = None
            cls.pinyin2vec = None

    @classmethod
    def load_dataset(cls, data_path, use_transcript=False):
        if use_transcript:
            print('Using manual transcripts for', data_path)
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        examples: List[Example] = []
        for data in datas:
            for utt in data:
                example = cls(utt)
                examples.append(example)
                if use_transcript:
                    example = cls(utt, 'manual_transcript')
                    examples.append(example)
        return examples

    def __init__(self, ex: dict, utt_key='asr_1best'):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex[utt_key]
        if Example.pinyin_vocab is not None:
            self.pinyin = pypinyin.pinyin(self.utt, style=pypinyin.NORMAL)
        self.slot = {}
        if 'semantic' in ex:
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slt}-{val}' for slt, val in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        if Example.pinyin_vocab is not None:
            self.pinyin_idx = [Example.pinyin_vocab[c[0]] for c in self.pinyin]
        else:
            self.pinyin_idx = None
        label = Example.label_vocab
        self.tag_id = [label.convert_tag_to_idx(tag) for tag in self.tags]
