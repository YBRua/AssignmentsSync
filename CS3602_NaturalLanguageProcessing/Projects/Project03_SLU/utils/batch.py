#-*- coding:utf-8 -*-
import torch

from utils.example import Example
from argparse import Namespace
from typing import List


def from_example_list(
        args: Namespace,
        ex_list: List[Example],
        device='cpu',
        train=True):
    ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    # for debuggin purposes
    # for ex in ex_list:
    #     assert len(ex.pinyin_idx) == len(ex.input_idx)
    if ex_list[0].pinyin_idx is not None:
        pinyin_ids = [ex.pinyin_idx + [pad_idx] * (max_len - len(ex.pinyin_idx)) for ex in ex_list]
        batch.pinyin_ids = torch.tensor(pinyin_ids, dtype=torch.long, device=device)
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        batch.tag_mask = None

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]