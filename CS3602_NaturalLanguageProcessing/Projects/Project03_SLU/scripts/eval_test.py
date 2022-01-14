#coding=utf8
import os
import gc
import sys
import time
import json
import torch

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from argparse import ArgumentParser

from utils.initialization import set_torch_device
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from utils.ontology import Ontology
from model.slu_ontology_guided_tagging import OntologyGuidedTagging


def parse_args_for_eval():
    parser = ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='root of data')
    parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size. Only support 1')
    parser.add_argument('--model_save', default='model.bin')

    # parser.add_argument('--pinyin', action='store_true', help='whether to enable pinyin fallback')

    parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='root of data')
    parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    parser.add_argument('--num_layer', default=2, type=int, help='number of layer')

    return parser.parse_args()


def decode_test(args, model: OntologyGuidedTagging, dataset, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred, _, _ = model.decode(Example.label_vocab, current_batch)
            # print("pred", pred)
            predictions.extend(pred)
            # print('predictions', predictions)
    torch.cuda.empty_cache()
    gc.collect()
    return predictions


if __name__ == '__main__':
    args = parse_args_for_eval()
    device = set_torch_device(args.device)
    print("Initialization finished ...")

    train_path = os.path.join(args.dataroot, 'train.json')
    dev_path = os.path.join(args.dataroot, 'development.json')
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    output_path = os.path.join(args.dataroot, 'test.json')
    ontology_path = os.path.join(args.dataroot, 'ontology.json')
    model_path = args.model_save

    if args.batch_size != 1:
        raise ValueError(f"Only support batchsize = 1, got {args.batch_size}")

    Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)

    test_dataset = Example.load_dataset(test_path)

    print('Enabling pinyin-calibrated ontology projection')

    args.vocab_size = Example.word_vocab.vocab_size
    args.pad_idx = Example.word_vocab[PAD]
    args.num_tags = Example.label_vocab.num_tags
    args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

    ontology = Ontology(args.dataroot, True)

    model = OntologyGuidedTagging(args, ontology).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])

    start_time = time.time()
    predictions = decode_test(args, model, test_dataset, device)
    # print(predictions)

    with open(test_path, 'r', encoding='utf-8') as f:
        tests = json.load(f)

    idx = 0
    for sample in tests:
        for utt in sample:
            utt['pred'] = [pred.split('-') for pred in predictions[idx]]
            idx += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tests, f, ensure_ascii=False)
