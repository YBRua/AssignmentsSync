import os
import nltk
import kenlm
from argparse import ArgumentParser
from nltk.corpus import words

from dataloader import load_text, load_tokens
from ngram import NGram


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-o', '--output', type=str,
        default='./lm.arpa',
        help='Output path for .arpa format language model.')
    parser.add_argument(
        '-p', '--path', type=str,
        default='./hw1_dataset',
        help='Path to train, dev and test datasets.')
    parser.add_argument(
        '-v', '--vocabulary',
        choices=['COCA20K', 'NLTK'],
        default='COCA20K',
        help='Vocabulary to be used.')
    parser.add_argument(
        '-t', '--tokenizer',
        choices=['NAIVE', 'NLTK'],
        default='NLTK',
        help='Tokenizing method to be used.')
    # parser.add_argument(
    #     '-s', '--stopwords', type=bool,
    #     default=False,
    #     help='Drop stop words if True')

    return parser.parse_args()


def get_vocab(args):
    if args.vocabulary == 'COCA20K':
        with open(os.path.join(args.path, 'COCA_20000.txt'), 'r') as f:
            wordlist = f.read(-1).split('\n')
        return wordlist
    elif args.vocabulary == 'NLTK':
        nltk.download('words')
        wordlist = set(words.words())
        return wordlist
    else:
        raise ValueError(f'Invalid vocab option {args.vocabulary}')


def main():
    args = parse_args()
    print(f'Loading vocabulary list {args.vocabulary}.')
    vocab = get_vocab(args)
    print('Loading training tokens.')
    train_tokens = load_tokens(args, './train_set.txt')
    ngram = NGram(vocab)
    print('Training.')
    ngram = ngram.train(train_tokens)
    print('Done training. Exporting lm as arpa.')
    ngram.dump_to_arpa(args.output)
    print(f'Output arpa to {args.output}')

    print('Running evaluations.')
    train_text = load_text(args, './train_set.txt')
    dev_text = load_text(args, './dev_set.txt')
    test_text = load_text(args, './test_set.txt')

    ngram_mdl = kenlm.Model(args.output)
    train_ppl = ngram_mdl.perplexity(train_text)
    dev_ppl = ngram_mdl.perplexity(dev_text)
    test_ppl = ngram_mdl.perplexity(test_text)

    print('Train perplexity: {:.4f}'.format(train_ppl))
    print('Dev perplexity: {:.4f}'.format(dev_ppl))
    print('Test perplexity: {:.4f}'.format(test_ppl))


if __name__ == '__main__':
    main()
