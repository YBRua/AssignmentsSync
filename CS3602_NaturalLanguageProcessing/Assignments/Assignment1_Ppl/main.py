from argparse import ArgumentParser
from arpa_reader import ArpaNGramReader, read_arpa


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--path',
        type=str,
        default='./models/cs3602_1.arpa',
        help='path to arpa language model')

    return parser.parse_args()


def main():
    args = parse_args()
    lines = read_arpa(args.path)
    ngram = ArpaNGramReader().parse_arpa_lines(lines)
    while True:
        test_sent = input('Input sentence. Type "exit!!" to exit.\n')
        if test_sent == 'exit!!':
            print('See you next time.')
            break
        else:
            ppl = ngram.perplexity(test_sent)
            print('Perplexity:', ppl)


if __name__ == '__main__':
    main()
