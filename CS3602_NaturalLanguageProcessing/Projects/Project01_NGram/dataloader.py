import os
import nltk
# from nltk.corpus import stopwords

from utils import SENTENSE_START, SENTENSE_END


def load_tokens(
        args,
        dataset='./train_set.txt',):
    with open(os.path.join(args.path, dataset)) as f:
        dataset = f.read(-1).lower()
    if args.tokenizer == 'NAIVE':
        tokens = dataset.split(' ')
    elif args.tokenizer == 'NLTK':
        nltk.download('punkt')
        tokens = nltk.word_tokenize(dataset)
    else:
        raise ValueError(f'Invalid tokenizer {args.tokenizer}')
    # if args.stopwords:
    #     nltk.download('stopwords')
    #     stopwords_en = set(stopwords.words('english'))
    #     tokens = [tok for tok in tokens if tok not in stopwords_en]
    tokens = [SENTENSE_START] + tokens + [SENTENSE_END]

    return tokens


def load_text(args, dataset='./dev_set.txt'):
    with open(os.path.join(args.path, dataset)) as f:
        datatext = f.read(-1).lower()
    return datatext
