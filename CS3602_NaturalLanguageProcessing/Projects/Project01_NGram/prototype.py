# %%
from tqdm import tqdm
import math
from collections import Counter, defaultdict
from dataloader import load_tokens
from discount import GoodTuringDiscount
from utils import LOG10_NINF, SENTENSE_END, SENTENSE_START, UNKNOWN, is_sentence_delimiter
from nltk.corpus import words


class Atlas(defaultdict):
    def __init__(self):
        super().__init__(Atlas)


# %%
tokens = load_tokens()
vocab_count = Counter(tokens)
vocab_count = sorted(vocab_count.items(), key=lambda x: x[1], reverse=True)

atlas_entry = Atlas()


# %% unigram
def count_unigram(vocab_count, threshold):
    with open('./vocabulary/COCA_20000.txt', 'r') as f:
        wordlist20k = f.read(-1).split('\n')
    wordlist = set(words.words())
    wordlist = set(wordlist20k)
    uni_vocab = {}
    uni_vocab[(UNKNOWN,)] = 0
    oov = set([])
    for token, count in vocab_count:
        if token in wordlist or is_sentence_delimiter(token):
            uni_vocab[(token,)] = count
        else:
            uni_vocab[(UNKNOWN,)] = uni_vocab[(UNKNOWN,)] + count
            oov.add(token)

    return uni_vocab, oov


def count_ngram(tokens, order, oov):
    ngram_list = []  # dict: {ngram_tuple -> count}
    if order <= 1:
        raise ValueError(f'Order should be greater than 1, got {order}')
    for i in range(len(tokens)-order+1):
        history = tuple(
            [UNKNOWN if t in oov else t for t in tokens[i:i+order-1]])
        current = (
            UNKNOWN if tokens[i+order-1] in oov else tokens[i+order-1],)
        ngram = history + current
        ngram_list.append(ngram)

    return ngram_list


threshold = 100
uni_vocab, oov = count_unigram(vocab_count, threshold)

# %%
bi_vocab = Counter(count_ngram(tokens, 2, oov))
tri_vocab = Counter(count_ngram(tokens, 3, oov))

# %%
vocabs = [uni_vocab, bi_vocab, tri_vocab]
cocs = [Counter(vocabs[i].values()) for i in range(len(vocabs))]
discounts = []

for idx in range(3):
    order = idx + 1
    gtmax = 1 if order == 1 else 7
    gtmin = 1

    discounts.append(GoodTuringDiscount(cocs[idx], gtmax, gtmin))


# %%
probs = []
no_discount_ngrams = []
for idx in range(3):
    order = idx + 1
    current_probs = {}
    current_no_discount_ngrams = set([])
    for ngram, count in vocabs[idx].items():
        if order == 1:
            history_count = sum(vocabs[0].values())
        else:
            history_count = vocabs[idx-1][ngram[:-1]]

        prob = discounts[idx](count) / history_count

        if prob > 0 and ngram != (SENTENSE_START,):
            log_prob = math.log10(prob)
            current_probs[ngram] = log_prob
            current_no_discount_ngrams.add(ngram)
        else:
            current_probs[ngram] = LOG10_NINF
            if ngram == (SENTENSE_START,):
                current_no_discount_ngrams.add(ngram)

        node = atlas_entry
        for word in ngram:
            node = node[word]

    if order == 1 and (SENTENSE_END,) not in current_no_discount_ngrams:
        current_no_discount_ngrams.add((SENTENSE_END,))
        current_probs[(SENTENSE_END,)] = LOG10_NINF

    probs.append(current_probs)
    no_discount_ngrams.append(current_no_discount_ngrams)

# %%
bows = []
word_vocab = uni_vocab.keys()
for idx in range(3):
    order = idx + 1
    if order == 3:
        bows.append({})
        continue
    # order < 3
    current_bows = {}
    for history, prob in tqdm(probs[idx].items()):
        numerator = 1
        denominator = 1
        curr_ngram_prob = probs[idx]
        higher_ngram_prob = probs[idx+1]

        node = atlas_entry
        # find all observed w[i] given w[0:i-i] in the search tree
        for history_word in history:
            node = node[history_word]

        for word in node.keys():
            ngram = history + (word,)
            if ngram in higher_ngram_prob:
                # all observed ngram w[0:i-1]'s
                numerator = numerator - 10 ** higher_ngram_prob[ngram]
                if len(ngram) > 1 and ngram[1:] in curr_ngram_prob:
                    # all observed ngram w[1:i]'s
                    denominator = denominator - \
                        10 ** curr_ngram_prob[ngram[1:]]
        if abs(numerator) < 1e-12:
            numerator = 0
        if abs(denominator) < 1e-12:
            denominator = 0

        if numerator == 0 and denominator == 0:
            current_bows[history] = 0
        elif numerator > 1e-12 and denominator == 0:
            if numerator == 1:
                current_bows[history] = LOG10_NINF
            else:
                log_margin = math.log10(1-numerator)
                for word in node.keys():
                    # reassign probability to make prob sum up to 1
                    # without using alpha (denominator = 0)
                    ngram = history + (word,)
                    higher_ngram_prob[ngram] -= log_margin
        elif numerator == 0:
            if denominator > 0:
                current_bows[history] = 0
            else:
                current_bows[history] = LOG10_NINF
        else:
            bow_log = math.log10(numerator) - math.log10(denominator)
            current_bows[history] = bow_log

    bows.append(current_bows)

    # if numerator == 0 and denominator == 0:
    #     # all possible ngrams have been observed
    #     current_bows[history] = 1
    # elif numerator > 1e-12 and denominator == 0:
    #     # all lower order ngrams have been observed
    #     assert numerator < 1, "1/0. Fucked up"
    #     log_margin = math.log10(1-numerator)
    #     for word in node.keys():
    #         # reassign probability to make prob sum up to 1
    #         # without using alpha (denominator = 0)
    #         ngram = history + (word,)
    #         higher_ngram_prob[ngram] -= log_margin
    # else:
    #     if numerator == 0:
    #         current_bows[history] = 1
    #     else:
    #         bow_log = math.log10(numerator) - math.log10(denominator)
    #         current_bows[history] = bow_log

# %%
with open('./lm.arpa', 'w') as lmfile:
    lmfile.write('\\data\\\n')
    for i in range(3):
        order = i + 1
        lmfile.write(f'ngram {order}={len(no_discount_ngrams[i])}\n')

    lmfile.write('\n')

    for i in range(3):
        order = i + 1
        lmfile.write(f'\\{order}-grams:\n')
        current_ngrams = no_discount_ngrams[i]
        current_probs = probs[i]
        current_bows = bows[i]
        for ngram in current_ngrams:
            text = ' '.join(ngram)
            log_prob = current_probs[ngram]
            if ngram in current_bows:
                log_bow = current_bows[ngram]
                lmfile.write(
                    '{:.4f}\t{:s}\t{:.4f}\n'.format(log_prob, text, log_bow))
            else:
                lmfile.write('{:.4f}\t{:s}\n'.format(log_prob, text))
        lmfile.write('\n')

    lmfile.write('\\end\\\n')

# %%
