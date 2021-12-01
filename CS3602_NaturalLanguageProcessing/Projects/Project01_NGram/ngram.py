import math
import utils
from discount import GoodTuringDiscount
from collections import Counter, defaultdict
from utils import LOG10_NINF, SENTENSE_END, SENTENSE_START, UNKNOWN


class Atlas(defaultdict):
    def __init__(self):
        super().__init__(Atlas)


class NGram():
    def __init__(self, vocabulary, order=3, eps=1e-10):
        self.order = order
        self.eps = eps
        self.vocabulary = vocabulary
        self.oov = set([])
        self.ngram_vocabs = {}  # order -> dict{ngram -> count}
        self.cocs = {}  # order -> dict_coc
        self.discounts = {}  # order -> discount function
        self.probs = {}  # order -> dict{ngram -> probs}
        self.bows = {}  # order -> dict{ngram -> bows}
        self.good_ngrams = {}  # order -> set of ngrams with available probs
        self.atlas = Atlas()

    def _count_ngram(self, tokens, order):
        """Counts order-n ngrams in tokens.
        Results are stored in self.ngram_vocabs,
        where key = order and value = Counter
        """
        if order == 1:
            uni_vocab = {}
            uni_vocab[(UNKNOWN,)] = 0
            vocab_counter = Counter(tokens)
            for tok, count in vocab_counter.items():
                if tok in self.vocabulary or utils.is_sentence_delimiter(tok):
                    uni_vocab[(tok,)] = count
                else:
                    uni_vocab[(UNKNOWN,)] = uni_vocab[(UNKNOWN,)] + count
                    self.oov.add(tok)
            self.ngram_vocabs[order] = uni_vocab

            # add current unigram to atlas
            for unigram in self.ngram_vocabs[order]:
                node = self.atlas
                for word in unigram:
                    node = node[word]
        else:
            ngrams = []
            for i in range(len(tokens)-order+1):
                history = tuple([
                    UNKNOWN if t in self.oov else t
                    for t in tokens[i:i+order-1]])
                current = (
                    UNKNOWN
                    if tokens[i+order-1] in self.oov else tokens[i+order-1],)
                ngram = history + current

                # add current ngram to atlas
                node = self.atlas
                for word in ngram:
                    node = node[word]

                ngrams.append(ngram)
            self.ngram_vocabs[order] = Counter(ngrams)
        return self

    def _set_discount(self, order, discounts):
        """Initialize GT discount for each order-n ngram."""
        self.discounts[order] = discounts
        return self

    def _compute_probs(self):
        """Compute probabilities for each ngram"""
        total = sum([len(self.ngram_vocabs[i]) for i in self.ngram_vocabs])
        current = 0
        for idx in range(self.order):
            order_ = idx + 1
            current_probs = {}
            no_discount_ngrams = set([])
            current_vocabs = self.ngram_vocabs[order_]
            for ngram, count in current_vocabs.items():
                current = current + 1
                if current % 1000 == 0:
                    print(
                        '  Computing prob. {:5d}k/{:5d}k'.format(
                            int(current/1000), int(total/1000)),
                        end='\r', flush=True)
                if order_ == 1:
                    # for unigram, history_count = N
                    history_count = sum(current_vocabs.values())
                else:
                    history_vocabs = self.ngram_vocabs[order_-1]
                    history_count = history_vocabs[ngram[:-1]]
                prob = self.discounts[order_](count) / history_count

                if prob > 0 and ngram != (SENTENSE_START,):
                    log_prob = math.log10(prob)
                    current_probs[ngram] = log_prob
                    no_discount_ngrams.add(ngram)
                else:
                    current_probs[ngram] = LOG10_NINF
                    if ngram == (SENTENSE_START,):
                        # add <s> to dict, as a special case
                        no_discount_ngrams.add(ngram)

            if order_ == 1 and (SENTENSE_END,) not in no_discount_ngrams:
                # add </s> to dict, in case it is not already included
                no_discount_ngrams.add((SENTENSE_END,))
                current_probs[(SENTENSE_END,)] = LOG10_NINF

            self.probs[order_] = current_probs
            self.good_ngrams[order_] = no_discount_ngrams

    def _compute_bows(self):
        """Compute backoff weights"""
        current = 0
        total = sum([len(self.probs[i]) for i in range(1, self.order)])
        for idx in range(self.order):
            order_ = idx + 1
            if order_ == self.order:
                self.bows[order_] = {}
                continue
            current_bows = {}
            for history, prob in self.probs[order_].items():
                current = current + 1
                if current % 1000 == 0:
                    print(
                        '  Computing bows. {:5d}k/{:5d}k'.format(
                            int(current/1000), int(total/1000)),
                        end='\r', flush=True)
                numerator = 1
                denominator = 1
                curr_ngram_prob = self.probs[order_]
                high_ngram_prob = self.probs[order_+1]

                node = self.atlas
                # find all observed w[i] given w[0:i-i] in the atlas
                for history_word in history:
                    node = node[history_word]

                for word in node.keys():
                    ngram = history + (word,)
                    if ngram in high_ngram_prob:
                        # all observed ngram given a history
                        numerator = numerator - 10 ** high_ngram_prob[ngram]
                        if len(ngram) > 1 and ngram[1:] in curr_ngram_prob:
                            # all observed low-order ngram
                            denominator = denominator - \
                                10 ** curr_ngram_prob[ngram[1:]]

                if abs(numerator) < self.eps:
                    numerator = 0
                if abs(denominator) < self.eps:
                    denominator = 0

                if numerator == 0 and denominator == 0:
                    current_bows[history] = 0
                elif numerator > 0 and denominator == 0:
                    if numerator == 1:
                        # problematic
                        current_bows[history] = LOG10_NINF
                    else:
                        log_margin = math.log10(1-numerator)
                        for word in node.keys():
                            # reassign probability to make prob sum up to 1
                            # without using alpha (denominator = 0)
                            ngram = history + (word,)
                            high_ngram_prob[ngram] -= log_margin
                elif numerator == 0:
                    if denominator > 0:
                        current_bows[history] = 0
                    else:
                        # problematic
                        current_bows[history] = LOG10_NINF
                else:
                    log_bow = math.log10(numerator) - math.log10(denominator)
                    current_bows[history] = log_bow

            self.bows[order_] = current_bows

    def train(self, tokens):
        print('  Counting NGrams.')
        for i in range(self.order):
            order_ = i + 1
            print(f'    Order {order_}')
            self._count_ngram(tokens, order_)

        for order, vocabs in self.ngram_vocabs.items():
            self.cocs[order] = Counter(vocabs.values())
        # currently only support GT Discount
        for i in range(self.order):
            order_ = i + 1
            gtmax = 1 if order_ == 1 else 7
            gtmin = 1
            self.discounts[order_] = GoodTuringDiscount(
                self.cocs[order_], gtmax, gtmin)
        print('  Computing prob.', end='\r')
        self._compute_probs()
        print()
        print('  Computing bows.', end='\r')
        self._compute_bows()
        print()

        return self

    def dump_to_arpa(self, fname):
        with open(fname, 'w') as lmfile:
            # write metadata
            lmfile.write('\\data\\\n')
            for i in range(self.order):
                o = i + 1
                lmfile.write(f'ngram {o}={len(self.good_ngrams[o])}\n')
            lmfile.write('\n')

            for i in range(self.order):
                o = i + 1
                lmfile.write(f'\\{o}-grams:\n')
                current_ngrams = self.good_ngrams[o]
                current_probs = self.probs[o]
                current_bows = self.bows[o]
                for ngram in current_ngrams:
                    text = ' '.join(ngram)
                    log_prob = current_probs[ngram]
                    if ngram in current_bows:
                        log_bow = current_bows[ngram]
                        lmfile.write(
                            '{:.4f}\t{:s}\t{:.4f}\n'.format(
                                log_prob, text, log_bow))
                    else:
                        lmfile.write('{:.4f}\t{:s}\n'.format(log_prob, text))
                lmfile.write('\n')

            lmfile.write('\\end\\\n')
