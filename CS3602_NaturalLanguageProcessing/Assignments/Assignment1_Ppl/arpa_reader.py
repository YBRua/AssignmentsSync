import re
import numpy as np


def read_arpa(path='./models/cs3602_1.arpa'):
    with open(path, 'r') as lmfile:
        lines = [ln.strip() for ln in lmfile.readlines()]
    return lines


class ArpaNGramReader():
    def __init__(self, order=3):
        self.ngram_metadata = {}  # \data\ field of .arpa file
        self.ngrams = {}  # ngram tuple -> log_prob
        self.bows = {}  # ngram tuple -> log_bow
        self.order = order
        self.LOG10_NINF = -99

    def parse_arpa_lines(self, lines):
        for line_pointer in range(len(lines)):
            if lines[line_pointer] == '':
                continue
            elif (
                    lines[line_pointer] == '\\data\\'
                    or lines[line_pointer] == '\\end\\'):
                continue
            elif lines[line_pointer].startswith('ngram'):
                # match metadata
                metadata = re.findall(r'\d+', lines[line_pointer])
                metadata = list(map(int, metadata))
                order, n_entries = metadata
                self.ngram_metadata[order] = n_entries
            elif lines[line_pointer].endswith('-grams:'):
                # match n-gram: section header
                order = re.findall(r'\d+', lines[line_pointer])
                order = int(order[0])
            else:
                # match n-gram entries
                entry = lines[line_pointer].split('\t')
                if len(entry) == 3:
                    # entry with backoff weight:
                    log_prob, ngram, bow = entry
                elif len(entry) == 2:
                    log_prob, ngram = entry
                    bow = 0.0
                else:
                    raise ValueError(f'Buggy entry in arpa: {entry}')
                ngram = tuple(ngram.split(' '))
                self.ngrams[ngram] = float(log_prob)
                self.bows[ngram] = float(bow)
        return self

    def compute_log_probability(self, words):
        if len(words) == 1:
            if words in self.ngrams:
                if words == ('<s>',):
                    return 0.0
                else:
                    return self.ngrams[words]
            else:
                return self.LOG10_NINF
        if words in self.ngrams:
            # no discounting
            return self.ngrams[words]
        elif words[:-1] in self.bows:
            log_bow = self.bows[words[:-1]]
            log_lower_prob = self.compute_log_probability(words[1:])
            return log_bow + log_lower_prob
        else:
            return self.compute_log_probability(words[1:])

    def perplexity(self, data):
        log_probs = []
        words = self._split_string_to_tuples(data)
        words = self._append_start_and_end(words)
        for i in range(len(words)-2):
            log_prob = self.compute_log_probability(words[i:i+self.order])
            log_probs.append(log_prob)
        if -99 in log_probs:
            print("log(0) encountered in log_probs. Ppl will be infty.")
        log_probs = np.array(log_probs)
        exponential = - np.mean(log_probs)
        return 10 ** exponential

    def _split_string_to_tuples(self, data):
        return tuple(data)

    def _append_start_and_end(self, words):
        return ('<s>',) * (self.order-1) + words + ('</s>',)
