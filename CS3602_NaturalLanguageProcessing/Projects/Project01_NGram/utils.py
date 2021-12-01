SENTENSE_START = '<s>'
SENTENSE_END = '</s>'
UNKNOWN = '<unk>'
LOG10_NINF = -99  # arpa style log10(-inf)

EPS = 1e-12


def is_sentence_delimiter(token):
    return token == SENTENSE_START or token == SENTENSE_END
