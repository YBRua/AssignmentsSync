import os
import re
from collections import Counter
from typing import Dict, List, Tuple

_UNK = "<unk>"


def tokenizer(line: str) -> List[str]:
    line = line.lower()
    tokens = list(filter(lambda x: len(x) > 0, re.split(r"[^\w]", line)))
    return tokens


class Vocab:
    VOCAB_FILE = "vocab.txt"
    _token_to_idx: Dict[str, int] = {}
    token_freq: List[Tuple[str, int]] = []

    def __init__(self, corpus: str = None, max_vocab_size: int = -1):
        """
        :param corpus:  Corpus file
        :param max_vocab_size: Maximum number of words, -1 indicates unlimited
        """
        if corpus is not None:
            self.build_vocab(corpus, max_vocab_size)

    def build_vocab(self, corpus: str, max_vocab_size: int = -1):
        """ Count word frequency and order it from highest to lowest """
        counter = Counter()
        with open(corpus, encoding="utf8") as f:
            for line in f:
                tokens = tokenizer(line)
                counter.update(tokens)

        print(f"Token number: {sum(counter.values())}")

        token_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        if max_vocab_size > 0:
            token_freq = token_freq[:max_vocab_size - 1]
        self.token_freq = token_freq + [(_UNK, token_freq[-1][1])]

        print(f"Vocab size: {len(self.token_freq)}")

        for i, (token, _freq) in enumerate(self.token_freq):
            self._token_to_idx[token] = i

    def __len__(self):
        return len(self.token_freq)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_idx

    def token_to_idx(self, token: str, warn=False) -> int:
        """ Map the token to index """
        if token not in self._token_to_idx:
            if warn:
                print(f"'{token}' not in vocab")
            token = _UNK
        return self._token_to_idx[token]

    def idx_to_token(self, idx: int) -> str:
        """ Map the index to token """
        assert 0 <= idx < len(self)
        return self.token_freq[idx][0]

    def save_vocab(self, path: str):
        with open(os.path.join(path, self.VOCAB_FILE), "w", encoding="utf8") as f:
            lines = [f"{token} {freq}" for token, freq in self.token_freq]
            f.write("\n".join(lines))

    @classmethod
    def load_vocab(cls, path: str):
        vocab = cls()

        with open(os.path.join(path, cls.VOCAB_FILE), encoding="utf8") as f:
            lines = f.read().split("\n")

        for i, line in enumerate(lines):
            token, freq = line.split()
            vocab.token_freq.append((token, int(freq)))
            vocab._token_to_idx[token] = i

        return vocab
