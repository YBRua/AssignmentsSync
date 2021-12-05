from utils.vocab import tokenizer


class Dataset:
    def __init__(self, corpus: str, window_size: int):
        self.corpus = corpus
        self.window_size = window_size

    def __iter__(self):
        with open(self.corpus, encoding="utf8") as f:
            for line in f:
                tokens = tokenizer(line)
                if len(tokens) <= 1:
                    continue
                for i, target in enumerate(tokens):
                    left_context = tokens[max(0, i - self.window_size): i]
                    right_context = tokens[i + 1: i + 1 + self.window_size]
                    context = left_context + right_context
                    yield context, target
