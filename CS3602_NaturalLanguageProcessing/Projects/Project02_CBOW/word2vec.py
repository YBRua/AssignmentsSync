import os
import pickle
import time
from os.path import join
from typing import List

import numpy as np

from utils.dataset import Dataset
from utils.vocab import Vocab


def one_hot(dim: int, idx: int):
    """ Get one-hot vector """
    v = np.zeros(dim)
    v[idx] = 1
    return v


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class CBOW:
    def __init__(self, vocab: Vocab, vector_dim: int):
        self.vocab = vocab
        self.vector_dim = vector_dim

        self.W1 = np.random.uniform(-1, 1, (len(self.vocab), self.vector_dim))  # V x N
        self.W2 = np.random.uniform(-1, 1, (self.vector_dim, len(self.vocab)))  # N x V

    def train(self, corpus: str, window_size: int, train_epoch: int, learning_rate: float, save_path: str = None):
        dataset = Dataset(corpus, window_size)

        for epoch in range(1, train_epoch + 1):
            start_time = time.time()
            avg_loss = self.train_one_epoch(dataset, learning_rate)
            end_time = time.time()
            print(f"Epoch {epoch}, loss: {avg_loss:.2f}. Cost {(end_time - start_time) / 60:.1f} min")
            if save_path is not None:
                self.save_model(save_path)

    def train_one_epoch(self, dataset: Dataset, learning_rate: float):
        steps, total_loss = 0, 0.0

        for steps, sample in enumerate(iter(dataset), start=1):
            context_tokens, target_token = sample
            loss = self.train_one_step(context_tokens, target_token, learning_rate)
            total_loss += loss

            if steps % 10000 == 0:
                print(f"Step: {steps}. Avg. loss: {total_loss / steps: .2f}")

        return total_loss / steps

    def train_one_step(self, context_tokens: List[str], target_token: str, learning_rate: float) -> float:
        """
        Predict the probability of the target token given context tokens.

        :param context_tokens:  List of tokens around the target token
        :param target_token:    Target (center) token
        :param learning_rate:   Learning rate of each step
        :return:    loss of the target token
        """

        # ==== TODO: Construct one-hot vectors ====

        # ==== TODO: Forward step ====

        # ==== TODO: Calculate loss ====

        # ==== TODO: Update parameters ====

        return loss

    def similarity(self, token1: str, token2: str):
        """ Calculate cosine similarity of token1 and token2 """
        v1 = self.W1[self.vocab.token_to_idx(token1)]
        v2 = self.W1[self.vocab.token_to_idx(token2)]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.dot(v1, v2)

    def most_similar_tokens(self, token: str, n: int):
        """ Find the n words most similar to the given token """
        norm_W1 = self.W1 / np.linalg.norm(self.W1, axis=1, keepdims=True)

        idx = self.vocab.token_to_idx(token, warn=True)
        v = norm_W1[idx]

        cosine_similarity = np.dot(norm_W1, v)
        nbest_idx = np.argsort(cosine_similarity)[-n:][::-1]

        results = []
        for idx in nbest_idx:
            _token = self.vocab.idx_to_token(idx)
            results.append((_token, cosine_similarity[idx]))

        return results

    def save_model(self, path: str):
        """ Save model and vocabulary to `path` """
        os.makedirs(path, exist_ok=True)
        self.vocab.save_vocab(path)

        with open(join(path, "wv.pkl"), "wb") as f:
            param = {"W1": self.W1, "W2": self.W2}
            pickle.dump(param, f)

        print(f"Save model to {path}")

    @classmethod
    def load_model(cls, path: str):
        """ Load model and vocabulary from `path` """
        vocab = Vocab.load_vocab(path)

        with open(join(path, "wv.pkl"), "rb") as f:
            param = pickle.load(f)

        W1, W2 = param["W1"], param["W2"]
        model = cls(vocab, W1.shape[1])
        model.W1, model.W2 = W1, W2

        print(f"Load model from {path}")
        return model
