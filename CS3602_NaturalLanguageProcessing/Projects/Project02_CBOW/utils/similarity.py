import numpy as np
from scipy.stats import spearmanr, pearsonr

from word2vec import CBOW


def get_test_data():
    with open("data/test.txt") as f:
        lines = f.read().split("\n")
    X = [line.split(" ") for line in lines if len(line) > 0]
    y = list(reversed([i / len(X) for i in range(len(X))]))
    return X, y


def evaluate_similarity(model: CBOW, X, y):
    """
    Calculate Spearman and pearson correlation between cosine similarity of the model
    and human rated similarity of word pairs

    :param model:   Trained word2vec
    :param X:       Word pairs, <word1, word2>
    :param y:       Human ratings
    :return:        Spearman and pearson correlation
    """
    index1 = []
    index2 = []

    for i, (word1, word2) in enumerate(X):
        index1.append(model.vocab.token_to_idx(word1, warn=False))
        index2.append(model.vocab.token_to_idx(word2, warn=False))

    vector = model.W1 / np.linalg.norm(model.W1, axis=1, keepdims=True)
    A = vector[index1]
    B = vector[index2]
    scores = (A * B).sum(axis=1)

    return spearmanr(scores, y), pearsonr(scores, y)
