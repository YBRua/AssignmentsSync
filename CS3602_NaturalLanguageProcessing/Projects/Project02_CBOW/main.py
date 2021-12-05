import random
import sys
import time

import numpy as np

from utils.vocab import Vocab
from word2vec import CBOW

# Check Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 6


def test1():
    random.seed(42)
    np.random.seed(42)

    vocab = Vocab(corpus="./data/debug.txt")
    cbow = CBOW(vocab, vector_dim=4)
    cbow.train(corpus="./data/debug.txt", window_size=3, train_epoch=10, learning_rate=1.0)

    print(cbow.most_similar_tokens("i", 5))
    print(cbow.most_similar_tokens("he", 5))
    print(cbow.most_similar_tokens("she", 5))

    # 注：如果实现正确，那么最终的loss将会停留在1.0左右，且'i','he','she'三者的相似性较高。


def test2():
    random.seed(42)
    np.random.seed(42)

    try:
        model = CBOW.load_model("ckpt")
    except FileNotFoundError:
        vocab = Vocab(corpus="./data/treebank.txt", max_vocab_size=-1)
        model = CBOW(vocab, vector_dim=12)

    start_time = time.time()
    model.train(corpus="./data/treebank.txt", window_size=4, train_epoch=10, learning_rate=1e-2, save_path="ckpt")
    end_time = time.time()

    print(f"Cost {(end_time - start_time) / 60:.1f} min")
    print(model.most_similar_tokens("i", 10))

    # 注：训练时间约1.5h，最终的loss将会降至7.0左右


def test3():
    from utils.similarity import get_test_data, evaluate_similarity

    model = CBOW.load_model("ckpt")
    spearman_result, pearson_result = evaluate_similarity(model, *get_test_data())
    print(f"spearman correlation: {spearman_result.correlation:.3f}")
    print(f"pearson correlation: {pearson_result[0]:.3f}")

    # 注：最终spearman相关系数将在0.3以上，pearson系数将在0.4以上


def main():
    test1()
    test2()
    test3()


if __name__ == '__main__':
    main()
