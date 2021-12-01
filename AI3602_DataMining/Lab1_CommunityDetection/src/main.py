import random
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from louvain import louvain
from utils import parse_args
from itertools import permutations


def main():
    random.seed("HK416A5")
    args = parse_args()
    EDGE_CSV_PATH = args.dataset_path
    OUTPUT_CSV_PATH = args.output
    edges = pd.read_csv(EDGE_CSV_PATH).to_numpy()
    graph_coo = sparse.coo_matrix(
        (np.ones(edges.shape[0]), (edges.T[0], edges.T[1])))
    graph_coo.sum_duplicates()

    global_node2comm = louvain(graph_coo)

    # reindex
    gt = pd.read_csv("./data/ground_truth.csv").to_numpy()
    best_acc = 0.0
    best_reindexer = (0,)
    for reindexer in permutations(range(5)):
        acc = 0.0
        for idx, lbl in gt:
            if reindexer[global_node2comm[idx]] == lbl:
                acc += 1
        acc /= len(gt)
        if acc > best_acc:
            best_acc = acc
            best_reindexer = reindexer

    print(f"Best ACC: {best_acc}.")

    for idx, cat in enumerate(global_node2comm):
        global_node2comm[idx] = best_reindexer[cat]

    with open(OUTPUT_CSV_PATH, 'w') as fo:
        fo.write('id, category\n')
        for id, cat in enumerate(global_node2comm):
            fo.write(f'{id}, {cat}\n')

    print(f"Result outputted to {OUTPUT_CSV_PATH}")
    print("See you next time.")


if __name__ == '__main__':
    main()
