import random
import numpy as np
import torch
import torch.nn as nn
from sparse_graph import SparseGraph
from typing import Tuple, List


def get_negative_tests(G: SparseGraph, size: int):
    n_nodes = G.coo.shape[0]
    negatives = []
    for i in range(size):
        src = choose_a_node(n_nodes)
        neighbours = G.get_neighbours(src)
        dst = choose_a_node(n_nodes)
        while dst == src or dst in neighbours:
            dst = choose_a_node(n_nodes)
        negatives.append([src, dst])


def choose_a_node(high, low=0) -> int:
    return random.randint(low, high-1)


def calc_auc(D0: torch.LongTensor, D1: torch.LongTensor, model: nn.Module):
    model.eval()
    pred_D0 = model.forward(D0)  # B,2,Emb
    pred_D1 = model.forward(D1)  # B,2,Emb
    prob_D0 = torch.cosine_similarity(pred_D0[:, 0, :], pred_D0[:, 1, :])
    prob_D1 = torch.cosine_similarity(pred_D1[:, 0, :], pred_D1[:, 1, :])
    prob_D1_ext = prob_D1.repeat_interleave(prob_D0.shape[0])
    prob_D0_ext = prob_D0.repeat(prob_D1.shape[0])
    auc = torch.sum(prob_D0_ext < prob_D1_ext).float()\
        / prob_D0.shape[0] / prob_D1.shape[0]
    return auc

def calc_auc_srx_ybr(e, edges, G: SparseGraph, sample=1000):
    e = e.cpu()
    N_NODES = G.coo.shape[0]
    auc = []
    for _ in range(sample):
        u1 = random.choice(edges)
        while True:
            u0 = np.random.randint(N_NODES, size=[2])
            if u0[1] not in G.get_neighbours(u0[0]):
                break
        auc.append(
            (torch.cosine_similarity(e[u0[0]], e[u0[1]], 0)
            < torch.cosine_similarity(e[u1[0]], e[u1[1]], 0)))
    return (np.mean(auc))
