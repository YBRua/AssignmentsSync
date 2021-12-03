import torch
import random
import torch.nn as nn
from sparse_graph import SparseGraph


def get_negative_tests(G: SparseGraph, size: int) -> torch.Tensor:
    n_nodes = G.coo.shape[0]
    negatives = []
    for i in range(size):
        src = choose_a_node(n_nodes)
        neighbours = G.get_neighbours(src)
        dst = choose_a_node(n_nodes)
        while dst == src or dst in neighbours:
            dst = choose_a_node(n_nodes)
        negatives.append([src, dst])

    return torch.tensor(negatives)


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
