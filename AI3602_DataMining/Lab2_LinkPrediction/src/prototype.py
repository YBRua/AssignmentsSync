# %%
import torch
import random
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.sparse as sparse
from model import SkipGram
from metric import calc_auc_srx_ybr
from sparse_graph import SparseGraph
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import List
from tqdm import tqdm

from walker import BiasedRandomWalker


def load_edges(path: str) -> np.ndarray:
    return pd.read_csv(path).to_numpy()


def build_graph_from_csv(path: str) -> SparseGraph:
    edges = load_edges(path).T
    srcs, dsts = edges[0], edges[1]
    coo = sparse.coo_matrix(
        (np.ones(edges.shape[1]), (srcs, dsts)))
    coo.sum_duplicates()
    return SparseGraph(coo)


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


# %%
class NegativeSamplingLoss(nn.Module):
    def __init__(self, k: int):
        self.k = k

    def __call__(self, zu: torch.Tensor, zv: torch.Tensor):
        pass


# %%
DATASET_PATH = './data/lab2_edge.csv'
graph = build_graph_from_csv(DATASET_PATH)
edges = load_edges(DATASET_PATH)


# %%
N_EPOCHS = 75
N_WALKS_PER_BATCH = 5
WALK_LENGTH = 15
N_NODES = graph.coo.shape[0]
IN_DIM = N_NODES
HIDDEN_DIM = 2048
EMBEDDING_DIM = 64
WINDOW_SIZE = 5
BATCH_SIZE = 128
N_NEG_SAMPLES = 12
EPSILON = 1e-7

DEVICE = 'cuda:0'

walker = BiasedRandomWalker(graph, 1, 1)
model = SkipGram(IN_DIM, EMBEDDING_DIM).to(DEVICE)
model.train()
optimizer = optim.SGD(model.parameters(), lr=10, momentum=0.9)

# %%
for e in range(N_EPOCHS):
    walk_data: List[List[int]] = []
    isolated_nodes = set([])
    nodes = list(range(N_NODES))
    random.shuffle(nodes)
    for node in nodes:
        if graph.get_degree(node) == 0:
            isolated_nodes.add(node)
            continue
        walk = walker(node, WALK_LENGTH)
        for centroid in range(WINDOW_SIZE, WALK_LENGTH - WINDOW_SIZE):
            walk_data.append(
                walk[centroid - WINDOW_SIZE: centroid + WINDOW_SIZE + 1])
    walk_data = torch.LongTensor(walk_data)

    data_loader = DataLoader(walk_data, batch_size=BATCH_SIZE, shuffle=True)

    tot_loss = 0
    t = tqdm(data_loader)
    model.train()
    for bidx, x in enumerate(t):
        current_bsize = x.shape[0]
        optimizer.zero_grad()
        x = x.to(DEVICE)
        preds = model.forward(x)
        cur_pred = preds[:, WINDOW_SIZE, :]
        ctx_pred = torch.cat(
            (preds[:, :WINDOW_SIZE, :], preds[:, WINDOW_SIZE + 1:, :]),
            axis=1)

        negs = torch.randint(
            N_NODES,
            (current_bsize, 2*WINDOW_SIZE, N_NEG_SAMPLES))
        negs = negs.reshape(current_bsize, -1).to(DEVICE)
        neg_pred = model.forward(negs)
        neg_pred = neg_pred.reshape(
            current_bsize, 2*WINDOW_SIZE, N_NEG_SAMPLES, EMBEDDING_DIM)
        pos_prob = torch.einsum(
            'ij, ikj -> ik', cur_pred, ctx_pred)
        neg_prob = torch.einsum(
            'ij, iklj -> ikl', cur_pred, neg_pred)
        pos_prob = torch.log(torch.sigmoid(pos_prob) + EPSILON)
        neg_prob = torch.mean(torch.log(torch.sigmoid(neg_prob) + EPSILON), axis=-1)
        loss = - torch.sum((pos_prob - neg_prob))
        loss /= current_bsize
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        t.set_description(f'Epoch: {e:2d} Loss: {tot_loss / (bidx+1):.4f}')
    model.eval()

    neg_tests = get_negative_tests(graph, 2000)
    pos_tests = torch.tensor(
        edges[random.choices(list(range(edges.shape[0])), k=2000)])
    neg_tests = neg_tests.to(DEVICE)
    pos_tests = pos_tests.to(DEVICE)

    auc = calc_auc_srx_ybr(model.emb.weight, edges, graph, 10000)

    print(f'Epoch: {e:2d} AUC: {auc:.4f}', file=sys.stderr)

# %%
