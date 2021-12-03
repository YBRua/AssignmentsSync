import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from metric import calc_auc, get_negative_tests

from model import SkipGram
from utils import load_edges, parse_args
from sparse_graph import SparseGraph
from walker import BiasedRandomWalker, RandomWalker
from loss import NegativeSamplingLoss


def train(
        args,
        edges: np.ndarray,
        graph: SparseGraph,
        model: nn.Module,
        optimizer: optim.Optimizer,
        walker: RandomWalker,
        loss_metric: NegativeSamplingLoss):
    # training configs
    N_NODES = graph.coo.shape[0]
    N_EPOCHS = args.epoch
    BATCH_SIZE = args.batchsize
    DEVICE = args.device

    # hyper params
    WALK_LENGTH = 15
    WINDOW_SIZE = 5

    for e in range(N_EPOCHS):
        trajectory: List[List[int]] = []
        isolated_nodes = set([])
        nodes = list(range(N_NODES))
        random.shuffle(nodes)
        for node in nodes:
            if graph.get_degree(node) == 0:
                isolated_nodes.add(node)
                continue
            walk = walker(node, WALK_LENGTH)
            for cent in range(WINDOW_SIZE, WALK_LENGTH - WINDOW_SIZE):
                trajectory.append(
                    walk[cent - WINDOW_SIZE: cent + WINDOW_SIZE + 1])
        trajectory = torch.LongTensor(trajectory)

        data_loader = DataLoader(
            trajectory, batch_size=BATCH_SIZE, shuffle=True)

        tot_loss = 0
        t = tqdm(data_loader)
        model.train()
        for bidx, x in enumerate(t):
            current_bsize = x.shape[0]
            optimizer.zero_grad()
            x = x.to(DEVICE)
            preds = model.forward(x)
            loss = loss_metric(model, preds, current_bsize)
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

        auc = calc_auc(neg_tests, pos_tests, model)

        print(f'Epoch: {e:2d} AUC: {auc:.4f}', file=sys.stderr)

    return model


def inference(model: nn.Module, tests:)


def main():
    args = parse_args()
    # training configs

    N_NEG_SAMPLES = args.neg_samples
    DEVICE = args.device

    # I/O configs
    EDGE_PATH = args.dataset_path
    TEST_PATH = args.testset_path
    OUTPUT_PATH = args.file_output
    SAVE_PATH = args.model_save
    PRETRAINED_MODEL = args.pretrained_path

    # build sparse graph
    edges = load_edges(EDGE_PATH)
    graph = SparseGraph.build_graph_from_csv(EDGE_PATH)
    N_NODES = graph.coo.shape[0]
    IN_DIM = N_NODES

    # hyper params
    EMBEDDING_DIM = 64
    LR = 10  # learning rate
    MMT = 0.9  # momentum

    # hyper params for biased random walker
    RETURN_PARAM = 1
    IO_PARAM = 1

    walker = BiasedRandomWalker(graph, RETURN_PARAM, IO_PARAM)
    model = SkipGram(IN_DIM, EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MMT)
    loss_metric = NegativeSamplingLoss(N_NODES, N_NEG_SAMPLES)

    if PRETRAINED_MODEL == '':
        # training mode
        model.train()
        model = train(
            args, edges, graph, model, optimizer, walker, loss_metric)
        torch.save(model.state_dict(), SAVE_PATH)
        model.eval()
    else:
        model.load_state_dict(torch.load(PRETRAINED_MODEL))
        model.eval()
        # inference mode
