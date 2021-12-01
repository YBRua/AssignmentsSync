import typing as t
import scipy.sparse as sparse
from collections import defaultdict as ddict


class Graph():
    def __init__(self,
                 adj_mat: sparse.coo_matrix,):
        self.coo = adj_mat
        self.out_degrees = self.coo.sum(axis=1).A1
        self.in_degrees = self.coo.sum(axis=0).A1
        self.neighbours: ddict[int, t.Set[int]] = ddict(set)
        self.in_neighbours: ddict[int, t.Set[int]] = ddict(set)
        self.out_neighbours: ddict[int, t.Set[int]] = ddict(set)
        self.edge_weights: t.Dict[t.Tuple[int, int], int] = {}
        for src, dst, weight in zip(self.coo.row, self.coo.col, self.coo.data):
            self.neighbours[src].add(dst)
            self.neighbours[dst].add(src)
            self.in_neighbours[dst].add(src)
            self.out_neighbours[src].add(dst)
            self.edge_weights[(src, dst)] = weight
        self.M = float(self.coo.sum())

    def set_node_to_comm(self, node2comm: t.List[int]):
        self.node2comm = node2comm

    def get_in_degree(self, node: int) -> int:
        return self.in_degrees[node]

    def get_out_degree(self, node: int) -> int:
        return self.out_degrees[node]

    def get_neighbours(self, node: int) -> t.Set[int]:
        return self.neighbours[node]

    def get_in_neighbours(self, node: int) -> t.Set[int]:
        return self.in_neighbours[node]

    def get_out_neighbours(self, node: int) -> t.Set[int]:
        return self.out_neighbours[node]
