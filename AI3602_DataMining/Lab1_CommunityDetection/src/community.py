import typing as t
from graph import Graph


class Community():
    def __init__(self, graph: Graph):
        self.nodes: t.Set[int] = set([])
        self.G = graph
        self.out_deg: int = 0
        self.in_deg: int = 0

    def add_node(self, node: int):
        self.nodes.add(node)
        self.out_deg += self.G.out_degrees[node]
        self.in_deg += self.G.in_degrees[node]

    def remove_node(self, node: int):
        self.nodes.remove(node)
        self.out_deg -= self.G.out_degrees[node]
        self.in_deg -= self.G.in_degrees[node]

    def intra_comm_in_degree(self, node: int) -> float:
        in_deg = 0.
        in_neighbours = self.G.get_in_neighbours(node)
        for neighbour in in_neighbours:
            if neighbour in self.nodes:
                in_deg += self.G.edge_weights[(neighbour, node)]

        return in_deg

    def intra_comm_out_degree(self, node: int) -> float:
        out_deg = 0.
        out_neighbours = self.G.get_out_neighbours(node)
        for neighbour in out_neighbours:
            if neighbour in self.nodes:
                out_deg += self.G.edge_weights[(node, neighbour)]

        return out_deg
