import typing as t
import random
import sys
from community import Community
import scipy.sparse as sparse
from graph import Graph
from collections import Counter

from tqdm import tqdm


def merge_community(
        idx1: int, idx2: int,
        communities: t.List[Community],
        global_node2comm: t.List[int]):
    comm1 = communities[idx1]
    comm2 = communities[idx2]
    for node in comm2.nodes.copy():
        comm1.add_node(node)
        comm2.remove_node(node)
        global_node2comm[node] = idx1


def rebuild_metagraph_coo(old_graph: Graph) -> sparse.coo_matrix:
    n_nodes = old_graph.coo.shape[0]
    reindexer = graph_reindex(old_graph)
    new_src = []
    new_dst = []
    edge_weights = []
    for node in range(n_nodes):
        comm_src = old_graph.node2comm[node]
        comm_src_ridx = reindexer[comm_src]
        for neighbour in old_graph.get_out_neighbours(node):
            comm_dst = old_graph.node2comm[neighbour]
            comm_dst_ridx = reindexer[comm_dst]
            new_src.append(comm_src_ridx)
            new_dst.append(comm_dst_ridx)
            edge_weights.append(old_graph.edge_weights[(node, neighbour)])
    new_coo = sparse.coo_matrix((edge_weights, (new_src, new_dst)))
    new_coo.sum_duplicates()

    return new_coo


def graph_reindex(graph: Graph) -> t.Dict[int, int]:
    reindexer: t.Dict[int, int] = {}
    comms = set(graph.node2comm)
    for idx, comm in enumerate(comms):
        reindexer[comm] = idx

    return reindexer


def _nodewise_delta_q(node: int, community: Community):
    graph = community.G
    intra_in_deg = float(community.intra_comm_in_degree(node))
    intra_out_deg = float(community.intra_comm_out_degree(node))
    # intra_in_deg = 10
    # intra_out_deg = 10
    in_deg = float(graph.get_in_degree(node))
    out_deg = float(graph.get_out_degree(node))
    comm_out_deg = float(community.out_deg)
    comm_in_deg = float(community.in_deg)
    return (intra_in_deg + intra_out_deg) / graph.M -\
        (in_deg * comm_out_deg + out_deg * comm_in_deg) / (graph.M ** 2)


def commwise_delta_q(comm1: Community, comm2: Community):
    M = comm1.G.M
    dq_loss = (comm1.in_deg * comm2.out_deg + comm1.out_deg * comm2.in_deg)
    dq_loss = dq_loss / M / M
    dq_gain = 0
    for node in comm2.nodes:
        dq_gain += comm1.intra_comm_out_degree(node)
        dq_gain += comm1.intra_comm_in_degree(node)

    dq_gain = dq_gain / M

    return dq_gain - dq_loss


def _init_communities(
        metagraph: Graph) -> t.Tuple[t.List[int], t.List[Community]]:
    node2comm: t.List[int] = []
    communities: t.List[Community] = []
    n_nodes = metagraph.coo.shape[0]
    for node in range(n_nodes):
        node2comm.append(node)
        community = Community(metagraph)
        community.add_node(node)
        communities.append(community)

    return node2comm, communities


def _louvain_phase_1(metagraph: Graph, communities: t.List[Community]):
    n_nodes = metagraph.coo.shape[0]
    num_iter = 0
    modularity = 0
    while True:
        num_iter += 1
        print("Iteration", num_iter, file=sys.stderr)
        changed = False
        num_nodes_changed = 0
        random_iterator = list(range(n_nodes))
        random.shuffle(random_iterator)
        for node in tqdm(random_iterator):
            old_comm_idx = metagraph.node2comm[node]
            old_comm = communities[old_comm_idx]
            old_comm.remove_node(node)

            best_comm_idx = old_comm_idx
            best_comm = old_comm

            metagraph.node2comm[node] = -1
            best_modularity = 0

            delta_q_rm = _nodewise_delta_q(node, old_comm)

            for neighbour in metagraph.get_neighbours(node):
                new_comm_idx = metagraph.node2comm[neighbour]
                if new_comm_idx == old_comm_idx:
                    continue
                new_comm = communities[new_comm_idx]
                delta_q_mv = _nodewise_delta_q(node, new_comm)
                d_q = delta_q_mv - delta_q_rm
                if d_q > best_modularity:
                    changed = True
                    best_modularity = d_q
                    best_comm = new_comm
                    best_comm_idx = new_comm_idx

            metagraph.node2comm[node] = best_comm_idx
            modularity += best_modularity

            best_comm.add_node(node)
            if old_comm_idx != best_comm_idx:
                num_nodes_changed += 1

        print(
            f'Modularity: {modularity:.4f} | '
            + f'Changed: {num_nodes_changed} | '
            + f'Communities: {len(set(metagraph.node2comm))}', file=sys.stderr)
        if not changed:
            break

    return


def _louvain_phase_2(
        metagraph: Graph,
        GRAPH: Graph,
        global_node2comm: t.List[int]) -> sparse.coo_matrix:
    N_NODES: int = GRAPH.coo.shape[0]
    reindexer = graph_reindex(metagraph)
    for node in range(N_NODES):
        metanode = global_node2comm[node]  # old metanode of current node
        new_metanode = metagraph.node2comm[metanode]  # new metanode
        new_metanode_ridx = reindexer[new_metanode]  # reindexed metanode
        global_node2comm[node] = new_metanode_ridx
    GRAPH.set_node_to_comm(global_node2comm)

    # construct new metagraph
    new_coo = rebuild_metagraph_coo(metagraph)

    return new_coo


def finalizing(
    global_node2comm: t.List[int],
    GRAPH: Graph,
    n_clusters: int
):
    N_NODES = GRAPH.coo.shape[0]
    community_counter = Counter(global_node2comm)

    # restore communities so they include nodes in original graph
    communities: t.List[Community] = []
    for i in range(len(community_counter)):
        communities.append(Community(GRAPH))
    for node in range(N_NODES):
        comm = global_node2comm[node]
        communities[comm].add_node(node)

    # merge nodes until the we reach the desired number of clusters
    while len(community_counter) > n_clusters:
        sorted_counter = sorted(
            community_counter.items(), key=lambda x: x[1])
        candidates = sorted_counter[:10]
        best_dq = -100
        best_pair: t.Tuple[int, int] = (-1, -1)
        for idx, _ in candidates:
            for idx2, _ in candidates:
                if idx == idx2:
                    continue
                comm = communities[idx]
                comm2 = communities[idx2]
                dq = commwise_delta_q(comm, comm2)
                if dq > best_dq:
                    best_dq = dq
                    best_pair = (idx, idx2)
        if best_pair[0] == -1 or best_pair[1] == -1:
            print('Early Stopping!')
            break
        merge_community(
            best_pair[0], best_pair[1], communities, global_node2comm)
        community_counter = Counter(global_node2comm)
    return


def louvain(graph_coo: sparse.coo_matrix):
    # get basic data of graph
    N_NODES = graph_coo.shape[0]

    # init global node2comm lookup table
    global_node2comm: t.List[int] = []
    for node in range(N_NODES):
        global_node2comm.append(node)

    # init graph
    GRAPH = Graph(graph_coo)

    # init meta graph
    # each node is an independent community at the beginning
    metagraph = Graph(graph_coo)
    node2comm, communities = _init_communities(metagraph)
    metagraph.set_node_to_comm(node2comm)

    # the main louvain algorithm
    while True:
        # phase 1
        _louvain_phase_1(metagraph, communities)

        # phase 2
        new_coo = _louvain_phase_2(
            metagraph,
            GRAPH,
            global_node2comm
        )

        # break if no change in structure
        if graph_coo.shape == new_coo.shape:
            if (graph_coo - new_coo).sum() == 0:
                print('No change in metagraph structure. Break.')
                break

        # update new meta graph
        graph_coo = new_coo
        metagraph = Graph(graph_coo)
        node2comm: t.List[int] = []
        communities: t.List[Community] = []

        # update communitites
        # new communitites only contain metanodes
        # global_node2comm maintains original nodes
        node2comm, communities = _init_communities(metagraph)

        metagraph.set_node_to_comm(node2comm)

    print("Main algorithm terminated. Running finalizing rounds.")
    # finalizing, make the number of clusters correct
    finalizing(global_node2comm, GRAPH, 5)

    # reindex
    reindexer = graph_reindex(GRAPH)
    for id, cat in enumerate(global_node2comm):
        global_node2comm[id] = reindexer[cat]

    print("Almost done.")
    return global_node2comm
