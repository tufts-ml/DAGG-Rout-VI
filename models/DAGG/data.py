import pickle
import torch
from torch.utils.data import Dataset
import networkx as nx
from datasets.preprocess import get_bfs_seq
from models.DAGG.helper import graph_to_matrix, get_attributes_len_for_graph_rnn


class Graph_Adj_Matrix_from_file(Dataset):
    """
    Dataset for reading graphs from files and returning adjacency like matrices
    max_prev_node has higher precedence than max_head_and_tail i.e
    :param args: Args object
    :param graph_list: List of graph indices to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    :random_bfs: Whether or not to do random_bfs
    """

    def __init__(self, args, graph_list, feature_map, random_bfs=False):
        # Path to folder containing dataset
        print('Start producing adj matrix')
        self.dataset_path = args.current_processed_dataset_path
        self.graph_list = graph_list
        self.feature_map = feature_map
        # No. of previous nodes to consider for edge prediction
        self.max_prev_node = args.max_prev_node
        # Head and tail of adjacency vector to consider for edge prediction
        self.max_head_and_tail = args.max_head_and_tail
        self.random_bfs = random_bfs

        if self.max_prev_node is None and self.max_head_and_tail is None:
            print('Please provide max_prev_node or max_head_and_tail')
            exit()

        self.max_nodes = feature_map['max_nodes']
        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(feature_map['node_forward']), len(feature_map['edge_forward']),
            self.max_prev_node, self.max_head_and_tail)
        self.feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        with open(self.dataset_path + 'graph' + str(self.graph_list[idx]) + '.dat', 'rb') as f:
            G = pickle.load(f)

        x_item = torch.zeros((self.max_nodes, self.feature_len))

        # get feature matrix for the min graph
        adj_feature_mat, perm = graph_to_matrix(
            G, self.feature_map['node_forward'], self.feature_map['edge_forward'],
            self.max_prev_node, self.max_head_and_tail, self.random_bfs)

        # prepare x_item
        x_item[0:adj_feature_mat.shape[0],
               :adj_feature_mat.shape[1]] = adj_feature_mat

        return {'x': x_item, 'len': len(adj_feature_mat), 'perm': perm}

    def collate_batch(self, batch):
        return batch


class Graph_Adj_Matrix(Dataset):
    """
    Mainly for testing purposes
    Dataset for taking graphs list and returning adjacency like matrices
    :param graph_list: List of graphs to be included in the dataset
    :param feature_map: feature_map for the dataset generated by the mapping
    :param max_prev_node: No of previous nodes to consider for edge prediction
    :param max_head_and_tail: Head and tail of adjacency vector to consider for edge prediction
    :random_bfs: Whether or not to do random_bfs
    """

    def __init__(self, args, feature_map, random_bfs=False):
        # self.graph_list = graph_list
        self.feature_map = feature_map
        self.max_prev_node = args.max_prev_node
        # Head and tail of adjacency vector to consider for edge prediction
        self.max_head_and_tail = args.max_head_and_tail
        self.random_bfs = random_bfs

        if self.max_prev_node is None and self.max_head_and_tail is None:
            print('Please provide max_prev_node or max_head_and_tail')
            exit()

        self.max_nodes = feature_map['max_nodes']
        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(len(
            feature_map['node_forward']), len(feature_map['edge_forward']), self.max_prev_node, self.max_head_and_tail)
        self.feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    def __call__(self, graph):
        # G = self.graph_list[idx]

        x_item = torch.zeros((self.max_nodes, self.feature_len))

        # get feature matrix for the min graph
        adj_feature_mat = graph_to_matrix(
            graph, self.feature_map['node_forward'], self.feature_map['edge_forward'], self.max_prev_node, self.max_head_and_tail, self.random_bfs)[0]

        # prepare x_item
        x_item[0:adj_feature_mat.shape[0],
               :adj_feature_mat.shape[1]] = adj_feature_mat

        return {'x': x_item, 'len': len(adj_feature_mat)}


class Graph_to_Adj_Matrix:
    def __init__(self, args, feature_map, random_bfs=False):
        print('Generating adj matrix...')
        self.feature_map = feature_map
        # No. of previous nodes to consider for edge prediction
        self.max_prev_node = args.max_prev_node
        # Head and tail of adjacency vector to consider for edge prediction
        self.max_head_and_tail = args.max_head_and_tail
        self.random_bfs = random_bfs
        if self.max_prev_node is None and self.max_head_and_tail is None:
            print('Please provide max_prev_node or max_head_and_tail')
            exit()

        self.max_nodes = feature_map['max_nodes']
        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(feature_map['node_forward']), len(feature_map['edge_forward']),
            self.max_prev_node, self.max_head_and_tail)
        self.feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    def __call__(self, graph, perm=None):
        # TODO given a graph and permutation, return an adj matrix, i.e., reimplement Graph_Adj_Matrix_from_file.__getitem__
        x_item = torch.zeros((self.max_nodes, self.feature_len))
        adj_feature_mat = self.graph_to_matrix(graph, self.feature_map['node_forward'], self.feature_map['edge_forward'], perm)
        x_item[0:adj_feature_mat.shape[0], :adj_feature_mat.shape[1]] = adj_feature_mat
        return {'x': x_item, 'len': len(adj_feature_mat)}

    def graph_to_matrix(self, in_graph, node_map, edge_map, perm):
        n = len(in_graph.nodes())
        len_node_vec, _, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(node_map), len(edge_map), self.max_prev_node, self.max_head_and_tail)

        # #-----------------------
        # perm = None
        # self.random_bfs = True
        # #------------------

        if perm is None:
            if self.random_bfs:
                n = len(in_graph.nodes())
                # Create a random permutaion of graph nodes
                perm = torch.randperm(n)
                adj = nx.to_numpy_matrix(in_graph, nodelist=perm.numpy(), dtype=int)
                G = nx.from_numpy_matrix(adj)
                start_id = 0
                # Construct bfs ordering starting from a random node
                bfs_seq = get_bfs_seq(G, start_id)
                seq = bfs_seq


        else:
            # TODO: examine the permutation is legal
            seq = perm
        # relabel graph
        seq = seq.cpu().numpy() #decide if really use learnable seq
        order_map = {seq[i]: i for i in range(n)}
        graph = nx.relabel_nodes(in_graph, order_map)
        #graph=in_graph





        # 3D adjacecny matrix in case of edge_features (each A[i, j] is a len_edge_vec size vector)
        adj_mat_2d = torch.ones((n, num_nodes_to_consider))
        adj_mat_2d.tril_(diagonal=-1)
        adj_mat_3d = torch.zeros((n, num_nodes_to_consider, len(edge_map)))

        node_mat = torch.zeros((n, len_node_vec))

        for v, data in graph.nodes.data():
            ind = node_map[data['label']]
            node_mat[v, ind] = 1

        for u, v, data in graph.edges.data():
            if self.max_prev_node is not None:
                if abs(u - v) <= self.max_prev_node:
                    adj_mat_3d[max(u, v), max(u, v) - min(u, v) -
                               1, edge_map[data['label']]] = 1
                    adj_mat_2d[max(u, v), max(u, v) - min(u, v) - 1] = 0

            elif self.max_head_and_tail is not None:
                if abs(u - v) <= self.max_head_and_tail[1]:
                    adj_mat_3d[max(u, v), max(u, v) - min(u, v) -
                               1, edge_map[data['label']]] = 1
                    adj_mat_2d[max(u, v), max(u, v) - min(u, v) - 1] = 0
                elif min(u, v) < self.max_head_and_tail[0]:
                    adj_mat_3d[max(u, v), self.max_head_and_tail[1] +
                               min(u, v), edge_map[data['label']]] = 1
                    adj_mat_2d[max(u, v), self.max_head_and_tail[1] + min(u, v)] = 0

        adj_mat = torch.cat((adj_mat_3d, adj_mat_2d.reshape(adj_mat_2d.size(
            0), adj_mat_2d.size(1), 1), torch.zeros((n, num_nodes_to_consider, 2))), dim=2)
        adj_mat = adj_mat.reshape((adj_mat.size(0), -1))

        return torch.cat((node_mat, adj_mat), dim=1)
