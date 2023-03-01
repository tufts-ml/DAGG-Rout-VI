import torch
import networkx as nx
from models.graph_rnn.helper import graph_to_matrix, get_attributes_len_for_graph_rnn

class Graph_to_att_Matrix():
    def __init__(self, args, feature_map):
        self.node_map = feature_map['node_forward']
        self.feature_map = feature_map
        # No. of previous nodes to consider for edge prediction
        self.max_prev_node = args.max_prev_node
        # Head and tail of adjacency vector to consider for edge prediction
        self.max_head_and_tail = None

        if self.max_prev_node is None and self.max_head_and_tail is None:
            print('Please provide max_prev_node or max_head_and_tail')
            exit()

        self.max_nodes = feature_map['max_nodes']
        self.len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(feature_map['node_forward']), len(feature_map['edge_forward']),
            self.max_prev_node, self.max_head_and_tail)
        self.feature_len = self.len_node_vec + num_nodes_to_consider * len_edge_vec

    def __call__(self, graph,perm):
        '''

        :param graph:nx_graph
        :param perm: [1,2,3]
        :return: [node_type], adj:numpy_array(n,n), number_of_nodes:int
        '''
        n = len(graph.nodes())
        seq = perm
        # relabel graph
        order_map = {seq[i]: i for i in range(n)}
        graph = nx.relabel_nodes(graph, order_map)
        matrix=nx.to_numpy_matrix(graph)
        adj_matrix = torch.tensor(matrix, dtype=torch.float)

        node_mat = torch.zeros((n, self.len_node_vec))

        for v, data in graph.nodes.data():
            ind = self.node_map[data['label']]
            node_mat[v, ind] = 1

        data = {'nlabel':node_mat, 'adj': adj_matrix, 'n':n}
        size = fact(n-1)
        conection=torch.zeros(size=(size,),dtype=torch.float)

        s=0
        for i in range(n-1):
            data[i+1] = adj_matrix[i+1, :i+1]
            #bug = adj_matrix[i + 1, :i + 1]
            e=s+i+1
            conection[s:e] = data[i+1]
            s=e

        conection= conection.type(torch.LongTensor).view(-1, 1)
        n_dims = int(torch.max(conection)) + 1
        c_one_hot = torch.zeros(conection.size()[0], n_dims).scatter_(1, conection, 1)
        c_one_hot = c_one_hot.view(size, -1)



        data['con'] = c_one_hot


        adj_feature_mat = self.graph_to_matrix(graph, self.feature_map['node_forward'],
                                               self.feature_map['edge_forward'])

        data['x'] = adj_feature_mat

        #for debug
        a=data['x'][0]
        b=adj_matrix[0]
        aa= data['x'][1]
        bb = adj_matrix[1]
        aaa = data['x'][2]
        bbb= adj_matrix[2]
        aaaa = data['x'][3]
        bbbb = adj_matrix[3]
        aaaaa = data['x'][4]
        bbbbb = adj_matrix[4]

        data['len'] = len(adj_feature_mat)







        return data

    def graph_to_matrix(self, in_graph, node_map, edge_map):
        n = len(in_graph.nodes())
        len_node_vec, _, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            len(node_map), len(edge_map), self.max_prev_node, self.max_head_and_tail)

        graph=in_graph

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






def fact(num):
    factorial=int(0)

    for i in range(0,num + 1):
               factorial = factorial+i
    return int(factorial)

