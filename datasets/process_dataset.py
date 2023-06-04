import os
import torch
import warnings
import pickle
from utils import caveman_special 

def default_label_graph(G):
    for node in G.nodes():
        G.nodes[node]['label'] = 'DEFAULT_LABEL'
    for edge in G.edges():
        G.edges[edge]['label'] = 'DEFAULT_LABEL'


def load_graph_dataset(args):

    if args.dataset =='caveman_small':

        # save dataset to file
        current_dataset_path = os.path.join(args.dataset_path, args.dataset, 'graphs.dat')

        if args.produce_graphs or (not os.path.isfile(current_dataset_path)):
            
            if not args.produce_graphs:
                warnings.warn(f"Data file does not exist -- will create a new dataset using this file path: {current_dataset_path}")
            
            # generate a new graph dataset to a list
            graphs = []
            for i in range(2, 3):
                for j in range(6, 11):
                    for k in range(100):
                        graph = caveman_special(i, j, p_edge=0.8) # default 0.8
                        default_label_graph(graph)
                        graphs.append(graph)
            
            dataset = GraphListDataset(graphs)
            
            # save dataset to the buffer file 
            if os.path.isfile(current_dataset_path):
                warnings.warn(f"This dataset file is over-written by the new dataset: {current_dataset_path}") 
            
            with open(current_dataset_path, 'wb') as f:
                pickle.dump(dataset, f)

        else: 
            
            if not os.path.isfile(current_dataset_path):
                raise Exception(f"Cannot load the dataset from: {current_dataset_path}") 
             
            with open(current_dataset_path, 'rb') as f:
                dataset = pickle.load(f)

        # TODO: setting args entries in args.py
        args.max_prev_node = 20
    
    else: 
        raise Exception(f"This dataset is not provided by this repo: {args.dataset}")

    return dataset 



class GraphListDataset(torch.utils.data.IterableDataset):

    def __init__(self, graph_list):
        super(GraphListDataset).__init__()

        if len(graph_list) == 0:
            raise Exception("Cannot generate an empty graph dataset")

        # record the list
        self.graph_list = graph_list

        # get simple statistics
        self.statistics = self._get_statistics()

        # mapping string labels to integers
        self.node_label_list = self._map_node_labels()
        self.edge_label_list = self._map_edge_labels()


    def __iter__(self):
        return iter(self.graph_list)

    def __len__(self):
        return len(self.graph_list)


    def _map_node_labels(self):

        if "label" not in self.graph_list[0].nodes[0]:
            print("From the structure of the first graph, the dataset does not contain node labels")
            return None
        
        label_set = set() 

        for graph in self.graph_list:
            for _, data in graph.nodes.data():
                label_set.add(data['label'])

        label_list = list(label_set)

        # integer representation of labels
        label_indices = range(len(label_list))

        label_dict = dict(zip(label_list, label_indices))

        # mapping string labels to integers
        for graph in self.graph_list:
            for _, data in graph.nodes.data():
                data['label'] = label_dict[data['label']]

        return label_list



    def _map_edge_labels(self):

        for _, _, data in self.graph_list[0].edges.data():
            if "label" not in data:
                print("From the structure of the first graph, the dataset does not contain edge labels")
                return None
            break
        
        label_set = set() 

        for graph in self.graph_list:
            for _, _, data in graph.edges.data():
                label_set.add(data['label'])

        label_list = list(label_set)

        # integer representation of labels
        label_indices = range(len(label_list))

        label_dict = dict(zip(label_list, label_indices))

        # mapping string labels to integers
        for graph in self.graph_list:
            for _, _, data in graph.edges.data():
                data['label'] = label_dict[data['label']]

        return label_list



    def _get_statistics(self):

        graph = self.graph_list[0]
        statistics = dict(max_num_nodes=graph.number_of_nodes(), 
                          min_num_nodes=graph.number_of_nodes(), 
                          max_num_edges=graph.number_of_edges(), 
                          min_num_edges=graph.number_of_edges())

       
        for graph in self.graph_list: 

            statistics["max_num_nodes"] = max(statistics["max_num_nodes"], graph.number_of_nodes())
            statistics["min_num_nodes"] = min(statistics["min_num_nodes"], graph.number_of_nodes())
 
            statistics["max_num_edges"] = max(statistics["max_num_edges"], graph.number_of_edges())
            statistics["min_num_edges"] = min(statistics["min_num_edges"], graph.number_of_edges())

        return statistics 




