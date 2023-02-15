import os
import random
import shutil
from statistics import mean
import torch
import numpy as np
import pandas as pd


from models.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from models.gran.model import predict_graphs as gen_graphs_gran
from models.graph_decoder.predict import predict_graphs
from utils import get_model_attribute, load_graphs, save_graphs, get_last_checkpoint

import metrics.stats

LINE_BREAK = '----------------------------------------------------------------------\n'


class ArgsEvaluate():
    def __init__(self, name, epoch):
        # Can manually select the device too
        '''

        :param name: str, output dir for a certain exp
        :param epoch: int, epoch to evaluate
        '''

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_path = 'output/' + name + '/model_save' + "/epoch_" +str(epoch) + ".dat"

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        # Number of graphs to produce from the model
        self.count = 1 # No larger than 50 for Enzymes(small dataset)
        self.batch_size = 1 # Must be a factor of count(important)

        self.metric_eval_batch_size = 200  #Prefer a factor of count



        # Specific to GraphRNN
        self.min_num_node = 0
        self.max_num_node = 20

        # Specific DFScodeRNN
        self.max_num_edges = 50  #149 for ENZYMES

        #Set for Likelihood
        self.need_llh = True
        self.llh_sample = 1000 #number of sample to estimate true likelihood
        self.test_number = 1  #number of test graphs to be estimated for likelihood



        self.graphs_save_path = 'output/' + name + '/generated_graphs/'
        self.current_graphs_save_path = self.graphs_save_path


        # self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
        #     self.train_args.time + '/' + str(self.num_epochs) + '/'


def patch_graph(graph):
    for u in graph.nodes():
        if 'label' in graph.nodes[u]:
            graph.nodes[u]['label'] = graph.nodes[u]['label'].split('-')[0]
        else:
            graph.nodes[u]['label'] = 'DEFAULT_LABEL'

    return graph


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """
    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)

    elif train_args.note == 'attGen-noZ':
        gen_graphs = predict_graphs(eval_args)
    elif train_args.note == 'GRAN':
        gen_graphs = gen_graphs_gran(eval_args)


    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print('MMD Node label - {:.6f}, MMD Edge label - {:.6f}'.format(
        mean(node_label_mmd), mean(edge_label_mmd)
    ))
    print('MMD Joint Node label and degree - {:.6f}'.format(
        mean(node_label_and_degree)
    ))
    print(LINE_BREAK)


#save the rsult to the csv
def save_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree,
    uni, nove, eval_args
):
    stats = {}
    stats['Node count avg Test'] = np.array([mean(node_count_avg_ref)])
    stats['Node count avg Generated'] = np.array([mean(node_count_avg_pred)])

    stats['Edge count avg avg Test'] = np.array([mean(edge_count_avg_ref)])
    stats['Edge count avg Generated'] = np.array([mean(edge_count_avg_pred)])

    stats['MMD Degree'] = np.array([mean(degree_mmd)])
    stats['MMD Clustering'] = np.array([mean(clustering_mmd)])
    stats['MMD Orbits'] = np.array([mean(orbit_mmd)])

    stats['MMD NSPDK'] = np.array([mean(nspdk_mmd)])
    stats['MMD Node label'] = np.array([mean(node_label_mmd)])
    stats['MMD Edge label'] = np.array([mean(edge_label_mmd)])
    stats['MMD Joint Node label and degree'] = np.array([mean(node_label_and_degree)])

    stats['novelity'] = np.array([nove])
    stats['uniqueness'] = np.array([uni])
    print(stats)
    hist = pd.DataFrame.from_dict(stats)
    hist.to_csv('output/'+ eval_args.train_args.fname+ '/stats.csv')
#
# MMD Degree - 0.501989, MMD Clustering - 0.236976, MMD Orbits - 0.233946
# MMD NSPDK - 0.193140
if __name__ == "__main__":


    eval_args = ArgsEvaluate(name='attGen-noZ_ENZYMES_gat_nobfs_2021_07_03_19_10_32', epoch=83)


    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    if eval_args.generate_graphs:
        generate_graphs(eval_args)

    random.seed(123)

    graphs = []
    for name in os.listdir(train_args.current_dataset_path):
        if name.endswith('.dat'):
            graphs.append(len(graphs))

    random.shuffle(graphs)
    graphs_test_indices = graphs#[int(0.90 * len(graphs)):]
    graphs_train_indices = graphs#[:int(0.90 * len(graphs))]

    graphs_pred_indices = []
    if not eval_args.generate_graphs:
        for name in os.listdir(eval_args.current_graphs_save_path):
            if name.endswith('.dat'):
                graphs_pred_indices.append(len(graphs_pred_indices))
    else:
        graphs_pred_indices = [i for i in range(eval_args.count)]

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print('Graphs generated - {}'.format(len(graphs_pred_indices)))
    #

    # novelity = metrics.stats.novelity(
    #     train_args.current_dataset_path, graphs_train_indices, eval_args.current_graphs_save_path,
    #     graphs_pred_indices, train_args.temp_path, timeout=60)
    #
    # uniqueness = metrics.stats.uniqueness(
    #     eval_args.current_graphs_save_path,
    #     graphs_pred_indices, train_args.temp_path, timeout=120)
    novelity = 0
    uniqueness = 0
    # exit()

    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []

    degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
    node_label_mmd, edge_label_mmd, node_label_and_degree = [0], [0], [0]

    print(len(graphs_test_indices))
    for i in range(0, len(graphs_pred_indices), eval_args.metric_eval_batch_size):
        batch_size = min(eval_args.metric_eval_batch_size,
                         len(graphs_pred_indices) - i)
        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
        graphs_ref = load_graphs(
            train_args.current_dataset_path, graphs_ref_indices)

        graphs_ref = [patch_graph(g) for g in graphs_ref]

        graphs_pred = load_graphs(
            eval_args.current_graphs_save_path, graphs_pred_indices[i: i + batch_size])

        graphs_pred = [patch_graph(g) for g in graphs_pred]

        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

        degree_mmd.append(metrics.stats.degree_stats(graphs_ref, graphs_pred))
        clustering_mmd.append(
            metrics.stats.clustering_stats(graphs_ref, graphs_pred))
        orbit_mmd.append(metrics.stats.orbit_stats_all(
            graphs_ref, graphs_pred))

        #nspdk_mmd.append(metrics.stats.nspdk_stats(graphs_ref, graphs_pred))

        #node_label_mmd.append(
            #metrics.stats.node_label_stats(graphs_ref, graphs_pred))
        #edge_label_mmd.append(
            #metrics.stats.edge_label_stats(graphs_ref, graphs_pred))
        #node_label_and_degree.append(
            #metrics.stats.node_label_and_degree_joint_stats(graphs_ref, graphs_pred))

        print('Running average of metrics:\n')

        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
            edge_label_mmd, node_label_and_degree
        )

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
        edge_label_mmd, node_label_and_degree
    )

    save_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
        edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
        nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree,
        novelity, uniqueness, eval_args
    )
