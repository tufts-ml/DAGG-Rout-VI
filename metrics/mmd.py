import os
import random
import shutil
from statistics import mean
import torch
import numpy as np
import pandas as pd
from utils import get_model_attribute, load_graphs, save_graphs, get_last_checkpoint
from models.DAGG.model import DAGG




def patch_graph(graph):
    for u in graph.nodes():
        if 'label' in graph.nodes[u]:
            graph.nodes[u]['label'] = graph.nodes[u]['label'].split('-')[0]
        else:
            graph.nodes[u]['label'] = 'DEFAULT_LABEL'

    return graph


def generate_graphs(args, DAGG):


    graphs = DAGG.sample(args.count)


    if os.path.isdir(args.current_graphs_save_path):
        shutil.rmtree(args.current_graphs_save_path)

    os.makedirs(args.current_graphs_save_path)

    save_graphs(args.current_graphs_save_path, graphs)


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



#save the rsult to the csv
def save_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree, args):
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
    print(stats)
    hist = pd.DataFrame.from_dict(stats)
    hist.to_csv('output/'+ args.fname+ '/stats.csv')