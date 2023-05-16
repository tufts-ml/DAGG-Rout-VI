import os
import random
import thirdy_party.stats
from statistics import mean
from metrics.mmd import generate_graphs,patch_graph,print_stats,save_stats
from utils import load_graphs
from metrics.likelihood import model_likelihood




#This function shoud be tested
def evaluate(args, p_model):
    #start evaluating mmd
    #get current graphs in the trainning set
    graphs_test_indices = []
    for name in os.listdir(args.current_dataset_path):
        if name.endswith('.dat'):
            graphs_test_indices.append(len(graphs_test_indices))

    #generate graphs
    generate_graphs(args,p_model)
    graphs_pred_indices = [i for i in range(args.count)]



    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []

    degree_mmd, clustering_mmd, shortest_path_mmd, orbit_mmd, nspdk_mmd = [], [], [], [], []
    node_label_mmd, edge_label_mmd, node_label_and_degree = [0], [0], [0]

    print(len(graphs_test_indices))
    for i in range(0, len(graphs_pred_indices), args.metric_eval_batch_size):
        batch_size = min(args.metric_eval_batch_size,
                         len(graphs_pred_indices) - i)
        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
        graphs_ref = load_graphs(
            args.current_dataset_path, graphs_ref_indices)

        graphs_ref = [patch_graph(g) for g in graphs_ref]

        graphs_pred = load_graphs(
            args.current_graphs_save_path, graphs_pred_indices[i: i + batch_size])

        graphs_pred = [patch_graph(g) for g in graphs_pred]

        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

        degree_mmd.append(thirdy_party.stats.degree_stats(graphs_ref, graphs_pred))
        clustering_mmd.append(
            thirdy_party.stats.clustering_stats(graphs_ref, graphs_pred))
        shortest_path_mmd.append(
            thirdy_party.stats.shotest_path_stats(graphs_ref, graphs_pred))

        orbit_mmd.append(thirdy_party.stats.orbit_stats_all(
            graphs_ref, graphs_pred))



        print('Running average of metrics:\n')

        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
            edge_label_mmd, node_label_and_degree
        )

    print('Evaluating {}, run at {}, epoch {}'.format(
        args.fname, args.time, args.num_epochs))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
        edge_label_mmd, node_label_and_degree
    )

    save_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
        edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
        nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree,
        args
    )
    #finish evaluating mmd
    #start evaluating graph likelihood
    graph_likelihood = model_likelihood(args, p_model, graphs_test_indices, args.sample_size)


