import os
import random
import thirdy_party.stats
from statistics import mean
from metrics.mmd import generate_graphs,patch_graph,print_stats,save_stats
from utils import load_graphs
from metrics.likelihood import model_likelihood

def evaluate(args, p_model, dataset_test):
    """
     
    """
    #generate graphs
    graphs = p_model.sample(args.count)

    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []

    degree_mmd, clustering_mmd, shortest_path_mmd, orbit_mmd, nspdk_mmd = [], [], [], [], []

    for i in range(0, len(graphs_pred_indices), args.metric_eval_batch_size):
        batch_size = min(args.metric_eval_batch_size,
                         len(graphs_pred_indices) - i)

        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
        graphs_ref = [dataset_test[i] for i in  graphs_ref_indices]

        graphs_pred = graphs[i: i + batch_size]

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



    print('Evaluating {}, run at {}, epoch {}'.format(
        args.fname, args.time, args.num_epochs))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd
    )



    save_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
        edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, args
    )


    graph_likelihood = model_likelihood(args, p_model, dataset_test, args.sample_size)

    print('Estimated log likelihood per graph is:')
    print(graph_likelihood)




def mmd(graph_ref_batch, graph_pred_batch):


    return ...


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
