from third_party import stats

def mmd(graph_ref_batch, graph_pred_batch):

    batch_degree_mmd = stats.degree_stats(graph_ref_batch, graph_pred_batch)
    batch_clustering_mmd = stats.clustering_stats(graph_ref_batch, graph_pred_batch)
    batch_orbit_mmd = stats.orbit_stats_all(graph_ref_batch, graph_pred_batch)

    return  batch_degree_mmd, batch_clustering_mmd, batch_orbit_mmd






