import os
import shutil
import pickle
import torch
import networkx as nx
import pynauty as pnt
import numpy as np
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL, python_exit_status
import torch.multiprocessing as multiprocessing
from torch.distributions.distribution import Distribution


def mkdir(path):
    if os.path.isdir(path):
        is_del = input('Delete ' + path + ' Y/N:')
        if is_del.strip().lower() == 'y':
            shutil.rmtree(path)
        else:
            exit()

    os.makedirs(path)


def load_graphs(graphs_path, graphs_indices=None):
    """
    Returns a list of graphs given graphs directory and graph indices (Optional)
    If graphs_indices are not provided all graphs will be loaded
    """

    graphs = []
    if graphs_indices is None:
        for name in os.listdir(graphs_path):
            if not name.endswith('.dat'):
                continue

            with open(graphs_path + name, 'rb') as f:
                graphs.append(pickle.load(f))
    else:
        for ind in graphs_indices:
            with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
                graphs.append(pickle.load(f))

    return graphs


def save_graphs(graphs_path, graphs):
    """
    Save networkx graphs to a directory with indexing starting from 0
    """
    for i in range(len(graphs)):
        with open(graphs_path + 'graph' + str(i) + '.dat', 'wb') as f:
            pickle.dump(graphs[i], f)


# Create Directories for outputs
def create_dirs(args):
    if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)

    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)

    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)

    if not os.path.isdir(args.logging_path):
        os.makedirs(args.logging_path)


def save_model(epoch, args, gmodel, qmodel):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)

    gmodel_path = args.current_model_save_path +'epoch' + '_' + 'gmodel'+'_' + str(epoch) + '.dat'

    torch.save(gmodel, gmodel_path)

    qmodel_path = args.current_model_save_path + 'epoch' + '_' + 'qmodel' + '_' + str(epoch) + '.dat'

    torch.save(qmodel, qmodel_path)


def load_model(args):
    gmodel_path = args.current_model_save_path + 'epoch' + '_' + 'gmodel' + '_' + str(epoch) + '.dat'
    gmodel =torch.load(gmodel_path)
    qmodel_path = args.current_model_save_path + 'epoch' + '_' + 'qmodel' + '_' + str(epoch) + '.dat
    qmodel = torch.load(qmodel_path)
    return gmodel, qmodel

def get_last_checkpoint(args, epoch):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = args.load_model_path + '/model_save/'
    # Checkpoint file names are in lexicographic order
    last_checkpoint_name = checkpoint_dir + 'epoch' + '_' + str(epoch) + '.dat'
    print('Last checkpoint is {}'.format(last_checkpoint_name))
    return last_checkpoint_name, epoch


def get_model_attribute(attribute, fname, device):

    checkpoint = torch.load(fname, map_location=device)

    return checkpoint[attribute]


def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)),1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1-p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max(connected_component_subgraphs(G), key=len)
    return G


def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    #print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G


def perturb_new(graph_list, p):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_remove_count = 0
        for (u, v) in list(G.edges()):
            if np.random.rand()<p:
                G.remove_edge(u, v)
                edge_remove_count += 1
        # randomly add the edges back
        for i in range(edge_remove_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u,v)) and (u!=v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def nx_to_nauty(nx_G):
    na_G = pnt.Graph(nx_G.number_of_nodes())
    for n_node in range(nx_G.number_of_nodes()):
        na_G.connect_vertex(n_node, list(nx_G.neighbors(n_node)))
    return na_G

def nauty_to_nx(na_G):
    raise NotImplementedError
    # pass

def smart_perm(x, permutation):
    assert x.size() == permutation.size()
    if x.ndimension() == 1:
        ret = x[permutation]
    elif x.ndimension() == 2:
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
    elif x.ndimension() == 3:
        d1, d2, d3 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2 * d3)).flatten(),
            torch.arange(d2).unsqueeze(1).repeat((1, d3)).flatten().unsqueeze(0).repeat((1, d1)).flatten(),
            permutation.flatten()
        ].view(d1, d2, d3)
    else:
        ValueError("Only 3 dimensions maximum")
    return ret
class PlackettLuce(Distribution):
    """
        Plackett-Luce distribution
    """
    arg_constraints = {"logits": constraints.real}
    def __init__(self, logits):
        # last dimension is for scores of plackett luce
        super(PlackettLuce, self).__init__()
        self.logits = logits
        self.size = self.logits.size()

    def sample(self, num_samples):
        # sample permutations using Gumbel-max trick to avoid cycles
        with torch.no_grad():
            logits = self.logits.unsqueeze(0).expand(num_samples, *self.size)
            u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
            z = self.logits - torch.log(-torch.log(u))
            samples = torch.sort(z, descending=True, dim=-1)[1]
        return samples

    def log_prob(self, samples):
        # samples shape is: num_samples x self.size
        # samples is permutations not permutation matrices
        if samples.ndimension() == self.logits.ndimension():  # then we already expanded logits
            logits = smart_perm(self.logits, samples)
        elif samples.ndimension() > self.logits.ndimension():  # then we need to expand it here
            logits = self.logits.unsqueeze(0).expand(*samples.size())
            logits = smart_perm(logits, samples)
        else:
            raise ValueError("Something wrong with dimensions")
        logp = (logits - reverse_logcumsumexp(logits, dim=-1)).sum(-1)
        return logp
class mp_sampler():
    def __init__(self, args, vf2=False):
        self.args = args
        self._workers = []
        self._index_queues = []
        self.device = args.device
        self._worker_result_queue = multiprocessing.Queue()
        self._workers_done_event = multiprocessing.Event()
        self._num_workers = args.mp_num_workers
        self.max_cr_iteration = args.max_cr_iteration
        wkl = worker_loop_vf2 if vf2 else worker_loop

        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing.Queue()  # type: ignore
            # index_queue.cancel_join_thread()
            w = multiprocessing.Process(
                target=wkl,
                args=(index_queue, self._worker_result_queue, self._workers_done_event))
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

    def compute_repetition(self, graph, perm):
        perm_length = len(perm)
        # subgraphs = ([graph.subgraph(perm[:i + 1]) for i in range(len(perm))])
        # for idx in range(perm_length):
        for idx in range(perm_length-1, -1, -1):
            # TODO: add early stop
            worker_id = idx % self._num_workers
            subgraph = graph.subgraph(perm[:idx + 1])
            self._index_queues[worker_id].put((subgraph, min(subgraph.number_of_nodes(), self.max_cr_iteration), perm[idx]))
        count = 0
        results = []
        while count != perm_length:
            results.append(self._worker_result_queue.get(timeout=MP_STATUS_CHECK_INTERVAL))
            count += 1
        self._worker_result_queue.empty()
        results = torch.tensor(results, dtype=torch.float32, requires_grad=False)
        log_rep = torch.sum(torch.log(results))
        return log_rep

    def __call__(self, graph, params, device, M=1, nobfs=True, max_cr_iteration=10):
        self.max_cr_iteration = max_cr_iteration
        self.device = device
        # perms = []
        # log_probs = torch.empty(M, device=device)

        # perm, log_prob = self.sample_legal_perm(graph, params)
        log_reps = torch.empty(M)
        perms = PlackettLuce(logits=params).sample(M)
        log_probs = PlackettLuce(logits=params).log_prob(perms)
        perms = perms.tolist()
        for m in range(M):
            if nobfs:
                rep = self.compute_repetition(graph, perms[m])
                log_reps[m].fill_(rep)
            else:

                # TODO: rewrite bfs sampler as a class function @Xu
                pass
        return perms, log_probs, log_reps

    def _shutdown_workers(self):
        if python_exit_status is True or python_exit_status is None:
            return
        self._workers_done_event.set()
        for w in self._workers:
            w.join(timeout=MP_STATUS_CHECK_INTERVAL)
            if w.is_alive():
                w.terminate()
        for q in self._index_queues:
            q.cancel_join_thread()
            q.close()

    def __del__(self):
        self._shutdown_workers()

