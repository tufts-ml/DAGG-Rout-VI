import numpy as np
import networkx
import pickle
from utils import load_model, get_model_attribute
import torch
from torch.utils.data._utils.collate import default_collate as collate
from models.gcn.helper import mp_sampler



def model_likelihood(model, graphs_indices, graphs_path, sample_size, eval_args):
    graphs=[]
    fact_nodes_number=[]
    #load test graphs
    train_args = eval_args.train_args

    for ind in graphs_indices:
        with open(graphs_path + 'graph' + str(ind) + '.dat', 'rb') as f:
            g = pickle.load(f)
            graphs.append(g)
            fact_nodes_number.append(np.math.factorial(g.number_of_nodes()))
    #load model


    train_args.device = eval_args.device
    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)




    load_model(eval_args.model_path, eval_args.device, model)
    #eval mode for model
    for _, net in model.items():
        net.eval()

    record_len = [g.number_of_nodes() for g in graphs]

    llg, l_rep= _get_log_likelihood(graphs, model, record_len, processor, eval_loss, train_args, feature_map, sample_size)
    mpg = _statistic(llg, l_rep)
    pg = mpg * fact_nodes_number
    print('Estimated probability is:')
    print(pg)

    return pg


def _get_log_likelihood(gs, model, record_len, processor, eval_loss, args, feature_map, sample_size):
    len_g = len(gs)
    ll_p = torch.empty((len_g, args.sample_size), device=args.device)
    reps = torch.empty((len_g, args.sample_size), device=args.device)
    rep_computer = mp_sampler(args)

    for i in range(sample_size):
        perms = _get_uniform_perm(record_len)
        data = [processor(graph, perms) for graph, perms in zip(gs, perms)]
        data = collate(data)
        ll_p_m = -eval_loss(args, model, data, feature_map)
        ll_p[:, i].copy_(ll_p_m)
        for j in range(len_g):
            log_rep = rep_computer.compute_repetition(gs[j], perms[j])
            reps[j, i] = log_rep

    return ll_p, reps


def _get_uniform_perm(record_len):
    return [np.array(torch.randperm(n)) for n in record_len]


def _statistic(llg, l_rep):
    tllg = llg-l_rep
    mtllg = torch.logsumexp(tllg, 1)-torch.empty(llg.shape[0]).fill_(torch.log(llg.shape[1]))
    return np.array(torch.exp(mtllg))









