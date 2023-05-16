import numpy as np
import pickle
import torch



def model_likelihood(args, model, graphs_indices, sample_size):
    '''
    This function is to estimate likehood of the given graphs with DAGG.
    '''
    graphs=[]
    fact_nodes_number=[]
    #load test graphs


    for ind in graphs_indices:
        with open(args.current_graphs_save_path + 'graph' + str(ind) + '.dat', 'rb') as f:
            g = pickle.load(f)
            graphs.append(g)
            fact_nodes_number.append(np.math.factorial(g.number_of_nodes()))

    record_len = [g.number_of_nodes() for g in graphs]

    llg= _get_log_likelihood(args, graphs, model, record_len, sample_size)
    mpg = _statistic(llg)
    pg = mpg * fact_nodes_number
    print('Estimated probability is:')
    print(pg)

    return pg


def _get_log_likelihood(args, gs, model, record_len, sample_size):
    len_g = len(gs)
    ll_p = torch.empty((len_g, args.sample_size), device=args.device)



    for i in range(sample_size):
        perms = _get_uniform_perm(record_len)
        ll_p_m =  -model(gs[i], perms)
        ll_p[:, i].copy_(ll_p_m)



    return ll_p


def _get_uniform_perm(record_len):
    return [np.array(torch.randperm(n)) for n in record_len]


def _statistic(llg):

    mtllg = torch.logsumexp(llg, 1)-torch.empty(llg.shape[0]).fill_(torch.log(llg.shape[1]))
    return np.array(torch.exp(mtllg))









