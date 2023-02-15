import numpy as np
import networkx
import pickle
from utils import load_model, get_model_attribute
from models.graph_rnn.model import create_model as creat_grnn
from models.dgmg.model import create_model as creat_dgmg
from models.graph_rnn.data import Graph_to_Adj_Matrix, Graph_Adj_Matrix
from models.dgmg.data import Graph_to_Action
import torch
from torch.utils.data._utils.collate import default_collate as collate
from models.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
from models.dgmg.train import evaluate_loss as eval_loss_dgmg
from models.gcn.helper import mp_sampler



def model_likelihood(graphs_indices, graphs_path, sample_size, eval_args):
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
    if train_args.note == 'GraphRNN':
        model = creat_grnn(train_args, feature_map)
        processor = Graph_to_Adj_Matrix(train_args, feature_map, random_bfs= False)
        eval_loss = eval_loss_graph_rnn
    if train_args.note == 'DGMG':
        model =  creat_dgmg(train_args, feature_map)
        processor = Graph_to_Action(train_args, feature_map)
        eval_loss = eval_loss_dgmg




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


if __name__ == '__main__':
    class ArgsEvaluate():
        def __init__(self, name, epoch):
            # Can manually select the device too
            '''

            :param name: str, output dir for a certain exp
            :param epoch: int, epoch to evaluate
            '''

            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

            self.model_path = 'output/' + name + '/model_save' + "/epoch_" + str(epoch) + ".dat"

            self.train_args = get_model_attribute(
                'saved_args', self.model_path, self.device)

            self.num_epochs = get_model_attribute(
                'epoch', self.model_path, self.device)

            # Whether to generate networkx format graphs for real datasets
            self.generate_graphs = True

            # Number of graphs to produce from the model
            self.count = 50
            self.batch_size = 25  # Must be a factor of count

            self.metric_eval_batch_size = 256

            # Specific to GraphRNN
            self.min_num_node = 0
            self.max_num_node = 50

            # Set for Likelihood
            self.need_llh = True
            self.llh_sample = 1000  # number of sample to estimate true likelihood
            self.test_number = 1  # number of test graphs to be estimated for likelihood

            self.graphs_save_path = 'output/' + name + '/generated_graphs/'
            self.current_graphs_save_path = self.graphs_save_path

            # self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
            #     self.train_args.time + '/' + str(self.num_epochs) + '/'

    eval_args = ArgsEvaluate(name='attGen-noZ_ENZYMES_gat_nobfs_2021_07_03_19_10_32', epoch=80)
    train_args = eval_args.train_args
    graph_path = 'datasets/Lung/graphs/'
    graphs_indices = [1,2]
    model_likelihood(graphs_indices, graph_path, sample_size=20, eval_args=eval_args)







