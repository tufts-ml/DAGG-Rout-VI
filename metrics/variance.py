import os
import random
import shutil
from statistics import mean
import torch
import numpy as np
import pandas as pd

# from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from models.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
#from models.dgmg.train import predict_graphs as gen_graphs_dgmg
from models.graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from models.gran.model import predict_graphs as gen_graphs_gran
from utils import get_model_attribute, load_graphs, save_graphs, get_last_checkpoint, load_model
from models.graph_rnn.model import create_model as create_model_rnn
from models.gran.model import create_model as create_model_gran
from models.data import Graph_from_file
from torch.utils.data import DataLoader
from models.graph_rnn.data import Graph_to_Adj_Matrix, Graph_Adj_Matrix
from models.dgmg.data import Graph_to_Action
from models.gran.data import GRANData
from model import create_models
from datasets.process_dataset import create_graphs
from train import evaluate_loss
import matplotlib as plt
from models.gcn.helper import legal_perms_sampler, mp_sampler
#from utils import load_model, get_model_attribute,load_graphs, save_graphs

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
        self.current_model_path = 'output/' + name + '/model_save'

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        self.sample_size = 16




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
def cal_var(grad):
    var = torch.var(grad, dim=0)
    var = torch.mean(var).cpu().item()
    return var
    

# MMD Degree - 0.501989, MMD Clustering - 0.236976, MMD Orbits - 0.233946
# MMD NSPDK - 0.193140
if __name__ == "__main__":


    eval_args = ArgsEvaluate(name='GraphRNN_Lung_gat_nobfs_2021_01_02_22_37_18', epoch=6)


    train_args = eval_args.train_args
    train_args.sample_size = eval_args.sample_size

    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))
    graphs = create_graphs(train_args)

    random.shuffle(graphs)
    graphs_train = graphs[: int(0.80 * len(graphs))]
    graphs_validate = graphs[int(0.80 * len(graphs)): int(0.90 * len(graphs))]
    sample_perm = mp_sampler(train_args)


    # if train_args.note == 'GraphRNN':
    #     model = create_model_rnn(train_args, feature_map)
    # elif train_args.note == 'DGMG':
    #     pass
    # elif train_args.note == 'Graphgen':
    #     pass
    # elif train_args.note == 'GRAN':
    #     model = create_model_gran(train_args, feature_map)

    model, gcn =create_models(train_args, feature_map, vf2=False)




    dataset_train = Graph_from_file(train_args, graphs_train, feature_map)
    dataset_validate = Graph_from_file(train_args, graphs_validate, feature_map)
    dataloader_train = DataLoader(
        dataset_train[:eval_args.batch_size], batch_size=eval_args.batch_size, shuffle=True, drop_last=True,
        num_workers=train_args.num_workers, collate_fn=dataset_train.collate_batch)
    dataloader_validate = DataLoader(
        dataset_validate, batch_size=eval_args.batch_size, shuffle=False, drop_last=True,
        num_workers=train_args.num_workers, collate_fn=dataset_validate.collate_batch)
    if train_args.note == 'GraphRNN':
        procesor = Graph_to_Adj_Matrix(train_args, feature_map, random_bfs=True)
    elif train_args.note == 'DGMG':
        pass
    elif train_args.note == 'Graphgen':
        pass
    elif train_args.note == 'GRAN':
        processor = GRANData(train_args, feature_map['max_nodes'])

    var_list = []
    for dat in os.listdir(eval_args.current_model_path):
        load_model(dat, eval_args.device, model)
        for _, net in model.items():
            net.train()

        if train_args.enable_gcn:
            gcn.train()

        batch_count = len(dataloader_train)
        gradients = torch.empty((16, 100), device=train_args.device)
        for i in range(100):
            for batch_id, graphs in enumerate(dataloader_train):

                for _, net in model.items():
                    net.zero_grad()

                if train_args.enable_gcn:
                    gcn.zero_grad()


                nll_p, fake_nll_q, elbo, ll_q = evaluate_loss(train_args, model, gcn, processor, sample_perm, graphs, feature_map,
                                                          1)
                loss = nll_p + fake_nll_q

                loss.backward()
            gradient = gcn.node_readout.FC_layers[0].bias.grad
            gradients[i,:].copy_(gradient)
        var = cal_var(gradients)
        var_list.append(var)

    plt.plot(np.arange(len(var_list)), var_list)
    plt.show()




