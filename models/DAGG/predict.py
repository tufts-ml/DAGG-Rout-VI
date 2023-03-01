import torch
import numpy as np
import random
import os
from utils import load_model, get_model_attribute,load_graphs, save_graphs
from model import create_models








def predict_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative graphRNN model
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args
    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)
    #evaluate num_nodes_pmf
    random.seed(123)
    graphs = []
    for name in os.listdir(train_args.current_dataset_path):
        if name.endswith('.dat'):
            graphs.append(len(graphs))
    random.shuffle(graphs)
    graphs_train_indices = graphs[:int(0.90 * len(graphs))]
    ref_graphs = load_graphs(train_args.current_dataset_path, graphs_indices=graphs[:2000])
    num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in ref_graphs])
    #max_num_nodes = len(num_nodes_pmf_train)
    num_nodes_pmf_train = num_nodes_pmf_train / num_nodes_pmf_train.sum()



    train_args.device = eval_args.device

    gmodel,pmodel,_ =  create_models(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, gmodel, pmodel)


    for _, net in gmodel.items():
        net.eval()
    model = gmodel['gmodel']
    max_num_node = train_args.max_prev_node

    graphs_gen = model.sample(num_nodes_pmf_train, max_num_node, eval_args.batch_size)

    return graphs_gen