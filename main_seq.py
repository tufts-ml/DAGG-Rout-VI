import random
import time
import pickle
from torch.utils.data import DataLoader
import torch
import os, json
from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node
from models.data import Graph_from_file, NumpyTupleDataset
from model import create_generative_model, create_inference_model
from train_seq import train


# TODO: add documentation to all functions 


if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(args.seed)

    # graphs = create_graphs(args)[:100] # for implementation test
    graphs = create_graphs(args)


    # Loading the feature map
    with open(args.current_processed_dataset_path +'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))
    print(args.__dict__)



    if args.note == 'GraphRNN' or args.note == 'attGen-noZ':
        start = time.time()
        if args.nobfs:
            args.max_prev_node = feature_map['max_nodes'] - 1
        if args.max_prev_node is None:
             args.max_prev_node = calc_max_prev_node(args.current_processed_dataset_path)

        args.max_head_and_tail = None
        print('max_prev_node:', args.max_prev_node)

        end = time.time()
        print('Time taken to calculate max_prev_node = {:.3f}s'.format(
            end - start))

    random.shuffle(graphs)
    graphs_train = graphs[: int(0.80 * len(graphs))]
    graphs_validate = graphs[int(0.80 * len(graphs)): int(0.90 * len(graphs))]
     # show graphs statistics
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    print('Training set: {}, Validation set: {}'.format(
            len(graphs_train), len(graphs_validate)))

    if args.graph_type in ['zinc', 'qm9']:
        dataset = NumpyTupleDataset.load(args.current_dataset_path, graphs, feature_map)
        dataset_train = torch.utils.data.Subset(dataset, graphs_train)  # 120,803
        dataset_validate = torch.utils.data.Subset(dataset, graphs_validate)
    else:

        dataset_train = Graph_from_file(args, graphs_train, feature_map)
        dataset_validate = Graph_from_file(args, graphs_validate, feature_map)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=NumpyTupleDataset.collate_batch)
    dataloader_validate = DataLoader(
        dataset_validate, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, collate_fn=NumpyTupleDataset.collate_batch)

    # save args
    with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)







    # "feature_map" specifies the properties of graphs to be generated 
    # TODO: separate the intialization of the generative and inference models 
    DAGG = create_generative_model(args, feature_map)
    Rout = create_inference_model(args, feature_map)

     
    train(args, DAGG, Rout, feature_map, dataloader_train, processor)

