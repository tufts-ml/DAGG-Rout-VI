import random
import os, json
import torch
import numpy as np
from args import Args
from utils import create_dirs,load_model
from models.DAGG.model import DAGG
from models.Rout.model import Rout
from data import create_dataset
from train import train
from evaluate import evaluate


if __name__ == '__main__':


    # preparation for model traing 
    args = Args()
    args = args.update_args()
    create_dirs(args)
    torch.manual_seed(args.seed)
    np.random.seed(0)

    if args.task == "train":

        # prepare the data
        dataloader_train, dataloader_validate, data_statistics = create_dataset(args)

        # save args
        with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # the autoregressive graph generative model
        p_model = DAGG(args, data_statistics).to(args.device)

        # the q distributions of node orders given training graphs  
        q_model = Rout(args, data_statistics).to(args.device)

        # minimize the variational lower bounds of training graphs under the p model
        train(args, p_model, q_model, data_statistics, dataloader_train, dataloader_validate)
        
    elif args.task == "evaluate":



        # load the p and q models
        p_model,qmodel = load_model(args)


        # compute MMD values from multiple graphs statistics 
        # compute the approximate log-likelihood from importance sampling
        evaluate(args, p_model)



    else:

        raise Exception("No such task in args.task:" + args.task)

