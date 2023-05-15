import random
import os, json
import torch
import numpy as np
from args import Args
from utils import create_dirs
from models.DAGG.model import DAGG
from models.Rout.model import Rout
from data import create_dataset
from train import train





if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)
    torch.manual_seed(args.seed)
    np.random.seed(0)

    dataloader_train, dataloader_validate, data_statistics = create_dataset(args)

    # save args
    with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    p_model = DAGG(args, data_statistics).to(args.device)
    q_model = Rout(args, data_statistics).to(args.device)
    train(args, p_model, q_model, data_statistics, dataloader_train, dataloader_validate)



