import random
import os, json
from args import Args
from utils import create_dirs
from models.DAGG.model import DAGG
from models.Rout.model import Rout
from data import create_dataset
from train_seq import train





if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    # TODO: set torch and numpy random seeds, or make sure that this line below decides 
    # the behavior of numpy and torch
    random.seed(args.seed) 

    dataloader_train, dataloader_validate, data_statistics = create_dataset(args)

    # save args
    with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    p_model = DAGG(args, data_statistics).to(args.device)
    q_model = Rout(args, data_statistics).to(args.device)
    
    #Rout = Rout(embedding_dim = args.gcn_out_dim,
    #             gcn_type=args.gcn_type,
    #             hidden_dim=args.gnn_hidden_dim,
    #             state=generation,
    #             args=args,
    #             data_statistics=data_statistics,
    #             n_encode_layers=2,
    #             tanh_clipping=10.,
    #             mask_inner=True,
    #             mask_logits=True,
    #             n_heads=8,
    #             checkpoint_encoder=False,
    #             shrink_size=None).to(args.device)

     
    train(args, p_model, q_model, dataloader_train, dataloader_validate)


    # evaluation?


