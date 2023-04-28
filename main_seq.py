import random
import os, json
from args import Args
from utils import create_dirs
from models.DAGG.model import DAGG
from models.Rout.attention_model import Rout
from models.Rout.generation import generation
from create_dataset import create_dataset
from train_seq import train





if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(args.seed)

    dataloader_train, dataloader_validate, feature_map = create_dataset(args)
    # save args
    with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)



    DAGG = DAGG(args, feature_map)
    Rout = Rout(embedding_dim = args.gcn_out_dim,
                 hidden_dim=32,
                 state=generation,
                 args=args,
                 featuremap=feature_map,
                 gcn_type=args.gcn_type,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None).to(args.device)

     
    train(args, DAGG, Rout, dataloader_train, dataloader_validate)

