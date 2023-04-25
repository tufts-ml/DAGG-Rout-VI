import random
import os, json
from args import Args
from utils import create_dirs
from model import create_generative_model, create_inference_model
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



    DAGG = create_generative_model(args, feature_map)
    Rout = create_inference_model(args, feature_map)

     
    train(args, DAGG, Rout, dataloader_train, dataloader_validate)

