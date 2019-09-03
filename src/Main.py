import argparse
import torch

# %% Args setup
from src.splitter import Splitter
from src.xgbtrainer import XGBTrainer
from src.train_xgb_emb import XGBEmbeddingTrainer

parser = argparse.ArgumentParser()

################## Model params
parser.add_argument('--num_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--embedding_size', type=int, default=20)
parser.add_argument('--random_state', type=int, default=0)
parser.add_argument('--load', type=bool, default=True)
parser.add_argument('--n_latent', type=int, default=300)

parser.add_argument('--model_name', type=str, default='xgb-emb')
parser.add_argument('--booster_file', type=str, default='xgb-booster')

parser.add_argument('--eta', type=float, default=0.2)
parser.add_argument('--max_depth', type=int, default=6)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)


#################### Finalizing args
args = parser.parse_args()
if args.random_state is not None:
    torch.manual_seed(args.random_state)
print("args = ", args)

def main():
    # split
    splitter = Splitter("../data/santander/train.csv", args)

    # train xgb
    trainer = XGBTrainer(splitter, args)
    train, valid, test = trainer.get_loaders()

    # train embedding
    xgb_emb_trainer = XGBEmbeddingTrainer(args, trainer.max_length)
    xgb_emb_trainer.init_model(train, valid)
    train_emb, valid_emb, test_emb = xgb_emb_trainer.get_embedding([train, valid, test], trainer.trees)



    pass

if __name__ == "__main__":
    main()