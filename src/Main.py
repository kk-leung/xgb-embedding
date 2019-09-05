import argparse

import torch

# %% Args setup
from src.splitter import Splitter
from src.train_mlp import MLPTrainer
from src.train_xgb_emb import XGBEmbeddingTrainer
from src.xgbtrainer import XGBTrainer

parser = argparse.ArgumentParser()

################## Model params
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--embedding_size', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--random_state', type=int, default=0)
parser.add_argument('--load', type=bool, default=True)
parser.add_argument('--print_eval', type=bool, default=True)

parser.add_argument('--model_name', type=str, default='xgb-emb')
parser.add_argument('--mlp_model_name', type=str, default='mlp')
parser.add_argument('--booster_file', type=str, default='xgb-booster')

parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--max_depth', type=int, default=7)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--mlp_num_epoch', type=int, default=100)
parser.add_argument('--mlp_lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--mlp_batch_size', type=int, default=32)
parser.add_argument('--mlp_dropout', type=float, default=0.3)
parser.add_argument('--mlp_weight_decay', type=float, default=0.1)
parser.add_argument('--n_latent', type=int, default=100)


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
    embs = xgb_emb_trainer.get_embedding([train, valid, test], trainer.trees)

    # MLP
    # MLPTrainer(args, splitter.num_input, mode='raw_only').run([splitter.train, splitter.valid, splitter.test], embs)

    # MLPTrainer(args, splitter.num_input, mode='emb_only').run([splitter.train, splitter.valid, splitter.test], embs)

    MLPTrainer(args, splitter.num_input, mode='both').run([splitter.train, splitter.valid, splitter.test], embs)

if __name__ == "__main__":
    main()
