import argparse

import torch

# %% Args setup
from src.Timer import Timer
from src.splitter import Splitter
from src.train_mlp import MLPTrainer
from src.train_xgb_emb import XGBEmbeddingTrainer
from src.train_xgb_svd import XgbSvdTrainer
from src.xgbtrainer import XGBTrainer

parser = argparse.ArgumentParser()

parser.add_argument('--load', type=bool, default=True)
parser.add_argument('--random_state', type=int, default=0)
parser.add_argument('--print_eval', type=bool, default=True)

parser.add_argument('--model_name', type=str, default='xgb-emb')
parser.add_argument('--mlp_model_name', type=str, default='mlp')
parser.add_argument('--booster_file', type=str, default='xgb-booster')
parser.add_argument('--svd_name', type=str, default='svd')

parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--max_depth', type=int, default=7)
parser.add_argument('--num_round', type=int, default=100)

parser.add_argument('--num_trees_for_embedding', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--embedding_size', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--mlp_num_epoch', type=int, default=10000)
parser.add_argument('--mlp_lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--mlp_batch_size', type=int, default=1024)
parser.add_argument('--mlp_dropout', type=float, default=0.3)
parser.add_argument('--mlp_weight_decay', type=float, default=0.1)
parser.add_argument('--n_latent', type=int, default=100)


#################### Finalizing args
args = parser.parse_args()
if args.random_state is not None:
    torch.manual_seed(args.random_state)
print("args = ", args)

timer = Timer()

def main():
    # split
    splitter = Splitter("../data/santander/train.csv", args)

    # train xgb
    trainer = XGBTrainer(splitter, args, timer)

    # train embedding
    # embs = train_xgb_emb(trainer)
    embs = train_xgb_svd_emb(trainer, all_trees=True)

    # MLP
    # MLPTrainer(args, splitter.num_input, mode='raw_only').run([splitter.train, splitter.valid, splitter.test], embs)
    MLPTrainer(args, splitter.num_input, mode='emb_only').run([splitter.train, splitter.valid, splitter.test], embs)
    # MLPTrainer(args, splitter.num_input, mode='both').run([splitter.train, splitter.valid, splitter.test], embs)

def train_xgb_svd_emb(trainer, all_trees=False):
    train, valid, test = trainer.get_one_hot_version()
    timer.toc("one hot done")
    xgb_emb_trainer = XgbSvdTrainer(args, trainer.num_nodes, all_trees=all_trees, timer=timer)
    timer.toc("begin to fit svd")
    xgb_emb_trainer.fit_svd(train)
    timer.toc("svd fit done")
    embs = xgb_emb_trainer.get_embedding([train, valid, test], trainer.trees)
    return embs

def train_xgb_emb(trainer):
    train, valid, test = trainer.get_loaders()
    xgb_emb_trainer = XGBEmbeddingTrainer(args, trainer.max_length)
    xgb_emb_trainer.init_model(train, valid)
    embs = xgb_emb_trainer.get_embedding([train, valid, test], trainer.trees)
    return embs


if __name__ == "__main__":
    main()
