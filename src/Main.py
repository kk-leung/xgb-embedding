import argparse

import torch

# %% Args setup
from src.Timer import Timer
from src.ieee_preprocess import IEEESplitter
from src.santandersplitter import SantanderSplitter
from src.train_mlp import MLPTrainer
from src.train_xgb_emb import XGBEmbeddingTrainer
from src.train_xgb_svd import XgbSvdTrainer
from src.xgbtrainer import XGBTrainer
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--load_xgb', type=bool, default=False)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--load_mlp', type=bool, default=False)
parser.add_argument('--random_state', type=int, default=0)
parser.add_argument('--print_eval', type=bool, default=True)

parser.add_argument('--model_name', type=str, default='ieee-xgb-emb')
parser.add_argument('--mlp_model_name', type=str, default='ieee-mlp')
parser.add_argument('--booster_file', type=str, default='ieee-xgb-booster')
parser.add_argument('--svd_name', type=str, default='svd')

parser.add_argument('--eta', type=float, default=0.009)
parser.add_argument('--max_depth', type=int, default=7)
parser.add_argument('--num_round', type=int, default=10000)
parser.add_argument('--xgb_silent', type=bool, default=False)

parser.add_argument('--num_trees_for_embedding', type=int, default=400)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--embedding_size', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--all_trees', type=bool, default=True)

parser.add_argument('--mlp_num_epoch', type=int, default=1000)
parser.add_argument('--mlp_lr', type=float, default=3e-6, help='learning rate')
parser.add_argument('--mlp_batch_size', type=int, default=64)
parser.add_argument('--mlp_dropout', type=float, default=0.5)
parser.add_argument('--mlp_weight_decay', type=float, default=0)
parser.add_argument('--n_latent', type=int, default=100)


#################### Finalizing args
args = parser.parse_args()
if args.random_state is not None:
    torch.manual_seed(args.random_state)
print("args = ", args)

timer = Timer()

def main():
    # split
    splitter = IEEESplitter(args)
    # splitter = SantanderSplitter("../data/santander/train.csv", args)

    # train xgb
    trainer = XGBTrainer(splitter, args, timer)

    pred = trainer.predict(splitter.test[0])
    splitter.export(pred, "xgb_pred.csv")

    # train embedding
    # embs = train_xgb_emb(trainer)
    # embs = train_xgb_svd_emb(trainer)
    #
    # train_emb, valid_emb, test_emb = embs
    # np.save("../data/data/ieee/train_np", train_emb)
    # np.save("../data/data/ieee/valid_np", valid_emb)
    # np.save("../data/data/ieee/test_np", test_emb)

    train_emb = np.load("../data/data/ieee/train_np.npy")
    valid_emb = np.load("../data/data/ieee/valid_np.npy")
    test_emb = np.load("../data/data/ieee/test_np.npy")
    embs = train_emb, valid_emb, test_emb
    print(train_emb.shape, valid_emb.shape, test_emb.shape)
    timer.toc("load done")

    # MLP
    # pred = MLPTrainer(args, splitter.num_input, mode='raw_only').run([splitter.train, splitter.valid, splitter.test], embs)
    # pred = MLPTrainer(args, splitter.num_input, mode='emb_only').run([splitter.train, splitter.valid, splitter.test], embs)
    pred = MLPTrainer(args, splitter.num_input, mode='both').run([splitter.train, splitter.valid, splitter.test], embs)
    splitter.export(pred, "raw_pred.csv")

def train_xgb_svd_emb(trainer):
    train, valid, test = trainer.get_one_hot_version()
    timer.toc("one hot done")
    xgb_emb_trainer = XgbSvdTrainer(args, trainer.num_nodes, timer=timer)
    timer.toc("begin to fit svd")
    xgb_emb_trainer.fit_svd(train)
    timer.toc("svd fit done")
    train, valid, test = trainer.get_leaf_version()
    embs = xgb_emb_trainer.get_embedding([train, valid, test], trainer.trees, args)
    timer.toc("get embedding done")
    return embs

def train_xgb_emb(trainer):
    train, valid, test = trainer.get_loaders()
    xgb_emb_trainer = XGBEmbeddingTrainer(args, trainer.max_length)
    xgb_emb_trainer.init_model(train, valid)
    embs = xgb_emb_trainer.get_embedding([train, valid, test], trainer.trees, args)
    return embs


if __name__ == "__main__":
    main()
