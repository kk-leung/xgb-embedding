from time import time

import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset

from src.splitter import Splitter
from src.xgb_dump_parser import DecisionTree


class XGBTrainer:
    def __init__(self, splitter: Splitter, args):

        # Load and split
        self.timer = timer()
        Xtrain, ytrain = splitter.train
        Xvalid, yvalid = splitter.valid
        Xtest, ytest = splitter.test

        zero_count = (ytrain == 0).sum()
        one_count = len(ytrain) - zero_count
        print("zero_count:", zero_count, "one_count", one_count)
        self.weight = [1 / np.sqrt(zero_count), 1 / np.sqrt(one_count)]
        print("weights", self.weight)

        scale_pos_weight = np.sqrt(zero_count / one_count)

        dtrain = xgb.DMatrix(Xtrain, ytrain)
        dvalid = xgb.DMatrix(Xvalid, yvalid)
        dtest = xgb.DMatrix(Xtest, ytest)
        param = {
            'objective': 'binary:logistic',
            'eta': args.eta,
            'subsample': 0.6,
            'colsample_bytree': 0.8,
            'max_depth': args.max_depth,
            'scale_pos_weight': scale_pos_weight,
            'lambda': 0.3,
            'alpha': 0.6,
            'gamma': 0.3,
            'eval_metric': 'auc',
            'silent': 0,
            'tree_method': 'gpu_hist'
        }
        watchlist = [(dtrain, 'train'), (dvalid, 'valid'), (dtest, 'test')]
        self.timer.toc("load and split done")

        # train

        if args.load is False:
            # booster = xgb.train(param, dtrain, evals=watchlist, num_boost_round=args.num_round)
            booster = xgb.train(param, dtrain, num_boost_round=args.num_round)
            booster.save_model(args.booster_file)
        else:
            booster = xgb.Booster()
            booster.load_model(args.booster_file)
            booster.set_param(param)
        print(booster.eval_set(watchlist))
        dump = booster.get_dump(with_stats=True)
        self.trees = [DecisionTree(tree, Xtrain.shape[1]) for tree in dump[:args.num_trees_for_embedding]]
        self.leaf_to_index = [tree.leaf_to_index for tree in self.trees]
        self.max_length = max([len(leaf_to_index) for leaf_to_index in self.leaf_to_index])
        self.num_trees = len(self.trees)
        self.timer.toc("train done. Max length = " + str(self.max_length))

        # predict leaf
        train_pred = booster.predict(dtrain, pred_leaf=True, ntree_limit=args.num_trees_for_embedding)
        valid_pred = booster.predict(dvalid, pred_leaf=True, ntree_limit=args.num_trees_for_embedding)
        test_pred = booster.predict(dtest, pred_leaf=True, ntree_limit=args.num_trees_for_embedding)
        self.timer.toc("predict done")
        booster = None

        train_leaf = self.parse_predict_leaf(train_pred)
        valid_leaf = self.parse_predict_leaf(valid_pred)
        test_leaf = self.parse_predict_leaf(test_pred)
        self.timer.toc("index done")

        train_dataset = XGBLeafDataset(train_leaf, ytrain)
        valid_dataset = XGBLeafDataset(valid_leaf, yvalid)
        test_dataset = XGBLeafDataset(test_leaf, ytest)


        sampler = torch.utils.data.WeightedRandomSampler(self.weight, len(train_dataset))

        # self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    def parse_predict_leaf(self, pred_leaves):
        def func(leaf_index, leaf):
            return self.leaf_to_index[leaf_index][leaf]

        a = np.vectorize(func)
        indexes = np.arange(0, self.num_trees).reshape(1, -1)  # .repeat(len(pred_leaves), 0)
        return a(indexes, pred_leaves)

    def get_loaders(self):
        return self.train_loader, self.valid_loader, self.test_loader


class XGBLeafDataset(Dataset):

    def __init__(self, leaves, y):
        self.leaves = leaves
        self.y = y

    def __len__(self):
        return self.leaves.shape[0]

    def __getitem__(self, index):
        return torch.LongTensor(self.leaves[index, :]), torch.LongTensor(self.y[index:index + 1])


class timer:
    def __init__(self):
        self.cur = time()

    def tic(self):
        self.cur = time()

    def toc(self, msg, reset=False):
        print("[{:7.2f}] {}".format(time() - self.cur, msg))
        if reset:
            self.cur = time()
