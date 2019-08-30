import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from torch.utils.data import DataLoader

from xgb_dump_parser import decision_tree
import numpy as np
from time import time

class xgb_trainer:
    def __init__(self, path, args):

        #Load and split
        self.timer = timer()
        df = pd.read_csv(path)
        X, y = df.iloc[:, 2:], df['target']
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, random_state=args.random_state, train_size=0.5, test_size=0.5)
        dtrain = xgb.DMatrix(Xtrain.values, ytrain)
        dvalid = xgb.DMatrix(Xvalid.values, yvalid)
        param = {
            'objective': 'binary:logistic',
            'eta': args.eta,
            'subsample': 0.6,
            'colsample_bytree': 0.8,
            'max_depth': args.max_depth,
            'lambda': 0.3,
            'alpha': 0.6,
            'gamma': 0.3,
            'eval_metric': 'error',
            'silent': 0,
            'tree_method': 'gpu_hist'
        }
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        self.timer.toc("load and split done")

        #train
        booster = xgb.train(param, dtrain, args.num_round)
        print(booster.eval_set(watchlist))
        dump = booster.get_dump(with_stats=True)
        self.trees = [decision_tree(tree, Xtrain.shape[1]) for tree in dump]
        self.leaf_to_index = [tree.leaf_to_index for tree in self.trees]
        self.max_length = max([len(leaf_to_index) for leaf_to_index in self.leaf_to_index])
        self.num_trees = args.num_round
        self.timer.toc("train done")

        # predict leaf
        train_pred = booster.predict(dtrain, pred_leaf=True)
        valid_pred = booster.predict(dvalid, pred_leaf=True)
        booster = None
        train_leaf = np.array([self.parse_each_predict_leaf(x, self.leaf_to_index) for x in train_pred])
        valid_leaf = np.array([self.parse_each_predict_leaf(x, self.leaf_to_index) for x in valid_pred])


        train, valid_and_test = train_test_split(train_leaf, random_state=args.random_state, train_size=0.6, test_size=0.4)
        valid, test = train_test_split(valid_and_test, random_state=args.random_state, train_size=0.5, test_size=0.5)
        test1, test2 = train_test_split(test, random_state=args.random_state, train_size=0.5, test_size=0.5)
        self.timer.toc("split2 done")
        self.train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=True)
        self.test1_loader = DataLoader(test1, batch_size=test1.shape[0], shuffle=False)
        self.test2_loader = DataLoader(test2, batch_size=test2.shape[0], shuffle=False)

    def parse_each_predict_leaf(self,  leaves, leaf_to_index):
        return [leaf_to_index[i][leaf] for i, leaf in enumerate(leaves)]

    def get_loaders(self):
        return self.train_loader, self.valid_loader, self.test1_loader, self.test2_loader


class timer:
    def __init__(self):
        self.cur = time()

    def tic(self):
        self.cur = time()

    def toc(self, msg, reset=False):
        print("[{:7.2f}] {}".format(time() - self.cur, msg))
        if reset:
            self.cur = time()