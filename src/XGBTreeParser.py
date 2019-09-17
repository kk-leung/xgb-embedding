from src.Timer import Timer
from src.xgb_dump_parser import DecisionTree
import numpy as np


class XGBTreeParser:
    def __init__(self, booster, num_input, timer: Timer = None):
        self.timer = Timer() if timer is None else timer
        dump = booster.get_dump(with_stats=True)
        self.trees = [DecisionTree(tree, num_input) for tree in dump]
        self.leaf_to_index = [tree.leaf_to_index for tree in self.trees]
        self.num_nodes = [len(leaf_to_index) for leaf_to_index in self.leaf_to_index]
        self.max_length = max(self.num_nodes)
        self.num_trees = len(self.trees)
        self.timer.toc("init done. Max length = " + str(self.max_length))

    def _parse_predict_leaf(self, pred_leaves):
        def func(leaf_index, leaf):
            return self.leaf_to_index[leaf_index][leaf]

        a = np.vectorize(func)
        indexes = np.arange(0, self.num_trees).reshape(1, -1)  # .repeat(len(pred_leaves), 0)
        return a(indexes, pred_leaves)

    # Input pred_leaves. shape [num_sample, num trees]
    def get_one_hot_version(self, pred_leaves):
        transpose = np.transpose(self._parse_predict_leaf(pred_leaves))
        return np.concatenate([np.eye(self.num_nodes[i])[tree] for i, tree in enumerate(transpose)], axis=1)

class XGBEmbedding:
    def __init__(self, embedding_size, timer: Timer = None):
        self.timer = Timer() if timer is None else timer
