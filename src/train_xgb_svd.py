import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.Timer import Timer
from src.xgbembeddingevaluator import XGBEmbeddingEvaluator


class XgbSvdTrainer:
    def __init__(self, args, num_nodes, svd_name='xgb-svd', timer: Timer = None):
        self.timer = Timer() if timer is None else timer
        self.args = args
        self.num_nodes = num_nodes
        self.max_depth = np.ceil(np.log(max(num_nodes))).astype(int)
        self.svd_name = svd_name
        self.cumsum = np.cumsum([0] + self.num_nodes)
        self.embedding_size = args.embedding_size
        self.embeddings = None
        if args.load:
            model_name = "{:s}_{:d}_{:d}".format(svd_name, self.max_depth, len(num_nodes))
            with open(model_name, "rb") as f:
                self.svds = pickle.load(f)
                self._get_weights()
        else:
            self.svds = TruncatedSVD(n_components=self.embedding_size)

    def fit_svd(self, x, save=True):
        if not self.args.load:
            self.svds.fit(x)
            model_name = "{:s}_{:d}_{:d}".format(self.svd_name, self.max_depth, len(self.num_nodes))
            self._get_weights()
            if save:
                with open(model_name, "wb") as f:
                    pickle.dump(self.svds, f)

    def _get_weights(self):
        max_length = max(self.num_nodes)
        embeddings = np.zeros((len(self.num_nodes), max_length, self.embedding_size))
        for i in range(len(self.num_nodes)):
            embeddings[i, :self.num_nodes[i], :] = \
                np.transpose(self.svds.components_)[self.cumsum[i]:self.cumsum[i + 1], :]
        self.embeddings = embeddings
        return embeddings  # (n_trees, n_nodes, e)

    def inference(self, x):
        # (bs, n_trees, e)
        return np.stack([self.embeddings[i][x[:, i]] for i in range(len(self.embeddings))], axis=1)

    def get_embedding(self, x_list, trees, args):
        # XGBEmbeddingEvaluator(self.embeddings, trees, print_eval=args.print_eval)
        return [self.inference(x) for x in x_list]
