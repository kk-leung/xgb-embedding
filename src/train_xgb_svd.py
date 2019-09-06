from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle

from src.Timer import Timer
from src.xgbembeddingevaluator import XGBEmbeddingEvaluator


class XgbSvdTrainer:
    def __init__(self, args, num_nodes, all_trees=False, timer: Timer = None):
        self.timer = Timer() if timer is None else timer
        self.args = args
        self.num_nodes = num_nodes
        self.cumsum = np.cumsum([0] + self.num_nodes)
        self.embedding_size = args.embedding_size
        self.all_trees = all_trees
        if args.load:
            model_name = "{:s}_{:d}_{:d}_{}".format(self.args.svd_name, self.args.max_depth,
                                                    self.args.num_trees_for_embedding, str(self.all_trees))
            with open(model_name, "rb") as f:
                self.svds = pickle.load(f)
        else:
            if self.all_trees:
                self.svds = [TruncatedSVD(n_components=self.embedding_size)]
            else:
                self.svds = [TruncatedSVD(n_components=self.embedding_size) for _ in range(len(num_nodes))]

    def fit_svd(self, x):
        if not self.args.load:
            if self.all_trees:
                self.svds[0].fit(x)
            else:
                for i, svd in enumerate(self.svds):
                    self.timer.toc("fit " + str(i))
                    svd.fit(x[:, self.cumsum[i]:self.cumsum[i + 1]])
            model_name = "{:s}_{:d}_{:d}_{}".format(self.args.svd_name, self.args.max_depth,
                                                    self.args.num_trees_for_embedding, str(self.all_trees))
            with open(model_name, "wb") as f:
                pickle.dump(self.svds, f)

    def get_weights(self):
        max_length = max(self.num_nodes)
        embeddings = np.zeros((len(self.num_nodes), max_length, self.args.embedding_size))
        if self.all_trees:
            for i in range(len(self.num_nodes)):
                embeddings[i, :self.num_nodes[i], :] = \
                    np.transpose(self.svds[0].components_)[self.cumsum[i]:self.cumsum[i+1], :]
        else:
            for i, svd in enumerate(self.svds):
                embeddings[i, :self.num_nodes[i], :] = np.transpose(svd.components_)
        return embeddings  # (n_trees, n_nodes, e)

    def _inference(self, x):
        if self.all_trees:
            return self.svds[0].transform(x)
        else:
            transformed = [svd.transform(x[:, self.cumsum[i]:self.cumsum[i + 1]]) for i, svd in enumerate(self.svds)]
            return np.stack(transformed, axis=1)

    def get_embedding(self, x_list, trees):
        weights = self.get_weights()
        XGBEmbeddingEvaluator(weights, trees, print_eval=self.args.print_eval)
        return [self._inference(x) for x in x_list]
